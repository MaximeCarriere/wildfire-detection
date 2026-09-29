"""XP7 shared core — simulated (fake) quantization for YOLOv5, in plain torch.

Everything in XP7 that measures *accuracy* runs through this module; everything
that measures *speed* runs through TensorRT (``lib/trt_export.py``). The split is
deliberate and it is the reason the page can attribute damage: an engine build
couples the quantization choice to TensorRT's kernel selection, layer fusion and
its own fallback logic, so a mAP change measured on an engine cannot be pinned on
the rounding. Fake-quant isolates the rounding.

**What "fake-quant" means.** The tensor is rounded to the grid an INT8 tensor
would land on and then immediately written back as a float. No integer kernel
runs and nothing gets faster — the *numerical* error of quantization is
reproduced exactly while the model stays an ordinary float model that the frozen
evaluation harness can score. That is the standard way to ask "what does this
rounding cost?" without building 120 engines.

**What a scale is.** Linear quantization maps a float ``r`` to an integer ``q``
by ``r = S(q - Z)``. ``S`` (the *scale*) is the step size between representable
values; ``Z`` (the *zero point*) is the integer that means exactly 0.0. Choosing
``S`` requires knowing the range the tensor occupies, which for weights is just
read off the tensor and for activations has to be *measured on sample data* —
that measurement is **calibration**, and which numbers it reports is XP7-E2.

Three axes are configurable here because three of XP7's experiments move them:

* ``bits`` and what gets quantized (weights only, or weights + input
  activations) — E1, E5, E8.
* **granularity**: one scale per tensor, one per output channel, or one per
  group of ``group_size`` input weights — E3, E8.
* **range method**: how the clipping range is chosen from the calibration data —
  ``minmax``, ``percentile``, ``entropy``, ``mse`` — E2.

Design note on where the quantizers sit. Activations are quantized at each
convolution's **input** by a forward pre-hook, not at its output. That is what
TensorRT's QDQ graphs do (a Q/DQ pair in front of every quantized conv), and it
means "quantize layer N" is a local, reversible edit — which is what the
per-layer sweep in E1 needs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

#: Histogram resolution for the calibrators that need a distribution rather than
#: just a min and a max. 2048 is TensorRT's own choice for its entropy
#: calibrator, kept so our ``entropy`` arm is comparable to the engine's.
HIST_BINS = 2048


# --------------------------------------------------------------------------- #
# quantization arithmetic
# --------------------------------------------------------------------------- #

def qrange(bits: int, *, symmetric: bool) -> tuple[int, int]:
    """The integer interval ``bits`` bits are allowed to use.

    Symmetric signed quantization deliberately gives up the most-negative code
    (-128 for INT8): keeping the interval symmetric about zero is what lets the
    zero point be exactly 0, which is what makes the integer matmul cheap —
    with ``Z != 0`` every product picks up a cross-term that has to be
    precomputed and added back (Lecture 05's precompute trick).
    """
    if symmetric:
        return -(2 ** (bits - 1) - 1), 2 ** (bits - 1) - 1
    return 0, 2 ** bits - 1


def quant_params(lo: torch.Tensor, hi: torch.Tensor, *, bits: int,
                 symmetric: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Turn an observed range into ``(scale, zero_point)``.

    ``lo``/``hi`` may be scalars (per-tensor) or vectors (per-channel, per-group);
    the arithmetic is elementwise either way, so granularity is entirely a
    question of what shape the caller observed.
    """
    qmin, qmax = qrange(bits, symmetric=symmetric)
    if symmetric:
        amax = torch.maximum(lo.abs(), hi.abs())
        scale = amax / qmax
        zp = torch.zeros_like(scale)
    else:
        lo = torch.minimum(lo, torch.zeros_like(lo))     # 0 must be representable
        hi = torch.maximum(hi, torch.zeros_like(hi))
        scale = (hi - lo) / (qmax - qmin)
        zp = torch.round(qmin - lo / scale.clamp_min(1e-12))
    # A dead tensor (all zeros) has no range; give it a harmless unit step rather
    # than dividing by zero and poisoning the whole forward pass with NaN.
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    return scale, zp


def fake_quant(x: torch.Tensor, scale: torch.Tensor, zp: torch.Tensor, *,
               bits: int, symmetric: bool) -> torch.Tensor:
    """Round ``x`` onto the integer grid and map it straight back to float.

    Computed in float32 regardless of the model's dtype: the point is to measure
    the error of the *INT8* grid, and doing the division in float16 would add
    float16's own rounding error to the measurement.
    """
    qmin, qmax = qrange(bits, symmetric=symmetric)
    dtype = x.dtype
    xf = x.float()
    q = torch.clamp(torch.round(xf / scale + zp), qmin, qmax)
    return ((q - zp) * scale).to(dtype)


# --------------------------------------------------------------------------- #
# range estimation (calibration)
# --------------------------------------------------------------------------- #

def _kl_threshold(hist: torch.Tensor, bin_width: float, *, bits: int) -> float:
    """TensorRT's entropy calibrator, reimplemented so its choice is inspectable.

    Sweep a candidate clip point ``i`` over the histogram of ``|x|``. Everything
    beyond ``i`` is *saturated* into the last kept bin (that is what clipping
    does), the kept part is then squeezed into the ``2**(bits-1)`` levels the
    integer grid actually offers, and the KL divergence between the two
    distributions is scored. The ``i`` with the least divergence wins.

    This is the routine XP10 caught red-handed: on this detector it decides the
    input tensor only needs 0 to 0.45 of its 0-1 range, clipping the bright sky
    that faint smoke has to be seen against. E2 re-measures it per layer.
    """
    levels = 2 ** (bits - 1)
    nbins = hist.numel()
    h = hist.double().cpu()
    total = h.sum()
    if total <= 0:
        return (nbins + 0.5) * bin_width
    nz = (h > 0).double()
    csum = torch.cat([torch.zeros(1, dtype=torch.float64), h.cumsum(0)])

    # The inner loop over the 128 output levels is vectorised with index_add_:
    # bin j of the kept range belongs to level ``j * levels // i``, so the
    # per-level sums and non-empty counts are two scatter-adds rather than 128
    # Python iterations. That turns a sweep costing ~245k interpreter steps per
    # tensor into ~10k tensor ops, which matters because this routine runs on 60
    # tensors per arm.
    best_i, best_kl = nbins, float("inf")
    for i in range(levels, nbins + 1):
        p = h[:i].clone()
        p[-1] += total - csum[i]                  # outliers saturate, they do not vanish
        psum = p.sum()
        if psum <= 0:
            continue
        p = p / psum

        lvl = (torch.arange(i, dtype=torch.long) * levels) // i
        hi_ = h[:i]
        nzi = nz[:i]
        lsum = torch.zeros(levels, dtype=torch.float64).index_add_(0, lvl, hi_)
        lcnt = torch.zeros(levels, dtype=torch.float64).index_add_(0, lvl, nzi)
        q = torch.where(nzi > 0, (lsum / lcnt.clamp_min(1.0))[lvl],
                        torch.zeros(i, dtype=torch.float64))
        qsum = q.sum()
        if qsum <= 0:
            continue
        q = q / qsum

        mask = p > 0
        kl = torch.sum(p[mask] * torch.log(p[mask] / q[mask].clamp_min(1e-12))).item()
        if kl < best_kl:
            best_kl, best_i = kl, i
    return (best_i + 0.5) * bin_width


def _mse_threshold(hist: torch.Tensor, bin_width: float, *, bits: int,
                   symmetric: bool = True) -> float:
    """Pick the clip point that minimises squared error, scored on the histogram.

    The trade every clipping method makes, stated numerically: clip tighter and
    the step size shrinks (less rounding error on the bulk) but the tail is
    crushed (large error on the outliers). ``minmax`` is the ``frac = 1.0``
    endpoint of this same sweep, which is why the two are comparable.
    """
    centres = (torch.arange(hist.numel(), dtype=torch.float64) + 0.5) * bin_width
    h = hist.double()
    _, qmax = qrange(bits, symmetric=symmetric)
    best_err, best_t = float("inf"), centres[-1].item()
    for frac in torch.linspace(0.3, 1.0, 71).tolist():
        t = centres[-1].item() * frac
        if t <= 0:
            continue
        s = t / qmax
        recon = torch.clamp(torch.round(centres / s), -qmax, qmax) * s
        err = (h * (centres - recon) ** 2).sum().item()
        if err < best_err:
            best_err, best_t = err, t
    return best_t


@dataclass
class Observer:
    """Accumulates what a tensor's range looks like across the calibration pass.

    Two accumulation modes, because the methods need different evidence.
    ``minmax`` needs two running numbers; ``percentile``/``entropy``/``mse`` need
    the shape of the distribution, so a histogram of ``|x|`` is accumulated on a
    range fixed by the first batch. Fixing the range on batch one is a real
    approximation — a later batch that exceeds it is clamped into the top bin —
    and it is the same approximation TensorRT's calibrator makes.
    """
    method: str = "minmax"
    bits: int = 8
    symmetric: bool = True
    percentile: float = 99.99
    lo: torch.Tensor | None = None
    hi: torch.Tensor | None = None
    hist: torch.Tensor | None = None
    bin_width: float = 0.0
    n_batches: int = 0

    @property
    def needs_hist(self) -> bool:
        return self.method in ("percentile", "entropy", "mse")

    @torch.no_grad()
    def collect(self, x: torch.Tensor) -> None:
        x = x.detach().float()
        self.n_batches += 1
        lo, hi = x.min().reshape(1), x.max().reshape(1)
        self.lo = lo if self.lo is None else torch.minimum(self.lo, lo)
        self.hi = hi if self.hi is None else torch.maximum(self.hi, hi)
        if not self.needs_hist:
            return
        a = x.abs()
        if self.hist is None:
            top = max(a.max().item(), 1e-8)
            self.bin_width = top / HIST_BINS
            self.hist = torch.zeros(HIST_BINS, device=x.device, dtype=torch.float64)
        idx = torch.clamp((a.flatten() / self.bin_width).long(), 0, HIST_BINS - 1)
        self.hist += torch.bincount(idx, minlength=HIST_BINS).double()

    @torch.no_grad()
    def range(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.lo is None:
            raise RuntimeError("observer saw no data — did the calibration pass run?")
        if self.method == "minmax":
            return self.lo, self.hi
        hist = self.hist.cpu()
        if self.method == "percentile":
            cum = torch.cumsum(hist, 0) / hist.sum()
            i = int(torch.searchsorted(cum, self.percentile / 100.0).item())
            t = (min(i, HIST_BINS - 1) + 0.5) * self.bin_width
        elif self.method == "entropy":
            t = _kl_threshold(hist, self.bin_width, bits=self.bits)
        else:
            t = _mse_threshold(hist, self.bin_width, bits=self.bits,
                               symmetric=self.symmetric)
        t = max(t, 1e-8)
        dev = self.lo.device
        # A clipping method reports a magnitude. Where the tensor is one-sided
        # (post-ReLU/SiLU activations mostly are) the observed sign structure is
        # kept, so an asymmetric arm still gets the one-sided range it exists for.
        if self.lo.item() >= 0:
            return torch.zeros(1, device=dev), torch.full((1,), t, device=dev)
        return torch.full((1,), -t, device=dev), torch.full((1,), t, device=dev)


# --------------------------------------------------------------------------- #
# weight quantization, at three granularities
# --------------------------------------------------------------------------- #

@torch.no_grad()
def quantize_weight(w: torch.Tensor, *, bits: int = 8, granularity: str = "per_channel",
                    group_size: int = 128, symmetric: bool = True) -> torch.Tensor:
    """Fake-quantize a conv weight ``(out, in, kh, kw)``.

    **per_tensor** — one scale for the whole kernel. One number to ship, and the
    coarsest: a single output filter with an unusually large weight sets the step
    size for all the others, so every quiet filter loses resolution to it.

    **per_channel** — one scale per *output* filter. This is free at inference:
    the conv's output channel ``c`` is a sum of products all sharing scale
    ``S_c``, so ``S_c`` factors straight out of the accumulator and folds into the
    bias/BN rescale that follows. Nothing in the integer kernel changes.

    **per_group** — one scale per ``group_size`` consecutive input weights within
    a filter. Not free (the scale changes mid-accumulation), so it is only used
    where it buys enough to be worth it, which the literature puts at 4 bits and
    E8 tests.

    Weights are quantized symmetrically throughout. Trained conv weights sit
    roughly symmetrically about zero, so a zero point buys almost nothing, and
    ``Z = 0`` is what keeps the integer matmul free of cross-terms.
    """
    out = w.shape[0]
    if granularity == "per_tensor":
        lo, hi = w.min().reshape(1), w.max().reshape(1)
        s, z = quant_params(lo, hi, bits=bits, symmetric=symmetric)
        return fake_quant(w, s, z, bits=bits, symmetric=symmetric)

    if granularity == "per_channel":
        flat = w.reshape(out, -1).float()
        s, z = quant_params(flat.min(1).values, flat.max(1).values,
                            bits=bits, symmetric=symmetric)
        shape = (out,) + (1,) * (w.dim() - 1)
        return fake_quant(w, s.reshape(shape), z.reshape(shape),
                          bits=bits, symmetric=symmetric)

    if granularity == "per_group":
        flat = w.reshape(out, -1).float()
        n = flat.shape[1]
        pad = (-n) % group_size
        if pad:
            flat = torch.cat([flat, torch.zeros(out, pad, device=w.device)], 1)
        g = flat.reshape(out, -1, group_size)
        s, z = quant_params(g.min(2).values, g.max(2).values,
                            bits=bits, symmetric=symmetric)
        qg = fake_quant(g, s.unsqueeze(-1), z.unsqueeze(-1),
                        bits=bits, symmetric=symmetric)
        flat_q = qg.reshape(out, -1)[:, :n]
        return flat_q.reshape(w.shape).to(w.dtype)

    raise ValueError(f"unknown granularity {granularity!r}")


# --------------------------------------------------------------------------- #
# attaching quantization to a live model
# --------------------------------------------------------------------------- #

def conv_layers(model) -> list[tuple[str, nn.Conv2d]]:
    """Every Conv2d in the network, in forward order.

    All 60 of them, the 3 Detect head convolutions included — unlike XP6's
    ``prunable_layers``, which excludes the head because its channel count is
    protocol-fixed. Quantization has no such constraint, and the head is the part
    the lecture and XP10 both expect to be the fragile one, so excluding it would
    remove the interesting half of the question.
    """
    return [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Conv2d)]


def head_conv_names(model) -> list[str]:
    """The names of the Detect head convolutions — E6's ``head-out`` split."""
    from lib.prune_utils import detect_head_convs
    ids = {id(m) for m in detect_head_convs(model)}
    return [n for n, m in conv_layers(model) if id(m) in ids]


@dataclass
class QuantConfig:
    """One complete quantization setting. Every experiment is a diff on this."""
    weight_bits: int = 8
    act_bits: int = 8                       # 0 or None-equivalent: see quantize_acts
    quantize_weights: bool = True
    quantize_acts: bool = True
    granularity: str = "per_channel"
    group_size: int = 128
    act_symmetric: bool = True
    method: str = "minmax"
    percentile: float = 99.99

    def label(self) -> str:
        parts = [f"w{self.weight_bits}" if self.quantize_weights else "wfp",
                 f"a{self.act_bits}" if self.quantize_acts else "afp",
                 self.granularity, self.method]
        if self.method == "percentile":
            parts.append(f"p{self.percentile}")
        parts.append("sym" if self.act_symmetric else "asym")
        return "_".join(parts)


class Quantizer:
    """Applies a :class:`QuantConfig` to a chosen subset of a model's convs.

    Lifecycle, and it matters that these are separate steps:

    1. ``observe()`` — hooks collect activation statistics, the model runs
       unmodified. This is the calibration pass.
    2. ``freeze()`` — the observed ranges become fixed ``(scale, zero_point)``
       pairs. **Static** quantization: the scales are now constants, which is what
       TensorRT executes. Nothing is recomputed per image afterwards.
    3. ``enable()`` — weights are replaced by their quantized values and the
       activation hooks start rounding. The model is now scoreable.
    4. ``restore()`` — original weights back, hooks removed. Required between
       sweep cells, and E1 restores 120 times.

    ``targets`` is the layer subset. E1 passes one name; E5/E6 pass many; passing
    none means every conv.
    """

    def __init__(self, model, config: QuantConfig, targets: list[str] | None = None):
        self.model = model
        self.cfg = config
        all_names = [n for n, _ in conv_layers(model)]
        self.targets = list(targets) if targets is not None else all_names
        unknown = set(self.targets) - set(all_names)
        if unknown:
            raise KeyError(f"not convolutions in this model: {sorted(unknown)}")
        self.mods = dict(conv_layers(model))
        self.observers: dict[str, Observer] = {}
        self.frozen: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._orig: dict[str, torch.Tensor] = {}
        self._handles: list = []
        self._mode = "off"

    # -- calibration ------------------------------------------------------- #

    def observe(self) -> "Quantizer":
        """Attach collecting hooks. Call, run the calibration images, then freeze."""
        self._detach()
        if not self.cfg.quantize_acts:
            self._mode = "observe"
            return self
        for name in self.targets:
            obs = Observer(method=self.cfg.method, bits=self.cfg.act_bits,
                           symmetric=self.cfg.act_symmetric,
                           percentile=self.cfg.percentile)
            self.observers[name] = obs

            def hook(_m, inputs, _obs=obs):
                _obs.collect(inputs[0])

            self._handles.append(self.mods[name].register_forward_pre_hook(hook))
        self._mode = "observe"
        return self

    def freeze(self) -> "Quantizer":
        """Turn observed ranges into fixed scales, and drop the observers."""
        for name, obs in self.observers.items():
            lo, hi = obs.range()
            self.frozen[name] = quant_params(lo, hi, bits=self.cfg.act_bits,
                                             symmetric=self.cfg.act_symmetric)
        self._detach()
        self._mode = "frozen"
        return self

    def adopt(self, frozen: dict) -> "Quantizer":
        """Take activation scales measured by a different quantizer.

        Sound, and it is what makes E1 affordable. During ``observe()`` the model
        runs **unmodified** — the hooks only read. So one calibration pass over a
        quantizer targeting all 60 convolutions records, for each layer, the
        activation distribution that layer sees when every layer is in float. That
        is precisely the distribution a single-layer arm sees too, because in that
        arm every *preceding* layer is still in float. One pass therefore yields
        the correct scale for all 120 single-layer cells instead of 120 passes.

        It stops being sound the moment two quantized layers feed each other,
        which is why the whole-network experiments (E2, E3, E5, E6) each run their
        own calibration pass rather than adopting E1's.
        """
        if self.cfg.quantize_acts:
            missing = set(self.targets) - set(frozen)
            if missing:
                raise KeyError(f"no frozen scale for {sorted(missing)}")
        self.frozen = {n: frozen[n] for n in self.targets if n in frozen}
        self._mode = "frozen"
        return self

    def scale_report(self) -> dict:
        """The scales themselves, so a result can show *why* a method lost.

        XP10's lesson: the 52-point drop was diagnosed by reading the calibration
        numbers, not by guessing at causes. Every arm here can be asked what range
        it chose for each tensor.
        """
        return {name: {"scale": float(s.flatten()[0]), "zero_point": float(z.flatten()[0]),
                       "represented_max": float((s.flatten()[0] *
                                                 qrange(self.cfg.act_bits,
                                                        symmetric=self.cfg.act_symmetric)[1]))}
                for name, (s, z) in self.frozen.items()}

    # -- application ------------------------------------------------------- #

    def enable(self) -> "Quantizer":
        """Quantize the weights and start rounding the activations."""
        if self.cfg.quantize_acts and not self.frozen:
            raise RuntimeError("activations are quantized but no scales are frozen — "
                               "run observe(), the calibration images, then freeze()")
        self._detach()
        if self.cfg.quantize_weights:
            for name in self.targets:
                m = self.mods[name]
                self._orig[name] = m.weight.data.clone()
                m.weight.data = quantize_weight(
                    m.weight.data, bits=self.cfg.weight_bits,
                    granularity=self.cfg.granularity, group_size=self.cfg.group_size)
        if self.cfg.quantize_acts:
            bits, sym = self.cfg.act_bits, self.cfg.act_symmetric
            for name in self.targets:
                s, z = self.frozen[name]

                def hook(_m, inputs, _s=s, _z=z):
                    return (fake_quant(inputs[0], _s, _z, bits=bits, symmetric=sym),
                            *inputs[1:])

                self._handles.append(self.mods[name].register_forward_pre_hook(hook))
        self._mode = "on"
        return self

    def rebase(self) -> "Quantizer":
        """Treat the model's **current** weights as the new float reference.

        Needed after a recovery or QAT fine-tune. ``enable()`` stashes the
        pre-quantization weights so ``restore()`` can undo itself, but training
        happens *after* that stash: the weights in the modules are now the trained
        ones, and calling ``restore()`` would write the stale pre-training copy
        back over them and silently discard every epoch. ``rebase()`` drops the
        stash and the hooks instead, so the next ``enable()`` re-quantizes the
        weights training actually produced.

        The hooks must stay attached *during* training — that is what makes it
        quantization-aware rather than plain fine-tuning — so this is called
        afterwards, never before.
        """
        self._detach()
        self._orig.clear()
        self._mode = "off"
        return self

    def restore(self) -> "Quantizer":
        """Put the model back exactly as it was found."""
        self._detach()
        for name, w in self._orig.items():
            self.mods[name].weight.data = w
        self._orig.clear()
        self._mode = "off"
        return self

    def _detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.restore()
        return False


class DetectOutputQuantizer:
    """Quantizes the Detect module's *decoded output*, per-tensor, as TensorRT does.

    Needed because the conv-level :class:`Quantizer` above cannot express the
    pathology XP10 diagnosed. A TensorRT INT8 engine quantizes the whole graph,
    and YOLOv5's Detect layer ends by concatenating, into one tensor:

    * box coordinates, **in pixels**, spanning 0 to the input resolution, and
    * objectness and class scores, in **[0, 1]**.

    INT8 carries one scale per tensor. Set it from the boxes and 256 levels are
    stretched across 0-512, so every box edge is quantized to ~2 px and every
    probability collapses into a fraction of a single step. Box regression is
    destroyed while classification limps on — and that asymmetry is the
    fingerprint XP10 saw (mAP50 0.2554 with the head unpinned, tiny plumes down
    99%, yet the night slice still holding 0.48).

    This class reproduces that one effect in isolation, which is what lets E6's
    third arm ("run the decode outside the engine in float") be *measured* rather
    than asserted. Without it, fake-quant has nothing to say about arm 3: the
    conv-level simulation never quantizes the decode, so arms 2 and 3 would be
    numerically identical for a reason that is an artefact of the simulation
    rather than a fact about the hardware.
    """

    def __init__(self, model, *, bits: int = 8, symmetric: bool = True,
                 method: str = "minmax"):
        self.detect = next((m for m in model.modules() if type(m).__name__ == "Detect"), None)
        if self.detect is None:
            raise RuntimeError("no Detect module found")
        self.bits, self.symmetric = bits, symmetric
        self.observer = Observer(method=method, bits=bits, symmetric=symmetric)
        self.frozen: tuple[torch.Tensor, torch.Tensor] | None = None
        self._handles: list = []

    def observe(self) -> "DetectOutputQuantizer":
        self._detach()

        def hook(_m, _inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            self.observer.collect(t)

        self._handles.append(self.detect.register_forward_hook(hook))
        return self

    def freeze(self) -> "DetectOutputQuantizer":
        lo, hi = self.observer.range()
        self.frozen = quant_params(lo, hi, bits=self.bits, symmetric=self.symmetric)
        self._detach()
        return self

    def enable(self) -> "DetectOutputQuantizer":
        if self.frozen is None:
            raise RuntimeError("calibrate the decode output before quantizing it")
        s, z = self.frozen
        bits, sym = self.bits, self.symmetric

        def hook(_m, _inp, out):
            if isinstance(out, (tuple, list)):
                return (fake_quant(out[0], s, z, bits=bits, symmetric=sym), *out[1:])
            return fake_quant(out, s, z, bits=bits, symmetric=sym)

        self._detach()
        self._handles.append(self.detect.register_forward_hook(hook))
        return self

    def report(self) -> dict:
        if self.frozen is None:
            return {}
        s, z = self.frozen
        _, qmax = qrange(self.bits, symmetric=self.symmetric)
        return {"scale": float(s.flatten()[0]), "zero_point": float(z.flatten()[0]),
                "represented_max": float(s.flatten()[0] * qmax)}

    def restore(self) -> "DetectOutputQuantizer":
        self._detach()
        return self

    def _detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


# --------------------------------------------------------------------------- #
# what the compressed model actually weighs
# --------------------------------------------------------------------------- #

#: Bytes per stored scale and zero point. TensorRT and ONNX QDQ both carry scales
#: as FP32 and zero points as the integer type of the tensor they belong to.
SCALE_BYTES = 4
ZERO_POINT_BYTES = 1


def n_weight_scales(conv: nn.Conv2d, *, granularity: str, group_size: int) -> int:
    """How many scale factors this layer's weights need at a given granularity."""
    if granularity == "per_tensor":
        return 1
    if granularity == "per_channel":
        return conv.out_channels
    if granularity == "per_group":
        per_filter = conv.weight[0].numel()
        return conv.out_channels * max(1, -(-per_filter // group_size))
    raise ValueError(f"unknown granularity {granularity!r}")


@torch.no_grad()
def model_size_report(model, cfg: QuantConfig, targets: list[str] | None = None) -> dict:
    """Bytes the model occupies under a :class:`QuantConfig`, itemised.

    Counted rather than measured, and deliberately so: these arms are scored with
    fake-quant, which leaves every tensor a float, so a file on disk would report
    the FP32 size of a simulation and mean nothing. What a deployment actually
    stores is determined by the configuration — bit-width, which layers are
    quantized, and how many scale factors the granularity implies — and all three
    are known exactly. The **engine** sizes in E4 are the measured counterpart and
    they are larger, because an engine carries its own tactic metadata, workspace
    descriptors and fused-kernel tables on top of the weights.

    Three things are itemised separately because they behave differently:

    * **quantized weights** — the part that shrinks with the bit-width.
    * **untouched parameters** — batch-norm, biases, and any convolution left out
      of ``targets``. These stay FP16 and are the reason an "INT8 model" is never
      a quarter the size of an FP32 one.
    * **scale overhead** — the price of granularity. Per-tensor costs 4 bytes a
      layer; per-channel costs 4 bytes per output filter; per-group costs 4 bytes
      per group. This is what E3 is buying when it pays for per-channel, and it is
      why E8's group-wise INT4 is not simply "half of INT8".
    """
    convs = dict(conv_layers(model))
    sel = set(targets if targets is not None else convs)

    total_params = sum(p.numel() for p in model.parameters())
    quantized_numel = sum(convs[n].weight.numel() for n in sel)

    w_bits = cfg.weight_bits if cfg.quantize_weights else 16
    weight_bytes = quantized_numel * w_bits / 8

    n_scales = sum(n_weight_scales(convs[n], granularity=cfg.granularity,
                                   group_size=cfg.group_size) for n in sel)
    scale_bytes = n_scales * SCALE_BYTES        # weights are symmetric: no zero point

    act_scales = len(sel) if cfg.quantize_acts else 0
    act_bytes = act_scales * (SCALE_BYTES + (0 if cfg.act_symmetric else ZERO_POINT_BYTES))

    untouched = (total_params - quantized_numel) * 2      # FP16
    total = weight_bytes + scale_bytes + act_bytes + untouched
    fp16_baseline = total_params * 2

    return {
        # Exact bytes alongside the rounded display values: the itemisation is a
        # claim about where the model's size goes, and a claim that only holds to
        # three decimal places is not auditable.
        "bytes": {
            "quantized_weights": int(weight_bytes),
            "untouched_params": int(untouched),
            "weight_scales": int(scale_bytes),
            "activation_scales": int(act_bytes),
            "total": int(total),
            "fp16_baseline": int(fp16_baseline),
        },
        "params_total_m": round(total_params / 1e6, 4),
        "params_quantized_m": round(quantized_numel / 1e6, 4),
        "params_quantized_pct": round(quantized_numel / total_params * 100, 1),
        "weight_bits": w_bits,
        "granularity": cfg.granularity,
        "quantized_weights_mb": round(weight_bytes / 1e6, 3),
        "untouched_params_mb": round(untouched / 1e6, 3),
        "weight_scales": n_scales,
        "weight_scale_overhead_kb": round(scale_bytes / 1e3, 2),
        "activation_scales": act_scales,
        "activation_scale_overhead_kb": round(act_bytes / 1e3, 2),
        "total_mb": round(total / 1e6, 3),
        "fp16_baseline_mb": round(fp16_baseline / 1e6, 3),
        "compression_x": round(fp16_baseline / total, 3),
        "note": "counted from the configuration, not measured on disk — fake-quant "
                "leaves every tensor a float. E4's engine sizes are the measured "
                "counterpart and are larger by the engine's own metadata.",
    }
