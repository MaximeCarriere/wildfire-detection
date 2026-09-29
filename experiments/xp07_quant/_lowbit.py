"""Below-8-bit machinery for E8: error-compensated rounding, and codebooks.

Two ideas live here, and they belong to different eras of the lecture.

**Error-compensated rounding (GPTQ / AdaRound family).** Round-to-nearest treats
each weight independently, which is only optimal if the weights are independent —
and they are not, because they are summed against correlated activations. If
weight ``j`` is rounded up, the layer's output is biased, and that bias can be
*paid back* by adjusting the weights not yet quantized. GPTQ does exactly this:
quantize the columns of a layer in order, and after each one push its rounding
error into the remaining columns, weighted by the inverse Hessian of the layer's
reconstruction loss. The Hessian is ``2 X X^T`` over the layer's input
activations, so this needs calibration data and is the reason E8 is more than a
one-line change from E1.

At 8 bits this is presumed a no-op — the rounding error is already tiny relative
to what the network tolerates — and E8 is where that presumption gets checked for
free. At 4 bits the literature says it is the difference between usable and not.

**The GPTQ implementation here is checked against its own premise**, because an
error-compensation routine that silently does nothing looks exactly like an
error-compensation routine that works on an easy layer. The test is the
conditioning of the Hessian:

* fed **white noise**, ``X X^T`` is near-identity (condition number ~1.4), the
  weights are effectively independent, and GPTQ matches plain round-to-nearest
  to within 0.0% — which is the correct answer, not a failure.
* fed a **spatially smooth, channel-correlated input** of the kind a real feature
  map is (condition number ~1.4e6), the same code cuts the layer's output error
  by **67%** against round-to-nearest.

A zero Hessian — a layer no calibration data reached — falls back to RTN and says
so. It used to sail through instead: every diagonal entry is "dead", so the usual
bookkeeping zeroed every input column, Cholesky'd the resulting identity happily,
and returned an empty layer reporting success.

**K-means codebooks (Deep Compression, Han et al. 2015).** Instead of a uniform
grid, cluster the weights and store a 4-bit *index* into a 16-entry table of FP16
centroids. The weights are no longer on a regular lattice, so no integer kernel
can use them directly: the model is decoded back to float before it runs. That
makes this a **storage** result, not a speed one, which is where it belongs in
2026 — the lecture's method placed at rest rather than in the inner loop.
Huffman coding on top exploits the fact that the cluster histogram is far from
uniform.
"""
from __future__ import annotations

import math
from collections import Counter

import torch

from _quant import fake_quant, quant_params


# --------------------------------------------------------------------------- #
# GPTQ-style error-compensated rounding
# --------------------------------------------------------------------------- #

@torch.no_grad()
def gptq_quantize(weight: torch.Tensor, hessian: torch.Tensor, *, bits: int = 4,
                  group_size: int = 128, damp_frac: float = 0.01,
                  block: int = 128) -> tuple[torch.Tensor, dict]:
    """Quantize ``weight`` to ``bits``, compensating each column's error forward.

    ``weight`` is ``(out, in*kh*kw)``; ``hessian`` is ``(in*kh*kw, in*kh*kw)``,
    the accumulated ``X X^T`` for this layer over the calibration set. Returns the
    quantized weight **and** a dict saying whether it had to fall back.

    The damping term is not optional. ``H`` is estimated from a few hundred images
    and is routinely singular (a dead input channel gives an all-zero row), so a
    fraction of the mean diagonal is added before the Cholesky. Columns whose
    Hessian diagonal is zero carry no information about the loss and are quantized
    plainly, without compensation.
    """
    w = weight.clone().float()
    n = w.shape[1]
    h = hessian.clone().float()

    dead = torch.diag(h) == 0
    if bool(dead.all()):
        # No calibration data reached this layer, so there is no reconstruction
        # loss to minimise. The usual GPTQ bookkeeping would zero every "dead"
        # input column — which here is *all* of them, silently returning an empty
        # layer that still Choleskys cleanly and reports success. Fall back.
        return _rtn_grouped(weight, bits=bits, group_size=group_size), {
            "fallback": True,
            "reason": "Hessian is entirely zero — the layer saw no calibration data"}
    h[dead, dead] = 1.0
    w[:, dead] = 0.0          # these columns cannot affect the output
    h += torch.eye(n, device=h.device) * (damp_frac * torch.diag(h).mean())

    # Upper Cholesky of the inverse: hinv[j, j:] is the row used to spread
    # column j's error over the columns still to come.
    try:
        hinv = torch.linalg.cholesky(torch.cholesky_inverse(
            torch.linalg.cholesky(h)), upper=True)
    except Exception as exc:
        # A Hessian too ill-conditioned even for damping. Fall back to plain
        # round-to-nearest and *say so* — a caller comparing the output against
        # RTN to detect this would be guessing, and would guess wrong whenever
        # GPTQ legitimately agreed with RTN on a layer.
        return _rtn_grouped(weight, bits=bits, group_size=group_size), {
            "fallback": True, "reason": f"{type(exc).__name__}: {exc}"}

    q_out = torch.zeros_like(w)
    for i0 in range(0, n, block):
        i1 = min(i0 + block, n)
        w_blk = w[:, i0:i1].clone()
        q_blk = torch.zeros_like(w_blk)
        err_blk = torch.zeros_like(w_blk)
        hinv_blk = hinv[i0:i1, i0:i1]

        for j in range(i1 - i0):
            col = w_blk[:, j]
            gcol = (i0 + j) // group_size
            gs, ge = gcol * group_size, min((gcol + 1) * group_size, n)
            grp = w[:, gs:ge]
            s, z = quant_params(grp.min(1).values, grp.max(1).values,
                                bits=bits, symmetric=True)
            qcol = fake_quant(col, s, z, bits=bits, symmetric=True)
            q_blk[:, j] = qcol

            d = hinv_blk[j, j]
            e = (col - qcol) / d
            # Pay the error forward, inside the block now and past it below.
            w_blk[:, j:] -= e.unsqueeze(1) @ hinv_blk[j, j:].unsqueeze(0)
            err_blk[:, j] = e

        q_out[:, i0:i1] = q_blk
        if i1 < n:
            w[:, i1:] -= err_blk @ hinv[i0:i1, i1:]

    return q_out.reshape(weight.shape).to(weight.dtype), {"fallback": False}


@torch.no_grad()
def _rtn_grouped(weight: torch.Tensor, *, bits: int, group_size: int) -> torch.Tensor:
    """Plain round-to-nearest with group-wise scales — GPTQ's control arm."""
    from _quant import quantize_weight
    return quantize_weight(weight, bits=bits, granularity="per_group",
                           group_size=group_size)


class HessianCollector:
    """Accumulates ``X X^T`` per convolution over the calibration pass.

    The rows of ``X`` are the *unfolded* patches the convolution actually
    multiplies, not the raw feature map — that is what makes the Hessian match the
    layer's real reconstruction loss. ``im2col`` is done with ``unfold`` using the
    layer's own stride, padding and dilation, so the statistic is correct for
    strided and dilated layers too.
    """

    def __init__(self, conv):
        self.conv = conv
        n = conv.in_channels // conv.groups * conv.kernel_size[0] * conv.kernel_size[1]
        self.h = torch.zeros(n, n, dtype=torch.float32, device=conv.weight.device)
        self.n_samples = 0
        self._handle = None

    def attach(self, chunk: int = 4096):
        """Start accumulating. ``chunk`` bounds the peak memory, and has to.

        The unfolded patch matrix for a 3x3 convolution with 512 input channels at
        512 px is ``4608 x (B * 4096)`` — about **600 MB per batch of 8** in
        float32. On an 8 GB board that is shared with the display, the model and
        the CUDA context, materialising it is an out-of-memory crash rather than a
        slow path. The Hessian it accumulates into is only ``4608 x 4608``
        (85 MB), so the fix is to stream: unfold, then fold the columns into the
        Hessian ``chunk`` at a time and free each slice. The result is identical —
        ``X X^T`` is a sum over columns — it just never holds all of them at once.
        """
        def hook(m, inputs):
            x = inputs[0].detach().float()
            cols = torch.nn.functional.unfold(
                x, m.kernel_size, dilation=m.dilation,
                padding=m.padding, stride=m.stride)          # (B, n, L)
            cols = cols.transpose(0, 1).reshape(cols.shape[1], -1)   # (n, B*L)
            for i in range(0, cols.shape[1], chunk):
                blk = cols[:, i:i + chunk]
                self.h += blk @ blk.t()
                self.n_samples += blk.shape[1]
            del cols

        self._handle = self.conv.register_forward_pre_hook(hook)
        return self

    def detach(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return self


# --------------------------------------------------------------------------- #
# K-means codebook + Huffman (storage only)
# --------------------------------------------------------------------------- #

@torch.no_grad()
def kmeans_codebook(weight: torch.Tensor, *, bits: int = 4,
                    iters: int = 25) -> tuple[torch.Tensor, torch.Tensor]:
    """1-D k-means over a layer's weights. Returns ``(centroids, indices)``.

    Initialised on the quantiles of the weight distribution rather than at random,
    which for a unimodal bell-shaped tensor converges in a handful of iterations
    and — more importantly for a study — is deterministic.
    """
    k = 2 ** bits
    flat = weight.detach().float().flatten()
    qs = torch.linspace(0, 1, k, device=flat.device)
    centroids = torch.quantile(flat, qs)
    for _ in range(iters):
        idx = torch.argmin((flat.unsqueeze(1) - centroids.unsqueeze(0)).abs(), dim=1)
        for j in range(k):
            m = idx == j
            if m.any():
                centroids[j] = flat[m].mean()
    idx = torch.argmin((flat.unsqueeze(1) - centroids.unsqueeze(0)).abs(), dim=1)
    return centroids, idx


def huffman_bits(indices: torch.Tensor) -> float:
    """Bits per symbol a Huffman code would need for this index stream.

    The true Huffman code length, built from the symbol histogram, not the Shannon
    entropy — they differ, and quoting entropy as though it were a file size is a
    common way to overstate compression by a few percent.
    """
    counts = Counter(indices.flatten().tolist())
    if len(counts) <= 1:
        return 0.0
    import heapq
    freqs = list(counts.values())
    total = sum(freqs)
    # The total code length of a Huffman tree equals the sum of the weights of its
    # internal nodes, so the tree never has to be built — only merged.
    heap = list(freqs)
    heapq.heapify(heap)
    bits_total = 0
    while len(heap) > 1:
        a = heapq.heappop(heap)
        b = heapq.heappop(heap)
        bits_total += a + b
        heapq.heappush(heap, a + b)
    return bits_total / total
