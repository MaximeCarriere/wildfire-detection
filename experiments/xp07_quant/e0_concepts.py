#!/usr/bin/env python3
"""XP7 E0 — the numbers the explainer figures are drawn from.

Not an experiment. The XP6 page explains a technique with figures built from the
model actually under study rather than from a textbook schematic, and this
collects what XP7's equivalents need, so every "what quantization is" picture on
the page is a measurement of *this* detector.

Four things are recorded:

1. **A weight distribution and its grid.** One representative convolution's
   weights, as a histogram, with the INT8 step size a per-tensor scale gives it.
   This is what "snapping to a grid" actually looks like on a trained layer.
2. **Per-output-channel weight ranges.** The lecture's per-channel-vs-per-tensor
   figure (Lec06 p14) reproduced on YOLOv5s: if one filter's range is far wider
   than its neighbours', a single per-tensor scale is set by that filter and
   every other filter loses resolution to it. Whether that is true here is a
   measurement, not an assumption.
3. **The input activation histogram, and where each method clips it.** The whole
   E2 story in one panel: the network input is normalised to [0, 1] and every
   daylight frame has sky near 1.0. Each calibration method's chosen clip point
   is recorded against that histogram.
4. **The Detect output tensor, channel by channel.** The mechanism behind the
   fragile head: YOLOv5's decode concatenates box coordinates in pixels with
   objectness and class probabilities in [0, 1]. The per-channel ranges of that
   one tensor are recorded so the figure can show the two populations and the
   single scale that has to serve both.

Cheap by design — 64 calibration images, a handful of forward passes.

Usage
    python e0_concepts.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _calib                                                      # noqa: E402
import torch                                                       # noqa: E402
from _calib import log                                             # noqa: E402
from _quant import (HIST_BINS, Observer, conv_layers,              # noqa: E402
                    head_conv_names, qrange, quant_params)

#: A mid-network convolution: past the fragile early layers, wide enough that
#: per-channel has something to say. Named rather than chosen by index so the
#: figure caption can state exactly which layer it is showing.
SHOWCASE_CONV = "model.8.cv1.conv"

#: Tensors whose full activation histogram is recorded, so the clipping
#: explainer can be drawn from real distributions rather than a sketch.
#:
#: Two are needed, because the two calibration failure modes are opposite and one
#: tensor can only show one of them. The **network input** is bounded in [0, 1]
#: with mass right up to the edge — clipping anything there is pure loss, which is
#: where entropy calibration fails. An **internal activation** is a long-tailed
#: distribution where a handful of outliers sit far above the bulk — refusing to
#: clip there spends the whole range on values that almost never occur, which is
#: where min-max fails. E2's headline, that both ends of the sweep are wrong,
#: needs both pictures.
HISTOGRAM_LAYERS = ["model.0.conv", "model.24.m.0", "model.17.cv3.conv"]


def weight_panel(model) -> dict:
    """One layer's weights, its per-tensor grid, and its per-channel ranges."""
    mods = dict(conv_layers(model))
    conv = mods[SHOWCASE_CONV]
    w = conv.weight.data.float()
    flat = w.reshape(w.shape[0], -1)

    hist = torch.histc(w.flatten().cpu(), bins=200,
                       min=float(w.min()), max=float(w.max()))
    s_t, _ = quant_params(w.min().reshape(1), w.max().reshape(1),
                          bits=8, symmetric=True)
    s_c, _ = quant_params(flat.min(1).values, flat.max(1).values,
                          bits=8, symmetric=True)

    per_channel_absmax = flat.abs().max(1).values
    return {
        "layer": SHOWCASE_CONV,
        "shape": list(w.shape),
        "hist_counts": hist.tolist(),
        "hist_min": float(w.min()), "hist_max": float(w.max()),
        "per_tensor_scale": float(s_t.flatten()[0]),
        "per_channel_scales": [float(x) for x in s_c],
        "per_channel_absmax": [float(x) for x in per_channel_absmax],
        "per_channel_min": [float(x) for x in flat.min(1).values],
        "per_channel_max": [float(x) for x in flat.max(1).values],
        "widest_over_narrowest": float(per_channel_absmax.max() / per_channel_absmax.min()),
    }


def activation_panel(model, n_images: int) -> dict:
    """The input tensor's distribution, and every method's clip point on it."""
    from lib.trt_export import _letterbox_batch

    mods = dict(conv_layers(model))
    first = mods["model.0.conv"]
    observers = {
        "minmax": Observer(method="minmax", bits=8),
        "percentile_99.9": Observer(method="percentile", bits=8, percentile=99.9),
        "percentile_99.99": Observer(method="percentile", bits=8, percentile=99.99),
        "entropy": Observer(method="entropy", bits=8),
        "mse": Observer(method="mse", bits=8),
    }
    handles = [first.register_forward_pre_hook(
        lambda _m, inp, _o=o: _o.collect(inp[0])) for o in observers.values()]

    paths = _calib.calib_paths(n_images)
    with torch.no_grad():
        for i in range(0, len(paths), 8):
            arr = _letterbox_batch(paths[i:i + 8], _calib.RES, _calib.YOLOV5_REPO)
            model(torch.from_numpy(arr).cuda().half())
    for h in handles:
        h.remove()

    _, qmax = qrange(8, symmetric=True)
    clips = {}
    for name, obs in observers.items():
        lo, hi = obs.range()
        s, z = quant_params(lo, hi, bits=8, symmetric=True)
        clips[name] = {"represented_max": float(s.flatten()[0] * qmax),
                       "scale": float(s.flatten()[0])}
    ref = observers["entropy"]
    return {
        "tensor": "network input (model.0.conv input), normalised to [0,1]",
        "n_images": n_images,
        "hist_counts": ref.hist.cpu().tolist(),
        "bin_width": ref.bin_width,
        "observed_min": float(ref.lo.item()), "observed_max": float(ref.hi.item()),
        "clips": clips,
    }


def histogram_panels(model, n_images: int) -> dict:
    """Full |activation| histograms for a few named layers, for the figures."""
    from lib.trt_export import _letterbox_batch

    mods = dict(conv_layers(model))
    obs = {n: Observer(method="mse", bits=8) for n in HISTOGRAM_LAYERS}
    handles = [mods[n].register_forward_pre_hook(
        lambda _m, inp, _o=obs[n]: _o.collect(inp[0])) for n in HISTOGRAM_LAYERS]

    paths = _calib.calib_paths(n_images)
    with torch.no_grad():
        for i in range(0, len(paths), 8):
            arr = _letterbox_batch(paths[i:i + 8], _calib.RES, _calib.YOLOV5_REPO)
            model(torch.from_numpy(arr).cuda().half())
    for h in handles:
        h.remove()

    out = {}
    for name, o in obs.items():
        methods = {}
        for m, pct in (("minmax", None), ("percentile", 99.99), ("entropy", None),
                       ("mse", None)):
            probe = Observer(method=m, bits=8, percentile=pct or 99.99)
            probe.lo, probe.hi = o.lo, o.hi
            probe.hist, probe.bin_width, probe.n_batches = o.hist, o.bin_width, o.n_batches
            lo, hi = probe.range()
            s_, _ = quant_params(lo, hi, bits=8, symmetric=True)
            key = m if pct is None else f"{m}_{pct}"
            methods[key] = round(float(s_.flatten()[0] * qrange(8, symmetric=True)[1]), 6)
        out[name] = {"hist_counts": o.hist.cpu().tolist(), "bin_width": o.bin_width,
                     "observed_max": float(o.hi.item()), "clips": methods}
    return out


def detect_panel(model, n_images: int) -> dict:
    """Per-channel ranges of the Detect output — the two-populations mechanism."""
    from lib.trt_export import _letterbox_batch

    detect = next(m for m in model.modules() if type(m).__name__ == "Detect")
    captured = {}

    def hook(_m, _i, out):
        t = out[0] if isinstance(out, (tuple, list)) else out
        captured["t"] = t.detach().float()

    h = detect.register_forward_hook(hook)
    paths = _calib.calib_paths(min(n_images, 16))
    with torch.no_grad():
        arr = _letterbox_batch(paths[:8], _calib.RES, _calib.YOLOV5_REPO)
        model(torch.from_numpy(arr).cuda().half())
    h.remove()

    t = captured["t"]                       # (B, anchors, 5 + nc)
    per_ch = t.reshape(-1, t.shape[-1])
    amax = per_ch.abs().max(0).values
    names = ["x", "y", "w", "h", "obj", "smoke", "fire"][:t.shape[-1]]
    lo, hi = per_ch.min().reshape(1), per_ch.max().reshape(1)
    s, _ = quant_params(lo, hi, bits=8, symmetric=True)
    step = float(s.flatten()[0])
    return {
        "tensor": "Detect decoded output",
        "shape": list(t.shape),
        "channel_names": names,
        "channel_absmax": [float(x) for x in amax],
        "per_tensor_step": step,
        "probability_levels": float(1.0 / step) if step > 0 else None,
        "box_px_granularity": step,
        "note": "one INT8 scale serves both populations; the step size is set by the "
                "pixel-valued box channels and is what a probability in [0,1] is "
                "quantized with",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=64)
    ap.add_argument("--out", default="xp07_concepts.json")
    args = ap.parse_args()

    model = _calib.load_base().half()
    log("e0", "collecting weight panel")
    weights = weight_panel(model)
    log("e0", f"  {weights['layer']}: widest channel range is "
              f"{weights['widest_over_narrowest']:.1f}x the narrowest")

    log("e0", "collecting activation panel")
    acts = activation_panel(model, args.images)
    for k, v in acts["clips"].items():
        log("e0", f"  {k:17s} represents 0-{v['represented_max']:.4f}")

    log("e0", "collecting Detect output panel")
    det = detect_panel(model, args.images)
    log("e0", f"  channel absmax: "
              + ", ".join(f"{n}={v:.2f}" for n, v in
                          zip(det["channel_names"], det["channel_absmax"])))
    log("e0", f"  one step = {det['per_tensor_step']:.3f}; a probability in [0,1] "
              f"gets {det['probability_levels']:.1f} levels")

    log("e0", "collecting activation histograms for the clipping explainer")
    hists = histogram_panels(model, args.images)
    for n, h in hists.items():
        log("e0", f"  {n:22s} max {h['observed_max']:.3f} | " +
                  ", ".join(f"{k}={v:g}" for k, v in h["clips"].items()))

    _calib.write_json(args.out, {
        "experiment": "xp07_e0_concepts",
        "histograms": hists,
        "purpose": "measured inputs for the explainer figures — every teaching "
                   "picture on the XP7 page is drawn from this detector, not a schematic",
        "resolution": _calib.RES,
        "calibration_list": str(_calib.CALIB_LIST.relative_to(_calib.REPO)),
        "weights": weights,
        "activations": acts,
        "detect_output": det,
        "head_convs": head_conv_names(model),
        "n_convs": len(conv_layers(model)),
    })


if __name__ == "__main__":
    main()
