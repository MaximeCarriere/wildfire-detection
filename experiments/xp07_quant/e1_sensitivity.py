#!/usr/bin/env python3
"""XP7 E1 — per-layer INT8 sensitivity: which layers can't take it?

Axis: **target**. Mirrors XP6-E1 exactly, one axis over.

XP6 built a map of where pruning damage was cheap and where it was fatal, and
that map turned out to be the whole story: the fragile layers were exactly the
ones that freed the least. This builds the same map for quantization. For each of
the 60 convolutions, quantize **only that one** to INT8, leave the other 59 in
FP16, score, restore, move on.

Two arms per layer, because they separate the two things "INT8" can mean:

* **W8** — the layer's weights only. Activations stay float.
* **W8A8** — the weights *and* the tensor arriving at that layer. This is the
  full integer path, and the literature puts nearly all of the accuracy risk
  here, because activations are where outliers live.

The prediction being tested, from the lecture and from XP10's post-mortem:
convolutions are robust, and the detection head is fragile because YOLOv5's
Detect layer emits box coordinates in pixels alongside objectness and class
probabilities in [0,1]. One scale per tensor cannot serve both — 256 levels
stretched over 0-512 px gives ~2 px of granularity per box edge while the
probabilities are squeezed into a fraction of one step. If that story is right,
the three ``model.24.m.*`` convolutions are the three worst cells in this sweep,
and E6 gets its split for free.

**Val split, never test.** The output configures E6, so scoring it on test would
mean the final number is an exam the configuration has already seen.

**No retraining.** This is the raw numerical damage of the rounding. Recovery is
a separate axis (E7) and confounds this one.

One calibration pass serves all 120 cells — see ``Quantizer.adopt`` for why that
is sound here and not in the whole-network experiments.

Usage
    python e1_sensitivity.py                  # full sweep, both arms, val
    python e1_sensitivity.py --limit 4        # smoke test on 4 layers
    python e1_sensitivity.py --val-images 300 # subsampled val, marked as such
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import _calib                                                      # noqa: E402
from _calib import log                                             # noqa: E402
from _quant import (QuantConfig, Quantizer, conv_layers,            # noqa: E402
                    head_conv_names)

ARMS = {
    "w8":   QuantConfig(weight_bits=8, quantize_acts=False, granularity="per_channel"),
    "w8a8": QuantConfig(weight_bits=8, act_bits=8, granularity="per_channel"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only the first N layers")
    ap.add_argument("--val-images", type=int, default=0, help="subsample val (smoke tests)")
    ap.add_argument("--calib-images", type=int, default=512)
    ap.add_argument("--arms", nargs="*", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--out", default="xp07e1_sensitivity.json")
    args = ap.parse_args()

    samples = _calib.split_samples("val", limit=args.val_images)
    layers = [n for n, _ in conv_layers(_calib.load_base())]
    if args.limit:
        layers = layers[:args.limit]
    log("e1", f"{len(layers)} layers x {len(args.arms)} arms on {len(samples)} val images")

    # --- the FP16 line every row is read against -------------------------- #
    model = _calib.load_base()
    head = set(head_conv_names(model))
    t0 = time.time()
    baseline = _calib.score(model, samples, "fp16_baseline")
    per_eval = time.time() - t0
    log("e1", f"FP16 baseline map50={baseline['map50']:.4f} ({per_eval:.0f}s/eval)")
    log("e1", f"estimated sweep time ~{per_eval * len(layers) * len(args.arms) / 60:.0f} min")

    # --- one calibration pass, all 60 activation scales ------------------- #
    master = Quantizer(model, ARMS["w8a8"])
    _calib.run_calibration(model, master, args.calib_images, tag="e1")
    scales = dict(master.frozen)
    master.restore()

    rows, done = [], 0
    for arm in args.arms:
        cfg = ARMS[arm]
        for name in layers:
            t0 = time.time()
            q = Quantizer(model, cfg, targets=[name]).adopt(scales).enable()
            acc = _calib.score(model, samples, f"e1_{arm}_{name}")
            q.restore()
            done += 1
            drop = (baseline["map50"] - acc["map50"]) / baseline["map50"] * 100
            rows.append({
                "layer": name, "arm": arm, "is_head": name in head,
                "map50": acc["map50"], "map5095": acc["map5095"],
                "small_plume": acc["small_plume"]["map50"],
                "tiny_plume": acc["tiny_plume"]["map50"],
                "rel_drop_pct": round(drop, 3),
                "act_scale": (None if not cfg.quantize_acts
                              else round(float(scales[name][0].flatten()[0]), 8)),
                "seconds": round(time.time() - t0, 1),
            })
            log("e1", f"[{done}/{len(layers) * len(args.arms)}] {arm} {name:28s} "
                      f"map50={acc['map50']:.4f} ({drop:+.1f}%)")

    payload = {
        "experiment": "xp07_e1_sensitivity",
        "question": "per-layer INT8 sensitivity: which layers can't take it?",
        "axis": "target",
        "split": "val",
        "subsampled": bool(args.val_images),
        "n_val_images": len(samples),
        "resolution": _calib.RES,
        "calibration": {"n_images": args.calib_images,
                        "method": ARMS["w8a8"].method,
                        "list": str(_calib.CALIB_LIST.relative_to(_calib.REPO))},
        "baseline_fp16": baseline,
        "head_convs": sorted(head),
        "rows": rows,
    }
    _calib.write_json(args.out, payload)

    for arm in args.arms:
        sub = [r for r in rows if r["arm"] == arm]
        if not sub:
            continue
        worst = sorted(sub, key=lambda r: r["map50"])[:5]
        log("e1", f"--- {arm}: five worst layers ---")
        for r in worst:
            log("e1", f"    {r['layer']:28s} map50={r['map50']:.4f} "
                      f"({r['rel_drop_pct']:+.1f}%) {'HEAD' if r['is_head'] else ''}")


if __name__ == "__main__":
    main()
