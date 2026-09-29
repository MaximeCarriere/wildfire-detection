#!/usr/bin/env python3
"""XP7 E2 — calibration: does the range-picking method matter at INT8?

Axis: **range**. This is quantization's "criterion" experiment — the decision
every toolchain makes silently, and the one XP10 got burned by.

**What calibration is.** A weight tensor's range can be read straight off the
tensor. An activation's cannot: it depends on the data. So the network is run over
sample images and each quantized tensor's range is *measured*. That measurement
fixes the scale, and the scale is then a constant for every inference afterwards
(**static** quantization — what TensorRT executes; dynamic per-inference scaling
is a CPU-runtime feature and is out of scope).

Five methods, all on the whole network at W8A8 with no retraining, so the damage
is attributable to the range choice alone:

* **minmax** — the observed minimum and maximum. Clips nothing, so no data is
  ever saturated; pays for it with a step size set by the single worst outlier.
* **percentile 99.9 / 99.99** — discard the extreme tail by count.
* **entropy (KL)** — TensorRT's default. Picks the clip point that minimises KL
  divergence between the float and quantized distributions. **This is the setting
  XP10 caught costing 52 mAP points**: it decided the network's input tensor,
  normalised to 0-1, only needed 0-0.45, flattening the bright sky that faint
  grey smoke has to be distinguished against.
* **mse** — the clip point that minimises squared reconstruction error. minmax is
  the ``frac = 1.0`` endpoint of the same sweep, which makes the two comparable.

Then, for the winning method only, **how much data**: 8 / 32 / 128 / 512 images.
The sizes are nested prefixes of one frozen list, so a difference between two
sizes is a difference in the amount of data and not in the draw.

**The mechanism is recorded, not just the ranking.** Every arm writes out the
range it chose for every quantized tensor. If the methods tie, the histograms say
why they tied; if one loses, its scale table says where it went wrong. That is how
XP10's bug was actually found — by reading the numbers the tool produced, after
three plausible hypotheses had been tested and eliminated.

The winning (method, size) is frozen here and used by every later arm.

Usage
    python e2_calibration.py                    # methods, then the size sweep
    python e2_calibration.py --skip-sizes
    python e2_calibration.py --val-images 300   # smoke test
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import _calib                                                      # noqa: E402
from _calib import CALIB_SIZES, log                                # noqa: E402
from _quant import (QuantConfig, Quantizer,                        # noqa: E402
                    model_size_report)

BASE = QuantConfig(weight_bits=8, act_bits=8, granularity="per_channel")

METHODS = [
    ("minmax",          replace(BASE, method="minmax")),
    ("percentile_99.9", replace(BASE, method="percentile", percentile=99.9)),
    ("percentile_99.99", replace(BASE, method="percentile", percentile=99.99)),
    ("entropy",         replace(BASE, method="entropy")),
    ("mse",             replace(BASE, method="mse")),
]

#: The tensor XP10's bug lived on: the network input, normalised to [0, 1]. Every
#: arm reports what range it thinks this needs, because that one number explains
#: the entropy result entirely.
INPUT_TENSOR = "model.0.conv"


def run_arm(cfg: QuantConfig, samples, n_calib: int, tag: str) -> dict:
    """Calibrate the whole network with this setting, then score it.

    Each arm gets its own calibration pass. Unlike E1 these are not shareable:
    once every layer is quantized, the tensor arriving at layer N is the *output
    of a quantized* layer N-1, so the distribution to be measured depends on the
    setting being measured.
    """
    model = _calib.load_base()
    q = Quantizer(model, cfg)
    t0 = time.time()
    _calib.run_calibration(model, q, n_calib, tag=tag)
    t_cal = time.time() - t0
    report = q.scale_report()
    size = model_size_report(model, cfg)
    q.enable()
    acc = _calib.score(model, samples, tag)
    q.restore()
    del model
    return {
        "map50": acc["map50"], "map5095": acc["map5095"],
        "small_plume": acc["small_plume"]["map50"],
        "tiny_plume": acc["tiny_plume"]["map50"],
        "background": acc["background"],
        "fingerprint": acc["fingerprint"],
        "input_represented_max": round(report[INPUT_TENSOR]["represented_max"], 6),
        "size": size,
        "calib_seconds": round(t_cal, 1),
        "scales": {k: round(v["represented_max"], 6) for k, v in report.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-images", type=int, default=0)
    ap.add_argument("--methods", nargs="*", default=[m for m, _ in METHODS])
    ap.add_argument("--method-calib-images", type=int, default=512)
    ap.add_argument("--skip-sizes", action="store_true")
    ap.add_argument("--out", default="xp07e2_calibration.json")
    args = ap.parse_args()

    samples = _calib.split_samples("val", limit=args.val_images)
    baseline = _calib.score(_calib.load_base(), samples, "fp16_baseline")
    log("e2", f"FP16 baseline map50={baseline['map50']:.4f} on {len(samples)} val images")

    method_rows = {}
    for name, cfg in METHODS:
        if name not in args.methods:
            continue
        r = run_arm(cfg, samples, args.method_calib_images, f"e2_{name}")
        method_rows[name] = r
        log("e2", f"{name:17s} map50={r['map50']:.4f} tiny={r['tiny_plume']} "
                  f"input_range=0-{r['input_represented_max']:.4f}")

    best = max(method_rows, key=lambda k: method_rows[k]["map50"]) if method_rows else None
    if best:
        log("e2", f"best method: {best} (map50={method_rows[best]['map50']:.4f})")

    size_rows = {}
    if best and not args.skip_sizes:
        best_cfg = dict(METHODS)[best]
        for n in CALIB_SIZES:
            r = run_arm(best_cfg, samples, n, f"e2_{best}_n{n}")
            size_rows[str(n)] = r
            log("e2", f"{best} @ {n:4d} images  map50={r['map50']:.4f} "
                      f"input_range=0-{r['input_represented_max']:.4f}")

    spread = (max(r["map50"] for r in method_rows.values())
              - min(r["map50"] for r in method_rows.values())) if method_rows else None

    payload = {
        "experiment": "xp07_e2_calibration",
        "question": "calibration method and set size — does it matter at INT8?",
        "axis": "range",
        "split": "val",
        "subsampled": bool(args.val_images),
        "n_val_images": len(samples),
        "setting": "whole network W8A8, per-channel weights, no retraining",
        "calibration_list": str(_calib.CALIB_LIST.relative_to(_calib.REPO)),
        "baseline_fp16": baseline,
        "methods": method_rows,
        "method_spread_map50": None if spread is None else round(spread, 4),
        "best_method": best,
        "sizes": size_rows,
        "frozen_choice": {"method": best, "n_images": _calib.CALIB_N},
    }
    _calib.write_json(args.out, payload)


if __name__ == "__main__":
    main()
