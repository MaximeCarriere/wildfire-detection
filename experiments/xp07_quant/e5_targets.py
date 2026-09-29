#!/usr/bin/env python3
"""XP7 E5 — weights-only vs W8A8: where does the accuracy risk actually live?

Axis: **target**.

The two things "INT8" can mean, measured against each other on the same frozen
calibration:

* **W8 (weights only)** — the weights are stored and loaded as 8-bit; the
  arithmetic still happens in float. On a GPU this saves *memory and bandwidth*
  and very little compute: the convolution runs a mixed kernel that dequantizes
  on the way in. Activations are never rounded, so the outlier problem does not
  arise.
* **W8A8** — weights *and* activations, so the multiply-accumulate itself is
  integer. This is the entire 2x-TOPS story on Ampere tensor cores, and it is
  also where all of the accuracy risk is, because activations are data-dependent
  and that is where outliers live.

**Why this is the experiment that keeps the others honest.** If W8 is nearly free
and W8A8 costs real accuracy, then the speed and the accuracy loss are not coming
from the same place, and any "INT8 costs X for Y speedup" claim needs to say which
of the two it means. Both arms go to the board in E4's table, so the compute-vs-
bandwidth split gets measured rather than reasoned about.

A third arm, **A8 only** (activations quantized, weights left float), is included
as the control that isolates the activation contribution directly. It is not a
deployable configuration and is not claimed as one; it is here because with W8 and
W8A8 alone the two effects cannot be separated from their interaction.

Usage
    python e5_targets.py
    python e5_targets.py --val-images 300
    python e5_targets.py --recover-epochs 12
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _arms                                                       # noqa: E402
import _calib                                                      # noqa: E402
from _calib import log                                             # noqa: E402
from _quant import QuantConfig                                     # noqa: E402

BASE = QuantConfig(weight_bits=8, act_bits=8, granularity="per_channel", method="minmax")

ARMS = [
    ("w8_only",  replace(BASE, quantize_weights=True,  quantize_acts=False)),
    ("a8_only",  replace(BASE, quantize_weights=False, quantize_acts=True)),
    ("w8a8",     replace(BASE, quantize_weights=True,  quantize_acts=True)),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-images", type=int, default=0)
    ap.add_argument("--calib-images", type=int, default=512)
    ap.add_argument("--recover-epochs", type=int, default=0)
    ap.add_argument("--out", default="xp07e5_targets.json")
    args = ap.parse_args()

    samples = _calib.split_samples("val", limit=args.val_images)
    baseline = _calib.score(_calib.load_base(), samples, "fp16_baseline")
    log("e5", f"FP16 baseline map50={baseline['map50']:.4f} on {len(samples)} val images")

    rows = [_arms.run_arm(n, c, samples, n_calib=args.calib_images,
                          recover_epochs=args.recover_epochs) for n, c in ARMS]

    got = {r["arm"]: r["damage"]["map50"] for r in rows}
    b = baseline["map50"]
    attribution = {
        "cost_of_weights_pct": round((b - got["w8_only"]) / b * 100, 3),
        "cost_of_activations_pct": round((b - got["a8_only"]) / b * 100, 3),
        "cost_of_both_pct": round((b - got["w8a8"]) / b * 100, 3),
        "interaction_pct": round(((b - got["w8a8"]) - (b - got["w8_only"])
                                  - (b - got["a8_only"])) / b * 100, 3),
    }
    for k, v in attribution.items():
        log("e5", f"{k:26s} {v:+.3f}")

    _calib.write_json(args.out, _arms.payload(
        "xp07_e5_targets", "weights-only vs W8A8 — where does the accuracy risk live?",
        "target", rows, baseline, samples,
        extra={"subsampled": bool(args.val_images), "attribution": attribution,
               "recovery_epochs": args.recover_epochs or None}))


if __name__ == "__main__":
    main()
