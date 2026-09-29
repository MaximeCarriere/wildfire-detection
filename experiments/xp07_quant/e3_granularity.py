#!/usr/bin/env python3
"""XP7 E3 — granularity: how many scale factors does this network need?

Axis: **granularity**.

**What granularity means.** A scale is a step size, and it has to be shared by
some set of numbers. Share it across the whole weight tensor (**per-tensor**) and
one unusually large filter sets the step size for every other filter in the layer.
Share it per **output channel** — one scale per filter, which is what "per-channel"
means — and each filter gets a step size matched to its own magnitude.

**Per-channel is free at inference, which is why a win here wins outright.**
Output channel ``c`` of a convolution is a sum of products that all share scale
``S_c``, so ``S_c`` factors straight out of the accumulator and folds into the
bias and batch-norm rescale that follow. The integer kernel is unchanged; nothing
is slower. The only cost is storing ``out_channels`` floats instead of one.

Two arms on the weight axis, activations per-tensor throughout because that is
what the runtime supports:

* **per_tensor** weights
* **per_channel** weights

Then a second decision on the same script, the one Lecture 05 concept this plan
otherwise never tests: **symmetric vs asymmetric activations.**

**What a zero point is for.** Symmetric quantization forces the integer 0 to mean
float 0.0 and spends the range symmetrically about it. Post-activation tensors are
*one-sided* — SiLU is bounded below at about -0.278 and unbounded above — so a
symmetric range spends nearly half its codes on values that never occur. An
asymmetric range adds a zero point ``Z`` that slides the interval onto the data.
The cost is that ``Z != 0`` puts a cross-term into every integer product which has
to be precomputed and added back, so it is not free the way per-channel is.

Weights stay symmetric in every arm: trained conv weights sit roughly
symmetrically about zero, so a zero point buys them almost nothing while costing
the matmul the same cross-term.

Damage is always reported; 12-epoch recovery is behind ``--recover-epochs``
because on this board it is a multi-hour job per arm and XP6 requires the two to
be stated separately rather than blended.

Usage
    python e3_granularity.py
    python e3_granularity.py --val-images 300        # smoke test
    python e3_granularity.py --recover-epochs 12     # adds the recovered column
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

BASE = QuantConfig(weight_bits=8, act_bits=8, method="minmax")

ARMS = [
    ("per_tensor_sym",   replace(BASE, granularity="per_tensor",  act_symmetric=True)),
    ("per_channel_sym",  replace(BASE, granularity="per_channel", act_symmetric=True)),
    ("per_channel_asym", replace(BASE, granularity="per_channel", act_symmetric=False)),
    ("per_tensor_asym",  replace(BASE, granularity="per_tensor",  act_symmetric=False)),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-images", type=int, default=0)
    ap.add_argument("--calib-images", type=int, default=512)
    ap.add_argument("--recover-epochs", type=int, default=0)
    ap.add_argument("--arms", nargs="*", default=[a for a, _ in ARMS])
    ap.add_argument("--out", default="xp07e3_granularity.json")
    args = ap.parse_args()

    samples = _calib.split_samples("val", limit=args.val_images)
    baseline = _calib.score(_calib.load_base(), samples, "fp16_baseline")
    log("e3", f"FP16 baseline map50={baseline['map50']:.4f} on {len(samples)} val images")

    rows = []
    for name, cfg in ARMS:
        if name not in args.arms:
            continue
        rows.append(_arms.run_arm(name, cfg, samples, n_calib=args.calib_images,
                                  recover_epochs=args.recover_epochs))

    # The two questions this script answers, extracted so the page can quote them.
    def find(n):
        return next((r for r in rows if r["arm"] == n), None)

    pt, pc = find("per_tensor_sym"), find("per_channel_sym")
    sym, asym = find("per_channel_sym"), find("per_channel_asym")
    deltas = {
        "per_channel_minus_per_tensor": (
            None if not (pt and pc) else
            round(pc["damage"]["map50"] - pt["damage"]["map50"], 4)),
        "asymmetric_minus_symmetric_acts": (
            None if not (sym and asym) else
            round(asym["damage"]["map50"] - sym["damage"]["map50"], 4)),
    }
    log("e3", f"per_channel - per_tensor = {deltas['per_channel_minus_per_tensor']}")
    log("e3", f"asym - sym activations   = {deltas['asymmetric_minus_symmetric_acts']}")

    _calib.write_json(args.out, _arms.payload(
        "xp07_e3_granularity",
        "per-tensor vs per-channel weight scales, and symmetric vs asymmetric activations",
        "granularity", rows, baseline, samples,
        extra={"subsampled": bool(args.val_images),
               "setting": "whole network W8A8, minmax calibration, weights always symmetric",
               "deltas": deltas,
               "recovery_epochs": args.recover_epochs or None}))


if __name__ == "__main__":
    main()
