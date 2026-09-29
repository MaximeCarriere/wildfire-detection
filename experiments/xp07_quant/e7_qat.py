#!/usr/bin/env python3
"""XP7 E7 — PTQ vs QAT, fairly.

Axis: **effort**. The analog of XP6-E7, with the same fairness discipline.

**What the three words mean.** *PTQ* (post-training quantization) picks scales
from calibration data and never touches the weights. *QAT* (quantization-aware
training) inserts the rounding into the forward pass and then trains through it,
so the weights learn to sit where rounding hurts least — the gradient is passed
straight through ``round`` as if it were the identity, which is the
straight-through estimator. *Recovery* is ordinary fine-tuning applied after
quantization, with the rounding also live.

**The fairness problem, and how it is handled.** Comparing "PTQ, 0 epochs" with
"QAT, 12 epochs" measures quantization method *and* training budget at once, and
credits the whole difference to the method. So three arms:

1. **ptq** — the best PTQ recipe (whatever E2+E3+E6 composed), zero epochs.
2. **ptq_recovered** — the same recipe, then 12 epochs of fine-tuning. This
   controls the training budget.
3. **qat** — fake-quant inserted from the start, 12 epochs.

Arms 2 and 3 get the identical budget — 12 epochs, the same number every XP6
recovery got — so the PTQ-vs-QAT difference is a method difference. Arm 1 is
reported alongside and labelled as the zero-budget point, never silently compared
against a trained arm.

**The guardrail from XP6's bug list.** Before any recovered or QAT number is
believed, the loop must be shown to return the *unquantized* model to its starting
accuracy. A fine-tune that quietly degrades the baseline would make QAT look bad
(or, with a bad baseline, look good) for reasons that have nothing to do with
quantization. ``--prove-loop`` runs that control first and refuses to continue if
the round trip loses more than ``--loop-tolerance`` mAP50.

Expected: at INT8, QAT buys little — PTQ on CNNs is already near-lossless, and if
E2 found calibration barely matters then there is not much left for training to
repair. If E8 runs, QAT is re-asked at 4 bits, where the literature says it earns
its cost.

**Cost warning.** 12 epochs over 15,500 images at 512 px on a Jetson Orin Nano is
a multi-hour job per arm. This script is written to be run deliberately, on the
3090 if one is available, and it states its wall clock in its JSON.

Usage
    python e7_qat.py --prove-loop                    # the guardrail, on its own
    python e7_qat.py --post-epochs 12
    python e7_qat.py --post-epochs 1 --val-images 300 # smoke test the plumbing
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _arms                                                       # noqa: E402
import _calib                                                      # noqa: E402
from _calib import log                                             # noqa: E402
from _quant import QuantConfig, Quantizer                          # noqa: E402

#: The composed best-PTQ recipe. E2 chose minmax, E3 per-channel; E6 decides
#: whether the head stays float. Overridable so the script does not silently
#: encode a stale conclusion.
BEST_PTQ = QuantConfig(weight_bits=8, act_bits=8, granularity="per_channel",
                       method="minmax", act_symmetric=True)


def prove_loop(samples, epochs: int, tolerance: float) -> dict:
    """Fine-tune the *unquantized* model and check it comes back where it started.

    This is a control, not a result. If it fails, every recovered number in XP7 is
    suspect and the run stops.
    """
    from lib.finetune import finetune
    model = _calib.load_base()
    before = _calib.score(model, samples, "loop_before")
    log("e7", f"round-trip control: before map50={before['map50']:.4f}, {epochs} epochs")
    t0 = time.time()
    finetune(model, repo=_calib.YOLOV5_REPO, epochs=epochs, res=_calib.RES, batch=8)
    after = _calib.score(model, samples, "loop_after")
    delta = after["map50"] - before["map50"]
    ok = delta >= -tolerance
    log("e7", f"round-trip control: after map50={after['map50']:.4f} "
              f"(delta {delta:+.4f}) -> {'PASS' if ok else 'FAIL'}")
    del model
    return {"epochs": epochs, "before": before["map50"], "after": after["map50"],
            "delta": round(delta, 4), "tolerance": tolerance, "passed": bool(ok),
            "seconds": round(time.time() - t0, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-images", type=int, default=0)
    ap.add_argument("--calib-images", type=int, default=512)
    ap.add_argument("--post-epochs", type=int, default=12)
    ap.add_argument("--prove-loop", action="store_true")
    ap.add_argument("--loop-tolerance", type=float, default=0.01)
    ap.add_argument("--skip-arms", action="store_true", help="only run the guardrail")
    ap.add_argument("--out", default="xp07e7_qat.json")
    args = ap.parse_args()

    samples = _calib.split_samples("val", limit=args.val_images)
    baseline = _calib.score(_calib.load_base(), samples, "fp16_baseline")
    log("e7", f"FP16 baseline map50={baseline['map50']:.4f} on {len(samples)} val images")

    control = None
    if args.prove_loop:
        control = prove_loop(samples, args.post_epochs, args.loop_tolerance)
        if not control["passed"]:
            _calib.write_json(args.out, {
                "experiment": "xp07_e7_qat", "aborted": True,
                "reason": "the recovery loop does not round-trip the unquantized model; "
                          "no QAT or recovered number from this loop can be believed",
                "round_trip_control": control, "baseline_fp16": baseline})
            raise SystemExit("round-trip control failed — see the JSON")

    rows = []
    if not args.skip_arms:
        # 1. PTQ, zero epochs.
        rows.append(_arms.run_arm("ptq", BEST_PTQ, samples,
                                  n_calib=args.calib_images, recover_epochs=0))
        # 2. PTQ then the same budget QAT gets.
        rows.append(_arms.run_arm("ptq_recovered", BEST_PTQ, samples,
                                  n_calib=args.calib_images,
                                  recover_epochs=args.post_epochs))
        # 3. QAT: the hooks are live for the whole fine-tune, which is what
        #    run_arm's recovery path already does — the distinction from arm 2 is
        #    that QAT trains from the quantized starting point rather than
        #    quantizing a model that was trained in float. On a PTQ-initialised
        #    network with static scales those are the same computation, so the
        #    honest statement is that arm 3 here IS arm 2, and the label is
        #    dropped rather than double-counted.
        log("e7", "arms 2 and 3 coincide for static-scale QAT from a PTQ init — "
                  "see the note in the source; reporting two arms, not three")

    _calib.write_json(args.out, _arms.payload(
        "xp07_e7_qat", "does QAT beat the best PTQ arm at a matched training budget?",
        "effort", rows, baseline, samples,
        extra={"subsampled": bool(args.val_images),
               "recipe": BEST_PTQ.label(),
               "post_epochs": args.post_epochs,
               "round_trip_control": control,
               "fairness_note": "arm 1 is the zero-budget point and is never compared "
                                "against a trained arm without saying so"}))


if __name__ == "__main__":
    main()
