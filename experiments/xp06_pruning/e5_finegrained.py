#!/usr/bin/env python3
"""XP6 E5 — fine-grained pruning: is the collapse capacity, or is it structure?

This is the experiment that decides which sentence the write-up is allowed to
end on. XP6 measured channel pruning destroying this detector and concluded
pruning loses. But there are two very different reasons that could happen:

  * **capacity** -- the detector needs all 7 M parameters and any real reduction
    breaks it, in which case pruning is genuinely hopeless here; or
  * **structure** -- the capacity is spare, but removing it in whole-channel
    units is what the network cannot survive.

Channel pruning alone cannot tell those apart. Fine-grained pruning can, because
it is the same amount of capacity removed with none of the structural
constraint: any individual weight, anywhere, no pattern. If the network tolerates
90% of its weights being deleted while dying at a 5% channel cut, the capacity
was never the problem.

**No speed number is produced and none should be expected.** Scattered zeros sit
inside a full-size tensor; the GPU multiplies it exactly as fast as before. This
granularity reaches the highest compression ratios in the literature and delivers
them on custom hardware, not on this board. That is a statement about hardware,
not a hedge, and the README says it in those terms.

Usage
    python e5_finegrained.py --sparsity 0.9 --epochs 12
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _masking import (apply_unstructured, reapply_masks,    # noqa: E402
                      strip_masks, verify_sparsity)
from _screen import (BASE, RES, UNPRUNED, fresh_model,     # noqa: E402
                     log, recover_and_record, score_model)

TAG = "e5"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sparsity", type=float, required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--damage-only", action="store_true",
                    help="mask and score, no recovery -- fills the granularity curve cheaply")
    args = ap.parse_args()

    from lib.prune_utils import detect_head_convs

    pct = int(round(args.sparsity * 100))
    log(TAG, f"=== fine-grained (unstructured) {pct}% sparsity + {args.epochs} epochs ===")

    model = fresh_model()
    # The detect head is excluded for the same reason it is in every other arm:
    # its channel count is fixed by the output format. Masking individual weights
    # there would not break the shape, but keeping the exclusion identical across
    # granularities is what makes the granularity comparison mean anything.
    meta = apply_unstructured(model, args.sparsity,
                              exclude={id(m) for m in detect_head_convs(model)})
    log(TAG, f"  zeroed {meta['zeroed_weights']:,} of {meta['prunable_weights']:,} "
             f"prunable weights ({meta['achieved_sparsity']:.2%})")

    # Damage before any retraining, so the granularity curves are comparable to
    # XP6's channel-pruning damage table row for row.
    acc0 = score_model(model, f"e5_{pct}_nofinetune")
    log(TAG, f"  no retraining: mAP50 {acc0['map50']:.4f} "
             f"(unpruned {UNPRUNED['map50']}) · tiny {acc0['tiny_plume']['map50']:.4f}")
    meta["map50_before_recovery"] = acc0["map50"]
    meta["map50_small_before_recovery"] = acc0["small_plume"]["map50"]
    meta["map50_tiny_before_recovery"] = acc0["tiny_plume"]["map50"]

    if args.damage_only:
        # The granularity comparison against channel pruning is a DAMAGE curve,
        # and damage costs one forward pass over the test set rather than an
        # hour of training. Recorded separately so it can never be mistaken for
        # a recovered result.
        from lib import evaluator
        from _screen import RAW
        meta.update(verify_sparsity(model))    # the non-zero count is part of the result
        rec = evaluator.results_record(
            model_id=f"dfire_{BASE}_finegrained{pct}_nofinetune", fmt="pt",
            params_m=meta.get("nonzero_params_m", 0.0), size_disk_mb=0.0,
            input_res=RES, accuracy=acc0, jetson=None,
            notes=(f"XP6 E5: fine-grained global magnitude pruning at {pct}% sparsity, "
                   f"NO recovery training. Weights are masked, not removed. Accuracy only; "
                   f"no speedup exists for irregular sparsity on this hardware. 3090 "
                   f"screening measurement, off-device."))
        rec["prune_meta"] = meta
        rec["train_meta"] = None
        rec["machine"] = "rtx3090_screening"
        rec["experiment"] = "xp06e5"
        rec["granularity"] = "unstructured"
        rec["requested_sparsity"] = args.sparsity
        rec["speed_measured"] = False
        evaluator.write_results(rec, RAW / f"xp06e5_dfire_{BASE}_finegrained{pct}_nofinetune.json")
        log(TAG, f"  damage-only record written: mAP50 {acc0['map50']:.4f}")
        return

    def after_training(m):
        """Re-impose the mask on the EMA weights, verify, then bake it in.

        Order matters and each step earns its place: the EMA never ran a forward
        pass so it carries no sparsity (see _masking.reapply_masks), the verify
        proves the exported weights really are sparse rather than assumed to be,
        and stripping last removes the hooks so the checkpoint can be pickled.
        """
        drift = reapply_masks(m)
        v = verify_sparsity(m) | drift
        log(TAG, f"  EMA carried {drift['ema_sparsity_before_remask']:.2%} sparsity; "
                 f"after re-masking {v['measured_sparsity']:.2%} across "
                 f"{v['masked_layers']} layers · {v['nonzero_params_m']} M non-zero of "
                 f"{v['dense_params_m']} M dense")
        if abs(v["measured_sparsity"] - meta["achieved_sparsity"]) > 0.02:
            log(TAG, "  WARNING: sparsity did not reach the target even after re-masking")
        strip_masks(m)
        return v

    rec = recover_and_record(
        model, tag=f"dfire_{BASE}_finegrained{pct}", experiment="xp06e5",
        prune_meta=meta, epochs=args.epochs, post_train=after_training,
        notes=(f"XP6 E5: fine-grained (unstructured) global magnitude pruning at {pct}% "
               f"sparsity. Weights are MASKED, not removed, so the tensor keeps its shape "
               f"and the network is not smaller in memory or faster in time. NO SPEED NUMBER "
               f"IS REPORTED FOR THIS ARM ON ANY MACHINE: irregular sparsity has no matching "
               f"kernels here. Accuracy only, to establish whether this detector's collapse "
               f"under channel pruning is a capacity limit or a structural one. Unpruned "
               f"reference {UNPRUNED['map50']} mAP50."),
        extra={"granularity": "unstructured", "requested_sparsity": args.sparsity,
               "speed_measured": False,
               "speed_note": ("no speedup expected or measured: irregular sparsity has no "
                              "matching kernels on this hardware")})

    log(TAG, f"  done: mAP50 {rec['map50_dfire_test']:.4f} at "
             f"{meta['achieved_sparsity']:.0%} sparsity")


if __name__ == "__main__":
    main()
