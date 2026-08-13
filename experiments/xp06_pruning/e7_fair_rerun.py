#!/usr/bin/env python3
"""XP6 E7 — the fair one-shot versus iterative rerun, and a longer budget.

XP6 published a comparison that its own limitations section flags as unclean,
and this fixes it. Both arms were given 12 epochs, but the iterative model
reached its final architecture only after the last cut, so it trained just 4
epochs *in the shape that was measured*, while the one-shot model trained all 12
in its final shape. Equal total budget is not equal recovery budget, and that
alone could produce iterative's deficit without saying anything about iterative
pruning.

The fix is to hold the post-cut budget equal. Both arms now get the same number
of epochs after their final cut. Iterative additionally keeps its between-step
recovery, so it receives *more* total training than one-shot, not less --
deliberately, because the point is to remove every excuse. If iterative still
loses when it has both equal post-cut training and more total training, that is
a real disagreement with the literature rather than a budgeting artifact.

The criterion is L2, matching XP6's published arms exactly. E2 established that
L2 is a poor choice on this detector, but changing it here would fix a different
experiment than the one that needs fixing: this is a clean rerun of a specific
published comparison, not a search for the best model.

The second arm of this experiment asks a different question. XP6 used 12 epochs
because that is what the board can manage overnight. On a desktop GPU 50 epochs
is affordable, and if the verdict against pruning flips at 50 epochs then the
finding is about **who can afford to prune**, not about pruning.

Usage
    python e7_fair_rerun.py --mode oneshot   --post-epochs 12
    python e7_fair_rerun.py --mode iterative --post-epochs 12 --between 2 --steps 4
    python e7_fair_rerun.py --mode oneshot   --post-epochs 50 --criterion lamp
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _screen import (BASE, RES, UNPRUNED, fresh_model,   # noqa: E402
                     log, recover_and_record)

TAG = "e7"
RATIO = 0.25


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["oneshot", "iterative"], required=True)
    ap.add_argument("--post-epochs", type=int, default=12,
                    help="epochs AFTER the final cut -- held equal across arms")
    ap.add_argument("--between", type=int, default=2,
                    help="iterative only: epochs between increments")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--criterion", default="l2")
    args = ap.parse_args()

    from lib.finetune import finetune
    from lib.prune_utils import prune_channels, prune_iterative
    from lib.detectors import YOLOV5_REPO

    label = f"{args.mode}_post{args.post_epochs}"
    if args.criterion != "l2":
        label += f"_{args.criterion}"
    log(TAG, f"=== {args.mode} {RATIO:.0%} · {args.criterion} · "
             f"{args.post_epochs} epochs after the final cut ===")

    model = fresh_model()
    total_epochs = args.post_epochs

    if args.mode == "oneshot":
        meta = prune_channels(model, RATIO, res=RES, importance=args.criterion)
        meta["method"] = "oneshot"
        meta["post_cut_epochs"] = args.post_epochs
        meta["total_epochs"] = args.post_epochs
    else:
        between_runs = []

        def between_recover(m):
            between_runs.append(
                finetune(m, repo=YOLOV5_REPO, epochs=args.between, res=RES,
                         batch=32, log_every=400))

        # final_recover=None means prune_iterative does NOT train after the last
        # cut; recover_and_record then supplies the full post-cut budget below,
        # which is what makes the two arms comparable.
        meta = prune_iterative(model, RATIO, steps=args.steps,
                               recover=between_recover, res=RES,
                               importance=args.criterion)
        meta["between_epochs_each"] = args.between
        meta["between_epochs_total"] = args.between * args.steps
        meta["post_cut_epochs"] = args.post_epochs
        total_epochs = args.between * args.steps + args.post_epochs
        meta["total_epochs"] = total_epochs
        meta["between_step_runs"] = between_runs
        log(TAG, f"  {args.steps} increments x {args.between} epochs = "
                 f"{args.between * args.steps} between-step epochs, then "
                 f"{args.post_epochs} post-cut")

    log(TAG, f"  params {meta['params_m_before']} -> {meta['params_m_after']} M "
             f"({meta['params_reduction']:.1%}) · MACs -{meta['macs_reduction']:.1%}")

    recover_and_record(
        model, tag=f"dfire_{BASE}_fair_{label}", experiment="xp06e7",
        prune_meta=meta, epochs=args.post_epochs,
        notes=(f"XP6 E7: {args.mode} pruning at {RATIO:.0%} with the {args.criterion} "
               f"criterion, given {args.post_epochs} epochs AFTER the final cut. This is the "
               f"confound fix: XP6 gave both arms 12 total epochs, but iterative's final "
               f"architecture existed for only 4 of them while one-shot trained all 12 in its "
               f"final shape. Here the post-cut budget is equal and iterative additionally "
               f"keeps its {meta.get('between_epochs_total', 0)} between-step epochs, so it "
               f"receives MORE total training ({total_epochs}) than one-shot, not less. "
               f"Unpruned reference {UNPRUNED['map50']} mAP50 at {UNPRUNED['params_m']} M."),
        extra={"arm": args.mode, "criterion": args.criterion,
               "post_cut_epochs": args.post_epochs,
               "total_epochs": total_epochs,
               "fixes_confound": "equal epochs after the final cut"})


if __name__ == "__main__":
    main()
