#!/usr/bin/env python3
"""XP6 E9 — the recoverable frontier: damage, one-shot and iterative across the ratio.

E7 compared one-shot against iterative pruning at a single point, 25% of channels,
and found iterative losing. The classic figure this reproduces (Han et al., Deep
Compression) plots the same comparison as a *curve* and shows the two arms lying
on top of each other until roughly 90% of the parameters are gone, separating only
at the extreme. E7's single point sits at 40% removed, which on that curve is the
flat part where no difference is predicted. So E7 answered its question honestly
and in the one regime least able to distinguish the arms.

This sweeps the ratio instead, and measures three series at every point:

* **damage** — pruned, never trained. Costs no training time at all and is the only
  series this repo can produce on the Jetson.
* **one-shot** — cut once, then ``--post-epochs`` of recovery.
* **iterative** — cut in ``--steps`` increments with ``--between`` epochs each,
  then the *same* post-cut budget as one-shot. E7 established that holding the
  post-cut budget equal is what makes the comparison fair; that convention is kept
  here rather than re-litigated.

**Plotted against measured parameter reduction, never the channel ratio.** A 25%
channel cut removes 39.6% of the parameters on this network and the two numbers
are not interchangeable — E5 and E6 both had to make this correction. The default
ratios were chosen by measuring, and land on 24.7 to 95.5% of the parameters, which
spans the reference figure's whole x-range.

**Criterion is L1, not the L2 that E7 used.** E7 held L2 because it was rerunning a
specific published comparison and changing the rule would have fixed a different
experiment. This is a new measurement with no such obligation, and E2 established
L2 is a poor rule on this detector; running a frontier with a criterion known to be
bad would confound "iterative does not help" with "the criterion was wrong four
times instead of once".

**round_to is left at 1 deliberately.** E3's rounding is a real and free speed win,
but it changes the achieved size, and this experiment's x-axis *is* achieved size.
Apply rounding after choosing a point on this curve, not while measuring it.

Usage
    python e9_frontier.py --plan                    # ratios, sizes, time estimate
    python e9_frontier.py --arm damage              # no training, safe anywhere
    python e9_frontier.py --arm oneshot             # ~2.7 h on a 3090
    python e9_frontier.py --arm iterative           # ~4.4 h on a 3090
    python e9_frontier.py --arm oneshot --ratios 0.45 0.55   # resume a subset
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _screen import (BASE, RAW, RES, UNPRUNED, fresh_model,  # noqa: E402
                     log, recover_and_record, score_model)

TAG = "e9"

#: Channel ratios, not parameter ratios. Measured on this network these land on
#: 24.7 / 40.1 / 55.5 / 68.4 / 79.1 / 87.3 / 93.4 / 95.5 percent of the parameters
#: removed, so the sweep covers the reference figure's whole x-range.
#:
#: The last two matter more than their cost suggests. In the reference the two
#: trained arms lie on top of each other until roughly 90% and separate only past
#: it, so a sweep stopping at 87% would reproduce the flat part and miss the only
#: region where iterative pruning is predicted to earn its extra training.
RATIOS = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.80]

CRITERION = "l1"


def _pruned(model, ratio: float, criterion: str) -> dict:
    from lib.prune_utils import prune_channels
    return prune_channels(model, ratio, res=RES, importance=criterion, round_to=1)


def run_plan(ratios, criterion) -> None:
    """Cut at every ratio and report the achieved size. No scoring, no training.

    Worth running first every time: the channel-to-parameter conversion is not
    linear and picking ratios without measuring it is how E2's parameter column
    ended up non-constant.
    """
    rows = []
    for r in ratios:
        model = fresh_model()
        meta = _pruned(model, r, criterion)
        rows.append((r, meta["params_m_after"], meta["params_reduction"],
                     meta["macs_reduction"]))
        del model
    log(TAG, f"{'channel cut':>12s} {'params':>9s} {'params cut':>11s} {'MACs cut':>9s}")
    for r, p, pr, mr in rows:
        log(TAG, f"{r:11.0%} {p:9.3f} {pr:11.2%} {mr:9.2%}")
    log(TAG, "")
    log(TAG, f"time estimate on a 3090 at ~1.6 min/epoch for a pruned model:")
    log(TAG, f"  damage    {len(rows)} x score only            ~{len(rows) * 3:.0f} min")
    log(TAG, f"  oneshot   {len(rows)} x 12 epochs             ~{len(rows) * 20:.0f} min")
    log(TAG, f"  iterative {len(rows)} x (8 between + 12 post) ~{len(rows) * 33:.0f} min")


def run_damage(ratios, criterion) -> None:
    """The unretrained series: what the cut costs before anything repairs it.

    Written as side data with no ``model_id`` because no model is produced and
    nothing here is deployable; the figure reads the file explicitly.
    """
    base = fresh_model()
    ref = score_model(base, "e9_unpruned")
    log(TAG, f"unpruned here: mAP50 {ref['map50']:.4f} (record says {UNPRUNED['map50']})")
    del base

    points = []
    for r in ratios:
        model = fresh_model()
        meta = _pruned(model, r, criterion)
        acc = score_model(model, f"e9_damage_{int(r * 100)}")
        points.append({
            "channel_ratio": r,
            "params_m_after": meta["params_m_after"],
            "params_reduction": meta["params_reduction"],
            "macs_reduction": meta["macs_reduction"],
            "map50": acc["map50"],
            "map50_tiny": acc["tiny_plume"]["map50"],
        })
        log(TAG, f"  cut {r:.0%} -> {meta['params_reduction']:.1%} of params gone, "
                 f"mAP50 {acc['map50']:.4f}")
        del model

    rec = {"experiment": "xp06e9_damage", "base_model": BASE, "input_res": RES,
           "criterion": CRITERION, "granularity": "channels",
           "unpruned_here": {"map50": ref["map50"],
                             "map50_tiny": ref["tiny_plume"]["map50"]},
           "points": points,
           "notes": ("XP6 E9: channel pruning damage across the ratio, no recovery training. "
                     "The unretrained series of the Han-style frontier figure. Scored against "
                     "an unpruned baseline measured on the same machine in the same run, so "
                     "the deltas hold even if the machine is not the screening box. No "
                     "model_id: side data for the E9 figure, not a results record.")}
    path = RAW / "xp06e9_damage.json"
    path.write_text(json.dumps(rec, indent=2) + "\n")
    log(TAG, f"wrote {path.name}")


def run_trained(mode: str, ratios, args) -> None:
    """One record per ratio, through the frozen harness, exactly like E7's arms."""
    from lib.detectors import YOLOV5_REPO
    from lib.finetune import finetune
    from lib.prune_utils import prune_iterative

    for r in ratios:
        pct = int(round(r * 100))
        log(TAG, f"=== {mode} · {r:.0%} channels · {CRITERION} · "
                 f"{args.post_epochs} epochs after the final cut ===")
        model = fresh_model()

        if mode == "oneshot":
            meta = _pruned(model, r, CRITERION)
            meta["method"] = "oneshot"
            total = args.post_epochs
        else:
            between_runs = []

            def between_recover(m):
                between_runs.append(
                    finetune(m, repo=YOLOV5_REPO, epochs=args.between, res=RES,
                             batch=32, log_every=400))

            # final_recover=None on purpose: prune_iterative must NOT train after
            # the last cut, so that recover_and_record supplies the whole post-cut
            # budget and both arms get the same one. This is E7's fix, kept.
            meta = prune_iterative(model, r, steps=args.steps,
                                   recover=between_recover, res=RES,
                                   importance=CRITERION, round_to=1)
            meta["between_epochs_each"] = args.between
            meta["between_epochs_total"] = args.between * args.steps
            meta["between_step_runs"] = between_runs
            total = args.between * args.steps + args.post_epochs

        meta["method"] = mode
        meta["channel_ratio"] = r
        meta["post_cut_epochs"] = args.post_epochs
        meta["total_epochs"] = total
        log(TAG, f"  params {meta['params_m_before']} -> {meta['params_m_after']} M "
                 f"({meta['params_reduction']:.1%}) · MACs -{meta['macs_reduction']:.1%}")

        recover_and_record(
            model, tag=f"dfire_{BASE}_frontier_{mode}_{pct}", experiment="xp06e9",
            prune_meta=meta, epochs=args.post_epochs,
            notes=(f"XP6 E9: {mode} channel pruning at {r:.0%} with the {CRITERION} criterion, "
                   f"{args.post_epochs} epochs after the final cut. One point on the recoverable "
                   f"frontier; the arm is only interpretable against the other points at the "
                   f"same ratio, never alone. Post-cut budget is held equal across arms per E7, "
                   f"so iterative receives more total training ({total}) than one-shot, not "
                   f"less. Criterion is L1 rather than E7's L2 because E2 found L2 poor on this "
                   f"detector and an iterative arm would apply it once per step. Plotted "
                   f"against measured parameter reduction, not the channel ratio. Unpruned "
                   f"reference {UNPRUNED['map50']} mAP50."),
            extra={"arm": mode, "criterion": CRITERION, "channel_ratio": r,
                   "post_cut_epochs": args.post_epochs, "total_epochs": total,
                   "frontier_point": True})
        del model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["damage", "oneshot", "iterative"])
    ap.add_argument("--plan", action="store_true",
                    help="report achieved sizes and a time estimate, run nothing")
    ap.add_argument("--ratios", type=float, nargs="+", default=None,
                    help="override the sweep, e.g. to resume after an interruption")
    ap.add_argument("--post-epochs", type=int, default=12,
                    help="epochs AFTER the final cut -- held equal across arms")
    ap.add_argument("--between", type=int, default=2,
                    help="iterative only: epochs between increments")
    ap.add_argument("--steps", type=int, default=4)
    args = ap.parse_args()

    ratios = args.ratios if args.ratios else RATIOS

    if args.plan or not args.arm:
        run_plan(ratios, CRITERION)
        return
    if args.arm == "damage":
        run_damage(ratios, CRITERION)
    else:
        run_trained(args.arm, ratios, args)


if __name__ == "__main__":
    main()
