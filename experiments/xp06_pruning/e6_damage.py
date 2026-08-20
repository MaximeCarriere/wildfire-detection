#!/usr/bin/env python3
"""XP6 E6b — the three allocations scored *before* any recovery training.

E6 compared uniform, global and sensitivity-driven allocation after 12 epochs of
recovery and found them 0.4 points apart, which is the basis for its "allocation
is second-order" verdict. That verdict rests on a measurement that was never
taken: nobody scored the three models at the moment they were cut. Their records
carry ``map50_before_recovery = None`` while E4 and E5 both store theirs.

The omission matters because E5 established the exact failure mode it invites.
Weight and channel pruning look nothing alike as damage and converge once both
are allowed to retrain, so 12 epochs is demonstrably capable of erasing a real
structural difference on this detector. If allocation behaves the same way, then
E6 measured how much damage retraining can absorb, not how much the allocations
differ -- and those are different claims with the same number attached.

So this scores the damage and changes nothing else. Same plan file, same L1
criterion, same matched sizes found by the same bisection; the only difference is
that no optimizer ever runs.

**Scored on the Jetson, not the 3090 that ran E6.** Accuracy is not a
device-dependent quantity the way throughput is, but FP16 arithmetic on two
different architectures is not bit-identical either, so the unpruned baseline is
re-scored here and every comparison below is against *that* number rather than
against the 0.7764 in the screening records. The two agreeing is a check, not an
assumption.

This writes side data with no ``model_id``, so ``analysis/make_figures.py``
correctly ignores it as a results record and reads it explicitly instead.

Usage
    python e6_damage.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _screen import (BASE, RAW, RES, UNPRUNED, fresh_model,  # noqa: E402
                     log, score_model)

from e6_allocation import (CRITERION, build_global,          # noqa: E402
                           build_sensitivity, build_uniform)

TAG = "e6dmg"


def main() -> None:
    plan_path = RAW / "xp06e6_allocation_plan.json"
    if not plan_path.exists():
        raise SystemExit("run e6_allocation.py --plan first")
    plan = json.loads(plan_path.read_text())

    from lib.prune_utils import model_stats

    # The reference for every delta below, measured on this machine rather than
    # quoted from the screening box.
    base = fresh_model()
    ref = score_model(base, "e6b_unpruned")
    log(TAG, f"unpruned here: mAP50 {ref['map50']:.4f} "
             f"(screening box recorded {UNPRUNED['map50']})")
    del base

    builders = {
        "global": lambda m: build_global(m),
        "uniform": lambda m: build_uniform(m, plan["uniform_knob"]),
        "sensitivity": lambda m: build_sensitivity(m, plan["sensitivity_scale"]),
    }

    arms = {}
    for arm, build in builders.items():
        model = fresh_model()
        meta = build(model)
        after = model_stats(model, RES)
        acc = score_model(model, f"e6b_{arm}_damage")
        arms[arm] = {
            "params_m_after": round(after["params_m"], 4),
            "params_reduction": round(1 - after["params_m"] / 7.025023, 5),
            "map50_damage": acc["map50"],
            "map50_tiny_damage": acc["tiny_plume"]["map50"],
            "map5095_damage": acc["map5095"],
            "allocation": arm,
        }
        log(TAG, f"  {arm:12s} {arms[arm]['params_m_after']} M "
                 f"({arms[arm]['params_reduction']:.1%} cut) -> "
                 f"mAP50 {acc['map50']:.4f} · tiny {acc['tiny_plume']['map50']:.4f}")
        del model

    rec = {
        "experiment": "xp06e6b_damage", "base_model": BASE, "input_res": RES,
        "criterion": CRITERION, "machine": "jetson_orin_nano",
        "unpruned_here": {"map50": ref["map50"],
                          "map50_tiny": ref["tiny_plume"]["map50"]},
        "unpruned_screening_box": UNPRUNED["map50"],
        "arms": arms,
        "notes": (
            "XP6 E6b: the three E6 allocations scored with no recovery training, the "
            "measurement E6's records left at null. Matched sizes and L1 criterion are "
            "unchanged from E6; only the optimizer is absent. Scored on the Orin Nano "
            "with the unpruned baseline re-scored alongside, so every delta is against "
            "a same-machine reference. No model_id: this is side data for the E6 "
            "figure, not a results record."),
    }
    path = RAW / "xp06e6b_damage.json"
    path.write_text(json.dumps(rec, indent=2) + "\n")
    log(TAG, f"wrote {path.name}")


if __name__ == "__main__":
    main()
