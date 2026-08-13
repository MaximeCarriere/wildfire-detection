#!/usr/bin/env python3
"""XP6 E6 — ratio allocation: same total cut, spread three different ways.

The lecture separates *how much* to prune from *where to spend it*, and gives
three answers: uniform (the same ratio in every layer), global (one threshold
across the network, per-layer ratios fall out), and sensitivity driven (measure
what each layer tolerates, then allocate). XP6 used global and never compared it
to anything.

E1 made that comparison worth running. Its map shows fragility tracking depth
almost perfectly, and, more usefully, shows that **the fragile layers are the
ones that free the least**: halving model.0.conv costs 84% of accuracy to free
0.16% of the parameters, while halving model.21.conv costs 0.9% to free 5.13%.
A global magnitude threshold cannot see any of that. It ranks channels by weight
size, and nothing makes an early-layer channel's weights systematically smaller,
so the budget lands partly where the trade is catastrophic.

**The comparison is held at matched total parameter reduction**, found by search
rather than assumed, so the three arms differ only in *where* the damage lands.
Comparing allocations at equal *ratio* rather than equal *size* would compare
three different models and prove nothing.

**Criterion is L1, deliberately, not LAMP.** LAMP normalises magnitudes per layer
and so already makes an implicit allocation decision; using it here would confound
the one axis this experiment exists to isolate. L1 is a pure per-channel statistic,
so allocation is the only thing that varies.

Usage
    python e6_allocation.py --plan          # show the three allocations, no training
    python e6_allocation.py --arm sensitivity --epochs 12
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _screen import (BASE, RAW, RES, UNPRUNED, fresh_model,  # noqa: E402
                     log, recover_and_record)

CRITERION = "l1"
TARGET_RATIO = 0.25          # the headline ratio, matching XP6 and E2
SAFE_RETENTION = 0.95        # a layer "tolerates" a ratio if it holds this much
TAG = "e6"


def safe_ratios(path: Path = RAW / "xp06e1_sensitivity.json") -> dict:
    """Largest single-layer ratio each layer held at >= SAFE_RETENTION, from E1.

    A layer that failed the threshold even at the mildest ratio gets 0.0: it is
    protected outright. This is the whole point of the arm -- the allocation is
    *measured*, not guessed.
    """
    d = json.loads(path.read_text())
    out: dict[str, float] = {}
    for r in d["rows"]:
        if "retained" not in r:
            continue
        best = out.get(r["layer"], 0.0)
        if r["retained"] >= SAFE_RETENTION and r["ratio"] > best:
            out[r["layer"]] = r["ratio"]
        out.setdefault(r["layer"], 0.0)
    return out


def measure_allocation(build) -> dict:
    """Prune a throwaway model and report what it actually cost. No training."""
    import torch
    from lib.prune_utils import model_stats
    model = fresh_model()
    before = model_stats(model, RES)
    meta = build(model)
    after = model_stats(model, RES)
    del model
    torch.cuda.empty_cache()
    meta |= {"params_m_before": round(before["params_m"], 4),
             "params_m_after": round(after["params_m"], 4),
             "params_reduction": round(1 - after["params_m"] / before["params_m"], 5),
             "gmacs_after": round(after["gmacs"], 4),
             "macs_reduction": round(1 - after["gmacs"] / before["gmacs"], 5)}
    return meta


def build_global(model, ratio=TARGET_RATIO):
    from lib.prune_utils import prune_channels
    m = prune_channels(model, ratio, res=RES, importance=CRITERION, global_pruning=True)
    m["allocation"] = "global"
    return m


def build_uniform(model, ratio):
    from lib.prune_utils import prune_channels
    m = prune_channels(model, ratio, res=RES, importance=CRITERION, global_pruning=False)
    m["allocation"] = "uniform"
    return m


def build_sensitivity(model, scale):
    """Per-layer ratios from E1, scaled to hit the target size."""
    from lib.prune_utils import prune_channels
    safe = safe_ratios()
    named = dict(model.named_modules())
    ratio_dict, applied = {}, {}
    for name, r in safe.items():
        if name not in named:
            continue
        rr = max(0.0, min(0.95, r * scale))
        if rr > 0:
            ratio_dict[named[name]] = rr
            applied[name] = round(rr, 4)
    m = prune_channels(model, 0.0, res=RES, importance=CRITERION,
                       pruning_ratio_dict=ratio_dict, global_pruning=False)
    m["allocation"] = "sensitivity"
    m["scale"] = round(scale, 4)
    m["protected_layers"] = sorted(n for n, r in safe.items() if r == 0.0)
    m["per_layer_ratio"] = applied
    return m


def search(build_fn, target: float, lo: float, hi: float, name: str) -> tuple[float, dict]:
    """Bisect a knob until the achieved parameter reduction matches the target.

    Matching by *measured size* rather than by nominal ratio is what makes the
    three arms comparable; the nominal knob means something different in each.
    """
    best = None
    for i in range(9):
        mid = (lo + hi) / 2
        meta = measure_allocation(lambda m: build_fn(m, mid))
        got = meta["params_reduction"]
        log(TAG, f"  search {name}: knob {mid:.4f} -> {got:.2%} params cut "
                 f"(target {target:.2%})")
        best = (mid, meta)
        if abs(got - target) < 0.004:
            break
        if got < target:
            lo = mid
        else:
            hi = mid
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true", help="show allocations, do not train")
    ap.add_argument("--arm", choices=["global", "uniform", "sensitivity"])
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()

    plan_path = RAW / "xp06e6_allocation_plan.json"

    if args.plan or not plan_path.exists():
        log(TAG, f"criterion {CRITERION}, target = the size global pruning reaches at "
                 f"{TARGET_RATIO:.0%}")
        g = measure_allocation(build_global)
        target = g["params_reduction"]
        log(TAG, f"  global: {g['params_m_after']} M ({target:.2%} cut, "
                 f"MACs -{g['macs_reduction']:.1%})")

        u_knob, u = search(build_uniform, target, 0.05, 0.80, "uniform")
        s_knob, s = search(build_sensitivity, target, 0.05, 3.0, "sensitivity")

        safe = safe_ratios()
        plan = {
            "experiment": "xp06_e6_allocation", "criterion": CRITERION,
            "target_params_reduction": target, "safe_retention_threshold": SAFE_RETENTION,
            "protected_layers": sorted(n for n, r in safe.items() if r == 0.0),
            "n_protected": sum(1 for r in safe.values() if r == 0.0),
            "n_layers": len(safe),
            "uniform_knob": u_knob, "sensitivity_scale": s_knob,
            "arms": {"global": g, "uniform": u, "sensitivity": s},
            "notes": (
                "Allocation comparison at MATCHED achieved parameter reduction, found by "
                "bisection rather than assumed, so the three arms differ only in where the "
                "damage lands. Per-layer ratios come from E1's single-layer sweep on val. "
                "Criterion is L1 rather than LAMP because LAMP normalises per layer and "
                "would confound the allocation axis."),
        }
        plan_path.write_text(json.dumps(plan, indent=2, default=str) + "\n")
        log(TAG, f"wrote {plan_path.name}")
        log(TAG, "")
        log(TAG, f"{'arm':13s} {'params':>9s} {'cut':>8s} {'MACs cut':>9s}")
        for k, v in plan["arms"].items():
            log(TAG, f"{k:13s} {v['params_m_after']:9.3f} {v['params_reduction']:8.2%} "
                     f"{v['macs_reduction']:9.2%}")
        log(TAG, f"protected outright by the sensitivity map: {plan['n_protected']} "
                 f"of {plan['n_layers']} layers")
        if args.plan:
            return

    plan = json.loads(plan_path.read_text())
    if not args.arm:
        return

    arm = args.arm
    log(TAG, f"=== {arm} allocation, {CRITERION}, {args.epochs} epochs ===")
    model = fresh_model()
    if arm == "global":
        meta = build_global(model)
    elif arm == "uniform":
        meta = build_uniform(model, plan["uniform_knob"])
    else:
        meta = build_sensitivity(model, plan["sensitivity_scale"])

    from lib.prune_utils import model_stats
    after = model_stats(model, RES)
    meta["params_m_after"] = round(after["params_m"], 4)
    meta["params_reduction"] = round(1 - after["params_m"] / 7.025023, 5)
    meta["macs_reduction"] = round(1 - after["gmacs"] / 5.075417, 5)
    log(TAG, f"  {meta['params_m_after']} M params ({meta['params_reduction']:.1%} cut), "
             f"MACs -{meta['macs_reduction']:.1%}")

    recover_and_record(
        model, tag=f"dfire_{BASE}_alloc_{arm}", experiment="xp06e6",
        prune_meta=meta, epochs=args.epochs,
        notes=(f"XP6 E6: ratio allocation '{arm}' with the {CRITERION} criterion, matched to "
               f"{plan['target_params_reduction']:.1%} parameter reduction across all three "
               f"arms so only the per-layer distribution varies. Unpruned reference on this "
               f"machine: {UNPRUNED['map50']} mAP50 at {UNPRUNED['params_m']} M."),
        extra={"allocation": arm, "criterion": CRITERION,
               "target_params_reduction": plan["target_params_reduction"]})


if __name__ == "__main__":
    main()
