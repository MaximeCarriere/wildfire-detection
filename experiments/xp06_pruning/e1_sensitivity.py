#!/usr/bin/env python3
"""XP6 E1 — per-layer sensitivity: which layers can this detector afford to lose?

XP6 found something that does not match the textbook curves: cutting **2%** of
this model's channels costs 9 accuracy points, and 5% costs 88%. Global pruning
picks its victims by one threshold across the whole network, so it had no map of
where the damage was cheap and where it was fatal, and it may simply have been
spending its budget in the worst possible place.

This builds that map. For each prunable convolution, prune **only that layer** at
a sweep of ratios, measure, and move on. A layer whose curve stays flat can
absorb heavy pruning; a layer whose curve falls off a cliff has to be protected.

Two deliberate choices:

* **The val split, never test.** This is a diagnostic whose output feeds the
  allocation in E6, so it is choosing a configuration. Test stays untouched, or
  the final number becomes a score on an exam the configuration has already seen.
* **No retraining.** This measures the raw structural damage of removing a
  layer's channels. Recovery is a separate axis and confounds this one.

Structured pruning removes *groups*, not lone layers: torch-pruning's dependency
graph couples each convolution to its batch norm and to anything a residual add
or concatenation ties it to. So "prune layer N" means "prune the group layer N
belongs to", and the achieved parameter reduction is recorded per row rather than
assumed from the ratio.

Output is one JSON holding the whole grid, not one file per cell -- 300 cells of
a diagnostic sweep are a single result, and analysis/make_figures.py draws the
heatmap from it.

Usage
    python e1_sensitivity.py                      # full sweep
    python e1_sensitivity.py --limit 4            # smoke test on 4 layers
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from lib import data as dataset                    # noqa: E402
from lib import evaluator                          # noqa: E402
from lib.detectors import YOLOV5_REPO, Yolov5Detector   # noqa: E402
from lib.prune_utils import (load_yolov5, model_stats,  # noqa: E402
                             prunable_layers)

RAW = REPO / "results" / "raw"
WEIGHTS = REPO / "weights"
BASE = "yolov5s"
RES = 512
RATIOS = [0.1, 0.2, 0.3, 0.5, 0.7]
SPLIT = "val"


def log(msg: str) -> None:
    print(f"[e1] {msg}", flush=True)


def score(model, samples, tag: str) -> dict:
    """Val accuracy for a live model, through the frozen harness."""
    det = Yolov5Detector.from_model(model, input_res=RES, half=True, name=tag)
    acc = evaluator.evaluate_accuracy(det, samples)
    del det
    return acc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only the first N layers")
    ap.add_argument("--ratios", nargs="*", type=float, default=RATIOS)
    args = ap.parse_args()

    import torch
    from lib.prune_utils import _build_pruner, _make_importance

    samples = dataset.load_samples(SPLIT)
    log(f"{len(samples)} {SPLIT} images, ratios {args.ratios}")

    # Baseline on the same split, so every cell is a delta against a number
    # measured here rather than one quoted from another page.
    base_model = load_yolov5(WEIGHTS / f"{BASE}.pt", YOLOV5_REPO)
    base_stats = model_stats(base_model, RES)
    layers = prunable_layers(base_model)
    names = [n for n, _ in layers]
    if args.limit:
        names = names[:args.limit]
    log(f"{len(prunable_layers(base_model))} prunable convolutions, sweeping {len(names)}")

    t0 = time.perf_counter()
    base_acc = score(base_model, samples, "e1_baseline")
    log(f"baseline {SPLIT} mAP50 {base_acc['map50']:.4f} "
        f"({base_stats['params_m']:.3f} M params, {time.perf_counter()-t0:.0f}s)")
    del base_model
    torch.cuda.empty_cache()

    rows = []
    total = len(names) * len(args.ratios)
    done = 0
    for name in names:
        for ratio in args.ratios:
            done += 1
            # Reload every cell: pruning is destructive and irreversible.
            model = load_yolov5(WEIGHTS / f"{BASE}.pt", YOLOV5_REPO)
            target = dict(model.named_modules())[name]
            example = torch.randn(1, 3, RES, RES, device="cuda:0")
            row = {"layer": name, "ratio": ratio}
            try:
                model.eval()
                pruner = _build_pruner(
                    model, example, _make_importance("l2"),
                    ratio=0.0,                       # everything else untouched
                    round_to=1,
                    pruning_ratio_dict={target: ratio},
                    global_pruning=False,            # the dict means exactly what it says
                )
                pruner.step()
                after = model_stats(model, RES)
                acc = score(model, samples, f"e1_{name}_{ratio}")
                row |= {
                    "params_m": round(after["params_m"], 4),
                    "params_reduction": round(
                        1 - after["params_m"] / base_stats["params_m"], 5),
                    "gmacs": round(after["gmacs"], 4),
                    "macs_reduction": round(1 - after["gmacs"] / base_stats["gmacs"], 5),
                    "map50": acc["map50"],
                    "map50_small_plume": acc["small_plume"]["map50"],
                    "map50_tiny_plume": acc["tiny_plume"]["map50"],
                    "retained": round(acc["map50"] / base_acc["map50"], 5),
                }
                log(f"  [{done:4d}/{total}] {name:34s} r={ratio:.1f} "
                    f"mAP50 {acc['map50']:.4f} ({row['retained']:.1%} retained, "
                    f"-{row['params_reduction']:.2%} params)")
            except Exception as e:
                # A layer that cannot be pruned at this ratio is a finding, not a
                # crash: record why and carry on.
                row |= {"error": f"{type(e).__name__}: {e}"}
                log(f"  [{done:4d}/{total}] {name:34s} r={ratio:.1f} SKIPPED {row['error'][:90]}")
            rows.append(row)
            del model
            torch.cuda.empty_cache()

    out = {
        "protocol_version": evaluator.PROTOCOL_VERSION,
        "experiment": "xp06_e1_sensitivity",
        "split": SPLIT,
        "input_res": RES,
        "base_model": BASE,
        "importance": "l2",
        "retrained": False,
        "baseline": {
            "map50": base_acc["map50"],
            "map50_small_plume": base_acc["small_plume"]["map50"],
            "map50_tiny_plume": base_acc["tiny_plume"]["map50"],
            "params_m": round(base_stats["params_m"], 4),
            "gmacs": round(base_stats["gmacs"], 4),
        },
        "ratios": args.ratios,
        "n_layers_swept": len(names),
        "minutes": round((time.perf_counter() - t0) / 60, 2),
        "notes": (
            f"3090 screening measurement, off-device. Per-layer sensitivity: each row "
            f"prunes ONE convolution group at the stated ratio with everything else left "
            f"alone, no retraining, scored on the {SPLIT} split ({len(samples)} images) so "
            f"that test stays clean for the configuration this sweep selects. Structured "
            f"pruning removes dependency groups, so the achieved parameter reduction is "
            f"measured per row rather than inferred from the ratio."),
        "rows": rows,
    }
    path = RAW / "xp06e1_sensitivity.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    log(f"wrote {path.relative_to(REPO)} ({out['minutes']} min)")

    ok = [r for r in rows if "map50" in r]
    if ok:
        log("")
        log("most fragile layers (mean accuracy retained across ratios):")
        by_layer: dict[str, list] = {}
        for r in ok:
            by_layer.setdefault(r["layer"], []).append(r["retained"])
        ranked = sorted(by_layer.items(), key=lambda kv: sum(kv[1]) / len(kv[1]))
        for name, vals in ranked[:10]:
            log(f"  {name:36s} {sum(vals)/len(vals):.1%}")
        log("most tolerant:")
        for name, vals in ranked[-5:]:
            log(f"  {name:36s} {sum(vals)/len(vals):.1%}")


if __name__ == "__main__":
    main()
