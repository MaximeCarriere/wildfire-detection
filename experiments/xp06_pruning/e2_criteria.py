#!/usr/bin/env python3
"""XP6 E2 — does the choice of importance criterion matter, and does any of it beat random?

Pruning has to answer "which channels go?", and the lecture offers five families
of answer: magnitude (L1/L2 structural norms), scaling (reuse the batch-norm
gamma, Network Slimming), second order (Taylor and Hessian, in the spirit of
Optimal Brain Damage), geometric median (FPGM), and layer-adaptive magnitude
(LAMP). XP6 used exactly one of them, L2, and never asked whether that choice was
doing any work.

**Random pruning is in this sweep and it is not a joke entry.** It is the control
that separates "this criterion is smart" from "any 25% cut plus retraining lands
here". A leaderboard without it says which criterion won; a leaderboard with it
says whether winning meant anything. Given how badly this model responds to
pruning at all, it may be the most informative run in the set.

**Why the damage stage sweeps low ratios.** XP6 established that a 25% cut with
no retraining scores exactly 0.0000 on test, and a criterion comparison where
every entry reads zero ranks nothing. So damage is measured at 2% and 5% as well,
where the model still produces usable output and the criteria are separable. That
the ranking has to be taken at 2% in order to exist at all is itself a statement
about this detector.

Stage 2 then fine-tunes at 25% one-shot, matching XP6's existing arm so the
result drops straight into the same table.

Selection note: the damage stage runs on **val**, because it chooses which
criteria earn a recovery run. Test is measured too, at the ratio XP6 published,
purely so the numbers are comparable to that page -- it never decides anything.

Usage
    python e2_criteria.py --stage damage
    python e2_criteria.py --stage recover --criteria l2 random bn --epochs 12
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from lib import data as dataset                        # noqa: E402
from lib import evaluator                              # noqa: E402
from lib.detectors import YOLOV5_REPO, Yolov5Detector  # noqa: E402
from lib.prune_utils import (NEEDS_GRADIENTS, accumulate_gradients,   # noqa: E402
                             load_yolov5, model_stats, prune_channels,
                             save_pruned)

RAW = REPO / "results" / "raw"
WEIGHTS = REPO / "weights"
BASE = "yolov5s"
RES = 512
BATCH = 32
CRITERIA = ["l2", "l1", "bn", "taylor", "hessian", "fpgm", "lamp", "random"]
DAMAGE_RATIOS = [0.02, 0.05, 0.25]
HEADLINE_RATIO = 0.25          # XP6's published one-shot arm


def log(msg: str) -> None:
    print(f"[e2] {msg}", flush=True)


def build_pruned(criterion: str, ratio: float):
    """A fresh model pruned by one criterion, plus whatever meta that took."""
    model = load_yolov5(WEIGHTS / f"{BASE}.pt", YOLOV5_REPO)
    grad_meta = None
    if criterion in NEEDS_GRADIENTS:
        # Taylor and Hessian are statistics of the loss, not of the weights, so
        # they need a backward pass before they mean anything. Train split only.
        t0 = time.perf_counter()
        grad_meta = accumulate_gradients(model, repo=YOLOV5_REPO, res=RES, batch=BATCH)
        log(f"    gradients: {grad_meta['grad_images']} train images, "
            f"mean loss {grad_meta['grad_mean_loss']} ({time.perf_counter()-t0:.0f}s)")
    meta = prune_channels(model, ratio, res=RES, importance=criterion,
                          grad_meta=grad_meta)
    return model, meta


def stage_damage(criteria, ratios) -> None:
    import torch

    val = dataset.load_samples("val")
    test = dataset.load_samples("test")
    log(f"{len(val)} val / {len(test)} test images; criteria {criteria}; ratios {ratios}")

    base = load_yolov5(WEIGHTS / f"{BASE}.pt", YOLOV5_REPO)
    base_stats = model_stats(base, RES)
    base_val = evaluator.evaluate_accuracy(
        Yolov5Detector.from_model(base, input_res=RES, name="e2_base"), val)
    log(f"baseline val mAP50 {base_val['map50']:.4f} ({base_stats['params_m']:.3f} M)")
    del base
    torch.cuda.empty_cache()

    rows = []
    for criterion in criteria:
        for ratio in ratios:
            t0 = time.perf_counter()
            log(f"  {criterion} @ {ratio:.0%}")
            row = {"criterion": criterion, "ratio": ratio}
            try:
                model, meta = build_pruned(criterion, ratio)
                acc = evaluator.evaluate_accuracy(
                    Yolov5Detector.from_model(model, input_res=RES,
                                              name=f"e2_{criterion}_{ratio}"), val)
                row |= {
                    "params_m": meta["params_m_after"],
                    "params_reduction": meta["params_reduction"],
                    "macs_reduction": meta["macs_reduction"],
                    "val_map50": acc["map50"],
                    "val_map50_small_plume": acc["small_plume"]["map50"],
                    "val_map50_tiny_plume": acc["tiny_plume"]["map50"],
                    "val_retained": round(acc["map50"] / base_val["map50"], 5),
                }
                # The ratio XP6 published, on the split XP6 published it on, so
                # this table can sit next to that one without an asterisk.
                if abs(ratio - HEADLINE_RATIO) < 1e-9:
                    tacc = evaluator.evaluate_accuracy(
                        Yolov5Detector.from_model(model, input_res=RES,
                                                  name=f"e2_{criterion}_test"), test)
                    row |= {"test_map50": tacc["map50"],
                            "test_map50_small_plume": tacc["small_plume"]["map50"],
                            "test_map50_tiny_plume": tacc["tiny_plume"]["map50"]}
                if grad := {k: v for k, v in meta.items() if k.startswith("grad_")}:
                    row |= grad
                log(f"    val mAP50 {acc['map50']:.4f} ({row['val_retained']:.1%} retained) "
                    f"· {meta['params_m_after']} M · -{meta['macs_reduction']:.1%} MACs "
                    f"({time.perf_counter()-t0:.0f}s)")
                del model
                torch.cuda.empty_cache()
            except Exception as e:
                row |= {"error": f"{type(e).__name__}: {e}"}
                log(f"    FAILED {row['error'][:160]}")
            rows.append(row)

    out = {
        "protocol_version": evaluator.PROTOCOL_VERSION,
        "experiment": "xp06_e2_criteria_damage",
        "input_res": RES, "base_model": BASE, "retrained": False,
        "selection_split": "val", "comparison_split": "test",
        "baseline": {"val_map50": base_val["map50"],
                     "params_m": round(base_stats["params_m"], 4)},
        "criteria": criteria, "ratios": ratios,
        "notes": (
            "3090 screening measurement, off-device. Importance-criterion sweep at fixed "
            "ratios, one-shot, NO retraining. Ranking is taken on val because it selects "
            "which criteria earn a recovery run; test is measured only at the 25% ratio XP6 "
            "published so the two tables are directly comparable. Low ratios are included "
            "because at 25% every criterion scores zero without retraining, which ranks "
            "nothing. 'random' is the control: it says how much of a criterion's benefit is "
            "the criterion rather than the cut plus the retraining."),
        "rows": rows,
    }
    path = RAW / "xp06e2_criteria_damage.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    log(f"wrote {path.relative_to(REPO)}")

    log("")
    log(f"{'criterion':10s} " + " ".join(f"{f'r={r:.0%}':>12s}" for r in ratios))
    for criterion in criteria:
        cells = []
        for r in ratios:
            m = next((x for x in rows if x["criterion"] == criterion and x["ratio"] == r), {})
            cells.append(f"{m['val_map50']:12.4f}" if "val_map50" in m else f"{'ERR':>12s}")
        log(f"{criterion:10s} " + " ".join(cells))


def stage_recover(criteria, epochs: int) -> None:
    import torch
    from lib.finetune import finetune

    test = dataset.load_samples("test")
    for criterion in criteria:
        tag = f"dfire_{BASE}_pruned{int(HEADLINE_RATIO*100)}_{criterion}_recovered"
        log(f"=== {criterion} @ {HEADLINE_RATIO:.0%} one-shot + {epochs} epochs ===")
        t0 = time.perf_counter()
        model, meta = build_pruned(criterion, HEADLINE_RATIO)
        meta["method"] = "oneshot"
        log(f"  params {meta['params_m_before']} -> {meta['params_m_after']} M "
            f"({meta['params_reduction']:.1%}) · MACs -{meta['macs_reduction']:.1%}")

        train_meta = finetune(model, repo=YOLOV5_REPO, epochs=epochs, res=RES, batch=BATCH)
        log(f"  recovery {train_meta['total_minutes']} min, "
            f"final loss {train_meta['history'][-1]['loss']}")

        path = save_pruned(model, WEIGHTS / f"{BASE}_pruned25_{criterion}_recovered.pt", meta)
        acc = evaluator.evaluate_accuracy(
            Yolov5Detector.from_model(model, input_res=RES, name=tag), test)
        log(f"  test mAP50 {acc['map50']:.4f} · small {acc['small_plume']['map50']:.4f} "
            f"· tiny {acc['tiny_plume']['map50']:.4f} "
            f"· silent {acc['background']['correctly_silent_rate']}")

        rec = evaluator.results_record(
            model_id=tag, fmt="pt", params_m=meta["params_m_after"],
            size_disk_mb=path.stat().st_size / 1e6, input_res=RES, accuracy=acc,
            jetson=None,          # rule zero: no speed or power number from a 3090
            notes=(f"XP6 E2: {HEADLINE_RATIO:.0%} channels pruned one-shot with the "
                   f"'{criterion}' importance criterion, {epochs}-epoch recovery at {RES}px, "
                   f"batch {BATCH}. 3090 SCREENING MEASUREMENT, off-device: accuracy only. "
                   f"No latency, throughput or energy number is recorded here because a "
                   f"TensorRT engine is compiled per GPU architecture and a 3090 figure says "
                   f"nothing about a 15 W Orin. torch {torch.__version__}."))
        rec["prune_meta"] = meta
        rec["train_meta"] = train_meta
        rec["machine"] = "rtx3090_screening"
        evaluator.write_results(rec, RAW / f"xp06e2_{tag}.json")
        log(f"  total {(time.perf_counter()-t0)/60:.1f} min")
        del model
        torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["damage", "recover"], default="damage")
    ap.add_argument("--criteria", nargs="*", default=CRITERIA)
    ap.add_argument("--ratios", nargs="*", type=float, default=DAMAGE_RATIOS)
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()

    if args.stage == "damage":
        stage_damage(args.criteria, args.ratios)
    else:
        stage_recover(args.criteria, args.epochs)


if __name__ == "__main__":
    main()
