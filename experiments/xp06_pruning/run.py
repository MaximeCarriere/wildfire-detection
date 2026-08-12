#!/usr/bin/env python3
"""XP6 — Structured channel pruning, with and without recovery.

The question the whole series is pointed at: **does cutting the model beat simply
feeding it a smaller image?** XP2 set the bar — YOLOv5s at 512 px, and under
TensorRT that is 0.7776 mAP50 at 474 img/s. A pruned model has to land above that
line to have earned its complexity.

Two stages, because they answer different things and cost wildly different amounts:

* ``--stage damage`` — prune and measure immediately, no retraining. Shows the raw
  structural damage and takes minutes.
* ``--stage recover`` — the same models after a recovery fine-tune. Measured at
  ~7 min/epoch (512 px, batch 8), so a 20-epoch recovery is ~2.5 h per level.

The teacher-supervised recovery arm PLAN.md asks for is **not** implemented: it
needs XP3's distillation machinery, and XP1 found the teacher only 1.4 mAP points
ahead of the student, so that arm is expensive and unpromising. Stated, not hidden.

Usage
    python run.py --stage damage
    python run.py --stage recover --ratios 0.25 0.5 --epochs 20
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from lib import data as dataset          # noqa: E402
from lib import evaluator                # noqa: E402
from lib.detectors import YOLOV5_REPO    # noqa: E402
from lib.prune_utils import (load_yolov5, model_stats,   # noqa: E402
                             prune_channels, save_pruned)

HERE = Path(__file__).resolve().parent
RAW = REPO / "results" / "raw"
WEIGHTS = REPO / "weights"
BASE = "yolov5s"
RATIOS = [0.25, 0.5, 0.7]
RES = 512                # the deployment resolution, per XP2/XP9


def log(msg: str) -> None:
    print(f"[xp06] {msg}", flush=True)


def _quick(samples, n=250):
    return samples[::max(1, len(samples) // n)][:n]


def measure(model_path: Path, tag: str, prune_meta: dict, quick: bool,
            train_meta: dict | None = None) -> dict:
    """Score a pruned checkpoint through the frozen harness."""
    from lib.detectors import Yolov5Detector

    det = Yolov5Detector(model_path, input_res=RES, half=True, name=tag)
    test = dataset.load_samples("test")
    acc_images = _quick(test) if quick else test

    t0 = time.perf_counter()
    acc = evaluator.evaluate_accuracy(det, acc_images)
    log(f"  mAP50 {acc['map50']} · small {acc['small_plume']['map50']} · "
        f"tiny {acc['tiny_plume']['map50']} · bgFA {acc['background']['false_alarm_rate']} "
        f"({time.perf_counter()-t0:.0f}s)")

    tk = dict(runs=1, warmup=10, frames=50, allow_short=True) if quick else {}
    pk = dict(runs=1, warmup_batches=2, measured_batches=5) if quick else {}
    jetson = evaluator.measure_latency(det, [s.image for s in test[:64]], **tk)
    jetson |= evaluator.measure_throughput(det, [s.image for s in test[:32]], **pk)
    jetson |= evaluator.measure_memory()
    log(f"  {jetson['latency_ms_median']} ms · {jetson['fps_batched']} fps batched")

    rec = evaluator.results_record(
        model_id=tag, fmt="pt", params_m=prune_meta["params_m_after"],
        size_disk_mb=model_path.stat().st_size / 1e6, input_res=RES,
        accuracy=acc, jetson=jetson,
        notes=f"XP6 structured channel pruning of {BASE} at {RES}px. "
              f"Requested channel ratio {prune_meta['requested_channel_ratio']}; achieved "
              f"{prune_meta['params_reduction']:.1%} parameter and "
              f"{prune_meta['macs_reduction']:.1%} MAC reduction. "
              + ("Recovery fine-tuned." if train_meta else "NO recovery fine-tune — this is "
                 "the raw structural damage."),
    )
    rec["prune_meta"] = prune_meta
    rec["train_meta"] = train_meta
    if quick:
        rec["notes"] += " QUICK MODE: not protocol-compliant."
    evaluator.write_results(rec, RAW / f"xp06_{tag}.json")
    return rec


def stage_damage(ratios, quick: bool) -> list[dict]:
    out = []
    for r in ratios:
        tag = f"dfire_{BASE}_pruned{int(r*100)}_nofinetune"
        log(f"=== prune {r:.0%} channels, no recovery ===")
        model = load_yolov5(WEIGHTS / f"{BASE}.pt", YOLOV5_REPO)
        meta = prune_channels(model, r, res=RES)
        log(f"  params {meta['params_m_before']} -> {meta['params_m_after']} M "
            f"({meta['params_reduction']:.1%} cut) · MACs {meta['macs_reduction']:.1%} cut")
        path = save_pruned(model, WEIGHTS / f"{BASE}_pruned{int(r*100)}_raw.pt", meta)
        del model
        out.append(measure(path, tag, meta, quick))
    return out


def stage_recover(ratios, epochs: int, quick: bool) -> list[dict]:
    from lib.finetune import finetune

    out = []
    for r in ratios:
        tag = f"dfire_{BASE}_pruned{int(r*100)}_recovered"
        log(f"=== prune {r:.0%} channels + {epochs}-epoch recovery ===")
        model = load_yolov5(WEIGHTS / f"{BASE}.pt", YOLOV5_REPO)
        meta = prune_channels(model, r, res=RES)
        log(f"  params {meta['params_m_before']} -> {meta['params_m_after']} M; "
            f"fine-tuning …")
        train_meta = finetune(model, repo=YOLOV5_REPO, epochs=epochs, res=RES, batch=8)
        log(f"  recovery took {train_meta['total_minutes']} min")
        path = save_pruned(model, WEIGHTS / f"{BASE}_pruned{int(r*100)}_recovered.pt", meta)
        del model
        out.append(measure(path, tag, meta, quick, train_meta))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["damage", "recover"], default="damage")
    ap.add_argument("--ratios", nargs="*", type=float, default=RATIOS)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    recs = (stage_damage(args.ratios, args.quick) if args.stage == "damage"
            else stage_recover(args.ratios, args.epochs, args.quick))

    log("")
    log(f"=== XP6 {args.stage} · the line to beat is FP16@512: 0.7776 mAP50 ===")
    log(f"{'model':44s} {'params':>7s} {'MACcut':>7s} {'mAP50':>7s} {'small':>7s} "
        f"{'tiny':>7s} {'fps':>7s}")
    for r in recs:
        j, pm = r["jetson"], r["prune_meta"]
        log(f"{r['model_id']:44s} {r['params_m']:7.2f} {pm['macs_reduction']:7.1%} "
            f"{r['map50_dfire_test']:7.4f} {r['map50_small_plume']:7.4f} "
            f"{(r['map50_tiny_plume'] or 0):7.4f} {j['fps_batched']:7.1f}")


if __name__ == "__main__":
    main()
