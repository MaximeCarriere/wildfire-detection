#!/usr/bin/env python3
"""XP6 stage 2 — prune, recover, deploy, and judge against the real frontier.

The point of pruning is speed, and XP9 established that **speed measured in
PyTorch on this board is meaningless** — it is kernel-launch-bound, so a smaller
model looks exactly as fast as a larger one. A pruned model therefore has to be
taken all the way to a TensorRT engine before its speed means anything, and only
then can it be compared to the line it must beat:

    YOLOv5s TensorRT FP16 @512 — 0.7776 mAP50, 474 img/s, 52 J / 1000 frames

So each level runs the whole chain: prune -> recovery fine-tune -> ONNX ->
TensorRT FP16 -> measure through the frozen harness. Both the PyTorch and the
engine numbers are recorded, because the gap between them is itself XP8's finding.

Usage
    python recover_and_deploy.py --ratio 0.25 --epochs 15
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from lib import data as dataset            # noqa: E402
from lib import evaluator                  # noqa: E402
from lib.detectors import YOLOV5_REPO      # noqa: E402
from lib.prune_utils import (load_yolov5, prune_channels,   # noqa: E402
                             save_pruned)
from lib.trt_export import (build_fp16_engine,              # noqa: E402
                            export_onnx_from_model)

RAW = REPO / "results" / "raw"
WEIGHTS = REPO / "weights"
BASE = "yolov5s"
RES = 512


def log(msg: str) -> None:
    print(f"[xp06] {msg}", flush=True)


def measure(detector, tag: str, fmt: str, params_m: float, size_mb: float,
            prune_meta: dict, train_meta: dict | None, note: str) -> dict:
    test = dataset.load_samples("test")
    t0 = time.perf_counter()
    acc = evaluator.evaluate_accuracy(detector, test)
    log(f"  [{fmt}] mAP50 {acc['map50']} · small {acc['small_plume']['map50']} · "
        f"tiny {acc['tiny_plume']['map50']} · "
        f"bgFA {acc['background']['false_alarm_rate']} ({time.perf_counter()-t0:.0f}s)")

    jetson = evaluator.measure_latency(detector, [s.image for s in test[:64]])
    jetson |= evaluator.measure_throughput(detector, [s.image for s in test[:32]])
    jetson |= evaluator.measure_memory()
    log(f"  [{fmt}] {jetson['latency_ms_median']} ms · {jetson['fps_batched']} fps · "
        f"{jetson.get('power_w_mean')} W · {jetson.get('energy_j_per_1000_frames')} J/1k")

    rec = evaluator.results_record(
        model_id=tag, fmt=fmt, params_m=params_m, size_disk_mb=size_mb,
        input_res=RES, accuracy=acc, jetson=jetson, notes=note)
    rec["prune_meta"] = prune_meta
    rec["train_meta"] = train_meta
    evaluator.write_results(rec, RAW / f"xp06_{tag}.json")
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio", type=float, required=True)
    ap.add_argument("--epochs", type=int, default=15)
    args = ap.parse_args()

    from lib.detectors import TRTDetector, Yolov5Detector
    from lib.finetune import finetune

    pct = int(args.ratio * 100)
    log(f"=== prune {pct}% channels -> recover {args.epochs} epochs -> TensorRT ===")

    model = load_yolov5(WEIGHTS / f"{BASE}.pt", YOLOV5_REPO)
    meta = prune_channels(model, args.ratio, res=RES)
    log(f"  params {meta['params_m_before']} -> {meta['params_m_after']} M "
        f"({meta['params_reduction']:.1%}) · MACs -{meta['macs_reduction']:.1%}")

    train_meta = finetune(model, repo=YOLOV5_REPO, epochs=args.epochs, res=RES, batch=8)
    log(f"  recovery: {train_meta['total_minutes']} min, "
        f"final loss {train_meta['history'][-1]['loss']}")

    pt_path = save_pruned(model, WEIGHTS / f"{BASE}_pruned{pct}_recovered.pt", meta)

    # PyTorch measurement — accuracy is what matters here; the speed number is
    # kept only so the PyTorch-vs-engine gap stays visible.
    det = Yolov5Detector(pt_path, input_res=RES, half=True)
    measure(det, f"dfire_{BASE}_pruned{pct}_recovered", "pt",
            meta["params_m_after"], pt_path.stat().st_size / 1e6, meta, train_meta,
            f"XP6: {pct}% channels pruned, {args.epochs}-epoch recovery fine-tune at {RES}px. "
            f"PyTorch runtime — speed here is kernel-launch-bound (see XP9) and is not the "
            f"number to judge pruning by.")
    del det

    log("  exporting to ONNX and building a TensorRT FP16 engine …")
    onnx = export_onnx_from_model(model, WEIGHTS / f"{BASE}_pruned{pct}.onnx",
                                  res=RES, repo=YOLOV5_REPO)
    del model
    engine = build_fp16_engine(onnx, WEIGHTS / f"{BASE}_pruned{pct}_fp16_{RES}.engine",
                               res=RES)
    log(f"  engine: {engine.name} ({engine.stat().st_size/1e6:.1f} MB)")

    trt_det = TRTDetector(engine, input_res=RES, fmt="trt_fp16",
                          params_m=meta["params_m_after"])
    rec = measure(trt_det, f"dfire_{BASE}_pruned{pct}_recovered_trt", "trt_fp16",
                  meta["params_m_after"], engine.stat().st_size / 1e6, meta, train_meta,
                  f"XP6: {pct}% channels pruned + recovery, TensorRT FP16 @{RES}px. This is "
                  f"the number to judge against the unpruned frontier "
                  f"(0.7776 mAP50, 474 img/s).")

    j = rec["jetson"]
    log("")
    log(f"=== verdict for {pct}% pruning ===")
    log(f"  pruned+recovered TRT : {rec['map50_dfire_test']:.4f} mAP50 · "
        f"{j['fps_batched']:.1f} fps · {j.get('energy_j_per_1000_frames')} J/1k")
    log(f"  unpruned frontier    : 0.7776 mAP50 · 474.0 fps · 52.1 J/1k")
    better = rec["map50_dfire_test"] >= 0.7776 and j["fps_batched"] >= 474.0
    log(f"  beats the dumb knob? {'YES' if better else 'NO'}")


if __name__ == "__main__":
    main()
