#!/usr/bin/env python3
"""XP9 — TensorRT FP16.

PLAN.md puts TensorRT in Phase 3, after distillation and pruning. XP2 is the
reason it moved here instead: PyTorch eager inference on this board is
**kernel-launch-bound at batch 1** — a YOLOv5s forward pass costs ~22 ms whether
the input is 640x640 or 320x320, and at 320px eight images cost the same
wall-clock as one. In that regime model size barely affects measured speed, so
every speed comparison in Phases 1-2 (XP5's size->speed table, XP8's FLOPs-vs-FPS
plot) would have been measuring the runtime rather than the models.

TensorRT fuses the graph and removes most of those launches. Establishing it
*first* is what makes the later compression numbers mean anything.

Two questions this answers:
  1. How much of the batch-1 latency was launch overhead rather than compute?
  2. Does FP16 quantization cost accuracy? (Expected: ~nothing. Measured anyway,
     including the tiny-plume slice, because "expected" is not "measured".)

Usage
    python run.py                     # every engine found in weights/
    python run.py --engines yolov5s_fp16_640
    python run.py --quick
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from lib import data as dataset          # noqa: E402
from lib import evaluator                # noqa: E402

RAW = REPO / "results" / "raw"
WEIGHTS = REPO / "weights"

#: Parameter counts come from the source checkpoints, not the engines — an engine
#: has no parameter count, and inventing one would put a fabricated number in the
#: schema. Measured in XP1.
SOURCE_PARAMS_M = {"yolov5s": 7.025, "yolov5l": 46.144}


def log(msg: str) -> None:
    print(f"[xp09] {msg}", flush=True)


def _quick_subset(samples: list, n: int = 200) -> list:
    if len(samples) <= n:
        return samples
    return samples[::len(samples) // n][:n]


def parse_engine_name(path: Path) -> dict:
    """weights/yolov5s_fp16_640.engine -> {base, precision, res}."""
    stem = path.stem                      # yolov5s_fp16_640
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"unexpected engine name {path.name}")
    return {"base": parts[0], "precision": parts[1], "res": int(parts[-1])}


def run_engine(path: Path, quick: bool) -> dict:
    from lib.detectors import TRTDetector

    meta = parse_engine_name(path)
    base, prec, res = meta["base"], meta["precision"], meta["res"]
    model_id = f"dfire_{base}_trt_{prec}@{res}"
    log(f"--- {model_id} ({path.name}, {path.stat().st_size / 1e6:.1f} MB) ---")

    det = TRTDetector(path, input_res=res, fmt=f"trt_{prec}",
                      params_m=SOURCE_PARAMS_M[base],
                      source_weights=WEIGHTS / f"{base}.pt", name=model_id)

    test = dataset.load_samples("test")
    acc_images = _quick_subset(test) if quick else test

    t0 = time.perf_counter()
    acc = evaluator.evaluate_accuracy(det, acc_images)
    bg = acc["background"]
    log(f"  mAP50 {acc['map50']} · 50-95 {acc['map5095']} · "
        f"fire {acc['per_class']['fire']['map50']} · smoke {acc['per_class']['smoke']['map50']}")
    log(f"  small {acc['small_plume']['map50']} · tiny {acc['tiny_plume']['map50']} · "
        f"bgFA {bg['false_alarm_rate']} ({time.perf_counter() - t0:.0f}s)")

    timing_kwargs = dict(runs=1, warmup=10, frames=50, allow_short=True) if quick else {}
    tput_kwargs = dict(runs=1, warmup_batches=2, measured_batches=5) if quick else {}
    jetson = evaluator.measure_latency(det, [s.image for s in test[:64]], **timing_kwargs)
    jetson |= evaluator.measure_throughput(det, [s.image for s in test[:32]], **tput_kwargs)
    jetson |= evaluator.measure_memory()
    log(f"  b1 {jetson['latency_ms_median']} ms / {jetson['fps_batch1']} fps · "
        f"bN {jetson['fps_batched']} fps · {jetson.get('power_w_mean')} W")

    record = evaluator.results_record(
        model_id=model_id, fmt=f"trt_{prec}",
        params_m=SOURCE_PARAMS_M[base],
        size_disk_mb=det.size_disk_mb(),
        input_res=res, accuracy=acc, jetson=jetson,
        notes=f"TensorRT {prec.upper()} engine built from the published D-Fire "
              f"{base} checkpoint via ONNX (dynamic batch profile 1-16). Same letterbox, "
              f"same NMS, same thresholds as the PyTorch path, so the comparison isolates "
              f"the runtime. Disk size is the engine, not the checkpoint.",
    )
    if quick:
        record["notes"] += " QUICK MODE: NOT protocol-compliant."

    evaluator.write_results(record, RAW / f"xp09_{base}_trt_{prec}_{res}.json")
    return record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", nargs="*", default=None,
                    help="engine stems, e.g. yolov5s_fp16_640; default = all fp16 engines")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.engines:
        paths = [WEIGHTS / f"{e}.engine" for e in args.engines]
    else:
        paths = sorted(WEIGHTS.glob("*_fp16_*.engine"))
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"missing engines: {[p.name for p in missing]} — build them first")
    # Zero-byte placeholders tell the build script to skip a config (its guard is a
    # plain existence check). They are not engines and must never be deserialized.
    skipped = [p for p in paths if p.stat().st_size <= 1_000_000]
    for p in skipped:
        log(f"skipping placeholder {p.name} ({p.stat().st_size} bytes)")
    paths = [p for p in paths if p.stat().st_size > 1_000_000]
    if not paths:
        raise SystemExit(f"no usable engines found in {WEIGHTS}")

    records = [run_engine(p, args.quick) for p in paths]

    log("")
    log("=== XP9: TensorRT FP16 ===")
    log(f"{'model':34s} {'MB':>7s} {'mAP50':>7s} {'small':>7s} {'tiny':>7s} "
        f"{'b1 ms':>7s} {'b1 fps':>7s} {'bN fps':>8s} {'W':>6s}")
    for r in records:
        j = r["jetson"]
        log(f"{r['model_id']:34s} {r['size_disk_mb']:7.1f} {r['map50_dfire_test']:7.4f} "
            f"{r['map50_small_plume']:7.4f} {(r['map50_tiny_plume'] or 0):7.4f} "
            f"{j['latency_ms_median']:7.2f} {j['fps_batch1']:7.2f} {j['fps_batched']:8.1f} "
            f"{(j.get('power_w_mean') or 0):6.2f}")


if __name__ == "__main__":
    main()
