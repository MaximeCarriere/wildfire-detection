#!/usr/bin/env python3
"""XP2 — Resolution sweep: the null hypothesis of model compression.

PLAN.md puts this second, before any clever technique, for a reason: input
resolution is the cheapest knob there is. Drop it and you get speed for free, no
retraining, no toolchain, no research. **Every later technique — distillation,
pruning, 2:4 sparsity, INT8 — has to beat this frontier to justify its
complexity.** If plain 416px YOLOv5s dominates a pruned 640px model, the series
says so.

This is the *inference-resolution* arm: the models are evaluated at resolutions
they were never trained for. That is the honest cheap version of the knob — what
you get by editing one number in a config. A retrained-at-resolution arm (which
recovers some of the loss, at the cost of a training run per resolution) is the
separate second half of XP2 and needs a GPU.

The counter-metric matters as much as the headline. Low resolution is exactly
what kills distant plumes, so ``map50_small_plume`` and the tighter
``map50_tiny_plume`` are what turn "3x faster" into "3x faster and blind to the
thing you built it for".

Usage
    python run.py                          # both models x 4 resolutions
    python run.py --models s --res 640 320
    python run.py --quick                  # wiring check, NOT a measurement
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

HERE = Path(__file__).resolve().parent
RAW = REPO / "results" / "raw"

MODELS = {
    "s": {"weights": "yolov5s.pt", "model_id": "dfire_yolov5s_published"},
    "l": {"weights": "yolov5l.pt", "model_id": "dfire_yolov5l_published"},
}
# 160 px is deliberately past the point of usefulness. The sweep from 640 down to
# 320 shows a knob worth turning; without a level where the detector plainly fails,
# a reader can reasonably ask whether the curve ever bends. It does, and 160 px is
# where. All values are multiples of 32, the network's output stride.
RESOLUTIONS = [640, 512, 416, 320, 256, 160]


def log(msg: str) -> None:
    print(f"[xp02] {msg}", flush=True)


def _quick_subset(samples: list, n: int = 200) -> list:
    if len(samples) <= n:
        return samples
    return samples[::len(samples) // n][:n]


def run_one(key: str, res: int, quick: bool, check_repro: bool) -> dict:
    from lib.detectors import Yolov5Detector

    spec = MODELS[key]
    weights = REPO / "weights" / spec["weights"]
    model_id = f"{spec['model_id']}@{res}"
    log(f"--- {model_id} ---")

    det = Yolov5Detector(weights, input_res=res, half=True, name=model_id)

    test = dataset.load_samples("test")
    acc_images = _quick_subset(test) if quick else test

    t0 = time.perf_counter()
    acc = evaluator.evaluate_accuracy(det, acc_images)
    bg = acc["background"]
    log(f"  mAP50 {acc['map50']} · 50-95 {acc['map5095']} · "
        f"fire {acc['per_class']['fire']['map50']} · smoke {acc['per_class']['smoke']['map50']}")
    log(f"  small-plume {acc['small_plume']['map50']} · tiny-plume {acc['tiny_plume']['map50']} · "
        f"bg false-alarm {bg['false_alarm_rate']} ({time.perf_counter() - t0:.0f}s)")

    timing_kwargs = dict(runs=1, warmup=10, frames=50, allow_short=True) if quick else {}
    jetson = evaluator.measure_latency(det, [s.image for s in test[:64]], **timing_kwargs)
    # Batch-1 latency on this box is launch-bound and says little about the model;
    # the batched number is what exposes compute cost. Both are recorded.
    tput_kwargs = dict(runs=1, warmup_batches=2, measured_batches=5) if quick else {}
    jetson |= evaluator.measure_throughput(det, [s.image for s in test[:32]], **tput_kwargs)
    jetson |= evaluator.measure_memory()
    log(f"  {jetson['latency_ms_median']} ms · {jetson['fps_batch1']} fps · "
        f"{jetson.get('power_w_mean')} W")

    # Bit-reproducibility is a property of the harness, not of the resolution;
    # asserting it once per model keeps the sweep affordable without dropping it.
    if check_repro:
        second = evaluator.evaluate_accuracy(det, acc_images)
        if second["fingerprint"] != acc["fingerprint"]:
            raise SystemExit(f"{model_id}: predictions not bit-reproducible — results void.")
        log(f"  reproducible ✓ {acc['fingerprint']}")

    record = evaluator.results_record(
        model_id=model_id,
        fmt="pt",
        params_m=det.params_m(),
        size_disk_mb=det.size_disk_mb(),
        input_res=res,
        accuracy=acc,
        jetson=jetson,
        notes=f"XP2 inference-resolution arm: published D-Fire weights evaluated at {res}px "
              f"without retraining. Params and disk size are unchanged by resolution — the "
              f"speed comes entirely from the smaller input.",
    )
    if quick:
        record["notes"] += " QUICK MODE: reduced counts, NOT protocol-compliant."

    evaluator.write_results(record, RAW / f"xp02_{spec['model_id']}_{res}.json")
    return record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["s", "l"], choices=["s", "l"])
    ap.add_argument("--res", nargs="*", type=int, default=RESOLUTIONS)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    records = []
    for key in args.models:
        for i, res in enumerate(args.res):
            records.append(run_one(key, res, args.quick, check_repro=(i == 0)))

    log("")
    log("=== XP2 resolution frontier ===")
    log(f"{'model':34s} {'res':>5s} {'mAP50':>7s} {'small':>7s} {'tiny':>7s} {'bgFA':>7s} "
        f"{'b1 ms':>7s} {'b1 fps':>7s} {'bN fps':>8s} {'W':>6s}")
    for r in records:
        j = r["jetson"]
        log(f"{r['model_id']:34s} {r['input_res']:5d} {r['map50_dfire_test']:7.4f} "
            f"{r['map50_small_plume']:7.4f} "
            f"{(r['map50_tiny_plume'] if r['map50_tiny_plume'] is not None else float('nan')):7.4f} "
            f"{r['bg_false_alarm_rate']:7.4f} "
            f"{j['latency_ms_median']:7.2f} {j['fps_batch1']:7.2f} {j['fps_batched']:8.1f} "
            f"{(j.get('power_w_mean') or float('nan')):6.2f}")
    log("b1 = batch 1 (launch-bound on this box) · bN = batched (compute-bound, the real axis)")


if __name__ == "__main__":
    main()
