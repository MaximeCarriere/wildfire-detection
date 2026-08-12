#!/usr/bin/env python3
"""XP1 — Baselines: the D-Fire authors' published YOLOv5s / YOLOv5l.

PLAN.md's XP1 called for fine-tuning YOLOv8m/s/n ourselves on a rented GPU. This
runs the *published* D-Fire models first instead, for a reason that costs nothing
and settles a lot: they were trained by the dataset's own authors on D-Fire's
official train split, and because XP0 preserved that official split, our test
split is provably outside their training data. So they can be measured honestly,
today, with no GPU rental — and the result is the accuracy ceiling the whole
compression arc starts from.

What this does NOT give us: the YOLOv8n-class control floor ("just use a smaller
model"), which still needs one training run. That gap is stated in the README
rather than papered over.

Usage
    python run.py                 # both models, full protocol
    python run.py --models s      # one of them
    python run.py --quick         # wiring check, NOT a measurement
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


def log(msg: str) -> None:
    print(f"[xp01] {msg}", flush=True)


def _quick_subset(samples: list, n: int = 200) -> list:
    """Strided, not [:n] — D-Fire's filenames are grouped by source, so the first
    200 test images carry no ground truth at all (see XP0)."""
    if len(samples) <= n:
        return samples
    return samples[::len(samples) // n][:n]


def run_model(key: str, quick: bool) -> dict:
    from lib.detectors import Yolov5Detector

    spec = MODELS[key]
    weights = REPO / "weights" / spec["weights"]
    log(f"=== {spec['model_id']} ({weights.name}) ===")

    det = Yolov5Detector(weights, input_res=evaluator.DEFAULT_INPUT_RES, half=True,
                         name=spec["model_id"])
    log(f"loaded: {det.params_m():.3f} M params · {det.size_disk_mb():.2f} MB · "
        f"classes {det.names}")

    test = dataset.load_samples("test")
    acc_images = _quick_subset(test) if quick else test

    log(f"accuracy over {len(acc_images)} test images (threshold-free) …")
    t0 = time.perf_counter()
    acc = evaluator.evaluate_accuracy(det, acc_images)
    log(f"  mAP50 {acc['map50']} · mAP50-95 {acc['map5095']} "
        f"({time.perf_counter() - t0:.1f}s)")
    log(f"  fire {acc['per_class']['fire']['map50']} · "
        f"smoke {acc['per_class']['smoke']['map50']} · "
        f"small-plume {acc['small_plume']['map50']}")

    log("timing …")
    timing_kwargs = dict(runs=1, warmup=10, frames=50, allow_short=True) if quick else {}
    jetson = evaluator.measure_latency(det, [s.image for s in test[:64]], **timing_kwargs)
    # Batch-1 latency on this box is launch-bound and says little about the model;
    # the batched number is what exposes compute cost. Both are recorded.
    tput_kwargs = dict(runs=1, warmup_batches=2, measured_batches=5) if quick else {}
    jetson |= evaluator.measure_throughput(det, [s.image for s in test[:32]], **tput_kwargs)
    jetson |= evaluator.measure_memory()
    log(f"  {jetson['latency_ms_median']} ms median · {jetson['fps_batch1']} fps · "
        f"{jetson.get('power_w_mean')} W · {jetson.get('temp_c_peak')} °C")

    log("re-running accuracy for the bit-reproducibility assertion …")
    second = evaluator.evaluate_accuracy(det, acc_images)
    if second["fingerprint"] != acc["fingerprint"]:
        raise SystemExit(
            f"{spec['model_id']}: predictions not bit-reproducible "
            f"({acc['fingerprint']} vs {second['fingerprint']}) — no accuracy number from "
            f"this run can be trusted.")
    log(f"  reproducible ✓ {acc['fingerprint']}")

    record = evaluator.results_record(
        model_id=spec["model_id"],
        fmt="pt",
        params_m=det.params_m(),
        size_disk_mb=det.size_disk_mb(),
        accuracy=acc,
        jetson=jetson,
        notes="Published D-Fire weights from the dataset authors "
              "(github.com/pedbrgs/Fire-Detection, MIT), trained on D-Fire's official train "
              "split; XP0 preserved that split, so this test set is outside their training "
              "data. Anchor-based YOLOv5, not the anchor-free 'u' variants. Trained with the "
              "authors' recipe, not ours — see README for what that costs.",
    )
    if quick:
        record["notes"] += " QUICK MODE: reduced counts, NOT protocol-compliant."

    evaluator.write_results(record, RAW / f"xp01_{spec['model_id']}.json")
    return record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["s", "l"], choices=["s", "l"])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    records = [run_model(k, args.quick) for k in args.models]

    log("")
    log("=== XP1 baseline table ===")
    hdr = f"{'model':28s} {'params':>8s} {'mAP50':>7s} {'50-95':>7s} {'fire':>7s} {'smoke':>7s} {'small':>7s} {'ms':>7s} {'fps':>7s}"
    log(hdr)
    for r in records:
        j = r["jetson"]
        log(f"{r['model_id']:28s} {r['params_m']:8.2f} {r['map50_dfire_test']:7.4f} "
            f"{r['map5095_dfire_test']:7.4f} {r['map50_fire_class']:7.4f} "
            f"{r['map50_smoke_class']:7.4f} {r['map50_small_plume']:7.4f} "
            f"{j['latency_ms_median']:7.2f} {j['fps_batch1']:7.2f}")


if __name__ == "__main__":
    main()
