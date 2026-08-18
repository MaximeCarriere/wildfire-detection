#!/usr/bin/env python3
"""XP6 E5b — removing channels versus zeroing weights, measured as speed.

E5 answered accuracy and found the counter-intuitive half: this detector keeps
0.7425 mAP50 after 90% of its individual weights are zeroed, and collapses to
0.0956 after 5% of its *channels* are deleted. On accuracy, weights are by far the
cheaper thing to remove.

The obvious follow-up is the other half of the trade. Channel pruning produces a
genuinely narrower network, so it *should* be the one that buys speed, while
zeroed weights sit in a full-size grid and should buy none. That is the standard
story and this page has been repeating it without ever measuring both sides on
the same axis. Two facts already on record make it worth checking rather than
assuming:

* XP6's 25% channel cut measured **381 img/s against the unpruned 474**. Deleting
  a quarter of the channels made the engine *slower*.
* E4 measured 50% of the weights zeroed at essentially unpruned speed, which is
  the expected half, and found TensorRT declining sparse kernels even when the
  pattern allowed them.

So the arms are matched on **fraction of parameters removed** rather than on the
nominal ratio, because a 25% channel cut and a 25% weight cut are not the same
amount of network, and plotted against throughput.

**Speed does not depend on the trained values, only on the shapes**, so the
channel arms are built from the un-retrained architectures. Retraining changes
weights, never widths, and using the raw checkpoints avoids a fine-tune per point
for a number that could not move. Accuracy is never reported from these engines;
E5 and E2 own that, measured through the frozen harness on retrained models.

Usage
    python e5_speed.py                 # build what is missing, then measure
    python e5_speed.py --skip-build
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib import data as dataset                     # noqa: E402
from lib import evaluator                           # noqa: E402
from lib.detectors import YOLOV5_REPO               # noqa: E402
from lib.prune_utils import (detect_head_convs,     # noqa: E402
                             load_yolov5, model_stats)
from lib.trt_export import build_fp16_engine, export_onnx_from_model  # noqa: E402

import _masking                                     # noqa: E402

RAW = REPO / "results" / "raw"
WEIGHTS = REPO / "weights"
BASE = "yolov5s"
RES = 512
RUNS = 7
DENSE_PARAMS = 7.025

# (tag, kind, level). Channel levels are the ratios whose architectures XP6 saved;
# weight levels are sparsities. Both are converted to "fraction of parameters
# removed" at measurement time, which is the only axis on which they compare.
ARMS = [
    ("dense",      "dense",   0.00),
    ("chan25",     "channel", 0.25),
    ("chan50",     "channel", 0.50),
    ("chan70",     "channel", 0.70),
    ("weight50",   "weight",  0.50),
    ("weight90",   "weight",  0.90),
]


def log(m: str) -> None:
    print(f"[e5speed] {m}", flush=True)


def build_variant(tag: str, kind: str, level: float) -> tuple[Path, dict]:
    """Produce the ONNX for one arm and report how much of the model it removed."""
    onnx = WEIGHTS / f"{BASE}_e5b_{tag}.onnx"
    meta_path = WEIGHTS / f"{BASE}_e5b_{tag}.meta.json"
    if onnx.exists() and meta_path.exists():
        return onnx, json.loads(meta_path.read_text())

    if kind == "channel":
        # The architecture is what decides speed, so the un-retrained checkpoint
        # is the right input: same widths, no fine-tune needed.
        src = WEIGHTS / f"{BASE}_pruned{int(level * 100)}_raw.pt"
        if not src.exists():
            raise SystemExit(f"missing {src.name}; run run.py --stage damage first")
        model = load_yolov5(src, YOLOV5_REPO)
        params = model_stats(model, RES)["params_m"]
        meta = {"kind": "channel", "level": level, "params_m": round(params, 4),
                "nonzero_params_m": round(params, 4),
                "params_removed_frac": round(1 - params / DENSE_PARAMS, 4),
                "note": "channels genuinely deleted; the tensors are smaller"}
    else:
        model = load_yolov5(WEIGHTS / f"{BASE}.pt", YOLOV5_REPO)
        if kind == "weight":
            exclude = {id(c) for c in detect_head_convs(model)}
            m = _masking.apply_unstructured(model, level, exclude)
            nz = DENSE_PARAMS * (1 - m["achieved_sparsity"] * m["prunable_weights"]
                                 / (DENSE_PARAMS * 1e6))
            meta = {"kind": "weight", "level": level,
                    "params_m": DENSE_PARAMS,
                    "nonzero_params_m": round(nz, 4),
                    "params_removed_frac": round(1 - nz / DENSE_PARAMS, 4),
                    "achieved_sparsity": m["achieved_sparsity"],
                    "note": "weights zeroed, not removed; the tensors keep their shape"}
        else:
            meta = {"kind": "dense", "level": 0.0, "params_m": DENSE_PARAMS,
                    "nonzero_params_m": DENSE_PARAMS, "params_removed_frac": 0.0,
                    "note": "unpruned control"}

    log(f"  exporting {tag}: {meta['nonzero_params_m']} M non-zero "
        f"({meta['params_removed_frac']:.1%} of the model removed)")
    export_onnx_from_model(model, onnx, res=RES, repo=YOLOV5_REPO)
    meta_path.write_text(json.dumps(meta, indent=2))
    del model
    return onnx, meta


def measure(engine: Path, tag: str, meta: dict) -> dict:
    from lib.detectors import TRTDetector

    test = dataset.load_samples("test")
    det = TRTDetector(engine, input_res=RES, fmt="trt_fp16",
                      params_m=meta["nonzero_params_m"])
    j = evaluator.measure_latency(det, [s.image for s in test[:64]], runs=RUNS)
    j |= evaluator.measure_throughput(det, [s.image for s in test[:32]], runs=RUNS)
    j |= evaluator.measure_memory()
    j["fps_batched_std"] = round(j["fps_batched_se"] * RUNS ** 0.5, 3)
    j["latency_ms_median_std"] = round(j["latency_ms_median_se"] * RUNS ** 0.5, 4)

    rec = evaluator.results_record(
        model_id=f"{BASE}_e5b_{tag}", fmt="trt_fp16",
        params_m=meta["nonzero_params_m"],
        size_disk_mb=engine.stat().st_size / 1e6, input_res=RES,
        accuracy=None, jetson=j,
        notes="XP6 E5b: throughput only, comparing channel removal against weight "
              "zeroing at matched fractions of the model removed. Built from "
              "un-retrained architectures where channels were cut, because widths "
              "decide speed and retraining does not change them. NO ACCURACY is "
              "reported from these engines; E2 and E5 own accuracy, measured on "
              "retrained models through the frozen harness.")
    rec["granularity_meta"] = meta
    evaluator.write_results(rec, RAW / f"xp06e5b_{tag}.json")
    log(f"  {tag}: {j['fps_batched']:.1f} +/- {j['fps_batched_std']:.1f} img/s, "
        f"{meta['nonzero_params_m']} M non-zero, {j.get('power_w_mean')} W")
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-build", action="store_true")
    args = ap.parse_args()

    prepared = []
    for tag, kind, level in ARMS:
        engine = WEIGHTS / f"{BASE}_e5b_{tag}_fp16_{RES}.engine"
        if engine.exists():
            meta_path = WEIGHTS / f"{BASE}_e5b_{tag}.meta.json"
            prepared.append((tag, engine, json.loads(meta_path.read_text())))
            continue
        if args.skip_build:
            log(f"  skipping {tag}, no engine")
            continue
        onnx, meta = build_variant(tag, kind, level)
        log(f"  building engine for {tag} ...")
        build_fp16_engine(onnx, engine, res=RES)
        prepared.append((tag, engine, meta))

    log("")
    for tag, engine, meta in prepared:
        measure(engine, tag, meta)


if __name__ == "__main__":
    main()
