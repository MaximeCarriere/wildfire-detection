#!/usr/bin/env python3
"""Entropy vs MinMax calibration — the fourth INT8 hypothesis, and the one with
direct evidence behind it.

Decoding the calibration cache (``scale = max_abs / 127``, confirmed by the
anchor-grid constants landing on exactly 0.5) shows TensorRT's default entropy
calibrator assigning the **input tensor** a range of 0.4475 — while the input is
normalised to [0,1] and the calibrator is verifiably fed data reaching 1.0.
Entropy calibration minimises KL divergence and clips outliers to do it; here the
"outliers" are bright sky, which is where faint smoke has to be seen.

MinMax calibration clips nothing. NVIDIA recommends it for detection. This builds
one engine each way, changing nothing else, and scores both against FP16.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from lib import data as dataset                      # noqa: E402
from lib import evaluator                            # noqa: E402
from lib.detectors import TRTDetector, YOLOV5_REPO   # noqa: E402
from lib.trt_export import build_int8_engine, make_calibrator   # noqa: E402

WEIGHTS = REPO / "weights"
RES = 640


def build(algorithm: str, tag: str) -> Path:
    engine = WEIGHTS / f"yolov5s_{tag}_{RES}.engine"
    if engine.exists():
        print(f"skip {engine.name} (exists)")
        return engine
    cache = WEIGHTS / f"yolov5s_{tag}_{RES}.calib"
    cache.unlink(missing_ok=True)
    calib = dataset.load_split("calib")
    calibrator = make_calibrator(calib, RES, YOLOV5_REPO, cache,
                                 batch_size=8, algorithm=algorithm)
    build_int8_engine(WEIGHTS / "yolov5s.onnx", engine, calibrator=calibrator,
                      res=RES, max_batch=16)
    return engine


def main() -> None:
    build("minmax", "int8mm")

    test = [s for s in dataset.load_samples("test") if s.boxes][:120]
    print(f"\nscoring on the same {len(test)} images used for every sanity check")
    print(f"{'engine':30s} {'mAP50':>7s} {'small':>7s} {'tiny':>7s}")
    for name in (f"yolov5s_int8mm_{RES}", f"yolov5s_int8_{RES}",
                 f"yolov5s_int8rand_{RES}", f"yolov5s_fp16_{RES}"):
        p = WEIGHTS / f"{name}.engine"
        if not p.exists():
            print(f"{name:30s} (missing)")
            continue
        det = TRTDetector(p, input_res=RES, fmt="trt", params_m=7.025)
        a = evaluator.evaluate_accuracy(det, test)
        print(f"{name:30s} {a['map50']:7.4f} {a['small_plume']['map50']:7.4f} "
              f"{a['tiny_plume']['map50']:7.4f}", flush=True)


if __name__ == "__main__":
    main()
