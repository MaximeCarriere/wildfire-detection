#!/usr/bin/env python3
"""XP10 step 1 — build INT8 engines calibrated on the frozen XP0 calibration set.

Restricted to YOLOv5s: it is the model the resolution frontier showed you would
actually deploy (v5s@512 beats v5l@640 to within 0.7 mAP points at 5.4x the
throughput), and engine builds on this board are expensive enough that spending
them on the dominated model would be poor use of the hardware.

The calibration set is the one XP0 froze — 500 training images stratified to
cover night, fog, backlight and small plumes. That stratification is the whole
experiment: INT8 picks activation ranges from this data, so conditions missing
here are conditions the engine may silently fail on. XP10's slice analysis is
what checks whether it worked.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from lib import data as dataset            # noqa: E402
from lib.detectors import YOLOV5_REPO      # noqa: E402
from lib.trt_export import build_int8_engine, make_calibrator   # noqa: E402

WEIGHTS = REPO / "weights"
RESOLUTIONS = [640, 512]
BASE = "yolov5s"

#: MinMax, not TensorRT's default entropy calibrator. Entropy minimises KL
#: divergence by clipping outliers, and XP10 measured what it clips here: the
#: input tensor's range came out at 0.4475 against a true [0,1] input, saturating
#: every pixel brighter than 0.45 — most of a sky-facing frame. Switching to
#: MinMax took mAP50 from 0.5387 to 0.8497 and tiny-plume from 0.0048 to 0.3204
#: (FP16 reference: 0.9285 / 0.4234). See experiments/xp10_int8_slices/README.md.
ALGORITHM = "minmax"
SUFFIX = "int8mm"


def main() -> None:
    calib_paths = dataset.load_split("calib")
    print(f"calibration set: {len(calib_paths)} images from {dataset.SPLITS/'calib.txt'}")
    missing = [p for p in calib_paths if not p.exists()]
    if missing:
        raise SystemExit(f"{len(missing)} calibration images missing, e.g. {missing[:3]}")

    onnx = WEIGHTS / f"{BASE}.onnx"
    if not onnx.exists():
        raise SystemExit(f"{onnx} not found — export ONNX first")

    for res in RESOLUTIONS:
        engine = WEIGHTS / f"{BASE}_{SUFFIX}_{res}.engine"
        if engine.exists():
            print(f"skip {engine.name} (exists)")
            continue
        # A fresh cache per resolution: activation ranges are resolution-specific,
        # and reusing a 640 cache for a 512 engine would silently mis-scale them.
        cache = WEIGHTS / f"{BASE}_{SUFFIX}_{res}.calib"
        calibrator = make_calibrator(calib_paths, res, YOLOV5_REPO, cache,
                                     batch_size=8, algorithm=ALGORITHM)
        build_int8_engine(onnx, engine, calibrator=calibrator, res=res, max_batch=16)

    print("INT8 ENGINES DONE")


if __name__ == "__main__":
    main()
