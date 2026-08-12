#!/usr/bin/env python3
"""Is the INT8 collapse caused by the calibration set, not the decode?

XP10's INT8 engines lose ~42% of mAP50 against FP16, with tiny-plume accuracy
~99% gone. Pinning the Detect head and the decode tail to FP16 moved that barely
at all (0.5211 -> 0.5387 on the sanity subset), which rules the decode out as the
main cause and points upstream.

The remaining suspect is the calibration data itself. XP0 built the frozen
calibration set to PLAN.md §1's instruction that it "MUST include: night, fog,
sunset/backlight, small distant plumes" — and it does, to the tune of **450 of
500 images**: 150 small-plume, 100 night (lowest-15% luminance), 100 fog
(lowest-15% contrast), 100 backlight, and only 50 random. Calibration sets the
activation ranges, so a set dominated by dark, low-contrast frames calibrates
ranges far narrower than ordinary daylight needs, and everything else clips.

This builds a second engine calibrated on 500 *uniformly random* training images,
changing nothing else, and scores both against FP16 on the same subset. If the
random-calibration engine recovers, the cause is calibration-set composition —
a finding about how to build calibration sets, not about INT8.
"""
from __future__ import annotations

import random
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


def main() -> None:
    train = dataset.load_split("train")
    rnd = random.Random(dataset.SEED)
    random_calib = sorted(rnd.sample([str(p) for p in train], 500))
    (dataset.SPLITS / "calib_random.txt").write_text(
        "\n".join(str(Path(p).relative_to(dataset.DATA)) for p in random_calib) + "\n")
    print(f"random calibration set: {len(random_calib)} images (seed {dataset.SEED})")

    engine = WEIGHTS / f"yolov5s_int8rand_{RES}.engine"
    if not engine.exists():
        cache = WEIGHTS / f"yolov5s_int8rand_{RES}.calib"
        cache.unlink(missing_ok=True)     # must NOT reuse the stratified ranges
        calibrator = make_calibrator([Path(p) for p in random_calib], RES,
                                     YOLOV5_REPO, cache, batch_size=8)
        build_int8_engine(WEIGHTS / "yolov5s.onnx", engine, calibrator=calibrator,
                          res=RES, max_batch=16)

    test = [s for s in dataset.load_samples("test") if s.boxes][:120]
    print(f"\nscoring on the same {len(test)} images used for every sanity check")
    print(f"{'engine':28s} {'mAP50':>7s} {'small':>7s} {'tiny':>7s}")
    for name in (f"yolov5s_int8rand_{RES}", f"yolov5s_int8_{RES}", f"yolov5s_fp16_{RES}"):
        p = WEIGHTS / f"{name}.engine"
        if not p.exists():
            print(f"{name:28s} (missing)")
            continue
        det = TRTDetector(p, input_res=RES, fmt="trt", params_m=7.025)
        a = evaluator.evaluate_accuracy(det, test)
        print(f"{name:28s} {a['map50']:7.4f} {a['small_plume']['map50']:7.4f} "
              f"{a['tiny_plume']['map50']:7.4f}", flush=True)


if __name__ == "__main__":
    main()
