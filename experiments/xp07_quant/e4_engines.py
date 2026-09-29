#!/usr/bin/env python3
"""XP7 E4 — the INT8 engine on the board. **Must run on the Jetson.**

Axis: none. This is the payoff measurement, the analog of XP6-E4's speed half,
and the one experiment whose numbers the whole XP reduces to.

Four engines from the same ONNX at 512 px, measured at batch 16 on the Orin with
XP6's discipline (identical build settings, warm-die control, and one arm rebuilt
three times to bound tactic noise before any cross-arm claim is made):

===========================  ==========================================
arm                          precision
===========================  ==========================================
``fp16``                     ``--fp16`` — the 0.7776 @ 474 line
``int8``                     ``--int8``, no float fallback permitted
``int8_fp16``                ``--int8 --fp16`` — TensorRT chooses per layer
``int8_head_fp16``           INT8 with the Detect head pinned to FP16
===========================  ==========================================

**Why "INT8" is a request and not a fact.** XP6's verdict was that the compiler is
the real subject: 2:4 sparsity held its accuracy and then TensorRT simply declined
to use the sparse kernels, so the hardware paid nothing. The same trap exists
here, so every engine is interrogated for **which layers actually ran in INT8**
and which fell back to FP16. "INT8 is 2x faster" is a claim about kernels chosen,
not kernels available.

**Engine size on disk is a result, not bookkeeping.** On an 8 GB shared-memory
board it is a deployment fact, and it is the honest version of "INT8 halves the
model" — INT8 halves the *weights*; the engine adds its own overhead, and the
measured file is what has to fit.

Two build paths exist and they are not the same experiment:

* **(A) QDQ ONNX** — scales computed upstream and carried into the graph as
  Q/DQ node pairs, which TensorRT then executes. Portable, inspectable in Netron,
  and the only path on which E2's and E3's decisions survive into the engine.
* **(B) TensorRT internal calibration** — hand the builder our calibration set
  and let it pick its own ranges. Used **once**, as the "what the vendor does by
  default" control.

Path B is what ``lib/trt_export.py`` already implements and what XP10 measured.
Path A is ``e4_qdq.py``. This script runs B plus the FP16 line.

Accuracy is scored on the **full 4,306-image test set** — these are final numbers,
not configuration choices, so test is the right split here and only here.

Usage
    python e4_engines.py                       # all arms: build, score, measure
    python e4_engines.py --arms fp16 int8_fp16
    python e4_engines.py --variance int8_fp16  # three rebuilds, tactic noise
    python e4_engines.py --skip-accuracy       # speed and size only
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import _calib                                                      # noqa: E402
from _calib import log                                             # noqa: E402

TRTEXEC = "/usr/src/tensorrt/bin/trtexec"
ONNX = _calib.WEIGHTS / "yolov5s.onnx"

#: YOLOv5s parameter count. A property of the *network*, so it is carried over from
#: the source checkpoint rather than read off the engine, which no longer knows it.
PARAMS_M = 7.03

#: name -> (int8 requested, FP16 fallback permitted, Detect head pinned to FP16).
#:
#: The middle flag is not cosmetic. ``build_int8_engine`` used to set the FP16
#: builder flag unconditionally, which made the "INT8 everything" and
#: "INT8 + FP16 fallback" arms build *byte-identical engines* — two rows of a
#: table that could never disagree. ``allow_fp16`` now carries the distinction,
#: so the arm that forbids float fallback actually forbids it.
ARMS = {
    # name              int8    fp16   pin the Detect head to FP16
    "fp16":            (False, True,  False),
    "int8":            (True,  False, False),
    "int8_fp16":       (True,  True,  False),
    "int8_head_fp16":  (True,  True,  True),
}


def engine_path(arm: str) -> Path:
    return _calib.WEIGHTS / f"xp07_yolov5s_{arm}_512.engine"


def layer_precision_report(engine: Path, log_text: str) -> dict:
    """How many layers actually ran in INT8, parsed from the builder's own log.

    This is the XP6 lesson applied: the compiler's report, not our request, is
    the data. ``--verbose`` makes trtexec print the chosen format for every
    layer; the counts below are of what it says it did.
    """
    counts = {"int8": 0, "fp16": 0, "fp32": 0, "other": 0}
    per_layer = []
    for line in log_text.splitlines():
        low = line.lower()
        if "tactic" not in low and "reformat" not in low and "-> " not in low:
            continue
        if "int8" in low:
            counts["int8"] += 1
            per_layer.append(("int8", line.strip()[:160]))
        elif "half" in low or "fp16" in low:
            counts["fp16"] += 1
        elif "float" in low or "fp32" in low:
            counts["fp32"] += 1
        else:
            counts["other"] += 1
    return {"counts": counts, "int8_layer_sample": [l for _, l in per_layer[:10]]}


def build(arm: str, *, res: int = _calib.RES, max_batch: int = 16,
          calib_images: int = 512, rebuild_tag: str = "",
          calib_algorithm: str = "minmax") -> dict:
    """Build one engine and return where it landed plus what the builder said."""
    int8, fp16, pin_head = ARMS[arm]
    out = engine_path(arm) if not rebuild_tag else engine_path(f"{arm}_{rebuild_tag}")
    log_path = out.with_suffix(".build.log")

    t0 = time.time()
    if not int8:
        from lib.trt_export import build_fp16_engine
        build_fp16_engine(ONNX, out, res=res, max_batch=max_batch,
                          trtexec=TRTEXEC, log_path=log_path,
                          detailed_profiling=True)
    else:
        from lib.trt_export import build_int8_engine, make_calibrator
        cache = out.with_suffix(".calib")
        calib = make_calibrator(_calib.calib_paths(calib_images), res,
                                _calib.YOLOV5_REPO, cache, algorithm=calib_algorithm)
        build_int8_engine(ONNX, out, calibrator=calib, res=res, max_batch=max_batch,
                          fp16_head=pin_head, fp16_head_convs=3 if pin_head else 0,
                          allow_fp16=fp16, detailed_profiling=True)
    secs = time.time() - t0
    text = log_path.read_text(errors="ignore") if log_path.exists() else ""
    return {
        "engine": str(out.relative_to(_calib.REPO)),
        "engine_mb": round(out.stat().st_size / 1e6, 2),
        "build_seconds": round(secs, 1),
        "int8_requested": int8,
        "fp16_fallback_allowed": fp16,
        "head_pinned_fp16": pin_head,
        "calibration_algorithm": calib_algorithm if int8 else None,
        # A placeholder. The Python-API build path writes no trtexec log, so this
        # scrape returns zeros; e4_precision.py overwrites it by reading the built
        # engine with TensorRT's inspector, which is the number the plan asks for.
        "precision_report": layer_precision_report(out, text),
    }


def measure(arm: str, engine: Path, samples, *, skip_accuracy: bool,
            energy_batches: int = 200) -> dict:
    """Accuracy, throughput and energy per 1,000 frames for one built engine.

    Energy is integrated over a dedicated inference window rather than reused from
    the throughput run: ``measure_throughput`` interleaves warmup and several timed
    repeats, so a J/1k computed over the whole call would include idle stretches
    the deployed system never has. Here the loop runs flat out for
    ``energy_batches`` batches and the board's VDD_IN rail is integrated over
    exactly that window.
    """
    from lib import evaluator
    from lib.detectors import TRTDetector

    det = TRTDetector(engine, input_res=_calib.RES,
                      fmt="int8" if "int8" in arm else "fp16",
                      params_m=PARAMS_M, source_weights=_calib.BASE_WEIGHTS,
                      name=f"xp07_{arm}")
    row: dict = {}
    if not skip_accuracy:
        acc = _calib.evaluate(det, samples)
        row.update({"map50": acc["map50"], "map5095": acc["map5095"],
                    # Per class as well as aggregate: E9 ranks flame and distant
                    # smoke separately because they rank the techniques oppositely,
                    # and an arm with no per-class numbers drops off those charts.
                    "per_class": acc["per_class"],
                    "small_plume": acc["small_plume"]["map50"],
                    "tiny_plume": acc["tiny_plume"]["map50"],
                    "background": acc["background"],
                    "fingerprint": acc["fingerprint"]})

    images = [s.image for s in samples[:256]]
    row.update(evaluator.measure_throughput(det, images, batch=16))

    try:
        import torch
        from lib.power_logger import PowerLogger, power_mode
        batch = torch.cat(det.prepare_frames(images[:16]), dim=0)
        for _ in range(10):                       # warm the die before integrating
            det.infer_batch(batch)
        with PowerLogger() as plog:
            t0 = time.perf_counter()
            for _ in range(energy_batches):
                det.infer_batch(batch)
            t1 = time.perf_counter()
        n_images = energy_batches * 16
        joules = plog.energy_joules(t0, t1)
        row["power"] = {
            **plog.summary(t0, t1),
            "power_mode": power_mode(),
            "energy_j": round(joules, 2),
            "j_per_1k": round(joules / n_images * 1000, 2) if n_images else None,
            "window_s": round(t1 - t0, 2),
            "images": n_images,
        }
    except Exception as exc:                      # rail unreadable -> say so, do not guess
        row["power"] = {"error": f"{type(exc).__name__}: {exc}", "j_per_1k": None}
    del det
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--calib-images", type=int, default=512)
    ap.add_argument("--test-images", type=int, default=0, help="subsample test (smoke only)")
    ap.add_argument("--skip-accuracy", action="store_true")
    ap.add_argument("--variance", default="", help="arm to rebuild three times")
    ap.add_argument("--calib-algorithm", default="minmax",
                    choices=["minmax", "entropy"],
                    help="TensorRT's own calibrator; E2 found percentile beats both, "
                         "but TRT offers only these two on the path-B builder")
    ap.add_argument("--merge", action="store_true",
                    help="update the existing JSON's arms instead of replacing it, "
                         "so one arm can be re-run without redoing the other three")
    ap.add_argument("--out", default="xp07e4_engines.json")
    args = ap.parse_args()

    if not ONNX.exists():
        raise SystemExit(f"{ONNX} missing — export it first (lib.trt_export.export_onnx_from_model)")

    samples = _calib.split_samples("test", limit=args.test_images)
    log("e4", f"{len(args.arms)} arms on {len(samples)} test images, batch 16")

    rows = {}
    for arm in args.arms:
        log("e4", f"--- {arm} ---")
        info = build(arm, calib_images=args.calib_images,
                     calib_algorithm=args.calib_algorithm)
        info.update(measure(arm, _calib.REPO / info["engine"], samples,
                            skip_accuracy=args.skip_accuracy))
        rows[arm] = info
        log("e4", f"{arm:16s} map50={info.get('map50')} "
                  f"fps={info.get('fps_batched')} size={info['engine_mb']}MB "
                  f"int8_layers={info['precision_report']['counts']['int8']}")

    variance = {}
    if args.variance:
        fps = []
        for k in range(3):
            info = build(args.variance, calib_images=args.calib_images, rebuild_tag=f"v{k}")
            m = measure(args.variance, _calib.REPO / info["engine"], samples,
                        skip_accuracy=True)
            fps.append(m["fps_batched"])
            log("e4", f"variance rebuild {k}: {m['fps_batched']} img/s")
        variance = {"arm": args.variance, "fps": fps,
                    "spread_pct": round((max(fps) - min(fps)) / min(fps) * 100, 2)}

    payload = {
        "experiment": "xp07_e4_engines",
        "question": "the INT8 engine on the board: latency, energy, and what actually ran in INT8",
        "axis": "none — the payoff measurement",
        "board": "Jetson Orin Nano Super",
        "build_path": "B — TensorRT internal calibration (the vendor default control)",
        "split": "test", "n_test_images": len(samples),
        "subsampled": bool(args.test_images),
        "resolution": _calib.RES, "batch": 16,
        "calibration": {"n_images": args.calib_images, "algorithm": "minmax",
                        "list": str(_calib.CALIB_LIST.relative_to(_calib.REPO))},
        "fp16_line": _calib.FP16_LINE,
        "arms": rows,
        "rebuild_variance": variance,
    }
    if args.merge and (_calib.RAW / args.out).exists():
        prev = json.loads((_calib.RAW / args.out).read_text())
        merged = {**(prev.get("arms") or {}), **rows}
        payload = {**prev, **payload, "arms": merged}
        if not variance and prev.get("rebuild_variance"):
            payload["rebuild_variance"] = prev["rebuild_variance"]
        log("e4", f"merged {len(rows)} arm(s) into {len(merged)} existing")
    _calib.write_json(args.out, payload)


if __name__ == "__main__":
    main()
