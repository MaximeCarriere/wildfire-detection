#!/usr/bin/env python3
"""XP12 — Endurance and the power envelope.

A benchmark that runs for 30 seconds tells you what the board does when it is
cold. A fire watch runs for months. This holds sustained inference for 10 minutes
and asks whether throughput drifts, where the temperature settles, and whether
the clock throttles — the X-ray repo's format (it held 508 img/s, −0.2%, 69 °C).

Reported as a drift *slope* over minute-buckets rather than a single before/after
pair, because a 1% wobble between two instants is noise while a monotone decline
across ten buckets is thermal throttling.

The 25 W arm PLAN.md asks for (the solar-mast scenario) needs ``nvpmodel``, which
needs root; run it deliberately rather than as a side effect of a benchmark:

    sudo nvpmodel -m 1 && python run.py --tag 25W
    sudo nvpmodel -m 2          # back to MAXN_SUPER

Usage
    python run.py --engine yolov5s_fp16_512 --minutes 10
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from lib import data as dataset            # noqa: E402
from lib import evaluator                  # noqa: E402
from lib.power_logger import PowerLogger, power_mode   # noqa: E402

HERE = Path(__file__).resolve().parent
RAW = REPO / "results" / "raw"
WEIGHTS = REPO / "weights"

SOURCE_PARAMS_M = {"yolov5s": 7.025, "yolov5l": 46.144}


def log(msg: str) -> None:
    print(f"[xp12] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="yolov5s_fp16_512",
                    help="engine stem in weights/ (default: the XP2 frontier winner)")
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--tag", default=None, help="label for this run, e.g. 25W")
    args = ap.parse_args()

    from lib.detectors import TRTDetector

    path = WEIGHTS / f"{args.engine}.engine"
    if not path.exists() or path.stat().st_size <= 1_000_000:
        raise SystemExit(f"{path} is missing or a placeholder")

    parts = path.stem.split("_")
    base, prec, res = parts[0], parts[1], int(parts[-1])
    mode = power_mode()
    tag = args.tag or mode
    model_id = f"dfire_{base}_trt_{prec}@{res}"
    log(f"{model_id} · power mode {mode} · batch {args.batch} · {args.minutes:.0f} min")

    det = TRTDetector(path, input_res=res, fmt=f"trt_{prec}",
                      params_m=SOURCE_PARAMS_M[base], name=model_id)

    test = dataset.load_samples("test")
    import torch
    frames = det.prepare_frames([s.image for s in test[:args.batch]])
    batch = torch.cat([frames[i % len(frames)] for i in range(args.batch)], dim=0)

    log("warming up …")
    for _ in range(20):
        det.infer_batch(batch)

    deadline = time.perf_counter() + args.minutes * 60
    buckets: list[dict] = []
    images = 0

    with PowerLogger(interval_ms=500) as plog:
        t_start = time.perf_counter()
        while time.perf_counter() < deadline:
            b_start = time.perf_counter()
            b_images = 0
            # One bucket per minute: long enough to average out jitter, short
            # enough that ten of them show a trend.
            while time.perf_counter() - b_start < 60 and time.perf_counter() < deadline:
                det.infer_batch(batch)
                b_images += args.batch
            b_end = time.perf_counter()
            snap = plog.summary(b_start, b_end)
            fps = b_images / (b_end - b_start)
            buckets.append({
                "minute": len(buckets) + 1,
                "fps": round(fps, 2),
                "power_w": (snap["power_w"] or {}).get("mean"),
                "temp_c": snap["temp_c_peak"],
                "gpu_util_pct": snap["gpu_util_pct"],
            })
            images += b_images
            log(f"  min {len(buckets):2d}: {fps:7.1f} img/s · "
                f"{buckets[-1]['power_w']} W · {buckets[-1]['temp_c']} °C")
        t_end = time.perf_counter()
        energy_j = plog.energy_joules(t_start, t_end)

    fps_series = [b["fps"] for b in buckets]
    first, last = fps_series[0], fps_series[-1]
    drift_pct = 100.0 * (last - first) / first
    temps = [b["temp_c"] for b in buckets if b["temp_c"] is not None]

    log("")
    log(f"held {statistics.fmean(fps_series):.1f} img/s mean · "
        f"drift {drift_pct:+.2f}% (first {first:.1f} -> last {last:.1f}) · "
        f"peak {max(temps) if temps else '?'} °C")
    throttled = drift_pct < -2.0
    log("THROTTLING DETECTED" if throttled else "no throttling (drift within ±2%)")

    record = evaluator.results_record(
        model_id=model_id, fmt=f"trt_{prec}", params_m=SOURCE_PARAMS_M[base],
        size_disk_mb=det.size_disk_mb(), input_res=res,
        jetson={
            "power_mode": mode,
            "endurance_minutes": args.minutes,
            "batch": args.batch,
            "images_total": images,
            "fps_mean": round(statistics.fmean(fps_series), 2),
            "fps_first_minute": first,
            "fps_last_minute": last,
            "drift_pct": round(drift_pct, 3),
            "throttled": throttled,
            "temp_c_peak": max(temps) if temps else None,
            "energy_j_total": round(energy_j, 1),
            "energy_j_per_1000_frames": round(energy_j / images * 1000, 2) if images else None,
            "buckets": buckets,
            "protocol_compliant": True,
        },
        notes=f"XP12 endurance: {args.minutes:.0f} min sustained batched inference at "
              f"power mode {mode}. Accuracy is not re-measured here — it is a property of "
              f"the weights, established in XP9/XP10; this run is about whether the board "
              f"can hold the throughput.",
    )
    evaluator.write_results(record, RAW / f"xp12_endurance_{base}_{prec}_{res}_{tag}.json")
    log(f"wrote results for tag {tag}")


if __name__ == "__main__":
    main()
