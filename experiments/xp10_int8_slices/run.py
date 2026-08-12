#!/usr/bin/env python3
"""XP10 — INT8 PTQ and the failure-slice analysis.

Aggregate mAP is the number that gets quoted and the number that hides things.
XP2 already showed it: dropping 640->320 cost 6% of aggregate mAP50 and 77% of
tiny-plume accuracy. The same question is now asked of INT8 quantization —
**does it quietly kill early detection while the headline number looks fine?**

So every INT8 engine is scored not once but several times: aggregate, per class,
the two plume size tiers, and the shooting conditions the calibration set was
stratified to cover (night / fog / backlight). Each INT8 engine is measured
alongside its FP16 twin at the same resolution, so the delta isolates
quantization rather than mixing in the resolution effect.

The condition slices reuse the *exact* luminance proxies XP0 used to build the
calibration set (lib.data.image_stats), so "the conditions we calibrated for" and
"the conditions we score" are the same definition rather than two similar ones.

Usage
    python run.py                 # every int8 engine, vs its fp16 twin
    python run.py --quick
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from lib import data as dataset          # noqa: E402
from lib import evaluator                # noqa: E402

HERE = Path(__file__).resolve().parent
RAW = REPO / "results" / "raw"
WEIGHTS = REPO / "weights"

SOURCE_PARAMS_M = {"yolov5s": 7.025, "yolov5l": 46.144}

#: Quantile width for each condition slice, matching the calibration stratification.
SLICE_FRAC = 0.15


def log(msg: str) -> None:
    print(f"[xp10] {msg}", flush=True)


def _quick_subset(samples: list, n: int = 250) -> list:
    if len(samples) <= n:
        return samples
    return samples[::len(samples) // n][:n]


def condition_slices(samples: list, cache: Path) -> dict[str, set[str]]:
    """Partition the test split by shooting condition, using XP0's proxies.

    Luminance stats over 4,306 images take a couple of minutes, so they are
    cached — they are a property of the dataset, not of any model, and every
    engine scored here must use identical slices or the comparison is void.
    """
    if cache.exists():
        stats = json.loads(cache.read_text())
    else:
        log(f"computing luminance stats for {len(samples)} test images (cached after) …")
        t0 = time.perf_counter()
        stats = {s.rel: dataset.image_stats(s.image) for s in samples}
        cache.write_text(json.dumps(stats))
        log(f"  done in {time.perf_counter() - t0:.0f}s")

    n = max(1, int(len(samples) * SLICE_FRAC))

    def lowest(key):
        return {s.rel for s in sorted(samples, key=lambda s: (stats[s.rel][key], s.rel))[:n]}

    def widest_spread():
        return {s.rel for s in sorted(
            samples, key=lambda s: (-(stats[s.rel]["p95"] - stats[s.rel]["mean"]), s.rel))[:n]}

    return {"night": lowest("mean"), "fog": lowest("std"), "backlight": widest_spread()}


def score_slices(det, samples: list, slices: dict[str, set[str]]) -> dict:
    """Aggregate + per-condition accuracy."""
    by_rel = {s.rel: s for s in samples}
    out = {"all": evaluator.evaluate_accuracy(det, samples)}
    for name, rels in slices.items():
        subset = [by_rel[r] for r in sorted(rels) if r in by_rel]
        if subset:
            out[name] = evaluator.evaluate_accuracy(det, subset)
    return out


def run_engine(path: Path, samples: list, slices: dict, quick: bool) -> dict:
    from lib.detectors import TRTDetector

    parts = path.stem.split("_")           # yolov5s_int8_640
    base, prec, res = parts[0], parts[1], int(parts[-1])
    model_id = f"dfire_{base}_trt_{prec}@{res}"
    log(f"--- {model_id} ({path.stat().st_size / 1e6:.1f} MB) ---")

    det = TRTDetector(path, input_res=res, fmt=f"trt_{prec}",
                      params_m=SOURCE_PARAMS_M[base],
                      source_weights=WEIGHTS / f"{base}.pt", name=model_id)

    scored = score_slices(det, samples, slices)
    acc = scored["all"]
    log(f"  ALL      mAP50 {acc['map50']} · small {acc['small_plume']['map50']} · "
        f"tiny {acc['tiny_plume']['map50']} · bgFA {acc['background']['false_alarm_rate']}")
    for name in ("night", "fog", "backlight"):
        if name in scored:
            s = scored[name]
            log(f"  {name:8s} mAP50 {s['map50']} · small {s['small_plume']['map50']} · "
                f"tiny {s['tiny_plume']['map50']}")

    timing_kwargs = dict(runs=1, warmup=10, frames=50, allow_short=True) if quick else {}
    tput_kwargs = dict(runs=1, warmup_batches=2, measured_batches=5) if quick else {}
    jetson = evaluator.measure_latency(det, [s.image for s in samples[:64]], **timing_kwargs)
    jetson |= evaluator.measure_throughput(det, [s.image for s in samples[:32]], **tput_kwargs)
    jetson |= evaluator.measure_memory()
    log(f"  b1 {jetson['latency_ms_median']} ms · bN {jetson['fps_batched']} fps · "
        f"{jetson.get('power_w_mean')} W")

    record = evaluator.results_record(
        model_id=model_id, fmt=f"trt_{prec}", params_m=SOURCE_PARAMS_M[base],
        size_disk_mb=det.size_disk_mb(), input_res=res, accuracy=acc, jetson=jetson,
        notes=f"TensorRT {prec.upper()} engine. INT8 is calibrated on the frozen XP0 "
              f"500-image calibration set (night/fog/backlight/small-plume stratified) and "
              f"built with FP16 fallback enabled, which is the configuration anyone actually "
              f"deploys. Condition slices in slices_detail.",
    )
    record["slices_detail"] = {
        name: {"map50": s["map50"], "map50_small_plume": s["small_plume"]["map50"],
               "map50_tiny_plume": s["tiny_plume"]["map50"],
               "map50_fire": s["per_class"]["fire"]["map50"],
               "map50_smoke": s["per_class"]["smoke"]["map50"],
               "n_images": s["n_images"],
               "bg_false_alarm_rate": (s["background"] or {}).get("false_alarm_rate")}
        for name, s in scored.items()
    }
    if quick:
        record["notes"] += " QUICK MODE: NOT protocol-compliant."

    evaluator.write_results(record, RAW / f"xp10_{base}_{prec}_{res}.json")
    return record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", nargs="*", default=None)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.engines:
        paths = [WEIGHTS / f"{e}.engine" for e in args.engines]
    else:
        # INT8 engines and their FP16 twins at the same resolutions, so the delta
        # attributable to quantization can be read straight off the table.
        int8 = sorted(WEIGHTS.glob("*_int8_*.engine"))
        twins = [WEIGHTS / p.name.replace("_int8_", "_fp16_") for p in int8]
        paths = int8 + [t for t in twins if t.exists()]
    # Zero-byte placeholders are used to tell the build script to skip a config
    # (its guard is a plain existence check). Never try to deserialize one.
    paths = [p for p in paths if p.exists() and p.stat().st_size > 1_000_000]
    if not paths:
        raise SystemExit("no INT8 engines found — run scripts/build_int8.py first")

    samples = dataset.load_samples("test")
    if args.quick:
        samples = _quick_subset(samples)
    cache = HERE / f"condition_stats{'_quick' if args.quick else ''}.json"
    slices = condition_slices(samples, cache)
    log("slices: " + " · ".join(f"{k}={len(v)}" for k, v in slices.items())
        + f" (of {len(samples)} test images)")

    records = [run_engine(p, samples, slices, args.quick) for p in paths]

    log("")
    log("=== XP10: INT8 vs FP16, sliced by condition (mAP50) ===")
    log(f"{'model':30s} {'MB':>6s} {'all':>7s} {'night':>7s} {'fog':>7s} {'backlt':>7s} "
        f"{'tiny':>7s} {'bN fps':>8s} {'W':>6s}")
    for r in records:
        j, s = r["jetson"], r["slices_detail"]
        log(f"{r['model_id']:30s} {r['size_disk_mb']:6.1f} {s['all']['map50']:7.4f} "
            f"{s.get('night', {}).get('map50', 0):7.4f} {s.get('fog', {}).get('map50', 0):7.4f} "
            f"{s.get('backlight', {}).get('map50', 0):7.4f} "
            f"{(r['map50_tiny_plume'] or 0):7.4f} {j['fps_batched']:8.1f} "
            f"{(j.get('power_w_mean') or 0):6.2f}")


if __name__ == "__main__":
    main()
