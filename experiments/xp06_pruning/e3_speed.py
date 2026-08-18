#!/usr/bin/env python3
"""XP6 E3b — does channel-count regularity buy speed on the board?

E3 answered accuracy: rounding surviving channel counts up to a multiple of
8/16/32 barely moves mAP50 (0.7458 -> 0.7377 across the four). It could not
answer speed, because a TensorRT engine is compiled per GPU architecture and the
accuracy screening ran on a desktop card. This script answers speed, on the
Jetson, and is the direct test of XP6's most surprising observation: that a
*larger* pruned model ran *faster* than a smaller one, hypothesised to be
channel-count regularity defeating the compiler's kernel tiling.

`round_to` forces surviving widths onto a multiple, so the same nominal 25% cut
lands on aligned shapes. Four engines, one per rounding, each a plain dense FP16
build. The only variable is the shape of the surviving channels.

**A caveat this script cannot remove, only record.** Rounding also shrinks the
model: round_to=1 keeps 4.21 M parameters, round_to=32 keeps 3.52 M. So the arms
are not size-matched, and if 32 measures faster, regularity and size both
contribute. The per-arm parameter count and the count of layers landing on a
multiple of 32 are recorded so the two effects can at least be read apart. A
clean isolation would need a fifth arm tuned to match 1's size at 32's
alignment; that is left as follow-up.

**And a prior, from E4b on this same board.** At 512px on an Orin Nano the
runtime is kernel-launch and memory bound, not compute bound: TensorRT declined
every sparse kernel because the layers are too small for multiplication to
dominate. Regularity helps a compute kernel choose a better tile, so if the
workload is not compute bound the effect may be small or absent. Measuring it is
what turns that reasoning into a result.

Speed is mean +/- standard deviation over RUNS independent timing runs.

Usage
    python e3_speed.py                 # build four engines on the board, then measure
    python e3_speed.py --skip-build    # measure engines already on disk
    python e3_speed.py --reverse       # measure in reverse order (drift control)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from lib import data as dataset                        # noqa: E402
from lib import evaluator                              # noqa: E402
from lib.detectors import YOLOV5_REPO                  # noqa: E402
from lib.trt_export import build_fp16_engine           # noqa: E402

RAW = REPO / "results" / "raw"
WEIGHTS = REPO / "weights"
BASE = "yolov5s"
RES = 512
RUNS = 7          # a std over the protocol's 3 samples is not a std
ROUND_TO = [1, 8, 16, 32]


def log(m: str) -> None:
    print(f"[e3speed] {m}", flush=True)


def arm_meta(r: int) -> dict:
    """Parameters and alignment for this arm, from the committed accuracy record,
    so the speed number can be read against how regular the model actually is."""
    import json
    p = RAW / f"xp06e3_dfire_{BASE}_round{r}.json"
    if not p.exists():
        return {"round_to": r}
    d = json.loads(p.read_text())
    m = d["prune_meta"]
    return {"round_to": r, "params_m": d["params_m"],
            "macs_reduction": m.get("macs_reduction"),
            "widths_divisible_by_8": m.get("widths_divisible_by_8"),
            "widths_divisible_by_16": m.get("widths_divisible_by_16"),
            "widths_divisible_by_32": m.get("widths_divisible_by_32"),
            "n_conv_layers": m.get("n_conv_layers"),
            "screening_map50": d.get("map50_dfire_test")}


def measure(engine: Path, tag: str, meta: dict) -> dict:
    from lib.detectors import TRTDetector

    test = dataset.load_samples("test")
    det = TRTDetector(engine, input_res=RES, fmt="trt_fp16",
                      params_m=meta.get("params_m", 0.0))

    j = evaluator.measure_latency(det, [s.image for s in test[:64]], runs=RUNS)
    j |= evaluator.measure_throughput(det, [s.image for s in test[:32]], runs=RUNS)
    j |= evaluator.measure_memory()
    j["fps_batched_std"] = round(j["fps_batched_se"] * RUNS ** 0.5, 3)
    j["latency_ms_median_std"] = round(j["latency_ms_median_se"] * RUNS ** 0.5, 4)

    rec = evaluator.results_record(
        model_id=tag, fmt="trt_fp16", params_m=meta.get("params_m", 0.0),
        size_disk_mb=engine.stat().st_size / 1e6, input_res=RES,
        accuracy=None, jetson=j,
        notes=(f"XP6 E3b: round_to={meta['round_to']} dense FP16 engine at {RES}px, measured "
               f"on the Jetson. Regularity test: {meta.get('widths_divisible_by_32')} of "
               f"{meta.get('n_conv_layers')} conv layers land on a multiple of 32 here. NOT "
               f"size-matched to the other arms ({meta.get('params_m')} M params), so a speed "
               f"difference mixes regularity with size; both are recorded."))
    rec["regularity_meta"] = meta
    evaluator.write_results(rec, RAW / f"xp06e3b_{tag}.json")
    log(f"  {tag}: {j['fps_batched']:.1f} +/- {j['fps_batched_std']:.1f} img/s batched, "
        f"{j['latency_ms_median']:.2f} +/- {j['latency_ms_median_std']:.2f} ms batch-1, "
        f"{j.get('power_w_mean')} W · {meta.get('widths_divisible_by_32')}/"
        f"{meta.get('n_conv_layers')} aligned")
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--reverse", action="store_true",
                    help="measure the arms back to front, tagged _rev, so anything that "
                         "drifts with elapsed time is separated from arm identity.")
    args = ap.parse_args()

    built = {}
    for r in ROUND_TO:
        onnx = WEIGHTS / f"dfire_{BASE}_round{r}.onnx"
        engine = WEIGHTS / f"dfire_{BASE}_round{r}_fp16_{RES}.engine"
        if args.skip_build and engine.exists():
            built[r] = engine
            continue
        if not onnx.exists():
            raise SystemExit(
                f"{onnx} missing. Copy the four round_to ONNX files from the 3090 first:\n"
                f"  scp weights/dfire_yolov5s_round*.onnx board:{WEIGHTS}/")
        log(f"building {engine.name}")
        build_fp16_engine(onnx, engine, res=RES)
        built[r] = engine

    log("")
    order = list(reversed(ROUND_TO)) if args.reverse else ROUND_TO
    recs = []
    for r in order:
        tag = f"{BASE}_round{r}" + ("_rev" if args.reverse else "")
        recs.append(measure(built[r], tag, arm_meta(r)))

    log("")
    base = next((x for x in recs if x["regularity_meta"]["round_to"] == 1), recs[0])
    b = base["jetson"]["fps_batched"]
    log(f"{'round_to':9s} {'params':>8s} {'w/32':>6s} {'img/s':>18s} {'vs round_to=1':>13s}")
    for x in sorted(recs, key=lambda x: x["regularity_meta"]["round_to"]):
        m, j = x["regularity_meta"], x["jetson"]
        log(f"{m['round_to']:<9d} {m.get('params_m', 0):7.2f}M "
            f"{m.get('widths_divisible_by_32', 0):>3d}/60 "
            f"{j['fps_batched']:8.1f} +/- {j['fps_batched_std']:5.1f} "
            f"{j['fps_batched'] / b:12.2f}x")
    log("Regularity confirmed only if fps rises with round_to beyond what the shrinking "
        "parameter count alone would give.")


if __name__ == "__main__":
    main()
