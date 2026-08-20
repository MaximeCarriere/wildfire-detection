#!/usr/bin/env python3
"""XP6 E4b — does 2:4 sparsity actually run faster on this board?

E4 answered accuracy: 2:4 masking costs nothing after recovery. It could not
answer speed, because a TensorRT engine is compiled per GPU architecture and the
accuracy screening ran on a desktop card. This script answers speed, on the
Jetson, and it is built to separate three things that are easy to conflate:

1. **The pattern's cost.** 2:4 weights built as an ordinary dense engine. Same
   arithmetic as unpruned, so this should match it. If it does not, the
   difference is noise and bounds every other claim here.
2. **Whether the compiler uses the hardware.** The same 2:4 weights built with
   ``--sparsity=enable``. Orin's GPU is Ampere class and has the circuitry; that
   does not mean TensorRT selects sparse kernels for *this* network. The verbose
   build log is kept so the decision is read, not assumed.
3. **What free-form sparsity buys.** Unstructured 50% built normally. Expected:
   exactly nothing, because scattered zeros have no matching kernels. Measuring
   it is what turns "expected" into "measured".

Every arm is the *same* network with the *same* shape. Nothing here is smaller.
The only variable is which weights are zero and what the compiler was allowed to
do about it.

Speed is reported as mean +/- standard deviation over independent timing runs.
``lib.evaluator`` reports the standard error, which is the more useful number for
"is this difference real"; the standard deviation is se * sqrt(runs) and both are
recorded, so neither has to be recomputed from the other.

Usage
    python e4_speed.py                 # build all four engines, then measure
    python e4_speed.py --skip-build    # measure engines already on disk
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
from lib.prune_utils import detect_head_convs, load_yolov5   # noqa: E402
from lib.trt_export import build_fp16_engine, export_onnx_from_model  # noqa: E402

import _masking                                     # noqa: E402

RAW = REPO / "results" / "raw"
WEIGHTS = REPO / "weights"
BASE = "yolov5s"
RES = 512
RUNS = 7          # more than the protocol's 3: a std over 3 samples is not a std


def log(m: str) -> None:
    print(f"[e4speed] {m}", flush=True)


def make_variant(kind: str):
    """Load yolov5s and impose the requested zero pattern. Returns (model, meta)."""
    model = load_yolov5(WEIGHTS / f"{BASE}.pt", YOLOV5_REPO)
    exclude = {id(c) for c in detect_head_convs(model)}
    if kind == "dense":
        meta = {"granularity": "dense", "achieved_sparsity": 0.0}
    elif kind == "free50":
        meta = _masking.apply_unstructured(model, 0.5, exclude)
    elif kind == "sparse24":
        meta = _masking.apply_2to4(model, exclude)
    else:
        raise ValueError(kind)
    return model, meta


def measure(engine: Path, tag: str, note: str, meta: dict, sparsity_flag: bool) -> dict:
    from lib.detectors import TRTDetector

    test = dataset.load_samples("test")
    det = TRTDetector(engine, input_res=RES, fmt="trt_fp16", params_m=7.025)

    j = evaluator.measure_latency(det, [s.image for s in test[:64]], runs=RUNS)
    j |= evaluator.measure_throughput(det, [s.image for s in test[:32]], runs=RUNS)
    j |= evaluator.measure_memory()

    # std = se * sqrt(runs). Recorded next to the se so the README can quote the
    # spread without a reader having to redo the algebra or guess the run count.
    j["fps_batched_std"] = round(j["fps_batched_se"] * RUNS ** 0.5, 3)
    j["fps_batch1_std"] = round(j["fps_se"] * RUNS ** 0.5, 3)
    j["latency_ms_median_std"] = round(j["latency_ms_median_se"] * RUNS ** 0.5, 4)

    rec = evaluator.results_record(
        model_id=tag, fmt="trt_fp16", params_m=7.025,
        size_disk_mb=engine.stat().st_size / 1e6, input_res=RES,
        accuracy=None, jetson=j, notes=note)
    rec["sparsity_meta"] = meta | {"trt_sparsity_flag": sparsity_flag}
    evaluator.write_results(rec, RAW / f"xp06e4b_{tag}.json")

    log(f"  {tag}: {j['fps_batched']:.1f} +/- {j['fps_batched_std']:.1f} img/s batched, "
        f"{j['latency_ms_median']:.2f} +/- {j['latency_ms_median_std']:.2f} ms batch-1, "
        f"{j.get('power_w_mean')} W")
    return rec


def sparse_kernel_report(log_path: Path) -> dict:
    """Read the build log rather than trusting that the flag did something."""
    if not log_path.exists():
        return {"log": None}
    lines = log_path.read_text().splitlines()
    hits = [l.strip() for l in lines if "spars" in l.lower()]
    return {"log": log_path.name, "n_sparsity_lines": len(hits),
            "sparsity_lines": hits[:25]}


ARMS = [
    # (kind, suffix, sparsity flag, note)
    ("dense", "dense", False,
     "XP6 E4b: unpruned dense FP16 control at 512px. The line every other arm is "
     "measured against."),
    ("free50", "free50", False,
     "XP6 E4b: unstructured 50% sparsity, weights MASKED not removed, ordinary dense "
     "engine. Scattered zeros have no matching kernel on this hardware, so this "
     "measures whether free-form sparsity buys any speed at all. It is the same "
     "arithmetic as dense."),
    ("sparse24", "sparse24_nosparse", False,
     "XP6 E4b: 2:4 weights built WITHOUT --sparsity=enable. The control that isolates "
     "kernel selection from the pattern itself: same zeros as the arm below, compiler "
     "not permitted to exploit them."),
    ("sparse24", "sparse24_sparse", True,
     "XP6 E4b: 2:4 weights built WITH --sparsity=enable on Ampere-class Orin. A "
     "successful build is not proof that sparse kernels were selected; the verbose "
     "build log records the decision and is summarised in sparsity_meta."),
]


def run_variance(args, ob: str) -> None:
    """Rebuild the dense arm from the identical onnx and measure each build.

    Every arm in ARMS differs in its weights, so a speed gap between them is always
    open to "maybe the zeros did something". These builds remove that: same file,
    same flags, same board, different invocation of the autotuner. Whatever spread
    appears here is the floor below which no E4 comparison means anything.
    """
    onnx = WEIGHTS / f"{BASE}_e4b_dense.onnx"
    meta = json.loads((WEIGHTS / f"{BASE}_e4b_dense.meta.json").read_text())
    for rep in ("a", "b", "c"):
        # 'a' is the published dense engine; b and c are its resamples.
        suffix = "dense" if rep == "a" else f"dense{rep}"
        engine = WEIGHTS / f"{BASE}_e4b_{suffix}{ob}_fp16_{RES}.engine"
        blog = WEIGHTS / f"{BASE}_e4b_{suffix}{ob}_build.log"
        if not engine.exists():
            log(f"building {engine.name} (opt_batch={args.opt_batch})")
            build_fp16_engine(onnx, engine, res=RES, sparsity=False, log_path=blog,
                              opt_batch=args.opt_batch)
        measure(engine, f"{BASE}_var{rep}{ob}",
                f"XP6 E4b variance control, build {rep} of 3. Identical onnx, identical "
                f"flags, autotuned independently. Measures TensorRT's build-to-build "
                f"spread so the gaps between the four E4b arms can be read against a "
                f"floor instead of assumed real. optShapes batch {args.opt_batch}.",
                meta | {"variance_build": rep, "opt_batch": args.opt_batch}, False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--opt-batch", type=int, default=1,
                    help="batch size TensorRT autotunes for (--optShapes). The first "
                         "run of this experiment tuned at 1 and reported throughput at "
                         "16, so the kernels were chosen under conditions never "
                         "measured; sparse tactics are exactly the case that gap hides, "
                         "because their metadata cost is amortised by batch. Results are "
                         "tagged _optb<N> so they never overwrite the batch-1 set.")
    ap.add_argument("--variance", action="store_true",
                    help="measure the extra dense engines built from the SAME onnx with "
                         "the SAME flags. Nothing about them differs except the build, so "
                         "their spread IS TensorRT's build-to-build noise -- the control "
                         "that decides whether E4's unexplained 3.5%% needs explaining.")
    ap.add_argument("--reverse", action="store_true",
                    help="measure the arms in reverse order, tagging results _rev. "
                         "The four arms are timed back to back, so anything that drifts "
                         "with elapsed time (clocks settling, die warming) is aliased "
                         "onto arm identity. Reversing the order separates the two.")
    args = ap.parse_args()

    # Batch-1 keeps the original unsuffixed names so the published engines and
    # result files are reproduced exactly; any other tuning batch lands beside them.
    ob = "" if args.opt_batch == 1 else f"_optb{args.opt_batch}"

    if args.variance:
        run_variance(args, ob)
        return

    built = {}
    for kind, suffix, flag, _note in ARMS:
        onnx = WEIGHTS / f"{BASE}_e4b_{kind}.onnx"
        engine = WEIGHTS / f"{BASE}_e4b_{suffix}{ob}_fp16_{RES}.engine"
        blog = WEIGHTS / f"{BASE}_e4b_{suffix}{ob}_build.log"
        if args.skip_build and engine.exists():
            built[suffix] = (engine, blog, {})
            continue
        if not onnx.exists():
            log(f"building {kind} -> {onnx.name}")
            model, meta = make_variant(kind)
            export_onnx_from_model(model, onnx, res=RES, repo=YOLOV5_REPO)
            (WEIGHTS / f"{BASE}_e4b_{kind}.meta.json").write_text(json.dumps(meta, indent=2))
            del model
        meta = json.loads((WEIGHTS / f"{BASE}_e4b_{kind}.meta.json").read_text())
        log(f"building engine {engine.name} (sparsity={flag}, opt_batch={args.opt_batch})")
        build_fp16_engine(onnx, engine, res=RES, sparsity=flag, log_path=blog,
                          opt_batch=args.opt_batch)
        built[suffix] = (engine, blog, meta)

    log("")
    order = list(reversed(ARMS)) if args.reverse else ARMS
    for kind, suffix, flag, note in order:
        engine, blog, meta = built[suffix]
        if not meta:
            meta = json.loads((WEIGHTS / f"{BASE}_e4b_{kind}.meta.json").read_text())
        meta = dict(meta) | sparse_kernel_report(blog)
        tag = f"{BASE}_{suffix}{ob}" + ("_rev" if args.reverse else "")
        measure(engine, tag, note, meta, flag)


if __name__ == "__main__":
    main()
