#!/usr/bin/env python3
"""XP7 E4, path A — QDQ engines. **Must run on the Jetson.**

Axis: none — this is the other half of the payoff measurement.

``e4_engines.py`` is **path B**: hand TensorRT our calibration set and let the
builder pick its own ranges. That is the vendor-default control, and it is what
XP10 measured. It has one serious limitation: the choices E2 and E3 made do not
survive it. TensorRT's calibrator uses its own algorithm and its own granularity,
so "we chose minmax, per-channel" is a statement about our experiments and not
about the engine that ships.

**Path A fixes that.** Q/DQ nodes are inserted into the ONNX graph itself, each
carrying an explicit scale and zero point that *we* computed. TensorRT then
recognises the pattern and executes the quantization we specified rather than one
it invented. Three consequences worth stating:

* **What a QDQ node is.** A ``QuantizeLinear`` followed by a ``DequantizeLinear``,
  with the scale as a graph initializer. Numerically it is a no-op in float — it
  rounds and un-rounds — but TensorRT reads the pair as an instruction: run the
  region between them in INT8 at this scale.
* **It is inspectable.** The graph can be opened in Netron and the scale on every
  tensor read off it. A calibration cache cannot be inspected without decoding it,
  which is what XP10 had to do by hand to find its bug.
* **Per-channel weight scales are expressible.** Path B's calibrator is
  per-tensor for weights; E3's per-channel result can only reach the board here.

Implemented with ``onnxruntime.quantization``, which is the standard and portable
producer of QDQ graphs. The scales it computes come from the same frozen
calibration list every other XP7 arm uses, through a ``CalibrationDataReader`` that
preprocesses images with ``lib.trt_export._letterbox_batch`` — the identical path
inference uses, because a preprocessing mismatch here calibrates the ranges of a
distribution the model never sees.

Usage
    python e4_qdq.py                                  # percentile 99.99, per-channel
    python e4_qdq.py --method minmax                  # what XP10 shipped, for comparison
    python e4_qdq.py --exclude-head                   # E6's head-out, as an engine
    python e4_qdq.py --skip-build                     # measure an existing engine
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _calib                                                      # noqa: E402
from _calib import log                                             # noqa: E402

ONNX = _calib.WEIGHTS / "yolov5s.onnx"
#: A batch-dynamic-only export, used for path A only. See ``static_hw_onnx``.
ONNX_FIXED_HW = _calib.WEIGHTS / "xp07_yolov5s_b_dynamic_512.onnx"
TRTEXEC = "/usr/src/tensorrt/bin/trtexec"


def static_hw_onnx(res: int = _calib.RES) -> Path:
    """Re-export the model with dynamic **batch only**, not dynamic resolution.

    The repo's standard ONNX marks batch, height *and* width dynamic, so one file
    serves every input size. Path A cannot use it. The QDQ graph has to be built
    without ONNX Runtime's pre-processing (its shape inference fails on this graph,
    and forcing it OOMs an 8 GB board), and TensorRT then cannot resolve the neck's
    concatenations without shape information:

        Error Code 4: Error while computing output extent of node /model.11/Concat_1

    Fixing the spatial dimensions removes the ambiguity and costs nothing that is
    actually used: **every engine in this repository is built at 512x512**, and the
    batch axis — the one the throughput measurement varies — stays dynamic.
    """
    if ONNX_FIXED_HW.exists():
        return ONNX_FIXED_HW
    import torch
    from lib.trt_export import export_onnx_from_model

    log("e4a", f"exporting a batch-dynamic ONNX at {res}x{res} (path A only)")
    model = _calib.load_base()
    for m in model.modules():
        if type(m).__name__ == "Detect":
            m.inplace, m.dynamic, m.export = False, False, True
    model = model.float().eval()
    dummy = torch.zeros(1, 3, res, res, device="cuda:0")
    with torch.no_grad():
        model(dummy)
    torch.onnx.export(
        model, dummy, str(ONNX_FIXED_HW), verbose=False, opset_version=13,
        do_constant_folding=True, input_names=["images"], output_names=["output0"],
        dynamic_axes={"images": {0: "batch"}, "output0": {0: "batch"}}, dynamo=False)
    del model
    torch.cuda.empty_cache()
    log("e4a", f"wrote {ONNX_FIXED_HW.name} "
               f"({ONNX_FIXED_HW.stat().st_size / 1e6:.1f} MB)")
    return ONNX_FIXED_HW


class LetterboxReader:
    """Feeds the frozen calibration set to onnxruntime's quantizer.

    Preprocessing is ``lib.trt_export._letterbox_batch``, the same function the
    path-B calibrator and the inference detector use. One batch at a time, because
    the quantizer holds the activations it is shown.
    """

    def __init__(self, paths, res: int, input_name: str = "images", batch: int = 1):
        self.paths = list(paths)
        self.res, self.input_name, self.batch = res, input_name, batch
        self.i = 0

    def get_next(self):
        from lib.trt_export import _letterbox_batch
        if self.i >= len(self.paths):
            return None
        chunk = self.paths[self.i:self.i + self.batch]
        self.i += self.batch
        return {self.input_name: _letterbox_batch(chunk, self.res, _calib.YOLOV5_REPO)}

    def rewind(self):
        self.i = 0


def head_output_names(onnx_path: Path, n: int = 3) -> list[str]:
    """The names of the last ``n`` convolution nodes — E6's head-out split, in
    the vocabulary the ONNX quantizer expects (node names, not module names)."""
    import onnx
    model = onnx.load(str(onnx_path))
    convs = [node.name for node in model.graph.node if node.op_type == "Conv"]
    return convs[-n:]


def build_qdq_onnx(src: Path, dst: Path, *, method: str, per_channel: bool,
                   n_calib: int, exclude: list[str], symmetric_acts: bool,
                   percentile: float = 99.99, preprocess: bool = False) -> dict:
    """Write a QDQ copy of the graph with our scales baked in.

    ``percentile`` is not decoration. E2 measured that on this detector the
    clipping rule is worth **0.62 mAP50**, and that the winner is percentile
    99.99 — a method TensorRT's own calibrator does not offer at all, since it
    exposes only entropy and min-max, *both* of which E2 found to be the wrong
    answer. Path A exists precisely so that choice can reach an engine, so the
    percentile value has to be passed through rather than left at the library
    default (99.999, which clips ten times less and lands closer to min-max).
    """
    from onnxruntime.quantization import (CalibrationMethod, QuantFormat,
                                          QuantType, quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    methods = {"minmax": CalibrationMethod.MinMax,
               "entropy": CalibrationMethod.Entropy,
               "percentile": CalibrationMethod.Percentile}
    if method not in methods:
        raise SystemExit(f"--method must be one of {sorted(methods)}")

    # Pre-processing is an optimisation, not a requirement, and on this board it
    # is not survivable. The strict form fails outright — the ONNX carries dynamic
    # batch, height and width so one engine serves every batch size, and ONNX
    # Runtime's symbolic shape inference gives up with "Incomplete symbolic shape
    # inference". The permissive form then got **OOM-killed** (exit 137): the
    # graph optimiser's working set does not fit alongside everything else on an
    # 8 GB shared-memory board. Skipping it costs the quantizer some shape
    # bookkeeping and nothing else, so it is the default here.
    prepped = Path(src)
    prep_mode = "skipped (default: the optimiser OOMs on this board)"
    if preprocess:
        prepped = dst.with_name(dst.stem + "_prepped.onnx")
        prep_mode = "full"
        try:
            quant_pre_process(str(src), str(prepped), skip_symbolic_shape=False)
        except Exception as exc:
            log("e4a", f"symbolic shape inference failed ({type(exc).__name__}); "
                       f"retrying without it")
            prep_mode = "no-symbolic-shape"
            try:
                quant_pre_process(str(src), str(prepped), skip_symbolic_shape=True)
            except Exception as exc2:
                log("e4a", f"pre-processing failed ({type(exc2).__name__}); "
                           f"quantizing the raw graph")
                prep_mode = "skipped (failed)"
                prepped = Path(src)

    reader = LetterboxReader(_calib.calib_paths(n_calib), _calib.RES)
    extra = {
        "ActivationSymmetric": symmetric_acts,
        "WeightSymmetric": True,
        # ONNX Runtime quantizes biases to INT32 by default. TensorRT's
        # DequantizeLayer accepts only INT8, FP8 and INT4, so the resulting graph
        # fails to parse outright:
        #   "IDequantizeLayer::setPrecision ... condition: isQuantized(dataType)".
        # Biases are a rounding error of the model's size (0.038 MB of 7.08 here,
        # per E2's itemisation) and are folded into the convolution anyway, so
        # leaving them in float costs nothing and is what makes the graph loadable.
        "QuantizeBias": False,
    }
    if method == "percentile":
        extra["CalibPercentile"] = percentile
        extra["CalibMovingAverage"] = False
    t0 = time.time()
    quantize_static(
        str(prepped), str(dst), reader,
        quant_format=QuantFormat.QDQ,
        # Convolutions only. Left to itself the quantizer also wraps Concat,
        # Resize and the shape-carrying tensors of the neck, and TensorRT then
        # cannot propagate shapes through them:
        #   Error Code 4: Error while computing output extent of /model.11/Concat_1
        # Restricting the op set is less a workaround than a statement of what
        # this study actually quantizes: E1 through E6 all move convolutions and
        # their input activations, and nothing else.
        op_types_to_quantize=["Conv"],
        activation_type=QuantType.QInt8 if symmetric_acts else QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
        calibrate_method=methods[method],
        nodes_to_exclude=exclude,
        extra_options=extra,
    )
    return {"qdq_onnx": str(dst.relative_to(_calib.REPO)),
            "preprocessing": prep_mode,
            "qdq_mb": round(dst.stat().st_size / 1e6, 2),
            "quantize_seconds": round(time.time() - t0, 1),
            "calibration_images": n_calib, "method": method,
            "per_channel": per_channel, "excluded_nodes": exclude,
            "percentile": percentile if method == "percentile" else None,
            "quantize_bias": False,
            "op_types_quantized": ["Conv"],
            "activations": "symmetric int8" if symmetric_acts else "asymmetric uint8"}


def build_engine(qdq: Path, engine: Path, *, res: int, max_batch: int) -> dict:
    """Let TensorRT execute the scales in the graph.

    ``--int8`` is still passed: it tells the builder INT8 kernels are permitted.
    The *ranges* now come from the graph's Q/DQ nodes rather than from a calibrator,
    which is the whole difference between path A and path B.
    """
    import subprocess
    log_path = engine.with_suffix(".build.log")
    cmd = [TRTEXEC, f"--onnx={qdq}", f"--saveEngine={engine}", "--int8", "--fp16",
           f"--minShapes=images:1x3x{res}x{res}",
           f"--optShapes=images:{max_batch}x3x{res}x{res}",
           f"--maxShapes=images:{max_batch}x3x{res}x{res}",
           # Engine metadata for the inspector. NOT --verbose: XP6 measured that
           # turning the logger up makes a 3-minute build a 20-minute one, and it
           # is not what IEngineInspector reads anyway.
           "--profilingVerbosity=detailed"]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    log_path.write_text(proc.stdout + "\n" + proc.stderr)
    if not engine.exists():
        raise RuntimeError(f"trtexec failed; see {log_path}")
    from e4_engines import layer_precision_report
    return {"engine": str(engine.relative_to(_calib.REPO)),
            "engine_mb": round(engine.stat().st_size / 1e6, 2),
            "build_seconds": round(time.time() - t0, 1),
            "precision_report": layer_precision_report(engine, log_path.read_text())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="percentile",
                    choices=["minmax", "entropy", "percentile"],
                    help="E2 measured percentile 99.99 as the winner by 0.62 mAP50 over "
                         "the worst arm, and by 57 points of tiny-plume accuracy over "
                         "min-max; it is the default here for that reason")
    ap.add_argument("--percentile", type=float, default=99.99,
                    help="only used with --method percentile; the library default is "
                         "99.999, which clips ten times less")
    ap.add_argument("--per-channel", action="store_true", default=True)
    ap.add_argument("--per-tensor", dest="per_channel", action="store_false")
    ap.add_argument("--asymmetric-acts", action="store_true")
    ap.add_argument("--exclude-head", action="store_true",
                    help="leave the last 3 convolutions out of the quantized region")
    # 32, not 512: E2 measured that calibration size is worth nothing past 32
    # images (99.8% of the unquantized mAP50 at 32 against 99.6% at 512), and the
    # ONNX Runtime calibrator holds every activation it is shown, which on an 8 GB
    # board is the difference between finishing and being OOM-killed.
    ap.add_argument("--calib-images", type=int, default=32)
    ap.add_argument("--preprocess", action="store_true",
                    help="run ONNX Runtime's graph pre-processing; it OOMs on this "
                         "board, so it is off by default")
    ap.add_argument("--test-images", type=int, default=0)
    ap.add_argument("--max-batch", type=int, default=16)
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--skip-accuracy", action="store_true")
    ap.add_argument("--out", default="xp07e4_qdq.json")
    args = ap.parse_args()

    if not ONNX.exists():
        raise SystemExit(f"{ONNX} missing — export it first")
    src_onnx = static_hw_onnx()

    tag = (f"{args.method}{args.percentile:g}".replace(".", "")
           if args.method == "percentile" else args.method)
    name = f"qdq_{tag}_{'pc' if args.per_channel else 'pt'}"
    if args.exclude_head:
        name += "_headout"
    if args.asymmetric_acts:
        name += "_asym"
    qdq_path = _calib.WEIGHTS / f"xp07_{name}.onnx"
    engine_path = _calib.WEIGHTS / f"xp07_{name}_512.engine"

    info: dict = {}
    if not args.skip_build:
        exclude = head_output_names(src_onnx) if args.exclude_head else []
        log("e4a", f"quantizing graph: {args.method}, "
                   f"{'per-channel' if args.per_channel else 'per-tensor'}, "
                   f"exclude={exclude}")
        info.update(build_qdq_onnx(src_onnx, qdq_path, method=args.method,
                                   per_channel=args.per_channel,
                                   n_calib=args.calib_images, exclude=exclude,
                                   symmetric_acts=not args.asymmetric_acts,
                                   percentile=args.percentile,
                                   preprocess=args.preprocess))
        log("e4a", "building the engine from the QDQ graph")
        info.update(build_engine(qdq_path, engine_path, res=_calib.RES,
                                 max_batch=args.max_batch))
        log("e4a", f"engine {info['engine_mb']} MB, "
                   f"int8 layers reported: {info['precision_report']['counts']['int8']}")

    samples = _calib.split_samples("test", limit=args.test_images)
    from e4_engines import measure           # shares E4's power/throughput path
    info.update(measure(name, engine_path, samples, skip_accuracy=args.skip_accuracy))
    log("e4a", f"{name}: map50={info.get('map50')} fps={info.get('fps_batched')}")

    _calib.write_json(args.out, {
        "experiment": "xp07_e4_qdq",
        "question": "the INT8 engine built from OUR scales, not TensorRT's",
        "axis": "none — the payoff measurement, path A",
        "build_path": "A — QDQ ONNX; E2's and E3's choices survive into the engine",
        "board": "Jetson Orin Nano Super",
        "split": "test", "n_test_images": len(samples),
        "subsampled": bool(args.test_images),
        "resolution": _calib.RES, "batch": args.max_batch,
        "calibration_list": str(_calib.CALIB_LIST.relative_to(_calib.REPO)),
        "fp16_line": _calib.FP16_LINE,
        "arms": {name: info},
    })


if __name__ == "__main__":
    main()
