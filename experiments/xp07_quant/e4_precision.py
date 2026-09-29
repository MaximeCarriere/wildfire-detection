#!/usr/bin/env python3
"""XP7 E4 — what actually ran in INT8. **Must run on the Jetson.**

XP6's verdict was that the compiler is the real subject of these experiments: 2:4
sparsity held its accuracy and TensorRT then declined to use the sparse kernels,
so the hardware paid nothing at all. The same trap is live for INT8, and the
plan's rule is explicit — **"INT8" is a request; what ran is data.**

This reads the built engine itself rather than the builder's console output.
``build_int8_engine`` goes through the TensorRT Python API, which writes no
trtexec log, so a log-scraping report silently returns zero for every arm — a
failure mode that looks exactly like "TensorRT refused INT8" and is in fact
"nobody was reading". The engine inspector has no such ambiguity: it reports, per
layer, the tactic and the input/output formats the builder actually chose.

Run it after ``e4_engines.py``; it patches the precision block into that
experiment's JSON in place rather than rebuilding anything.

Usage
    python e4_precision.py                       # every xp07 engine found
    python e4_precision.py --engine weights/xp07_yolov5s_int8_512.engine
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _calib                                                      # noqa: E402
from _calib import log                                             # noqa: E402

E4_JSON = _calib.RAW / "xp07e4_engines.json"

#: How TensorRT's tactic names encode the arithmetic a kernel actually runs in.
#:
#: The obvious field to read would be the layer's output ``Format/Datatype`` — but
#: these engines are built with a **dynamic batch profile**, and for those the
#: inspector reports ``"N/A due to dynamic shapes"`` for every tensor. Reading it
#: anyway yields "unknown" for all 166 layers, which then counts as zero INT8 and
#: is indistinguishable from TensorRT having refused INT8. That is the same silent
#: zero this script exists to prevent, arriving by a different route.
#:
#: The tactic name is not subject to that: it names the kernel, and the kernel
#: names its datatypes. ``sm80_xmma_fprop_implicit_gemm_interleaved_i8i8_i8i32_f32``
#: is an INT8 convolution accumulating into INT32; ``..._f16f16_f16f16_f16`` is an
#: FP16 one. Checked in order, longest-evidence first.
_TACTIC_PRECISION = [
    ("int8", ("i8i8", "int8", "imma", "i8i32")),
    ("fp16", ("f16f16", "hmma", "h884", "h1688", "_fp16")),
    ("fp32", ("f32f32", "sgemm", "ffma", "_fp32", "float")),
]


def inspect(engine_path: Path) -> dict:
    """Per-layer precision of a built engine, straight from TensorRT."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
    if engine is None:
        raise RuntimeError(f"could not deserialize {engine_path}")

    insp = engine.create_engine_inspector()
    counts: Counter = Counter()
    conv_counts: Counter = Counter()
    layers = []
    parsed = 0
    for i in range(engine.num_layers):
        raw = insp.get_layer_information(i, trt.LayerInformationFormat.JSON)
        info = None
        if isinstance(raw, str):
            try:
                info = json.loads(raw)
            except Exception:
                info = None
        if isinstance(info, dict):
            parsed += 1
        else:
            # LAYER_NAMES_ONLY verbosity returns a bare quoted name. Counting those
            # as "no INT8 found" is precisely the silent-zero failure this script
            # exists to remove, so it is recorded and raised on below.
            info = {"Name": str(raw).strip('"')}

        name = info.get("Name", f"layer_{i}")
        tactic = str(info.get("TacticName") or "")
        low = tactic.lower()
        prec = next((p for p, keys in _TACTIC_PRECISION if any(k in low for k in keys)),
                    "unknown")
        counts[prec] += 1
        layer_type = str(info.get("LayerType") or "")
        is_conv = ("Conv" in layer_type) or ("conv" in name.lower())
        if is_conv:
            conv_counts[prec] += 1
        layers.append({"name": name[:140], "precision": prec,
                       "layer_type": layer_type, "tactic": tactic[:90],
                       "is_convolution": bool(is_conv)})

    # Two ways to end up reporting nothing, and both must be loud.
    if parsed and counts.get("unknown", 0) == parsed:
        raise RuntimeError(
            f"{Path(engine_path).name}: every one of {parsed} layers came back with an "
            f"unrecognised tactic, so no precision could be read. Reporting zero INT8 "
            f"layers here would be indistinguishable from TensorRT refusing INT8. "
            f"Check _TACTIC_PRECISION against this TensorRT version's tactic naming.")
    if parsed == 0:
        raise RuntimeError(
            f"{Path(engine_path).name} carries no per-layer metadata — it was built "
            f"with the default LAYER_NAMES_ONLY profiling verbosity, so TensorRT "
            f"cannot say what precision anything ran in. Rebuild with "
            f"detailed_profiling=True (lib.trt_export) and re-run. Reporting zero "
            f"INT8 layers here would be indistinguishable from TensorRT refusing "
            f"INT8, which is the exact confusion this report exists to prevent.")

    total_conv = sum(conv_counts.values())
    return {
        "engine": str(Path(engine_path).name),
        "num_layers": engine.num_layers,
        "counts": dict(counts),
        "convolution_counts": dict(conv_counts),
        "convolutions_total": total_conv,
        "convolutions_in_int8": conv_counts.get("int8", 0),
        "convolutions_int8_pct": (round(conv_counts.get("int8", 0) / total_conv * 100, 1)
                                  if total_conv else None),
        "layers": layers,
        "layers_with_metadata": parsed,
        "source": "tensorrt engine inspector, DETAILED profiling verbosity "
                  "(not a log scrape)",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="", help="one engine; default is every xp07 engine")
    ap.add_argument("--patch-json", default=str(E4_JSON))
    args = ap.parse_args()

    if args.engine:
        paths = [Path(args.engine)]
    else:
        paths = sorted(_calib.WEIGHTS.glob("xp07_*.engine"))
    if not paths:
        raise SystemExit("no xp07 engines found — run e4_engines.py first")

    reports = {}
    for p in paths:
        try:
            r = inspect(p)
        except Exception as exc:
            log("e4p", f"{p.name}: {type(exc).__name__}: {exc}")
            continue
        reports[p.stem] = r
        log("e4p", f"{p.name}: {r['convolutions_in_int8']}/{r['convolutions_total']} "
                   f"convolutions in INT8 ({r['convolutions_int8_pct']}%) | "
                   f"all layers: {r['counts']}")

    doc_path = Path(args.patch_json)
    if doc_path.exists():
        doc = json.loads(doc_path.read_text())
        for arm, row in (doc.get("arms") or {}).items():
            stem = Path(row.get("engine", "")).stem
            if stem in reports:
                # Replace the log-scraped block; it cannot see a Python-API build.
                row["precision_report"] = reports[stem]
        doc["precision_report_source"] = "tensorrt engine inspector"
        doc_path.write_text(json.dumps(doc, indent=2) + "\n")
        log("e4p", f"patched {doc_path.name}")

    _calib.write_json("xp07e4_precision.json", {
        "experiment": "xp07_e4_precision",
        "question": "how many of the convolutions did TensorRT actually place in INT8?",
        "why": "XP6 found the compiler declines optimisations it advertises; a request "
               "for INT8 is not evidence that INT8 ran",
        "reports": reports,
    })


if __name__ == "__main__":
    main()
