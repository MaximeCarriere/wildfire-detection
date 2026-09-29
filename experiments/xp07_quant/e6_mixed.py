#!/usr/bin/env python3
"""XP7 E6 — mixed precision: act on E1's map.

Axis: **target**. The analog of XP6-E6, and the EdgeFirst "smart quantization"
idea tested honestly: leave the quantization-hostile part of the graph in float
and quantize the 99% of FLOPs that are robust.

Three arms at the same nominal W8A8 setting:

* **uniform** — all 60 convolutions INT8, and the Detect decode output quantized
  too. This is what asking TensorRT for ``--int8`` with no constraints produces,
  and it is the control.
* **head_out** — the 57 backbone/neck convolutions INT8, the 3 Detect head
  convolutions left in FP16. **The split is taken from E1's measured map, not
  chosen by hand** — the layers excluded are the ones E1 ranked worst, and
  ``--from-e1`` reads them out of E1's JSON so the choice is auditable.
* **head_out_decode_out** — additionally the box decode and sigmoid run in float
  outside the quantized region. On the board this is graph surgery: the engine
  ends at the three head convolutions and emits ten raw output buffers instead of
  two, with the decode done on the CPU afterwards.

**Why the third arm needs its own machinery.** The conv-level simulation never
quantizes the decode, so arms 2 and 3 would be numerically identical for a reason
that is an artefact of the simulation rather than a fact about hardware.
``DetectOutputQuantizer`` therefore reproduces the one effect that distinguishes
them: a single per-tensor scale over a tensor that concatenates box coordinates in
pixels (0-512) with probabilities in [0, 1]. Arms 1 and 2 have it on, arm 3 has it
off, and the gap between arm 2 and arm 3 is exactly the cost of quantizing the
decode.

EdgeFirst's claim is that arm 3 recovers ~5 mAP points for sub-20 ms of CPU work.
Here the head is small but the board is launch-bound (XP9), so ten output buffers
and a CPU decode is a real cost, not a footnote — which is why arm 3 also gets
board numbers in E4 rather than only an accuracy number here.

Usage
    python e6_mixed.py                                # all three arms
    python e6_mixed.py --from-e1 --n-worst 3          # split chosen by E1's map
    python e6_mixed.py --from-e1 --rank-by tiny_plume # protect distant smoke instead
    python e6_mixed.py --val-images 300
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _arms                                                       # noqa: E402
import _calib                                                      # noqa: E402
from _calib import log                                             # noqa: E402
from _quant import (QuantConfig, conv_layers, head_conv_names)     # noqa: E402

CFG = QuantConfig(weight_bits=8, act_bits=8, granularity="per_channel", method="minmax")
E1_JSON = _calib.RAW / "xp07e1_sensitivity.json"


def float_layers_from_e1(n_worst: int, arm: str = "w8a8",
                         rank_by: str = "map50") -> tuple[list[str], dict]:
    """The N layers E1 ranked worst — the split, chosen by measurement.

    ``rank_by`` decides *worst at what*, and on this model the two answers differ.
    Ranked on aggregate mAP50 the worst three are ``model.24.m.2``,
    ``model.24.m.0`` and ``model.17.cv3.conv``; ranked on tiny plumes they are
    ``model.24.m.0``, ``model.20.cv3.conv`` and ``model.17.m.0.cv2.conv``. Only
    one layer appears in both. Since E9 established that distant smoke is the
    capability actually at stake, which ranking to protect is a real decision and
    not a detail — so it is a flag, and the choice is recorded in the JSON.

    If E1 has not been run this raises rather than silently falling back to the
    hand-picked head, because "the split came from the map" has to be true.
    """
    if not E1_JSON.exists():
        raise SystemExit(f"{E1_JSON} missing — run e1_sensitivity.py first, or drop --from-e1")
    doc = json.loads(E1_JSON.read_text())
    rows = [r for r in doc["rows"] if r["arm"] == arm and r.get(rank_by) is not None]
    if not rows:
        raise SystemExit(f"E1 has no {arm} rows carrying {rank_by!r}")
    worst = sorted(rows, key=lambda r: r[rank_by])[:n_worst]
    return ([r["layer"] for r in worst],
            {"source": E1_JSON.name, "arm": arm, "n_worst": n_worst,
             "ranked_by": rank_by,
             "chosen": [{"layer": r["layer"], "map50": r["map50"],
                         "tiny_plume": r["tiny_plume"],
                         "rel_drop_pct": r["rel_drop_pct"], "is_head": r["is_head"]}
                        for r in worst]})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-images", type=int, default=0)
    ap.add_argument("--calib-images", type=int, default=512)
    ap.add_argument("--from-e1", action="store_true",
                    help="take the float-layer set from E1's measured map")
    ap.add_argument("--n-worst", type=int, default=3)
    ap.add_argument("--rank-by", default="map50", choices=["map50", "tiny_plume"],
                    help="worst at what — the two rankings disagree on this model")
    ap.add_argument("--recover-epochs", type=int, default=0)
    ap.add_argument("--out", default="xp07e6_mixed.json")
    args = ap.parse_args()

    samples = _calib.split_samples("val", limit=args.val_images)
    baseline = _calib.score(_calib.load_base(), samples, "fp16_baseline")
    log("e6", f"FP16 baseline map50={baseline['map50']:.4f} on {len(samples)} val images")

    model = _calib.load_base()
    all_names = [n for n, _ in conv_layers(model)]
    head = head_conv_names(model)
    del model

    if args.from_e1:
        float_set, evidence = float_layers_from_e1(args.n_worst, rank_by=args.rank_by)
        log("e6", f"E1 (ranked by {args.rank_by}) says leave in float: {float_set}")
    else:
        float_set, evidence = head, {"source": "the Detect head, by hand (no --from-e1)"}

    quantized = [n for n in all_names if n not in set(float_set)]

    rows = [
        _arms.run_arm("uniform", CFG, samples, n_calib=args.calib_images,
                      targets="all", quantize_decode=True,
                      recover_epochs=args.recover_epochs),
        _arms.run_arm("head_out", CFG, samples, n_calib=args.calib_images,
                      targets=quantized, quantize_decode=True,
                      recover_epochs=args.recover_epochs),
        _arms.run_arm("head_out_decode_out", CFG, samples, n_calib=args.calib_images,
                      targets=quantized, quantize_decode=False,
                      recover_epochs=args.recover_epochs),
    ]

    got = {r["arm"]: r["damage"]["map50"] for r in rows}
    gaps = {
        "keeping_head_float_gains": round(got["head_out"] - got["uniform"], 4),
        "keeping_decode_float_gains": round(got["head_out_decode_out"] - got["head_out"], 4),
        "total_vs_uniform": round(got["head_out_decode_out"] - got["uniform"], 4),
        "remaining_gap_to_fp16": round(baseline["map50"] - got["head_out_decode_out"], 4),
    }
    for k, v in gaps.items():
        log("e6", f"{k:30s} {v:+.4f} mAP50")

    _calib.write_json(args.out, _arms.payload(
        "xp07_e6_mixed", "mixed precision: E1's map applied — keep the head in FP16",
        "target", rows, baseline, samples,
        extra={"subsampled": bool(args.val_images),
               "float_layers": float_set, "split_evidence": evidence,
               "n_quantized": len(quantized), "gaps": gaps,
               "recovery_epochs": args.recover_epochs or None}))


if __name__ == "__main__":
    main()
