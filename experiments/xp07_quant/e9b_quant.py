#!/usr/bin/env python3
"""XP7 E9b — every quantization decision in this study, against every other.

E9 asks which *family* of compression wins and answers it across the whole
series. This asks the narrower and more immediately useful question: **within
quantization, which decisions actually matter, and what does each one cost?**

Every arm XP7 measured is collected here and tagged with the axis it moves —
range (calibration), granularity, target, bit-width, effort — so the table reads
as a ranking of decisions rather than a list of runs.

**The two halves cannot be read against each other, and the script refuses to
pretend otherwise.** Simulated arms are fake-quant scored on the 1,721-image
**validation** split; engines are TensorRT scored on the 4,306-image **test**
split. The same model scores 0.9494 on one and 0.7776 on the other, so a chart
mixing them would invent a 17-point difference out of the split alone. They are
emitted as separate blocks, each normalised to *its own* unquantized baseline, so
"% of the unquantized model kept" is comparable everywhere even though the raw
mAP50 is not.

Usage
    python e9b_quant.py
    python e9b_quant.py --print-table
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _calib                                                      # noqa: E402
from _calib import log                                             # noqa: E402

RAW = _calib.RAW


def side(name: str):
    p = RAW / name
    return json.loads(p.read_text()) if p.exists() else None


def _row(label, axis, acc, size_mb, base, *, note=None, fps=None, j=None,
         int8_convs=None):
    """One arm, normalised against the baseline of its own split."""
    bmap, btiny = base
    return {
        "label": label, "axis": axis,
        "map50": acc.get("map50"),
        "tiny_plume": acc.get("tiny_plume"),
        # `is None`, not falsy: an arm that scored exactly 0.0 is the most
        # informative measurement in this study (E6's decode-quantized arms), and
        # treating it as missing drops it out of every spread and ranking below.
        "kept_pct": None if (acc.get("map50") is None or not bmap)
                    else round(acc["map50"] / bmap * 100, 1),
        "tiny_kept_pct": None if (acc.get("tiny_plume") is None or not btiny)
                         else round(acc["tiny_plume"] / btiny * 100, 1),
        "size_mb": size_mb, "fps_batched": fps, "j_per_1k": j,
        "convs_in_int8": int8_convs, "note": note,
    }


def simulated_rows() -> tuple[list[dict], dict]:
    """Every fake-quant arm, val split, each normalised to the val baseline."""
    rows = []
    base = (None, None)

    e2 = side("xp07e2_calibration.json")
    if e2:
        bl = e2["baseline_fp16"]
        base = (bl["map50"], bl["tiny_plume"]["map50"])
        rows.append(_row("unquantized FP16", "—",
                         {"map50": bl["map50"], "tiny_plume": bl["tiny_plume"]["map50"]},
                         e2.get("fp16_baseline_mb"), base, note="the reference"))
        for name, r in sorted(e2["methods"].items(), key=lambda kv: -kv[1]["map50"]):
            rows.append(_row(f"calibration: {name}", "range", r,
                             (r.get("size") or {}).get("total_mb"), base,
                             note=f"input range 0–{r['input_represented_max']:.4f}"))

    for fname, axis, prefix in (("xp07e3_granularity.json", "granularity", "granularity"),
                                ("xp07e5_targets.json", "target", "target"),
                                ("xp07e6_mixed.json", "target", "mixed (by mAP50)"),
                                ("xp07e6_mixed_tiny.json", "target", "mixed (by tiny)")):
        d = side(fname)
        if not d:
            continue
        for r in d["rows"]:
            dam = r["damage"]
            note = None
            if "mixed" in prefix:
                note = f"{r['n_layers_quantized']} convs INT8"
                if r.get("quantize_decode_output"):
                    note += ", decode INT8"
            rows.append(_row(f"{prefix}: {r['arm']}", axis,
                             {"map50": dam["map50"], "tiny_plume": dam["tiny_plume"]},
                             (r.get("size") or {}).get("total_mb"), base, note=note))

    e8 = side("xp07e8_lowbit.json")
    if e8:
        for name, r in e8["arms"].items():
            rows.append(_row(f"bit-width: {name}", "bit-width", r,
                             (r.get("size") or {}).get("total_mb"), base,
                             note="weights only, activations FP16"))
        cb = e8.get("codebook")
        if cb:
            rows.append(_row("bit-width: k-means codebook 4-bit + Huffman", "bit-width",
                             cb, cb.get("huffman_mb"), base,
                             note=f"{cb['mean_huffman_bits_per_weight']} bits/weight, "
                                  f"no integer kernel"))

    e10 = side("xp07e10_compose.json")
    if e10:
        for arm, r in (e10.get("arms") or {}).items():
            if r.get("pruned_fp16"):
                rows.append(_row("composition: XP6 pruned model, FP16", "composition",
                                 r["pruned_fp16"], None, base,
                                 note="pruning's own damage, before quantizing"))
            rows.append(_row(f"composition: {arm}", "composition", r["damage"], None,
                             base, note="XP6's pruned model + XP7's INT8"))
    return rows, base


def engine_rows() -> tuple[list[dict], dict]:
    """Every built engine, test split, normalised to the FP16 engine."""
    rows = []
    e4 = side("xp07e4_engines.json")
    base = (None, None)
    if not e4:
        return rows, base
    fp16 = (e4.get("arms") or {}).get("fp16") or {}
    base = (fp16.get("map50"), fp16.get("tiny_plume"))

    def add(doc, src):
        for arm, r in (doc.get("arms") or {}).items():
            rep = r.get("precision_report") or {}
            pw = r.get("power") or {}
            n8, tot = rep.get("convolutions_in_int8"), rep.get("convolutions_total")
            rows.append(_row(
                f"engine: {arm}", "engine", r, r.get("engine_mb"), base,
                fps=r.get("fps_batched"),
                j=pw.get("j_per_1k") or r.get("energy_j_per_1000_frames"),
                int8_convs=None if n8 is None else f"{n8}/{tot}",
                note=src))

    add(e4, "path B — TensorRT's own calibrator, min-max")
    qdq = side("xp07e4_qdq.json")
    if qdq:
        add(qdq, "path A — our QDQ graph, percentile 99.99, 8 calib images")

    # XP10's engines, from the frozen results schema
    for f, label in (("xp10_yolov5s_int8mm_512.json", "engine: XP10 INT8 min-max"),
                     ("xp10_yolov5s_int8_512.json", "engine: XP10 INT8 entropy")):
        d = side(f)
        if not d:
            continue
        j = d.get("jetson") or {}
        rows.append(_row(label, "engine",
                         {"map50": d.get("map50_dfire_test"),
                          "tiny_plume": d.get("map50_tiny_plume")},
                         d.get("size_disk_mb"), base,
                         fps=j.get("fps_batched"),
                         j=j.get("energy_j_per_1000_frames"),
                         note="XP10, batch-1 energy protocol"))
    return rows, base


def print_block(title, rows, base, show_speed=False):
    print(f"\n### {title}\n")
    print(f"Baseline: mAP50 {base[0]}, tiny plumes {base[1]}.\n")
    cols = "| arm | axis | size | mAP50 | kept | tiny | kept |"
    sep = "|---|---|---:|---:|---:|---:|---:|"
    if show_speed:
        cols += " img/s | J/1k | convs INT8 |"
        sep += "---:|---:|---:|"
    print(cols); print(sep)
    for r in rows:
        f = lambda v, n=4: "—" if v is None else f"{v:.{n}f}"
        pc = lambda v: "—" if v is None else f"{v:.1f}%"   # 0.0% is a result
        line = (f"| {r['label']} | {r['axis']} | {f(r['size_mb'], 2)} MB | "
                f"{f(r['map50'])} | {pc(r['kept_pct'])} | {f(r['tiny_plume'])} | "
                f"{pc(r['tiny_kept_pct'])} |")
        if show_speed:
            line += (f" {f(r['fps_batched'], 1)} | {f(r['j_per_1k'], 1)} | "
                     f"{r['convs_in_int8'] or '—'} |")
        print(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--print-table", action="store_true")
    ap.add_argument("--out", default="xp07e9b_quant.json")
    args = ap.parse_args()

    sim, sim_base = simulated_rows()
    eng, eng_base = engine_rows()
    log("e9b", f"{len(sim)} simulated arms (val), {len(eng)} engines (test)")

    def spread(rows, key="kept_pct"):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return None if not vals else round(max(vals) - min(vals), 1)

    by_axis = {}
    for r in sim:
        if r["axis"] in ("—", "composition"):
            continue
        by_axis.setdefault(r["axis"], []).append(r)
    ranking = sorted(
        ({"axis": a, "arms": len(v),
          "map50_spread_pct": spread(v), "tiny_spread_pct": spread(v, "tiny_kept_pct")}
         for a, v in by_axis.items()),
        key=lambda x: -(x["map50_spread_pct"] or 0))

    log("e9b", "how much each axis is worth (spread across its arms, % of baseline):")
    for r in ranking:
        log("e9b", f"   {r['axis']:12s} {r['map50_spread_pct']:6.1f}% mAP50  "
                   f"{r['tiny_spread_pct'] or 0:6.1f}% tiny   ({r['arms']} arms)")

    if args.print_table:
        print_block("Simulated arms — validation split, fake-quant", sim, sim_base)
        print_block("Built engines — test split, TensorRT", eng, eng_base, show_speed=True)

    _calib.write_json(args.out, {
        "experiment": "xp07_e9b_quant",
        "question": "within quantization, which decisions matter and what does each cost?",
        "axis": "none — the summary of XP7 itself",
        "split_warning": "simulated arms are val (baseline 0.9494); engines are test "
                         "(baseline 0.7776). Raw mAP50 is not comparable across the two "
                         "blocks; 'kept %' is, because each is normalised to its own "
                         "unquantized baseline.",
        "simulated": {"baseline": {"map50": sim_base[0], "tiny_plume": sim_base[1]},
                      "rows": sim},
        "engines": {"baseline": {"map50": eng_base[0], "tiny_plume": eng_base[1]},
                    "rows": eng},
        "axis_ranking": ranking,
    })


if __name__ == "__main__":
    main()
