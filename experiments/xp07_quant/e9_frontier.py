#!/usr/bin/env python3
"""XP7 E9 — the frontier: every technique in the series, against every other.

Axis: none. This is the last experiment of the compression arc and the only one
whose subject is the *other* experiments. It reads committed JSON and draws
nothing it did not find in a file.

**The question the whole series reduces to.** Five families of technique have now
been measured on this board — feed the network a smaller image (XP2), delete
channels (XP6), zero weights in the pattern the hardware understands (XP6-E4),
compute in fewer bits (XP7/XP10), and put a cheap gate in front (XP15). Each was
judged inside its own study against the same line. Nobody has yet put them on one
chart. **Which one actually buys the most, and at what?**

Four axes, because "best" means different things to different deployments and a
single ranking would hide that:

* **accuracy** — mAP50 on the frozen 4,306-image test set, with the small- and
  tiny-plume slices carried alongside, because a technique that holds aggregate
  accuracy while destroying distant smoke has not held accuracy.
* **throughput** — images/s at batch 16 on the Orin. The compute-bound number.
* **latency** — batch-1 milliseconds. What one live camera feed actually waits,
  and a different question from throughput on a launch-bound board (XP2).
* **size** — the engine on disk, in MB, plus peak memory. On an 8 GB
  shared-memory box this is a deployment constraint, not bookkeeping.

Energy (J per 1,000 frames) rides along as the marker area, since on a 15 W
fanless box it is the constraint that decides whether the thing can run at all.

**Two pieces of measurement discipline this script cannot paper over.**

1. *PyTorch throughput and TensorRT throughput are not the same measurement.*
   XP2 showed eager PyTorch on this board is kernel-launch-bound, so a `pt` row's
   img/s says more about the runtime than the model. Rows are therefore tagged
   with their runtime and the headline chart uses engine rows only.
2. *Some arms had accuracy and speed measured on different artifacts.* XP6 scored
   the round-to-32 model in PyTorch and timed its engine separately, which is how
   "0.7377 @ 641.9" came to be quoted. That pairing is legitimate — the engine is
   built from exactly that checkpoint — but it is a pairing, not a single
   measurement, so every paired row names both sources in the output and is
   flagged in the chart.

Usage
    python e9_frontier.py                    # table + figure from committed JSON
    python e9_frontier.py --engines-only
    python e9_frontier.py --print-table
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

#: The comparison set. One row per *claim the series actually makes*, not one row
#: per JSON — the ablation arms stay in their own pages. ``accuracy_from`` is
#: given separately only where the engine record carries no accuracy of its own;
#: where it is None the engine row scored itself.
#:
#: (label, family, engine_json, accuracy_json_or_None)
ROWS = [
    # --- the baseline, and what resolution alone buys ---------------------
    ("YOLOv5s FP16 @512  (the line)", "baseline",
     "xp09_yolov5s_trt_fp16_512.json", None),
    ("YOLOv5s FP16 @640", "resolution",
     "xp09_yolov5s_trt_fp16_640.json", None),
    ("YOLOv5l FP16 @640  (bigger model)", "baseline",
     "xp09_yolov5l_trt_fp16_640.json", None),

    # --- quantization (XP10's measured engines) ---------------------------
    ("INT8 min-max @512", "quantization",
     "xp10_yolov5s_int8mm_512.json", None),
    ("INT8 entropy @512  (the default)", "quantization",
     "xp10_yolov5s_int8_512.json", None),
    ("INT8 min-max @640", "quantization",
     "xp10_yolov5s_int8mm_640.json", None),

    # --- pruning ----------------------------------------------------------
    ("Pruned 25%, one-shot + 12ep", "pruning",
     "xp06_dfire_yolov5s_pruned25_recovered_trt.json", None),
    ("Pruned 25%, iterative + 12ep", "pruning",
     "xp06_dfire_yolov5s_pruned25_iter_recovered_trt.json", None),
    ("Pruned 25%, widths rounded to 32", "pruning",
     "xp06e3b_yolov5s_round32.json", "xp06e3_dfire_yolov5s_round32.json"),
    ("Pruned 25%, widths rounded to 16", "pruning",
     "xp06e3b_yolov5s_round16.json", "xp06e3_dfire_yolov5s_round16.json"),

    # --- structured sparsity ----------------------------------------------
    ("2:4 sparsity, sparse engine", "sparsity",
     "xp06e4b_yolov5s_sparse24_sparse.json", "xp06e4_dfire_yolov5s_sparse24.json"),
    ("2:4 sparsity, dense engine", "sparsity",
     "xp06e4b_yolov5s_sparse24_nosparse.json", "xp06e4_dfire_yolov5s_sparse24.json"),
]

#: XP7's own engines, appended when E4 has run. Same row shape.
XP7_SOURCES = ["xp07e4_engines.json", "xp07e4_qdq.json", "xp07e10_compose.json"]

FAMILY_ORDER = ["baseline", "resolution", "pruning", "sparsity", "quantization",
                "composition"]

#: The four things "accuracy" can mean here, and why each gets its own ranking.
#: Aggregate mAP50 is the headline, but it averages over two classes of very
#: different difficulty and over every plume size, so a technique can hold it
#: while losing the capability the product exists for. XP10 is the worked example:
#: INT8 min-max costs 8% of aggregate mAP50 and 58% of tiny-plume accuracy.
METRICS = {
    "map50":      ("aggregate mAP50", "both classes, every plume size"),
    "map50_fire": ("flame / fire class", "visible flame — the harder of the two classes"),
    "map50_smoke": ("smoke class", "the easier class, and the larger share of the data"),
    "tiny_plume": ("tiny plumes (<0.1% of frame)",
                   "distant smoke — what early detection actually is"),
}

#: Floors expressed as a share of the FP16 line's value on *that* metric, so the
#: tables are comparable across metrics whose absolute scales differ by 5x.
RETENTION_FLOORS = [0.98, 0.95, 0.90, 0.80, 0.60]


def _load(name: str) -> dict | None:
    p = RAW / name
    return json.loads(p.read_text()) if p.exists() else None


def _row_from_records(label, family, engine_json, accuracy_json) -> dict | None:
    eng = _load(engine_json)
    if not eng:
        log("e9", f"absent: {engine_json}")
        return None
    j = eng.get("jetson") or {}
    if not j.get("fps_batched"):
        return None

    acc_rec = eng
    paired = False
    if eng.get("map50_dfire_test") is None and accuracy_json:
        acc_rec = _load(accuracy_json) or {}
        paired = True

    return {
        "label": label, "family": family,
        "runtime": eng.get("format"),
        "input_res": eng.get("input_res"),
        "map50": acc_rec.get("map50_dfire_test"),
        "map50_fire": acc_rec.get("map50_fire_class"),
        "map50_smoke": acc_rec.get("map50_smoke_class"),
        "small_plume": acc_rec.get("map50_small_plume"),
        "tiny_plume": acc_rec.get("map50_tiny_plume"),
        "size_disk_mb": eng.get("size_disk_mb"),
        "params_m": eng.get("params_m"),
        "fps_batched": j.get("fps_batched"),
        "ms_per_image_batched": j.get("ms_per_image_batched"),
        "latency_ms_median": j.get("latency_ms_median"),
        "fps_batch1": j.get("fps_batch1"),
        "j_per_1k": j.get("energy_j_per_1000_frames"),
        # The two studies integrated power over different work. The older pages
        # measured during their batch-1 latency protocol, where the GPU idles
        # between kernel launches; XP7-E4 integrates over a flat-out batch-16
        # window. On the *same* FP16 engine that is 11.6 W / 52.1 J per 1k against
        # 20.5 W / 43.3 J per 1k -- nearly twice the power and 17% less energy per
        # frame. Both are correct; they are answers to different questions, and
        # comparing them without saying so would be a silent error.
        "j_per_1k_protocol": "batch-1 latency window (XP9/XP10 protocol)",
        "power_w": j.get("power_w_mean"),
        "mem_mb": j.get("mem_mb"),
        "accuracy_paired_from": accuracy_json if paired else None,
        "sources": [engine_json] + ([accuracy_json] if paired else []),
    }


def _rows_from_xp7() -> list[dict]:
    """XP7's own engines, in the same row shape, once E4 has written them."""
    out = []
    for src in XP7_SOURCES:
        doc = _load(src)
        if not doc:
            continue
        for arm, r in (doc.get("arms") or {}).items():
            if not r.get("fps_batched"):
                continue
            # XP7 rebuilt the FP16 line as its own control, so it is a
            # baseline row, not a quantization one -- filing it under
            # "quantization" would put an unquantized engine in that family's
            # colour and inflate what quantization appears to achieve.
            family = ("composition" if "e10" in src
                      else "baseline" if arm == "fp16"
                      else "quantization")
            power = r.get("power") or {}
            out.append({
                "label": ("XP7 FP16 (line, rebuilt)" if arm == "fp16"
                          else f"XP7 {arm}"), "family": family,
                "runtime": "trt_int8" if "int8" in arm or "qdq" in arm else "trt_fp16",
                "input_res": doc.get("resolution"),
                "map50": r.get("map50"),
                "map50_fire": (r.get("per_class") or {}).get("fire", {}).get("map50"),
                "map50_smoke": (r.get("per_class") or {}).get("smoke", {}).get("map50"),
                "small_plume": r.get("small_plume"),
                "tiny_plume": r.get("tiny_plume"),
                "size_disk_mb": r.get("engine_mb"), "params_m": None,
                "fps_batched": r.get("fps_batched"),
                "ms_per_image_batched": r.get("ms_per_image_batched"),
                "latency_ms_median": None, "fps_batch1": None,
                "j_per_1k": power.get("j_per_1k"),
                "j_per_1k_protocol": "flat-out batch-16 window (XP7-E4 protocol)",
                "power_w": (power.get("power_w") or {}).get("mean"), "mem_mb": None,
                "int8_layers": (r.get("precision_report") or {}).get("counts", {}).get("int8"),
                "accuracy_paired_from": None, "sources": [src],
            })
    return out


def verdict(rows: list[dict]) -> dict:
    """Who wins on each axis, and does anything beat the line on all of them."""
    line = next((r for r in rows if r["family"] == "baseline"
                 and r["input_res"] == 512), None)
    scored = [r for r in rows if r["map50"] is not None]

    def best(key, reverse=True, among=None):
        pool = [r for r in (among or scored) if r.get(key) is not None]
        if not pool:
            return None
        r = sorted(pool, key=lambda x: x[key], reverse=reverse)[0]
        return {"label": r["label"], "family": r["family"], key: r[key],
                "map50": r["map50"]}

    dominating = []
    if line:
        for r in scored:
            if r is line:
                continue
            if (r["map50"] >= line["map50"]
                    and (r["fps_batched"] or 0) >= line["fps_batched"]
                    and (r["size_disk_mb"] or 1e9) <= line["size_disk_mb"]):
                dominating.append(r["label"])

    # The practical question, asked once per metric: if you must keep this much
    # of the line's capability on THIS measure, what is the fastest arm that does?
    budgets = {}
    for floor in (0.77, 0.75, 0.73, 0.70):
        pool = [r for r in scored if r["map50"] >= floor and r["fps_batched"]]
        if pool:
            w = max(pool, key=lambda r: r["fps_batched"])
            budgets[f"map50>={floor}"] = {
                "fastest": w["label"], "family": w["family"],
                "fps": w["fps_batched"], "map50": w["map50"],
                "size_mb": w["size_disk_mb"], "j_per_1k": w["j_per_1k"]}

    per_metric = {}
    for key, (title, _) in METRICS.items():
        base = line.get(key) if line else None
        pool = [r for r in rows if r.get(key) is not None and r.get("fps_batched")]
        if not base or not pool:
            continue
        floors = {}
        for frac in RETENTION_FLOORS:
            ok = [r for r in pool if r[key] >= base * frac]
            if not ok:
                continue
            w = max(ok, key=lambda r: r["fps_batched"])
            floors[f"keep>={frac:.0%}"] = {
                "absolute": round(base * frac, 4),
                "fastest": w["label"], "family": w["family"],
                "fps": w["fps_batched"], "value": w[key],
                "kept_pct": round(w[key] / base * 100, 1),
                "size_mb": w["size_disk_mb"], "j_per_1k": w["j_per_1k"]}
        per_metric[key] = {
            "title": title, "line_value": base,
            "retention": sorted(
                ({"label": r["label"], "family": r["family"], "value": r[key],
                  "kept_pct": round(r[key] / base * 100, 1),
                  "fps": r["fps_batched"], "size_mb": r["size_disk_mb"],
                  "j_per_1k": r["j_per_1k"]} for r in pool),
                key=lambda x: -x["kept_pct"]),
            "fastest_at_each_floor": floors,
        }

    return {
        "line": None if not line else {k: line[k] for k in
                                       ("label", "map50", "fps_batched",
                                        "size_disk_mb", "j_per_1k")},
        "best_accuracy": best("map50"),
        "fastest": best("fps_batched"),
        "smallest": best("size_disk_mb", reverse=False),
        "lowest_energy": best("j_per_1k", reverse=False),
        "best_tiny_plume": best("tiny_plume"),
        "dominates_the_line_on_accuracy_speed_and_size": dominating,
        "fastest_at_each_accuracy_floor": budgets,
        "per_metric": per_metric,
    }


def print_table(rows: list[dict]) -> None:
    head = (f"{'technique':36s} {'family':13s} {'mAP50':>7s} {'fire':>7s} "
            f"{'smoke':>7s} {'small':>7s} {'tiny':>7s} "
            f"{'MB':>6s} {'img/s':>7s} {'ms b1':>6s} {'J/1k':>6s}")
    print(head); print("-" * len(head))
    for fam in FAMILY_ORDER:
        for r in [x for x in rows if x["family"] == fam]:
            f = lambda v, n=4: "—" if v is None else f"{v:.{n}f}"
            print(f"{r['label']:36s} {r['family']:13s} {f(r['map50']):>7s} "
                  f"{f(r['map50_fire']):>7s} {f(r['map50_smoke']):>7s} "
                  f"{f(r['small_plume']):>7s} "
                  f"{f(r['tiny_plume']):>7s} {f(r['size_disk_mb'],1):>6s} "
                  f"{f(r['fps_batched'],1):>7s} {f(r['latency_ms_median'],2):>6s} "
                  f"{f(r['j_per_1k'],1):>6s}"
                  + ("   [paired]" if r["accuracy_paired_from"] else ""))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--print-table", action="store_true")
    ap.add_argument("--out", default="xp07e9_frontier.json")
    args = ap.parse_args()

    rows = [r for r in (_row_from_records(*spec) for spec in ROWS) if r]
    rows += _rows_from_xp7()
    log("e9", f"{len(rows)} board-measured arms across "
              f"{len({r['family'] for r in rows})} technique families")

    v = verdict(rows)
    if v["line"]:
        log("e9", f"the line: {v['line']['map50']} @ {v['line']['fps_batched']} img/s, "
                  f"{v['line']['size_disk_mb']} MB")
    log("e9", f"dominates the line on accuracy+speed+size: "
              f"{v['dominates_the_line_on_accuracy_speed_and_size'] or 'nothing'}")
    for k, b in v["fastest_at_each_accuracy_floor"].items():
        log("e9", f"  fastest at {k}: {b['fastest']} ({b['family']}) "
                  f"{b['fps']} img/s, {b['size_mb']} MB")

    for key, blk in (v.get("per_metric") or {}).items():
        best = next(iter(blk["fastest_at_each_floor"].items()), None)
        log("e9", f"{blk['title']}: line={blk['line_value']}, "
                  f"best retention {blk['retention'][0]['label']} "
                  f"({blk['retention'][0]['kept_pct']}%)")

    if args.print_table:
        print()
        print_table(rows)

    _calib.write_json(args.out, {
        "experiment": "xp07_e9_frontier",
        "question": "every technique in the series against every other — "
                    "accuracy, size, latency, throughput, energy",
        "axis": "none — the summary of the whole compression arc",
        "board": "Jetson Orin Nano Super",
        "caveats": {
            "runtime": "PyTorch and TensorRT throughput are different measurements; "
                       "the headline chart uses engine rows only",
            "pairing": "rows with accuracy_paired_from had accuracy and speed measured "
                       "on the same checkpoint but in separate runs",
            "energy": "j_per_1k comes from two protocols -- see j_per_1k_protocol on "
                      "each row. On the same FP16 engine the batch-1 window reports "
                      "52.1 J/1k at 11.6 W and the flat-out batch-16 window reports "
                      "43.3 J/1k at 20.5 W. Energy is therefore comparable WITHIN a "
                      "protocol and not across one, which is why the figure says so "
                      "rather than averaging them into one marker scale.",
        },
        "fp16_line": _calib.FP16_LINE,
        "rows": rows,
        "verdict": v,
    })


if __name__ == "__main__":
    main()
