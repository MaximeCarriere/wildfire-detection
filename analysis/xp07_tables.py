"""Markdown tables for the XP7 quantization README, built from the committed JSON.

Same reasoning as ``xp06_tables.py``: a number retyped into a README is a number
that can drift from the measurement it claims to report, silently, and nothing
catches it. The figures are generated; the tables should be too.

Run it and paste, or diff its output against the README to check the page still
matches the evidence:

    python analysis/xp07_tables.py             # every table that has data
    python analysis/xp07_tables.py --table e2  # just one

Tables whose experiment has not run print nothing rather than a placeholder, so
an empty section is visibly empty instead of quietly stale.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

RAW = REPO / "results" / "raw"


def side(name: str):
    p = RAW / name
    return json.loads(p.read_text()) if p.exists() else None


def f(v, n=4):
    return "—" if v is None else f"{v:.{n}f}"


def keep(v, base, n=0):
    """Share of the unquantized model retained, as a percentage."""
    return "—" if (v is None or not base) else f"{v / base * 100:.{n}f}%"


def head(title: str) -> None:
    print(f"\n### {title}\n")


# --------------------------------------------------------------------------
# E1 — per-layer sensitivity
# --------------------------------------------------------------------------

def t_e1() -> None:
    d = side("xp07e1_sensitivity.json")
    if not d:
        return
    bl = d["baseline_fp16"]
    bmap, btiny = bl["map50"], bl["tiny_plume"]["map50"]
    head("E1. Which layers can't take INT8")
    print(f"Val split, {d['n_val_images']} images, no retraining. "
          f"Unquantized {bmap:.4f} mAP50, {btiny:.4f} on tiny plumes.\n")

    for arm in ("w8", "w8a8"):
        rows = [r for r in d["rows"] if r["arm"] == arm]
        if not rows:
            continue
        drops = [r["rel_drop_pct"] for r in rows]
        h = [r for r in rows if r["is_head"]]
        nh = [r for r in rows if not r["is_head"]]
        label = "weights only" if arm == "w8" else "weights + input activation"
        print(f"**{arm.upper()} ({label})** — worst cell {max(drops):+.2f}%, "
              f"median {sorted(drops)[len(drops) // 2]:+.2f}%; "
              f"head mean {sum(x['rel_drop_pct'] for x in h) / len(h):+.3f}% vs "
              f"non-head {sum(x['rel_drop_pct'] for x in nh) / len(nh):+.3f}%\n")

    rows = [r for r in d["rows"] if r["arm"] == "w8a8"]
    print("Ranked on distant smoke, W8A8 — the ordering aggregate mAP50 hides:\n")
    print("| layer | tiny-plume mAP50 | kept | aggregate mAP50 fell | head? |")
    print("|---|---:|---:|---:|---|")
    print(f"| unquantized | {btiny:.4f} | 100% | — | |")
    for r in sorted(rows, key=lambda r: r["tiny_plume"])[:5]:
        print(f"| `{r['layer']}` | {r['tiny_plume']:.4f} | {keep(r['tiny_plume'], btiny)} | "
              f"{r['rel_drop_pct']:.1f}% | {'yes' if r['is_head'] else 'no'} |")

    for rank in ("map50", "tiny_plume"):
        worst = [r["layer"] for r in sorted(rows, key=lambda r: r[rank])[:3]]
        print(f"\nThree worst ranked by `{rank}`: {', '.join('`' + w + '`' for w in worst)}")


# --------------------------------------------------------------------------
# E2 — calibration
# --------------------------------------------------------------------------

def t_e2() -> None:
    d = side("xp07e2_calibration.json")
    if not d:
        return
    bl = d["baseline_fp16"]
    bmap, btiny = bl["map50"], bl["tiny_plume"]["map50"]
    head("E2. Calibration: method and size")
    print(f"{d['setting']}, {d['n_val_images']} val images.\n")
    print("| method | model size | mAP50 | kept | small plumes | tiny plumes | kept | input range |")
    print("|---|---:|---:|---:|---:|---:|---:|---|")
    fp16_mb = d.get("fp16_baseline_mb")
    print(f"| unquantized FP16 | {f(fp16_mb, 2)} MB | {bmap:.4f} | 100% | "
          f"{bl['small_plume']['map50']:.4f} | {btiny:.4f} | 100% | — |")
    for name, r in sorted(d["methods"].items(), key=lambda kv: -kv[1]["map50"]):
        mb = (r.get("size") or {}).get("total_mb")
        print(f"| {name} | {f(mb, 2)} MB | {r['map50']:.4f} | {keep(r['map50'], bmap)} | "
              f"{f(r['small_plume'])} | {f(r['tiny_plume'])} | {keep(r['tiny_plume'], btiny)} | "
              f"0–{r['input_represented_max']:.4f} |")
    print(f"\nSpread across methods: **{d['method_spread_map50']} mAP50**. "
          f"Best: **{d['best_method']}**.")
    sizes = {(r.get('size') or {}).get('total_mb') for r in d["methods"].values()}
    if len(sizes) == 1:
        print(f"Every method ships the same {next(iter(sizes))} MB model — the spread is free.")

    if d.get("sizes"):
        print("\n| calibration images | mAP50 | kept | tiny plumes | kept |")
        print("|---|---:|---:|---:|---:|")
        for n in sorted(d["sizes"], key=int):
            r = d["sizes"][n]
            print(f"| {n} | {r['map50']:.4f} | {keep(r['map50'], bmap)} | "
                  f"{f(r['tiny_plume'])} | {keep(r['tiny_plume'], btiny)} |")


# --------------------------------------------------------------------------
# E3 / E5 / E6 — the whole-network arms, all the same row shape
# --------------------------------------------------------------------------

def _arm_table(fname: str, title: str, note: str = "") -> None:
    d = side(fname)
    if not d:
        return
    bl = d["baseline_fp16"]
    bmap, btiny = bl["map50"], bl["tiny_plume"]["map50"]
    head(title)
    if note:
        print(note + "\n")
    print(f"Val split, {d['n_val_images']} images. Unquantized {bmap:.4f} mAP50, "
          f"{btiny:.4f} tiny.\n")
    print("| arm | layers quantized | size | mAP50 | kept | tiny plumes | kept | recovered |")
    print("|---|---:|---:|---:|---:|---:|---:|---|")
    for r in d["rows"]:
        dam, sz = r["damage"], (r.get("size") or {})
        rec = r.get("recovered")
        print(f"| {r['arm']} | {r['n_layers_quantized']} | {f(sz.get('total_mb'), 2)} MB | "
              f"{dam['map50']:.4f} | {keep(dam['map50'], bmap)} | {f(dam['tiny_plume'])} | "
              f"{keep(dam['tiny_plume'], btiny)} | "
              f"{f(rec['map50']) if rec else 'not run'} |")


def t_e3() -> None:
    _arm_table("xp07e3_granularity.json", "E3. Granularity of the weight scales")


def t_e5() -> None:
    d = side("xp07e5_targets.json")
    _arm_table("xp07e5_targets.json", "E5. Weights-only vs W8A8")
    if d and d.get("attribution"):
        print("\nWhere the loss comes from, as a share of the unquantized model:\n")
        for k, v in d["attribution"].items():
            print(f"- `{k}`: {v:+.3f}%")


def t_e6() -> None:
    for fname, how in (("xp07e6_mixed.json", "ranked by aggregate mAP50"),
                       ("xp07e6_mixed_tiny.json", "ranked by tiny-plume damage")):
        d = side(fname)
        if not d:
            continue
        _arm_table(fname, f"E6. Mixed precision — split {how}",
                   f"Layers left in FP16: "
                   f"{', '.join('`' + n + '`' for n in d.get('float_layers', []))}")
        if d.get("gaps"):
            print()
            for k, v in d["gaps"].items():
                print(f"- `{k}`: {v:+.4f} mAP50")


# --------------------------------------------------------------------------
# E4 — the board
# --------------------------------------------------------------------------

def t_e4() -> None:
    d = side("xp07e4_engines.json")
    if not d:
        return
    head("E4. The INT8 engine on the board")
    print(f"Full {d['n_test_images']}-image test set at {d['resolution']} px, "
          f"batch {d['batch']}, {d['board']}. Build path: {d['build_path']}.\n")
    print("| arm | engine | mAP50 | fire | tiny plumes | img/s | J/1k | convs in INT8 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for arm, r in d["arms"].items():
        pw = r.get("power") or {}
        rep = r.get("precision_report") or {}
        n8, tot = rep.get("convolutions_in_int8"), rep.get("convolutions_total")
        pc = (r.get("per_class") or {}).get("fire", {}).get("map50")
        print(f"| {arm} | {f(r.get('engine_mb'), 2)} MB | {f(r.get('map50'))} | "
              f"{f(pc)} | {f(r.get('tiny_plume'))} | {f(r.get('fps_batched'), 1)} | "
              f"{f(pw.get('j_per_1k'), 1)} | "
              f"{'—' if n8 is None else f'{n8}/{tot}'} |")
    v = d.get("rebuild_variance") or {}
    if v:
        print(f"\nTactic noise, `{v['arm']}` built {len(v['fps'])} times: "
              f"{', '.join(f'{x:.1f}' for x in v['fps'])} img/s "
              f"(**spread {v['spread_pct']}%**) — any speed claim smaller than this is noise.")


def t_e8() -> None:
    d = side("xp07e8_lowbit.json")
    if not d:
        return
    bl = d["baseline_fp16"]
    bmap, btiny = bl["map50"], bl["tiny_plume"]["map50"]
    head("E8. Below 8 bits, and storage-only compression")
    print(f"{d['board_claim']}. Activations: {d['activations']}.\n")
    print("| arm | size | scales | mAP50 | kept | tiny plumes | kept |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for name, r in d["arms"].items():
        sz = r.get("size") or {}
        print(f"| {name} | {f(sz.get('total_mb'), 3)} MB | "
              f"{f(sz.get('weight_scale_overhead_kb'), 1)} KB | "
              f"{f(r['map50'])} | {keep(r['map50'], bmap)} | "
              f"{f(r['tiny_plume'])} | {keep(r['tiny_plume'], btiny)} |")
    cb = d.get("codebook")
    if cb:
        print(f"\n**K-means codebook, 4-bit indices:** {cb['fp16_mb']} MB FP16 → "
              f"{cb['indices_mb']} MB raw indices → **{cb['huffman_mb']} MB Huffman** "
              f"({cb['mean_huffman_bits_per_weight']} bits/weight), "
              f"mAP50 {f(cb['map50'])}.")


# --------------------------------------------------------------------------
# E9 — the cross-technique frontier
# --------------------------------------------------------------------------

def t_e9() -> None:
    d = side("xp07e9_frontier.json")
    if not d:
        return
    head("E9. Every technique in the series against every other")
    print("Full test set, 512 px unless stated, every row an engine measured on the Orin.\n")
    print("| technique | family | mAP50 | fire | smoke | tiny plumes | MB | img/s | "
          "batch-1 ms | J/1k |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    fams = ["baseline", "resolution", "pruning", "sparsity", "quantization", "composition"]
    for fam in fams:
        for r in [x for x in d["rows"] if x["family"] == fam]:
            print(f"| {r['label']} | {fam} | {f(r['map50'])} | {f(r['map50_fire'])} | "
                  f"{f(r['map50_smoke'])} | {f(r['tiny_plume'])} | "
                  f"{f(r['size_disk_mb'], 1)} | {f(r['fps_batched'], 1)} | "
                  f"{f(r['latency_ms_median'], 2)} | {f(r['j_per_1k'], 1)} |"
                  + ("  <!-- paired -->" if r.get("accuracy_paired_from") else ""))

    v = d["verdict"]
    dom = v["dominates_the_line_on_accuracy_speed_and_size"]
    print(f"\n**Dominates the line on accuracy, speed and size at once:** "
          f"{', '.join(dom) if dom else '**nothing**'}.\n")
    print("| if you must keep | fastest arm that clears it | img/s | MB | J/1k |")
    print("|---|---|---:|---:|---:|")
    for k, b in v["fastest_at_each_accuracy_floor"].items():
        print(f"| {k.replace('map50>=', 'mAP50 ≥ ')} | {b['fastest']} ({b['family']}) | "
              f"{b['fps']:.0f} | {f(b['size_mb'], 1)} | {f(b['j_per_1k'], 0)} |")

    for key in ("map50_fire", "tiny_plume"):
        blk = (v.get("per_metric") or {}).get(key)
        if not blk:
            continue
        print(f"\n**{blk['title']}** — the line scores {blk['line_value']:.4f}.\n")
        print("| technique | value | keeps | img/s | MB |")
        print("|---|---:|---:|---:|---:|")
        for r in blk["retention"]:
            print(f"| {r['label']} | {f(r['value'])} | **{r['kept_pct']:.0f}%** | "
                  f"{r['fps']:.0f} | {f(r['size_mb'], 1)} |")


def t_cost() -> None:
    d = side("xp07e7_cost.json")
    if not d:
        return
    head("E7/E10. What 12 epochs actually costs on this board")
    print(f"Measured on the real {d['mode']} loop at {d['resolution']} px, "
          f"batch {d['batch']}, over {d['timed_steps']} timed steps.\n")
    print(f"- **{d['images_per_s']} img/s** forward+backward")
    print(f"- one epoch over {d['train_split_images']} images: "
          f"**{d['minutes_per_epoch']} min**")
    print(f"- 12 epochs, one trained arm: **{d['hours_for_12_epochs']} h**")
    print(f"- E7 as specified (control + 2 trained arms): "
          f"**{d['hours_for_e7_as_specified']} h**")
    print(f"\n{d['note']}")


TABLES = {"e1": t_e1, "e2": t_e2, "e3": t_e3, "e4": t_e4,
          "e5": t_e5, "e6": t_e6, "e8": t_e8, "e9": t_e9, "cost": t_cost}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", choices=sorted(TABLES))
    args = ap.parse_args()
    for key in ([args.table] if args.table else
                ["e1", "e2", "e3", "e5", "e6", "e4", "e8", "e9", "cost"]):
        TABLES[key]()
    print()


if __name__ == "__main__":
    main()
