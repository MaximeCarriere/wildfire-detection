#!/usr/bin/env python3
"""Markdown tables for the XP6 pruning README, built from the committed JSON.

The repo's figures are generated and never hand edited, for a reason that
applies just as strongly to tables: a number retyped into a README is a number
that can drift from the measurement it claims to report, silently, and nothing
catches it. So the tables are generated too.

Run it and paste, or diff its output against the README to check the page still
matches the evidence:

    python analysis/xp06_tables.py            # every table
    python analysis/xp06_tables.py --table e2 # just one
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

RAW = REPO / "results" / "raw"

#: Re-measured on the screening machine; every table leads with it.
UNPRUNED = {"params": "7.03 M", "map50": 0.7764, "small": 0.6038,
            "tiny": 0.1380, "silent": 0.9736}


def records() -> list[dict]:
    out = []
    for p in sorted(RAW.glob("*.json")):
        d = json.loads(p.read_text())
        if "model_id" in d:
            d["_src"] = p.name
            out.append(d)
    return out


def side(name: str):
    p = RAW / name
    return json.loads(p.read_text()) if p.exists() else None


def f(v, n=4):
    return "-" if v is None else f"{v:.{n}f}"


def pct(v, n=1):
    return "-" if v is None else f"{100*v:.{n}f}%"


def head(title: str) -> None:
    print(f"\n### {title}\n")


def row_unpruned(extra_cols: int = 0, first: str = "**none (unpruned)**") -> str:
    tail = " | ".join(["-"] * extra_cols)
    return (f"| {first} | {UNPRUNED['params']} | **{UNPRUNED['map50']:.4f}** | "
            f"{UNPRUNED['small']:.4f} | {UNPRUNED['tiny']:.4f} | "
            f"{UNPRUNED['silent']:.4f} |" + (f" {tail} |" if extra_cols else ""))


def t_e1() -> None:
    d = side("xp06e1_sensitivity.json")
    if not d:
        return
    head("E1. Which layers can be pruned, and what each cut actually buys")
    rows = [r for r in d["rows"] if "retained" in r and r["ratio"] == 0.5]
    rows.sort(key=lambda r: r["retained"])
    print("At a 50% cut applied to that layer alone, val split, no retraining.")
    print(f"Baseline val mAP50 {d['baseline']['map50']:.4f}.\n")
    print("| layer | accuracy retained | parameters freed |")
    print("|---|---:|---:|")
    for r in rows[:5]:
        print(f"| `{r['layer']}` | **{pct(r['retained'])}** | {pct(r['params_reduction'], 2)} |")
    print("| … | | |")
    for r in rows[-5:]:
        print(f"| `{r['layer']}` | {pct(r['retained'])} | **{pct(r['params_reduction'], 2)}** |")

    def stage(n):
        m = re.match(r"model\.(\d+)\.", n)
        return int(m.group(1)) if m else -1
    early = [r for r in rows if stage(r["layer"]) <= 4]
    deep = [r for r in rows if stage(r["layer"]) >= 6]
    print(f"\nStages 0-4 retain {sum(r['retained'] for r in early)/len(early):.1%} on average; "
          f"stages 6-23 retain {sum(r['retained'] for r in deep)/len(deep):.1%}.")


def t_e2() -> None:
    d = side("xp06e2_criteria_damage.json")
    if not d:
        return
    head("E2. Criterion, at a fixed 25% one-shot cut")
    base = d["baseline"]["val_map50"]
    order = ["l1", "fpgm", "taylor", "lamp", "hessian", "l2", "random", "bn"]
    pretty = {"l1": "L1 magnitude", "l2": "**L2 magnitude** (XP6's arm)", "bn": "BN scale",
              "taylor": "Taylor", "hessian": "Hessian", "fpgm": "FPGM", "lamp": "LAMP",
              "random": "**random** (control)"}
    cells = {(r["criterion"], r["ratio"]): r for r in d["rows"] if "val_map50" in r}
    print("Damage with no retraining. Val split for the ranking, since it selects which")
    print("criteria earn a recovery run; test at 25% so the numbers sit beside XP6's table.\n")
    print(f"| criterion | val @2% | val @5% | val @25% | test @25% |")
    print("|---|---:|---:|---:|---:|")
    print(f"| none (unpruned) | {base:.4f} | {base:.4f} | {base:.4f} | {UNPRUNED['map50']:.4f} |")
    for c in order:
        if (c, 0.05) not in cells:
            continue
        r25 = cells.get((c, 0.25), {})
        print(f"| {pretty[c]} | {f(cells[(c,0.02)]['val_map50'])} | "
              f"{f(cells[(c,0.05)]['val_map50'])} | {f(r25.get('val_map50'))} | "
              f"{f(r25.get('test_map50'))} |")

    recs = [r for r in records() if r.get("experiment") == "xp06e2"]
    if recs:
        print("\nAfter 12 epochs of recovery, full test set:\n")
        print("| criterion | params | mAP50 | small plumes | tiny plumes | correctly silent |")
        print("|---|---:|---:|---:|---:|---:|")
        print(row_unpruned())
        for r in sorted(recs, key=lambda r: -r["map50_dfire_test"]):
            name = r["model_id"].split("_pruned25_")[-1].replace("_recovered", "")
            print(f"| {name} | {r['params_m']:.2f} M | {f(r['map50_dfire_test'])} | "
                  f"{f(r['map50_small_plume'])} | {f(r['map50_tiny_plume'])} | "
                  f"{f(r['bg_correctly_silent_rate'])} |")
        old = next((r for r in records()
                    if r["model_id"] == "dfire_yolov5s_pruned25_recovered"), None)
        if old:
            print(f"| L2 (XP6, same protocol) | {old['params_m']:.2f} M | "
                  f"{f(old['map50_dfire_test'])} | {f(old['map50_small_plume'])} | "
                  f"{f(old['map50_tiny_plume'])} | {f(old.get('bg_correctly_silent_rate'))} |")


def t_e5() -> None:
    recs = [r for r in records() if r.get("granularity") == "unstructured"]
    if not recs:
        return
    head("E5. Granularity: individual weights versus whole channels")
    print("No speed number appears here, on any machine: irregular zeros have no matching")
    print("kernels on this hardware. Accuracy only, which is the question being asked.\n")
    print("| sparsity | non-zero params | mAP50, no retraining | mAP50 after 12 epochs | small | tiny |")
    print("|---|---:|---:|---:|---:|---:|")
    print(f"| **none (unpruned)** | {UNPRUNED['params']} | {UNPRUNED['map50']:.4f} | "
          f"**{UNPRUNED['map50']:.4f}** | {UNPRUNED['small']:.4f} | {UNPRUNED['tiny']:.4f} |")
    for r in sorted(recs, key=lambda r: r["prune_meta"]["requested_sparsity"]):
        m = r["prune_meta"]
        nz = m.get("nonzero_params_m") or 0.0
        # A damage-only run has no recovery. Printing its damage figure in the
        # recovered column would silently claim a training budget it never had.
        recovered = r.get("train_meta") is not None
        after = f(r["map50_dfire_test"]) if recovered else "not retrained"
        small = f(r["map50_small_plume"]) if recovered else "-"
        tiny = f(r["map50_tiny_plume"]) if recovered else "-"
        print(f"| {m['requested_sparsity']:.0%} of all weights | {nz:.2f} M | "
              f"{f(m.get('map50_before_recovery'))} | {after} | {small} | {tiny} |")
    # "_nofinetune" alone now also matches the fine-grained damage records, which
    # are a different granularity and must not be listed as channel pruning.
    chan = [r for r in records() if "_nofinetune" in r["model_id"]
            and "requested_channel_ratio" in r.get("prune_meta", {})]
    if chan:
        print("\nFor comparison, whole-channel removal with no retraining:\n")
        print("| channels cut | params | mAP50 |")
        print("|---|---:|---:|")
        for r in sorted(chan, key=lambda r: r["prune_meta"]["requested_channel_ratio"]):
            print(f"| {r['prune_meta']['requested_channel_ratio']:.0%} | "
                  f"{r['params_m']:.2f} M | {f(r['map50_dfire_test'])} |")


def t_e6() -> None:
    recs = [r for r in records() if r.get("experiment") == "xp06e6"]
    if not recs:
        return
    head("E6. Ratio allocation, at matched size")
    plan = side("xp06e6_allocation_plan.json") or {}
    print(f"All three arms use the L1 criterion and are matched to "
          f"{plan.get('target_params_reduction', 0):.1%} parameter reduction by search, so")
    print("only the per-layer distribution of the cut varies.\n")
    print("| allocation | params | MACs cut | mAP50 | small plumes | tiny plumes | correctly silent |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    print(row_unpruned(extra_cols=0, first="**none (unpruned)**").replace(
        f"| {UNPRUNED['params']} |", f"| {UNPRUNED['params']} | - |"))
    for a in ("global", "uniform", "sensitivity"):
        r = next((x for x in recs if x.get("allocation") == a), None)
        if not r:
            continue
        print(f"| {a} | {r['params_m']:.2f} M | {pct(r['prune_meta'].get('macs_reduction'))} | "
              f"{f(r['map50_dfire_test'])} | {f(r['map50_small_plume'])} | "
              f"{f(r['map50_tiny_plume'])} | {f(r['bg_correctly_silent_rate'])} |")


def t_e7() -> None:
    recs = [r for r in records() if r.get("experiment") == "xp06e7"]
    if not recs:
        return
    head("E7. One-shot versus iterative, with equal post-cut training")
    print("| arm | epochs after final cut | total epochs | params | mAP50 | small | tiny |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for r in sorted(recs, key=lambda r: (r.get("arm", ""), r.get("post_cut_epochs", 0))):
        print(f"| {r.get('arm')} | {r.get('post_cut_epochs')} | {r.get('total_epochs')} | "
              f"{r['params_m']:.2f} M | {f(r['map50_dfire_test'])} | "
              f"{f(r['map50_small_plume'])} | {f(r['map50_tiny_plume'])} |")


def t_e3() -> None:
    recs = [r for r in records() if r.get("experiment") == "xp06e3"]
    if not recs:
        return
    head("E3. Regularity (accuracy here; throughput needs the board)")
    print("| round_to | params | MACs cut | widths divisible by 32 | mAP50 | throughput |")
    print("|---|---:|---:|---:|---:|---|")
    for r in sorted(recs, key=lambda r: r.get("round_to", 0)):
        m = r["prune_meta"]
        print(f"| {r.get('round_to')} | {r['params_m']:.2f} M | "
              f"{pct(m.get('macs_reduction'))} | "
              f"{m.get('widths_divisible_by_32')} of {m.get('n_conv_layers')} | "
              f"{f(r['map50_dfire_test'])} | not measured here |")


def t_e4() -> None:
    recs = [r for r in records() if r.get("granularity") == "2:4"]
    if not recs:
        return
    head("E4. 2:4 sparsity (accuracy here; sparse-kernel selection and speed need the board)")
    print("| model | non-zero params | mAP50 before recovery | mAP50 after | small | tiny |")
    print("|---|---:|---:|---:|---:|---:|")
    for r in recs:
        m = r["prune_meta"]
        print(f"| 2:4 masked | {m.get('nonzero_params_m', 0):.2f} M | "
              f"{f(m.get('map50_before_recovery'))} | {f(r['map50_dfire_test'])} | "
              f"{f(r['map50_small_plume'])} | {f(r['map50_tiny_plume'])} |")


TABLES = {"e1": t_e1, "e2": t_e2, "e3": t_e3, "e4": t_e4,
          "e5": t_e5, "e6": t_e6, "e7": t_e7}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", choices=sorted(TABLES))
    args = ap.parse_args()
    for key in ([args.table] if args.table else sorted(TABLES)):
        TABLES[key]()
    print()


if __name__ == "__main__":
    main()
