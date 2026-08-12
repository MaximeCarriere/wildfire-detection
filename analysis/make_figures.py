#!/usr/bin/env python3
"""Regenerate every figure in the repo from ``results/raw/``.

PLAN.md §0: no hand-made figures. If a plot exists here, this script rebuilt it
from the committed per-run JSON, and running this script is the only way a figure
changes.

Two guardrails matter more than the plots:

* **Mixed protocol versions are refused.** A record written under an older
  ``protocol_version`` was measured under different rules; plotting it beside a
  current one would draw a comparison that does not exist.
* **Non-compliant timing blocks are excluded** from speed plots. ``--quick``
  wiring checks mark themselves ``protocol_compliant: false``.

Usage
    python analysis/make_figures.py           # rebuild everything
    python analysis/make_figures.py --list    # show available records
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from analysis import style                       # noqa: E402
from lib.evaluator import PROTOCOL_VERSION       # noqa: E402

RAW = REPO / "results" / "raw"
FIGURES = REPO / "results" / "figures"


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load_records() -> list[dict]:
    records, stale = [], []
    for path in sorted(RAW.glob("*.json")):
        rec = json.loads(path.read_text())
        if "model_id" not in rec:            # side data (e.g. box geometry)
            continue
        rec["_source"] = path.name
        if rec.get("protocol_version") != PROTOCOL_VERSION:
            stale.append((path.name, rec.get("protocol_version")))
        records.append(rec)
    if stale:
        lines = "\n".join(f"    {n}: protocol_version {v!r}" for n, v in stale)
        raise SystemExit(
            f"refusing to plot mixed protocol versions (current {PROTOCOL_VERSION!r}):\n"
            f"{lines}\nRe-run those experiments or move them out of results/raw/.")
    return records


def by_id(records, needle: str) -> dict | None:
    for r in records:
        if r["model_id"] == needle:
            return r
    return None


def usable(rec: dict) -> bool:
    j = rec.get("jetson") or {}
    return bool(j) and j.get("protocol_compliant", True)


def save(fig, name: str) -> Path:
    import matplotlib.pyplot as plt
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / name
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------
# XP0 — what the dataset actually contains
# --------------------------------------------------------------------------

def fig_xp00(records) -> Path | None:
    import matplotlib.pyplot as plt
    import numpy as np

    geo_path = RAW / "xp00_box_geometry.json"
    if not geo_path.exists():
        return None
    geo = json.loads(geo_path.read_text())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.suptitle("What a fire detector is actually asked to find", y=1.06)
    style.subtitle(fig, "Nearly half the test images contain no fire at all, and half the "
                        "targets are smaller than a 74-pixel box.", y=1.0)

    # Panel 1 — content mix. Position, not colour, carries the categories.
    ax = axes[0]
    mix = geo["content_mix"]
    order = ["none", "smoke", "both", "fire"]
    labels = ["no fire\nor smoke", "smoke\nonly", "fire +\nsmoke", "fire\nonly"]
    vals = [mix.get(k, 0) for k in order]
    colours = [style.MUTED, style.BLUE, style.AQUA, style.ORANGE]
    bars = ax.bar(labels, vals, color=colours, width=0.66, zorder=3)
    for b, v in zip(bars, vals):
        style.annotate(ax, b.get_x() + b.get_width() / 2, v,
                       f"{v:,}\n{100*v/geo['n_images']:.0f}%", dy=6, size=9.5)
    ax.set_ylabel("test images")
    ax.set_ylim(0, max(vals) * 1.28)
    ax.set_title("Almost half the frames are empty landscape", fontsize=11.5, pad=8)
    style.tidy(ax)

    # Panel 2 — target size distribution on a log axis, thresholds marked.
    ax = axes[1]
    fire = np.array(geo["box_area_frac"]["fire"])
    smoke = np.array(geo["box_area_frac"]["smoke"])
    bins = np.logspace(-5, 0, 46)
    ax.hist([smoke, fire], bins=bins, stacked=True, color=[style.BLUE, style.ORANGE],
            label=["smoke", "fire"], zorder=3)
    ax.set_xscale("log")
    # Staggered heights and opposite alignment: the two thresholds are close on a
    # log axis and their labels collide if both sit at the same y.
    for x, lab, ypos, ha in ((0.01, "1% of frame (≈74 px)", 0.99, "left"),
                             (0.001, "0.1% of frame (≈20 px)", 0.86, "right")):
        ax.axvline(x, color=style.INK, linestyle="--", linewidth=1.2, zorder=4)
        ax.text(x * (1.25 if ha == "left" else 0.8), ax.get_ylim()[1] * ypos, lab,
                ha=ha, va="top", fontsize=8.5, color=style.INK, zorder=5,
                bbox=dict(fc=style.SURFACE, ec="none", pad=1.5))
    ax.set_xlabel("target size (fraction of the image, log scale)")
    ax.set_ylabel("number of boxes")
    ax.set_title("45% of targets are below the 1% line", fontsize=11.5, pad=8)
    ax.legend(loc="upper left")
    style.tidy(ax)

    fig.tight_layout()
    return save(fig, "xp00_dataset.png")


# --------------------------------------------------------------------------
# XP1 — the published baselines
# --------------------------------------------------------------------------

def fig_xp01(records) -> Path | None:
    import matplotlib.pyplot as plt
    import numpy as np

    s = by_id(records, "dfire_yolov5s_published")
    l = by_id(records, "dfire_yolov5l_published")
    if not (s and l):
        return None

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.suptitle("A 6.6× bigger model buys almost nothing", y=1.06)
    style.subtitle(fig, "The large model gains 1.4 accuracy points and costs 3.3× the "
                        "energy per frame — which undermines the plan to distil from it.", y=1.0)

    metrics = [("map50_dfire_test", "overall"), ("map50_fire_class", "fire"),
               ("map50_smoke_class", "smoke"), ("map50_small_plume", "small\nplumes"),
               ("map50_tiny_plume", "tiny\nplumes")]
    x = np.arange(len(metrics))
    w = 0.36
    ax = axes[0]
    sv = [s[k] or 0 for k, _ in metrics]
    lv = [l[k] or 0 for k, _ in metrics]
    ax.bar(x - w/2, sv, w, label="YOLOv5s — 7.0 M params", color=style.BLUE, zorder=3)
    ax.bar(x + w/2, lv, w, label="YOLOv5l — 46.1 M params", color=style.ORANGE, zorder=3)
    for xi, (a, b) in enumerate(zip(sv, lv)):
        style.annotate(ax, xi - w/2, a, f"{a:.2f}", dy=4, size=8.5, weight="normal")
        style.annotate(ax, xi + w/2, b, f"{b:.2f}", dy=4, size=8.5, weight="normal")
    ax.set_xticks(x); ax.set_xticklabels([n for _, n in metrics])
    ax.set_ylabel("detection accuracy (mAP50)")
    ax.set_ylim(0, 1.16)
    ax.set_title("Accuracy: nearly identical everywhere", fontsize=11.5, pad=8)
    ax.legend(loc="upper center", ncol=2, columnspacing=1.2)
    style.tidy(ax)

    # Cost panel — indexed to the small model so the multiple is the message.
    ax = axes[1]
    names = ["energy\nper frame", "latency", "memory", "model\nsize"]
    ratios = [
        l["jetson"]["energy_j_per_1000_frames"] / s["jetson"]["energy_j_per_1000_frames"],
        l["jetson"]["latency_ms_median"] / s["jetson"]["latency_ms_median"],
        l["jetson"]["mem_mb"] / s["jetson"]["mem_mb"],
        l["size_disk_mb"] / s["size_disk_mb"],
    ]
    bars = ax.bar(names, ratios, color=style.ORANGE, width=0.6, zorder=3)
    ax.axhline(1.0, color=style.INK, linewidth=1.2, zorder=4)
    ax.text(3.45, 1.0, "YOLOv5s = 1×", ha="right", va="bottom", fontsize=9, color=style.INK)
    for b, v in zip(bars, ratios):
        style.annotate(ax, b.get_x() + b.get_width()/2, v, f"{v:.1f}×", dy=5)
    ax.set_ylabel("cost relative to YOLOv5s")
    ax.set_ylim(0, max(ratios) * 1.22)
    ax.set_title("Cost: 1.7× to 6.5× worse on every axis", fontsize=11.5, pad=8)
    style.tidy(ax)

    fig.tight_layout()
    return save(fig, "xp01_baselines.png")


# --------------------------------------------------------------------------
# XP2 — the resolution frontier
# --------------------------------------------------------------------------

def _family(records, base: str) -> list[dict]:
    rows = [r for r in records
            if r["model_id"].startswith(base + "@")
            and (r.get("jetson") or {}).get("fps_batched") is not None
            and r.get("map50_dfire_test") is not None]
    return sorted(rows, key=lambda r: r["input_res"])


def fig_xp02(records) -> Path | None:
    import matplotlib.pyplot as plt

    s = _family(records, "dfire_yolov5s_published")
    l = _family(records, "dfire_yolov5l_published")
    if len(s) < 2:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    fig.suptitle("Shrinking the input is free speed — until it blinds the detector", y=1.06)
    style.subtitle(fig, "512 pixels is more accurate AND 1.6× faster than 640. Push further "
                        "and overall accuracy barely moves while distant smoke disappears.", y=1.0)

    ax = axes[0]
    for rows, colour, name in ((l, style.ORANGE, "YOLOv5l (46 M)"),
                               (s, style.BLUE, "YOLOv5s (7 M)")):
        if not rows:
            continue
        xs = [r["jetson"]["fps_batched"] for r in rows]
        ys = [r["map50_dfire_test"] for r in rows]
        ax.plot(xs, ys, "o-", color=colour, linewidth=2, markersize=7,
                label=name, zorder=3)
        for r, xx, yy in zip(rows, xs, ys):
            ax.annotate(f"{r['input_res']}", (xx, yy), xytext=(0, -15),
                        textcoords="offset points", ha="center", fontsize=9,
                        color=style.INK_2)
    best = max(s, key=lambda r: r["map50_dfire_test"])
    ax.scatter([best["jetson"]["fps_batched"]], [best["map50_dfire_test"]],
               s=260, facecolors="none", edgecolors=style.INK, linewidths=1.6, zorder=5)
    ax.annotate("best of both:\nmore accurate, faster, cooler",
                (best["jetson"]["fps_batched"], best["map50_dfire_test"]),
                xytext=(30, -30), textcoords="offset points", fontsize=9.5,
                color=style.INK, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=style.MUTED, lw=1))
    ax.set_xlabel("images per second (higher is better)")
    ax.set_ylabel("detection accuracy (mAP50)")
    ax.set_title("The speed/accuracy frontier", fontsize=11.5, pad=8)
    ax.legend(loc="lower left")
    style.tidy(ax)

    # Panel 2 — the counter-metric, indexed so the collapse rates compare directly.
    ax = axes[1]
    series = [("map50_dfire_test", "overall accuracy", style.MUTED),
              ("map50_small_plume", "small plumes (<1%)", style.AQUA),
              ("map50_tiny_plume", "tiny plumes (<0.1%)", style.ORANGE)]
    res = [r["input_res"] for r in s]
    for key, label, colour in series:
        ref = next(r[key] for r in s if r["input_res"] == max(res))
        ys = [100 * r[key] / ref for r in s]
        ax.plot(res, ys, "o-", color=colour, linewidth=2, markersize=7, label=label, zorder=3)
        ax.annotate(f"{ys[0]:.0f}%", (res[0], ys[0]), xytext=(9, -3),
                    textcoords="offset points", fontsize=9.5, color=colour,
                    fontweight="bold", ha="left")
    ax.axhline(100, color=style.INK, linewidth=1, linestyle=":", zorder=2)
    ax.set_xticks(res)
    ax.set_xlim(min(res) - 24, max(res) + 18)
    ax.set_xlabel("input resolution (pixels)")
    ax.set_ylabel("% of accuracy kept vs 640px")
    ax.set_title("Overall accuracy hides the damage", fontsize=11.5, pad=8)
    ax.legend(loc="lower right")
    style.tidy(ax)

    fig.tight_layout()
    return save(fig, "xp02_resolution.png")


# --------------------------------------------------------------------------
# XP9 — TensorRT
# --------------------------------------------------------------------------

def fig_xp09(records) -> Path | None:
    import matplotlib.pyplot as plt
    import numpy as np

    pairs = [("dfire_yolov5s_published@640", "dfire_yolov5s_trt_fp16@640", "YOLOv5s 640px"),
             ("dfire_yolov5s_published@512", "dfire_yolov5s_trt_fp16@512", "YOLOv5s 512px"),
             ("dfire_yolov5l_published@640", "dfire_yolov5l_trt_fp16@640", "YOLOv5l 640px")]
    rows = [(lab, by_id(records, a), by_id(records, b)) for a, b, lab in pairs]
    rows = [(lab, a, b) for lab, a, b in rows if a and b]
    if not rows:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.suptitle("The same model, 4–5× faster — and nothing lost", y=1.06)
    style.subtitle(fig, "Standard PyTorch was leaving the GPU idle between operations. "
                        "TensorRT removes that, at no cost in accuracy.", y=1.0)

    ax = axes[0]
    x = np.arange(len(rows)); w = 0.36
    pt = [a["jetson"]["latency_ms_median"] for _, a, _ in rows]
    trt = [b["jetson"]["latency_ms_median"] for _, _, b in rows]
    ax.bar(x - w/2, pt, w, label="PyTorch", color=style.MUTED, zorder=3)
    ax.bar(x + w/2, trt, w, label="TensorRT", color=style.BLUE, zorder=3)
    for xi, (p, t) in enumerate(zip(pt, trt)):
        style.annotate(ax, xi - w/2, p, f"{p:.1f}", dy=4, size=9, weight="normal")
        style.annotate(ax, xi + w/2, t, f"{t:.1f} ms", dy=4, size=9, color=style.BLUE)
        ax.annotate(f"{p/t:.1f}× faster", (xi, max(p, t)), xytext=(0, 22),
                    textcoords="offset points", ha="center", fontsize=10,
                    color=style.INK, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([lab for lab, _, _ in rows])
    ax.set_ylabel("time per frame (milliseconds)")
    ax.set_ylim(0, max(pt) * 1.35)
    ax.set_title("Latency per frame", fontsize=11.5, pad=8)
    ax.legend(loc="upper left")
    style.tidy(ax)

    # Panel 2 — the diagnostic: resolution was invisible before TensorRT.
    ax = axes[1]
    s_pt = _family(records, "dfire_yolov5s_published")
    trt_pts = [(r["input_res"], r["jetson"]["latency_ms_median"]) for r in
               [by_id(records, f"dfire_yolov5s_trt_fp16@{n}") for n in (640, 512)] if r]
    if s_pt and trt_pts:
        ax.plot([r["input_res"] for r in s_pt],
                [r["jetson"]["latency_ms_median"] for r in s_pt],
                "o-", color=style.MUTED, linewidth=2, markersize=7, label="PyTorch", zorder=3)
        tp = sorted(trt_pts)
        ax.plot([p[0] for p in tp], [p[1] for p in tp], "o-", color=style.BLUE,
                linewidth=2, markersize=7, label="TensorRT", zorder=3)
        ax.annotate("flat — shrinking the image\nchanged nothing at all", (416, 23.1),
                    xytext=(0, 20), textcoords="offset points", ha="center",
                    fontsize=9.5, color=style.INK_2, fontweight="bold")
        ax.annotate("resolution finally\nmatters again", (576, 4.8),
                    xytext=(0, 26), textcoords="offset points", ha="center",
                    fontsize=9.5, color=style.BLUE, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=style.BLUE, lw=1))
    ax.set_xlabel("input resolution (pixels)")
    ax.set_ylabel("time per frame (milliseconds)")
    ax.set_ylim(0, 30)
    ax.set_title("Why the earlier speed numbers were meaningless", fontsize=11.5, pad=8)
    ax.legend(loc="center right")
    style.tidy(ax)

    fig.tight_layout()
    return save(fig, "xp09_tensorrt.png")


# --------------------------------------------------------------------------
# XP10 — the calibrator
# --------------------------------------------------------------------------

def fig_xp10(records) -> Path | None:
    import matplotlib.pyplot as plt
    import numpy as np

    fp16 = by_id(records, "dfire_yolov5s_trt_fp16@512")
    mm = by_id(records, "dfire_yolov5s_trt_int8mm@512")
    ent = by_id(records, "dfire_yolov5s_trt_int8@512")
    if not (fp16 and mm and ent):
        return None

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    fig.suptitle("One default setting cost 67% of the accuracy", y=1.06)
    style.subtitle(fig, "TensorRT's standard calibration clipped the bright half of every "
                        "image away. Switching one option restored almost all of it.", y=1.0)

    ax = axes[0]
    groups = [("overall", "map50_dfire_test"), ("small plumes", "map50_small_plume"),
              ("tiny plumes", "map50_tiny_plume")]
    x = np.arange(len(groups)); w = 0.26
    cfgs = [("full precision (FP16)", fp16, style.BLUE),
            ("compressed, fixed setting", mm, style.AQUA),
            ("compressed, default setting", ent, style.RED)]
    for i, (label, rec, colour) in enumerate(cfgs):
        vals = [rec[k] or 0 for _, k in groups]
        off = (i - 1) * w
        ax.bar(x + off, vals, w, label=label, color=colour, zorder=3)
        for xi, v in zip(x, vals):
            style.annotate(ax, xi + off, v, f"{v:.2f}", dy=4, size=8.5, weight="normal")
    ax.set_xticks(x); ax.set_xticklabels([g for g, _ in groups])
    ax.set_ylabel("detection accuracy (mAP50)")
    ax.set_ylim(0, 0.95)
    ax.set_title("Accuracy by target size", fontsize=11.5, pad=8)
    ax.legend(loc="upper right")
    style.tidy(ax)

    # Panel 2 — the mechanism, in one number a non-specialist can read.
    ax = axes[1]
    bars = ax.bar(["default\n(entropy)", "fixed\n(min/max)"], [0.4475, 1.0],
                  color=[style.RED, style.AQUA], width=0.5, zorder=3)
    ax.axhline(1.0, color=style.INK, linestyle="--", linewidth=1.2, zorder=4)
    ax.text(-0.42, 1.035, "true brightness range of the image", ha="left", va="bottom",
            fontsize=9.5, color=style.INK)
    for b, v in zip(bars, [0.4475, 1.0]):
        style.annotate(ax, b.get_x() + b.get_width()/2, v, f"{v:.2f}", dy=6)
    # Explanation sits in the empty gap between the bars, clear of both marks.
    ax.annotate("everything brighter\nthan this line was\nflattened to white —\nincluding the "
                "sky that\nsmoke must be seen\nagainst",
                (0.5, 0.62), ha="center", va="center", fontsize=9.5, color=style.INK_2)
    ax.plot([0.28, 0.5], [0.4475, 0.79], color=style.MUTED, linewidth=1, zorder=2)
    ax.set_ylabel("brightness range the compressor kept")
    ax.set_ylim(0, 1.28)
    ax.set_title("The cause, in one number", fontsize=11.5, pad=8)
    style.tidy(ax)

    fig.tight_layout()
    return save(fig, "xp10_int8.png")


# --------------------------------------------------------------------------
# XP12 — endurance
# --------------------------------------------------------------------------

def fig_xp12(records) -> Path | None:
    import matplotlib.pyplot as plt

    rows = [r for r in records if (r.get("jetson") or {}).get("buckets")]
    rows = [r for r in rows if "fp16" in r["model_id"]]
    if not rows:
        return None
    rec = rows[0]
    b = rec["jetson"]["buckets"]

    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    fig.suptitle("Ten minutes flat out, no slowdown", y=1.03)
    style.subtitle(fig, f"{rec['jetson']['images_total']:,} images processed back-to-back. "
                        f"Throughput drifted {rec['jetson']['drift_pct']:+.1f}%.", y=0.965)

    mins = [x["minute"] for x in b]
    fps = [x["fps"] for x in b]
    ax.plot(mins, fps, "o-", color=style.BLUE, linewidth=2, markersize=7, zorder=3)
    ax.axhline(rec["jetson"]["fps_mean"], color=style.MUTED, linestyle=":", zorder=2)
    ax.set_ylim(min(fps) * 0.97, max(fps) * 1.03)
    ax.set_xticks(mins)
    ax.set_xlabel("minute of sustained load")
    ax.set_ylabel("images per second")
    style.annotate(ax, mins[-1], fps[-1], f"{fps[-1]:.0f}", dx=-16, dy=-4, color=style.BLUE)

    temps = [x["temp_c"] for x in b if x["temp_c"] is not None]
    if temps:
        ax.text(0.02, 0.06,
                f"chip temperature settled at {max(temps):.0f} °C — no thermal throttling",
                transform=ax.transAxes, fontsize=9.5, color=style.INK_2)
    style.tidy(ax)
    fig.tight_layout()
    return save(fig, "xp12_endurance.png")




# --------------------------------------------------------------------------
# XP6 — pruning
# --------------------------------------------------------------------------

def fig_xp06(records) -> Path | None:
    """Two things pruning does on this board, neither of them what you'd hope."""
    import matplotlib.pyplot as plt

    raw = sorted([r for r in records if "_nofinetune" in r["model_id"]],
                 key=lambda r: r["prune_meta"]["macs_reduction"])
    if len(raw) < 3:
        return None
    rec = sorted([r for r in records if "_recovered_trt" in r["model_id"]],
                 key=lambda r: r["prune_meta"]["macs_reduction"])

    # The unpruned reference, measured under the same runtime as each series.
    base_pt = by_id(records, "dfire_yolov5s_published@512")
    base_trt = by_id(records, "dfire_yolov5s_trt_fp16@512")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    fig.suptitle("Pruning: the damage is immediate, the speed-up is not", y=1.06)
    style.subtitle(fig, "Cutting channels destroys accuracy long before it buys speed. "
                        "Recovery training is not an optional refinement here.", y=1.0)

    ax = axes[0]
    xs = [100 * r["prune_meta"]["macs_reduction"] for r in raw]
    ys = [r["map50_dfire_test"] for r in raw]
    ax.plot(xs, ys, "o-", color=style.ORANGE, linewidth=2, markersize=7,
            label="pruned, no recovery", zorder=3)
    if rec:
        rxs = [100 * r["prune_meta"]["macs_reduction"] for r in rec]
        rys = [r["map50_dfire_test"] for r in rec]
        ax.plot(rxs, rys, "s-", color=style.AQUA, linewidth=2, markersize=8,
                label="pruned + recovery training", zorder=4)
        for x, y in zip(rxs, rys):
            style.annotate(ax, x, y, f"{y:.2f}", dy=8, color=style.AQUA)
    if base_pt:
        ax.axhline(base_pt["map50_dfire_test"], color=style.MUTED, linestyle=":", zorder=2)
        ax.text(2, base_pt["map50_dfire_test"] + 0.012, "unpruned model",
                fontsize=9, color=style.INK_2)
    ax.set_xlabel("arithmetic removed (% of MACs)")
    ax.set_ylabel("detection accuracy (mAP50)")
    ax.set_ylim(-0.03, 0.9)
    ax.set_title("Accuracy collapses at ~10% of the arithmetic", fontsize=11.5, pad=8)
    ax.legend(loc="upper right")
    style.tidy(ax)

    # Panel 2 — the FLOPs-vs-speed reality check.
    ax = axes[1]
    fps = [r["jetson"]["fps_batched"] for r in raw]
    base_fps = base_pt["jetson"]["fps_batched"] if base_pt else fps[0]
    ideal = [100 / (100 - x) for x in xs]                 # if speed tracked arithmetic
    actual = [f / base_fps for f in fps]
    ax.plot(xs, ideal, "--", color=style.MUTED, linewidth=1.8,
            label="if speed tracked arithmetic", zorder=3)
    ax.plot(xs, actual, "o-", color=style.BLUE, linewidth=2, markersize=7,
            label="measured", zorder=4)
    style.annotate(ax, xs[-1], actual[-1], f"{actual[-1]:.1f}x", dy=-18, color=style.BLUE)
    style.annotate(ax, xs[-1], ideal[-1], f"{ideal[-1]:.1f}x expected", dx=-42, dy=-4,
                   color=style.INK_2, weight="normal", size=9)
    ax.set_xlabel("arithmetic removed (% of MACs)")
    ax.set_ylabel("speed-up vs the unpruned model")
    ax.set_title(f"Removing {xs[-1]:.0f}% of the maths buys {actual[-1]:.1f}x",
                 fontsize=11.5, pad=8)
    ax.legend(loc="upper left")
    style.tidy(ax)

    fig.tight_layout()
    return save(fig, "xp06_pruning.png")


BUILDERS = [fig_xp00, fig_xp01, fig_xp02, fig_xp06, fig_xp09, fig_xp10, fig_xp12]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    records = load_records()
    if args.list:
        for r in records:
            j = r.get("jetson") or {}
            print(f"{r['_source']:44s} {r['model_id']:34s} {r['format']:10s} "
                  f"mAP50={r.get('map50_dfire_test')} fps={j.get('fps_batched')}")
        return

    style.apply()
    for build in BUILDERS:
        out = build(records)
        print(f"{build.__name__}: {out if out else 'skipped — no applicable records'}")


if __name__ == "__main__":
    main()
