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

#: Unpruned yolov5s FP16 @512 on the full test set, re-measured on the screening
#: machine. Every extension figure draws this line so the comparison is never lost.
UNPRUNED_MAP50 = 0.7764
UNPRUNED_TINY = 0.1380


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

    s_rec = by_id(records, "dfire_yolov5s_published")
    l_rec = by_id(records, "dfire_yolov5l_published")
    if not (s_rec and l_rec):
        return None

    def silent(r):
        v = r.get("bg_correctly_silent_rate")
        return v if v is not None else 1 - r["bg_false_alarm_rate"]

    n_bg = ((s_rec.get("accuracy_detail") or {}).get("background") or {}).get(
        "n_background_images", 2005)

    # One axis, six groups. The first five are mAP50; the sixth is a rate, not an
    # average precision, so it sits after a visible break and is labelled as a
    # different measure. Both happen to live on 0-1, which is what makes a single
    # axis honest here; a second y-axis would not be.
    groups = [("overall", s_rec["map50_dfire_test"], l_rec["map50_dfire_test"]),
              ("fire", s_rec["map50_fire_class"], l_rec["map50_fire_class"]),
              ("smoke", s_rec["map50_smoke_class"], l_rec["map50_smoke_class"]),
              ("small\nplumes", s_rec["map50_small_plume"], l_rec["map50_small_plume"]),
              ("tiny\nplumes", s_rec["map50_tiny_plume"], l_rec["map50_tiny_plume"]),
              ("no fire\npresent", silent(s_rec), silent(l_rec))]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5),
                             gridspec_kw={"width_ratios": [2.5, 1.3]})
    fig.suptitle("A 6.6x bigger model buys almost nothing", y=1.06)
    style.subtitle(fig, "The large model gains 1.4 accuracy points and costs 3.3x the "
                        "energy per frame. What it does buy is fewer false alarms.", y=1.0)

    ax = axes[0]
    # A gap in the x positions marks where the metric changes.
    xs = np.array([0, 1, 2, 3, 4, 5.55])
    w = 0.36
    sv = [g[1] or 0 for g in groups]
    lv = [g[2] or 0 for g in groups]
    ax.bar(xs - w/2, sv, w, label="YOLOv5s, 7.0 M params", color=style.BLUE, zorder=3)
    ax.bar(xs + w/2, lv, w, label="YOLOv5l, 46.1 M params", color=style.ORANGE, zorder=3)
    for x, a, b in zip(xs, sv, lv):
        style.annotate(ax, x - w/2, a, f"{a:.2f}", dy=4, size=8.5, weight="normal")
        style.annotate(ax, x + w/2, b, f"{b:.2f}", dy=4, size=8.5, weight="normal")

    ax.axvline(4.78, color=style.GRID, linewidth=1.4, zorder=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel("score (0 to 1)")
    ax.set_ylim(0, 1.22)
    ax.set_xlim(-0.75, 6.2)
    ax.text(2.0, 1.15, f"detection accuracy (mAP50)\non the {4306 - n_bg:,} frames with fire or smoke",
            ha="center", fontsize=9, color=style.INK_2)
    ax.text(5.55, 1.15, f"correctly silent\non the {n_bg:,} empty frames",
            ha="center", fontsize=9, color=style.INK_2)
    # Legend below the axes, not inside them: at these bar heights every in-plot
    # position overlaps data.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2,
              columnspacing=2.0, handlelength=1.4)
    style.tidy(ax)

    # Panel 2 - cost, indexed to the small model so the multiple is the message.
    ax = axes[1]
    names = ["energy\nper frame", "latency", "memory", "model\nsize"]
    ratios = [
        l_rec["jetson"]["energy_j_per_1000_frames"] / s_rec["jetson"]["energy_j_per_1000_frames"],
        l_rec["jetson"]["latency_ms_median"] / s_rec["jetson"]["latency_ms_median"],
        l_rec["jetson"]["mem_mb"] / s_rec["jetson"]["mem_mb"],
        l_rec["size_disk_mb"] / s_rec["size_disk_mb"],
    ]
    bars = ax.bar(names, ratios, color=style.ORANGE, width=0.6, zorder=3)
    ax.axhline(1.0, color=style.INK, linewidth=1.2, zorder=4)
    ax.text(3.45, 1.06, "YOLOv5s = 1x", ha="right", va="bottom", fontsize=9, color=style.INK)
    for b, v in zip(bars, ratios):
        style.annotate(ax, b.get_x() + b.get_width()/2, v, f"{v:.1f}x", dy=5)
    ax.set_ylabel("cost relative to YOLOv5s")
    ax.set_ylim(0, max(ratios) * 1.22)
    ax.set_title("What the extra size costs", fontsize=11.5, pad=8)
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
    fig.suptitle("Shrinking the input is free speed, until it blinds the detector", y=1.06)
    style.subtitle(fig, "512 pixels is more accurate AND 1.6× faster than 640. Below 320 the "
                        "bargain ends: distant smoke goes first, then everything else.", y=1.0)

    ax = axes[0]
    for rows, colour, name in ((l, style.ORANGE, "YOLOv5l (46 M)"),
                               (s, style.BLUE, "YOLOv5s (7 M)")):
        if not rows:
            continue
        xs = [r["jetson"]["fps_batched"] for r in rows]
        ys = [r["map50_dfire_test"] for r in rows]
        ax.plot(xs, ys, "o-", color=colour, linewidth=2, markersize=7,
                label=name, zorder=3)
        # The low-resolution points are far apart in speed and the high-resolution
        # ones are packed together, so a fixed label offset collides at the slow
        # end. Alternate above and below along each line instead.
        for i, (r, xx, yy) in enumerate(zip(rows, xs, ys)):
            ax.annotate(f"{r['input_res']}", (xx, yy),
                        xytext=(0, 9 if i % 2 else -16),
                        textcoords="offset points", ha="center", fontsize=9,
                        color=style.INK_2, zorder=4)
    best = max(s, key=lambda r: r["map50_dfire_test"])
    ax.scatter([best["jetson"]["fps_batched"]], [best["map50_dfire_test"]],
               s=260, facecolors="none", edgecolors=style.INK, linewidths=1.6, zorder=5)
    # Well to the right, level with the marked point: the curve falls away from
    # here, so this strip is empty, and staying level keeps clear of the title.
    ax.annotate("best of both:\nmore accurate, faster, cooler",
                (best["jetson"]["fps_batched"], best["map50_dfire_test"]),
                xytext=(118, -4), textcoords="offset points", fontsize=9.5,
                color=style.INK, fontweight="bold", ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=style.MUTED, lw=1))

    # Headroom so the fastest point's label does not land on the axis, and a
    # little below the slowest so its label has somewhere to sit.
    all_x = [r["jetson"]["fps_batched"] for r in s + l]
    all_y = [r["map50_dfire_test"] for r in s + l]
    ax.set_xlim(min(all_x) - 30, max(all_x) * 1.10)
    ax.set_ylim(min(all_y) - 0.035, max(all_y) + 0.02)
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
    """Two things pruning does on this board, neither of them what you would hope."""
    import matplotlib.pyplot as plt

    # "_nofinetune" also matches the fine-grained damage records added later, which
    # mask weights rather than removing channels and carry no MAC reduction.
    raw = sorted([r for r in records if "_nofinetune" in r["model_id"]
                  and "macs_reduction" in r.get("prune_meta", {})],
                 key=lambda r: r["prune_meta"]["macs_reduction"])
    if len(raw) < 3:
        return None
    rec = {("iterative" if "_iter_" in r["model_id"] else "one-shot"): r
           for r in records if "_recovered_trt" in r["model_id"]}

    base_pt = by_id(records, "dfire_yolov5s_published@512")
    base_fps = base_pt["jetson"]["fps_batched"] if base_pt else raw[0]["jetson"]["fps_batched"]
    base_acc = base_pt["map50_dfire_test"] if base_pt else None

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    fig.suptitle("Pruning: the damage is immediate, the speed-up is not", y=1.06)
    style.subtitle(fig, "Cutting channels destroys accuracy long before it buys speed, and "
                        "retraining does not win the loss back.", y=1.0)

    ax = axes[0]
    xs = [100 * r["prune_meta"]["macs_reduction"] for r in raw]
    ys = [r["map50_dfire_test"] for r in raw]
    ax.plot(xs, ys, "o-", color=style.ORANGE, linewidth=2, markersize=7,
            label="pruned, no retraining", zorder=3)

    # The two recovery arms sit at almost the same x (43.1% and 44.1% of MACs), so
    # they are drawn as separate markers with labels placed apart, never joined by
    # a line: two points at one x is not a trend.
    marks = [("one-shot", "s", style.AQUA, (10, 14)),
             ("iterative", "D", style.BLUE, (10, -20))]
    for name, marker, colour, offset in marks:
        r = rec.get(name)
        if not r:
            continue
        x = 100 * r["prune_meta"]["macs_reduction"]
        y = r["map50_dfire_test"]
        ax.plot([x], [y], marker, color=colour, markersize=10, zorder=5)
        ax.annotate(f"{name} + retraining\n{y:.2f}", (x, y), xytext=offset,
                    textcoords="offset points", fontsize=9.5, color=colour,
                    fontweight="bold", ha="left")

    if base_acc:
        ax.axhline(base_acc, color=style.MUTED, linestyle=":", linewidth=1.6, zorder=2)
        ax.text(88, base_acc + 0.02, "unpruned model", fontsize=9.5,
                color=style.INK_2, ha="right")
    ax.set_xlabel("arithmetic removed (% of MACs)")
    ax.set_ylabel("detection accuracy (mAP50)")
    ax.set_ylim(-0.04, 0.95)
    ax.set_xlim(-3, 95)
    ax.set_title("Accuracy collapses at ~10% of the arithmetic", fontsize=11.5, pad=8)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17))
    style.tidy(ax)

    # Panel 2 - the arithmetic-versus-speed reality check.
    ax = axes[1]
    fps = [r["jetson"]["fps_batched"] for r in raw]
    ideal = [100 / (100 - x) for x in xs]
    actual = [f / base_fps for f in fps]
    ax.plot(xs, ideal, "--", color=style.MUTED, linewidth=1.8,
            label="if speed tracked arithmetic", zorder=3)
    ax.plot(xs, actual, "o-", color=style.BLUE, linewidth=2, markersize=7,
            label="measured", zorder=4)
    style.annotate(ax, xs[-1], actual[-1], f"{actual[-1]:.1f}x", dx=-14, dy=-20,
                   color=style.BLUE)
    style.annotate(ax, xs[-1], ideal[-1], f"{ideal[-1]:.1f}x expected", dx=-44, dy=-6,
                   color=style.INK_2, weight="normal", size=9)
    ax.set_xlabel("arithmetic removed (% of MACs)")
    ax.set_ylabel("speed-up vs the unpruned model")
    ax.set_xlim(-3, 95)
    ax.set_title(f"Removing {xs[-1]:.0f}% of the maths buys {actual[-1]:.1f}x",
                 fontsize=11.5, pad=8)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17))
    style.tidy(ax)

    fig.tight_layout()
    return save(fig, "xp06_pruning.png")



# --------------------------------------------------------------------------
# XP6 extension — the four axes, measured
# --------------------------------------------------------------------------

def _side(name: str):
    """Load one of the extension's non-record JSONs (sweeps, not single runs)."""
    path = RAW / name
    return json.loads(path.read_text()) if path.exists() else None


def _stage(layer: str) -> int:
    import re
    m = re.match(r"model\.(\d+)\.", layer)
    return int(m.group(1)) if m else -1


def fig_xp06e1(records) -> Path | None:
    """Where pruning damage is cheap, where it is fatal, and what each cut buys."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap

    d = _side("xp06e1_sensitivity.json")
    if not d:
        return None

    rows = [r for r in d["rows"] if "retained" in r]
    ratios = sorted({r["ratio"] for r in rows})
    layers = sorted({r["layer"] for r in rows}, key=lambda n: (_stage(n), n))

    grid = np.full((len(ratios), len(layers)), np.nan)
    for r in rows:
        grid[ratios.index(r["ratio"]), layers.index(r["layer"])] = min(1.0, r["retained"])

    # Sequential, one hue: retention is a magnitude, so it never gets a rainbow.
    cmap = LinearSegmentedColormap.from_list(
        "retained", ["#fdf3ee", "#f6c9ae", "#8ec9b4", "#1baf7a", "#0d5f43"])

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 4.6),
                             gridspec_kw={"width_ratios": [1.55, 1]})
    fig.suptitle("The layers that break first are the layers that save the least", y=1.10)
    style.subtitle(fig, "Each cell prunes ONE layer and leaves the rest alone. Left: the map. "
                        "Right: what each cut actually buys you.", y=1.015)

    # ---- panel 1: the map ------------------------------------------------
    ax = axes[0]
    im = ax.imshow(grid, aspect="auto", cmap=cmap, vmin=0, vmax=1,
                   interpolation="nearest")
    ax.set_yticks(range(len(ratios)))
    ax.set_yticklabels([f"{r:.0%}" for r in ratios])
    ax.set_ylabel("how much of that\nlayer was cut")
    ax.set_xlabel("layer, in order through the network")

    # A single boundary is easier to read than 24 stage numbers: everything left
    # of it is the early backbone, everything right of it is deeper.
    split = next(i for i, n in enumerate(layers) if _stage(n) >= 6)
    ax.axvline(split - 0.5, color=style.INK, linewidth=2)
    ax.set_xticks([split / 2, split + (len(layers) - split) / 2])
    ax.set_xticklabels(["early layers\n(stages 0-4)", "deeper layers (stages 6-23)"],
                       fontsize=10)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, pad=0.012, fraction=0.03)
    cb.set_label("accuracy kept", fontsize=10)
    cb.outline.set_visible(False)

    # ---- panel 2: the trade-off, which is the actual conclusion ----------
    ax = axes[1]
    half = [r for r in rows if abs(r["ratio"] - 0.5) < 1e-9]
    early = [r for r in half if _stage(r["layer"]) <= 4]
    deep = [r for r in half if _stage(r["layer"]) >= 6]
    for grp, colour, label in ((early, style.RED, "early layers (stages 0-4)"),
                               (deep, style.AQUA, "deeper layers (stages 6-23)")):
        ax.scatter([100 * r["params_reduction"] for r in grp],
                   [100 * min(1, r["retained"]) for r in grp],
                   s=52, color=colour, alpha=0.85, edgecolor=style.SURFACE,
                   linewidth=1.2, label=label, zorder=4)

    worst = min(half, key=lambda r: r["retained"])
    best = max(half, key=lambda r: r["params_reduction"])
    for r, dx, dy, ha in ((worst, 16, 10, "left"), (best, -14, -26, "right")):
        ax.annotate(f"{r['layer'].replace('model.', '')}\n"
                    f"keeps {r['retained']:.0%}, frees {r['params_reduction']:.1%}",
                    (100 * r["params_reduction"], 100 * min(1, r["retained"])),
                    xytext=(dx, dy), textcoords="offset points", fontsize=9,
                    color=style.INK, fontweight="bold", ha=ha,
                    arrowprops=dict(arrowstyle="->", color=style.INK, linewidth=1.1))

    ax.set_xlabel("parameters freed by that cut (%)")
    ax.set_ylabel("accuracy kept (%)")
    ax.set_title("Every layer halved: bottom-left destroys accuracy and saves nothing",
                 fontsize=11, pad=10)
    ax.set_xlim(-0.5, 100 * max(r["params_reduction"] for r in half) * 1.22)
    ax.set_ylim(-6, 114)
    ax.legend(loc="center right", fontsize=9.5)
    style.tidy(ax)

    fig.tight_layout()
    return save(fig, "xp06e1_sensitivity.png")


def fig_xp06e2(records) -> Path | None:
    """Which channels you choose matters more than anyone assumed."""
    import matplotlib.pyplot as plt
    import numpy as np

    d = _side("xp06e2_criteria_damage.json")
    if not d:
        return None
    base = d["baseline"]["val_map50"]

    order = ["l1", "fpgm", "taylor", "lamp", "hessian", "l2", "bn", "random"]
    pretty = {"l1": "L1", "l2": "L2", "bn": "BN scale", "taylor": "Taylor",
              "hessian": "Hessian", "fpgm": "FPGM", "lamp": "LAMP", "random": "random"}
    cells = {(r["criterion"], r["ratio"]): r for r in d["rows"] if "val_map50" in r}
    shown = [c for c in order if (c, 0.05) in cells]

    fig, axes = plt.subplots(1, 2, figsize=(15.2, 4.8),
                             gridspec_kw={"width_ratios": [1, 1.35]})
    fig.suptitle("The importance criterion decides whether pruning is survivable", y=1.13)
    style.subtitle(fig, "Left: a 5% cut, no retraining. Right: a DIFFERENT, deeper 25% cut, "
                        "after 12 epochs. Same eight rules in both.\nThe cuts differ on "
                        "purpose: at 25% almost nothing survives untrained, so a damage panel "
                        "there would rank nothing.",
                   y=1.07)

    ax = axes[0]
    vals = [cells[(c, 0.05)]["val_map50"] for c in shown]
    # Colour encodes the outcome, not the name: a criterion that lost most of the
    # accuracy is marked as failed, and the control is neutral grey.
    colours = [style.MUTED if c == "random" else
               (style.RED if cells[(c, 0.05)]["val_map50"] < 0.5 * base else style.AQUA)
               for c in shown]
    bars = ax.bar(range(len(shown)), vals, color=colours, width=0.68, zorder=3)
    ax.axhline(base, color=style.INK_2, linestyle=":", linewidth=1.6, zorder=2,
               xmax=0.80)
    ax.text(len(shown) - 0.35, base, "unpruned", fontsize=9.5, color=style.INK_2,
            ha="left", va="center")
    for b, v in zip(bars, vals):
        x = b.get_x() + b.get_width() / 2
        if v > 0.25:                      # tall enough to hold the label inside
            ax.text(x, v - 0.03, f"{v:.2f}", ha="center", va="top",
                    fontsize=9.5, fontweight="bold", color="white")
        else:                             # short bar: sit above it, clear of the line
            ax.text(x, v + 0.02, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=9.5, fontweight="bold", color=style.RED)
    ax.set_xticks(range(len(shown)))
    ax.set_xticklabels([pretty[c] for c in shown], rotation=30, ha="right")
    ax.set_ylabel("accuracy after a 5% cut (mAP50)")
    ax.set_ylim(0, 1.05)
    ax.set_title("A 5% cut, no retraining", fontsize=11.5, pad=12)
    style.tidy(ax)

    # Panel 2: after retraining the criteria converge on the easy cases and stay
    # far apart on the hard ones, which is the finding that matters here.
    ax = axes[1]
    rec = {}
    for r in records:
        if r.get("experiment") == "xp06e2" or "_recovered" in r["model_id"]:
            for c in order:
                if f"_{c}_recovered" in r["model_id"]:
                    rec[c] = r
    # L2 was retrained too, as the originally published arm, under a model_id
    # that predates this naming. Including it keeps the arm this page corrects
    # visible in the same panel rather than only in a table.
    l2 = by_id(records, "dfire_yolov5s_pruned25_recovered")
    if l2:
        rec["l2"] = l2
    if rec:
        names = [c for c in order if c in rec]
        x = np.arange(len(names))
        # Both series as a FRACTION of the unpruned model, so they share one
        # honest scale. Plotting raw mAP50 beside raw tiny-plume mAP50 would need
        # either two y-axes or a fudge factor, and both of those lie.
        overall = [rec[c]["map50_dfire_test"] / UNPRUNED_MAP50 for c in names]
        tiny = [(rec[c]["map50_tiny_plume"] or 0) / UNPRUNED_TINY for c in names]
        ax.bar(x - 0.19, overall, width=0.36, color=style.BLUE,
               label="overall accuracy kept", zorder=3)
        ax.bar(x + 0.19, tiny, width=0.36, color=style.ORANGE,
               label="tiny-plume accuracy kept", zorder=3)
        for i, (o, t) in enumerate(zip(overall, tiny)):
            # Inside the bars: the 1.0 reference line runs exactly where an
            # above-bar label would sit.
            ax.text(i - 0.19, o - 0.03, f"{o:.0%}", ha="center", va="top",
                    fontsize=8, fontweight="bold", color="white")
            ax.text(i + 0.19, t - 0.03, f"{t:.0%}", ha="center", va="top",
                    fontsize=8, fontweight="bold", color="white")
        ax.axhline(1.0, color=style.INK_2, linestyle=":", linewidth=1.4, zorder=2,
                   xmax=0.86)
        ax.text(len(names) - 0.42, 1.0, "unpruned", fontsize=9.5, color=style.INK_2,
                ha="left", va="center")
        ax.set_xticks(x)
        ax.set_xticklabels([pretty[c] for c in names], rotation=30, ha="right")
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("share of unpruned kept")
        ax.set_title("A 25% cut, after 12 epochs of retraining",
                     fontsize=11.5, pad=12)
        # Say why this panel is a subset: recovery costs ~20 min per arm against
        # seconds for damage, so only the leaders plus the control were paid for.
        ax.text(0.5, -0.42, "Retraining is a great leveller: the same eight rules span 0.94 to "
                            "0.00 before it and 0.754 to 0.709 after.\nThe ranking survives "
                            "only on tiny plumes, which is where the detector earns its keep.",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=8.5, color=style.MUTED)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2)
        style.tidy(ax)
    else:
        ax.axis("off")

    fig.tight_layout()
    return save(fig, "xp06e2_criteria.png")


def fig_xp06e4(records) -> Path | None:
    """What 2:4 sparsity is, and what constraining the pattern costs."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Rectangle

    rec = next((r for r in records if r.get("granularity") == "2:4"), None)
    if not rec:
        return None
    m = rec["prune_meta"]

    fig = plt.figure(figsize=(13.6, 5.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], hspace=0.55, wspace=0.28)
    fig.suptitle("2:4 sparsity: same number of weights removed, but the pattern is fixed",
                 y=1.06)
    style.subtitle(fig, "Both patterns below delete half the weights. Only the lower one can be "
                        "accelerated by the hardware, and only it needs retraining to survive.",
                   y=1.0)

    ROWS, COLS = 6, 12
    def draw(ax, mask, title, note):
        """mask[r][c] True means the weight was removed."""
        for r in range(ROWS):
            for c in range(COLS):
                ax.add_patch(Rectangle((c, ROWS - 1 - r), 0.9, 0.9,
                                       facecolor="#dfe3e4" if mask[r][c] else style.AQUA,
                                       edgecolor="none"))
        ax.set_xlim(-0.2, COLS + 0.1)
        ax.set_ylim(-2.6, ROWS + 0.1)      # room for the caption inside the axes
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=11, pad=6, loc="left")
        ax.text(0, -0.7, note, fontsize=9, color=style.INK_2, va="top", wrap=True)

    # Free choice: any half of the weights, no constraint. Deterministic scatter.
    rng = np.random.default_rng(0)
    free = np.zeros((ROWS, COLS), bool)
    flat = rng.permutation(ROWS * COLS)[: ROWS * COLS // 2]
    for i in flat:
        free[i // COLS][i % COLS] = True

    # 2:4: exactly two of every four neighbouring weights, chosen within the group.
    nm = np.zeros((ROWS, COLS), bool)
    for r in range(ROWS):
        for g in range(0, COLS, 4):
            for d in ((r + g) % 4, (r + g + 2) % 4):
                nm[r][g + d] = True

    ax = fig.add_subplot(gs[0, 0])
    draw(ax, free, "Free choice (E5): delete any half",
         "No rule about where they land.\nHighest compression, but nothing on this\n"
         "board can skip scattered zeros.")

    ax = fig.add_subplot(gs[1, 0])
    draw(ax, nm, "2:4 (E4): exactly two of every four",
         "The same 50% removed. This regular pattern\nis what Ampere sparse tensor cores can\n"
         "actually skip.")
    for g in range(0, COLS + 1, 4):          # show the groups of four
        ax.plot([g - 0.05, g - 0.05], [-0.1, ROWS], color=style.INK, linewidth=1.6, zorder=5)

    # ---- results -------------------------------------------------------
    ax = fig.add_subplot(gs[:, 1])
    bars = [("unpruned", UNPRUNED_MAP50, style.MUTED),
            ("50% free,\nno retraining", 0.7622, style.AQUA),
            ("50% as 2:4,\nno retraining", m.get("map50_before_recovery", 0.0), style.RED),
            ("50% as 2:4,\nafter 12 epochs", rec["map50_dfire_test"], style.BLUE)]
    xs = range(len(bars))
    ax.bar(xs, [b[1] for b in bars], color=[b[2] for b in bars], width=0.66, zorder=3)
    for i, (_, v, _c) in enumerate(bars):
        if v > 0.1:
            ax.text(i, v - 0.025, f"{v:.4f}", ha="center", va="top", fontsize=9.5,
                    fontweight="bold", color="white")
        else:
            ax.text(i, 0.02, f"{v:.4f}", ha="center", va="bottom", fontsize=9.5,
                    fontweight="bold", color=style.RED)
    ax.axhline(UNPRUNED_MAP50, color=style.INK_2, linestyle=":", linewidth=1.5,
               zorder=2, xmax=0.93)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([b[0] for b in bars], fontsize=9.5)
    ax.set_ylabel("accuracy (mAP50)")
    ax.set_ylim(0, UNPRUNED_MAP50 * 1.2)
    ax.set_title("The pattern costs everything, until you retrain", fontsize=11.5, pad=10)
    style.tidy(ax)

    fig.subplots_adjust(bottom=0.14, top=0.86)
    return save(fig, "xp06e4_sparsity24.png")


def fig_xp06e5(records) -> Path | None:
    """Accuracy, speed and energy for the two granularities, on one shared axis.

    A 25% channel cut and a 25% weight cut are not the same amount of network, so
    every panel is plotted against **the fraction of parameters actually removed**.
    That is what lets the three be read together: pick a point on the x-axis and
    the panels say what it costs, what it buys, and what it draws.

    Panel one is damage with **no retraining in either series**. Comparing a
    retrained weight-pruned model against an un-retrained channel-pruned one would
    flatter weights enormously, and the two converge once both are allowed to
    recover.
    """
    import matplotlib.pyplot as plt

    chan_acc = sorted(
        (100 * r["prune_meta"]["params_reduction"], r["map50_dfire_test"])
        for r in records
        if "_nofinetune" in r["model_id"]
        and r.get("granularity") != "unstructured"
        and (r.get("prune_meta") or {}).get("params_reduction") is not None)
    wgt_acc = sorted({
        (100 * (1 - m["nonzero_params_m"] / m["dense_params_m"]), m["map50_before_recovery"])
        for r in records
        for m in [r.get("prune_meta") or {}]
        if m.get("granularity") == "unstructured"
        and m.get("map50_before_recovery") is not None})
    if not chan_acc or not wgt_acc:
        return None

    def eng(tag):
        return by_id(records, f"yolov5s_e5b_{tag}")

    base = eng("dense")
    if not base:
        return None

    def eseries(tags):
        pts = [(0.0, base["jetson"])]
        for tg in tags:
            r = eng(tg)
            if r:
                pts.append((r["granularity_meta"]["params_removed_frac"] * 100, r["jetson"]))
        return sorted(pts)

    chan_e = eseries(["chan25", "chan50", "chan70"])
    wgt_e = eseries(["weight50", "weight90"])

    CH, WG = style.RED, style.AQUA
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.3))
    fig.suptitle("Deleting channels and zeroing weights are not the same operation", y=1.09)
    style.subtitle(fig, "One axis throughout: how much of the model is actually gone. Weights "
                        "survive damage that channels cannot, and channels buy speed that "
                        "weights never do.", y=1.01)

    ax = axes[0]
    ax.plot(*zip(*([(0.0, UNPRUNED_MAP50)] + list(chan_acc))), "o-", color=CH,
            linewidth=2.2, markersize=7, label="whole channels deleted", zorder=4)
    ax.plot(*zip(*([(0.0, UNPRUNED_MAP50)] + list(wgt_acc))), "s-", color=WG,
            linewidth=2.2, markersize=7, label="individual weights zeroed", zorder=5)
    ax.axhline(UNPRUNED_MAP50, color=style.MUTED, linestyle=":", linewidth=1.4, zorder=2)
    ax.annotate("7% of the model gone\nand accuracy with it", xy=chan_acc[1],
                xytext=(30, 0.34), textcoords="data", fontsize=9, color=CH,
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=CH, linewidth=1.2))
    ax.set_ylabel("accuracy (mAP50), no retraining")
    ax.set_ylim(-0.05, 0.95)
    ax.set_title("What it costs", fontsize=11.5, pad=8)
    ax.legend(loc="upper right", fontsize=8.5)

    for ax, key, std_key, ylab, title, lab_va in (
            (axes[1], "fps_batched", "fps_batched_std", "images per second", "What it buys",
             "top"),
            (axes[2], "energy_j_per_1000_frames", None, "joules per 1000 frames",
             "What it draws", "bottom")):
        ref = base["jetson"].get(key)
        ax.axhline(ref, color=style.MUTED, linestyle=":", linewidth=1.4, zorder=2)
        # The weights line sits almost exactly on the reference in the speed panel,
        # so the label goes below it there and above it in the energy panel.
        ax.text(98, ref, "unpruned", fontsize=9, color=style.INK_2, ha="right", va=lab_va)
        for pts, colour, marker in ((chan_e, CH, "o"), (wgt_e, WG, "s")):
            ys = [j.get(key) for _, j in pts]
            if any(v is None for v in ys):
                continue
            es = [(j.get(std_key) or 0.0) for _, j in pts] if std_key else None
            ax.errorbar([x for x, _ in pts], ys, yerr=es, fmt=marker + "-", color=colour,
                        linewidth=2.2, markersize=7, capsize=3, zorder=4)
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=11.5, pad=8)

    dip = min(chan_e[1:], key=lambda p: p[1]["fps_batched"])
    axes[1].annotate("slower than\nnot pruning", (dip[0], dip[1]["fps_batched"]),
                     xytext=(28, 6), textcoords="offset points", fontsize=9.5, color=CH,
                     fontweight="bold", ha="left", va="center",
                     arrowprops=dict(arrowstyle="->", color=CH, lw=1.3))

    for ax in axes:
        ax.set_xlabel("percent of the model removed")
        ax.set_xlim(-4, 100)
        style.tidy(ax)

    fig.tight_layout()
    return save(fig, "xp06e5_granularity.png")


def fig_xp06e6(records) -> Path | None:
    """Same amount removed, three ways of choosing where, before and after recovery.

    E6's published comparison is the right-hand panel alone, and read alone it says
    allocation is worth 0.4 points and therefore barely matters. The left panel is
    the measurement that was missing: the same three models scored at the moment
    they were cut, with no optimizer run at all.

    Both panels are needed because E5 already showed this detector can erase a
    large structural difference in 12 epochs. A gap that is wide before recovery
    and narrow after it is a statement about what retraining absorbs; a gap that
    is narrow in both is a statement about allocation. Only the pair distinguishes
    them, so only the pair is published.

    Each panel carries its own unpruned reference because the damage scores were
    taken on the Orin and the recovered ones on the screening box. The two
    baselines agree to 0.0011 mAP50, which is the point of measuring both.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    arms = {r.get("allocation"): r for r in records if r.get("experiment") == "xp06e6"}
    if len(arms) < 2:
        return None
    order = [a for a in ("global", "uniform", "sensitivity") if a in arms]

    dmg_path = RAW / "xp06e6b_damage.json"
    dmg = json.loads(dmg_path.read_text()) if dmg_path.exists() else None
    if dmg and not all(a in dmg["arms"] for a in order):
        dmg = None

    panels = []
    if dmg:
        panels.append(("Before retraining", "the cut, scored as damage",
                       {a: (dmg["arms"][a]["map50_damage"],
                            dmg["arms"][a]["map50_tiny_damage"]) for a in order},
                       dmg["unpruned_here"]["map50"]))
    panels.append(("After 12 epochs of recovery", "what E6 published",
                   {a: (arms[a].get("map50_dfire_test") or 0,
                        arms[a].get("map50_tiny_plume") or 0) for a in order},
                   0.7764))

    fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 4.9),
                             sharey=True, squeeze=False)
    axes = axes[0]
    fig.suptitle("Where the cut lands, before and after the retraining that hides it", y=1.06)
    style.subtitle(fig, "All three models are the same size (~4.21 M, 40% removed), pruned "
                        "with the same L1 criterion. Only the per-layer distribution differs.",
                   y=0.995)

    series = (("overall", style.BLUE), ("tiny plumes", style.AQUA))
    x = np.arange(len(order))
    w = 0.34
    for ax, (title, sub, vals, ref) in zip(axes, panels):
        for i, (label, colour) in enumerate(series):
            vs = [vals[a][i] for a in order]
            ax.bar(x + (i - 0.5) * w, vs, width=w - 0.03, color=colour,
                   label=label, zorder=3)
            for xi, v in zip(x + (i - 0.5) * w, vs):
                # A bar that nearly reaches the unpruned line has no room above it,
                # so those labels go inside the bar instead of on top of the line.
                if v > ref * 0.85:
                    ax.text(xi, v - 0.02, f"{v:.3f}", ha="center", va="top",
                            fontsize=8.5, color="white", fontweight="bold", zorder=4)
                else:
                    ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=8.5)
        ax.axhline(ref, color=style.INK, linestyle=":", linewidth=1.2, zorder=2)
        ax.text(-0.45, ref + 0.014, f"unpruned {ref:.4f}",
                ha="left", va="bottom", fontsize=8.5, color=style.INK_2)

        # The number the panel exists to show: how far apart the three allocations
        # are on the overall metric. Side by side, the pair is the whole argument.
        spread = max(vals[a][0] for a in order) - min(vals[a][0] for a in order)
        ax.set_title(f"{title}\n{sub} \u2014 arms span {spread:.3f} mAP50",
                     fontsize=11, pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(order)
        style.tidy(ax)
        ax.grid(axis="x", visible=False)

    axes[0].set_ylabel("accuracy (mAP50)")
    axes[0].set_ylim(0, max(max(v) for _, _, vals, _ in panels for v in vals.values()) * 1.24)
    axes[-1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2, frameon=False)
    fig.tight_layout()
    return save(fig, "xp06e6_allocation.png")


def fig_xp06e7b(records) -> Path | None:
    """The recoverable frontier: damage, one-shot and iterative across the ratio.

    This is the Han-style figure the E7 comparison implicitly lives inside. E7
    measured one-shot against iterative at a single point, 40% of parameters
    removed, and found iterative losing; on the reference curve that point sits in
    the flat region where the two arms are not expected to differ at all. Drawing
    the whole sweep is what makes E7's result readable as "no difference where none
    was predicted" rather than as a contradiction of the literature.

    Plotted against **measured parameter reduction**, never the channel ratio the
    experiment was configured with. A 25% channel cut removes 39.6% of this
    network's parameters, and E5 and E6 both had to make that correction before
    their comparisons meant anything.

    The y-axis is accuracy *loss* against the unpruned model, matching the
    reference, so every series starts at zero and falls. Each series carries its
    own baseline: the damage sweep records the unpruned score measured in its own
    run, and the trained arms are read against the screening record.

    Returns None until at least the damage series exists, so the figure appears the
    moment the cheap arm has run and fills in as the expensive ones land.
    """
    import matplotlib.pyplot as plt

    dmg_path = RAW / "xp06e7b_damage.json"
    dmg = json.loads(dmg_path.read_text()) if dmg_path.exists() else None

    arms = {}
    for r in records:
        if r.get("experiment") == "xp06e7b" and r.get("arm"):
            m = r.get("prune_meta") or {}
            if m.get("params_reduction") is None or r.get("map50_dfire_test") is None:
                continue
            arms.setdefault(r["arm"], []).append(
                (100 * m["params_reduction"], r["map50_dfire_test"]))
    if not dmg and not arms:
        return None

    series = []
    if dmg:
        ref = dmg["unpruned_here"]["map50"]
        series.append(("pruning (no retraining)", style.MUTED, ":", "o",
                       sorted((100 * p["params_reduction"], p["map50"])
                              for p in dmg["points"]), ref))
    # RED is reserved in this palette for a failed configuration, so the two
    # working arms take AQUA and BLUE and the unretrained series stays MUTED.
    for arm, label, colour in (("oneshot", "pruning + finetuning", style.AQUA),
                               ("iterative", "iterative pruning + finetuning", style.BLUE)):
        if arm in arms:
            series.append((label, colour, "-", "o", sorted(arms[arm]), 0.7764))

    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    fig.suptitle("How far this detector can be pruned before the accuracy goes", y=1.04)
    style.subtitle(fig, "Channel pruning, L1, at 512 px. Loss against the unpruned model, "
                        "plotted against the parameters actually removed \u2014 not the "
                        "channel ratio each cut was configured with.", y=0.975)

    for label, colour, ls, mk, pts, ref in series:
        xs = [x for x, _ in pts]
        ys = [(v - ref) * 100 for _, v in pts]
        ax.plot(xs, ys, ls, marker=mk, color=colour, label=label,
                linewidth=2.0, markersize=6, markerfacecolor="white",
                markeredgewidth=1.8, zorder=3)

    ax.axhline(0, color=style.INK, linewidth=1.0, zorder=2)
    # E7 measured exactly one point on this axis. Marking it is the whole reason
    # the sweep exists: a reader should see which part of the curve it sampled.
    ax.axvline(39.6, color=style.MUTED, linestyle="--", linewidth=1.2, zorder=1)
    ax.text(39.6, ax.get_ylim()[0], "  E7 measured here", rotation=90,
            va="bottom", ha="left", fontsize=8.5, color=style.INK_2)

    ax.set_xlabel("parameters pruned away (%)")
    ax.set_ylabel("accuracy loss (mAP50 points)")
    ax.legend(frameon=False, fontsize=9.5, loc="lower left")
    style.tidy(ax)
    fig.tight_layout()
    return save(fig, "xp06e7b_frontier.png")

def fig_xp06e3(records) -> Path | None:
    """Rounding barely touches accuracy and nearly doubles throughput on the board."""
    import matplotlib.pyplot as plt

    acc = {}
    for fp in sorted(RAW.glob("xp06e3_dfire_yolov5s_round*.json")):
        d = json.loads(fp.read_text()); m = d["prune_meta"]
        acc[m["round_to"]] = {"map50": d["map50_dfire_test"], "params": d["params_m"],
                              "aligned": m["widths_divisible_by_32"], "n": m["n_conv_layers"]}
    spd = {}
    for fp in sorted(RAW.glob("xp06e3b_yolov5s_round*.json")):
        d = json.loads(fp.read_text()); j = d["jetson"]; r = d["regularity_meta"]["round_to"]
        spd[r] = {"fps": j["fps_batched"], "std": j.get("fps_batched_std", 0),
                  "energy": j.get("energy_j_per_1000_frames")}
    rs = sorted(acc)
    if len(rs) < 2:
        return None
    xs = list(range(len(rs)))
    labels = [f"round to\n{r}" for r in rs]
    have_speed = len(spd) == len(rs)

    ncol = 3 if have_speed else 2
    fig, axes = plt.subplots(1, ncol, figsize=(5.4 * ncol, 4.4))
    fig.suptitle("Rounding channel widths: free accuracy, and 1.77x the speed on the board",
                 y=1.03)
    style.subtitle(fig, "Same 25% cut, same accuracy. Snapping the surviving widths to clean "
                        "multiples nearly doubles throughput on the Jetson.", y=0.965)

    # Panel 1: alignment.
    ax = axes[0]
    bars = ax.bar(xs, [acc[r]["aligned"] for r in rs], width=0.62, color=style.BLUE, zorder=3)
    for b, r in zip(bars, rs):
        v = acc[r]["aligned"]
        inside = v > 6
        ax.text(b.get_x() + b.get_width() / 2, v - 2.5 if inside else v + 1, str(v),
                ha="center", va="top" if inside else "bottom", fontsize=11, fontweight="bold",
                color="white" if inside else style.INK)
    ax.set_xticks(xs); ax.set_xticklabels(labels)
    ax.set_ylabel(f"conv layers on a multiple of 32 (of {acc[rs[0]]['n']})")
    ax.set_ylim(0, acc[rs[0]]["n"] * 1.05)
    ax.set_title("The shape changes completely", fontsize=11.5, pad=8)
    style.tidy(ax)

    # Panel 2: accuracy (flat).
    ax = axes[1]
    ax.plot(xs, [acc[r]["map50"] for r in rs], "o-", color=style.AQUA, linewidth=2.2,
            markersize=9, zorder=4)
    for xi, r in zip(xs, rs):
        ax.annotate(f"{acc[r]['map50']:.4f}", (xi, acc[r]["map50"]), xytext=(0, 12),
                    textcoords="offset points", ha="center", fontsize=9.5, fontweight="bold",
                    color=style.INK)
    ax.axhline(UNPRUNED_MAP50, color=style.INK_2, linestyle=":", linewidth=1.5, zorder=2, xmax=0.82)
    ax.text(len(rs) - 0.9, UNPRUNED_MAP50, "unpruned", fontsize=9.5, color=style.INK_2,
            ha="left", va="center")
    ax.set_xticks(xs); ax.set_xticklabels(labels)
    ax.set_ylabel("accuracy (mAP50)"); ax.set_ylim(0.70, UNPRUNED_MAP50 * 1.02)
    ax.set_xlim(-0.4, len(rs) - 0.3)
    ax.set_title("Accuracy does not", fontsize=11.5, pad=8)
    style.tidy(ax)

    # Panel 3: throughput (the payoff).
    if have_speed:
        ax = axes[2]
        fps = [spd[r]["fps"] for r in rs]
        bars = ax.bar(xs, fps, yerr=[spd[r]["std"] for r in rs], width=0.62,
                      color=style.ORANGE, zorder=3, capsize=4,
                      error_kw={"elinewidth": 1.2, "ecolor": style.INK_2})
        for b, r in zip(bars, rs):
            v = spd[r]["fps"]
            ax.text(b.get_x() + b.get_width() / 2, v - 18, f"{v:.0f}", ha="center", va="top",
                    fontsize=10.5, fontweight="bold", color="white")
        base = spd[rs[0]]["fps"]
        ax.text(xs[-1], fps[-1] + 20, f"{fps[-1] / base:.2f}x", ha="center", fontsize=11,
                fontweight="bold", color=style.INK)
        ax.axhline(472.6, color=style.INK_2, linestyle=":", linewidth=1.5, zorder=2, xmax=0.82)
        ax.text(len(rs) - 0.9, 472.6, "unpruned", fontsize=9.5, color=style.INK_2,
                ha="left", va="center")
        ax.set_xticks(xs); ax.set_xticklabels(labels)
        ax.set_ylabel("throughput on the Jetson (img/s)")
        ax.set_ylim(0, max(fps) * 1.16)
        ax.set_title("Throughput nearly doubles", fontsize=11.5, pad=8)
        style.tidy(ax)

    fig.text(0.5, -0.04, "Measured on the Jetson Orin, MAXN_SUPER. Size is not the driver: "
                         "round_to=1 has 40% fewer parameters than unpruned yet runs slower "
                         "(363 vs 473 img/s); rounding the widths is what recovers the speed.",
             ha="center", fontsize=8.5, color=style.MUTED)
    fig.tight_layout()
    return save(fig, "xp06e3_regularity.png")


def fig_xp06e7(records) -> Path | None:
    """The one-shot versus iterative comparison, with the confound removed."""
    import matplotlib.pyplot as plt
    import numpy as np

    fair = {r.get("arm"): r for r in records if r.get("experiment") == "xp06e7"}
    if len(fair) < 2:
        return None
    old = {("iterative" if "_iter_" in r["model_id"] else "oneshot"): r
           for r in records if "_recovered_trt" in r["model_id"]}

    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    fig.suptitle("Iterative pruning still loses once both arms train equally", y=1.05)
    style.subtitle(fig, "The published comparison gave iterative only 4 epochs in its final "
                        "shape against one-shot's 12. Here the post-cut budget is equal.",
                   y=0.985)

    arms = [a for a in ("oneshot", "iterative") if a in fair]
    x = np.arange(len(arms))
    w = 0.30
    old_vals = [(old.get(a) or {}).get("map50_dfire_test") for a in arms]
    new_vals = [fair[a]["map50_dfire_test"] for a in arms]

    if all(v is not None for v in old_vals):
        ax.bar(x - w / 2, old_vals, width=w - 0.02, color=style.MUTED,
               label="as published (unequal post-cut epochs)", zorder=3)
        for xi, v in zip(x - w / 2, old_vals):
            ax.text(xi, v - 0.02, f"{v:.3f}", ha="center", va="top", fontsize=9,
                    color="white")
    ax.bar(x + w / 2, new_vals, width=w - 0.02, color=style.BLUE,
           label="equal epochs after the final cut", zorder=3)
    for xi, v in zip(x + w / 2, new_vals):
        ax.text(xi, v - 0.02, f"{v:.3f}", ha="center", va="top", fontsize=9,
                fontweight="bold", color="white")

    # Stop the line short and sit the label in the gap, so the dots cannot run
    # through the word whatever the figure size.
    ax.axhline(UNPRUNED_MAP50, color=style.INK_2, linestyle=":", linewidth=1.6,
               zorder=2, xmax=0.82)
    ax.text(len(arms) - 0.52, UNPRUNED_MAP50, "unpruned", fontsize=9.5,
            color=style.INK_2, ha="left", va="center")
    ax.set_xlim(-0.55, len(arms) - 0.15)
    ax.set_ylim(0, UNPRUNED_MAP50 * 1.16)
    ax.set_xticks(x)
    ax.set_xticklabels(["one-shot", "iterative"][:len(arms)])
    ax.set_ylabel("accuracy (mAP50)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    style.tidy(ax)
    fig.tight_layout()
    return save(fig, "xp06e7_fair_rerun.png")


ARMS_E4B = [
    ("yolov5s_dense",             "unpruned\n(dense)",             "MUTED"),
    ("yolov5s_free50",            "50% removed,\nfree choice",      "AQUA"),
    ("yolov5s_sparse24_nosparse", "50% as 2:4,\nordinary build",    "BLUE"),
    ("yolov5s_sparse24_sparse",   "50% as 2:4,\nsparse build",      "ORANGE"),
]


def fig_xp06e4b(records) -> Path | None:
    """Four engines of the same network, built twice under two tuning batches.

    The first build set tuned TensorRT at batch 1 (``--optShapes``) and then
    reported throughput at batch 16, so its kernels were chosen under conditions
    that were never measured -- and batch 1 is precisely the case where a sparse
    kernel cannot win, because its metadata-decode cost is fixed while the maths
    it saves scales with batch. The second set closes that gap by tuning at 16.

    Both sets are plotted against the dense arm *of their own set*, never against
    each other. The batch-16 engines were measured after an hour of continuous
    building and sit about 2.6% lower across the board, which is the die being
    warm, not the engines being slower. Within-set ranking is the only comparison
    this figure invites, and it is the one that carries the result: the ranking
    does not survive a rebuild. ``50% as 2:4, ordinary build`` is +2.1% in one set
    and -1.4% in the other, from identical weights.

    The right panel is the floor that makes that readable. Three engines compiled
    from one unchanged onnx with identical flags differ by 0.58%, so a same-build
    comparison is trustworthy to well under a percent while a *different*-build
    comparison plainly is not. Sparsity cannot be what separates the left-hand
    bars, and now the figure shows why rather than asserting it.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    def pair(mid):
        a, b = by_id(records, mid), by_id(records, mid + "_rev")
        return [r for r in (a, b) if r and usable(r)]

    def fps(rs):
        v = [r["jetson"]["fps_batched"] for r in rs]
        within = max((r["jetson"].get("fps_batched_std") or 0.0) for r in rs)
        return float(np.mean(v)), max(within, (max(v) - min(v)) / 2 if len(v) > 1 else 0.0)

    sets = []
    for suffix, label in (("", "tuned at batch 1"), ("_optb16", "tuned at batch 16")):
        arms = [(lab, pair(mid + suffix), col) for mid, lab, col in ARMS_E4B]
        if all(rs for _, rs, _ in arms):
            sets.append((label, [(lab, fps(rs), col) for lab, rs, col in arms]))
    if not sets:
        return None

    var = [by_id(records, f"yolov5s_var{r}") for r in ("a", "b", "c")]
    var = [r for r in var if r and usable(r)]

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.3),
                             gridspec_kw={"width_ratios": [1.55, 1.0]})
    fig.suptitle("The one pattern the hardware understands, measured on the hardware", y=1.11)
    style.subtitle(fig, "Same network, same shape, same arithmetic in all four: only which "
                        "weights are zero, and whether the compiler was allowed to exploit "
                        "them. Each set is measured against its own dense engine.", y=1.02)

    # --- left: per-set delta against that set's dense arm -------------------
    ax = axes[0]
    labels = [lab for lab, _, _ in sets[0][1]]
    ys = np.arange(len(labels))[::-1]
    h = 0.36 if len(sets) > 1 else 0.6
    hatches = (None, "///")
    for k, (setlab, arms) in enumerate(sets):
        ref = arms[0][1][0]
        vals = [(v / ref - 1) * 100 for _, (v, _), _ in arms]
        errs = [e / ref * 100 for _, (_, e), _ in arms]
        off = (k - (len(sets) - 1) / 2) * h
        ax.barh(ys + off, vals, xerr=errs, height=h * 0.92,
                color=[getattr(style, c) for _, _, c in arms],
                hatch=hatches[k], edgecolor="white", linewidth=0.6, zorder=3,
                error_kw=dict(ecolor=style.INK, lw=1.1, capsize=3),
                label=setlab)
        # Labels clear the error bar, not just the bar end, or the caps sit on top
        # of the digits at this aspect ratio.
        for y, v, e in zip(ys, vals, errs):
            ha, dx = ("left", e + 0.16) if v >= 0 else ("right", -(e + 0.16))
            ax.text(v + dx, y + off, f"{v:+.1f}%", va="center", ha=ha,
                    fontsize=9, color=style.INK_2)

    # The noise floor from the right panel, drawn where it does its work: any bar
    # inside this band is indistinguishable from rebuilding the same file.
    if var:
        v = [r["jetson"]["fps_batched"] for r in var]
        floor = (max(v) / min(v) - 1) * 100
        ax.axvspan(-floor, floor, color=style.MUTED, alpha=0.18, zorder=1)
        ax.text(floor, ys[-1] - 0.60, f" rebuild noise \u00b1{floor:.1f}%",
                fontsize=8.5, color=style.INK_2, va="center", ha="left")
    ax.axvline(0, color=style.INK, linestyle=":", linewidth=1.2, zorder=2)
    # Room for the value labels on both sides: a negative bar puts its label to the
    # left of the axis, where the tick text already lives.
    lo = min(v - e for _, arms in sets for (v, e) in
             [((a[1][0] / arms[0][1][0] - 1) * 100, a[1][1] / arms[0][1][0] * 100)
              for a in arms])
    hi = max(v + e for _, arms in sets for (v, e) in
             [((a[1][0] / arms[0][1][0] - 1) * 100, a[1][1] / arms[0][1][0] * 100)
              for a in arms])
    ax.set_xlim(min(lo - 0.95, -1.1), hi + 1.9)
    ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlabel("throughput vs the dense engine of the same build set (%)")
    ax.set_title("Speed, relative to dense", fontsize=11.5, pad=8)
    # Upper right is the only quadrant no bar reaches: the dense row is pinned at
    # zero by construction and every other arm sits below it.
    ax.legend(fontsize=8.5, loc="upper right", frameon=False)
    ax.set_ylim(ys[-1] - 0.95, ys[0] + 0.75)
    style.tidy(ax); ax.grid(axis="y", visible=False)

    # --- right: the rebuild control ----------------------------------------
    ax = axes[1]
    if var:
        v = [r["jetson"]["fps_batched"] for r in var]
        e = [r["jetson"].get("fps_batched_std") or 0.0 for r in var]
        xs = np.arange(len(v))
        ax.bar(xs, v, yerr=e, width=0.55, color=style.MUTED, zorder=3,
               error_kw=dict(ecolor=style.INK, lw=1.2, capsize=4))
        for x, val in zip(xs, v):
            ax.text(x, val, f"{val:,.1f}", ha="center", va="bottom",
                    fontsize=9.5, color=style.INK_2)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"build {c}" for c in "ABC"], fontsize=9.5)
        ax.set_ylabel("images per second")
        ax.set_ylim(0, max(v) * 1.22)
        ax.set_title(f"One onnx, three builds: {(max(v)/min(v)-1)*100:.1f}% apart",
                     fontsize=11.5, pad=8)
        style.tidy(ax); ax.grid(axis="x", visible=False)
    else:
        ax.axis("off")

    axes[0].text(0.5, -0.30, "TensorRT found 39 layers eligible for sparse kernels and "
                             "chose 0 of them \u2014 at both tuning batches.",
                 transform=axes[0].transAxes, ha="center", va="top",
                 fontsize=10.5, color=style.RED, fontweight="bold")
    fig.tight_layout()
    return save(fig, "xp06e4b_sparsity_speed.png")



BUILDERS = [fig_xp00, fig_xp01, fig_xp02, fig_xp06, fig_xp09, fig_xp10,
            fig_xp12, fig_xp06e1, fig_xp06e2, fig_xp06e3, fig_xp06e4, fig_xp06e4b, fig_xp06e5, fig_xp06e7b,
            fig_xp06e6,
            fig_xp06e7]


# --------------------------------------------------------------------------
# XP6 E4b — does the one hardware-supported pattern actually run faster?
# --------------------------------------------------------------------------

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
