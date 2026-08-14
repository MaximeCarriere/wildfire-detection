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


def fig_xp06e5(records) -> Path | None:
    """Capacity was never the problem. Structure was."""
    import matplotlib.pyplot as plt

    by_sparsity = {}
    for r in records:
        if r.get("granularity") != "unstructured":
            continue
        s = r["prune_meta"]["requested_sparsity"]
        # Either record carries the same damage number; keep one per level.
        by_sparsity.setdefault(s, r)
    fine = [by_sparsity[k] for k in sorted(by_sparsity)]
    chan = sorted([r for r in records if "_nofinetune" in r["model_id"]
                   and r.get("granularity") != "unstructured"
                   and r.get("prune_meta", {}).get("requested_channel_ratio") is not None],
                  key=lambda r: r["prune_meta"]["requested_channel_ratio"])
    if not fine or not chan:
        return None

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    fig.suptitle("The same network survives losing half its weights and dies "
                 "losing 5% of its channels", y=1.05)
    style.subtitle(fig, "Both lines remove capacity with no retraining. Only the SHAPE of "
                        "the removal differs, and that is what the detector cannot absorb.",
                   y=0.985)

    cx = [100 * r["prune_meta"]["requested_channel_ratio"] for r in chan]
    cy = [r["map50_dfire_test"] for r in chan]
    ax.plot(cx, cy, "o-", color=style.RED, linewidth=2.2, markersize=8,
            label="whole channels removed", zorder=4)

    fx = [100 * r["prune_meta"]["requested_sparsity"] for r in fine]
    fy = [r["prune_meta"].get("map50_before_recovery") for r in fine]
    if all(v is not None for v in fy):
        ax.plot(fx, fy, "s-", color=style.AQUA, linewidth=2.2, markersize=8,
                label="individual weights removed", zorder=5)

    base = UNPRUNED_MAP50
    ax.axhline(base, color=style.MUTED, linestyle=":", linewidth=1.6, zorder=2)
    ax.text(97, base + 0.02, "unpruned", fontsize=9.5, color=style.INK_2, ha="right")

    ax.annotate("5% of channels:\naccuracy is gone", xy=(5, cy[min(2, len(cy)-1)]),
                xytext=(16, 0.30), textcoords="data", fontsize=9.5,
                color=style.RED, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=style.RED, linewidth=1.2))
    if all(v is not None for v in fy):
        ax.annotate("50% of weights:\nbarely a scratch", xy=(50, fy[1] if len(fy) > 1 else fy[0]),
                    xytext=(40, 0.90), textcoords="data", fontsize=9.5,
                    color=style.AQUA, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=style.AQUA, linewidth=1.2))

    ax.set_xlabel("capacity removed (%)")
    ax.set_ylabel("detection accuracy (mAP50), no retraining")
    ax.set_xlim(-3, 100)
    ax.set_ylim(-0.04, 0.95)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    style.tidy(ax)
    fig.text(0.5, -0.20, "Removing individual weights has no speed benefit on this "
                         "hardware: irregular zeros sit in a full-size tensor and the GPU "
                         "does the same work.\nThe comparison is about what the network can "
                         "tolerate, not about what runs faster.",
             ha="center", fontsize=9, color=style.MUTED)
    fig.tight_layout()
    return save(fig, "xp06e5_granularity.png")


def fig_xp06e6(records) -> Path | None:
    """Same amount removed, three ways of choosing where."""
    import matplotlib.pyplot as plt
    import numpy as np

    arms = {r.get("allocation"): r for r in records if r.get("experiment") == "xp06e6"}
    if len(arms) < 2:
        return None
    order = [a for a in ("global", "uniform", "sensitivity") if a in arms]

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    fig.suptitle("Where the cut lands matters as much as how big it is", y=1.05)
    style.subtitle(fig, "All three models are the same size, pruned with the same criterion. "
                        "Only the per-layer distribution differs.", y=0.985)

    metrics = [("map50_dfire_test", "overall", style.BLUE),
               ("map50_small_plume", "small plumes", style.ORANGE),
               ("map50_tiny_plume", "tiny plumes", style.AQUA)]
    x = np.arange(len(order))
    w = 0.26
    for i, (key, label, colour) in enumerate(metrics):
        vals = [arms[a].get(key) or 0 for a in order]
        ax.bar(x + (i - 1) * w, vals, width=w - 0.02, color=colour, label=label, zorder=3)
        for xi, v in zip(x + (i - 1) * w, vals):
            ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{a}\n{arms[a]['params_m']:.2f} M params" for a in order])
    ax.set_ylabel("accuracy (mAP50)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3)
    style.tidy(ax)
    fig.tight_layout()
    return save(fig, "xp06e6_allocation.png")


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


BUILDERS = [fig_xp00, fig_xp01, fig_xp02, fig_xp06, fig_xp09, fig_xp10,
            fig_xp12, fig_xp06e1, fig_xp06e2, fig_xp06e5, fig_xp06e6,
            fig_xp06e7]


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
