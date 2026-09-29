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

    from matplotlib.patches import Rectangle, FancyArrowPatch
    fig = plt.figure(figsize=(17.6, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.52, 1.55, 1], wspace=0.40)
    sch = fig.add_subplot(gs[0, 0])
    axes = [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])]
    fig.suptitle("The layers that break first are the layers that save the least", y=1.10)
    style.subtitle(fig, "Each cell prunes ONE layer and leaves the rest alone. Left: the map. "
                        "Right: what each cut actually buys you.", y=1.015)

    # --- panel 1: how the map is built ------------------------------------
    # One layer is pruned while the other 56 are left whole, the model is scored,
    # and that one number becomes one cell of the heatmap. Repeat for every layer
    # at every depth. The schematic shows a single trial and names the sweep.
    sch.set_xlim(0, 10); sch.set_ylim(-1.0, 9.4); sch.axis("off")
    sch.set_title("1. How the map\nis built", fontsize=10.5, pad=6, loc="left")
    cut_idx = 1
    for k in range(5):
        y = 7.4 - k * 1.45
        cut = k == cut_idx
        sch.add_patch(Rectangle((0.4, y), 3.2, 1.15, facecolor="#e6eaec",
                                edgecolor="none", zorder=2))
        if cut:
            sch.add_patch(Rectangle((0.4, y), 3.2 * 0.5, 1.15, facecolor=style.RED,
                                    edgecolor="none", zorder=3))
            sch.text(3.8, y + 0.55, "cut", ha="left", va="center", fontsize=8,
                     color=style.RED, fontweight="bold")
        else:
            sch.add_patch(Rectangle((0.4, y), 3.2, 1.15, facecolor=style.AQUA,
                                    edgecolor="none", zorder=3, alpha=0.85))
    sch.text(2.0, 7.4 + 1.5, "the network", ha="center", fontsize=8.5,
             fontweight="bold", color=style.INK)
    sch.text(2.0, -0.9, "prune ONE layer,\nleave the other 56", ha="center", va="top",
             fontsize=7.8, color=style.INK_2)
    # arrow to a single heatmap cell
    sch.add_patch(FancyArrowPatch((6.0, 5.6), (7.4, 5.6), arrowstyle="-|>",
                                  mutation_scale=12, color=style.INK_2, linewidth=1.3))
    sch.text(6.7, 6.15, "score", ha="center", fontsize=7.8, color=style.INK_2)
    sch.add_patch(Rectangle((7.8, 4.9), 1.4, 1.4, facecolor="#8ec9b4", edgecolor="white",
                            linewidth=1.5, zorder=3))
    sch.text(8.5, 3.9, "one cell\nof the map", ha="center", va="top", fontsize=7.8,
             color=style.INK_2)
    sch.text(5.0, 1.4, "repeat: 57 layers x 5 depths", ha="center", fontsize=8,
             color=style.INK, fontweight="bold")

    # ---- panel 2 (data): the map -----------------------------------------
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
    xmax = 100 * max(r["params_reduction"] for r in half) * 1.22
    # Place both callouts in the empty middle band, clear of the point clouds
    # (deep layers sit high, early layers hug the left edge).
    for r, tx, ty, ha in ((worst, 0.26 * xmax, 26, "left"),
                          (best, 0.60 * xmax, 60, "center")):
        ax.annotate(f"{r['layer'].replace('model.', '')}\n"
                    f"keeps {r['retained']:.0%}, frees {r['params_reduction']:.1%}",
                    xy=(100 * r["params_reduction"], 100 * min(1, r["retained"])),
                    xytext=(tx, ty), textcoords="data", fontsize=8.8,
                    color=style.INK, fontweight="bold", ha=ha, va="center",
                    arrowprops=dict(arrowstyle="->", color=style.INK, linewidth=1.1))

    ax.set_xlabel("parameters freed by that cut (%)")
    ax.set_ylabel("accuracy kept (%)")
    ax.set_title("Every layer halved:\nbottom-left destroys accuracy, saves nothing",
                 fontsize=11, pad=8)
    ax.set_xlim(-0.5, xmax)
    ax.set_ylim(-6, 114)
    ax.legend(loc="lower right", fontsize=9)
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

    from matplotlib.patches import Rectangle
    fig = plt.figure(figsize=(19.0, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.62, 1, 1.35], wspace=0.24)
    sch = fig.add_subplot(gs[0, 0])
    axes = [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])]
    fig.suptitle("The importance criterion decides whether pruning is survivable", y=1.13)
    style.subtitle(fig, "Left: a 5% cut, no retraining. Right: a DIFFERENT, deeper 25% cut, "
                        "after 12 epochs. Same eight rules in both.\nThe cuts differ on "
                        "purpose: at 25% almost nothing survives untrained, so a damage panel "
                        "there would rank nothing.",
                   y=1.07)

    # --- panel 1: what a criterion does, and why the rule matters ---------
    # Every rule scores each channel and deletes the lowest. Two rules given the
    # SAME six channels score them differently, so they delete different ones:
    # L2 squares the weights (one big weight rescues a channel), L1 does not.
    # Illustrative scores chosen so the two disagree on which two die.
    sch.set_xlim(-2.3, 6.3); sch.set_ylim(-2.0, 10.4); sch.axis("off")
    sch.set_title("1. A rule gives every channel\na score, then cuts the lowest",
                  fontsize=10.5, pad=6, loc="left")
    chans = list("ABCDEF")
    scores = {"L2": [0.90, 0.50, 0.85, 0.30, 0.62, 0.20],
              "L1": [0.90, 0.50, 0.42, 0.58, 0.62, 0.20]}
    for row, rule in enumerate(("L2", "L1")):
        yb = 5.2 - row * 4.5
        vals = scores[rule]
        cut = set(sorted(range(6), key=lambda i: vals[i])[:2])
        sch.text(-2.1, yb + 1.2, rule, ha="left", va="center", fontsize=10,
                 fontweight="bold", color=style.INK)
        for i, v in enumerate(vals):
            col = style.RED if i in cut else style.AQUA
            sch.add_patch(Rectangle((i - 0.32, yb), 0.64, 2.6 * v, facecolor=col,
                                    edgecolor="none", zorder=3))
            if i in cut:
                sch.text(i, yb - 0.12, "cut", ha="center", va="top", fontsize=6.8,
                         color=style.RED, fontweight="bold")
    # channel labels once, under the lower row
    for i in range(6):
        sch.text(i, -1.15, chans[i], ha="center", fontsize=7.5, color=style.INK_2)
    sch.text(2.5, -1.85, "one bar = one channel (A-F);  green kept, red cut",
             ha="center", fontsize=7, color=style.INK_2)
    # single callout: a bar's height is the score
    sch.annotate("bar height = that rule's\nimportance score for the channel",
                 xy=(0, 5.2 + 2.6 * 0.9), xytext=(0.9, 9.8), textcoords="data",
                 fontsize=6.9, color=style.INK_2, ha="left", va="top",
                 arrowprops=dict(arrowstyle="->", color=style.INK_2, linewidth=1.0))
    # the punchline, in the clear band between the two rows
    sch.text(2.3, 3.55, "same 6 channels, different scores:\nL2 cuts D, L1 cuts C",
             ha="center", va="center", fontsize=7, color=style.INK, style="italic")

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
    import numpy as np
    from matplotlib.patches import Rectangle
    fig = plt.figure(figsize=(17.6, 4.4))
    gs = fig.add_gridspec(1, 4, width_ratios=[0.62, 1, 1, 1], wspace=0.32)
    sch = fig.add_subplot(gs[0, 0])
    axes = [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[0, 3])]
    fig.suptitle("Deleting channels and zeroing weights are not the same operation", y=1.09)
    style.subtitle(fig, "One axis throughout: how much of the model is actually gone. Weights "
                        "survive damage that channels cannot, and channels buy speed that "
                        "weights never do.", y=1.01)

    # --- panel 1: the two operations, on the same weight grid -------------
    sch.set_xlim(0, 8); sch.set_ylim(0, 15.4); sch.axis("off")
    sch.set_title("1. Two ways to\ndelete half a layer", fontsize=10.5, pad=6, loc="left")
    R, C, cell = 6, 6, 0.78

    def grid(y0, dead, colour, title, note):
        for r in range(R):
            for c in range(C):
                on = (r, c) not in dead
                sch.add_patch(Rectangle((1 + c * cell, y0 + (R - 1 - r) * cell),
                                        cell * 0.86, cell * 0.86,
                                        facecolor=colour if on else "#dfe3e4",
                                        edgecolor="none"))
        sch.text(1, y0 + R * cell + 0.15, title, fontsize=9, fontweight="bold",
                 color=colour, va="bottom")
        sch.text(1, y0 - 0.55, note, fontsize=7.8, color=style.INK_2, va="top")

    # channels: whole columns removed (a slice). weights: scattered, same count.
    chan_dead = {(r, c) for r in range(R) for c in (1, 4)}
    grid(9.4, chan_dead, CH, "channels: whole slices",
         "the layer genuinely\nnarrows, 7.03 M to 4.2 M")
    rng = np.random.default_rng(1)
    flat = rng.permutation(R * C)[: R * C // 2]
    wgt_dead = {(i // C, i % C) for i in flat}
    grid(1.8, wgt_dead, WG, "weights: scattered holes",
         "same count gone, but the\ngrid keeps its full shape")

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

    from matplotlib.patches import Rectangle
    fig = plt.figure(figsize=(4.6 + 6.2 * len(panels), 4.9))
    gs = fig.add_gridspec(1, len(panels) + 1, width_ratios=[0.7] + [1] * len(panels),
                          wspace=0.16)
    sch = fig.add_subplot(gs[0, 0])
    axes = [fig.add_subplot(gs[0, i + 1], sharey=(None)) for i in range(len(panels))]
    if len(axes) > 1:
        axes[1].sharey(axes[0])
    fig.suptitle("Where the cut lands, before and after the retraining that hides it", y=1.06)
    style.subtitle(fig, "All three models are the same size (~4.21 M, 40% removed), pruned "
                        "with the same L1 criterion. Only the per-layer distribution differs.",
                   y=0.995)

    # --- panel 1: the three ways to spread one fixed total cut ------------
    # Six layers, early ones narrow and fragile (E1), deep ones wide. Each row is
    # a strategy: the bar height is how much of that layer it removes. All three
    # remove the same TOTAL; they differ only in where. Illustrative shape, the
    # point the two data panels then score.
    sch.set_xlim(-2.4, 6.3); sch.set_ylim(-0.6, 10.4); sch.axis("off")
    sch.set_title("1. Three ways to spread\nthe same total cut", fontsize=10.5, pad=6,
                  loc="left")
    profiles = [("uniform", style.INK_2, [.40, .40, .40, .40, .40, .40]),
                ("global", style.RED, [.72, .60, .42, .34, .30, .30]),
                ("sensitivity", style.BLUE, [.05, .12, .38, .52, .60, .66])]
    for row, (name, colour, prof) in enumerate(profiles):
        yb = 7.2 - row * 3.4
        sch.axvspan(-0.4, 1.4, ymin=(yb) / 11.0, ymax=(yb + 2.0) / 11.0,
                    color=style.RED, alpha=0.07, zorder=0)
        for x, cut in enumerate(prof):
            sch.add_patch(Rectangle((x - 0.34, yb), 0.68, 1.8, facecolor="#e6eaec",
                                    edgecolor="none", zorder=2))
            sch.add_patch(Rectangle((x - 0.34, yb), 0.68, 1.8 * cut, facecolor=colour,
                                    edgecolor="none", zorder=3))
        sch.text(-0.7, yb + 0.9, name, ha="right", va="center", fontsize=8.8,
                 fontweight="bold", color=colour)
    sch.text(0.5, 7.2 + 2.15, "fragile\n(E1)", ha="center", fontsize=7,
             color=style.RED, va="bottom")
    sch.text(2.75, -0.5, "early layers  ->  deep layers", ha="center", fontsize=7.3,
             color=style.INK_2)

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
    """The recoverable frontier: what the two schedules are, and where they diverge.

    Three panels. First the method, drawn as two training timelines so a reader
    can see what "one-shot" and "iterative" physically differ by, with the equal
    post-cut budget made visible. Then the Han-style frontier, accuracy loss
    against measured parameter reduction. Then the gap between the arms, which
    crosses zero: the single number E7 could not see from its one point.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, FancyArrowPatch

    dmg_path = RAW / "xp06e7b_damage.json"
    dmg = json.loads(dmg_path.read_text()) if dmg_path.exists() else None

    arms = {}
    for r in records:
        if r.get("experiment") == "xp06e7b" and r.get("arm"):
            m = r.get("prune_meta") or {}
            if m.get("params_reduction") is None or r.get("map50_dfire_test") is None:
                continue
            arms.setdefault(r["arm"], []).append(
                (100 * m["params_reduction"], r["map50_dfire_test"],
                 r.get("channel_ratio")))
    if not dmg and not arms:
        return None
    have_gap = "oneshot" in arms and "iterative" in arms

    fig = plt.figure(figsize=(16.5, 4.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.02, 1.05, 0.95], wspace=0.28)
    ax0, ax1 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2]) if have_gap else None
    fig.suptitle("One-shot versus iterative pruning, across the whole ratio", y=1.05)
    style.subtitle(fig, "Same criterion, same final size, same 12 epochs of final training. "
                        "The only difference is whether the channels come out all at once or "
                        "gradually.", y=0.965)

    # --- panel 1: the two schedules as timelines --------------------------
    ax0.set_xlim(0, 21); ax0.set_ylim(0, 10); ax0.axis("off")
    ax0.set_title("1. What the two schedules do", fontsize=11, pad=8, loc="left")
    th = 1.3   # block height

    def cut(x, y, label=None):
        ax0.add_patch(Rectangle((x - 0.12, y - 0.25), 0.24, th + 0.5,
                                facecolor=style.RED, edgecolor="none", zorder=4))
        if label:
            ax0.text(x, y + th + 0.55, label, ha="center", fontsize=8,
                     color=style.RED, fontweight="bold")

    def train(x, w, y, colour, label=None):
        ax0.add_patch(Rectangle((x, y), w, th, facecolor=colour, edgecolor="white",
                                linewidth=1.2, zorder=3))
        if label:
            ax0.text(x + w / 2, y + th / 2, label, ha="center", va="center",
                     fontsize=8, color="white", fontweight="bold")

    # one-shot lane (top): one cut, then 12 epochs. Aligned so its 12-epoch block
    # ends at x=20, the same place iterative's does.
    yo = 6.4
    ax0.text(0, yo + th + 1.0, "one-shot", fontsize=10, fontweight="bold", color=style.AQUA)
    cut(8, yo, "cut 40%\nat once")
    train(8.2, 11.8, yo, style.AQUA, "12 epochs")

    # iterative lane (bottom): 4 small cuts, 2 epochs between the first three, then
    # the same 12-epoch block, ending at the same x.
    yi = 2.2
    ax0.text(0, yi + th + 1.0, "iterative", fontsize=10, fontweight="bold", color=style.BLUE)
    x = 1.0
    for k in range(3):
        cut(x, yi, "cut" if k == 0 else None)
        train(x + 0.2, 1.7, yi, style.BLUE, "2")
        x += 2.1
    cut(x, yi, None)
    train(x + 0.2, 11.8, yi, style.BLUE, "12 epochs")

    # bracket showing the equal post-cut budget under both final blocks
    ax0.add_patch(FancyArrowPatch((8.4, yo - 0.5), (8.4, yi + th + 0.35),
                                  arrowstyle="-", color=style.INK_2, linewidth=1.0,
                                  linestyle=(0, (3, 3)), zorder=2))
    ax0.text(14.2, 0.7, "same 12 epochs after the final cut", ha="center",
             fontsize=8.5, color=style.INK_2, style="italic")
    ax0.text(4.4, 0.7, "iterative pays extra\nbetween-step training", ha="center",
             fontsize=8, color=style.BLUE)

    # --- panel 2: the frontier --------------------------------------------
    series = []
    if dmg:
        ref = dmg["unpruned_here"]["map50"]
        series.append(("pruned, no retraining", style.MUTED, ":", "o",
                       sorted((100 * p["params_reduction"], p["map50"])
                              for p in dmg["points"])))
    for arm, label, colour in (("oneshot", "one-shot + retraining", style.AQUA),
                               ("iterative", "iterative + retraining", style.BLUE)):
        if arm in arms:
            series.append((label, colour, "-", "o",
                           sorted((x, v) for x, v, _ in arms[arm])))

    for label, colour, ls, mk, pts in series:
        xs = [x for x, _ in pts]; ys = [(v - 0.7764) * 100 for _, v in pts]
        ax1.plot(xs, ys, ls, marker=mk, color=colour, label=label, linewidth=2.0,
                 markersize=6, markerfacecolor="white", markeredgewidth=1.8, zorder=3)
    ax1.axhline(0, color=style.INK, linewidth=1.0, zorder=2)
    # Mark E7's single point without crossing the data: a short tick at the top of
    # the axis and a horizontal label in the empty band below the trained curves.
    ax1.axvline(39.6, color=style.MUTED, linestyle="--", linewidth=1.2, zorder=1)
    ax1.annotate("E7 measured only here", xy=(39.6, -2), xytext=(44, -13),
                 textcoords="data", fontsize=8.5, color=style.INK_2, ha="left",
                 va="center", arrowprops=dict(arrowstyle="->", color=style.INK_2,
                                               linewidth=1.0))
    ax1.set_xlabel("parameters pruned away (%)")
    ax1.set_ylabel("accuracy loss (mAP50 points)")
    ax1.set_title("2. How far it can be pruned", fontsize=11, pad=8, loc="left")
    # Legend in the empty rectangle on the right, clear of both the trained curves
    # (top) and the plunging no-retraining curve (left).
    ax1.legend(frameon=False, fontsize=8.5, loc="center right",
               bbox_to_anchor=(1.0, 0.42))
    style.tidy(ax1)

    # --- panel 3: the gap that crosses zero -------------------------------
    if have_gap:
        om = {rr: (x, v) for x, v, rr in arms["oneshot"]}
        im = {rr: (x, v) for x, v, rr in arms["iterative"]}
        pts = sorted((om[k][0], (im[k][1] - om[k][1]) * 100) for k in om if k in im)
        xs = [x for x, _ in pts]; gaps = [g for _, g in pts]
        ax2.axhline(0, color=style.INK, linewidth=1.0, zorder=2)
        # Shade where iterative wins for good: after the last point it trails at.
        # The low-ratio wobble around zero is noise, not a win for either arm.
        last_neg = max((x for x, g in pts if g < 0), default=xs[0])
        nxt = next((x for x, _ in pts if x > last_neg), last_neg)
        cross = (last_neg + nxt) / 2
        ax2.axvspan(cross, 100, color=style.BLUE, alpha=0.09, zorder=0)
        ax2.plot(xs, gaps, "-o", color=style.BLUE, linewidth=2.0, markersize=6,
                 markerfacecolor="white", markeredgewidth=1.8, zorder=3)
        top = max(gaps)
        ax2.set_ylim(min(gaps) - 1.2, top + 2.4)
        ax2.text((cross + 100) / 2, top + 1.1, "iterative wins", ha="center", fontsize=9.5,
                 color=style.BLUE, fontweight="bold")
        ax2.text(cross - 2, top + 1.1, "tie / one-shot", ha="right", fontsize=9.5,
                 color=style.INK_2, fontweight="bold")
        ax2.set_xlabel("parameters pruned away (%)")
        ax2.set_ylabel("iterative advantage (mAP50 points)")
        ax2.set_title("3. Where iterative starts to win", fontsize=11, pad=8, loc="left")
        style.tidy(ax2)

    fig.tight_layout()
    return save(fig, "xp06e7b_frontier.png")

def fig_xp06e9(records) -> Path | None:
    """Whether NetAdapt's latency lookup table is a usable cost model on this board.

    A schematic and four data panels, in the order the question actually has to be
    asked. Each data panel could have ended the experiment, and the figure is
    arranged so a reader can see which one nearly did.

    Leftmost: the method, hung on the real network -- named layers lifted out of
    the fused engine and timed alone, a sweep filling one row of the table, and
    the two totals posed as the question that panel 4 then resolves.
    Second: is layer latency informative in channel count? Only if the curve bends.
    Third: how repeatable is one entry, which sets the finest width grid a search
    could resolve. Fourth: do per-layer times sum to the real engine, where they
    plainly do not. Right: the one that decides it -- does the table predict the
    *direction* of a real cut's effect, scored against engines E3 already built and
    measured on this board.

    The fourth and fifth panels disagree deliberately. NetAdapt never consumes
    absolute latency, only differences, so a per-layer overhead that does not vary
    with width cancels out and the 2.16x total error is survivable. Showing the
    failed total next to the correct ranking is the whole argument of the section.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    path = RAW / "xp06e9_netadapt_lut.json"
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    shape, noise, slope = d.get("shape"), d.get("noise"), d.get("slope")
    if not (shape and slope):
        return None

    # One tall schematic column, then the four data panels in a 2x2 block, the
    # same shape fig_xp06e4 uses. A single row of five would be ~5.6:1 and turn
    # unreadable at README width; this keeps the figure near 1.8:1.
    fig = plt.figure(figsize=(15.5, 8.5))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 0.95, 0.95], hspace=0.42, wspace=0.30)
    axes = [fig.add_subplot(gs[:, 0]),
            fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]),
            fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])]
    fig.suptitle("Can the pruning ratio be searched automatically against measured latency?",
                 y=1.045)
    style.subtitle(fig, "NetAdapt ranks candidate cuts with a table of layer latency versus "
                        "channel count. Before writing the search, the table has to be shown "
                        "to describe this hardware. It half does.", y=1.005)

    # --- 1. method schematic ------------------------------------------------
    # The data panels judge a cost model, but a reader cannot judge whether the
    # model is valid without first knowing how its table was built: each conv
    # layer is lifted out of the fused engine, compiled as its own tiny TensorRT
    # engine, and timed alone. The panel shows that on the real network -- named
    # layers with measured times, not generic boxes -- because the contrast that
    # matters (fused layers share input/output handling, isolated ones each pay
    # their own) is what lets panel 4 fail while panel 5 survives. Everything
    # numeric is read from the record; ``unpruned_table`` is newer than some
    # copies of it, so its absence degrades the rows rather than raising.
    from matplotlib.patches import Rectangle

    ax = axes[0]
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("1. How the table is built\nlayers timed alone vs the fused engine",
                 fontsize=10.5, pad=8)
    table = slope.get("unpruned_table") or []
    rows = table[:3]
    n_more = (slope.get("n_convs") or len(table)) - len(rows)
    lut_ms, real_ms = slope["unpruned_lut_ms"], slope["unpruned_real_ms"]
    arrow = dict(arrowstyle="->", color=style.INK_2, linewidth=1.1)
    fx, fw, ix, iw, rh = 4.55, 1.4, 7.45, 1.3, 0.95   # fused col, isolated col, row pitch

    ax.text(fx + fw / 2, 15.15, "deployed:\none fused engine", fontsize=8, color=style.INK,
            va="bottom", ha="center", fontweight="bold")
    ax.text(8.5, 15.15, "for the table:\neach conv alone", fontsize=8, color=style.INK,
            va="bottom", ha="center", fontweight="bold")
    for i in range(3):
        e = rows[i] if i < len(rows) else None
        y0 = 14.55 - rh * (i + 1)                     # bottom of this row
        rc = y0 + rh / 2
        ax.add_patch(Rectangle((fx, y0), fw, rh, facecolor=style.BLUE,
                               edgecolor="white", linewidth=1.2))
        ax.annotate("", xy=(ix - 0.05, rc), xytext=(fx + fw + 0.15, rc), arrowprops=arrow)
        ax.add_patch(Rectangle((ix, y0 + 0.08), iw, 0.16, facecolor=style.MUTED,
                               edgecolor="none"))
        ax.add_patch(Rectangle((ix, y0 + 0.24), iw, 0.47, facecolor=style.BLUE,
                               edgecolor="none"))
        ax.add_patch(Rectangle((ix, y0 + 0.71), iw, 0.16, facecolor=style.MUTED,
                               edgecolor="none"))
        if e:
            ax.text(0.15, rc + 0.04, e["layer"], fontsize=8, color=style.INK, va="bottom")
            ax.text(0.15, rc - 0.08,
                    f"{e['cin']}\u2192{e['cout']}  k{e['k']}  {e['hw']}\u00d7{e['hw']}",
                    fontsize=7.5, color=style.INK_2, va="top")
            ax.text(8.95, rc, f"{e['ms']:.1f}", fontsize=8.5, color=style.INK, va="center")
    ax.add_patch(Rectangle((fx, 14.55 - 3 * rh), fw, 3 * rh, facecolor="none",
                           edgecolor=style.INK, linewidth=1.3))
    ax.text(fx + fw / 2, 11.5, "\u22ee", fontsize=10, color=style.INK_2, va="top", ha="center")
    ax.text(ix + iw / 2, 11.5, "\u22ee", fontsize=10, color=style.INK_2, va="top", ha="center")
    if rows:
        ax.text(0.15, 11.45, f"+ {n_more} more convs", fontsize=7.5, color=style.INK_2,
                va="top")
        ax.text(8.95, 11.45, "ms", fontsize=7.5, color=style.INK_2, va="top")
    ax.text(fx + fw / 2, 10.35, f"one engine\n{real_ms:.1f} ms", fontsize=8.5,
            color=style.INK, va="top", ha="center", fontweight="bold")
    ax.text(8.5, 10.35, f"summed\n{lut_ms:.1f} ms", fontsize=8.5,
            color=style.INK, va="top", ha="center", fontweight="bold")
    ax.text(0.15, 8.85, "fused, neighbours share input/output\n"
                       "handling; alone, each conv pays its own\n"
                       "(grey), so the right column costs more",
            fontsize=7.5, color=style.INK_2, va="top")

    # One row of the table, shown with its real entries: NetAdapt consumes
    # latency as a function of width, and the sweep is what fills that row.
    pts = shape["points"]
    picks = [pts[0], pts[len(pts) // 2], pts[-1]]
    ax.text(0.15, 7.05, "sweeping a layer's width fills its table row", fontsize=8.5,
            color=style.INK, va="top", fontweight="bold")
    ax.text(0.15, 6.5,
            f"a {shape['cin']}-in {shape['k']}\u00d7{shape['k']} conv at "
            f"{shape['hw']}\u00d7{shape['hw']}, three of its entries:",
            fontsize=7.5, color=style.INK_2, va="top")
    ax.add_patch(Rectangle((0.5, 4.25), 8.6, 1.5, facecolor="white",
                           edgecolor=style.INK_2, linewidth=1.0))
    ax.plot([2.9, 2.9], [4.25, 5.75], color=style.INK_2, linewidth=1.0)
    ax.plot([0.5, 9.1], [5.0, 5.0], color=style.INK_2, linewidth=1.0)
    ax.text(0.7, 5.37, "out ch", fontsize=8, color=style.INK_2, va="center")
    ax.text(0.7, 4.62, "ms", fontsize=8, color=style.INK_2, va="center")
    for cx, pp in zip((4.0, 5.9, 7.8), picks):
        ax.text(cx, 5.37, f"{pp['out_ch']}", fontsize=8.5, color=style.INK,
                va="center", ha="center", fontweight="bold")
        ax.text(cx, 4.62, f"{pp['ms']:.2f}", fontsize=8.5, color=style.INK,
                va="center", ha="center")
    ax.text(0.5, 3.9, "panel 2 plots this row in full", fontsize=7.5,
            color=style.INK_2, va="top")

    ax.text(0.15, 2.95, f"the question: can a table whose rows sum to\n"
                       f"{lut_ms:.1f} ms describe an engine that runs in\n"
                       f"{real_ms:.1f} ms? panels 2\u20135 take it apart",
            fontsize=8.5, color=style.INK, va="top")
    ax.set_xlim(0, 9.7)
    ax.set_ylim(0.9, 16.2)

    # --- 2. shape -----------------------------------------------------------
    ax = axes[1]
    # Two changes from the obvious plot, both because the obvious plot is read
    # backwards. The x-axis counts channels *removed*, so it runs in the direction
    # a reader thinks of pruning; and the y-axis is a percentage of the unpruned
    # layer's own time, so the 100% line means "no faster than not pruning at all"
    # and every point above it is a cut that cost time instead of saving it. In raw
    # milliseconds those points look unremarkable.
    full = max(p["out_ch"] for p in shape["points"])
    base = next(p["ms"] for p in shape["points"] if p["out_ch"] == full)
    pts = sorted(shape["points"], key=lambda p: -p["out_ch"])
    xs = [(1 - p["out_ch"] / full) * 100 for p in pts]
    ys = [p["ms"] / base * 100 for p in pts]
    chans = [p["out_ch"] for p in pts]

    ax.axhspan(100, max(ys) * 1.08, color=style.RED, alpha=0.07, zorder=1)
    ax.axhline(100, color=style.RED, linewidth=1.4, zorder=2)
    ax.text(2, max(ys) * 1.05, "slower than not pruning at all", fontsize=8.5,
            color=style.RED, ha="left", va="top")
    ax.plot([0, 100], [100, 0], ":", color=style.MUTED, linewidth=1.5, zorder=2,
            label="if time fell with the channels")
    ax.plot(xs, ys, "-", color=style.BLUE, linewidth=1.6, zorder=3, alpha=0.75)
    for c, x, y in zip(chans, xs, ys):
        aligned = c % 32 == 0
        ax.plot([x], [y], "o", color=style.ORANGE if aligned else style.BLUE,
                markersize=9 if aligned else 5,
                markerfacecolor="white" if aligned else style.BLUE,
                markeredgewidth=2 if aligned else 0, zorder=4)
    ax.set_xlabel("channels removed (%)")
    ax.set_ylabel("layer time, % of unpruned")
    ax.set_title("2. Cutting channels often costs time\n"
                 "4 of 15 widths run slower than the full layer", fontsize=10.5, pad=8)
    # The aligned-width marker goes in the legend rather than floating as a caption:
    # at this panel size any free corner is already claimed by the proportional line.
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], linestyle=":", color=style.MUTED, linewidth=1.5,
               label="if time fell with the channels"),
        Line2D([], [], linestyle="none", marker="o", color=style.ORANGE,
               markerfacecolor="white", markeredgewidth=2, markersize=8,
               label="width is a multiple of 32")],
        fontsize=8, frameon=False, loc="lower left")
    style.tidy(ax)

    # --- 3. noise -----------------------------------------------------------
    # Two different questions share this panel: rebuilding an entry, and merely
    # re-timing one. Both bound how finely a search can rank candidates, and the
    # rebuild number is much the larger of the two.
    ax = axes[2]
    drift = d.get("drift") or {}
    if noise and noise.get("repeats"):
        labels, worst = [], 0.0
        for i, r in enumerate(noise["repeats"]):
            rel = [(v / np.mean(r["ms"]) - 1) * 100 for v in r["ms"]]
            ax.plot([i] * len(rel), rel, "o", color=style.BLUE, markersize=7,
                    markerfacecolor="white", markeredgewidth=1.8, zorder=3)
            labels.append(f"{r['out_ch']} ch\nrebuilt")
            worst = max(worst, r["spread_pct"])
        if drift.get("entries"):
            rel = [e["drift_pct"] for e in drift["entries"]]
            ax.plot([len(labels)] * len(rel), rel, "o", color=style.AQUA,
                    markersize=6, markerfacecolor="white", markeredgewidth=1.6,
                    zorder=3)
            labels.append("same engine\nre-timed")
        ax.axhline(0, color=style.INK, linewidth=1.0, zorder=2)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_xlim(-0.5, len(labels) - 0.5)
        ax.set_ylabel("deviation from the mean (%)")
        ax.set_title(f"3. An entry is only good to ~{worst:.0f}%\n"
                     f"which sets the finest usable grid", fontsize=10.5, pad=8)
        style.tidy(ax)
        ax.grid(axis="x", visible=False)

    # --- 4. composition -----------------------------------------------------
    ax = axes[3]
    vals = [slope["unpruned_lut_ms"], slope["unpruned_real_ms"]]
    ax.bar([0, 1], vals, width=0.55, color=[style.RED, style.MUTED], zorder=3)
    for x, v in zip([0, 1], vals):
        ax.text(x, v + max(vals) * 0.02, f"{v:.1f}", ha="center", fontsize=9.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["60 layers\nsummed", "real\nengine"], fontsize=9)
    ax.set_ylabel("latency at batch 16 (ms)")
    ax.set_ylim(0, max(vals) * 1.22)
    ax.set_title(f"4. Totals do not compose\n{slope['ratio_lut_over_real']:.2f}x too high",
                 fontsize=10.5, pad=8)
    style.tidy(ax)
    ax.grid(axis="x", visible=False)

    # --- 5. slope, the decisive one -----------------------------------------
    ax = axes[4]
    arms = slope["arms"]
    ys = np.arange(len(arms))[::-1]
    h = 0.34
    ax.barh(ys + h / 2, [a["predicted_saving_ms"] for a in arms], height=h * 0.9,
            color=style.BLUE, label="table predicts", zorder=3)
    ax.barh(ys - h / 2, [a["actual_saving_ms"] for a in arms], height=h * 0.9,
            color=style.AQUA, label="hardware delivers", zorder=3)
    for y, a in zip(ys, arms):
        for off, v in ((h / 2, a["predicted_saving_ms"]), (-h / 2, a["actual_saving_ms"])):
            ha, dx = ("left", 0.4) if v >= 0 else ("right", -0.4)
            ax.text(v + dx, y + off, f"{v:+.1f}", va="center", ha=ha, fontsize=8.5,
                    color=style.INK_2)
    ax.axvline(0, color=style.INK, linewidth=1.2, zorder=2)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"round_to={a['arm'].replace('round', '')}" for a in arms],
                       fontsize=9.5)
    # Deliberately says "saved": panel 2 plots a time and this plots a difference,
    # and a reader carrying the first convention into the second reads it backwards.
    ax.set_xlabel("latency SAVED vs unpruned, ms\n"
                  "a difference, not a time \u2014 negative = slower")
    ax.set_xlim(-16, 24)
    ok = all(a["sign_agrees"] for a in arms)
    ax.set_title("5. But the direction is right\n"
                 + ("both signs correct" if ok else "a sign is wrong"),
                 fontsize=10.5, pad=8)
    # Lower left is the only free quadrant: round_to=1 puts its bars left of zero
    # in the upper half, round_to=32 puts its bars right of zero in the lower half.
    ax.legend(fontsize=8.5, frameon=False, loc="lower left")
    style.tidy(ax)
    ax.grid(axis="y", visible=False)

    fig.tight_layout()
    return save(fig, "xp06e9_netadapt.png")

def _xp15_record():
    """XP15's results, preferring the quantized model over its float ancestor.

    Both are the same 4,306 frames and the same network; one is what was trained
    and the other is what runs on the board. Quantization moves scores enough to
    change which threshold satisfies the deployment rule, so a figure captioned
    "the gate" should show the one that ships.
    """
    for name in ("xp15_gate_96px_w1.0_int8_full.json", "xp15_gate_96px_w1.0.json"):
        p = RAW / name
        if p.exists():
            return json.loads(p.read_text())
    hits = sorted(RAW.glob("xp15_gate_*.json"))
    return json.loads(hits[0].read_text()) if hits else None

def fig_xp15(records) -> Path | None:
    """The ESP32 gate: what it wakes for, what it sleeps through, and at what cost.

    Four panels answering the only question that decides whether the port happens.

    The confusion panel is one row per ground-truth category rather than a square
    matrix, because the gate is binary: it has four kinds of input and two possible
    responses. Read the 'nothing' row as the false alarm rate and every other row
    as recall for that category.

    Recall is never pooled. On this split most positives are large, so a pooled
    number is dominated by frames where the fire is already obvious -- which is
    precisely the failure mode a wake-up gate has to be checked for, since a gate
    that only fires on obvious fire fires too late to be worth having.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Rectangle

    # The int8 record when it exists, because that is the model the board runs;
    # the float one otherwise. Chosen by name rather than by modification time,
    # which would swap the figure's subject whenever a file was touched.
    d = _xp15_record()
    if d is None:
        return None
    m = d["metrics"]
    conf, sw = m.get("confusion_pct"), d.get("threshold_sweep")
    if not conf:
        return None

    fig, axes = plt.subplots(1, 4, figsize=(19.5, 4.4),
                             gridspec_kw={"width_ratios": [1.15, 1.0, 1.1, 1.0]})
    which = "int8, as it runs on the board" if d.get("quantization") else "float32"
    fig.suptitle(f"A 180 KB int8 gate on a EUR 27 board: what does it wake the detector for?"
                 if d.get("quantization") else
                 "A 131 KB gate on a EUR 27 board: what would it wake the detector for?",
                 y=1.10)
    verdict = (f"PASSES the port rule at threshold {d['operating_point']['threshold']}"
               if d.get("operating_point") else
               "does not meet the port rule at any threshold")
    style.subtitle(fig, f"{d['input_res']}px grayscale, {d['params']/1000:.0f}k parameters. "
                        f"{verdict}. Model: {which}.",
                   y=1.02)

    # --- 1. the confusion matrix, at both operating points -------------------
    # Row-normalised: each row is one ground-truth category and sums to 100%, so a
    # cell reads "of the frames that were really this, the gate did that". Raw
    # counts would say almost nothing here, since 'none' has 2005 frames and 'fire'
    # 220 and the class balance would dominate every colour.
    #
    # Both thresholds are drawn because they are two different products rather than
    # a tuning detail: one wakes on nearly every fire and cries wolf on one empty
    # frame in seven, the other is quiet enough to deploy and sleeps through a
    # quarter of real fires.
    ax = axes[0]
    op = d.get("metrics_at_operating_point")
    rows = [c for c in ("none", "smoke", "fire", "both") if c in conf]
    mats = [(conf, m["threshold"])] + ([(op["confusion_pct"], op["threshold"])] if op else [])

    nr = len(rows)
    ax.set_xlim(0, 2); ax.set_ylim(0, len(mats) * (nr + 1.15))
    ax.axis("off")

    for k, (cm, thr) in enumerate(mats):
        y0 = (len(mats) - 1 - k) * (nr + 1.15)
        ax.text(1, y0 + nr + 0.62, f"threshold {thr}", ha="center", va="center",
                fontsize=9.5, fontweight="bold", color=style.INK)
        for col, key, lab in ((0, "asleep_pct", "stayed asleep"), (1, "woke_pct", "woke")):
            ax.text(col + 0.5, y0 + nr + 0.12, lab, ha="center", va="center",
                    fontsize=8.5, color=style.INK_2)
        for r, c in enumerate(rows):
            y = y0 + nr - 1 - r
            for col, key in ((0, "asleep_pct"), (1, "woke_pct")):
                v = cm[c][key]
                # Green where the gate did the right thing, red where it did not:
                # for 'none' the correct answer is to stay asleep, for every other
                # row it is to wake.
                good = (key == "asleep_pct") if c == "none" else (key == "woke_pct")
                base = style.AQUA if good else style.RED
                ax.add_patch(Rectangle((col, y), 1, 1, facecolor=base,
                                       alpha=0.15 + 0.75 * v / 100,
                                       edgecolor="white", linewidth=1.5, zorder=3))
                ax.text(col + 0.5, y + 0.5, f"{v:.0f}%", ha="center", va="center",
                        fontsize=10, fontweight="bold",
                        color="white" if v > 55 else style.INK, zorder=4)
            ax.text(-0.06, y + 0.5, f"{c}\n(n={cm[c]['n']})", ha="right", va="center",
                    fontsize=8.5, color=style.INK)

    ax.set_title("1. Confusion on the test set (%)\neach row sums to 100",
                 fontsize=10.5, pad=8)

    # --- 2. recall by how big the plume is ----------------------------------
    ax = axes[1]
    keys = [("recall_obvious", "obvious"), ("recall_small_plume", "small"),
            ("recall_tiny_plume", "tiny")]
    src = op or m          # the threshold the gate would actually ship with
    vals = [(lab, src[k]) for k, lab in keys if src.get(k) is not None]
    xs = np.arange(len(vals))
    ax.bar(xs, [v * 100 for _, v in vals], width=0.6, color=style.BLUE, zorder=3)
    for x, (_, v) in zip(xs, vals):
        ax.text(x, v * 100 - 4, f"{v*100:.0f}%", ha="center", va="top", fontsize=10,
                color="white", fontweight="bold", zorder=4)
    ax.axhline(70, color=style.RED, linestyle="--", linewidth=1.3, zorder=2)
    ax.text(-0.45, 73, "the bar: 70% on small", ha="left", fontsize=8,
            color=style.RED)
    ax.set_xticks(xs); ax.set_xticklabels([lab for lab, _ in vals], fontsize=9.5)
    ax.set_ylim(0, 108); ax.set_ylabel("recall (%)")
    ax.set_title(f"2. Recall at threshold {src['threshold']}\n"
                 f"the number that decides the port", fontsize=10.5, pad=8)
    style.tidy(ax); ax.grid(axis="x", visible=False)

    # --- 3. the operating curve --------------------------------------------
    ax = axes[2]
    if sw:
        fw = [r["false_wake_rate"] * 100 for r in sw]
        for key, lab, col in (("recall_all", "any fire/smoke", style.BLUE),
                              ("recall_small", "small plume", style.AQUA),
                              ("recall_tiny", "tiny plume", style.ORANGE)):
            v = [r[key] for r in sw]
            if all(x is not None for x in v):
                ax.plot(fw, [x * 100 for x in v], "-", color=col, linewidth=2,
                        label=lab, zorder=3)
        ax.axvline(5, color=style.RED, linestyle="--", linewidth=1.3, zorder=2)
        ax.text(5.6, 4, "5% false-wake budget", fontsize=8, color=style.RED, rotation=90)
        ax.set_xlabel("false wakes on empty frames (%)")
        ax.set_ylabel("recall (%)")
        ax.set_xlim(0, 60); ax.set_ylim(0, 104)
        ax.set_title("3. The trade-off you get to pick\nthreshold sweep",
                     fontsize=10.5, pad=8)
        ax.legend(fontsize=8.5, frameon=False, loc="lower right")
        style.tidy(ax)

    # --- 4. where the scores actually sit -----------------------------------
    ax = axes[3]
    hist = d.get("score_histogram") or {}
    edges = np.linspace(0, 1, 21)
    centres = (edges[:-1] + edges[1:]) / 2
    for c, col in (("none", style.RED), ("smoke", style.AQUA),
                   ("fire", style.ORANGE), ("both", style.BLUE)):
        if c in hist:
            h = np.array(hist[c], float)
            # Normalised per category: 'none' has 2005 frames and 'fire' 220, so
            # raw counts would show only the class balance.
            ax.plot(centres, h / max(h.sum(), 1) * 100, "-", color=col,
                    linewidth=1.8, label=c, zorder=3)
    for thr, lab, col in ((m["threshold"], "default", style.MUTED),
                          ((op or m)["threshold"], "shipping", style.INK)):
        ax.axvline(thr, color=col, linestyle=":", linewidth=1.4, zorder=2)
        ax.text(thr - 0.02, ax.get_ylim()[1] * 0.55, f"{lab} {thr}", fontsize=8,
                color=col, rotation=90, ha="right", va="center")
    ax.set_xlabel("gate score")
    ax.set_ylabel("% of that category's frames")
    ax.set_title("4. Are the classes separated?\nscore distribution", fontsize=10.5, pad=8)
    ax.legend(fontsize=8.5, frameon=False)
    style.tidy(ax)

    fig.tight_layout()
    return save(fig, "xp15_gate.png")

def fig_xp15_confusion(records) -> Path | None:
    """The gate's confusion matrix proper, at both thresholds, with what follows from it.

    Separate from the four-panel summary because this is the standard artefact and
    deserves to be readable on its own: the binary matrix a classifier is normally
    judged by, plus the four-way breakdown of what the positives actually were.

    Counts *and* row percentages are printed in every cell. The percentage is what
    the behaviour looks like; the count is what it rests on, and 'fire' has only
    220 frames, so a row read as a percentage alone would hide how thin it is.

    Built from the per-frame scores stored in the record rather than from its
    aggregates, so the matrix can be cut at any threshold without a rescore.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Rectangle

    d = _xp15_record()
    if d is None:
        return None
    pf = d.get("per_frame")
    if not pf:
        return None

    score = np.array(pf["score"], float)
    y = np.array(pf["label"], int)
    content = np.array(pf["content"])

    def matrix(t):
        """sklearn where it is importable, numpy where it is not.

        The confusion matrix is four comparisons and does not need a dependency, but
        sklearn is the reference implementation and worth deferring to when present.
        """
        pred = (score >= t).astype(int)
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                from sklearn.metrics import confusion_matrix
            return confusion_matrix(y, pred, labels=[0, 1])
        except Exception:
            return np.array([[int(((y == a) & (pred == b)).sum()) for b in (0, 1)]
                             for a in (0, 1)])

    thrs = [d["metrics"]["threshold"]]
    if d.get("operating_point"):
        thrs.append(d["operating_point"]["threshold"])

    fig, axes = plt.subplots(1, len(thrs) + 1, figsize=(5.0 * (len(thrs) + 1), 4.9),
                             gridspec_kw={"width_ratios": [1] * len(thrs) + [1.25]})
    fig.suptitle("The ESP32 gate on the D-Fire test set: 4,306 frames, one binary decision",
                 y=1.06)
    # Says where these were computed, because "the ESP32 gate" reads as if the chip
    # produced them. It produced 8 of them, to within 0.004 and with no decision
    # changed; the rest would take 25 minutes on the board to reconfirm.
    style.subtitle(fig, f"{d['input_res']}px grayscale, {d['params']/1000:.0f}k parameters, "
                        f"int8 \u2014 the model the board runs, scored off-device. Cells show "
                        f"count and row percentage; each row sums to 100%.", y=0.98)

    labels = ["no fire/smoke", "fire or smoke"]
    for ax, t in zip(axes, thrs):
        cm = matrix(t)
        tn, fp, fn, tp = cm.ravel()
        rows = cm.sum(1, keepdims=True)
        pct = cm / np.maximum(rows, 1) * 100
        for i in range(2):
            for j in range(2):
                # Correct on the diagonal here, because this matrix really is
                # true-versus-predicted with the same two classes on both axes.
                col = style.AQUA if i == j else style.RED
                ax.add_patch(Rectangle((j, 1 - i), 1, 1, facecolor=col,
                                       alpha=0.15 + 0.75 * pct[i, j] / 100,
                                       edgecolor="white", linewidth=2, zorder=3))
                ax.text(j + 0.5, 1 - i + 0.58, f"{pct[i, j]:.1f}%", ha="center",
                        va="center", fontsize=15, fontweight="bold", zorder=4,
                        color="white" if pct[i, j] > 55 else style.INK)
                ax.text(j + 0.5, 1 - i + 0.28, f"n = {cm[i, j]:,}", ha="center",
                        va="center", fontsize=9.5, zorder=4,
                        color="white" if pct[i, j] > 55 else style.INK_2)
        ax.set_xlim(0, 2); ax.set_ylim(0, 2); ax.set_aspect("equal")
        ax.set_xticks([0.5, 1.5]); ax.set_xticklabels(["stayed asleep", "woke"], fontsize=10)
        ax.set_yticks([1.5, 0.5]); ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel("what the gate did"); ax.set_ylabel("what was really there")
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        ax.set_title(f"threshold {t}\nprecision {prec:.2f} · recall {rec:.2f} · F1 {f1:.2f}",
                     fontsize=11, pad=10)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)

    # --- what the positives actually were -----------------------------------
    ax = axes[-1]
    cats = [c for c in ("smoke", "fire", "both") if (content == c).any()]
    ys = np.arange(len(cats))[::-1]
    h = 0.36
    for k, t in enumerate(thrs):
        woke = [float(((content == c) & (score >= t)).sum())
                / max((content == c).sum(), 1) * 100 for c in cats]
        off = (0.5 - k) * h if len(thrs) > 1 else 0
        ax.barh(ys + off, woke, height=h * 0.9, color=style.AQUA,
                hatch=None if k == 0 else "///", edgecolor="white", linewidth=0.6,
                zorder=3, label=f"threshold {t}")
        for yy, w in zip(ys, woke):
            ax.text(w + 1.5, yy + off, f"{w:.0f}%", va="center", fontsize=9,
                    color=style.INK_2)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{c}\n(n={int((content == c).sum())})" for c in cats], fontsize=10)
    ax.set_xlim(0, 112); ax.set_xlabel("% woken on")
    ax.set_title("Which positives it caught\nthe binary matrix pools these three",
                 fontsize=11, pad=10)
    # Above the bars, not inside them: every bar here runs past 65% so the lower
    # right corner a legend would normally take is occupied.
    ax.legend(fontsize=8.5, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.13), ncol=2)
    style.tidy(ax); ax.grid(axis="y", visible=False)

    fig.tight_layout()
    return save(fig, "xp15_confusion.png")

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

    from matplotlib.patches import Rectangle
    ndata = 3 if have_speed else 2
    fig = plt.figure(figsize=(4.7 + 5.0 * ndata, 4.4))
    gs = fig.add_gridspec(1, ndata + 1, width_ratios=[0.82] + [1] * ndata, wspace=0.30)
    sch = fig.add_subplot(gs[0, 0])
    axes = [fig.add_subplot(gs[0, i + 1]) for i in range(ndata)]
    fig.suptitle("Rounding channel widths: free accuracy, and 1.77x the speed on the board",
                 y=1.03)
    style.subtitle(fig, "Same 25% cut, same accuracy. Snapping the surviving widths to clean "
                        "multiples nearly doubles throughput on the Jetson.", y=0.965)

    # --- panel 1: what round_to does to one layer -------------------------
    # A layer pruned to 52 channels spills past the 32-wide tile the GPU works
    # in, paying for a second tile it barely fills; snapping to 32 fills one tile
    # exactly. This is the mechanism the three data panels then measure.
    sch.set_xlim(-3, 68); sch.set_ylim(-1.2, 9.2); sch.axis("off")
    sch.set_title("1. What round_to does\nto one layer's width", fontsize=10.5, pad=6,
                  loc="left")
    for t in (0, 32, 64):                                   # tile boundaries
        sch.plot([t, t], [-0.4, 8.4], color=style.INK_2, linewidth=1.0,
                 linestyle=(0, (3, 3)), zorder=1)
    sch.text(16, 8.6, "tile 1", ha="center", fontsize=7.5, color=style.INK_2)
    sch.text(48, 8.6, "tile 2", ha="center", fontsize=7.5, color=style.INK_2)

    def lane(y, width, colour, label):
        sch.add_patch(Rectangle((0, y), width, 1.5, facecolor=colour, edgecolor="white",
                                linewidth=1.0, zorder=3))
        sch.text(-2, y + 0.75, label, ha="right", va="center", fontsize=8.5,
                 fontweight="bold", color=colour)

    lane(5.4, 52, style.RED, "pruned\nto 52")
    # the wasted remainder of tile 2
    sch.add_patch(Rectangle((52, 5.4), 12, 1.5, facecolor="none", edgecolor=style.RED,
                            hatch="////", linewidth=0.0, zorder=2))
    sch.text(58, 4.7, "wasted", ha="center", va="top", fontsize=7.5, color=style.RED)
    lane(1.2, 32, style.BLUE, "rounded\nto 32")
    sch.text(16, 0.2, "fills tile 1 exactly", ha="center", va="top", fontsize=7.5,
             color=style.BLUE)
    sch.text(32, -0.9, "channels ->", ha="center", fontsize=7.5, color=style.INK_2)

    # Panel 1 (data): alignment.
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

    from matplotlib.patches import Rectangle, FancyArrowPatch
    fig = plt.figure(figsize=(16.4, 4.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.80, 1.55, 1.0], wspace=0.32)
    sch = fig.add_subplot(gs[0, 0])
    axes = [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])]
    fig.suptitle("The one pattern the hardware understands, measured on the hardware", y=1.11)

    # --- panel 1: the choice the compiler makes ---------------------------
    # The 2:4 weights are genuine, so 39 layers are eligible for a sparse kernel
    # that skips the two zeros. But TensorRT times a sparse and a dense kernel for
    # each and keeps the faster, and here it kept dense every time. The schematic
    # shows the decision the data then quantifies.
    sch.set_xlim(0, 10); sch.set_ylim(-0.6, 9.4); sch.axis("off")
    sch.set_title("1. The choice the\ncompiler makes", fontsize=10.5, pad=6, loc="left")
    sch.text(5, 8.7, "39 layers are in 2:4 form,\neligible for a sparse kernel",
             ha="center", va="top", fontsize=8, color=style.INK, fontweight="bold")

    def kbox(x, y, w, h, colour, title, sub):
        sch.add_patch(Rectangle((x, y), w, h, facecolor="none", edgecolor=colour,
                                linewidth=1.8, zorder=3))
        sch.text(x + w / 2, y + h - 0.35, title, ha="center", va="top", fontsize=8,
                 color=colour, fontweight="bold")
        sch.text(x + w / 2, y + 0.48, sub, ha="center", va="center", fontsize=6.8,
                 color=style.INK_2)

    # Stacked, not side by side: this column is the narrowest on the page, and two
    # boxes abreast leave each one too thin to hold its own label without spilling
    # over the neighbouring panel's tick labels.
    kbox(0.6, 6.05, 8.8, 1.65, style.ORANGE, "sparse kernel", "skips the 2 zeros, up to 2x")
    kbox(0.6, 4.05, 8.8, 1.65, style.INK_2, "dense kernel", "multiplies all four, zeros included")
    sch.text(5, 3.65, "TensorRT times both per layer,\nkeeps the faster",
             ha="center", va="top", fontsize=7.6, color=style.INK, style="italic")
    sch.add_patch(FancyArrowPatch((5, 2.75), (5, 1.95), arrowstyle="-|>", mutation_scale=13,
                                  color=style.INK_2, linewidth=1.4))
    sch.add_patch(Rectangle((0.6, 0.25), 8.8, 1.55, facecolor=style.INK_2, alpha=0.10,
                            edgecolor=style.INK_2, linewidth=1.3, zorder=2))
    sch.text(5, 1.02, "it kept DENSE every time\n0 of 39 used the sparse kernel",
             ha="center", va="center", fontsize=8.2, color=style.INK, fontweight="bold")
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





# --------------------------------------------------------------------------
# XP7 — quantization. The explainer figures are drawn from xp07_concepts.json,
# which is measured on this detector rather than sketched, so every teaching
# picture on the page is a fact about YOLOv5s and not about a textbook tensor.
# --------------------------------------------------------------------------

def fig_xp07_concepts(records) -> Path | None:
    """What a scale is, why granularity exists, and what calibration decides."""
    import matplotlib.pyplot as plt
    import numpy as np

    d = _side("xp07_concepts.json")
    if not d:
        return None

    w, a = d["weights"], d["activations"]
    fig, axes = plt.subplots(1, 3, figsize=(17.4, 4.6),
                             gridspec_kw={"width_ratios": [1, 1, 1.35]})
    fig.suptitle("Quantization is one decision repeated: where do you put the grid?", y=1.06)
    style.subtitle(fig, "All three panels are measured on this detector — YOLOv5s at 512 px — "
                        "not drawn as a schematic.", y=1.005)

    # --- panel 1: the grid ------------------------------------------------
    ax = axes[0]
    edges = np.linspace(w["hist_min"], w["hist_max"], len(w["hist_counts"]) + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    ax.fill_between(centres, w["hist_counts"], color=style.BLUE, alpha=0.30, zorder=2)
    ax.plot(centres, w["hist_counts"], color=style.BLUE, lw=1.4, zorder=3)

    s = w["per_tensor_scale"]
    top = max(w["hist_counts"])
    for k in range(-8, 9):                      # a few grid lines, not all 255
        ax.axvline(k * s, color=style.INK_2, lw=0.7, alpha=0.40, zorder=1)
    # One step, shaded. At this zoom a band reads where a 1-step arrow collapses.
    ax.axvspan(2 * s, 3 * s, color=style.ORANGE, alpha=0.35, zorder=2)
    ax.annotate(f"one step\nS = {s:.5f}", xy=(2.5 * s, top * 0.40),
                xytext=(5.6 * s, top * 0.80), ha="center", fontsize=9,
                color=style.ORANGE, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=style.ORANGE, lw=1.4))
    ax.set_xlim(-8.5 * s, 8.5 * s)
    ax.set_title(f"1. INT8 keeps 255 values. The scale\nsays which ones. ({w['layer']})",
                 fontsize=11, loc="left")
    ax.set_xlabel("weight value, zoomed to ±8 steps")
    ax.set_ylabel("number of weights")
    ax.set_yticks([])
    style.tidy(ax)

    # --- panel 2: granularity --------------------------------------------
    ax = axes[1]
    amax = np.array(w["per_channel_absmax"])
    order = np.argsort(-amax)
    ax.bar(range(len(amax)), amax[order], width=1.0, color=style.AQUA,
           alpha=0.85, zorder=3)
    pt = float(np.max(amax))
    ax.axhline(pt, color=style.RED, lw=1.6, zorder=4)
    ax.text(len(amax) * 0.45, pt * 1.02,
            f"per-tensor: every channel is given the widest channel's range ({pt:.2f})",
            fontsize=8.6, color=style.RED, fontweight="bold", va="bottom")
    ax.annotate(f"{w['widest_over_narrowest']:.1f}x spread\nwidest to narrowest",
                xy=(len(amax) * 0.80, amax[order][int(len(amax) * 0.80)]),
                xytext=(len(amax) * 0.52, pt * 0.45), fontsize=8.6,
                color=style.INK_2,
                arrowprops=dict(arrowstyle="->", color=style.INK_2, lw=1.1))
    ax.set_title("2. Per-channel gives each filter its own step,\nand costs nothing to run.",
                 fontsize=11, loc="left")
    ax.set_xlabel(f"the {len(amax)} output filters of {w['layer']}, widest first")
    ax.set_ylabel("largest weight magnitude")
    style.tidy(ax)

    # --- panel 3: calibration --------------------------------------------
    ax = axes[2]
    counts = np.array(a["hist_counts"], dtype=float)
    bw = a["bin_width"]
    x = (np.arange(len(counts)) + 0.5) * bw
    ax.fill_between(x, counts, color=style.BLUE, alpha=0.28, zorder=2)
    ax.plot(x, counts, color=style.BLUE, lw=1.2, zorder=3)
    ax.set_yscale("log")

    seen = {}
    for name, v in a["clips"].items():
        seen.setdefault(round(v["represented_max"], 3), []).append(name)
    peak = counts[counts > 0].max()
    for r, names in sorted(seen.items()):
        clipper = "entropy" in names
        col = style.RED if clipper else style.AQUA
        ax.axvline(r, color=col, lw=2.2, zorder=5)
        label = " / ".join(n.replace("percentile_", "pct ") for n in names)
        # The clipper is labelled to the left of its line, the keepers above the
        # axis on the right, so the two groups never share space.
        if clipper:
            ax.text(r - 0.03, peak * 0.6, f"{label}\nkeeps 0–{r:.4f}",
                    ha="right", va="center", fontsize=9.2, color=col, fontweight="bold")
        else:
            ax.text(0.99, peak * 2.0, f"{label}\nall keep 0–{r:.4f}", ha="right",
                    va="bottom", fontsize=9.2, color=col, fontweight="bold")
    ax.axvspan(min(seen), 1.0, color=style.RED, alpha=0.08, zorder=1)
    # Placed in the empty band between the histogram and the keeper labels, so
    # the sentence never sits on the data it is describing.
    ax.text(0.735, peak * 0.02,
            "entropy flattens everything in here\nto one value — and bright sky is\nexactly where faint smoke has to\nbe seen against it",
            ha="center", va="center", fontsize=8.4, color=style.RED)
    ax.set_ylim(top=peak * 14)
    ax.set_title("3. Calibration picks the clip point.\nOne method picks a different one.",
                 fontsize=11, loc="left")
    ax.set_xlabel("input image brightness, normalised to 0–1")
    ax.set_ylabel("pixels, log scale")
    ax.set_xlim(0, 1.02)
    style.tidy(ax)

    fig.tight_layout()
    return save(fig, "xp07_concepts.png")


def fig_xp07_head(records) -> Path | None:
    """Why the detection head cannot take a per-tensor INT8 scale."""
    import matplotlib.pyplot as plt
    import numpy as np

    d = _side("xp07_concepts.json")
    if not d:
        return None
    det = d["detect_output"]

    names = det["channel_names"]
    amax = np.array(det["channel_absmax"])
    step = det["per_tensor_step"]
    is_box = np.array([n in ("x", "y", "w", "h") for n in names])

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.6),
                             gridspec_kw={"width_ratios": [1.25, 1]})
    fig.suptitle("One tensor, two populations, and a single scale that cannot serve both",
                 y=1.06)
    style.subtitle(fig, "YOLOv5's Detect layer concatenates box coordinates in pixels with "
                        "probabilities in [0,1]. INT8 carries one scale per tensor.", y=1.0)

    # --- left: the two populations ---------------------------------------
    ax = axes[0]
    cols = [style.BLUE if b else style.ORANGE for b in is_box]
    ax.bar(names, amax, color=cols, zorder=3, width=0.68)
    ax.set_yscale("log")
    ax.axhline(step, color=style.RED, lw=1.8, ls="--", zorder=4)
    ax.text(len(names) - 0.4, step * 1.25,
            f"one INT8 step = {step:.2f}", ha="right", fontsize=9.5,
            color=style.RED, fontweight="bold")
    for i, (n, v) in enumerate(zip(names, amax)):
        ax.text(i, v * 1.3, f"{v:.2f}".rstrip("0").rstrip("."), ha="center",
                fontsize=9, color=style.INK, fontweight="bold")
    ax.text(1.5, amax.max() * 3.0, "box coordinates, in pixels", ha="center",
            fontsize=9.5, color=style.BLUE, fontweight="bold")
    ax.text(5.0, amax.max() * 3.0, "probabilities, in [0,1]", ha="center",
            fontsize=9.5, color=style.ORANGE, fontweight="bold")
    ax.set_ylim(top=amax.max() * 9)
    ax.set_ylabel("largest value seen in this channel, log scale")
    ax.set_xlabel("the 7 channels of the decoded Detect output")
    ax.set_title("The scale is set by the widest channel", fontsize=11, loc="left")
    style.tidy(ax)

    # --- right: what each population gets --------------------------------
    ax = axes[1]
    levels_box = amax[is_box].max() / step
    levels_prob = 1.0 / step
    bars = ax.barh(["box coordinate\n(range 0–%.0f)" % amax[is_box].max(),
                    "probability\n(range 0–1)"],
                   [levels_box, levels_prob],
                   color=[style.BLUE, style.ORANGE], zorder=3, height=0.34)
    ax.set_xscale("log")
    ax.set_xlim(0.02, 4000)
    ax.axvline(1.0, color=style.RED, lw=1.6, zorder=4)
    ax.text(1.0, 1.42, "one step", fontsize=9, color=style.RED,
            fontweight="bold", ha="center")
    ax.text(levels_box * 1.3, 0, f"{levels_box:.0f} levels", va="center",
            fontsize=10.5, color=style.BLUE, fontweight="bold")
    # Parked to the right of the "one step" line so the sentence crosses nothing.
    ax.text(2.6, 1, f"{levels_prob:.3f} of one level\n→ every probability rounds to zero",
            va="center", ha="left", fontsize=10.5, color=style.RED, fontweight="bold")
    ax.set_ylim(-0.6, 1.7)
    ax.set_xlabel("how many INT8 levels this quantity actually gets")
    ax.set_title("Box regression survives. The classifier does not.",
                 fontsize=11, loc="left")
    style.tidy(ax, ygrid=False)

    fig.tight_layout()
    return save(fig, "xp07_head.png")




def fig_xp07e9(records) -> Path | None:
    """Every technique in the series on four axes. The ranking is not one ranking."""
    import matplotlib.pyplot as plt
    import numpy as np

    d = _side("xp07e9_frontier.json")
    if not d:
        return None
    rows = [r for r in d["rows"] if r.get("map50") and r.get("fps_batched")]
    if not rows:
        return None
    line = next((r for r in rows if r["family"] == "baseline"
                 and r["input_res"] == 512), None)

    fam_colour = {
        "baseline": style.INK, "resolution": style.BLUE,
        "pruning": style.ORANGE, "sparsity": style.AQUA,
        "quantization": "#8e44ad", "composition": style.RED,
    }

    def area(j):
        return 70 if not j else max(50, min(900, j * 4.2))

    fig = plt.figure(figsize=(16.8, 9.4))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.24)
    fig.suptitle("No technique wins outright — the best one depends on the accuracy "
                 "you are willing to give up", y=0.975, fontsize=15)
    # Wrapped: an unbroken subtitle this long makes bbox_inches="tight" stretch
    # the whole figure to fit it.
    style.subtitle(fig, "Every point is one engine measured on the Jetson Orin Nano Super at "
                        "512 px unless labelled otherwise.\nMarker area is energy per 1,000 "
                        "frames (bigger dot, more joules) and is indicative only — XP7's engines "
                        "were integrated over a\nflat-out batch-16 window, the older rows over a "
                        "batch-1 one, and the same FP16 engine reads 43.3 vs 52.1 J/1k.",
                   y=0.958)

    def scatter(ax, xk, yk, xlabel, ylabel, title, logx=False, invert=False):
        for r in rows:
            c = fam_colour.get(r["family"], style.MUTED)
            ax.scatter([r[xk]], [r[yk]], s=area(r["j_per_1k"]), color=c,
                       alpha=0.80, zorder=4, edgecolor="white", linewidth=1.1)
        if logx:
            ax.set_xscale("log")
        if invert:
            ax.invert_xaxis()
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=12, loc="left")
        style.tidy(ax)

    # ---- panel 1: accuracy vs throughput --------------------------------
    ax = fig.add_subplot(gs[0, 0])
    if line:
        ax.axhline(line["map50"], color=style.INK_2, ls="--", lw=1.0, zorder=1)
        ax.axvline(line["fps_batched"], color=style.INK_2, ls="--", lw=1.0, zorder=1)
        xs = max(r["fps_batched"] for r in rows) * 1.10
        ax.fill_betweenx([line["map50"], 0.83], line["fps_batched"], xs,
                         color=style.AQUA, alpha=0.10, zorder=0)
        ax.text(line["fps_batched"] * 1.03, 0.822, "better than the line\n(nothing is here)",
                fontsize=9, color="#0d5f43", va="top", fontweight="bold")
    scatter(ax, "fps_batched", "map50", "throughput, images/s at batch 16 (higher is better)",
            "mAP50 on the 4,306-image test set", "1. Accuracy vs speed")
    ax.set_ylim(0.20, 0.83)
    for r in rows:
        if r["map50"] < 0.60:                    # the broken arm, labelled in place
            ax.annotate(r["label"], (r["fps_batched"], r["map50"]),
                        textcoords="offset points", xytext=(-8, 10), ha="right",
                        fontsize=8.2, color=style.RED, fontweight="bold")

    # ---- panel 2: accuracy vs size --------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    if line:
        ax.axhline(line["map50"], color=style.INK_2, ls="--", lw=1.0, zorder=1)
        ax.axvline(line["size_disk_mb"], color=style.INK_2, ls="--", lw=1.0, zorder=1)
    scatter(ax, "size_disk_mb", "map50", "engine on disk, MB, log scale (smaller is better →)",
            "mAP50 on the 4,306-image test set", "2. Accuracy vs size",
            logx=True, invert=True)
    ax.set_ylim(0.20, 0.83)
    ax.set_xticks([10, 15, 20, 40, 60, 95])
    ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    if line:
        ax.annotate("everything compressed lands in 9–17 MB,\nand none of it reaches the line",
                    xy=(11.5, 0.745), xytext=(11.5, 0.46), ha="center", fontsize=9,
                    color=style.INK_2,
                    arrowprops=dict(arrowstyle="->", color=style.INK_2, lw=1.0))

    # ---- panel 3: the slice that changes the ranking --------------------
    ax = fig.add_subplot(gs[1, 0])
    tiny = [r for r in rows if r.get("tiny_plume") is not None]
    if line and line.get("tiny_plume"):
        ax.axhline(line["tiny_plume"], color=style.INK_2, ls="--", lw=1.0, zorder=1)
        ax.text(max(r["fps_batched"] for r in tiny) * 0.99, line["tiny_plume"] * 1.06,
                f"the line: {line['tiny_plume']:.4f}", ha="right", fontsize=9,
                color=style.INK_2)
    placed = []
    for r in sorted(tiny, key=lambda x: x["fps_batched"]):
        c = fam_colour.get(r["family"], style.MUTED)
        ax.scatter([r["fps_batched"]], [r["tiny_plume"]], s=area(r["j_per_1k"]),
                   color=c, alpha=0.80, zorder=4, edgecolor="white", linewidth=1.1)
        keep = r["tiny_plume"] / line["tiny_plume"] * 100 if line and line["tiny_plume"] else None
        if keep is None or r is line:
            continue
        # Arms that land on top of each other get their labels pushed apart, so
        # two coincident dots do not print two numbers in the same place.
        crowded = any(abs(r["fps_batched"] - x) < 40 and abs(r["tiny_plume"] - y) < 0.02
                      for x, y in placed)
        ax.annotate(f"{keep:.0f}%", (r["fps_batched"], r["tiny_plume"]),
                    textcoords="offset points",
                    xytext=(26, -4) if crowded else (0, 12), ha="center",
                    fontsize=8.4, color=c, fontweight="bold")
        placed.append((r["fps_batched"], r["tiny_plume"]))
    ax.set_xlabel("throughput, images/s at batch 16")
    ax.set_ylabel("mAP50 on plumes under 0.1% of the frame")
    ax.set_ylim(-0.014, max(r["tiny_plume"] for r in tiny) * 1.30)
    ax.set_title("3. The same arms, scored on distant smoke only\n"
                 "(% = share of the FP16 line's tiny-plume accuracy kept)",
                 fontsize=12, loc="left")
    style.tidy(ax)

    # ---- panel 4: what to pick, per accuracy budget ---------------------
    ax = fig.add_subplot(gs[1, 1])
    budgets = d["verdict"]["fastest_at_each_accuracy_floor"]
    keys = list(budgets)[::-1]
    labels = [k.replace("map50>=", "keep mAP50 ≥ ") for k in keys]
    fps = [budgets[k]["fps"] for k in keys]
    cols = [fam_colour.get(budgets[k]["family"], style.MUTED) for k in keys]
    bars = ax.barh(labels, fps, color=cols, zorder=3, height=0.55)
    base = line["fps_batched"] if line else 0
    ax.axvline(base, color=style.INK_2, ls="--", lw=1.2, zorder=4)
    ax.text(base, -0.62, "the line\n474 img/s", fontsize=8.8, color=style.INK_2,
            ha="center", va="top", fontweight="bold")
    ax.set_ylim(-0.95, len(keys) - 0.35)
    for i, k in enumerate(keys):
        b = budgets[k]
        ax.text(b["fps"] * 1.01, i,
                f"  {b['fastest']}\n  {b['fps']:.0f} img/s · {b['size_mb']:.1f} MB"
                + (f" · {b['j_per_1k']:.0f} J/1k" if b.get("j_per_1k") else ""),
                va="center", fontsize=9, color=style.INK)
    ax.set_xlim(0, max(fps) * 1.85)
    ax.set_xlabel("throughput of the fastest arm that still clears the accuracy floor")
    ax.set_title("4. The practical answer: pick your floor, read off the winner",
                 fontsize=12, loc="left")
    style.tidy(ax, ygrid=False)

    handles = [plt.Line2D([], [], marker="o", ls="", color=c, markersize=9, label=f)
               for f, c in fam_colour.items()
               if any(r["family"] == f for r in rows)]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               bbox_to_anchor=(0.5, -0.01), frameon=False, fontsize=10.5)
    return save(fig, "xp07e9_frontier.png")




#: Family colours, shared by every XP7 frontier figure so a technique keeps its
#: colour whichever metric is being ranked.
XP07_FAMILY_COLOUR = {
    "baseline": style.INK, "resolution": style.BLUE,
    "pruning": style.ORANGE, "sparsity": style.AQUA,
    "quantization": "#8e44ad", "composition": style.RED,
}


def _xp07_metric_figure(key: str, fname: str, headline: str, blurb: str,
                        ylabel: str) -> Path | None:
    """One metric, two panels: who keeps it, and what that costs in speed.

    Aggregate mAP50 averages over two classes of very different difficulty and
    over every plume size, so it can stay flat while the capability that matters
    collapses. These per-metric figures are the check on that: same arms, same
    axes, one measure at a time.
    """
    import matplotlib.pyplot as plt

    d = _side("xp07e9_frontier.json")
    if not d:
        return None
    blk = ((d.get("verdict") or {}).get("per_metric") or {}).get(key)
    rows = [r for r in d["rows"] if r.get(key) is not None and r.get("fps_batched")]
    if not blk or not rows:
        return None
    base = blk["line_value"]
    line = next((r for r in rows if r["family"] == "baseline"
                 and r["input_res"] == 512), None)

    def area(j):
        return 70 if not j else max(50, min(900, j * 4.2))

    # The right panel's y labels are full technique names, so the gutter has to
    # be wide enough for them or they run into the left panel's data.
    fig = plt.figure(figsize=(17.6, 5.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.0], wspace=0.52)
    fig.suptitle(headline, y=1.045, fontsize=14.5)
    style.subtitle(fig, blurb, y=0.985)

    # ---- panel 1: this metric vs throughput -----------------------------
    ax = fig.add_subplot(gs[0, 0])
    ax.axhline(base, color=style.INK_2, ls="--", lw=1.1, zorder=1)
    xs = max(r["fps_batched"] for r in rows) * 1.12
    ax.fill_betweenx([base, max(r[key] for r in rows) * 1.25],
                     line["fps_batched"] if line else 0, xs,
                     color=style.AQUA, alpha=0.10, zorder=0)
    if line:
        ax.axvline(line["fps_batched"], color=style.INK_2, ls="--", lw=1.1, zorder=1)
    ax.text(xs * 0.99, base * 1.05, f"the line: {base:.4f}", ha="right", fontsize=9.5,
            color=style.INK_2, fontweight="bold")
    for r in rows:
        c = XP07_FAMILY_COLOUR.get(r["family"], style.MUTED)
        ax.scatter([r["fps_batched"]], [r[key]], s=area(r["j_per_1k"]), color=c,
                   alpha=0.82, zorder=4, edgecolor="white", linewidth=1.1)
    ax.set_xlim(0, xs)
    ax.set_ylim(-0.02 * max(r[key] for r in rows), max(r[key] for r in rows) * 1.25)
    ax.set_xlabel("throughput, images/s at batch 16 (higher is better)")
    ax.set_ylabel(ylabel)
    ax.set_title("Where each technique lands on this measure", fontsize=12, loc="left")
    style.tidy(ax)

    # ---- panel 2: how much of the line each one keeps -------------------
    ax = fig.add_subplot(gs[0, 1])
    ret = [r for r in blk["retention"]]
    labels = [r["label"] for r in ret][::-1]
    kept = [r["kept_pct"] for r in ret][::-1]
    cols = [XP07_FAMILY_COLOUR.get(r["family"], style.MUTED) for r in ret][::-1]
    ax.barh(labels, kept, color=cols, zorder=3, height=0.62)
    ax.axvline(100, color=style.INK_2, ls="--", lw=1.2, zorder=4)
    for i, r in enumerate(ret[::-1]):
        ax.text(r["kept_pct"] + 2.5, i,
                f"{r['kept_pct']:.0f}%   {r['fps']:.0f} img/s · {r['size_mb']:.1f} MB",
                va="center", fontsize=9, color=style.INK)
    ax.set_xlim(0, max(kept) * 1.62)
    ax.set_xlabel(f"share of the FP16 line's {blk['title']} that survives (%)")
    ax.set_title("How much of the line each technique keeps", fontsize=12, loc="left")
    style.tidy(ax, ygrid=False)

    handles = [plt.Line2D([], [], marker="o", ls="", color=c, markersize=9, label=f)
               for f, c in XP07_FAMILY_COLOUR.items()
               if any(r["family"] == f for r in rows)]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               bbox_to_anchor=(0.5, -0.09), frameon=False, fontsize=10.5)
    return save(fig, fname)


def fig_xp07e9_fire(records) -> Path | None:
    """The flame class on its own — the harder of the two, and the one that burns."""
    return _xp07_metric_figure(
        "map50_fire", "xp07e9_fire.png",
        "Flame: every technique costs more here than the headline number admits",
        "Fire is the harder class (the line scores 0.7184 on it against 0.8367 on smoke), "
        "so a technique's damage shows up on flame before it shows up on the average.",
        "mAP50 on the fire class, 4,306-image test set")


def fig_xp07e9_tiny(records) -> Path | None:
    """Distant smoke — the capability early detection actually depends on."""
    return _xp07_metric_figure(
        "tiny_plume", "xp07e9_tiny.png",
        "Distant smoke: the ranking that aggregate mAP50 hides",
        "Plumes under 0.1% of the frame — roughly 20x20 px. This is what early "
        "detection is, and it is where the cheapest-looking technique turns out to be "
        "the most expensive.",
        "mAP50 on plumes under 0.1% of the frame")




def fig_xp07e1(records) -> Path | None:
    """Where INT8 damage lives: not in the weights, and not spread out."""
    import matplotlib.pyplot as plt
    import numpy as np

    d = _side("xp07e1_sensitivity.json")
    if not d:
        return None
    bl = d["baseline_fp16"]
    base_map, base_tiny = bl["map50"], bl["tiny_plume"]["map50"]
    head = set(d["head_convs"])

    order = [r["layer"] for r in d["rows"] if r["arm"] == "w8"]
    idx = {n: i for i, n in enumerate(order)}
    by_arm = {a: {r["layer"]: r for r in d["rows"] if r["arm"] == a}
              for a in ("w8", "w8a8")}

    fig, axes = plt.subplots(1, 2, figsize=(17.0, 5.4))
    fig.suptitle("INT8 damage is not in the weights, and it is not spread out — "
                 "it is three layers of the detection head", y=1.05, fontsize=14.5)
    style.subtitle(fig, "Each point quantizes ONE of the 60 convolutions and leaves the "
                        "other 59 in FP16. Validation split, no retraining.", y=0.99)

    # ---- panel 1: aggregate mAP50 ---------------------------------------
    ax = axes[0]
    ax.axhline(100, color=style.INK_2, ls="--", lw=1.1, zorder=2)
    for arm, colour, label in (("w8", style.AQUA, "W8 — weights only"),
                               ("w8a8", style.BLUE, "W8A8 — weights + input activation")):
        xs = [idx[n] for n in order]
        ys = [by_arm[arm][n]["map50"] / base_map * 100 for n in order]
        ax.plot(xs, ys, color=colour, lw=1.4, marker="o", ms=3.4, zorder=3, label=label)
    for n in head:
        r = by_arm["w8a8"][n]
        ax.scatter([idx[n]], [r["map50"] / base_map * 100], s=130, facecolor="none",
                   edgecolor=style.RED, linewidth=2.0, zorder=5)
    worst = min(head, key=lambda n: by_arm["w8a8"][n]["map50"])
    lo = min(by_arm["w8a8"][n]["map50"] / base_map * 100 for n in order)
    ax.set_ylim(lo - 0.45, 100.2)
    ax.annotate("the three Detect head convolutions —\nthe only cells that move at all",
                xy=(idx[worst], by_arm["w8a8"][worst]["map50"] / base_map * 100),
                xytext=(len(order) * 0.30, lo + 0.25), ha="left", fontsize=9.5,
                color=style.RED, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=style.RED, lw=1.3))
    ax.set_xlabel("the 60 convolutions, in forward order")
    ax.set_ylabel("mAP50 kept, % of the unquantized model")
    ax.set_title("1. Quantize one layer: what does it cost?", fontsize=12, loc="left")
    ax.legend(loc="lower left", fontsize=9.5)
    style.tidy(ax)

    # ---- panel 2: the slice that finds the real victim -------------------
    ax = axes[1]
    ax.axhline(100, color=style.INK_2, ls="--", lw=1.1, zorder=2)
    xs = [idx[n] for n in order]
    ys = [by_arm["w8a8"][n]["tiny_plume"] / base_tiny * 100 for n in order]
    cols = [style.RED if n in head else style.BLUE for n in order]
    ax.bar(xs, ys, color=cols, width=0.82, zorder=3)
    ax.set_ylim(0, 125)

    # Both callouts go in the empty left half, stacked, so the arrows fan out to
    # the right instead of crossing each other over the bars.
    ranked = sorted(order, key=lambda n: by_arm["w8a8"][n]["tiny_plume"])[:2]
    for n, ty in zip(ranked, (40, 72)):
        r = by_arm["w8a8"][n]
        keep = r["tiny_plume"] / base_tiny * 100
        ax.annotate(f"{n} — {keep:.0f}% kept\n(its aggregate mAP50 fell only "
                    f"{r['rel_drop_pct']:.1f}%)",
                    xy=(idx[n], keep), xytext=(2, ty), ha="left", va="center",
                    fontsize=9.2, zorder=6,
                    color=style.RED if n in head else style.INK,
                    fontweight="bold",
                    # The callouts sit over the bars, so they carry their own
                    # background or the bar colour swallows the text.
                    bbox=dict(boxstyle="round,pad=0.35", fc=style.SURFACE,
                              ec="none", alpha=0.93),
                    arrowprops=dict(arrowstyle="->", lw=1.3, zorder=6,
                                    connectionstyle="arc3,rad=-0.12",
                                    color=style.RED if n in head else style.INK_2))
    ax.set_xlabel("the 60 convolutions, in forward order")
    ax.set_ylabel("tiny-plume mAP50 kept, % of the unquantized model")
    ax.set_title("2. The same 60 cells, scored on distant smoke only\n"
                 "(red = a Detect head convolution)", fontsize=12, loc="left")
    style.tidy(ax)

    fig.tight_layout()
    return save(fig, "xp07e1_sensitivity.png")




def fig_xp07e2(records) -> Path | None:
    """Both ends of the clipping sweep are wrong, and the middle was never tested."""
    import matplotlib.pyplot as plt
    import numpy as np

    d = _side("xp07e2_calibration.json")
    if not d:
        return None
    bl = d["baseline_fp16"]
    base_map, base_tiny = bl["map50"], bl["tiny_plume"]["map50"]
    meth = d["methods"]

    order = ["entropy", "minmax", "percentile_99.9", "mse", "percentile_99.99"]
    order = [m for m in order if m in meth]
    nice = {"entropy": "entropy\n(TRT default)",
            "minmax": "min-max\n(XP10's fix)",
            "percentile_99.9": "percentile\n99.9",
            "mse": "MSE", "percentile_99.99": "percentile\n99.99"}

    fig = plt.figure(figsize=(17.4, 5.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.25, 0.85], wspace=0.30)
    fig.suptitle("Calibration decides almost everything at INT8 — for free, and both ends "
                 "of the sweep are wrong", y=1.06, fontsize=14.5)
    style.subtitle(fig, "Whole network W8A8, per-channel weights, no retraining, "
                        "512 calibration images. Validation split.", y=1.0)

    # ---- panel 1: what each method costs --------------------------------
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(order))
    agg = [meth[m]["map50"] / base_map * 100 for m in order]
    tin = [meth[m]["tiny_plume"] / base_tiny * 100 for m in order]
    ax.bar(x - 0.20, agg, width=0.38, color=style.BLUE, zorder=3, label="aggregate mAP50")
    ax.bar(x + 0.20, tin, width=0.38, color=style.ORANGE, zorder=3,
           label="tiny plumes (<0.1%)")
    ax.axhline(100, color=style.INK_2, ls="--", lw=1.1, zorder=4)
    for i, (a, t) in enumerate(zip(agg, tin)):
        ax.text(i - 0.20, a + 2.5, f"{a:.0f}", ha="center", fontsize=8.6,
                color=style.BLUE, fontweight="bold")
        ax.text(i + 0.20, t + 2.5, f"{t:.0f}", ha="center", fontsize=8.6,
                color=style.ORANGE, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([nice[m] for m in order], fontsize=8.0)
    ax.set_ylim(0, 118)
    ax.set_ylabel("% of the unquantized model kept")
    ax.set_title("1. One setting, 62 points of mAP50", fontsize=12, loc="left")
    # Every arm ships the same bytes, which is what makes the spread free.
    sz = {m: (meth[m].get("size") or {}).get("total_mb") for m in order}
    same = {v for v in sz.values() if v}
    if len(same) == 1:
        fp16_mb = next(iter(meth.values()))["size"]["fp16_baseline_mb"]
        ax.text(0.5, 0.965, f"all five arms ship the same model: "
                            f"{next(iter(same)):.2f} MB, from {fp16_mb:.2f} MB FP16",
                transform=ax.transAxes, ha="center", va="top", fontsize=9.2,
                color=style.INK, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", fc="#eef5ee", ec="#8ec9b4", lw=1.0))
    ax.legend(loc="lower right", fontsize=9)
    style.tidy(ax)

    # ---- panel 2: the mechanism -----------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    layers = list(meth["minmax"]["scales"])
    xs = np.arange(len(layers))
    for m, colour, lw in (("minmax", style.RED, 1.6),
                          ("percentile_99.99", style.AQUA, 1.6),
                          ("entropy", "#8e44ad", 1.3)):
        ax.plot(xs, [meth[m]["scales"][n] for n in layers], color=colour, lw=lw,
                zorder=3, label=nice[m].replace("\n", " — "))
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.07)          # headroom below the lines for the callout + legend
    key = "model.24.m.0"
    if key in meth["minmax"]["scales"]:
        i = layers.index(key)
        ax.scatter([i, i], [meth["minmax"]["scales"][key],
                            meth["percentile_99.99"]["scales"][key]],
                   s=70, facecolor="none", edgecolor=style.INK, linewidth=1.8, zorder=5)
        ax.annotate(f"{key} — the stride-8 head,\nE1's worst layer: "
                    f"{meth['minmax']['scales'][key]:.0f} vs "
                    f"{meth['percentile_99.99']['scales'][key]:.0f}".replace(
                        " — the stride-8 head,\n", "\n(stride-8 head) "),
                    xy=(i, meth["minmax"]["scales"][key]), xytext=(1.5, 0.30),
                    ha="left", va="center", fontsize=8.8, color=style.INK,
                    fontweight="bold", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.35", fc=style.SURFACE, ec="none",
                              alpha=0.93),
                    arrowprops=dict(arrowstyle="->", color=style.INK_2, lw=1.2))
    ax.set_xlabel("the 60 convolutions, in forward order")
    ax.set_ylabel("range the method decides the tensor needs, log scale")
    ax.set_title("2. Why: min-max lets one outlier set the step size\n"
                 "(2.7x wider than percentile on the median layer)",
                 fontsize=12, loc="left")
    ax.legend(loc="lower right", fontsize=8.4)
    style.tidy(ax)

    # ---- panel 3: and how much data it needs ----------------------------
    ax = fig.add_subplot(gs[0, 2])
    sizes = d.get("sizes") or {}
    ns = sorted(int(k) for k in sizes)
    if ns:
        ax.plot(ns, [sizes[str(n)]["map50"] / base_map * 100 for n in ns],
                color=style.BLUE, marker="o", lw=1.8, zorder=3, label="aggregate mAP50")
        ax.plot(ns, [sizes[str(n)]["tiny_plume"] / base_tiny * 100 for n in ns],
                color=style.ORANGE, marker="o", lw=1.8, zorder=3, label="tiny plumes")
        ax.axhline(100, color=style.INK_2, ls="--", lw=1.1, zorder=4)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ns)
        ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
        ax.set_ylim(80, 108)
    ax.set_xlabel("calibration images (nested subsets)")
    ax.set_ylabel("% of the unquantized model kept")
    ax.set_title("3. …and 32 images is enough", fontsize=12, loc="left")
    ax.legend(loc="lower right", fontsize=9)
    style.tidy(ax)

    return save(fig, "xp07e2_calibration.png")




def fig_xp07e4(records) -> Path | None:
    """The board: what INT8 bought, and what TensorRT actually did with it."""
    import matplotlib.pyplot as plt
    import numpy as np

    d = _side("xp07e4_engines.json")
    if not d:
        return None
    arms = d.get("arms") or {}
    order = [a for a in ("fp16", "int8", "int8_fp16", "int8_head_fp16") if a in arms]
    if not order:
        return None
    nice = {"fp16": "FP16\n(the line)", "int8": "INT8\nno float fallback",
            "int8_fp16": "INT8 + FP16\nTRT chooses", "int8_head_fp16": "INT8, head\npinned FP16"}
    base = arms.get("fp16") or {}
    bmap = base.get("map50") or _calib_fp16("map50")
    btiny = base.get("tiny_plume")
    bfps = base.get("fps_batched")

    fig, axes = plt.subplots(1, 3, figsize=(17.2, 5.2))
    fig.suptitle("INT8 on the board: the speed is real, and so is what it costs "
                 "distant smoke", y=1.05, fontsize=14.5)
    style.subtitle(fig, "Full 4,306-image test set at 512 px, batch 16, Jetson Orin Nano "
                        "Super. Every engine built from the same ONNX.", y=0.995)

    x = np.arange(len(order))

    # ---- panel 1: accuracy, aggregate vs distant smoke -------------------
    ax = axes[0]
    agg = [(arms[a].get("map50") or 0) / bmap * 100 if bmap else 0 for a in order]
    tin = [(arms[a].get("tiny_plume") or 0) / btiny * 100 if btiny else 0 for a in order]
    ax.bar(x - 0.20, agg, width=0.38, color=style.BLUE, zorder=3, label="aggregate mAP50")
    ax.bar(x + 0.20, tin, width=0.38, color=style.ORANGE, zorder=3, label="tiny plumes")
    ax.axhline(100, color=style.INK_2, ls="--", lw=1.1, zorder=4)
    for i, (a, t) in enumerate(zip(agg, tin)):
        ax.text(i - 0.20, a + 2, f"{a:.0f}", ha="center", fontsize=8.8,
                color=style.BLUE, fontweight="bold")
        ax.text(i + 0.20, t + 2, f"{t:.0f}", ha="center", fontsize=8.8,
                color=style.ORANGE, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([nice[a] for a in order], fontsize=8.4)
    ax.set_ylim(0, 118)
    ax.set_ylabel("% of the FP16 engine kept")
    ax.set_title("1. What each engine keeps", fontsize=12, loc="left")
    ax.legend(loc="lower right", fontsize=9)
    style.tidy(ax)

    # ---- panel 2: speed, energy, size ------------------------------------
    ax = axes[1]
    fps = [arms[a].get("fps_batched") or 0 for a in order]
    ax.bar(x, fps, width=0.55, color=style.AQUA, zorder=3)
    if bfps:
        ax.axhline(bfps, color=style.INK_2, ls="--", lw=1.1, zorder=4)
    for i, a in enumerate(order):
        r = arms[a]
        j = (r.get("power") or {}).get("j_per_1k")
        ax.text(i, (r.get("fps_batched") or 0) + max(fps) * 0.02,
                f"{r.get('fps_batched', 0):.0f} img/s\n{r.get('engine_mb', 0):.1f} MB"
                + (f"\n{j:.0f} J/1k" if j else ""),
                ha="center", fontsize=8.8, color=style.INK, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([nice[a] for a in order], fontsize=8.4)
    ax.set_ylim(0, max(fps) * 1.34 if fps else 1)
    ax.set_ylabel("throughput, images/s at batch 16")
    ax.set_title("2. What it bought", fontsize=12, loc="left")
    style.tidy(ax)

    # ---- panel 3: what the compiler actually did -------------------------
    ax = axes[2]
    prec_order = ["int8", "fp16", "fp32", "unknown"]
    prec_col = {"int8": "#8e44ad", "fp16": style.BLUE, "fp32": style.MUTED,
                "unknown": "#cccccc"}
    have = False
    bottom = np.zeros(len(order))
    for pk in prec_order:
        vals = []
        for a in order:
            cc = ((arms[a].get("precision_report") or {}).get("convolution_counts") or {})
            vals.append(cc.get(pk, 0))
        if any(vals):
            have = True
            ax.bar(x, vals, bottom=bottom, width=0.55, color=prec_col[pk],
                   zorder=3, label=pk.upper())
            bottom += np.array(vals, dtype=float)
    if have:
        for i, a in enumerate(order):
            rep = arms[a].get("precision_report") or {}
            n8, tot = rep.get("convolutions_in_int8"), rep.get("convolutions_total")
            if tot:
                ax.text(i, bottom[i] + tot * 0.02, f"{n8}/{tot} in INT8",
                        ha="center", fontsize=8.8, color=style.INK, fontweight="bold")
        # Headroom for the legend, so it sits above the bars instead of on the
        # first one's label.
        ax.set_ylim(0, max(bottom) * 1.45)
        ax.legend(loc="upper center", fontsize=9, ncol=3, columnspacing=1.1,
                  title="convolutions ran in")
        ax.set_ylabel("convolutions in the built engine")
        ax.set_title("3. What TensorRT actually did\n"
                     '("INT8" is a request; this is the answer)', fontsize=12, loc="left")
    else:
        ax.text(0.5, 0.5, "per-layer precision not recorded\n"
                          "(engines need detailed profiling verbosity)",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=10, color=style.MUTED)
        ax.set_title("3. What TensorRT actually did", fontsize=12, loc="left")
    ax.set_xticks(x); ax.set_xticklabels([nice[a] for a in order], fontsize=8.4)
    style.tidy(ax)

    fig.tight_layout()
    return save(fig, "xp07e4_engines.png")


def _calib_fp16(key: str):
    """The XP9 line, for when an E4 run skipped its own FP16 arm."""
    return {"map50": 0.7776, "tiny_plume": 0.1376, "fps_batched": 474.0}.get(key)




def fig_xp07e5(records) -> Path | None:
    """The size is free and the speed is not: weights cost nothing, activations cost it all."""
    import matplotlib.pyplot as plt
    import numpy as np

    d = _side("xp07e5_targets.json")
    if not d:
        return None
    bl = d["baseline_fp16"]
    bmap, btiny = bl["map50"], bl["tiny_plume"]["map50"]
    rows = {r["arm"]: r for r in d["rows"]}
    order = [a for a in ("w8_only", "a8_only", "w8a8") if a in rows]
    nice = {"w8_only": "W8\nweights only", "a8_only": "A8\nactivations only\n(control)",
            "w8a8": "W8A8\nboth"}

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.8),
                             gridspec_kw={"width_ratios": [1.15, 1]})
    fig.suptitle("Quantizing the weights is free. Quantizing the activations is the whole cost.",
                 y=1.05, fontsize=14)
    style.subtitle(fig, "Whole network, min-max calibration, no retraining. Validation split, "
                        f"{d['n_val_images']} images.", y=0.995)

    # ---- panel 1: what each target keeps, and what it weighs -------------
    ax = axes[0]
    x = np.arange(len(order) + 1)
    agg = [100] + [rows[a]["damage"]["map50"] / bmap * 100 for a in order]
    tin = [100] + [rows[a]["damage"]["tiny_plume"] / btiny * 100 for a in order]
    mb = [(rows[order[0]].get("size") or {}).get("fp16_baseline_mb")] + \
         [(rows[a].get("size") or {}).get("total_mb") for a in order]
    ax.bar(x - 0.20, agg, width=0.38, color=style.BLUE, zorder=3, label="aggregate mAP50")
    ax.bar(x + 0.20, tin, width=0.38, color=style.ORANGE, zorder=3, label="tiny plumes")
    ax.axhline(100, color=style.INK_2, ls="--", lw=1.1, zorder=4)
    for i, (a, t, m) in enumerate(zip(agg, tin, mb)):
        ax.text(i - 0.20, a + 2, f"{a:.0f}", ha="center", fontsize=9,
                color=style.BLUE, fontweight="bold")
        ax.text(i + 0.20, t + 2, f"{t:.0f}", ha="center", fontsize=9,
                color=style.ORANGE, fontweight="bold")
        if m:
            ax.text(i, 112, f"{m:.2f} MB", ha="center", fontsize=9.5,
                    color=style.INK, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", fc="#eef2f6", ec="none"))
    ax.set_xticks(x)
    ax.set_xticklabels(["FP16\nunquantized"] + [nice[a] for a in order], fontsize=8.6)
    ax.set_ylim(0, 124)
    ax.set_ylabel("% of the unquantized model kept")
    ax.set_title("1. What each target costs — and what it weighs", fontsize=12, loc="left")
    ax.legend(loc="lower left", fontsize=9)
    style.tidy(ax)

    # ---- panel 2: the decomposition --------------------------------------
    ax = axes[1]
    att = d.get("attribution") or {}
    parts = [("weights alone", att.get("cost_of_weights_pct"), style.AQUA),
             ("activations alone", att.get("cost_of_activations_pct"), style.RED),
             ("interaction", att.get("interaction_pct"), style.MUTED)]
    parts = [(n, v, c) for n, v, c in parts if v is not None]
    labels = [n for n, _, _ in parts][::-1]
    vals = [v for _, v, _ in parts][::-1]
    cols = [c for _, _, c in parts][::-1]
    ax.barh(labels, vals, color=cols, zorder=3, height=0.5)
    total = att.get("cost_of_both_pct")
    if total:
        ax.axvline(total, color=style.INK_2, ls="--", lw=1.2, zorder=4)
        ax.text(total, len(vals) - 0.75, f"both together\n{total:.3f}%", fontsize=9,
                color=style.INK_2, ha="center", va="top", fontweight="bold")
    for i, v in enumerate(vals):
        ax.text(v + total * 0.03, i, f"{v:+.3f}%", va="center", fontsize=10.5,
                color=style.INK, fontweight="bold")
    ax.set_xlim(0, total * 1.42 if total else 1)
    ax.set_ylim(-0.6, len(vals) - 0.1)
    ax.set_xlabel("mAP50 given up, % of the unquantized model")
    ax.set_title("2. Where the loss actually comes from\n"
                 "(the two effects are additive — no conspiracy)", fontsize=12, loc="left")
    style.tidy(ax, ygrid=False)

    fig.tight_layout()
    return save(fig, "xp07e5_targets.png")




def fig_xp07e3(records) -> Path | None:
    """The granularity everyone tunes is worth nothing; the flag nobody mentions is worth 31x."""
    import matplotlib.pyplot as plt
    import numpy as np

    d = _side("xp07e3_granularity.json")
    if not d:
        return None
    bl = d["baseline_fp16"]
    bmap = bl["map50"]
    rows = {r["arm"]: r for r in d["rows"]}
    order = [a for a in ("per_tensor_sym", "per_channel_sym",
                         "per_channel_asym", "per_tensor_asym") if a in rows]
    nice = {"per_tensor_sym": "per-tensor\nsymmetric",
            "per_channel_sym": "per-channel\nsymmetric",
            "per_channel_asym": "per-channel\nasymmetric",
            "per_tensor_asym": "per-tensor\nasymmetric"}

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.9),
                             gridspec_kw={"width_ratios": [1.2, 1]})
    fig.suptitle("The scale count nobody needs, and the zero point nobody mentions",
                 y=1.05, fontsize=14)
    style.subtitle(fig, "Whole network W8A8, min-max calibration, no retraining. Weights are "
                        "symmetric in every arm.", y=0.995)

    # ---- panel 1: the four arms -----------------------------------------
    ax = axes[0]
    x = np.arange(len(order))
    vals = [rows[a]["damage"]["map50"] / bmap * 100 for a in order]
    cols = [style.MUTED if "sym" in a and "asym" not in a else style.AQUA for a in order]
    ax.bar(x, vals, width=0.58, color=cols, zorder=3)
    ax.axhline(100, color=style.INK_2, ls="--", lw=1.1, zorder=4)
    for i, a in enumerate(order):
        sz = rows[a].get("size") or {}
        ax.text(i, vals[i] + 0.35,
                f"{rows[a]['damage']['map50']:.4f}\n{sz.get('weight_scales', 0):,} scales\n"
                f"{sz.get('total_mb', 0):.3f} MB",
                ha="center", fontsize=8.8, color=style.INK, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([nice[a] for a in order], fontsize=9)
    ax.set_ylim(88, 104)
    ax.set_ylabel("% of the unquantized mAP50 kept")
    ax.set_title("1. Four settings, one that matters", fontsize=12, loc="left")
    style.tidy(ax)

    # ---- panel 2: the two decisions, side by side ------------------------
    ax = axes[1]
    dl = d.get("deltas") or {}
    pairs = [("per-channel vs\nper-tensor weights",
              dl.get("per_channel_minus_per_tensor"), style.MUTED),
             ("asymmetric vs symmetric\nactivations",
              dl.get("asymmetric_minus_symmetric_acts"), style.AQUA)]
    pairs = [(n, v, c) for n, v, c in pairs if v is not None]
    names = [n for n, _, _ in pairs]
    vals2 = [v for _, v, _ in pairs]
    ax.barh(names, vals2, color=[c for _, _, c in pairs], zorder=3, height=0.42)
    for i, v in enumerate(vals2):
        ax.text(v + max(vals2) * 0.03, i, f"{v:+.4f} mAP50", va="center",
                fontsize=11.5, color=style.INK, fontweight="bold")
    if len(vals2) == 2 and vals2[0]:
        ax.text(max(vals2) * 0.52, 0.5, f"{vals2[1] / vals2[0]:.0f}x", ha="center",
                fontsize=17, color=style.AQUA, fontweight="bold")
    ax.set_xlim(0, max(vals2) * 1.5)
    ax.set_xlabel("mAP50 gained")
    ax.set_title("2. What each decision is actually worth", fontsize=12, loc="left")
    style.tidy(ax, ygrid=False)

    fig.tight_layout()
    return save(fig, "xp07e3_granularity.png")




def fig_xp07e6(records) -> Path | None:
    """Three of sixty layers, chosen by measurement, recover the distant-smoke loss."""
    import matplotlib.pyplot as plt
    import numpy as np

    d = _side("xp07e6_mixed.json")
    dt = _side("xp07e6_mixed_tiny.json")
    e5 = _side("xp07e5_targets.json")
    if not d:
        return None
    bl = d["baseline_fp16"]
    bmap, btiny = bl["map50"], bl["tiny_plume"]["map50"]
    rows = {r["arm"]: r for r in d["rows"]}

    fig, axes = plt.subplots(1, 2, figsize=(15.4, 5.0),
                             gridspec_kw={"width_ratios": [1, 1.25]})
    fig.suptitle("The damage has an address: three of sixty convolutions",
                 y=1.05, fontsize=14.5)
    style.subtitle(fig, "Whole network W8A8, min-max calibration, no retraining. The layers left "
                        "in FP16 were read out of E1's map, not chosen by hand.", y=0.995)

    # ---- panel 1: the decode cliff --------------------------------------
    ax = axes[0]
    order = [a for a in ("uniform", "head_out", "head_out_decode_out") if a in rows]
    nice = {"uniform": "all 60 INT8\ndecode INT8",
            "head_out": "57 INT8\ndecode INT8",
            "head_out_decode_out": "57 INT8\ndecode FLOAT"}
    vals = [rows[a]["damage"]["map50"] / bmap * 100 for a in order]
    cols = [style.RED if v < 1 else style.AQUA for v in vals]
    ax.bar(np.arange(len(order)), vals, width=0.55, color=cols, zorder=3)
    ax.axhline(100, color=style.INK_2, ls="--", lw=1.1, zorder=4)
    for i, (a, v) in enumerate(zip(order, vals)):
        ax.text(i, v + 2.5, f"{rows[a]['damage']['map50']:.4f}", ha="center",
                fontsize=10.5, color=style.INK, fontweight="bold")
    ax.text(0.5, 46, "quantizing the decode output\ndoes not degrade the detector —\n"
                     "it switches it off", ha="center", fontsize=10, color=style.RED,
            fontweight="bold")
    dq = rows[order[0]].get("decode_output_scale") or {}
    if dq:
        ax.text(0.5, 26, f"one INT8 step = {dq['scale']:.2f}\n"
                         f"a probability gets {1/dq['scale']:.3f} of a level",
                ha="center", fontsize=9, color=style.INK_2)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([nice[a] for a in order], fontsize=9)
    ax.set_ylim(0, 118)
    ax.set_ylabel("% of the unquantized mAP50 kept")
    ax.set_title("1. Protecting the head convolutions does not help.\n"
                 "Protecting the tensor they feed does.", fontsize=12, loc="left")
    style.tidy(ax)

    # ---- panel 2: which three, and what it is worth ----------------------
    ax = axes[1]
    arms = []
    if e5:
        w = next((r for r in e5["rows"] if r["arm"] == "w8a8"), None)
        if w:
            arms.append(("nothing protected\n(all 60 INT8)", w["damage"], style.MUTED))
    arms.append(("3 layers, ranked by\naggregate mAP50",
                 rows["head_out_decode_out"]["damage"], style.BLUE))
    if dt:
        r = next((x for x in dt["rows"] if x["arm"] == "head_out_decode_out"), None)
        if r:
            arms.append(("3 layers, ranked by\ntiny-plume damage", r["damage"], style.AQUA))

    x = np.arange(len(arms))
    agg = [a[1]["map50"] / bmap * 100 for a in arms]
    tin = [a[1]["tiny_plume"] / btiny * 100 for a in arms]
    ax.bar(x - 0.20, agg, width=0.38, color=style.BLUE, zorder=3, label="aggregate mAP50")
    ax.bar(x + 0.20, tin, width=0.38, color=style.ORANGE, zorder=3,
           label="tiny plumes (<0.1%)")
    ax.axhline(100, color=style.INK_2, ls="--", lw=1.1, zorder=4)
    for i, (a, t) in enumerate(zip(agg, tin)):
        ax.text(i - 0.20, a + 2.5, f"{a:.0f}", ha="center", fontsize=9.5,
                color=style.BLUE, fontweight="bold")
        ax.text(i + 0.20, t + 2.5, f"{t:.0f}", ha="center", fontsize=9.5,
                color=style.ORANGE, fontweight="bold")
    # A straight annotation in the empty band above the bars; the earlier arc
    # crossed the middle bar's own label.
    ax.annotate("", xy=(len(arms) - 1 + 0.20, tin[-1] + 6), xytext=(0.20, tin[0] + 6),
                arrowprops=dict(arrowstyle="->", color=style.ORANGE, lw=1.8,
                                connectionstyle="arc3,rad=-0.34"))
    ax.text(len(arms) / 2 - 0.5, 140,
            f"{tin[-1] / tin[0]:.1f}x the distant-smoke accuracy, for 29 KB",
            ha="center", fontsize=10.5, color=style.ORANGE, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([a[0] for a in arms], fontsize=9)
    ax.set_ylim(0, 156)
    ax.set_ylabel("% of the unquantized model kept")
    ax.set_title("2. Which three layers — and it is a real decision", fontsize=12, loc="left")
    ax.legend(loc="upper left", fontsize=9)
    style.tidy(ax)

    fig.tight_layout()
    return save(fig, "xp07e6_mixed.png")




def fig_xp07e9b(records) -> Path | None:
    """Within quantization: which decision matters, and what does each one cost?"""
    import matplotlib.pyplot as plt
    import numpy as np

    d = _side("xp07e9b_quant.json")
    if not d:
        return None
    sim = d["simulated"]["rows"]
    eng = d["engines"]["rows"]
    ranking = d["axis_ranking"]
    if not sim or not ranking:
        return None

    axis_colour = {"range": style.RED, "target": "#8e44ad",
                   "bit-width": style.ORANGE, "granularity": style.MUTED,
                   "composition": style.BLUE, "engine": style.AQUA, "—": style.INK}
    nice = {"range": "range\n(calibration)", "target": "target\n(what is quantized)",
            "bit-width": "bit-width", "granularity": "granularity\n(scale count)"}

    fig = plt.figure(figsize=(17.0, 6.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.95, 1.5, 1.15], wspace=0.34)
    fig.suptitle("Within quantization, the decisions are not equal — and the one that "
                 "costs nothing matters most", y=1.04, fontsize=14.5)
    style.subtitle(fig, "Left and middle: fake-quant on the 1,721-image validation split. "
                        "Right: TensorRT engines on the 4,306-image test split. Each block "
                        "is normalised to its own unquantized baseline, because the two "
                        "splits are 17 points apart.", y=0.975)

    # ---- panel 1: how much each axis is worth ---------------------------
    ax = fig.add_subplot(gs[0, 0])
    order = list(reversed(ranking))
    names = [nice.get(r["axis"], r["axis"]) for r in order]
    vals = [r["map50_spread_pct"] for r in order]
    tiny = [r["tiny_spread_pct"] for r in order]
    y = np.arange(len(order))
    ax.barh(y - 0.19, vals, height=0.36, color=style.BLUE, zorder=3, label="aggregate mAP50")
    ax.barh(y + 0.19, tiny, height=0.36, color=style.ORANGE, zorder=3, label="tiny plumes")
    for i, (v, t) in enumerate(zip(vals, tiny)):
        ax.text(v + 3, i - 0.19, f"{v:.0f}", va="center", fontsize=9.5,
                color=style.BLUE, fontweight="bold")
        ax.text(t + 3, i + 0.19, f"{t:.0f}", va="center", fontsize=9.5,
                color=style.ORANGE, fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9.5)
    ax.set_xlim(0, max(max(vals), max(tiny)) * 1.28)
    ax.set_xlabel("spread across that axis's arms,\n% of the unquantized model")
    ax.set_title("1. What each decision is worth", fontsize=12, loc="left")
    ax.legend(loc="lower right", fontsize=9)
    style.tidy(ax, ygrid=False)

    # ---- panel 2: every simulated arm, accuracy vs distant smoke --------
    ax = fig.add_subplot(gs[0, 1])
    for r in sim:
        if r["kept_pct"] is None or r["tiny_kept_pct"] is None:
            continue
        c = axis_colour.get(r["axis"], style.MUTED)
        ax.scatter([r["kept_pct"]], [r["tiny_kept_pct"]], s=95, color=c, alpha=0.85,
                   zorder=4, edgecolor="white", linewidth=1.1)
    ax.axhline(100, color=style.INK_2, ls="--", lw=1.0, zorder=2)
    ax.axvline(100, color=style.INK_2, ls="--", lw=1.0, zorder=2)
    ax.plot([0, 135], [0, 135], color=style.MUTED, ls=":", lw=1.0, zorder=1)
    ax.text(52, 44, "below this line, distant smoke\nsuffers more than the headline",
            fontsize=8.6, color=style.MUTED, rotation=34, ha="center")
    for r in sim:
        lab = None
        if r["label"].startswith("calibration: minmax"):
            lab = "min-max\n(what we shipped)"
        elif r["label"].startswith("calibration: percentile_99.99"):
            lab = "percentile 99.99"
        elif "head_out_decode_out" in r["label"] and "tiny" in r["label"]:
            lab = "3 convs in FP16"
        elif r["label"].startswith("target: w8_only"):
            lab = "weights only"
        if lab and r["kept_pct"] is not None:
            # "weights only" and "percentile 99.99" both sit at (~100, ~97), so
            # they are pushed to opposite sides rather than stacked.
            off = (-9, 16) if lab.startswith("weights") else (-9, -18)
            if lab.startswith("3 convs"):
                off = (-9, 8)
            elif lab.startswith("min-max"):
                off = (-10, 4)
            ax.annotate(lab, (r["kept_pct"], r["tiny_kept_pct"]),
                        textcoords="offset points", xytext=off, ha="right",
                        fontsize=8.8, fontweight="bold",
                        color=axis_colour.get(r["axis"], style.INK))
    ax.set_xlim(25, 108); ax.set_ylim(-5, 140)
    ax.set_xlabel("aggregate mAP50 kept, %")
    ax.set_ylabel("tiny-plume mAP50 kept, %")
    ax.set_title("2. Every quantization arm measured\n"
                 "(the arms at 0 quantize the box-decode output)", fontsize=12, loc="left")
    handles = [plt.Line2D([], [], marker="o", ls="", color=axis_colour[a], markersize=8,
                          label=nice.get(a, a).replace("\n", " "))
               for a in ("range", "target", "granularity", "bit-width", "composition")
               if any(r["axis"] == a for r in sim)]
    ax.legend(handles=handles, loc="lower left", fontsize=8.6)
    style.tidy(ax)

    # ---- panel 3: what survived into an engine --------------------------
    ax = fig.add_subplot(gs[0, 2])
    eng = [r for r in eng if r.get("fps_batched")]
    short = {"engine: fp16": "FP16 (the line)", "engine: int8": "INT8",
             "engine: int8_fp16": "INT8 + FP16", "engine: int8_head_fp16": "INT8, head pinned",
             "engine: qdq_percentile9999_pc": "path A: percentile, 8 imgs",
             "engine: XP10 INT8 min-max": "XP10 INT8 min-max",
             "engine: XP10 INT8 entropy": "XP10 INT8 entropy"}
    # The three path-B INT8 arms land on the same point to within a pixel
    # (712-720 img/s, 30.6-30.7% tiny), so they are drawn once and labelled as a
    # group. Printing three labels there just overprints them.
    groups, used = [], set()
    for i, r in enumerate(eng):
        if i in used:
            continue
        near = [j for j in range(i, len(eng))
                if j not in used
                and abs(eng[j]["fps_batched"] - r["fps_batched"]) < 12
                and abs((eng[j]["tiny_kept_pct"] or 0) - (r["tiny_kept_pct"] or 0)) < 1.5]
        used.update(near)
        groups.append([eng[j] for j in near])

    for g in groups:
        r = g[0]
        lab = r["label"]
        if len(g) > 1:
            lab = f"{len(g)} path-B INT8 arms\n(identical to 0.1%)"
            c = style.AQUA
        else:
            lab = short.get(lab, lab)
            c = (style.INK if lab.startswith("FP16") else
                 style.RED if "entropy" in lab else
                 "#8e44ad" if "path A" in lab else style.AQUA)
        ax.scatter([r["fps_batched"]], [r["tiny_kept_pct"]], s=110, color=c,
                   alpha=0.85, zorder=4, edgecolor="white", linewidth=1.1)
        dy = 13 if r["tiny_kept_pct"] < 60 else -22
        ax.annotate(lab, (r["fps_batched"], r["tiny_kept_pct"]),
                    textcoords="offset points", xytext=(0, dy), ha="center",
                    fontsize=8.6, color=style.INK, fontweight="bold")

    ax.axhline(100, color=style.INK_2, ls="--", lw=1.0, zorder=2)
    ax.set_ylim(-8, 125)
    ax.set_xlabel("throughput, images/s at batch 16")
    ax.set_ylabel("tiny-plume mAP50 kept, %")
    ax.set_title("3. What reached an engine\n"
                 "(no engine keeps more than 46% of distant smoke)",
                 fontsize=12, loc="left")
    style.tidy(ax)

    return save(fig, "xp07e9b_quant.png")


BUILDERS = [fig_xp00, fig_xp01, fig_xp02, fig_xp06, fig_xp09, fig_xp10,
            fig_xp12, fig_xp06e1, fig_xp06e2, fig_xp06e3, fig_xp06e4, fig_xp06e4b, fig_xp06e5, fig_xp06e7b, fig_xp06e9, fig_xp15, fig_xp15_confusion,
            fig_xp06e6,
            fig_xp06e7,
            fig_xp07_concepts, fig_xp07_head, fig_xp07e9,
            fig_xp07e9_fire, fig_xp07e9_tiny, fig_xp07e1, fig_xp07e2, fig_xp07e4, fig_xp07e5, fig_xp07e3, fig_xp07e6, fig_xp07e9b]


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
