#!/usr/bin/env python3
"""Figures made of actual imagery — what the data looks like, and what the model
sees as you starve it of pixels.

Runs on the Jetson (it needs the dataset and the model). Two outputs:

* ``dataset_examples.png`` — six representative test frames with ground truth
  drawn on, chosen programmatically rather than cherry-picked by eye: the largest
  fire, the largest smoke plume, the darkest labelled frame, the lowest-contrast
  labelled frame, a genuinely tiny plume, and an empty landscape.
* ``xp02_resolution_visual.png`` — one frame run through the detector at 640, 512,
  416 and 320 pixels, with what it found at each. The image is picked by searching
  for a frame the detector gets right at 640 and loses by 320, which is the failure
  the numbers in XP2 describe.

Both state their selection rule in the caption, because "representative example"
is otherwise unfalsifiable.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from analysis import style              # noqa: E402
from lib import data as dataset         # noqa: E402
from lib import evaluator               # noqa: E402
from lib.detectors import YOLOV5_REPO as YOLOV5_REPO_PATH   # noqa: E402

FIGURES = REPO / "results" / "figures"
RESOLUTIONS = [640, 512, 416, 320]

FIRE, SMOKE = "#e34948", "#2a78d6"      # box colours: fire warm, smoke cool



def _mast_frames(samples):
    """Prefer fixed-camera frames (the `AoF` and `PublicDataset` sources).

    Two reasons, both deliberate. They are the deployment domain this project is
    about — a camera bolted to a mast looking at terrain — and they are wide,
    consistent, and free of identifiable people, which the web-scraped `WEB`
    images are not. Publishing a stranger's face to illustrate a threshold is not
    a trade worth making.
    """
    mast = [s for s in samples if not Path(s.rel).name.startswith("WEB")]
    return mast or samples


def _load_rgb(path: Path):
    import cv2
    im = cv2.imread(str(path))
    return im[..., ::-1]                 # BGR -> RGB


def _draw_boxes(ax, boxes, w, h, *, gt=True, scores=None, classes=None):
    """Ground truth as solid, predictions as solid with a confidence tag."""
    from matplotlib.patches import Rectangle

    for i, b in enumerate(boxes):
        if gt:
            x0, y0, x1, y1 = b.to_xyxy(w, h)
            cls = dataset.CLASS_NAMES[b.cls]
        else:
            x0, y0, x1, y1 = b
            cls = dataset.CLASS_NAMES[classes[i]]
        colour = FIRE if cls == "fire" else SMOKE
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                               edgecolor=colour, linewidth=2.0))
        # Clamp the label inside the frame: a box near the right edge otherwise
        # pushes its tag outside the axes and into the neighbouring panel.
        tag = cls if gt else f"{cls} {scores[i]:.2f}"
        tx = min(max(x0, 2), max(w - 0.16 * w, 2))
        ha = "left" if tx < w * 0.84 else "right"
        if ha == "right":
            tx = min(x1, w - 2)
        ax.text(tx, max(y0 - 4, 12), tag, color="white", fontsize=8, ha=ha,
                clip_on=False,
                bbox=dict(fc=colour, ec="none", pad=1.4, alpha=0.95))


def dataset_examples(samples) -> Path:
    """Six frames covering the range the detector has to cope with.

    Selected by rule, and restricted to 16:9 fixed-camera frames so the grid is
    uniform and the panels are comparable. An earlier version selected "darkest"
    and "lowest-contrast" separately and got two near-identical black rectangles;
    the bands below are chosen to be visually distinct as well as statistically.
    """
    import matplotlib.pyplot as plt
    from PIL import Image

    samples = _mast_frames(samples)

    def wide(s):
        with Image.open(s.image) as im:
            w, h = im.size
        return 1.6 <= w / h <= 1.85

    pool = [s for s in samples if wide(s)]
    labelled = [s for s in pool if s.boxes]

    def biggest(cls_name):
        best, area = None, -1
        for s in labelled:
            for b in s.boxes:
                if dataset.CLASS_NAMES[b.cls] == cls_name and b.area_frac > area:
                    best, area = s, b.area_frac
        return best

    def in_band(lo, hi):
        band = [s for s in labelled if lo <= min(b.area_frac for b in s.boxes) < hi]
        return max(band, key=lambda s: min(b.area_frac for b in s.boxes)) if band else None

    stats = {s.rel: dataset.image_stats(s.image) for s in labelled[::5]}
    night = min((s for s in labelled if s.rel in stats), key=lambda s: stats[s.rel]["mean"])
    hazy = min((s for s in labelled if s.rel in stats and stats[s.rel]["mean"] > 90),
               key=lambda s: stats[s.rel]["std"], default=None)
    empty = next(s for s in pool if s.is_background)

    picks = [(biggest("fire"), "large fire — the easy case"),
             (biggest("smoke"), "large smoke plume"),
             (in_band(0.002, 0.01), "small plume — under 1% of the frame"),
             (in_band(0.0, 0.001), "tiny plume — under 0.1% of the frame"),
             (night or hazy, "night: most of the frame is black"),
             (empty, "empty landscape — 47% of the test set")]
    picks = [(s, c) for s, c in picks if s is not None]

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 5.6))
    fig.suptitle("What the detector is looking at", y=1.04, fontsize=14, fontweight="bold")
    style.subtitle(fig, "Ground-truth boxes shown. Fixed cameras watching terrain — the "
                        "deployment case — not drone footage. Frames chosen by rule.", y=1.0)
    for ax, (s, caption) in zip(axes.ravel(), picks):
        im = _load_rgb(s.image)
        h, w = im.shape[:2]
        ax.imshow(im)
        _draw_boxes(ax, s.boxes, w, h, gt=True)
        ax.set_title(caption, fontsize=10, color=style.INK_2, pad=5)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    for ax in axes.ravel()[len(picks):]:
        ax.set_visible(False)
    fig.tight_layout()
    out = FIGURES / "dataset_examples.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out


def resolution_visual(samples) -> Path | None:
    """Find a frame the detector holds at 640 and drops by 320, then show all four."""
    import matplotlib.pyplot as plt
    from lib.detectors import Yolov5Detector

    weights = REPO / "weights" / "yolov5s.pt"
    dets = {r: Yolov5Detector(weights, input_res=r, half=True) for r in RESOLUTIONS}

    # The figure has to show the failure XP2 measured: a genuinely small target
    # that survives at full resolution and is gone by 320 px. Matching on
    # "fewer boxes overall" is not that — it selected a huge night fire whose
    # count merely wobbled. So: match detections against the smallest ground-truth
    # box by IoU, and require that match to exist at 640 and vanish at 320.
    samples = _mast_frames(samples)

    def iou(a, b):
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        ix0, iy0 = max(ax0, bx0), max(ay0, by0)
        ix1, iy1 = min(ax1, bx1), min(ay1, by1)
        inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
        if inter <= 0:
            return 0.0
        ua = (ax1-ax0)*(ay1-ay0) + (bx1-bx0)*(by1-by0) - inter
        return inter / ua if ua > 0 else 0.0

    def hits_small_gt(det, s, thr=0.30):
        """Does the detector find the smallest labelled target in this frame?"""
        import cv2
        im = cv2.imread(str(s.image))
        h, w = im.shape[:2]
        gt = min(s.boxes, key=lambda b: b.area_frac).to_xyxy(w, h)
        d = det.predict([s.image])[0]
        return any(sc >= thr and iou(gt, xy) >= 0.3 for xy, sc in zip(d.xyxy, d.scores))

    cands = [s for s in samples
             if s.boxes and 1 <= len(s.boxes) <= 3
             and 0.0004 < min(b.area_frac for b in s.boxes) < 0.006]
    # Daylight only: a dark frame renders as a black rectangle on a README page.
    bright = []
    for s in cands[:400]:
        if dataset.image_stats(s.image)["mean"] > 70:
            bright.append(s)
        if len(bright) >= 120:
            break

    chosen = None
    for s in bright:
        if hits_small_gt(dets[640], s) and not hits_small_gt(dets[320], s):
            chosen = s
            break
    chosen = chosen or (bright[0] if bright else cands[0])
    print(f"  resolution_visual frame: {chosen.rel} "
          f"(smallest target {min(b.area_frac for b in chosen.boxes)*100:.3f}% of frame)",
          flush=True)

    # Render what the MODEL receives, not the original photo. An earlier version
    # drew the full-resolution frame in all four panels and changed only the boxes,
    # so the panels looked identical and the figure silently made the opposite of
    # its point. Each panel below is the actual letterboxed network input at its
    # true pixel count, upscaled with nearest-neighbour for display so the lost
    # detail is visible rather than smoothed back in by the renderer.
    import cv2
    import numpy as np
    sys.path.insert(0, str(YOLOV5_REPO_PATH))
    from utils.augmentations import letterbox

    im0 = cv2.imread(str(chosen.image))
    h0, w0 = im0.shape[:2]
    gt_box = min(chosen.boxes, key=lambda b: b.area_frac)
    gx0, gy0, gx1, gy1 = gt_box.to_xyxy(w0, h0)

    fig, axes = plt.subplots(2, 4, figsize=(15.5, 7.0),
                             gridspec_kw={"height_ratios": [2.05, 1.0]})
    fig.suptitle("The same frame as the model actually receives it", y=1.02,
                 fontsize=14, fontweight="bold")
    style.subtitle(fig, "Top: the network input at each resolution. Bottom: the target "
                        "magnified. Fewer pixels reach the model each step — by 320 px the "
                        "plume is ~12 px across and is missed.", y=0.975)

    for col, res in enumerate(RESOLUTIONS):
        lb, ratio, (dw, dh) = letterbox(im0, (res, res), stride=32, auto=False)
        disp = lb[..., ::-1]                       # BGR -> RGB, res x res
        # Ground truth mapped into letterbox space.
        bx0, by0 = gx0 * ratio[0] + dw, gy0 * ratio[1] + dh
        bx1, by1 = gx1 * ratio[0] + dw, gy1 * ratio[1] + dh
        side = max(bx1 - bx0, by1 - by0)

        d = dets[res].predict([chosen.image])[0]
        keep = [i for i, sc in enumerate(d.scores) if sc >= 0.35]
        found_small = any(iou((gx0, gy0, gx1, gy1), d.xyxy[i]) >= 0.3 for i in keep)

        ax = axes[0, col]
        ax.imshow(disp, interpolation="nearest")
        for i in keep:                              # predictions, in letterbox space
            px0, py0, px1, py1 = d.xyxy[i]
            from matplotlib.patches import Rectangle
            ax.add_patch(Rectangle((px0 * ratio[0] + dw, py0 * ratio[1] + dh),
                                   (px1 - px0) * ratio[0], (py1 - py0) * ratio[1],
                                   fill=False, edgecolor=SMOKE
                                   if dataset.CLASS_NAMES[d.classes[i]] == "smoke" else FIRE,
                                   linewidth=1.6))
        colour = style.INK if found_small else "#d03b3b"
        ax.set_title(f"{res} × {res} px input\n"
                     f"{'target found' if found_small else 'target MISSED'}",
                     fontsize=11, color=colour, fontweight="bold", pad=6)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

        # Bottom row: the target region magnified, same crop in every panel.
        axz = axes[1, col]
        cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
        half = max(side * 3.0, 26)
        zx0, zx1 = int(max(0, cx - half)), int(min(res, cx + half))
        zy0, zy1 = int(max(0, cy - half)), int(min(res, cy + half))
        axz.imshow(disp[zy0:zy1, zx0:zx1], interpolation="nearest")
        axz.set_title(f"target ≈ {side:.0f} × {side:.0f} px", fontsize=9.5,
                      color=style.INK_2, pad=4)
        axz.set_xticks([]); axz.set_yticks([])
        for sp in axz.spines.values():
            sp.set_edgecolor(style.GRID)

    fig.tight_layout()
    out = FIGURES / "xp02_resolution_visual.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out


def plume_definition(samples) -> Path:
    """What "small plume" and "tiny plume" actually mean, in pixels.

    The terms carry every size-sensitive finding in this repo, so they get a
    picture rather than a threshold in prose. Each panel is a real test frame
    whose smallest box falls in that band, with the box measured.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    samples = _mast_frames(samples)

    def pick(lo, hi):
        best = None
        for s in samples:
            if not s.boxes:
                continue
            b = min(s.boxes, key=lambda b: b.area_frac)
            if lo <= b.area_frac < hi and (best is None or b.area_frac > best[1].area_frac):
                best = (s, b)
        return best

    bands = [(pick(0.05, 0.45), "obvious", "5-45% of the frame"),
             (pick(0.002, 0.01), "small plume", "under 1% of the frame"),
             (pick(0.0, 0.001), "tiny plume", "under 0.1% of the frame")]
    bands = [(p, t, c) for p, t, c in bands if p]

    fig, axes = plt.subplots(1, len(bands), figsize=(4.6 * len(bands), 4.4))
    fig.suptitle('What "small" and "tiny" plume mean', y=1.05,
                 fontsize=14, fontweight="bold")
    style.subtitle(fig, "The target is the same object; only its size on screen changes. "
                        "Compression damages the right-hand case first.", y=0.995)

    for ax, ((s, b), title, caption) in zip(np_atleast(axes), bands):
        im = _load_rgb(s.image)
        h, w = im.shape[:2]
        ax.imshow(im)
        x0, y0, x1, y1 = b.to_xyxy(w, h)
        colour = FIRE if dataset.CLASS_NAMES[b.cls] == "fire" else SMOKE
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                               edgecolor=colour, linewidth=2.4))
        # A zoom inset: at 15x15 px on a 1280-wide frame the target is otherwise
        # invisible on a README page, which would defeat the point of the figure.
        if b.area_frac < 0.02:
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            half = max((x1 - x0), (y1 - y0)) * 3.5 + 18
            ix0, ix1 = max(0, cx - half), min(w, cx + half)
            iy0, iy1 = max(0, cy - half), min(h, cy + half)
            inset = ax.inset_axes([0.62, 0.62, 0.36, 0.36])
            inset.imshow(im)
            inset.set_xlim(ix0, ix1); inset.set_ylim(iy1, iy0)
            inset.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                      edgecolor=colour, linewidth=1.8))
            inset.set_xticks([]); inset.set_yticks([])
            for sp2 in inset.spines.values():
                sp2.set_edgecolor(style.INK); sp2.set_linewidth(1.4)
            ax.indicate_inset_zoom(inset, edgecolor=style.INK, alpha=0.35, linewidth=0.9)
        px = (x1 - x0) * (y1 - y0)
        ax.set_title(f"{title}\n{caption} · {b.area_frac*100:.2f}% "
                     f"(≈{int(px**0.5)}×{int(px**0.5)} px)",
                     fontsize=10.5, color=style.INK, pad=6)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.tight_layout()
    out = FIGURES / "plume_definition.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out


def np_atleast(axes):
    try:
        return list(axes)
    except TypeError:
        return [axes]


def main() -> None:
    style.apply()
    FIGURES.mkdir(parents=True, exist_ok=True)
    samples = dataset.load_samples("test")
    print("plume_definition:", plume_definition(samples), flush=True)
    print("dataset_examples:", dataset_examples(samples), flush=True)
    print("resolution_visual:", resolution_visual(samples), flush=True)


if __name__ == "__main__":
    main()
