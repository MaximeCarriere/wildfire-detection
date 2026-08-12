"""Datasets, frozen splits, OOD quarantine, calibration set, small-plume tags.

PLAN.md §0/§1 in code. The rules this module exists to enforce:

* **Splits are frozen once, in XP0**, as explicit file lists committed to
  ``data/splits/``. Nothing downstream re-derives them; every XP calls
  :func:`load_split`. Lists hold paths relative to ``data/``, so they survive the
  move between the training box and the Jetson.
* **D-Fire's own train/test split is preserved.** The published split (17,221 /
  4,306) is the one every paper reports against; re-shuffling it would silently
  cost us that comparison. We carve only *val* out of the training pool.
* **OOD sets are quarantined.** FLAME / FLAME 2 / Boreal exist to be tested on,
  never trained on. :func:`assert_trainable` raises if a training file list
  touches anything on an OOD list — the harness refuses, rather than trusting
  the experimenter to remember.
* **Small plumes are tagged here**, not per-XP. ``map50_small_plume`` is the
  early-detection metric and the first casualty of most compression techniques,
  so its definition lives in exactly one place.
* **Seed 42 everywhere**, and every selection is a pure function of
  (file list, seed) so the splits regenerate bit-identically.

Label format is YOLO: one ``.txt`` per image, one line per box,
``<class> <cx> <cy> <w> <h>`` with all coordinates normalized to [0, 1].
Because w and h are normalized, ``w * h`` *is* the fraction of image area a box
covers — the small-plume test needs no image decode.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
SPLITS = DATA / "splits"

SEED = 42

#: Fraction-of-image-area below which a ground-truth box counts as a "small
#: plume" — the distant, faint smoke a fire watch has to catch early.
SMALL_PLUME_AREA_FRAC = 0.01

#: A second, much tighter tier. XP0 measured D-Fire's actual box-size
#: distribution and found the median box covers 1.34% of the image — so the 1%
#: threshold above lands almost exactly on the median and selects "the smaller
#: half" (45.4% of boxes) rather than the hard tail its name suggests. Boxes
#: under 0.1% (~20x20 px in a 640x640 frame, 10.6% of boxes) are the genuinely
#: tiny ones, and are where compression damage should show up first. Both tiers
#: are reported; neither replaces the other.
TINY_PLUME_AREA_FRAC = 0.001

#: D-Fire class ids. Asserted against the published box counts by
#: :func:`verify_class_map` in XP0 rather than taken on trust — the dataset's
#: own README numbers the classes differently from the label files.
CLASS_NAMES = {0: "smoke", 1: "fire"}

#: Published D-Fire box counts (Venâncio et al. 2022), used as the fingerprint
#: that tells us which integer means which class.
PUBLISHED_BOX_COUNTS = {"fire": 14692, "smoke": 11865}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


# --------------------------------------------------------------------------
# label / image plumbing
# --------------------------------------------------------------------------

@dataclass
class Box:
    cls: int
    cx: float
    cy: float
    w: float
    h: float

    @property
    def area_frac(self) -> float:
        """Fraction of the image this box covers (w, h are already normalized)."""
        return self.w * self.h

    @property
    def is_small_plume(self) -> bool:
        return self.area_frac < SMALL_PLUME_AREA_FRAC

    @property
    def is_tiny_plume(self) -> bool:
        return self.area_frac < TINY_PLUME_AREA_FRAC

    def to_xyxy(self, img_w: int, img_h: int) -> tuple[float, float, float, float]:
        x0 = (self.cx - self.w / 2) * img_w
        y0 = (self.cy - self.h / 2) * img_h
        return x0, y0, x0 + self.w * img_w, y0 + self.h * img_h


@dataclass
class Sample:
    """One image plus its ground truth.

    ``rel`` (path relative to ``data/``) is what gets written to the split files;
    ``stem`` is the identity used for fingerprints and set arithmetic.
    """
    rel: str
    image: Path
    label: Path | None
    boxes: list[Box] = field(default_factory=list)

    @property
    def stem(self) -> str:
        return Path(self.rel).stem

    @property
    def content(self) -> str:
        """D-Fire's four content categories, recomputed from the boxes.
        Used to stratify the val split so it carries the same mix as train."""
        classes = {b.cls for b in self.boxes}
        has_fire = any(CLASS_NAMES.get(c) == "fire" for c in classes)
        has_smoke = any(CLASS_NAMES.get(c) == "smoke" for c in classes)
        if has_fire and has_smoke:
            return "both"
        if has_fire:
            return "fire"
        if has_smoke:
            return "smoke"
        return "none"

    @property
    def has_small_plume(self) -> bool:
        return any(b.is_small_plume for b in self.boxes)

    @property
    def has_tiny_plume(self) -> bool:
        return any(b.is_tiny_plume for b in self.boxes)

    @property
    def is_background(self) -> bool:
        """No fire, no smoke — normal forest, sky, cloud, fog, sunset.

        46.6% of D-Fire's test split. These carry no boxes, so they contribute to
        mAP only through the false positives they provoke; the explicit
        false-alarm rate the evaluator reports is what actually makes them
        legible."""
        return not self.boxes


def read_label(path: Path | None) -> list[Box]:
    """Parse one YOLO label file. Missing or empty file -> no boxes (D-Fire's
    ~9.8k background images are legitimately empty, not broken)."""
    if path is None or not path.exists():
        return []
    boxes: list[Box] = []
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path}:{lineno}: expected 5 fields, got {len(parts)}")
        c, cx, cy, w, h = parts
        boxes.append(Box(int(c), float(cx), float(cy), float(w), float(h)))
    return boxes


def scan_dataset(images_dir: Path, labels_dir: Path, *, data_root: Path = DATA) -> list[Sample]:
    """Pair every image with its label file and read the boxes.

    Sorted by filename so the ordering is machine-independent — directory glob
    order is not, and the splits must regenerate identically on both boxes.
    """
    images_dir, labels_dir = Path(images_dir), Path(labels_dir)
    samples: list[Sample] = []
    for img in sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
        lbl = labels_dir / f"{img.stem}.txt"
        samples.append(Sample(
            rel=str(img.resolve().relative_to(Path(data_root).resolve())),
            image=img,
            label=lbl if lbl.exists() else None,
            boxes=read_label(lbl),
        ))
    if not samples:
        raise FileNotFoundError(f"no images found under {images_dir}")
    return samples


def verify_class_map(samples: list[Sample]) -> dict:
    """Check CLASS_NAMES against the published per-class box counts.

    D-Fire's README numbers the classes 1/2 while the label files use 0/1, so the
    mapping is worth *deriving* rather than assuming: fire has ~14.7k boxes and
    smoke ~11.9k, a gap far larger than any plausible mirror-to-mirror variation.
    Returns the observed counts and whether they corroborate CLASS_NAMES.
    """
    counts: dict[int, int] = {}
    for s in samples:
        for b in s.boxes:
            counts[b.cls] = counts.get(b.cls, 0) + 1
    if len(counts) < 2:
        return {"observed_by_id": counts, "consistent": False,
                "note": "fewer than two classes present — cannot fingerprint"}
    by_size = sorted(counts, key=lambda c: counts[c], reverse=True)
    derived = {by_size[0]: "fire", by_size[1]: "smoke"}   # fire is the larger class
    return {
        "observed_by_id": dict(sorted(counts.items())),
        "observed_named": {CLASS_NAMES.get(k, f"id{k}"): v for k, v in sorted(counts.items())},
        "derived_map": derived,
        "declared_map": CLASS_NAMES,
        "consistent": derived == CLASS_NAMES,
        "published": PUBLISHED_BOX_COUNTS,
    }


# --------------------------------------------------------------------------
# frozen splits
# --------------------------------------------------------------------------

def _stable_shuffle(samples: list[Sample], seed: int) -> list[Sample]:
    """Shuffle that depends only on (samples, seed) — not on directory order or
    Python's hash randomization."""
    out = sorted(samples, key=lambda s: s.rel)
    random.Random(seed).shuffle(out)
    return out


def build_splits(train_pool: list[Sample], test_pool: list[Sample], *,
                 val_frac: float = 0.10, seed: int = SEED,
                 out_dir: Path = SPLITS) -> dict:
    """Freeze train/val/test as committed file lists.

    ``test_pool`` is D-Fire's published test split and is written through
    untouched. ``train_pool`` is its published training split, from which val is
    carved — stratified by content category so val carries the same
    fire/smoke/both/none mix as train.

    Called exactly once, by XP0. Re-running must reproduce the same lists;
    ``manifest.json`` records the checksums that prove it.
    """
    if not 0.0 < val_frac < 0.5:
        raise ValueError("val_frac must be a small positive fraction")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    overlap = {s.stem for s in train_pool} & {s.stem for s in test_pool}
    if overlap:
        raise ValueError(f"{len(overlap)} stems appear in BOTH pools, e.g. {sorted(overlap)[:5]}")

    buckets: dict[str, list[Sample]] = {}
    for s in train_pool:
        buckets.setdefault(s.content, []).append(s)

    train: list[Sample] = []
    val: list[Sample] = []
    for content in sorted(buckets):
        group = _stable_shuffle(buckets[content], seed)
        n_val = int(len(group) * val_frac)
        val += group[:n_val]
        train += group[n_val:]

    splits = {"train": train, "val": val, "test": list(test_pool)}
    for name, group in splits.items():
        _write_list(out_dir / f"{name}.txt", sorted(s.rel for s in group))

    # The small-plume test subset: a *view* of the test split, not a split of its
    # own. Tagged here so mAP on distant plumes is defined in exactly one place.
    small = sorted(s.rel for s in test_pool if s.has_small_plume)
    _write_list(out_dir / "test_small_plume.txt", small)

    manifest = {
        "seed": seed,
        "source": "D-Fire published train/test split; val carved from train",
        "val_frac": val_frac,
        "small_plume_area_frac": SMALL_PLUME_AREA_FRAC,
        "class_names": CLASS_NAMES,
        "counts": {k: len(v) for k, v in splits.items()},
        "test_small_plume": len(small),
        "content_mix": {name: _content_mix(group) for name, group in splits.items()},
        "boxes": {name: _box_counts(group) for name, group in splits.items()},
        "checksums": {name: _list_checksum(sorted(s.rel for s in group))
                      for name, group in splits.items()},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _content_mix(samples: list[Sample]) -> dict:
    mix: dict[str, int] = {}
    for s in samples:
        mix[s.content] = mix.get(s.content, 0) + 1
    return dict(sorted(mix.items()))


def _box_counts(samples: list[Sample]) -> dict:
    counts: dict[str, int] = {}
    small = 0
    for s in samples:
        for b in s.boxes:
            name = CLASS_NAMES.get(b.cls, f"id{b.cls}")
            counts[name] = counts.get(name, 0) + 1
            if b.is_small_plume:
                small += 1
    counts["small_plume_boxes"] = small
    return dict(sorted(counts.items()))


def _write_list(path: Path, rels: list[str]) -> None:
    path.write_text("\n".join(rels) + "\n")


def _list_checksum(rels: list[str]) -> str:
    return hashlib.sha256("\n".join(rels).encode()).hexdigest()[:16]


def load_split(name: str, *, split_dir: Path = SPLITS, data_root: Path = DATA) -> list[Path]:
    """Read a frozen split as absolute image paths. The only sanctioned way for
    an XP to learn which images it may touch."""
    path = Path(split_dir) / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist — splits are frozen by XP0 (experiments/xp00_foundation)")
    return [Path(data_root) / ln.strip()
            for ln in path.read_text().splitlines() if ln.strip()]


def load_samples(name: str, *, split_dir: Path = SPLITS, data_root: Path = DATA) -> list[Sample]:
    """A frozen split as :class:`Sample` objects, labels read. What the evaluator
    consumes."""
    samples = []
    for img in load_split(name, split_dir=split_dir, data_root=data_root):
        lbl = img.parent.parent / "labels" / f"{img.stem}.txt"
        samples.append(Sample(
            rel=str(img.resolve().relative_to(Path(data_root).resolve())),
            image=img,
            label=lbl if lbl.exists() else None,
            boxes=read_label(lbl),
        ))
    return samples


# --------------------------------------------------------------------------
# OOD quarantine
# --------------------------------------------------------------------------

def quarantined(split_dir: Path = SPLITS) -> set[str]:
    """Every relative path on every ``ood_*.txt`` list. Training on any of these
    would silently invalidate the generalization claim the whole series rests on."""
    out: set[str] = set()
    for p in sorted(Path(split_dir).glob("ood_*.txt")):
        out |= {ln.strip() for ln in p.read_text().splitlines() if ln.strip()}
    return out


def assert_trainable(items, *, split_dir: Path = SPLITS, data_root: Path = DATA) -> None:
    """Raise if any item is quarantined. Call at the top of every training entry
    point — the harness refuses, so nobody has to remember.

    Accepts relative-path strings, absolute paths, or :class:`Sample` objects.
    """
    def as_rel(x) -> str:
        if isinstance(x, Sample):
            return x.rel
        p = Path(x)
        if p.is_absolute():
            try:
                return str(p.resolve().relative_to(Path(data_root).resolve()))
            except ValueError:
                return str(p)
        return str(p)

    banned = quarantined(split_dir)
    if not banned:
        return
    bad = sorted({as_rel(x) for x in items} & banned)
    if bad:
        raise PermissionError(
            f"{len(bad)} OOD-quarantined file(s) in a training list — refusing. "
            f"First few: {bad[:5]}. OOD sets are test-only (PLAN.md §1)."
        )


# --------------------------------------------------------------------------
# INT8 calibration set
# --------------------------------------------------------------------------

def image_stats(path: Path, max_side: int = 128) -> dict:
    """Cheap luminance statistics on a downscaled grayscale copy.

    Used only to *stratify* the calibration set by shooting condition. PIL's
    ``draft`` mode decodes JPEGs directly at reduced scale, so this stays fast
    enough to sweep the whole training set on the Jetson.
    """
    from PIL import Image

    with Image.open(path) as im:
        im.draft("L", (max_side, max_side))          # JPEG-only fast path; no-op otherwise
        im = im.convert("L")
        im.thumbnail((max_side, max_side))
        px = list(im.getdata())
    n = len(px)
    px_sorted = sorted(px)
    mean = sum(px) / n
    var = sum((p - mean) ** 2 for p in px) / n
    return {
        "mean": mean,
        "std": var ** 0.5,
        "p05": px_sorted[int(0.05 * (n - 1))],
        "p95": px_sorted[int(0.95 * (n - 1))],
    }


#: How many calibration images each condition gets. Sums to less than the total
#: so the remainder is topped up at random — a calibration set that is *only*
#: hard conditions would misrepresent the deployment distribution.
CALIB_QUOTA = {"small_plume": 150, "night": 100, "fog": 100, "backlight": 100}


def build_calibration_set(train_samples: list[Sample], *, n: int = 500, seed: int = SEED,
                          out_dir: Path = SPLITS,
                          stats: dict[str, dict] | None = None) -> dict:
    """Pick the frozen 500-image INT8 calibration set from the *training* split.

    PLAN.md §1 requires it to cover night, fog, sunset/backlight and small
    distant plumes — the conditions INT8 is most likely to quietly break. D-Fire
    carries no condition labels, so the conditions are approximated from
    luminance statistics, quantile-based so the thresholds adapt to the dataset
    rather than being magic numbers:

    * **night**      — lowest-quantile mean luminance.
    * **fog / haze** — lowest-quantile luminance spread (washed-out, low contrast).
    * **backlight**  — highest-quantile p95-minus-mean (blown sky over dark ground).
    * **small plume** — from the small-plume tag, a ground-truth property rather
      than a pixel one.

    These are proxies, and XP10's slice analysis is where they get put to work —
    if INT8 turns out to break on a condition the proxy mislabels, that shows up
    there and the proxy gets revisited (with a PROTOCOL_VERSION bump).

    Categories are filled in the order above and de-duplicated, then any shortfall
    is topped up at random, so the set is exactly ``n`` images and is a pure
    function of (train split, seed).
    """
    if len(train_samples) < n:
        raise ValueError(f"training pool has {len(train_samples)} images, need {n}")
    assert_trainable(train_samples, split_dir=out_dir)   # calibration data IS training data

    stats = stats or {s.rel: image_stats(s.image) for s in train_samples}

    def lowest(key: str, frac: float) -> list[Sample]:
        ranked = sorted(train_samples, key=lambda s: (stats[s.rel][key], s.rel))
        return ranked[:max(1, int(len(ranked) * frac))]

    def highest_spread(frac: float) -> list[Sample]:
        ranked = sorted(train_samples,
                        key=lambda s: (-(stats[s.rel]["p95"] - stats[s.rel]["mean"]), s.rel))
        return ranked[:max(1, int(len(ranked) * frac))]

    candidates = {
        "small_plume": [s for s in train_samples if s.has_small_plume],
        "night": lowest("mean", 0.15),
        "fog": lowest("std", 0.15),
        "backlight": highest_spread(0.15),
    }

    chosen: list[str] = []
    taken: set[str] = set()
    composition: dict[str, int] = {}
    for cat, quota in CALIB_QUOTA.items():
        avail = [s.rel for s in _stable_shuffle(candidates[cat], seed) if s.rel not in taken]
        pick = avail[:quota]
        composition[cat] = len(pick)
        taken.update(pick)
        chosen += pick

    remainder = [s.rel for s in _stable_shuffle(train_samples, seed + 1) if s.rel not in taken]
    topup = remainder[:max(0, n - len(chosen))]
    composition["random"] = len(topup)
    chosen += topup

    if len(chosen) != n:
        raise RuntimeError(f"calibration set is {len(chosen)} images, expected {n}")

    _write_list(Path(out_dir) / "calib.txt", sorted(chosen))
    meta = {
        "n": n,
        "seed": seed,
        "source": "train split only (never val/test, never OOD)",
        "composition": composition,
        "checksum": _list_checksum(sorted(chosen)),
        "condition_proxies": {
            "night": "lowest 15% mean luminance",
            "fog": "lowest 15% luminance std (low contrast)",
            "backlight": "highest 15% (p95 - mean) luminance (blown sky, dark ground)",
            "small_plume": f"has a GT box < {SMALL_PLUME_AREA_FRAC:.0%} of image area",
        },
    }
    (Path(out_dir) / "calib_manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


__all__ = [
    "Box", "Sample", "CLASS_NAMES", "SEED", "SMALL_PLUME_AREA_FRAC", "DATA", "SPLITS",
    "read_label", "scan_dataset", "verify_class_map",
    "build_splits", "load_split", "load_samples", "quarantined", "assert_trainable",
    "image_stats", "build_calibration_set",
]
