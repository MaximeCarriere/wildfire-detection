"""XP7's frozen calibration set, and the shared plumbing every E-script uses.

**Why XP7 freezes its own set instead of reusing XP0's.** XP0's ``calib.txt`` is
90% deliberately hard cases (night, fog, backlight, small plumes). XP10 measured
what that costs: swapping it for a random sample was worth **+22% accuracy** on
the INT8 engine, because calibration is asking "what range does this tensor
occupy in deployment?" and a set built from the tails answers a different
question. So XP7's set is a **random sample of the train split** — the deployment
distribution, hard cases present at their natural rate rather than at 90%.

**Why the subsets are nested.** E2 sweeps calibration size over 8 / 32 / 128 /
512 images. If each size were an independent sample, a difference between two
sizes would be confounded with a difference between two *draws*. Here size N is
the first N entries of the same ordered list, so the 32-image arm is the 8-image
arm plus 24 more, and the sweep isolates the amount of data.

The list is written once, checksummed, and committed. Every arm in every
experiment — and the TensorRT calibrator in E4 — reads this one file.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lib import data as dataset                                    # noqa: E402

RAW = REPO / "results" / "raw"
WEIGHTS = REPO / "weights"
SPLITS = REPO / "data" / "splits"
CALIB_LIST = SPLITS / "xp07_calib.txt"
CALIB_META = SPLITS / "xp07_calib_manifest.json"

#: Frozen once and never re-drawn. Changing either invalidates every XP7 number.
CALIB_N = 512
CALIB_SEED = 707

#: The sizes E2 sweeps, all nested prefixes of the same list.
CALIB_SIZES = [8, 32, 128, 512]

BASE_WEIGHTS = WEIGHTS / "yolov5s.pt"
YOLOV5_REPO = Path.home() / "yolov5"
RES = 512

#: The line every XP7 table's top row has to show, from XP9 on this board.
FP16_LINE = {"map50": 0.7776, "fps_batched": 474.0, "j_per_1k": 51.8,
             "small_plume": 0.6061, "tiny_plume": 0.1376, "engine_mb": 17.0}


def log(tag: str, msg: str) -> None:
    print(f"[{tag}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# the frozen set
# --------------------------------------------------------------------------- #

def build_calib_list(force: bool = False) -> dict:
    """Draw the 512-image list once, in a fixed shuffled order, and pin its hash."""
    if CALIB_LIST.exists() and not force:
        return json.loads(CALIB_META.read_text())

    train = dataset.load_samples("train")
    dataset.assert_trainable(train)                # calibration data IS training data
    ordered = dataset._stable_shuffle(list(train), CALIB_SEED)[:CALIB_N]
    rels = [s.rel for s in ordered]
    CALIB_LIST.write_text("\n".join(rels) + "\n")

    n_small = sum(1 for s in ordered if s.has_small_plume)
    meta = {
        "n": CALIB_N,
        "seed": CALIB_SEED,
        "source": "train split only, uniform random — the deployment distribution",
        "why_not_xp0_calib": "XP0's calib.txt is 90% hard cases; XP10 measured +22% "
                             "accuracy from a representative sample instead",
        "nested_sizes": CALIB_SIZES,
        "checksum": hashlib.sha256("\n".join(rels).encode()).hexdigest()[:16],
        "small_plume_images": n_small,
        "small_plume_frac": round(n_small / len(ordered), 4),
    }
    CALIB_META.write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def calib_paths(n: int | None = None) -> list[Path]:
    """The first ``n`` calibration images, as absolute paths. Nested by construction."""
    if not CALIB_LIST.exists():
        build_calib_list()
    rels = [r for r in CALIB_LIST.read_text().split("\n") if r.strip()]
    expect = json.loads(CALIB_META.read_text())["checksum"]
    got = hashlib.sha256("\n".join(rels).encode()).hexdigest()[:16]
    if got != expect:
        raise RuntimeError(f"calibration list changed under us ({got} != {expect}) — "
                           "every XP7 number is scored against the pinned set")
    n = len(rels) if n is None else n
    if n > len(rels):
        raise ValueError(f"asked for {n} calibration images, the frozen set holds {len(rels)}")
    return [dataset.DATA / r for r in rels[:n]]


# --------------------------------------------------------------------------- #
# models and scoring
# --------------------------------------------------------------------------- #

def load_base(device: str = "cuda:0", weights: Path | None = None):
    """The same YOLOv5s every XP7 arm starts from, in eval mode."""
    from lib.prune_utils import load_yolov5
    model = load_yolov5(weights or BASE_WEIGHTS, YOLOV5_REPO, device=device)
    return model.eval()


def assert_classes(samples) -> None:
    """The class-name assert the measurement discipline requires on every load."""
    dataset.verify_class_map(list(samples))


def run_calibration(model, quantizer, n_images: int, *, res: int = RES,
                    batch: int = 8, device: str = "cuda:0", tag: str = "calib") -> None:
    """Feed ``n_images`` of the frozen set through the model so observers fill.

    Preprocessing goes through ``lib.trt_export._letterbox_batch`` — the exact
    path inference uses. A mismatch here would measure the ranges of a
    distribution the model never actually sees, which is the single easiest way
    to produce a calibration that is wrong for a reason nobody can find later.
    """
    import torch
    from lib.trt_export import _letterbox_batch

    paths = calib_paths(n_images)
    quantizer.observe()
    half = next(model.parameters()).dtype == torch.float16
    with torch.no_grad():
        for i in range(0, len(paths), batch):
            chunk = paths[i:i + batch]
            arr = _letterbox_batch(chunk, res, YOLOV5_REPO)
            x = torch.from_numpy(arr).to(device)
            model(x.half() if half else x)
    quantizer.freeze()
    log(tag, f"calibrated on {len(paths)} images, {len(quantizer.frozen)} activation scales")


def score(model, samples, tag: str, *, res: int = RES, half: bool = True) -> dict:
    """Accuracy for a live model, through the frozen evaluation harness."""
    from lib.detectors import Yolov5Detector
    det = Yolov5Detector.from_model(model, input_res=res, half=half, name=tag)
    acc = evaluate(det, samples)
    del det
    return acc


def evaluate(detector, samples) -> dict:
    from lib import evaluator
    return evaluator.evaluate_accuracy(detector, samples)


def write_json(name: str, payload: dict) -> Path:
    """One JSON per experiment under results/raw, the XP6 convention."""
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / name
    path.write_text(json.dumps(payload, indent=2) + "\n")
    log("io", f"wrote {path.relative_to(REPO)}")
    return path


def split_samples(split: str, limit: int = 0):
    """A split, optionally subsampled *deterministically* for smoke tests.

    A subsampled run is marked as such in its JSON; a published number is never
    allowed to come from one.
    """
    samples = dataset.load_samples(split)
    assert_classes(samples)
    if limit:
        samples = dataset._stable_shuffle(list(samples), 0)[:limit]
    return samples
