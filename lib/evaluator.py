"""THE shared evaluation harness (PLAN.md §0).

Every number in this repo comes from here. No experiment implements its own mAP
computation and no experiment writes its own timing loop — if two XPs measured
accuracy differently, the compression arc they are supposed to form would be an
artifact of the measurement, not of the models.

The protocol is versioned. **If anything in this file changes the numbers, bump
:data:`PROTOCOL_VERSION` and re-run every prior XP** — a results.json carrying an
old protocol version is stale by definition, and ``analysis/make_figures.py``
refuses to plot mixed versions.

What is frozen here
-------------------
* **Accuracy** — COCO-style, threshold-free (detections swept at conf 0.001,
  max 300 per image), via ``pycocotools``. Reported as aggregate mAP50 /
  mAP50-95, **per class** (fire and smoke fail at different rates under
  compression — averaging that away would hide a core finding), on the
  small-plume slice, and on each OOD set.
* **Small-plume slice** — ground-truth boxes covering ≥ 1% of the image are
  marked ``iscrowd`` so pycocotools *ignores* them: detections landing on a big
  obvious plume are neither rewarded nor punished, and the score is purely
  "did it catch the distant faint one".
* **Latency** — batch 1, ≥ 50 warm-up frames, ≥ 1000 measured frames, median and
  p95, repeated ``runs`` times, reported mean ± 1 SE. CUDA is synchronized
  around every frame, so these are real end-to-end per-frame times.
* **Power** — sampled by :mod:`lib.power_logger` over exactly the measured
  window, with the nvpmodel power mode recorded alongside.
* **Reproducibility** — accuracy is deterministic given fixed weights and fixed
  data, so instead of error bars we assert bit-reproducibility:
  :func:`prediction_fingerprint` must be identical across two runs.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence

from lib import data as dataset
from lib.power_logger import PowerLogger, power_mode

#: Bump on ANY change that can move a number. See module docstring.
#:
#: 1.1 — added the ``tiny_plume`` (<0.1% area) tier alongside ``small_plume``, the
#:       background false-alarm rate, and batched throughput (batch-1 latency on
#:       this box measures kernel-launch overhead, not the model — see
#:       :func:`measure_throughput`). Purely additive: no 1.0 number changes
#:       meaning, but records must be re-run to carry the new fields.
PROTOCOL_VERSION = "1.1"

#: Frozen inference hyper-choices (PLAN.md §0). An XP may override one of these
#: only when it is that XP's explicit variable — never silently.
DEFAULT_INPUT_RES = 640
NMS_IOU = 0.45
DEMO_CONF = 0.25          # qualitative demos only, never for mAP
EVAL_CONF = 0.001         # threshold-free sweep
MAX_DET = 300

WARMUP_FRAMES = 50
MEASURE_FRAMES = 1000
TIMING_RUNS = 3


# --------------------------------------------------------------------------
# what a model has to look like to be measurable
# --------------------------------------------------------------------------

@dataclass
class ImageDets:
    """Detections for one image, in absolute pixel xyxy."""
    stem: str
    xyxy: list[tuple[float, float, float, float]]
    scores: list[float]
    classes: list[int]


class Detector(Protocol):
    """The contract every runtime (PyTorch, ONNX, TensorRT FP16/INT8/sparse)
    implements so the harness can measure it without knowing which it is."""

    name: str
    input_res: int

    def predict(self, images: Sequence[Path]) -> list[ImageDets]:
        """Threshold-free detections for accuracy scoring."""

    def prepare_frames(self, images: Sequence[Path]) -> list:
        """Decode + preprocess frames once, outside the timed loop."""

    def infer_frame(self, frame) -> None:
        """One batch-1 forward pass on a prepared frame. This is what gets timed."""

    def infer_batch(self, batch) -> None:
        """One forward pass on a stacked batch, for the compute-bound number."""


# --------------------------------------------------------------------------
# accuracy
# --------------------------------------------------------------------------

def _image_size(path: Path) -> tuple[int, int]:
    from PIL import Image
    with Image.open(path) as im:
        return im.size            # (w, h); header-only, no pixel decode


def _build_coco_gt(samples: Sequence[dataset.Sample], *, area_below: float | None = None) -> dict:
    """YOLO ground truth -> COCO dict.

    ``area_below`` restricts scoring to boxes covering less than that fraction of
    the image. It does not *drop* the larger boxes — it marks them ``iscrowd=1``.
    pycocotools then ignores both those boxes and any detection that matches them,
    which is exactly the semantics we want: score the model on distant plumes
    without inventing false positives out of the obvious ones.
    """
    images, annotations = [], []
    ann_id = 1
    for img_id, s in enumerate(samples, 1):
        w, h = _image_size(s.image)
        images.append({"id": img_id, "file_name": s.image.name, "width": w, "height": h})
        for b in s.boxes:
            x0, y0, x1, y1 = b.to_xyxy(w, h)
            ignore = area_below is not None and b.area_frac >= area_below
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": int(b.cls),
                "bbox": [x0, y0, x1 - x0, y1 - y0],
                "area": (x1 - x0) * (y1 - y0),
                "iscrowd": 1 if ignore else 0,
            })
            ann_id += 1
    categories = [{"id": cid, "name": name} for cid, name in sorted(dataset.CLASS_NAMES.items())]
    return {"images": images, "annotations": annotations, "categories": categories}


def _to_coco_dets(dets: Sequence[ImageDets], stem_to_id: dict[str, int]) -> list[dict]:
    out = []
    for d in dets:
        img_id = stem_to_id[d.stem]
        for (x0, y0, x1, y1), score, cls in zip(d.xyxy, d.scores, d.classes):
            out.append({
                "image_id": img_id,
                "category_id": int(cls),
                "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                "score": float(score),
            })
    return out


def _coco_eval(gt: dict, dets: list[dict], cat_ids: list[int] | None = None) -> dict:
    """Run COCOeval quietly and return {map50, map5095}.

    Three outcomes are kept distinct, because collapsing them would put a number
    in results.json that means something other than what it says:

    * **no detections** -> 0.0. A model that finds nothing scored zero; that is a
      result, not an error.
    * **no ground truth** for the evaluated categories -> ``None``. mAP is
      undefined, not zero. COCOeval signals this with its own ``-1`` sentinel,
      which is translated here so it can never be mistaken for a real score.
    * otherwise -> the score.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    wanted = set(cat_ids) if cat_ids is not None else None
    has_gt = any(
        a["iscrowd"] == 0 and (wanted is None or a["category_id"] in wanted)
        for a in gt["annotations"]
    )
    if not has_gt:
        return {"map50": None, "map5095": None}
    if not dets:
        return {"map50": 0.0, "map5095": 0.0}

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):        # pycocotools is chatty
        coco_gt = COCO()
        coco_gt.dataset = gt
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(list(dets))
        ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
        if cat_ids is not None:
            ev.params.catIds = cat_ids
        ev.params.maxDets = [1, 10, MAX_DET]
        ev.evaluate()
        ev.accumulate()

    # Deliberately NOT ev.summarize()/ev.stats. COCOeval.summarize() hardcodes
    # maxDets=100 for its headline AP@[.50:.95] entry; because we raise maxDets to
    # 300 (threshold-free sweeps need the headroom), that lookup finds no matching
    # entry and silently yields COCO's -1 "undefined" sentinel — a wrong number
    # that looks like a real one. Averaging the precision array ourselves is exact
    # and immune to that.
    return {
        "map5095": _mean_precision(ev),
        "map50": _mean_precision(ev, iou_thr=0.50),
    }


def _mean_precision(ev, iou_thr: float | None = None) -> float | None:
    """Average precision from COCOeval's accumulated array.

    ``precision`` has shape [IoU, recall, category, areaRng, maxDets]. We take
    area range 'all' (index 0) and our own maxDets (the last entry), average over
    the entries COCO left defined, and return None if it left none.
    """
    import numpy as np

    p = ev.eval["precision"]
    a_idx, m_idx = 0, len(ev.params.maxDets) - 1
    if iou_thr is None:
        s = p[:, :, :, a_idx, m_idx]
    else:
        t = np.where(np.isclose(ev.params.iouThrs, iou_thr))[0]
        s = p[t, :, :, a_idx, m_idx]
    valid = s[s > -1]
    return None if valid.size == 0 else float(valid.mean())


def prediction_fingerprint(dets: Sequence[ImageDets]) -> str:
    """Stable hash of a prediction set, for the bit-reproducibility assertion.

    Coordinates are rounded to 1e-3 px and scores to 1e-5 so the fingerprint
    survives irrelevant float noise but still catches any real change in output.
    """
    hasher = hashlib.sha256()
    for d in sorted(dets, key=lambda d: d.stem):
        hasher.update(d.stem.encode())
        order = sorted(range(len(d.scores)), key=lambda i: (-d.scores[i], d.classes[i], d.xyxy[i]))
        for i in order:
            x0, y0, x1, y1 = d.xyxy[i]
            hasher.update(f"|{d.classes[i]}:{x0:.3f},{y0:.3f},{x1:.3f},{y1:.3f}:"
                          f"{d.scores[i]:.5f}".encode())
    return hasher.hexdigest()[:16]


def evaluate_accuracy(detector: Detector, samples: Sequence[dataset.Sample]) -> dict:
    """Full accuracy block: aggregate, per class, and the small-plume slice.

    Returns the mAP fields of the results schema plus the fingerprint. The caller
    decides which schema keys these land in (in-distribution test vs an OOD set).
    """
    samples = list(samples)
    dets = detector.predict([s.image for s in samples])
    stem_to_id = {s.stem: i for i, s in enumerate(samples, 1)}

    gt = _build_coco_gt(samples)
    coco_dets = _to_coco_dets(dets, stem_to_id)

    agg = _coco_eval(gt, coco_dets)
    per_class = {}
    for cid, cname in sorted(dataset.CLASS_NAMES.items()):
        per_class[cname] = _coco_eval(gt, coco_dets, cat_ids=[cid])

    # Two size tiers, not one. XP0 found the 1% threshold sits on the median box
    # size, so it selects the smaller half rather than the hard tail; the 0.1%
    # tier is where compression damage to early detection should surface first.
    tiers = {
        "small_plume": (dataset.SMALL_PLUME_AREA_FRAC, lambda s: s.has_small_plume),
        "tiny_plume": (dataset.TINY_PLUME_AREA_FRAC, lambda s: s.has_tiny_plume),
    }
    slices, slice_counts = {}, {}
    for label, (frac, present) in tiers.items():
        stems = {s.stem for s in samples if present(s)}
        slice_counts[label] = len(stems)
        slices[label] = (_coco_eval(_build_coco_gt(samples, area_below=frac), coco_dets)
                         if stems else {"map50": None, "map5095": None})

    def rnd(m: dict) -> dict:
        return {k: (None if v is None else round(v, 4)) for k, v in m.items()}

    return {
        **rnd(agg),
        "per_class": {c: rnd(m) for c, m in per_class.items()},
        "small_plume": rnd(slices["small_plume"]),
        "tiny_plume": rnd(slices["tiny_plume"]),
        "background": background_false_alarms(samples, dets),
        "n_images": len(samples),
        "n_images_small_plume": slice_counts["small_plume"],
        "n_images_tiny_plume": slice_counts["tiny_plume"],
        "n_detections": len(coco_dets),
        "fingerprint": prediction_fingerprint(dets),
    }


def background_false_alarms(samples: Sequence[dataset.Sample], dets: Sequence[ImageDets],
                            conf: float = DEMO_CONF) -> dict | None:
    """How often the model cries wolf at an empty sky.

    46.6% of D-Fire's test split is background — normal forest, cloud, fog,
    sunset. mAP folds those false positives into one aggregate number, but for a
    detector that stares at nothing almost all the time, "what fraction of empty
    frames raise an alarm" is closer to the metric that decides deployability.
    XP14's cascade gate depends on it directly: every false wake costs power.

    Measured at the deployment threshold, not the threshold-free sweep — an alarm
    at conf 0.001 is not an alarm anyone would act on. The detections are already
    available from the accuracy pass, so this costs no extra inference.
    """
    by_stem = {d.stem: d for d in dets}
    bg = [s for s in samples if s.is_background]
    if not bg:
        return None

    alarms, boxes = 0, 0
    for s in bg:
        scores = [sc for sc in by_stem[s.stem].scores if sc >= conf] if s.stem in by_stem else []
        boxes += len(scores)
        alarms += bool(scores)
    return {
        "conf": conf,
        "n_background_images": len(bg),
        "n_alarmed": alarms,
        "false_alarm_rate": round(alarms / len(bg), 4),
        "false_boxes_per_bg_image": round(boxes / len(bg), 4),
    }


# --------------------------------------------------------------------------
# latency / throughput / power
# --------------------------------------------------------------------------

def _mean_se(values: Sequence[float]) -> tuple[float, float]:
    """Mean and 1 standard error. SE is 0.0 for a single run — reported, not hidden."""
    m = statistics.fmean(values)
    se = 0.0 if len(values) < 2 else statistics.stdev(values) / (len(values) ** 0.5)
    return m, se


def measure_latency(detector: Detector, images: Sequence[Path], *,
                    runs: int = TIMING_RUNS, warmup: int = WARMUP_FRAMES,
                    frames: int = MEASURE_FRAMES, log_power: bool = True,
                    allow_short: bool = False) -> dict:
    """Batch-1 latency, throughput and power, per PLAN.md §0.

    Frames are decoded and preprocessed *before* the timed loop and cycled if the
    image list is shorter than ``frames``, so what is measured is inference, not
    JPEG decoding or disk. Every per-frame time includes the device
    synchronization, so no work is left in flight and counted as free.

    ``allow_short`` exists only for wiring checks. It marks the result
    ``protocol_compliant: false``, so a short run can never be mistaken for a
    reportable measurement further downstream.
    """
    short = warmup < WARMUP_FRAMES or frames < MEASURE_FRAMES
    if short and not allow_short:
        raise ValueError(
            f"protocol requires >= {WARMUP_FRAMES} warm-up and >= {MEASURE_FRAMES} measured "
            f"frames (got {warmup}, {frames}) — lower these only by bumping PROTOCOL_VERSION, "
            f"or pass allow_short=True for a wiring check that is not a measurement")

    prepared = detector.prepare_frames(list(images))
    if not prepared:
        raise ValueError("no frames to time")

    def cycle(n):
        return [prepared[i % len(prepared)] for i in range(n)]

    warm = cycle(warmup)
    work = cycle(frames)

    run_medians, run_p95s, run_fps = [], [], []
    power_summaries, energies = [], []

    for _ in range(runs):
        for f in warm:
            detector.infer_frame(f)

        logger = PowerLogger() if log_power else None
        ctx = logger if logger is not None else contextlib.nullcontext()
        per_frame: list[float] = []
        with ctx:
            t0 = time.perf_counter()
            for f in work:
                s = time.perf_counter()
                detector.infer_frame(f)
                per_frame.append((time.perf_counter() - s) * 1000.0)   # ms
            t1 = time.perf_counter()

        per_frame.sort()
        run_medians.append(per_frame[len(per_frame) // 2])
        run_p95s.append(per_frame[int(0.95 * (len(per_frame) - 1))])
        run_fps.append(len(work) / (t1 - t0))
        if logger is not None:
            power_summaries.append(logger.summary(t0, t1))
            energies.append(logger.energy_joules(t0, t1))

    median_mean, median_se = _mean_se(run_medians)
    p95_mean, _ = _mean_se(run_p95s)
    fps_mean, fps_se = _mean_se(run_fps)

    out = {
        "power_mode": power_mode(),
        "latency_ms_median": round(median_mean, 3),
        "latency_ms_median_se": round(median_se, 4),
        "latency_ms_p95": round(p95_mean, 3),
        "fps_batch1": round(fps_mean, 2),
        "fps_se": round(fps_se, 3),
        "runs": runs,
        "warmup_frames": warmup,
        "measured_frames": frames,
        "timing_scope": "batch-1 forward + NMS, CUDA-synchronized; decode/letterbox excluded",
        "protocol_compliant": not short,
    }
    if power_summaries:
        watts = [p["power_w"]["mean"] for p in power_summaries if p["power_w"]["mean"] is not None]
        temps = [p["temp_c_peak"] for p in power_summaries if p["temp_c_peak"] is not None]
        utils = [p["gpu_util_pct"] for p in power_summaries if p["gpu_util_pct"] is not None]
        out |= {
            "power_w_mean": round(statistics.fmean(watts), 3) if watts else None,
            "energy_j_per_1000_frames": round(statistics.fmean(energies), 2) if energies else None,
            "gpu_util_pct": round(statistics.fmean(utils), 1) if utils else None,
            "temp_c_peak": max(temps) if temps else None,
        }
    return out


def measure_throughput(detector: Detector, images: Sequence[Path], *, batch: int = 16,
                       runs: int = TIMING_RUNS, warmup_batches: int = 5,
                       measured_batches: int = 60) -> dict:
    """Batched throughput — the compute-bound number, reported *alongside* batch-1
    latency because on this box the two say completely different things.

    XP2 established that PyTorch eager inference on the Orin Nano is
    **kernel-launch-bound at batch 1**: a YOLOv5s forward pass costs ~21 ms
    whether the input is 640x640 or 320x320, because the CPU cannot submit the
    model's ~200 kernel launches any faster and the GPU idles between them. At
    320px, batch 8 completes in the same wall-clock as batch 1.

    That makes batch-1 latency a measurement of *the runtime*, not of the model —
    useless for the questions this repo asks, since resolution, pruning and
    distillation all change compute while leaving the launch count roughly
    intact. Batching amortises the launches and lets model cost show through:
    8.65 ms/img at 640 vs 2.41 ms/img at 320, a 3.6x gap invisible at batch 1.

    Both numbers are real and both are reported. Batch-1 latency is what a single
    live camera feed actually experiences; batched throughput is what the
    hardware can actually do, and it is the axis every compression technique in
    this series should be judged on until TensorRT removes the launch bottleneck.
    """
    import torch

    prepared = detector.prepare_frames(list(images))
    if not prepared:
        raise ValueError("no frames to time")

    # prepare_frames returns batch-1 tensors; stack them into real batches.
    stacked = torch.cat([prepared[i % len(prepared)] for i in range(batch)], dim=0)

    for _ in range(warmup_batches):
        detector.infer_batch(stacked)

    per_run = []
    for _ in range(runs):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        for _ in range(measured_batches):
            detector.infer_batch(stacked)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        per_run.append(measured_batches * batch / elapsed)

    fps_mean, fps_se = _mean_se(per_run)
    return {
        "batch": batch,
        "fps_batched": round(fps_mean, 2),
        "fps_batched_se": round(fps_se, 3),
        "ms_per_image_batched": round(1000.0 / fps_mean, 3),
        "runs": runs,
        "measured_images": measured_batches * batch,
    }


def measure_memory() -> dict:
    """Peak CUDA memory for the current process, in MB. Zero if torch is absent
    or the runtime is TensorRT-native (which reports its own)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {"mem_mb": None}
        return {"mem_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1)}
    except Exception:
        return {"mem_mb": None}


# --------------------------------------------------------------------------
# the results record
# --------------------------------------------------------------------------

def results_record(*, model_id: str, fmt: str, params_m: float, size_disk_mb: float,
                   input_res: int = DEFAULT_INPUT_RES,
                   accuracy: dict | None = None,
                   ood: dict[str, dict] | None = None,
                   jetson: dict | None = None,
                   notes: str | None = None) -> dict:
    """Assemble one results.json in the frozen schema (PLAN.md §0).

    Every XP writes through this function so the keys are identical everywhere —
    the whole series is only comparable because nothing is allowed to invent its
    own field names. ``ood`` maps an OOD set name ("flame", "boreal") to an
    :func:`evaluate_accuracy` block; missing sets stay ``null`` rather than absent.
    """
    acc = accuracy or {}
    per_class = acc.get("per_class", {})
    ood = ood or {}

    rec = {
        "protocol_version": PROTOCOL_VERSION,
        "model_id": model_id,
        "params_m": round(params_m, 4),
        "size_disk_mb": round(size_disk_mb, 3),
        "format": fmt,
        "input_res": input_res,
        "nms_iou": NMS_IOU,
        "eval_conf": EVAL_CONF,
        "max_det": MAX_DET,
        "seed": dataset.SEED,

        "map50_dfire_test": acc.get("map50"),
        "map5095_dfire_test": acc.get("map5095"),
        "map50_fire_class": per_class.get("fire", {}).get("map50"),
        "map50_smoke_class": per_class.get("smoke", {}).get("map50"),
        "map50_small_plume": acc.get("small_plume", {}).get("map50"),
        "map50_tiny_plume": acc.get("tiny_plume", {}).get("map50"),
        "bg_false_alarm_rate": (acc.get("background") or {}).get("false_alarm_rate"),
        "map50_ood_flame": (ood.get("flame") or {}).get("map50"),
        "map50_ood_boreal": (ood.get("boreal") or {}).get("map50"),

        "jetson": jetson,
        "accuracy_detail": acc or None,
        "ood_detail": ood or None,
        "notes": notes,
    }
    return rec


def write_results(record: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n")
    return path


__all__ = [
    "PROTOCOL_VERSION", "DEFAULT_INPUT_RES", "NMS_IOU", "DEMO_CONF", "EVAL_CONF", "MAX_DET",
    "ImageDets", "Detector",
    "evaluate_accuracy", "prediction_fingerprint",
    "measure_latency", "measure_memory",
    "results_record", "write_results",
]
