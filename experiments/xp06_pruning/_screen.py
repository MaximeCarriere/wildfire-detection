"""Shared 3090 screening path for the XP6 extension experiments.

Every extension experiment does the same four things: damage a model, retrain
it, score it on the frozen test set, and write one JSON through the frozen
harness. This module is that path, so the experiments differ only in *how* they
damage the model, which is the only thing any of them is actually about.

**Rule zero is enforced here, not remembered.** ``jetson=None`` on every record
and no call into ``evaluator.measure_latency``: a TensorRT engine is compiled per
GPU architecture, so a 3090 speed number says nothing about a 15 W Orin and must
not exist in a results file where it could later be quoted. ``lib.power_logger``
would in fact crash off-device (it shells out to ``tegrastats``), which is the
codebase making the same point.

Accuracy from this machine is screening evidence. The finalists go back to the
board and the board's numbers are the ones that get published.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lib import data as dataset                        # noqa: E402
from lib import evaluator                              # noqa: E402
from lib.detectors import YOLOV5_REPO, Yolov5Detector  # noqa: E402
from lib.prune_utils import save_pruned                # noqa: E402

RAW = REPO / "results" / "raw"
WEIGHTS = REPO / "weights"
BASE = "yolov5s"
RES = 512
BATCH = 32

#: Measured on this machine, unpruned yolov5s FP16 @512 over the full test set.
#: Every extension table carries it as its top row so the comparison is never
#: lost, and it is re-measured here rather than quoted from another page.
UNPRUNED = {"params_m": 7.025, "map50": 0.7764, "small": 0.6038,
            "tiny": 0.1380, "silent": 0.9736}


def log(tag: str, msg: str) -> None:
    print(f"[{tag}] {msg}", flush=True)


def torch_version() -> str:
    import torch
    return torch.__version__


def score_model(model, name: str, split: str = "test") -> dict:
    """Accuracy only, through the frozen harness, on a *copy* of a live model.

    The copy is not defensive tidiness, it is required. ``from_model(half=True)``
    calls ``model.half()``, which mutates the module in place, so scoring a model
    that is about to be trained would hand the optimizer FP16 parameters and
    torch refuses to unscale FP16 gradients. Any arm that measures damage before
    recovery hits this; the arms that only score after training never did, which
    is exactly the sort of bug that hides until one experiment does things in a
    different order.

    Scoring a copy also keeps the measurement honest in the other direction: the
    evaluation runs at FP16 because that is the deployment precision, and the
    trainable model never takes the half-to-float round trip.
    """
    import copy
    samples = dataset.load_samples(split)
    det = Yolov5Detector.from_model(copy.deepcopy(model), input_res=RES,
                                    half=True, name=name)
    acc = evaluator.evaluate_accuracy(det, samples)
    del det
    return acc


def recover_and_record(model, *, tag: str, experiment: str, prune_meta: dict,
                       epochs: int, notes: str, extra: dict | None = None,
                       save_name: str | None = None, lr0: float | None = None,
                       post_train=None) -> dict:
    """Fine-tune, score on test, write one results JSON. Returns the record.

    ``post_train`` runs on the trained model before it is saved or scored, and
    whatever dict it returns is merged into ``prune_meta``. The masked
    granularities need it: their sparsity has to be *verified* after training
    (a mask that silently stopped holding would otherwise be reported as a
    result) and their hooks stripped before the checkpoint is pickled.
    """
    import torch
    from lib.finetune import finetune

    t0 = time.perf_counter()
    train_meta = finetune(model, repo=YOLOV5_REPO, epochs=epochs, res=RES,
                          batch=BATCH, lr0=lr0)
    log(experiment, f"  recovery {train_meta['total_minutes']} min, "
                    f"final loss {train_meta['history'][-1]['loss']}")

    if post_train is not None:
        prune_meta = dict(prune_meta) | (post_train(model) or {})

    path = save_pruned(model, WEIGHTS / f"{save_name or tag}.pt", prune_meta)
    acc = score_model(model, tag)
    log(experiment, f"  test mAP50 {acc['map50']:.4f} · small "
                    f"{acc['small_plume']['map50']:.4f} · tiny "
                    f"{acc['tiny_plume']['map50']:.4f} · silent "
                    f"{acc['background']['correctly_silent_rate']}")

    rec = evaluator.results_record(
        model_id=tag, fmt="pt", params_m=prune_meta.get("params_m_after", 0.0),
        size_disk_mb=path.stat().st_size / 1e6, input_res=RES, accuracy=acc,
        jetson=None,                      # rule zero, see module docstring
        notes=(f"{notes} 3090 SCREENING MEASUREMENT, off-device: accuracy only. No "
               f"latency, throughput or energy is recorded here because a TensorRT "
               f"engine is compiled per GPU architecture and a 3090 figure says nothing "
               f"about a 15 W Orin. torch {torch_version()}, batch {BATCH}, {epochs} "
               f"epochs at {RES}px."))
    rec["prune_meta"] = prune_meta
    rec["train_meta"] = train_meta
    rec["machine"] = "rtx3090_screening"
    rec["experiment"] = experiment
    if extra:
        rec |= extra
    evaluator.write_results(rec, RAW / f"{experiment}_{tag}.json")
    log(experiment, f"  total {(time.perf_counter()-t0)/60:.1f} min -> "
                    f"{experiment}_{tag}.json")

    del model
    torch.cuda.empty_cache()
    return rec


def fresh_model():
    from lib.prune_utils import load_yolov5
    return load_yolov5(WEIGHTS / f"{BASE}.pt", YOLOV5_REPO)
