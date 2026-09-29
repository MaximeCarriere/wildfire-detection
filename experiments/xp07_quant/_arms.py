"""Shared runner for the whole-network arms (E3, E5, E6).

Three experiments differ only in which :class:`QuantConfig` variants they compare
and whether the Detect decode is quantized, so the run-calibrate-score-restore
loop lives here once. Each arm gets its **own** calibration pass: once more than
one layer is quantized, the tensor arriving at layer N is the output of a
quantized layer N-1, so the distribution to be measured depends on the setting
being measured. (E1 is the exception, and ``Quantizer.adopt`` explains why.)

Optional 12-epoch recovery is wired in behind a flag rather than on by default.
XP6's discipline requires damage and recovered numbers to be reported separately —
damage alone misranks methods — but 12 epochs over 15,500 images is a 3090-scale
job, and on this board it is a multi-hour run per arm. The flag exists so the
number can be produced deliberately; ``recovered: null`` in a JSON means it was
not run, never that it was zero.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(HERE.parents[1]) not in sys.path:
    sys.path.insert(0, str(HERE.parents[1]))

import _calib                                                      # noqa: E402
from _calib import log                                             # noqa: E402
from _quant import (DetectOutputQuantizer, QuantConfig, Quantizer,  # noqa: E402
                    conv_layers, head_conv_names, model_size_report)


def run_arm(name: str, cfg: QuantConfig, samples, *, n_calib: int = 512,
            targets: str | list[str] = "all", quantize_decode: bool = False,
            recover_epochs: int = 0, tag: str | None = None) -> dict:
    """Calibrate, score the damage, optionally recover, and score again.

    ``targets`` accepts ``"all"``, ``"no_head"`` (the 57 non-Detect convolutions),
    or an explicit list of layer names.
    """
    tag = tag or f"arm_{name}"
    model = _calib.load_base()
    all_names = [n for n, _ in conv_layers(model)]
    head = head_conv_names(model)
    if targets == "all":
        sel = all_names
    elif targets == "no_head":
        sel = [n for n in all_names if n not in set(head)]
    else:
        sel = list(targets)

    q = Quantizer(model, cfg, targets=sel)
    dq = (DetectOutputQuantizer(model, bits=cfg.act_bits, method=cfg.method)
          if quantize_decode else None)

    # One calibration pass fills both the conv observers and, if present, the
    # decode observer — they are independent hooks on the same forward pass.
    t0 = time.time()
    if dq is not None:
        dq.observe()
    _calib.run_calibration(model, q, n_calib, tag=tag)
    if dq is not None:
        dq.freeze()
    t_cal = time.time() - t0

    q.enable()
    if dq is not None:
        dq.enable()
    t0 = time.time()
    damage = _calib.score(model, samples, tag)
    t_eval = time.time() - t0

    row = {
        "arm": name,
        "config": cfg.label(),
        "n_layers_quantized": len(sel),
        "quantize_decode_output": quantize_decode,
        "layers_left_float": sorted(set(all_names) - set(sel)),
        "calibration_images": n_calib,
        # Size is a property of the configuration, so it is reported for every arm
        # — including the ones that only move accuracy, where the point is that
        # they move accuracy *for free*.
        "size": model_size_report(model, cfg, targets=sel),
        "damage": {"map50": damage["map50"], "map5095": damage["map5095"],
                   "small_plume": damage["small_plume"]["map50"],
                   "tiny_plume": damage["tiny_plume"]["map50"],
                   "background": damage["background"],
                   "fingerprint": damage["fingerprint"]},
        "decode_output_scale": dq.report() if dq is not None else None,
        "recovered": None,
        "seconds": {"calibration": round(t_cal, 1), "evaluation": round(t_eval, 1)},
    }
    log("arm", f"{name:22s} damage map50={damage['map50']:.4f} "
               f"tiny={damage['tiny_plume']['map50']}")

    if recover_epochs:
        row["recovered"] = _recover(model, q, dq, samples, recover_epochs, tag)

    q.restore()
    if dq is not None:
        dq.restore()
    del model
    return row


def _recover(model, q, dq, samples, epochs: int, tag: str) -> dict:
    """Quantization-aware recovery: fine-tune with the fake-quant hooks live.

    The hooks stay attached during training, so the weights are updated against
    the rounded forward pass — the gradient flows through ``round`` as if it were
    the identity, which is the straight-through estimator every QAT
    implementation uses. Weights are re-quantized after training, because
    training moves them off the grid they were placed on.

    The guardrail XP6's bug list demands is enforced by the caller (``e7_qat.py``
    proves the loop returns the *unquantized* model to its starting accuracy
    before any recovered number here is believed).
    """
    from lib.finetune import finetune
    t0 = time.time()
    hist = finetune(model, repo=_calib.YOLOV5_REPO, epochs=epochs, res=_calib.RES, batch=8)
    cfg = q.cfg

    # Training moved the weights off the grid they were placed on, so they are
    # re-placed. rebase() -- not restore() -- because the weights to re-quantize
    # are the ones training produced; restore() would write back the stash taken
    # before training and throw the whole fine-tune away.
    q.rebase()
    if dq is not None:
        dq.restore()

    # Activation ranges are re-measured too: training changed the distributions,
    # so the scales frozen before it are no longer the ranges of this network.
    if cfg.quantize_acts:
        if dq is not None:
            dq.observe()
        _calib.run_calibration(model, q, 512, tag=f"{tag}_recal")
        if dq is not None:
            dq.freeze()
    q.enable()
    if dq is not None:
        dq.enable()

    acc = _calib.score(model, samples, f"{tag}_recovered")
    log("arm", f"{tag} recovered map50={acc['map50']:.4f} after {epochs} epochs")
    return {"epochs": epochs, "map50": acc["map50"], "map5095": acc["map5095"],
            "small_plume": acc["small_plume"]["map50"],
            "tiny_plume": acc["tiny_plume"]["map50"],
            "train_seconds": round(time.time() - t0, 1),
            "history": hist if isinstance(hist, dict) else None}


def payload(experiment: str, question: str, axis: str, rows: list[dict],
            baseline: dict, samples, *, extra: dict | None = None) -> dict:
    return {
        "experiment": experiment, "question": question, "axis": axis,
        "split": "val", "n_val_images": len(samples), "resolution": _calib.RES,
        "calibration_list": str(_calib.CALIB_LIST.relative_to(_calib.REPO)),
        "baseline_fp16": baseline,
        "rows": rows,
        **(extra or {}),
    }
