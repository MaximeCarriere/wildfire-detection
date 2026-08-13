#!/usr/bin/env python3
"""XP6 E4 — 2:4 structured sparsity, the one pattern with silicon behind it.

Every other granularity in this study trades accuracy for a speed-up that
ordinary hardware may or may not deliver. 2:4 is different: the rule is that of
every four neighbouring weights exactly two must be zero, and Ampere GPUs have
dedicated circuitry that skips those zeros for up to twice the peak throughput.
The rigidity is the feature. NVIDIA's published table reports a detector,
SSD-RN50, going from 24.8 to 24.8 box AP on COCO -- no measurable accuracy cost
at 50% sparsity.

The target board's GPU is Ampere class, so the hardware path exists on paper.
Whether TensorRT actually selects sparse kernels for *this* network on *this*
SKU is a measurement, and it is not one this machine can make.

**Three results, reported separately and never blurred together:**

  1. does accuracy hold under the 2:4 constraint (answered here);
  2. does the compiler actually choose sparse kernels (board, from its log);
  3. what throughput comes out (board).

It is entirely possible that (1) holds and (3) does not materialise. That is a
publishable result and it fits this project's recurring theme exactly, which is
that hardware does not do what the arithmetic promises.

Usage
    python e4_sparsity24.py --epochs 12
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _masking import (apply_2to4, reapply_masks, strip_masks,   # noqa: E402
                      verify_sparsity)
from _screen import (BASE, RES, UNPRUNED, WEIGHTS, fresh_model,  # noqa: E402
                     log, recover_and_record, score_model)

TAG = "e4"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--no-onnx", action="store_true")
    args = ap.parse_args()

    from lib.prune_utils import detect_head_convs

    log(TAG, f"=== 2:4 structured sparsity + {args.epochs} epochs ===")
    model = fresh_model()
    meta = apply_2to4(model, exclude={id(m) for m in detect_head_convs(model)})
    log(TAG, f"  {meta['achieved_sparsity']:.2%} of patterned weights zeroed "
             f"({meta['zeroed_weights']:,} of {meta['patterned_weights']:,})")
    if meta["n_layers_left_dense"]:
        # Reported rather than padded: a layer whose filter size is not divisible
        # by four cannot carry the pattern the hardware reads, and quietly
        # forcing one would misrepresent what the board is being handed.
        log(TAG, f"  {meta['n_layers_left_dense']} layers left dense (filter size not "
                 f"divisible by 4): {', '.join(meta['layers_left_dense'][:4])}")

    acc0 = score_model(model, "e4_2to4_nofinetune")
    log(TAG, f"  no retraining: mAP50 {acc0['map50']:.4f} (unpruned {UNPRUNED['map50']}) "
             f"· tiny {acc0['tiny_plume']['map50']:.4f}")
    meta["map50_before_recovery"] = acc0["map50"]
    meta["map50_tiny_before_recovery"] = acc0["tiny_plume"]["map50"]

    def after_training(m):
        drift = reapply_masks(m)
        v = verify_sparsity(m) | drift
        log(TAG, f"  EMA carried {drift['ema_sparsity_before_remask']:.2%}; after "
                 f"re-masking {v['measured_sparsity']:.2%} · {v['nonzero_params_m']} M "
                 f"non-zero of {v['dense_params_m']} M dense")
        strip_masks(m)
        return v

    tag = f"dfire_{BASE}_sparse24"
    rec = recover_and_record(
        model, tag=tag, experiment="xp06e4", prune_meta=meta, epochs=args.epochs,
        post_train=after_training,
        notes=(f"XP6 E4: 2:4 semi-structured sparsity (two of every four contiguous weights "
               f"zeroed), {args.epochs}-epoch recovery. Weights are MASKED, not removed: the "
               f"tensor keeps its shape and the model is not smaller on disk. THIS RECORD "
               f"ANSWERS ACCURACY ONLY. Whether TensorRT selects sparse kernels, and what "
               f"throughput results, must be measured on the Ampere-class Jetson with the "
               f"sparse weights builder flag; those two results are reported separately and "
               f"must not be conflated with this one. Unpruned reference "
               f"{UNPRUNED['map50']} mAP50."),
        extra={"granularity": "2:4", "speed_measured": False,
               "verdict_requires": "jetson_sparse_engine",
               "board_checklist": ["build engine with sparse weights enabled",
                                   "confirm from the builder log that sparse kernels were selected",
                                   "measure throughput and energy against the dense engine"]})

    if not args.no_onnx:
        try:
            import torch
            from lib.trt_export import export_onnx_from_model
            from lib.detectors import YOLOV5_REPO
            ck = torch.load(WEIGHTS / f"{tag}.pt", map_location="cuda:0",
                            weights_only=False)
            onnx = export_onnx_from_model(ck["model"], WEIGHTS / f"{tag}.onnx",
                                          res=RES, repo=YOLOV5_REPO)
            log(TAG, f"  exported {onnx.name} ({onnx.stat().st_size/1e6:.1f} MB) — build "
                     f"this on the board with sparsity enabled")
        except Exception as e:
            log(TAG, f"  ONNX export failed: {type(e).__name__}: {e}")

    log(TAG, f"  2:4: mAP50 {rec['map50_dfire_test']:.4f} "
             f"(accuracy answered; speed needs the board)")


if __name__ == "__main__":
    main()
