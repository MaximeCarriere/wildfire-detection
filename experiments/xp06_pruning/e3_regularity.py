#!/usr/bin/env python3
"""XP6 E3 — regularity: does rounding channel counts recover the missing speed?

XP6's most interesting observation was an accident. Its iterative arm kept *more*
parameters than its one-shot arm (4.51 M against 4.24 M) and ran substantially
*faster* (450 img/s against 381). Removing more arithmetic produced a slower
model. The proposed explanation was channel-count regularity: pruning leaves
layers at awkward widths like 47 instead of 64, and GPU kernels are written for
neat tile sizes, so a layer with 27% fewer channels can take exactly as long as
before.

That was a hypothesis with one accidental data point behind it. This turns it
into a controlled test. ``round_to`` makes the pruner round every surviving
channel count up to a multiple, so the same nominal cut lands on aligned widths
instead of arbitrary ones. Four arms, identical in every other respect, differing
only in alignment.

**This experiment cannot be finished on this machine and does not pretend to
be.** Its entire question is about throughput, and a TensorRT engine is compiled
per GPU architecture: a number from a desktop RTX 3090 says nothing about a 15 W
Orin. What runs here produces the four trained checkpoints and their ONNX
exports, which are architecture neutral and transfer cleanly. The board builds
the engines and produces the four throughput figures that decide it.

What this machine *can* settle is whether alignment costs accuracy, because if
round_to=32 were to cost several accuracy points the speed question would be
moot. That much is answered here.

Usage
    python e3_regularity.py --round-to 32 --epochs 12
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _screen import (BASE, RES, UNPRUNED, WEIGHTS, fresh_model,   # noqa: E402
                     log, recover_and_record)

TAG = "e3"
RATIO = 0.25
CRITERION = "l1"        # E2's evidence: L2 is a poor baseline on this detector


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round-to", type=int, required=True, choices=[1, 8, 16, 32])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--no-onnx", action="store_true")
    args = ap.parse_args()

    from lib.prune_utils import prune_channels

    log(TAG, f"=== {RATIO:.0%} one-shot, {CRITERION}, round_to={args.round_to} ===")
    model = fresh_model()
    meta = prune_channels(model, RATIO, res=RES, importance=CRITERION,
                          round_to=args.round_to)
    meta["method"] = "oneshot"
    log(TAG, f"  params {meta['params_m_before']} -> {meta['params_m_after']} M "
             f"({meta['params_reduction']:.1%}) · MACs -{meta['macs_reduction']:.1%}")

    # Record the surviving widths themselves. They are the independent variable,
    # and "how many layers landed on a multiple of 32" is the thing the board's
    # throughput number will have to be explained by.
    import torch.nn as nn
    widths = [m.out_channels for m in model.modules() if isinstance(m, nn.Conv2d)]
    meta["surviving_widths"] = widths
    for mult in (8, 16, 32):
        meta[f"widths_divisible_by_{mult}"] = sum(1 for w in widths if w % mult == 0)
    meta["n_conv_layers"] = len(widths)
    log(TAG, f"  widths divisible by 8/16/32: {meta['widths_divisible_by_8']}/"
             f"{meta['widths_divisible_by_16']}/{meta['widths_divisible_by_32']} "
             f"of {len(widths)}")

    tag = f"dfire_{BASE}_round{args.round_to}"
    rec = recover_and_record(
        model, tag=tag, experiment="xp06e3", prune_meta=meta, epochs=args.epochs,
        notes=(f"XP6 E3: {RATIO:.0%} channels pruned one-shot with {CRITERION}, surviving "
               f"channel counts rounded up to a multiple of {args.round_to}. THE HEADLINE "
               f"NUMBER FOR THIS EXPERIMENT IS THROUGHPUT AND IT IS NOT IN THIS FILE: it has "
               f"to be measured on the Jetson, because an engine is compiled per GPU "
               f"architecture. What this record establishes is the accuracy cost of "
               f"alignment, so the board's speed numbers can be read against a known "
               f"accuracy. Unpruned reference {UNPRUNED['map50']} mAP50."),
        extra={"round_to": args.round_to, "criterion": CRITERION,
               "verdict_requires": "jetson_throughput",
               "speed_measured": False})

    if not args.no_onnx:
        # Architecture-neutral hand-off to the board.
        try:
            import torch
            from lib.prune_utils import load_yolov5
            from lib.trt_export import export_onnx_from_model
            from lib.detectors import YOLOV5_REPO
            ck = torch.load(WEIGHTS / f"{tag}.pt", map_location="cuda:0",
                            weights_only=False)
            onnx = export_onnx_from_model(ck["model"], WEIGHTS / f"{tag}.onnx",
                                          res=RES, repo=YOLOV5_REPO)
            log(TAG, f"  exported {onnx.name} ({onnx.stat().st_size/1e6:.1f} MB) "
                     f"for engine building on the board")
        except Exception as e:
            log(TAG, f"  ONNX export failed: {type(e).__name__}: {e}")

    log(TAG, f"  round_to={args.round_to}: mAP50 {rec['map50_dfire_test']:.4f} at "
             f"{meta['params_m_after']} M")


if __name__ == "__main__":
    main()
