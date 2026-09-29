#!/usr/bin/env python3
"""XP7 E7/E10 cost probe — what does 12 epochs actually cost on this board?

Every "recovered" column in XP7, and both of E7's trained arms and E10's second
arm, cost 12 epochs over the 15,500-image training split. The plan budgeted a
desktop GPU for that. On a 15 W Jetson the number is the deciding factor for
whether those experiments happen at all — and so far it has been an *estimate*,
which is exactly the kind of number this series is supposed to measure instead of
assert.

This times the real loop: the real dataloader, the real augmentation, the real
forward and backward, with the fake-quant hooks live so it measures QAT and not
plain fine-tuning. It runs a warmup, then a fixed number of timed steps, and
projects from the measured rate.

It deliberately does **not** run a full epoch. A projection from 60 timed steps
is honest about being a projection; a single epoch measured once is not obviously
better and costs an hour to find out.

Usage
    python e7_cost.py                    # QAT loop, 60 timed steps
    python e7_cost.py --no-quant         # plain fine-tune, for the comparison
    python e7_cost.py --steps 120
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _calib                                                      # noqa: E402
import torch                                                       # noqa: E402
from _calib import log                                             # noqa: E402
from _quant import QuantConfig, Quantizer                          # noqa: E402

TRAIN_IMAGES = 15500
EPOCHS = 12


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60, help="timed optimiser steps")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--no-quant", action="store_true",
                    help="time a plain fine-tune instead of QAT")
    ap.add_argument("--calib-images", type=int, default=64)
    ap.add_argument("--out", default="xp07e7_cost.json")
    args = ap.parse_args()

    sys.path.insert(0, str(_calib.YOLOV5_REPO))
    from lib.finetune import base_hyp, build_loader
    from utils.loss import ComputeLoss
    from utils.torch_utils import smart_optimizer

    model = _calib.load_base()
    q = None
    if not args.no_quant:
        cfg = QuantConfig(weight_bits=8, act_bits=8, granularity="per_channel",
                          method="minmax")
        q = Quantizer(model, cfg)
        _calib.run_calibration(model, q, args.calib_images, tag="e7cost")
        q.enable()
        log("e7c", "fake-quant hooks live — this times QAT, not plain fine-tuning")

    detect = next(m for m in model.modules() if type(m).__name__ == "Detect")
    hyp = base_hyp(detect.nl, int(detect.nc), _calib.RES)
    model.hyp, model.gr, model.nc = hyp, 1.0, int(detect.nc)
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)

    loader = build_loader("train", _calib.RES, args.batch, _calib.YOLOV5_REPO,
                          hyp, augment=True)
    opt = smart_optimizer(model, "SGD", hyp["lr0"], hyp["momentum"], hyp["weight_decay"])
    loss_fn = ComputeLoss(model)
    scaler = torch.amp.GradScaler("cuda")

    it = iter(loader)
    done = 0
    t0 = None
    n_timed = 0
    while done < args.warmup + args.steps:
        try:
            imgs, targets = next(it)[:2]
        except StopIteration:
            it = iter(loader)
            continue
        imgs = imgs.to("cuda", non_blocking=True).float() / 255.0
        targets = targets.to("cuda")

        with torch.amp.autocast("cuda"):
            pred = model(imgs)
            loss = loss_fn(pred, targets)[0]
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

        done += 1
        if done == args.warmup:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        elif done > args.warmup:
            n_timed += 1

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    img_s = n_timed * args.batch / elapsed
    epoch_s = TRAIN_IMAGES / img_s
    total_h = epoch_s * EPOCHS / 3600

    log("e7c", f"{img_s:.2f} img/s forward+backward at {_calib.RES} px, "
               f"batch {args.batch}")
    log("e7c", f"one epoch over {TRAIN_IMAGES} images: {epoch_s / 60:.1f} min")
    log("e7c", f"{EPOCHS} epochs: {total_h:.1f} h per trained arm")
    log("e7c", f"E7 as specified (round-trip control + 2 trained arms): "
               f"{total_h * 3:.1f} h")

    _calib.write_json(args.out, {
        "experiment": "xp07_e7_cost",
        "question": "what does a 12-epoch recovery actually cost on this board?",
        "why": "every recovered column in XP7 costs this; it had been an estimate",
        "board": "Jetson Orin Nano Super",
        "mode": "plain fine-tune" if args.no_quant else "QAT (fake-quant hooks live)",
        "resolution": _calib.RES, "batch": args.batch,
        "warmup_steps": args.warmup, "timed_steps": n_timed,
        "elapsed_s": round(elapsed, 2),
        "images_per_s": round(img_s, 3),
        "train_split_images": TRAIN_IMAGES,
        "seconds_per_epoch": round(epoch_s, 1),
        "minutes_per_epoch": round(epoch_s / 60, 2),
        "hours_for_12_epochs": round(total_h, 2),
        "hours_for_e7_as_specified": round(total_h * 3, 2),
        "note": "projected from timed steps, not a measured full epoch — stated as a "
                "projection because that is what it is",
    })
    if q is not None:
        q.restore()


if __name__ == "__main__":
    main()
