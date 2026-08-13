"""On-device training loop for pruning recovery (XP6).

YOLOv5's own ``train.py`` cannot be used here: it rebuilds the network from an
architecture file and loads a state_dict into it, and a *pruned* network no longer
matches that file — the channel counts have changed. So this is a compact loop
over YOLOv5's dataloader, loss, optimizer and EMA, accepting an arbitrary live
model object.

**It is a port, not a re-invention, and that distinction was learned the hard
way.** A first version wrote its own optimizer schedule and skipped the parts of
train.py that looked incidental. Measured with a control — one epoch on the
*unpruned* model, which scores 0.818 — it produced **0.008**. It destroyed a
working model, and the two pruning-recovery numbers taken with it were therefore
measurements of the loop, not of pruning. Three omissions, each sufficient alone:

* **No gradient accumulation.** YOLOv5's hyper-parameters assume a nominal batch
  of 64 and accumulate ``64/batch`` steps before each optimizer step. Stepping on
  every batch of 8 applies 8x as many updates, each from an eighth of the data,
  at the full step size.
* **Unscaled loss gains.** YOLOv5 scales them to the dataset: the classification
  gain is ``cls x nc/80``, which for 2 classes is 0.0125, not the raw 0.5 — 40x
  too heavy. The term asking "fire or smoke?" drowned out the terms asking "is
  anything there, and where?", and objectness is exactly what collapsed.
* **No EMA.** YOLOv5's published accuracy comes from an exponential moving
  average of the weights. Without it you evaluate whatever noisy point the last
  small-batch update landed on.

Everything that decides a *result* still goes through ``lib.evaluator``; this file
only moves weights. ``sanity_check_unpruned`` in the XP6 folder is the control
that must pass before any recovery number is believed.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

NOMINAL_BATCH = 64      # the batch size YOLOv5's hyper-parameters assume


def _yolo_paths_file(split: str, out: Path) -> Path:
    """YOLOv5's dataloader takes a text file of image paths; ours are relative."""
    from lib import data as dataset

    paths = dataset.load_split(split)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(str(p) for p in paths) + "\n")
    return out


def base_hyp(nl: int, nc: int, res: int) -> dict:
    """YOLOv5's fine-tuning hyper-parameters, scaled the way train.py scales them.

    The scaling is not cosmetic — see the module docstring for what skipping it
    costs. Augmentation is deliberately milder than from-scratch training: this
    repairs an already-trained network rather than teaching one.
    """
    hyp = {
        "lr0": 0.0032, "lrf": 0.12, "momentum": 0.937, "weight_decay": 5e-4,
        "warmup_epochs": 2.0, "warmup_momentum": 0.5, "warmup_bias_lr": 0.05,
        "box": 0.05, "cls": 0.5, "cls_pw": 1.0, "obj": 1.0, "obj_pw": 1.0,
        "iou_t": 0.20, "anchor_t": 4.0, "fl_gamma": 0.0, "label_smoothing": 0.0,
        "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
        "degrees": 0.0, "translate": 0.1, "scale": 0.5, "shear": 0.0,
        "perspective": 0.0, "flipud": 0.0, "fliplr": 0.5,
        "mosaic": 0.5, "mixup": 0.0, "copy_paste": 0.0,
    }
    hyp["box"] *= 3 / nl
    hyp["cls"] *= nc / 80 * 3 / nl
    hyp["obj"] *= (res / 640) ** 2 * 3 / nl
    return hyp


def build_loader(split: str, res: int, batch: int, repo: Path, hyp: dict, *,
                 augment: bool, workers: int = 4):
    from lib import data as dataset

    # Refuse to train against splits that do not match the frozen manifest. This
    # matters most when training moves to another machine: weights can be made
    # anywhere, but if the split membership changes there, test images leak into
    # training and every published number silently becomes a comparison against a
    # model that has seen the exam paper.
    dataset.verify_splits()

    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from utils.dataloaders import create_dataloader

    listing = _yolo_paths_file(split, dataset.DATA / "splits" / "yolo" / f"{split}.txt")
    loader, _ = create_dataloader(
        str(listing), res, batch, 32, single_cls=False, hyp=hyp,
        augment=augment, cache=None, rect=not augment, workers=workers,
        prefix=f"{split}: ", shuffle=augment,
    )
    return loader


def finetune(model, *, repo: Path, epochs: int, res: int = 512, batch: int = 8,
             device: str = "cuda:0", log_every: int = 200, lr0: float | None = None) -> dict:
    """Recovery fine-tune. Returns the EMA weights in ``model`` and a history."""
    import torch
    from torch.amp import GradScaler, autocast

    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from utils.loss import ComputeLoss
    from utils.torch_utils import ModelEMA, smart_optimizer

    detect = next(m for m in model.modules() if type(m).__name__ == "Detect")
    nl, nc = detect.nl, int(detect.nc)
    hyp = base_hyp(nl, nc, res)
    if lr0 is not None:
        hyp["lr0"] = lr0

    loader = build_loader("train", res, batch, repo, hyp, augment=True)
    nb = len(loader)

    model.to(device)
    model.hyp = hyp
    model.gr = 1.0
    model.nc = nc

    # Nominal-batch accumulation, exactly as train.py does it.
    accumulate = max(round(NOMINAL_BATCH / batch), 1)
    hyp["weight_decay"] *= batch * accumulate / NOMINAL_BATCH
    opt = smart_optimizer(model, "SGD", hyp["lr0"], hyp["momentum"], hyp["weight_decay"])
    ema = ModelEMA(model)
    compute_loss = ComputeLoss(model)
    scaler = GradScaler("cuda")

    def lf(ep):                                    # linear decay to lrf
        return (1 - ep / epochs) * (1.0 - hyp["lrf"]) + hyp["lrf"]

    nw = max(round(hyp["warmup_epochs"] * nb), 100)
    history = []
    t_start = time.perf_counter()
    last_opt_step = -1

    for epoch in range(epochs):
        model.train()
        running, seen = 0.0, 0
        t_epoch = time.perf_counter()
        opt.zero_grad(set_to_none=True)

        for i, (imgs, targets, _paths, _shapes) in enumerate(loader):
            ni = i + nb * epoch
            imgs = imgs.to(device, non_blocking=True).float() / 255.0

            # Warm-up: ramp accumulation, per-group LR and momentum together.
            if ni <= nw:
                xi = [0, nw]
                accumulate = max(1, int(_interp(ni, xi, [1, NOMINAL_BATCH / batch])))
                for j, g in enumerate(opt.param_groups):
                    g["lr"] = _interp(ni, xi,
                                      [hyp["warmup_bias_lr"] if j == 0 else 0.0,
                                       hyp["lr0"] * lf(epoch)])
                    if "momentum" in g:
                        g["momentum"] = _interp(ni, xi,
                                                [hyp["warmup_momentum"], hyp["momentum"]])

            with autocast("cuda"):
                loss, _items = compute_loss(model(imgs), targets.to(device))

            scaler.scale(loss).backward()

            if ni - last_opt_step >= accumulate:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                ema.update(model)
                last_opt_step = ni

            running += float(loss.detach()) * imgs.shape[0]
            seen += imgs.shape[0]
            if log_every and i % log_every == 0:
                print(f"    epoch {epoch+1}/{epochs} step {i}/{nb} "
                      f"loss {running/max(seen,1):.4f} lr {opt.param_groups[0]['lr']:.5f} "
                      f"accum {accumulate}", flush=True)

        for g in opt.param_groups:                 # end-of-epoch schedule step
            g["lr"] = hyp["lr0"] * lf(epoch + 1)

        entry = {"epoch": epoch + 1, "loss": round(running / max(seen, 1), 5),
                 "minutes": round((time.perf_counter() - t_epoch) / 60, 2)}
        history.append(entry)
        print(f"  epoch {epoch+1}/{epochs}: loss {entry['loss']} "
              f"({entry['minutes']} min)", flush=True)

    # Hand back the EMA weights — that is what YOLOv5 evaluates and publishes.
    model.load_state_dict(ema.ema.state_dict())
    model.eval()
    return {"epochs": epochs, "res": res, "batch": batch, "hyp": hyp,
            "accumulate": accumulate, "ema": True,
            "total_minutes": round((time.perf_counter() - t_start) / 60, 2),
            "history": history}


def _interp(x, xp, fp):
    import numpy as np
    return float(np.interp(x, xp, fp))


__all__ = ["base_hyp", "build_loader", "finetune"]
