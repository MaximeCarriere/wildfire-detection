"""Minimal on-device training loop, for pruning recovery (XP6).

YOLOv5's own ``train.py`` cannot be used here: it builds the network from an
architecture file and loads a state_dict into it, and a *pruned* network no longer
matches its architecture file — the channel counts have changed. So this is a
compact loop over YOLOv5's own dataloader and loss, which keeps the augmentation
and the target-assignment identical to how the baseline was trained while
accepting an arbitrary live model object.

It is deliberately small. Everything that decides a *result* — accuracy, latency,
power — still goes through ``lib.evaluator``; this file only has to move weights.

Measured cost on this board (XP6, 512 px, batch 8): ~7 min per epoch over 15,500
images, so a 20-epoch recovery is roughly 2.5 hours. That is the whole reason
pruning is feasible here and the distillation ladder is not.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path


def _yolo_paths_file(split: str, out: Path, data_root: Path) -> Path:
    """YOLOv5's dataloader takes a text file of image paths; ours are relative."""
    from lib import data as dataset

    paths = dataset.load_split(split)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(str(p) for p in paths) + "\n")
    return out


def build_loader(split: str, res: int, batch: int, repo: Path, *, augment: bool,
                 workers: int = 4, cache_dir: Path | None = None):
    """A YOLOv5 dataloader over one of our frozen splits."""
    from lib import data as dataset

    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from utils.dataloaders import create_dataloader

    cache_dir = cache_dir or (dataset.DATA / "splits" / "yolo")
    listing = _yolo_paths_file(split, cache_dir / f"{split}.txt", dataset.DATA)

    hyp = {
        # YOLOv5's default fine-tuning hyper-parameters, with augmentation
        # deliberately mild: this is a *recovery* fine-tune of an already-trained
        # model, not training from scratch, so heavy mosaic/mixup mostly adds noise.
        "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
        "degrees": 0.0, "translate": 0.1, "scale": 0.5, "shear": 0.0,
        "perspective": 0.0, "flipud": 0.0, "fliplr": 0.5,
        "mosaic": 0.5 if augment else 0.0, "mixup": 0.0, "copy_paste": 0.0,
        "box": 0.05, "cls": 0.5, "cls_pw": 1.0, "obj": 1.0, "obj_pw": 1.0,
        "iou_t": 0.20, "anchor_t": 4.0, "fl_gamma": 0.0, "label_smoothing": 0.0,
    }
    loader, _ = create_dataloader(
        str(listing), res, batch, 32, single_cls=False, hyp=hyp,
        augment=augment, cache=None, rect=not augment, workers=workers,
        prefix=f"{split}: ", shuffle=augment,
    )
    return loader, hyp


def finetune(model, *, repo: Path, epochs: int, res: int = 512, batch: int = 8,
             lr0: float = 0.0032, device: str = "cuda:0", log_every: int = 50,
             on_epoch=None) -> dict:
    """Recovery fine-tune. Returns a short training history.

    The learning rate is an order of magnitude below from-scratch training and
    decays cosine to near zero: the network already knows the task and is being
    repaired, not taught.
    """
    import torch
    from torch.amp import GradScaler, autocast

    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from utils.loss import ComputeLoss

    loader, hyp = build_loader("train", res, batch, repo, augment=True)

    model.to(device).train()
    model.hyp = hyp                       # ComputeLoss reads gains off the model
    model.gr = 1.0
    nc = int(getattr(model, "nc", 2))
    model.nc = nc

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr0, momentum=0.937, weight_decay=5e-4, nesterov=True)
    compute_loss = ComputeLoss(model)
    scaler = GradScaler("cuda")

    nb = len(loader)
    total_steps = max(1, epochs * nb)
    history = []
    step = 0
    t_start = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        running, seen = 0.0, 0
        t_epoch = time.perf_counter()
        for i, (imgs, targets, _paths, _shapes) in enumerate(loader):
            imgs = imgs.to(device, non_blocking=True).float() / 255.0
            targets = targets.to(device)

            # Cosine decay to ~0, with a short warm-up: a pruned network's first
            # gradients are large and a flat high LR at step 0 undoes the surgery.
            frac = step / total_steps
            warm = min(1.0, (step + 1) / max(1, int(0.03 * total_steps)))
            for g in opt.param_groups:
                g["lr"] = lr0 * warm * (0.5 * (1 + math.cos(math.pi * frac)))

            with autocast("cuda"):
                pred = model(imgs)
                loss, _items = compute_loss(pred, targets)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running += float(loss.detach()) * imgs.shape[0]
            seen += imgs.shape[0]
            step += 1
            if log_every and i % log_every == 0:
                print(f"    epoch {epoch+1}/{epochs} step {i}/{nb} "
                      f"loss {running/max(seen,1):.4f} lr {opt.param_groups[0]['lr']:.5f}",
                      flush=True)

        entry = {"epoch": epoch + 1, "loss": round(running / max(seen, 1), 5),
                 "minutes": round((time.perf_counter() - t_epoch) / 60, 2)}
        history.append(entry)
        print(f"  epoch {epoch+1}/{epochs}: loss {entry['loss']} "
              f"({entry['minutes']} min)", flush=True)
        if on_epoch:
            on_epoch(epoch + 1, model)

    model.eval()
    return {"epochs": epochs, "res": res, "batch": batch, "lr0": lr0,
            "total_minutes": round((time.perf_counter() - t_start) / 60, 2),
            "history": history}


__all__ = ["build_loader", "finetune"]
