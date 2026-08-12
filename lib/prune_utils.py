"""Structured channel pruning for YOLOv5, via torch-pruning (XP6).

Structured pruning removes whole channels, so the resulting network is genuinely
smaller and denser rather than a sparse mask over the original — which is what
makes it a real speed candidate on this hardware, and what makes it awkward: a
detection backbone is full of residual adds and concatenations that force groups
of channels to be pruned together. ``torch_pruning``'s dependency graph works that
out; our job is to tell it what must not be touched.

**The detection head outputs are never pruned.** Each of YOLOv5's three head
convolutions emits exactly ``anchors x (5 + num_classes)`` channels, and that
number is fixed by the output format — prune it and the decode arithmetic no
longer describes the tensor it is given. Everything upstream is fair game.

Pruning damage is reported as measured accuracy, never as a proxy. XP2 and XP9
both showed proxies (FLOPs, parameter counts) diverging from what the board
actually does, so this module reports the structural change and leaves the
verdict to ``lib.evaluator``.
"""
from __future__ import annotations

from pathlib import Path


def load_yolov5(weights: str | Path, repo: Path, device: str = "cuda:0"):
    """Load an original-YOLOv5 checkpoint as a live, trainable model."""
    import sys

    import torch

    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    model = (ckpt.get("ema") or ckpt["model"]).float()
    for p in model.parameters():          # checkpoints ship frozen; pruning needs grads
        p.requires_grad_(True)
    return model.to(device)


def detect_head_convs(model) -> list:
    """The final per-stride convolutions, whose channel count is protocol-fixed."""
    import torch.nn as nn

    detect = None
    for m in model.modules():
        if type(m).__name__ == "Detect":
            detect = m
    if detect is None:
        raise RuntimeError("no Detect module found — is this a YOLOv5 detection model?")
    return [m for m in detect.modules() if isinstance(m, nn.Conv2d)]


def model_stats(model, res: int = 640, device: str = "cuda:0") -> dict:
    """Parameters and MACs at a given input size."""
    import torch
    import torch_pruning as tp

    x = torch.randn(1, 3, res, res, device=device)
    macs, params = tp.utils.count_ops_and_params(model, x)
    return {"params_m": params / 1e6, "gmacs": macs / 1e9}


def prune_channels(model, ratio: float, *, res: int = 640, device: str = "cuda:0",
                   importance: str = "group_norm") -> dict:
    """Remove ``ratio`` of channels structurally, and report what actually changed.

    ``ratio`` is a *channel* target, not a FLOPs target. The two differ — early
    layers carry far more compute per channel than late ones — so the achieved
    MAC reduction is measured and returned rather than assumed. PLAN.md asks for
    FLOPs targets; the honest mapping between the two is part of the result.
    """
    import torch
    import torch_pruning as tp

    model.eval()
    example = torch.randn(1, 3, res, res, device=device)
    before = model_stats(model, res, device)

    # torch-pruning 1.6 renamed GroupNormImportance -> GroupMagnitudeImportance.
    imp = {"group_norm": tp.importance.GroupMagnitudeImportance(p=2),
           "magnitude": tp.importance.MagnitudeImportance(p=2)}[importance]

    pruner = tp.pruner.MetaPruner(
        model, example,
        importance=imp,
        pruning_ratio=ratio,
        ignored_layers=detect_head_convs(model),   # see module docstring
        global_pruning=True,        # spend the budget where it costs least accuracy
    )
    pruner.step()

    after = model_stats(model, res, device)
    return {
        "requested_channel_ratio": ratio,
        "params_m_before": round(before["params_m"], 4),
        "params_m_after": round(after["params_m"], 4),
        "params_reduction": round(1 - after["params_m"] / before["params_m"], 4),
        "gmacs_before": round(before["gmacs"], 3),
        "gmacs_after": round(after["gmacs"], 3),
        "macs_reduction": round(1 - after["gmacs"] / before["gmacs"], 4),
    }


def save_pruned(model, path: Path, meta: dict | None = None) -> Path:
    """Save the whole model object, not a state_dict.

    A pruned network has different channel counts from its architecture file, so
    a state_dict cannot be reloaded into a fresh Model(cfg) — the shapes no longer
    match. This mirrors how YOLOv5 stores checkpoints.
    """
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.half().float(), "pruned_meta": meta or {}}, path)
    return path


__all__ = ["load_yolov5", "detect_head_convs", "model_stats", "prune_channels", "save_pruned"]
