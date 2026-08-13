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


#: The lecture's importance criteria (Lec03 §"which synapses to prune"), as
#: torch-pruning objects. Names map to sections of the lecture: magnitude (L1/L2
#: structural norms, Wen et al. NeurIPS 2016), scaling (BN gamma reused as the
#: per-channel scaling factor, Liu et al. ICCV 2017), second order (Taylor,
#: Molchanov et al.; Hessian, in the spirit of Optimal Brain Damage), geometric
#: median (FPGM), and layer-adaptive magnitude (LAMP).
#:
#: ``random`` is not a joke entry. It is the control that says how much of a
#: criterion's benefit is really coming from the criterion rather than from the
#: mere act of making the network smaller and retraining it. On a model that
#: collapses as fast as this one does, that control is the most informative run
#: in the set.
_IMPORTANCE_ALIASES = {"group_norm": "l2"}      # XP6's name for the L2 criterion


def _make_importance(name: str):
    """One of the lecture's criteria, by name. See :data:`_IMPORTANCE_ALIASES`."""
    import torch_pruning as tp

    name = _IMPORTANCE_ALIASES.get(name, name)
    table = {
        "l2":      lambda: tp.importance.GroupMagnitudeImportance(p=2),   # XP6 baseline
        "l1":      lambda: tp.importance.GroupMagnitudeImportance(p=1),
        "bn":      lambda: tp.importance.BNScaleImportance(),             # Network Slimming
        "taylor":  lambda: tp.importance.GroupTaylorImportance(),         # needs gradients
        "hessian": lambda: tp.importance.GroupHessianImportance(),        # needs gradients
        "fpgm":    lambda: tp.importance.FPGMImportance(p=2),
        "lamp":    lambda: tp.importance.LAMPImportance(p=2),
        "random":  lambda: tp.importance.RandomImportance(),              # the control
        # XP6 used this one directly; kept so old call sites still resolve.
        "magnitude": lambda: tp.importance.MagnitudeImportance(p=2),
    }
    if name not in table:
        raise KeyError(f"unknown importance {name!r}; have {sorted(table)}")
    return table[name]()


#: Criteria that are statistics of the *data*, not of the weights, and therefore
#: need a backward pass accumulated into ``.grad`` before ``pruner.step()``.
NEEDS_GRADIENTS = frozenset({"taylor", "hessian"})


def accumulate_gradients(model, *, repo, res: int = 512, batch: int = 8,
                         images: int = 512, device: str = "cuda:0") -> dict:
    """Run the real YOLOv5 loss backwards over a slice of *train*, leaving ``.grad``.

    Taylor and Hessian importance read gradients, so they measure how much the
    loss actually moves when a channel is removed rather than how large its
    weights happen to be. That makes them the only criteria here whose answer
    depends on the data, and it means they must never see val or test: the
    gradients come from the training split, through the same loader the recovery
    fine-tune uses, so the criterion is computed on data the model was already
    entitled to.

    Gradients are accumulated (never zeroed between batches) because the
    criterion wants an expectation over the sample, not the last batch's noise.
    """
    import sys

    import torch

    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from utils.loss import ComputeLoss

    from lib.finetune import base_hyp, build_loader

    detect = next(m for m in model.modules() if type(m).__name__ == "Detect")
    hyp = base_hyp(detect.nl, int(detect.nc), res)
    model.hyp, model.gr, model.nc = hyp, 1.0, int(detect.nc)

    loader = build_loader("train", res, batch, repo, hyp, augment=False)
    compute_loss = ComputeLoss(model)

    model.train()                      # the loss needs training-mode Detect outputs
    model.zero_grad(set_to_none=False)
    seen, total = 0, 0.0
    for imgs, targets, _paths, _shapes in loader:
        imgs = imgs.to(device, non_blocking=True).float() / 255.0
        loss, _items = compute_loss(model(imgs), targets.to(device))
        loss.backward()                # accumulate, do NOT zero
        total += float(loss.detach())
        seen += imgs.shape[0]
        if seen >= images:
            break

    model.eval()
    return {"grad_images": seen, "grad_batches": max(1, seen // batch),
            "grad_mean_loss": round(total / max(1, seen // batch), 5),
            "grad_split": "train", "grad_res": res, "grad_augment": False}


def prunable_layers(model) -> list:
    """Every Conv2d that is fair game, in forward order, as (name, module) pairs.

    This is the axis of the sensitivity sweep: prune exactly one of these at a
    time and the resulting curve says which layers this detector can afford to
    lose and which are load bearing. The detect head convolutions are excluded
    for the reason given in the module docstring, and 1-channel-out convolutions
    are excluded because there is nothing to remove without deleting the layer.
    """
    import torch.nn as nn

    banned = {id(m) for m in detect_head_convs(model)}
    return [(name, m) for name, m in model.named_modules()
            if isinstance(m, nn.Conv2d) and id(m) not in banned and m.out_channels > 1]


def model_stats(model, res: int = 640, device: str = "cuda:0") -> dict:
    """Parameters and MACs at a given input size."""
    import torch
    import torch_pruning as tp

    x = torch.randn(1, 3, res, res, device=device)
    macs, params = tp.utils.count_ops_and_params(model, x)
    return {"params_m": params / 1e6, "gmacs": macs / 1e9}


def prune_channels(model, ratio: float, *, res: int = 640, device: str = "cuda:0",
                   importance: str = "group_norm", round_to: int = 1,
                   pruning_ratio_dict: dict | None = None,
                   global_pruning: bool = True, grad_meta: dict | None = None) -> dict:
    """Remove ``ratio`` of channels structurally, and report what actually changed.

    ``ratio`` is a *channel* target, not a FLOPs target. The two differ — early
    layers carry far more compute per channel than late ones — so the achieved
    MAC reduction is measured and returned rather than assumed. PLAN.md asks for
    FLOPs targets; the honest mapping between the two is part of the result.

    The three arguments beyond the ratio are the lecture's other two axes:

    ``importance``
        *Which* channels go. See :func:`_make_importance`. ``taylor`` and
        ``hessian`` read ``.grad``, so :func:`accumulate_gradients` must have run
        on this model first; pass its return value as ``grad_meta`` so the record
        says what the criterion was computed on.
    ``round_to``
        Round surviving channel counts up to a multiple. XP6 found a *larger*
        pruned model running *faster* than a smaller one and guessed channel-count
        regularity was the cause; this is the knob that tests it. Structural only,
        the verdict is a throughput measurement on the board.
    ``pruning_ratio_dict`` / ``global_pruning``
        The ratio axis. Global (one threshold, per-layer ratios fall out) is what
        XP6 did. A dict assigns per-layer ratios explicitly, which is how both the
        single-layer sensitivity sweep and a sensitivity-driven allocation are
        expressed; those want ``global_pruning=False`` so each layer gets exactly
        the ratio it was given.
    """
    import torch

    model.eval()
    example = torch.randn(1, 3, res, res, device=device)
    before = model_stats(model, res, device)

    imp = _make_importance(importance)
    if importance in NEEDS_GRADIENTS and not any(
            p.grad is not None for p in model.parameters()):
        raise RuntimeError(
            f"{importance!r} importance reads gradients but none are present — call "
            f"accumulate_gradients(model, repo=...) before pruning, or the criterion "
            f"silently degenerates to a constant.")

    pruner = _build_pruner(model, example, imp, ratio, round_to,
                           pruning_ratio_dict, global_pruning)
    pruner.step()

    after = model_stats(model, res, device)
    out = {
        "requested_channel_ratio": ratio,
        "importance": importance,
        "round_to": round_to,
        "global_pruning": global_pruning,
        "per_layer_ratios": bool(pruning_ratio_dict),
        "params_m_before": round(before["params_m"], 4),
        "params_m_after": round(after["params_m"], 4),
        "params_reduction": round(1 - after["params_m"] / before["params_m"], 4),
        "gmacs_before": round(before["gmacs"], 3),
        "gmacs_after": round(after["gmacs"], 3),
        "macs_reduction": round(1 - after["gmacs"] / before["gmacs"], 4),
    }
    if grad_meta:
        out |= grad_meta
    return out


def _build_pruner(model, example, imp, ratio: float, round_to: int,
                  pruning_ratio_dict: dict | None, global_pruning: bool,
                  iterative_steps: int = 1):
    """The one place MetaPruner is constructed, so every arm shares its settings.

    ``ignored_layers`` is not optional and not a tuning knob. Each of YOLOv5's
    three head convolutions emits exactly ``anchors x (5 + num_classes)``
    channels; that number is fixed by the output format, and pruning it leaves
    the decode arithmetic describing a tensor that no longer exists.
    """
    import torch_pruning as tp

    kwargs = dict(
        importance=imp,
        pruning_ratio=ratio,
        ignored_layers=detect_head_convs(model),   # never the head, see docstring
        global_pruning=global_pruning,
        round_to=round_to,
        iterative_steps=iterative_steps,
    )
    if pruning_ratio_dict:
        kwargs["pruning_ratio_dict"] = pruning_ratio_dict
    return tp.pruner.MetaPruner(model, example, **kwargs)


def prune_iterative(model, ratio: float, *, steps: int, recover, res: int = 640,
                    device: str = "cuda:0", importance: str = "group_norm",
                    round_to: int = 1, pruning_ratio_dict: dict | None = None,
                    global_pruning: bool = True, final_recover=None) -> dict:
    """Reach ``ratio`` in ``steps`` increments, retraining between each.

    One-shot pruning removes everything in a single operation and only then lets
    the network train. Iterative pruning removes a slice, gives the survivors a
    chance to redistribute the work, and repeats. The literature generally finds
    the second preserves far more accuracy at the same final sparsity, and XP6's
    damage curve suggests why it should matter here: this model loses 9 mAP50
    points to a **2%** one-shot cut and 88% to a 5% cut, which is far steeper than
    typical published curves — the signature of survivors never getting a chance
    to compensate.

    ``recover`` is called as ``recover(model)`` after each cut. Splitting the same
    total training budget across the steps is what makes the comparison to
    one-shot fair: the two arms differ in *when* the training happens, not in how
    much of it there is.

    **That framing has a confound, and ``final_recover`` is the fix.** Equal
    *total* budget is not equal *recovery* budget. XP6 gave both arms 12 epochs,
    but the iterative model's final architecture only existed for the last 4 of
    them, while the one-shot model trained all 12 in its final shape, so part of
    iterative's deficit may be nothing but a shorter run in the shape being
    measured. Pass ``final_recover`` to give the last cut its own full budget, and
    the two arms then differ only in how the channels were removed.
    """
    import torch

    example = torch.randn(1, 3, res, res, device=device)
    before = model_stats(model, res, device)

    imp = _make_importance(importance)

    model.eval()
    pruner = _build_pruner(model, example, imp, ratio, round_to,
                           pruning_ratio_dict, global_pruning, iterative_steps=steps)

    per_step = []
    for i in range(steps):
        pruner.step()
        mid = model_stats(model, res, device)
        print(f"  iterative step {i+1}/{steps}: {mid['params_m']:.3f} M params, "
              f"{mid['gmacs']:.2f} GMACs", flush=True)
        last = i == steps - 1
        (final_recover if last and final_recover else recover)(model)
        per_step.append({"step": i + 1, "params_m": round(mid["params_m"], 4),
                         "gmacs": round(mid["gmacs"], 3),
                         "recovery": "final" if last and final_recover else "between"})

    after = model_stats(model, res, device)
    return {
        "method": "iterative",
        "steps": steps,
        "requested_channel_ratio": ratio,
        "importance": importance,
        "round_to": round_to,
        "global_pruning": global_pruning,
        "equal_post_cut_budget": final_recover is not None,
        "params_m_before": round(before["params_m"], 4),
        "params_m_after": round(after["params_m"], 4),
        "params_reduction": round(1 - after["params_m"] / before["params_m"], 4),
        "gmacs_before": round(before["gmacs"], 3),
        "gmacs_after": round(after["gmacs"], 3),
        "macs_reduction": round(1 - after["gmacs"] / before["gmacs"], 4),
        "per_step": per_step,
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


__all__ = ["load_yolov5", "detect_head_convs", "model_stats", "prune_channels",
           "prune_iterative", "save_pruned", "accumulate_gradients",
           "prunable_layers", "NEEDS_GRADIENTS"]
