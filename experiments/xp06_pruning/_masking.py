"""Weight masking for the two granularities that do not remove channels.

Channel pruning makes a genuinely smaller network. Fine-grained (E5) and 2:4
(E4) do not: they set individual weights to zero and leave the tensor its
original shape. That difference drives everything here.

**The trap this module exists to avoid.** A mask applied once, before training,
does not survive training. Gradients flow into the masked positions, momentum
carries them, and after a few hundred steps the "pruned" weights are back to
ordinary small non-zero numbers. The measurement then reports the accuracy of a
model that was never actually sparse, which is a silent failure of exactly the
kind XP6's write-up already catalogues. So the mask is re-applied through a
forward pre-hook, on every forward pass, for the whole of training and
evaluation. Sparsity is asserted afterwards rather than assumed.

Parameter counts here are **non-zero** counts. Reporting 7.03 M for a 90% sparse
model would be true of the file and useless as a description of the model, and
reporting 0.7 M would imply a smaller network that does not exist. Both numbers
are recorded, and the README says which is which.

No speed number is produced from either granularity. Scattered zeros have no
matching kernels on this hardware, and 2:4's hardware path exists only on the
board.
"""
from __future__ import annotations

import torch
import torch.nn as nn

_HOOK_ATTR = "_prune_mask_handle"


def prunable_conv_weights(model, exclude: set[int] | None = None):
    """Conv2d weights fair game for masking, skipping the detect head."""
    exclude = exclude or set()
    return [(n, m) for n, m in model.named_modules()
            if isinstance(m, nn.Conv2d) and id(m) not in exclude]


def _attach(module: nn.Module, mask: torch.Tensor) -> None:
    """Re-apply the mask before every forward. See the module docstring."""
    module.register_buffer("_prune_mask", mask, persistent=False)
    if getattr(module, _HOOK_ATTR, None) is not None:
        getattr(module, _HOOK_ATTR).remove()

    def pre_hook(mod, _inp):
        mod.weight.data.mul_(mod._prune_mask)

    setattr(module, _HOOK_ATTR, module.register_forward_pre_hook(pre_hook))
    module.weight.data.mul_(mask)


def apply_unstructured(model, sparsity: float, exclude: set[int] | None = None) -> dict:
    """Global magnitude pruning: one threshold over every prunable weight.

    Global rather than per-layer because that is what the fine-grained result in
    the lecture does, and because it lets the network decide which layers can
    spare the most, which is the whole appeal of the granularity.
    """
    convs = prunable_conv_weights(model, exclude)
    flat = torch.cat([m.weight.detach().abs().flatten() for _, m in convs])
    total = flat.numel()
    k = int(sparsity * total)
    if k < 1:
        raise ValueError(f"sparsity {sparsity} removes nothing from {total} weights")
    threshold = torch.kthvalue(flat.float().cpu(), k).values.item()

    zeroed = 0
    for _, m in convs:
        mask = (m.weight.detach().abs() > threshold).to(m.weight.dtype)
        zeroed += int((mask == 0).sum())
        _attach(m, mask)

    return {"granularity": "unstructured",
            "requested_sparsity": sparsity,
            "achieved_sparsity": round(zeroed / total, 5),
            "threshold": threshold,
            "prunable_weights": total,
            "zeroed_weights": zeroed}


def apply_2to4(model, exclude: set[int] | None = None) -> dict:
    """Keep the two largest of every four contiguous weights.

    The grouping runs along the flattened input dimension of each filter, which
    is the layout NVIDIA's sparse tensor cores read. Layers whose filter size is
    not divisible by four are left dense and reported, rather than being quietly
    padded into a pattern the hardware would not accept.
    """
    convs = prunable_conv_weights(model, exclude)
    zeroed = total = 0
    skipped = []
    for name, m in convs:
        w = m.weight.detach()
        out_c = w.shape[0]
        rest = w[0].numel()
        if rest % 4 != 0:
            skipped.append(f"{name}({rest})")
            continue
        g = w.reshape(out_c, rest // 4, 4).abs()
        keep = torch.zeros_like(g, dtype=torch.bool)
        idx = g.argsort(dim=-1, descending=True)[..., :2]
        keep.scatter_(-1, idx, True)
        mask = keep.reshape(w.shape).to(w.dtype)
        zeroed += int((mask == 0).sum())
        total += mask.numel()
        _attach(m, mask)

    return {"granularity": "2:4",
            "achieved_sparsity": round(zeroed / max(total, 1), 5),
            "patterned_weights": total,
            "zeroed_weights": zeroed,
            "layers_left_dense": skipped,
            "n_layers_left_dense": len(skipped)}


def reapply_masks(model) -> dict:
    """Re-impose the masks on the *final* weights, and report the drift first.

    This exists because of a genuine interaction with the training loop, and the
    drift number is worth keeping rather than hiding. ``lib.finetune`` finishes by
    loading the EMA weights into the model, and the EMA is a separate copy that
    never runs a forward pass, so the pre-forward hooks that enforce sparsity
    never fire on it. It averages the raw post-optimizer-step parameters, which
    are non-zero in the masked positions, and then overwrites the trained model
    with them.

    The training itself was genuinely sparse: every forward pass ran on masked
    weights, so the network learned to work without them. Only the exported
    average needed the mask re-imposed. Returns the sparsity measured *before*
    re-imposing, which is the honest record of how far the EMA drifted.
    """
    drift_zero = drift_total = 0
    for _, m in model.named_modules():
        if isinstance(m, nn.Conv2d) and hasattr(m, "_prune_mask"):
            w = m.weight.detach()
            drift_zero += int((w == 0).sum())
            drift_total += w.numel()
            m.weight.data.mul_(m._prune_mask)
    return {"ema_sparsity_before_remask": round(drift_zero / max(drift_total, 1), 5)}


def strip_masks(model) -> int:
    """Remove the hooks and buffers, leaving the zeros baked into the weights.

    Two reasons this has to happen before the checkpoint is written. The hooks
    close over a Python function, and ``save_pruned`` stores the whole model
    object, so pickling would fail on the closure. And a checkpoint that only
    becomes sparse when a hook happens to be attached is a checkpoint whose
    sparsity is not really in the weights. Call this *after* verifying, never
    before: it removes the thing that was keeping the model honest.
    """
    removed = 0
    for _, m in model.named_modules():
        h = getattr(m, _HOOK_ATTR, None)
        if h is not None:
            m.weight.data.mul_(m._prune_mask)      # final application
            h.remove()
            setattr(m, _HOOK_ATTR, None)
            removed += 1
        if hasattr(m, "_prune_mask"):
            del m._prune_mask
    return removed


def verify_sparsity(model) -> dict:
    """Measure what is actually zero, after everything. Never trust the plan."""
    zero = total = 0
    masked_layers = 0
    for _, m in model.named_modules():
        if isinstance(m, nn.Conv2d) and hasattr(m, "_prune_mask"):
            masked_layers += 1
            w = m.weight.detach()
            zero += int((w == 0).sum())
            total += w.numel()
    nonzero_params = sum(int((p.detach() != 0).sum()) for p in model.parameters())
    return {"masked_layers": masked_layers,
            "measured_sparsity": round(zero / max(total, 1), 5),
            "nonzero_params_m": round(nonzero_params / 1e6, 4),
            "dense_params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 4)}
