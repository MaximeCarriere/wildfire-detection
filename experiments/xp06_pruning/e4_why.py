#!/usr/bin/env python3
"""XP6 E4c — why does free-form 50% survive and patterned 50% collapse?

E4 measures *that* a 2:4 cut scores 0.0000 where an unstructured cut of the same
size scores 0.7622. This measures *why*, so the README explains a mechanism
instead of asserting a mystery.

Both cuts delete the same count of weights from the same layers by magnitude. The
only difference is the scope of the comparison: one global threshold, or a quota
inside every group of four. That difference is measurable in three ways, and all
three are recorded here:

* **What gets deleted.** A global threshold cannot remove a weight larger than
  itself. A per-group quota can remove any weight whose three neighbours happen
  to be larger, however important it is.
* **Where it gets deleted.** A global threshold spends the budget wherever it is
  cheapest, so per-layer sparsity varies enormously. 2:4 is uniform by
  construction, including in the layers E1 showed are fragile.
* **What breaks first.** XP6 found repeatedly that this detector fails through
  its confidence head rather than by degrading gracefully. Objectness is measured
  for both cuts so the exact-zero mAP is explained rather than left as a curiosity.

No accuracy number is produced here; that is E4's job through the frozen harness.
This writes one side-data JSON with no ``model_id``, so it is evidence for the
README and is correctly ignored by the figure builder.

Usage
    python e4_why.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch                                        # noqa: E402
import torch.nn as nn                               # noqa: E402

from lib import data as dataset                     # noqa: E402
from lib.detectors import YOLOV5_REPO               # noqa: E402
from lib.prune_utils import detect_head_convs, load_yolov5   # noqa: E402

import _masking                                     # noqa: E402

RAW = REPO / "results" / "raw"
WEIGHTS = REPO / "weights"
BASE = "yolov5s"
RES = 512
SPARSITY = 0.5
N_OBJ_IMAGES = 48


def log(m: str) -> None:
    print(f"[e4why] {m}", flush=True)


def _convs(model):
    exclude = {id(c) for c in detect_head_convs(model)}
    return [(n, m) for n, m in model.named_modules()
            if isinstance(m, nn.Conv2d) and id(m) not in exclude]


def deletion_stats() -> dict:
    """Which weights each rule deletes, and from where."""
    model = load_yolov5(WEIGHTS / f"{BASE}.pt", YOLOV5_REPO, device="cpu")
    convs = _convs(model)

    flat = torch.cat([m.weight.detach().abs().flatten() for _, m in convs]).float()
    total = flat.numel()
    thr = torch.kthvalue(flat, int(SPARSITY * total)).values.item()
    p90 = torch.kthvalue(flat, int(0.90 * total)).values.item()
    p75 = torch.kthvalue(flat, int(0.75 * total)).values.item()

    removed = {"free": [], "pattern": []}
    per_layer = []
    for name, m in convs:
        a = m.weight.detach().float().abs()
        keep_free = a > thr

        out_c, rest = a.shape[0], a[0].numel()
        g = a.reshape(out_c, rest // 4, 4)
        idx = g.argsort(dim=-1, descending=True)[..., :2]
        keep_pat = torch.zeros_like(g, dtype=torch.bool).scatter_(
            -1, idx, True).reshape(a.shape)

        removed["free"].append(a[~keep_free])
        removed["pattern"].append(a[~keep_pat])
        per_layer.append({
            "layer": name, "weights": int(a.numel()),
            "mean_abs_w": round(a.mean().item(), 6),
            "sparsity_free": round(1 - keep_free.float().mean().item(), 5),
            "sparsity_pattern": round(1 - keep_pat.float().mean().item(), 5),
        })

    out = {"prunable_weights": total, "global_threshold": round(thr, 8),
           "global_p75": round(p75, 6), "global_p90": round(p90, 6),
           "per_layer": per_layer}
    energy = (flat ** 2).sum()
    for key, chunks in removed.items():
        R = torch.cat(chunks)
        out[key] = {
            "removed": int(R.numel()),
            "removed_frac": round(R.numel() / total, 5),
            "max_abs_removed": round(R.max().item(), 6),
            "mean_abs_removed": round(R.mean().item(), 8),
            "l1_mass_removed_frac": round((R.sum() / flat.sum()).item(), 5),
            "l2_energy_removed_frac": round(((R ** 2).sum() / energy).item(), 5),
            "removed_above_p90": int((R > p90).sum()),
            "removed_above_p75": int((R > p75).sum()),
        }
    sp = [r["sparsity_free"] for r in per_layer]
    out["free"]["per_layer_sparsity_min"] = round(min(sp), 5)
    out["free"]["per_layer_sparsity_max"] = round(max(sp), 5)
    out["pattern"]["per_layer_sparsity_min"] = round(
        min(r["sparsity_pattern"] for r in per_layer), 5)
    out["pattern"]["per_layer_sparsity_max"] = round(
        max(r["sparsity_pattern"] for r in per_layer), 5)
    del model
    return out


def objectness_stats() -> dict:
    """What the confidence head does under each cut.

    mAP going to exactly 0.0000 could mean the network was destroyed or that its
    boxes fell under the detection threshold. Those are very different claims and
    only one of them is consistent with retraining recovering the accuracy, so it
    is measured rather than inferred.
    """
    import cv2
    import numpy as np
    sys.path.insert(0, str(YOLOV5_REPO))
    from utils.augmentations import letterbox

    samples = [s for s in dataset.load_samples("test") if s.boxes][:N_OBJ_IMAGES]
    ims = []
    for s in samples:
        im = cv2.imread(str(s.image))
        lb, _, _ = letterbox(im, (RES, RES), stride=32, auto=False)
        ims.append(lb[..., ::-1].transpose(2, 0, 1).copy())
    x = torch.from_numpy(np.stack(ims)).float().div(255).cuda()

    out = {"n_images": len(samples)}
    for name, apply in (("unpruned", None),
                        ("free", lambda m, e: _masking.apply_unstructured(m, SPARSITY, e)),
                        ("pattern", lambda m, e: _masking.apply_2to4(m, e))):
        model = load_yolov5(WEIGHTS / f"{BASE}.pt", YOLOV5_REPO)
        if apply is not None:
            apply(model, {id(c) for c in detect_head_convs(model)})
        model.cuda().eval()
        with torch.no_grad():
            obj = model(x)[0][..., 4]
        out[name] = {"max_objectness": round(obj.max().item(), 6),
                     "mean_objectness": round(obj.mean().item(), 8)}
        log(f"  objectness {name}: max {out[name]['max_objectness']}")
        del model
        torch.cuda.empty_cache()
    return out


def main() -> None:
    log("measuring which weights each rule deletes ...")
    dele = deletion_stats()
    log(f"  free: max removed {dele['free']['max_abs_removed']}, "
        f"L2 energy {dele['free']['l2_energy_removed_frac']:.1%}, "
        f"per-layer {dele['free']['per_layer_sparsity_min']:.1%}"
        f"-{dele['free']['per_layer_sparsity_max']:.1%}")
    log(f"  2:4 : max removed {dele['pattern']['max_abs_removed']}, "
        f"L2 energy {dele['pattern']['l2_energy_removed_frac']:.1%}, "
        f"above p90 {dele['pattern']['removed_above_p90']}")

    log("measuring the confidence head ...")
    obj = objectness_stats()

    rec = {"experiment": "xp06e4c_mechanism", "base_model": BASE, "input_res": RES,
           "sparsity": SPARSITY, "deletion": dele, "objectness": obj,
           "notes": "XP6 E4c: why an unstructured 50% cut survives and a 2:4 cut of the "
                    "same size collapses. Side data for the E4 README section; no model_id, "
                    "so analysis/make_figures.py correctly ignores it. Accuracy is never "
                    "measured here, only the structure of the deletion and the objectness "
                    "it produces."}
    path = RAW / "xp06e4c_mechanism.json"
    path.write_text(json.dumps(rec, indent=2))
    log(f"wrote {path.name}")


if __name__ == "__main__":
    main()
