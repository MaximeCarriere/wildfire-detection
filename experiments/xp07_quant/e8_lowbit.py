#!/usr/bin/env python3
"""XP7 E8 — below 8 bits, and storage-only compression. **Optional.**

Axis: **bit-width**. No board claim is made anywhere in this experiment: the Orin
has no INT4 convolution path, so every number here is an accuracy-or-size result
and saying otherwise would be the XP6-E4 mistake (a compression with no silicon to
execute it).

Three questions:

1. **INT4 weight-only, round-to-nearest vs error-compensated.** Group-wise scales
   at ``g = 128``. RTN rounds each weight to the nearest grid point independently;
   GPTQ-style rounding quantizes a layer's columns in order and pushes each
   column's error into the columns not yet done, weighted by the inverse Hessian
   of the layer's reconstruction loss. This is the middle rung of the effort
   ladder that E7 skips, placed at the bit-width where the literature says it
   earns its cost.
2. **Is error compensation a no-op at 8 bits?** The same two arms are also run at
   W8, because "presumed negligible" is cheap to check once the machinery exists,
   and a measured no-op is a better sentence than an assumption.
3. **K-means codebook (Deep Compression) + Huffman, as file size.** 4-bit indices
   into a 16-entry FP16 table per layer, then a Huffman code over the index
   stream. Deployed by decoding back to the E5 arm, so this is explicitly a result
   about the artifact at rest, not about the inner loop.

Activations stay in FP16 for every arm here. Quantizing activations to 4 bits on
a detector is a different and much harder experiment, and mixing it in would make
the bit-width result unattributable.

Usage
    python e8_lowbit.py                      # all arms
    python e8_lowbit.py --bits 4 --val-images 300
    python e8_lowbit.py --skip-gptq          # RTN and codebook only (much faster)
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
from _lowbit import (HessianCollector, gptq_quantize, huffman_bits,  # noqa: E402
                     kmeans_codebook)
from _quant import (QuantConfig, conv_layers, model_size_report,   # noqa: E402
                    quantize_weight)


def size_for(bits: int, granularity: str, group_size: int) -> dict:
    """What an arm at this bit-width and granularity actually weighs.

    E8 is the experiment whose question *is* size, so every arm carries the
    itemised count rather than a headline ratio — at 4 bits the group-wise scale
    tables stop being a rounding error and start being a real share of the file.
    """
    model = _calib.load_base(device="cpu")
    cfg = QuantConfig(weight_bits=bits, quantize_acts=False,
                      granularity=granularity, group_size=group_size)
    r = model_size_report(model, cfg)
    del model
    return r


def score_weight_map(fn, samples, tag: str) -> dict:
    """Apply ``fn(name, conv) -> new_weight`` to every conv, score, restore."""
    model = _calib.load_base()
    orig = {}
    t0 = time.time()
    for name, conv in conv_layers(model):
        orig[name] = conv.weight.data.clone()
        new = fn(name, conv)
        if new is not None:
            conv.weight.data = new
    t_q = time.time() - t0
    acc = _calib.score(model, samples, tag)
    for name, conv in conv_layers(model):
        conv.weight.data = orig[name]
    del model
    return {"map50": acc["map50"], "map5095": acc["map5095"],
            "small_plume": acc["small_plume"]["map50"],
            "tiny_plume": acc["tiny_plume"]["map50"],
            "quantize_seconds": round(t_q, 1)}


def gptq_all_layers(samples, *, bits: int, group_size: int, n_calib: int,
                    tag: str) -> dict:
    """GPTQ one layer at a time: collect that layer's Hessian, quantize, keep it.

    Layers are done in forward order and each layer's Hessian is measured on the
    *already quantized* prefix of the network, which is the sequential variant
    GPTQ actually specifies — the later layers are then compensating for the error
    the earlier ones introduced, not for a float network that no longer exists.
    """
    from lib.trt_export import _letterbox_batch
    model = _calib.load_base()
    paths = _calib.calib_paths(n_calib)
    names = [n for n, _ in conv_layers(model)]
    mods = dict(conv_layers(model))
    fallbacks = []

    for i, name in enumerate(names, 1):
        conv = mods[name]
        col = HessianCollector(conv).attach()
        with torch.no_grad():
            for j in range(0, len(paths), 8):
                arr = _letterbox_batch(paths[j:j + 8], _calib.RES, _calib.YOLOV5_REPO)
                x = torch.from_numpy(arr).to(conv.weight.device)
                model(x.half() if conv.weight.dtype == torch.float16 else x)
        col.detach()

        w = conv.weight.data
        flat = w.reshape(w.shape[0], -1)
        q, info = gptq_quantize(flat, col.h, bits=bits, group_size=group_size)
        conv.weight.data = q.reshape(w.shape).to(w.dtype)
        if info.get("fallback"):
            fallbacks.append({"layer": name, "reason": info.get("reason")})
        del col
        torch.cuda.empty_cache()
        if i % 10 == 0:
            log("e8", f"gptq {i}/{len(names)} layers")

    acc = _calib.score(model, samples, tag)
    del model
    return {"map50": acc["map50"], "map5095": acc["map5095"],
            "small_plume": acc["small_plume"]["map50"],
            "tiny_plume": acc["tiny_plume"]["map50"],
            "hessian_fallback_layers": fallbacks}


def codebook_sizes(bits: int = 4) -> dict:
    """Accuracy of the decoded model, and what the encoded file would weigh."""
    model = _calib.load_base()
    total_w = 0
    raw_bits = 0
    huff_bits = 0
    table_bytes = 0
    for name, conv in conv_layers(model):
        centroids, idx = kmeans_codebook(conv.weight.data, bits=bits)
        conv.weight.data = centroids[idx].reshape(conv.weight.shape).to(conv.weight.dtype)
        n = idx.numel()
        total_w += n
        raw_bits += n * bits
        huff_bits += n * huffman_bits(idx)
        table_bytes += centroids.numel() * 2           # FP16 codebook
    return {"model": model, "weights": total_w,
            "fp16_mb": round(total_w * 2 / 1e6, 3),
            "indices_mb": round(raw_bits / 8 / 1e6, 3),
            "huffman_mb": round((huff_bits / 8 + table_bytes) / 1e6, 3),
            "codebook_tables_kb": round(table_bytes / 1e3, 2),
            "mean_huffman_bits_per_weight": round(huff_bits / total_w, 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-images", type=int, default=0)
    ap.add_argument("--calib-images", type=int, default=128,
                    help="GPTQ Hessians; 128 is the usual sufficient size")
    ap.add_argument("--bits", nargs="*", type=int, default=[8, 4])
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--skip-gptq", action="store_true")
    ap.add_argument("--skip-codebook", action="store_true")
    ap.add_argument("--out", default="xp07e8_lowbit.json")
    args = ap.parse_args()

    samples = _calib.split_samples("val", limit=args.val_images)
    baseline = _calib.score(_calib.load_base(), samples, "fp16_baseline")
    log("e8", f"FP16 baseline map50={baseline['map50']:.4f} on {len(samples)} val images")

    rows = {}
    for bits in args.bits:
        key = f"w{bits}"
        sizes = {
            "per_group": size_for(bits, "per_group", args.group_size),
            "per_channel": size_for(bits, "per_channel", 0),
        }
        log("e8", f"{key} size: per-group {sizes['per_group']['total_mb']:.3f} MB "
                  f"({sizes['per_group']['weight_scale_overhead_kb']:.0f} KB of scales) | "
                  f"per-channel {sizes['per_channel']['total_mb']:.3f} MB "
                  f"({sizes['per_channel']['weight_scale_overhead_kb']:.0f} KB)")
        rows[f"{key}_rtn_pergroup"] = score_weight_map(
            lambda n, c, b=bits: quantize_weight(
                c.weight.data, bits=b, granularity="per_group",
                group_size=args.group_size),
            samples, f"e8_{key}_rtn")
        rows[f"{key}_rtn_pergroup"]["size"] = sizes["per_group"]
        log("e8", f"{key} RTN  g={args.group_size} "
                  f"map50={rows[f'{key}_rtn_pergroup']['map50']:.4f}")

        rows[f"{key}_rtn_perchannel"] = score_weight_map(
            lambda n, c, b=bits: quantize_weight(
                c.weight.data, bits=b, granularity="per_channel"),
            samples, f"e8_{key}_rtn_pc")
        rows[f"{key}_rtn_perchannel"]["size"] = sizes["per_channel"]
        log("e8", f"{key} RTN  per-channel "
                  f"map50={rows[f'{key}_rtn_perchannel']['map50']:.4f}")

        if not args.skip_gptq:
            rows[f"{key}_gptq_pergroup"] = gptq_all_layers(
                samples, bits=bits, group_size=args.group_size,
                n_calib=args.calib_images, tag=f"e8_{key}_gptq")
            rows[f"{key}_gptq_pergroup"]["size"] = sizes["per_group"]
            log("e8", f"{key} GPTQ g={args.group_size} "
                      f"map50={rows[f'{key}_gptq_pergroup']['map50']:.4f}")

    codebook = None
    if not args.skip_codebook:
        cb = codebook_sizes(bits=4)
        model = cb.pop("model")
        acc = _calib.score(model, samples, "e8_codebook4")
        del model
        codebook = {**cb, "map50": acc["map50"],
                    "small_plume": acc["small_plume"]["map50"],
                    "tiny_plume": acc["tiny_plume"]["map50"]}
        log("e8", f"codebook4 map50={codebook['map50']:.4f} "
                  f"fp16={codebook['fp16_mb']}MB -> huffman={codebook['huffman_mb']}MB")

    _calib.write_json(args.out, {
        "experiment": "xp07_e8_lowbit",
        "question": "INT4 / codebook / storage-only compression — what does it buy?",
        "axis": "bit-width",
        "board_claim": "none — the Orin has no INT4 convolution path; these are "
                       "accuracy and size results only",
        "split": "val", "n_val_images": len(samples),
        "subsampled": bool(args.val_images),
        "activations": "FP16 in every arm",
        "group_size": args.group_size,
        "gptq_calibration_images": args.calib_images,
        "baseline_fp16": baseline,
        "arms": rows,
        "codebook": codebook,
    })


if __name__ == "__main__":
    main()
