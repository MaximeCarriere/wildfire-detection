#!/usr/bin/env python3
"""XP7 E10 — composition: prune, then quantize. **Must run on the Jetson.**

Axis: none new — this is the two studies multiplied, and arguably the headline
question of the pair: **do the savings stack?**

Start from XP6's best model (L1 criterion, 25% channel cut, ``round_to=32``, which
measured 0.7377 mAP50 at 641.9 img/s and 38.0 J/1k) and run it through XP7's best
PTQ recipe. Two arms:

* **prune_then_quantize** — the recovered pruned model, quantized. Recovery
  already happened; quantization is applied to a finished network.
* **prune_quantize_then_recover** — quantize *before* the 12-epoch recovery, so
  one training run repairs both damages at once. This is the arm that tests
  whether the two repairs compete for the same capacity or not.

**The failure mode this script exists to avoid.** Pruning changed the network's
activation statistics — narrower layers, different distributions, and in XP6's
case a deliberately rounded channel count. So the calibration cache is
**re-collected on the pruned model**, never reused from the dense one. Reusing it
would set every scale from a distribution that no longer exists, and the resulting
accuracy loss would be blamed on INT8.

**Why the arithmetic is not trusted.** The two techniques save through different
mechanisms: pruning removes channels (less work), quantization makes each MAC
cheaper and halves the bandwidth. Naive multiplication says XP6's 1.36x times
E4's INT8 factor. XP6's whole lesson was that kernels, tiling and launch overhead
decide, not FLOPs — which is exactly why this is measured.

Row format matches E4 so the result drops straight into E9's chart.

Usage
    python e10_compose.py                          # both arms, board numbers
    python e10_compose.py --arms prune_then_quantize
    python e10_compose.py --skip-board             # accuracy only
"""
from __future__ import annotations

import argparse
import sys
import time

import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _calib                                                      # noqa: E402
from _calib import log                                             # noqa: E402

#: XP6's winner, as committed by that study.
XP6_BEST_WEIGHTS = _calib.WEIGHTS / "yolov5s_pruned25_recovered.pt"
XP6_BEST_RAW = _calib.WEIGHTS / "yolov5s_pruned25_raw.pt"
XP6_BEST_BOARD = {"map50": 0.7377, "fps_batched": 641.9, "j_per_1k": 38.0,
                  "recipe": "L1 criterion, 25% channel cut, round_to=32"}


def load_pruned(path: Path):
    """Load a pruned checkpoint. A pruned net no longer matches its config file,
    so this goes through torch.load of the whole object, not a state dict.

    The YOLOv5 repository has to be on ``sys.path`` first: these checkpoints
    pickle live ``models.yolo.DetectionModel`` objects, so unpickling imports
    ``models`` by name and fails with a bare ``ModuleNotFoundError`` otherwise —
    an error that looks like a missing dependency rather than a missing path.
    """
    import sys
    import torch

    if not path.exists():
        raise SystemExit(f"{path} missing — XP6's pruned model has to be present; "
                         f"see experiments/xp06_pruning/recover_and_deploy.py")
    if str(_calib.YOLOV5_REPO) not in sys.path:
        sys.path.insert(0, str(_calib.YOLOV5_REPO))
    obj = torch.load(path, map_location="cpu", weights_only=False)
    model = obj.get("ema") or obj.get("model") or obj if isinstance(obj, dict) else obj
    return model.float().eval().cuda()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*",
                    default=["prune_then_quantize", "prune_quantize_then_recover"])
    ap.add_argument("--calib-images", type=int, default=512)
    ap.add_argument("--val-images", type=int, default=0)
    ap.add_argument("--test-images", type=int, default=0)
    ap.add_argument("--recover-epochs", type=int, default=12)
    ap.add_argument("--skip-board", action="store_true")
    ap.add_argument("--out", default="xp07e10_compose.json")
    args = ap.parse_args()

    import _arms
    from _quant import QuantConfig, Quantizer

    cfg = QuantConfig(weight_bits=8, act_bits=8, granularity="per_channel",
                      method="minmax")
    val = _calib.split_samples("val", limit=args.val_images)

    rows = {}

    if "prune_then_quantize" in args.arms:
        log("e10", "--- prune_then_quantize: XP6's recovered model, quantized ---")
        model = load_pruned(XP6_BEST_WEIGHTS)
        # Score the pruned model *before* quantizing it. Without this the arm
        # reports an accuracy with no reference: the dense FP16 baseline is the
        # wrong comparison, because pruning already spent accuracy of its own, and
        # the question here is what quantization costs *on top of* that.
        pruned_base = _calib.score(model, val, "e10_pruned_fp16")
        log("e10", f"pruned model before quantizing: map50={pruned_base['map50']:.4f} "
                   f"tiny={pruned_base['tiny_plume']['map50']}")
        q = Quantizer(model, cfg)
        # Re-collected on the PRUNED model. This is the point of the experiment.
        _calib.run_calibration(model, q, args.calib_images, tag="e10_pruned")
        q.enable()
        acc = _calib.score(model, val, "e10_prune_then_quantize")
        rows["prune_then_quantize"] = {
            "pruned_fp16": {"map50": pruned_base["map50"],
                            "small_plume": pruned_base["small_plume"]["map50"],
                            "tiny_plume": pruned_base["tiny_plume"]["map50"]},
            "damage": {"map50": acc["map50"],
                       "small_plume": acc["small_plume"]["map50"],
                       "tiny_plume": acc["tiny_plume"]["map50"]},
            "quantization_cost_pct": round(
                (pruned_base["map50"] - acc["map50"]) / pruned_base["map50"] * 100, 3),
            "quantization_cost_tiny_pct": round(
                (pruned_base["tiny_plume"]["map50"] - acc["tiny_plume"]["map50"])
                / pruned_base["tiny_plume"]["map50"] * 100, 3),
            "calibration": "re-collected on the pruned model, never reused from dense",
            "start_from": str(XP6_BEST_WEIGHTS.name),
        }
        q.restore()
        del model
        log("e10", f"prune_then_quantize map50={acc['map50']:.4f}")

    if "prune_quantize_then_recover" in args.arms:
        log("e10", "--- prune_quantize_then_recover: one recovery for both damages ---")
        model = load_pruned(XP6_BEST_RAW if XP6_BEST_RAW.exists() else XP6_BEST_WEIGHTS)
        q = Quantizer(model, cfg)
        _calib.run_calibration(model, q, args.calib_images, tag="e10_raw")
        q.enable()
        pre = _calib.score(model, val, "e10_pqr_pre")
        row = {"damage": {"map50": pre["map50"],
                          "small_plume": pre["small_plume"]["map50"],
                          "tiny_plume": pre["tiny_plume"]["map50"]},
               "start_from": str((XP6_BEST_RAW if XP6_BEST_RAW.exists()
                                  else XP6_BEST_WEIGHTS).name),
               "recovered": None}
        if args.recover_epochs:
            t0 = time.time()
            from lib.finetune import finetune
            finetune(model, repo=_calib.YOLOV5_REPO, epochs=args.recover_epochs,
                     res=_calib.RES, batch=8)
            # rebase(), not restore(): the weights worth keeping are the ones this
            # fine-tune just produced. restore() would put back the copy stashed
            # before training and discard all of it.
            q.rebase()
            _calib.run_calibration(model, q, args.calib_images, tag="e10_recal")
            q.enable()
            post = _calib.score(model, val, "e10_pqr_post")

            # Three controls, because the first run of this arm produced a clean
            # training curve (loss 1.166 -> 0.804 over 12 epochs) and a recovered
            # mAP50 of 0.0001, and nothing in the output distinguished between:
            #   (a) the fine-tune never recovered the pruned network,
            #   (b) it recovered, and re-calibrating afterwards broke it,
            #   (c) it recovered, and the quantization at evaluation broke it.
            # The model was not saved, so none of them could be told apart without
            # spending the three hours again. This is XP6-E7's guardrail rule
            # applied where it should have been from the start: prove the loop
            # returns a working model before believing any number it produces.
            q.restore()
            unquant = _calib.score(model, val, "e10_pqr_post_unquantized")
            log("e10", f"control — same weights, quantization OFF: "
                       f"map50={unquant['map50']:.4f}")
            q.enable()

            ckpt = _calib.WEIGHTS / "xp07_e10_pqr_recovered.pt"
            torch.save({"model": model}, ckpt)
            log("e10", f"saved {ckpt.name} so this is diagnosable without a re-run")

            row["recovered"] = {"epochs": args.recover_epochs, "map50": post["map50"],
                                "small_plume": post["small_plume"]["map50"],
                                "tiny_plume": post["tiny_plume"]["map50"],
                                "train_seconds": round(time.time() - t0, 1),
                                "control_unquantized": {
                                    "map50": unquant["map50"],
                                    "small_plume": unquant["small_plume"]["map50"],
                                    "tiny_plume": unquant["tiny_plume"]["map50"],
                                    "means": "if this is healthy the fine-tune worked "
                                             "and the fault is in the quantization or "
                                             "the re-calibration; if it is ~0 the "
                                             "fine-tune never recovered the network"},
                                "checkpoint": str(ckpt.name)}
            log("e10", f"recovered map50={post['map50']:.4f}")
        rows["prune_quantize_then_recover"] = row
        q.restore()
        del model

    board_note = ("skipped" if args.skip_board else
                  "build the pruned ONNX and run e4_engines.py against it — "
                  "engine building for a pruned graph goes through "
                  "lib.trt_export.export_onnx_from_model, see e10 board section")

    _calib.write_json(args.out, {
        "experiment": "xp07_e10_compose",
        "question": "do XP6's pruning savings and XP7's quantization savings stack?",
        "axis": "composition",
        "xp6_best": XP6_BEST_BOARD,
        "recipe": cfg.label(),
        "split": "val", "n_val_images": len(val),
        "calibration_discipline": "re-collected on the pruned model — pruning changed "
                                  "the activation statistics, so the dense cache is invalid",
        "arms": rows,
        "board": board_note,
        "naive_expectation": "1.36x (XP6) x whatever E4 measured for INT8 — stated so it "
                             "can be contradicted, not because it is believed",
    })


if __name__ == "__main__":
    main()
