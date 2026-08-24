#!/usr/bin/env python3
"""XP15 — score the *quantized* gate on the whole test split, not a sample of it.

Everything published about this gate's accuracy comes from the float model:
`train_gate.py` scores 4,306 frames in float32 and the confusion matrix is built
from those. The thing that ships is the int8 TFLite model, and so far it has only
been compared against float on the 200 frames exported for the board.

200 frames is enough to catch a broken port. It is not enough to state an accuracy,
and the difference matters more here than usual: `to_tflite.py` already found that
quantization moves scores by a mean of 0.018 while flipping 3% of decisions at a
threshold of 0.99, because that threshold sits where the scores bunch. A 200-frame
estimate of a 3% effect is four or five frames.

So this runs the int8 interpreter over the full split and writes the same fields
`train_gate.py` writes, letting the figures be rebuilt from the model that will
actually run rather than from its float ancestor.

**Off-device on purpose.** The board reproduces these numbers to 0.004 with no
decisions changed, which is what stage B established; re-deriving 4,306 of them
over a serial cable at 340 ms each would take 25 minutes to confirm something
already measured.

Usage
    python eval_int8_full.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

RAW = REPO / "results" / "raw"
WEIGHTS = REPO / "weights"
TAG = "xp15int8"

CATEGORIES = ("none", "smoke", "fire", "both")


def log(m: str) -> None:
    print(f"[{TAG}] {m}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=96)
    ap.add_argument("--width", type=float, default=1.0)
    args = ap.parse_args()

    import numpy as np
    import tensorflow as tf
    from train_gate import CATEGORIES as _c, evaluate, load_split, sweep  # noqa: F401

    tag = f"gate_{args.res}px_w{args.width}"
    blob = (WEIGHTS / f"{tag}_int8.tflite").read_bytes()

    # Prefer the exported npz: it lets this run on a machine that has TensorFlow
    # but not the 3 GB of images, which is the usual split of tools here.
    npz = HERE / "board_data" / "frames_all_test.npz"
    if npz.exists():
        d = np.load(npz, allow_pickle=True)
        x, y = d["frames"], d["labels"]
        tiny, small, content = d["tiny"], d["small"], d["content"]
        log(f"loaded {npz.name}")
    else:
        x, y, tiny, small, content = load_split("test", args.res)
    log(f"{len(x)} test frames, int8 model {len(blob)/1024:.0f} KB")

    it = tf.lite.Interpreter(model_content=blob)
    it.allocate_tensors()
    inp, outp = it.get_input_details()[0], it.get_output_details()[0]
    is_, iz = inp["quantization"]
    os_, oz = outp["quantization"]

    # One frame at a time, exactly as the board does it: the interpreter is built
    # for batch 1 and feeding it otherwise would measure a different graph.
    p = np.empty(len(x), "float32")
    for i, f in enumerate(x):
        v = np.clip(np.round((f.astype("float32") / 255.0) / is_ + iz), -128, 127)
        it.set_tensor(inp["index"], v.astype(np.int8)[None, ..., None])
        it.invoke()
        logit = (it.get_tensor(outp["index"]).astype("float32") - oz) * os_
        p[i] = 1 / (1 + np.exp(-logit.squeeze()))
        if (i + 1) % 1000 == 0:
            log(f"  {i+1}/{len(x)}")

    sw = sweep(p, y, tiny, small)
    ok = [r for r in sw if r["false_wake_rate"] <= 0.05
          and r["recall_small"] is not None and r["recall_small"] >= 0.70]
    best = max(ok, key=lambda r: r["recall_small"]) if ok else None

    rec = {
        "experiment": "xp15_gate_int8_full", "input_res": args.res,
        "width_mult": args.width, "quantization": "int8 post-training",
        "params": 133889, "model_bytes": len(blob),
        "metrics": evaluate(p, y, tiny, small, content, 0.30),
        "metrics_at_operating_point": (
            evaluate(p, y, tiny, small, content, best["threshold"]) if best else None),
        "threshold_sweep": sw, "operating_point": best,
        "decision_rule_passed": best is not None,
        "score_histogram": {
            c: np.histogram(p[content == c], bins=20, range=(0, 1))[0].tolist()
            for c in CATEGORIES if (content == c).any()},
        "per_frame": {"score": [round(float(v), 5) for v in p],
                      "label": [int(v) for v in y],
                      "content": [str(c) for c in content],
                      "tiny": [bool(v) for v in tiny],
                      "small": [bool(v) for v in small]},
        "notes": ("XP15: the int8 TFLite model scored on the full 4,306-frame test split, "
                  "which is what the board runs. The published float numbers come from the "
                  "same frames through the float32 checkpoint; quantization moves scores by "
                  "a mean of 0.018 and flips decisions near the threshold, so an accuracy "
                  "claim about the deployed gate has to come from here. Scored off-device: "
                  "the board reproduces these to 0.004 with no decisions changed, which stage "
                  "B measured, and re-deriving them over serial at 340 ms each would take 25 "
                  "minutes to confirm a settled result."),
    }
    path = RAW / f"xp15_{tag}_int8_full.json"
    path.write_text(json.dumps(rec, indent=2) + "\n")

    m = rec["metrics_at_operating_point"] or rec["metrics"]
    log(f"  at threshold {m['threshold']}: {m['false_wake_rate']:.1%} false wakes, "
        f"small-plume recall {m['recall_small_plume']:.1%}")
    for c, row in m["confusion_pct"].items():
        log(f"    {c:6s} n={row['n']:4d}  woke {row['woke_pct']:5.1f}%")
    log(f"wrote {path.name}")


if __name__ == "__main__":
    main()
