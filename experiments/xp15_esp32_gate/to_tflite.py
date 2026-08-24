#!/usr/bin/env python3
"""XP15 stage B, step 2 — port the trained gate to int8 TFLite, and prove it still works.

**Why TFLite Micro rather than ESP-DL**, having first planned the opposite: the
deciding factor is what can be checked before anything is flashed. This path lets
the quantized model be run and compared against the original on a laptop, so a
port that has gone wrong is caught here rather than debugged through a serial
cable. ESP-DL takes ONNX directly and avoids a framework hop, which is the tidier
story, but nothing about it can be verified off-device.

**Why the weights are copied by hand rather than converted.** onnx2tf and friends
work often enough and fail opaquely, and this network is eight blocks of
conv/BN/ReLU. Rebuilding it in Keras and moving the tensors across is a page of
code whose failure mode is a numerical mismatch that gets asserted on, not a
mystery graph. The layouts differ and that is the whole difficulty:

    ordinary conv   torch (out, in, h, w)  ->  keras (h, w, in, out)
    depthwise conv  torch (ch, 1, h, w)    ->  keras (h, w, ch, 1)

**Three checks, each of which has to pass before the next matters:**

1. Keras float against torch float, on real frames. Catches a botched weight copy.
2. TFLite int8 against torch float, on the same frames. This is quantization
   damage, measured rather than assumed.
3. The decision the gate actually makes -- does int8 flip any frame across the
   threshold. A mean-absolute-error of 0.01 is meaningless if it moves frames from
   one side of 0.99 to the other, which is the only thing the deployed gate does.

Calibration frames come from the *training* split. Quantizing against test frames
would tune the ranges on the data the port is about to be judged on.

Run on any machine with tensorflow; the frames arrive as npz from
``export_frames.py`` so the dataset itself is not needed here.

Usage
    python to_tflite.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

WEIGHTS = REPO / "weights"
DATA = HERE / "board_data"
TAG = "xp15tflite"


def log(m: str) -> None:
    print(f"[{TAG}] {m}", flush=True)


def keras_gate(res: int, width: float):
    """The same graph as ``train_gate.build_model``, in Keras.

    Padding is spelled out as explicit ZeroPadding rather than 'same' because
    Keras resolves 'same' asymmetrically on even inputs while torch pads evenly,
    and a one-pixel disagreement on the stride-2 layers is enough to move the
    output. Being explicit costs three lines and removes the whole class of bug.
    """
    from tensorflow import keras
    from tensorflow.keras import layers as L

    w = lambda n: max(8, int(n * width))       # noqa: E731

    def conv_bn(x, o, k, s):
        x = L.ZeroPadding2D(k // 2)(x)
        x = L.Conv2D(o, k, strides=s, padding="valid", use_bias=False)(x)
        x = L.BatchNormalization(epsilon=1e-5)(x)
        return L.ReLU()(x)

    def dw(x, i, o, s=1):
        x = L.ZeroPadding2D(1)(x)
        x = L.DepthwiseConv2D(3, strides=s, padding="valid", use_bias=False)(x)
        x = L.BatchNormalization(epsilon=1e-5)(x)
        x = L.ReLU()(x)
        x = L.Conv2D(o, 1, use_bias=False)(x)
        x = L.BatchNormalization(epsilon=1e-5)(x)
        return L.ReLU()(x)

    inp = keras.Input((res, res, 1))
    x = conv_bn(inp, w(32), 3, 2)
    x = dw(x, w(32), w(64))
    x = dw(x, w(64), w(128), 2)
    x = dw(x, w(128), w(128))
    x = dw(x, w(128), w(256), 2)
    x = dw(x, w(256), w(256))
    x = L.GlobalAveragePooling2D()(x)
    out = L.Dense(1)(x)
    return keras.Model(inp, out)


def copy_weights(torch_model, kmodel):
    """Move every tensor across, in graph order, asserting the shapes agree."""
    import numpy as np
    import torch.nn as nn
    from tensorflow.keras import layers as L

    tw = [m for m in torch_model.modules()
          if isinstance(m, (nn.Conv2d, nn.BatchNorm2d, nn.Linear))]
    kw = [l for l in kmodel.layers
          if isinstance(l, (L.Conv2D, L.DepthwiseConv2D, L.BatchNormalization, L.Dense))]
    assert len(tw) == len(kw), f"{len(tw)} torch layers against {len(kw)} keras"

    for t, k in zip(tw, kw):
        if isinstance(t, nn.Conv2d):
            W = t.weight.detach().numpy()
            if t.groups > 1:                      # depthwise: (ch,1,h,w) -> (h,w,ch,1)
                assert isinstance(k, L.DepthwiseConv2D), f"{type(k)} for a depthwise conv"
                k.set_weights([W.transpose(2, 3, 0, 1)])
            else:                                 # (out,in,h,w) -> (h,w,in,out)
                assert isinstance(k, L.Conv2D), f"{type(k)} for a dense conv"
                k.set_weights([W.transpose(2, 3, 1, 0)])
        elif isinstance(t, nn.BatchNorm2d):
            k.set_weights([t.weight.detach().numpy(), t.bias.detach().numpy(),
                           t.running_mean.numpy(), t.running_var.numpy()])
        else:
            k.set_weights([t.weight.detach().numpy().T, t.bias.detach().numpy()])
    return kmodel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=96)
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--threshold", type=float, default=0.99)
    args = ap.parse_args()

    import numpy as np
    import tensorflow as tf
    import torch
    from train_gate import build_model

    test = np.load(DATA / "frames_test.npz", allow_pickle=True)
    calib = np.load(DATA / "frames_calib.npz")["frames"]
    frames, ref = test["frames"], test["ref_score"]
    log(f"{len(frames)} test frames, {len(calib)} calibration frames")

    tag = f"gate_{args.res}px_w{args.width}"
    tm = build_model(args.res, args.width)
    tm.load_state_dict(torch.load(WEIGHTS / f"{tag}.pt", map_location="cpu",
                                  weights_only=False)["state_dict"])
    tm.eval()

    km = copy_weights(tm, keras_gate(args.res, args.width))

    # --- check 1: did the weights land where they belong? ------------------
    x = frames.astype("float32")[..., None] / 255.0
    kf = tf.sigmoid(km.predict(x, verbose=0)).numpy().squeeze(1)
    d1 = float(np.abs(kf - ref).max())
    log(f"  keras float vs torch float: max |diff| {d1:.2e}")
    if d1 > 1e-3:
        raise SystemExit(f"weight copy is wrong (max diff {d1:.3e}) -- stopping before "
                         f"quantization, which would only hide it")

    # --- quantize ----------------------------------------------------------
    def representative():
        for f in calib:
            yield [f.astype("float32")[None, ..., None] / 255.0]

    conv = tf.lite.TFLiteConverter.from_keras_model(km)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = representative
    # int8 in and out, not just int8 weights: TFLite Micro on this chip has no
    # float kernels worth using, and a model with float endpoints quietly drags a
    # dequantize op into the graph.
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    blob = conv.convert()

    out = WEIGHTS / f"{tag}_int8.tflite"
    out.write_bytes(blob)
    log(f"  wrote {out.name} ({len(blob)/1024:.0f} KB int8, was "
        f"{sum(p.numel() for p in tm.parameters())*4/1024:.0f} KB float32)")

    # --- check 2: what did quantization cost? ------------------------------
    it = tf.lite.Interpreter(model_content=blob)
    it.allocate_tensors()
    inp, outp = it.get_input_details()[0], it.get_output_details()[0]
    is_, iz = inp["quantization"]
    os_, oz = outp["quantization"]

    q = []
    for f in frames:
        v = np.clip(np.round((f.astype("float32") / 255.0) / is_ + iz), -128, 127)
        it.set_tensor(inp["index"], v.astype(np.int8)[None, ..., None])
        it.invoke()
        logit = (it.get_tensor(outp["index"]).astype("float32") - oz) * os_
        q.append(1 / (1 + np.exp(-logit.squeeze())))
    q = np.array(q)

    d2 = float(np.abs(q - ref).max())
    log(f"  tflite int8 vs torch float: max |diff| {d2:.4f}, "
        f"mean {float(np.abs(q - ref).mean()):.4f}")

    # --- check 3: does it change what the gate decides? --------------------
    # The only thing that reaches the outside world is which side of the
    # threshold a frame lands on, so that is what has to be compared.
    flips = int(((ref >= args.threshold) != (q >= args.threshold)).sum())
    log(f"  decisions changed at threshold {args.threshold}: {flips} of {len(frames)} "
        f"({flips/len(frames):.1%})")
    if flips:
        which = np.where((ref >= args.threshold) != (q >= args.threshold))[0][:5]
        for i in which:
            log(f"    {test['stems'][i]}: float {ref[i]:.4f} -> int8 {q[i]:.4f}")

    np.savez_compressed(DATA / "int8_scores.npz", score=q.astype("float32"),
                        input_scale=is_, input_zero=iz,
                        output_scale=os_, output_zero=oz)
    log(f"wrote board_data/int8_scores.npz -- the reference the board must reproduce")


if __name__ == "__main__":
    main()
