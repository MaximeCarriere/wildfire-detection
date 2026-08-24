#!/usr/bin/env python3
"""XP15 stage A — train the tiny gate, and find out whether it is worth porting.

The gate does not detect anything. It answers one binary question, "is there fire
or smoke in this frame", and its only job is to decide whether to wake a real
detector. That framing is forced by a measurement this repo already has: XP2's
resolution sweep scores a *full* YOLOv5s at **0.0000** on tiny plumes at 160 px
while still reaching 0.458 mAP50 overall. An ESP32 runs at 96 px with a model
orders of magnitude smaller, so boxes are not on the table and distant plumes are
gone before the microcontroller is involved. What may survive is "there is obvious
fire or smoke in view".

**The point of this script is to fail cheaply if it is going to fail.** Nothing
about the port is hard to guess -- the board has 8 MB of PSRAM and a 240 MHz core
with SIMD, and a 300 KB int8 model will run on it. What is not guessable is
whether a 96 px classifier catches anything a fire watch would want, so that is
measured first, on a workstation, before any firmware exists.

**Recall is reported stratified by plume size, never pooled.** A gate that scores
90% overall by catching every wall of flame and no distant smoke is worse than
useless: it fires when the fire is already obvious. ``lib.data`` carries
``has_tiny_plume`` and ``has_small_plume`` per sample, so the breakdown costs
nothing and is the only number that decides whether stage B happens.

**The decision rule is stated here, before the run** (the convention XP5 set):

    Port to the device only if, at a false-wake rate <= 5% on background frames,
    recall on frames containing a *small* plume is >= 0.70.

Tiny plumes are deliberately not in the rule. At 96 px they are almost certainly
unreachable, and writing a bar this cannot clear would only produce a rigged
success or a foregone failure. They are measured and reported regardless, because
that number is the honest limit of the whole idea.

Usage
    python train_gate.py --epochs 30
    python train_gate.py --epochs 30 --res 128     # if 96 misses the bar
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

RAW = REPO / "results" / "raw"
WEIGHTS = REPO / "weights"
TAG = "xp15"

#: Wake the detector above this score. Deliberately low: a false wake costs one
#: Orin inference, a missed fire costs the whole point of the system.
DEFAULT_THRESHOLD = 0.30


def log(m: str) -> None:
    print(f"[{TAG}] {m}", flush=True)


def build_model(res: int, width: float = 0.25):
    """A small depthwise-separable CNN, sized to quantize to a few hundred KB.

    Deliberately not MobileNet from torchvision: the stock model carries a 1000-way
    classifier and an input stem tuned for 224 px, and trimming it to one output at
    96 px leaves something less predictable than writing the six blocks directly.
    Every op here has an int8 kernel in both ESP-DL and TFLite Micro, which is the
    real constraint -- an elegant architecture that lowers to an unsupported op is
    worth nothing on this target.
    """
    import torch.nn as nn

    def c(i, o, s=1):
        return nn.Sequential(nn.Conv2d(i, o, 3, s, 1, bias=False),
                             nn.BatchNorm2d(o), nn.ReLU(inplace=True))

    def dw(i, o, s=1):
        return nn.Sequential(
            nn.Conv2d(i, i, 3, s, 1, groups=i, bias=False),
            nn.BatchNorm2d(i), nn.ReLU(inplace=True),
            nn.Conv2d(i, o, 1, 1, 0, bias=False),
            nn.BatchNorm2d(o), nn.ReLU(inplace=True))

    w = lambda n: max(8, int(n * width))       # noqa: E731
    return nn.Sequential(
        c(1, w(32), 2),                        # grayscale in: the camera is mono-
        dw(w(32), w(64)),                      # capable and colour buys little for
        dw(w(64), w(128), 2),                  # smoke, which is grey by definition
        dw(w(128), w(128)),
        dw(w(128), w(256), 2),
        dw(w(256), w(256)),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(w(256), 1))


def load_split(split: str, res: int):
    """Frames and binary labels, plus the size flags used to stratify recall."""
    import cv2
    import numpy as np
    from lib import data as dataset

    xs, ys, tiny, small = [], [], [], []
    for s in dataset.load_samples(split):
        im = cv2.imread(str(s.image), cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        xs.append(cv2.resize(im, (res, res), interpolation=cv2.INTER_AREA))
        ys.append(0.0 if s.is_background else 1.0)
        tiny.append(bool(s.has_tiny_plume))
        small.append(bool(s.has_small_plume))
    return (np.stack(xs), np.array(ys, "float32"),
            np.array(tiny), np.array(small))


def evaluate(model, x, y, tiny, small, threshold: float) -> dict:
    """Recall by plume size, and the false-wake rate that pays for it.

    Pooled accuracy is not reported anywhere. On a set where most positives are
    large it is dominated by the easy cases, which is exactly the failure this
    experiment is trying to detect.
    """
    import numpy as np
    import torch

    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(torch.from_numpy(x).unsqueeze(1).float().div(255)))
        p = p.squeeze(1).cpu().numpy()
    fired = p >= threshold

    pos, neg = y > 0.5, y < 0.5
    big = pos & ~small & ~tiny
    out = {
        "threshold": threshold,
        "false_wake_rate": float(fired[neg].mean()) if neg.any() else None,
        "recall_all_positive": float(fired[pos].mean()) if pos.any() else None,
        "recall_obvious": float(fired[big].mean()) if big.any() else None,
        "recall_small_plume": float(fired[small].mean()) if small.any() else None,
        "recall_tiny_plume": float(fired[tiny].mean()) if tiny.any() else None,
        "n": {"positive": int(pos.sum()), "background": int(neg.sum()),
              "small_plume": int(small.sum()), "tiny_plume": int(tiny.sum())},
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=96)
    ap.add_argument("--width", type=float, default=0.25)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--no-onnx", action="store_true")
    args = ap.parse_args()

    import numpy as np
    import torch
    import torch.nn as nn

    dev = ("cuda" if torch.cuda.is_available() else
           "mps" if torch.backends.mps.is_available() else "cpu")
    log(f"device {dev} · {args.res}px · width {args.width}")

    xtr, ytr, _, _ = load_split("train", args.res)
    xte, yte, tiny, small = load_split("test", args.res)
    log(f"train {len(xtr)} · test {len(xte)} "
        f"({int(yte.sum())} positive, {int(small.sum())} small, {int(tiny.sum())} tiny)")

    model = build_model(args.res, args.width).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    log(f"{n_par/1000:.1f}k parameters -> about {n_par/1024:.0f} KB as int8")

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=3e-3, total_steps=args.epochs * (len(xtr) // args.batch + 1))
    # Positives are the minority and a missed fire is the expensive error, so the
    # loss is weighted rather than left to the class balance.
    pw = torch.tensor([(len(ytr) - ytr.sum()) / max(ytr.sum(), 1)], device=dev)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    log(f"positive class weight {pw.item():.2f}")

    t0 = time.perf_counter()
    for ep in range(args.epochs):
        model.train()
        idx = np.random.permutation(len(xtr))
        tot = 0.0
        for i in range(0, len(idx), args.batch):
            b = idx[i:i + args.batch]
            xb = torch.from_numpy(xtr[b]).unsqueeze(1).float().div(255).to(dev)
            # Horizontal flip only: a plume is not vertically symmetric and the
            # camera will be level, so the usual full augmentation stack would be
            # teaching invariances this deployment never encounters.
            if np.random.rand() < 0.5:
                xb = torch.flip(xb, [3])
            yb = torch.from_numpy(ytr[b]).unsqueeze(1).to(dev)
            loss = lossf(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(b)
        if (ep + 1) % 5 == 0 or ep == args.epochs - 1:
            log(f"  epoch {ep+1}/{args.epochs} loss {tot/len(idx):.4f}")

    model.cpu()
    m = evaluate(model, xte, yte, tiny, small, args.threshold)
    log(f"  false wakes {m['false_wake_rate']:.1%} · recall obvious "
        f"{m['recall_obvious']:.3f} · small {m['recall_small_plume']:.3f} · "
        f"tiny {m['recall_tiny_plume']:.3f}")

    passes = (m["false_wake_rate"] is not None and m["false_wake_rate"] <= 0.05
              and m["recall_small_plume"] is not None
              and m["recall_small_plume"] >= 0.70)
    log(f"  decision rule (<=5% false wakes AND >=0.70 small-plume recall): "
        f"{'PASS -> port it' if passes else 'FAIL -> do not port yet'}")

    tag = f"gate_{args.res}px_w{args.width}"
    if not args.no_onnx:
        WEIGHTS.mkdir(exist_ok=True)
        onnx_path = WEIGHTS / f"{tag}.onnx"
        torch.onnx.export(model, torch.randn(1, 1, args.res, args.res),
                          str(onnx_path), input_names=["frame"],
                          output_names=["score"], opset_version=13)
        log(f"  wrote {onnx_path.name} ({onnx_path.stat().st_size/1024:.0f} KB, float32)")

    rec = {"experiment": "xp15_gate_training", "input_res": args.res,
           "width_mult": args.width, "epochs": args.epochs,
           "params": n_par, "train_minutes": round((time.perf_counter() - t0) / 60, 2),
           "device": dev, "metrics": m, "decision_rule_passed": passes,
           "notes": ("XP15 stage A: a binary fire/smoke gate for the XIAO ESP32-S3, trained "
                     "off-device. Recall is stratified by plume size and never pooled, because "
                     "a gate that only sees obvious flame fires when the fire is already "
                     "obvious. The decision rule was fixed before the run: port only at <=5% "
                     "false wakes with >=0.70 recall on small plumes. Tiny plumes are measured "
                     "but excluded from the rule, since 96 px almost certainly cannot reach "
                     "them and a bar nobody can clear proves nothing. No model_id: this is not "
                     "a detector and does not belong in the detector tables.")}
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"xp15_{tag}.json"
    path.write_text(json.dumps(rec, indent=2) + "\n")
    log(f"wrote {path.name}")


if __name__ == "__main__":
    main()
