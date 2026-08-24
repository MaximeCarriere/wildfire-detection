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

    xs, ys, tiny, small, content = [], [], [], [], []
    for s in dataset.load_samples(split):
        im = cv2.imread(str(s.image), cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        xs.append(cv2.resize(im, (res, res), interpolation=cv2.INTER_AREA))
        ys.append(0.0 if s.is_background else 1.0)
        tiny.append(bool(s.has_tiny_plume))
        small.append(bool(s.has_small_plume))
        # 'none' / 'smoke' / 'fire' / 'both' -- the rows of the confusion matrix.
        content.append(s.content)
    return (np.stack(xs), np.array(ys, "float32"),
            np.array(tiny), np.array(small), np.array(content))


def scores(model, x, batch: int = 256):
    """Sigmoid scores for every frame, in batches.

    Batched deliberately: one forward pass over the whole test split allocates
    about 1.3 GB for the first layer's activations alone, which is enough to get
    the process killed on a board with 8 GB shared between CPU and GPU. The first
    version of this script did exactly that and lost a completed training run.
    """
    import numpy as np
    import torch

    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = torch.from_numpy(x[i:i + batch]).unsqueeze(1).float().div(255)
            out.append(torch.sigmoid(model(xb)).squeeze(1).cpu().numpy())
    return np.concatenate(out)


CATEGORIES = ("none", "smoke", "fire", "both")


def evaluate(p, y, tiny, small, content, threshold: float) -> dict:
    """Wake rates by ground-truth category and by plume size.

    The gate is binary, so the useful "confusion matrix" is one row per true
    category and two columns -- woke, stayed asleep. The 'none' row is the false
    alarm rate and every other row is recall for that category.

    Pooled accuracy is not reported anywhere. On a split where most positives are
    large it is dominated by the easy cases, which is exactly the failure this
    experiment exists to detect.
    """
    import numpy as np

    fired = p >= threshold
    pos, neg = y > 0.5, y < 0.5
    big = pos & ~small & ~tiny

    matrix = {}
    for c in CATEGORIES:
        m = content == c
        if not m.any():
            continue
        woke = float(fired[m].mean()) * 100
        matrix[c] = {"n": int(m.sum()), "woke_pct": round(woke, 2),
                     "asleep_pct": round(100 - woke, 2),
                     "mean_score": round(float(p[m].mean()), 4)}

    def r(mask):
        return round(float(fired[mask].mean()), 4) if mask.any() else None

    return {
        "threshold": threshold,
        "confusion_pct": matrix,
        "false_wake_rate": r(neg),
        "recall_all_positive": r(pos),
        "recall_obvious": r(big),
        "recall_small_plume": r(small),
        "recall_tiny_plume": r(tiny),
        "n": {"positive": int(pos.sum()), "background": int(neg.sum()),
              "small_plume": int(small.sum()), "tiny_plume": int(tiny.sum())},
    }


def sweep(p, y, tiny, small) -> list:
    """Recall against false wakes across thresholds: the gate's operating curve.

    A single threshold is one point on a trade-off the deployer gets to choose,
    so the curve is recorded rather than just the point this run happened to pick.
    """
    import numpy as np

    pos, neg = y > 0.5, y < 0.5
    rows = []
    for t in [i / 100 for i in range(1, 100, 2)]:
        f = p >= t
        rows.append({
            "threshold": round(t, 2),
            "false_wake_rate": round(float(f[neg].mean()), 4),
            "recall_all": round(float(f[pos].mean()), 4),
            "recall_small": round(float(f[small].mean()), 4) if small.any() else None,
            "recall_tiny": round(float(f[tiny].mean()), 4) if tiny.any() else None,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=96)
    ap.add_argument("--width", type=float, default=0.25)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--no-onnx", action="store_true")
    ap.add_argument("--eval-only", action="store_true",
                    help="load the saved checkpoint and re-score, skipping training. "
                         "Scoring is seconds and training is minutes, so any change to "
                         "how results are reported should not cost a retrain.")
    args = ap.parse_args()

    import numpy as np
    import torch
    import torch.nn as nn

    dev = ("cuda" if torch.cuda.is_available() else
           "mps" if torch.backends.mps.is_available() else "cpu")
    log(f"device {dev} · {args.res}px · width {args.width}")

    xtr, ytr, _, _, _ = load_split("train", args.res)
    xte, yte, tiny, small, content = load_split("test", args.res)
    log(f"train {len(xtr)} · test {len(xte)} "
        f"({int(yte.sum())} positive, {int(small.sum())} small, {int(tiny.sum())} tiny)")

    model = build_model(args.res, args.width).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    log(f"{n_par/1000:.1f}k parameters -> about {n_par/1024:.0f} KB as int8")

    tag = f"gate_{args.res}px_w{args.width}"
    if args.eval_only:
        ck = WEIGHTS / f"{tag}.pt"
        model.load_state_dict(torch.load(ck, map_location=dev,
                                         weights_only=False)["state_dict"])
        log(f"loaded {ck.name}, skipping training")
        args.epochs = 0

    # Built only when there is training to do: OneCycleLR rejects zero total steps,
    # so constructing it unconditionally makes --eval-only crash before it scores.
    opt = sched = None
    if args.epochs:
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=3e-3, total_steps=args.epochs * (len(xtr) // args.batch + 1))
    # Positives are the minority and a missed fire is the expensive error, so the
    # loss is weighted rather than left to the class balance.
    pw = torch.tensor([(len(ytr) - ytr.sum()) / max(ytr.sum(), 1)], device=dev)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    log(f"positive class weight {pw.item():.2f}")

    t0 = time.perf_counter()
    for ep in range(args.epochs):  # zero iterations under --eval-only
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
    WEIGHTS.mkdir(exist_ok=True)
    # Saved before scoring, not after: the first version of this script evaluated
    # first and was killed doing it, throwing away a finished training run.
    ckpt = WEIGHTS / f"{tag}.pt"
    if not args.eval_only:
        torch.save({"state_dict": model.state_dict(), "res": args.res,
                    "width": args.width}, ckpt)
        log(f"  saved {ckpt.name}")

    p_scores = scores(model, xte)
    m = evaluate(p_scores, yte, tiny, small, content, args.threshold)
    log(f"  false wakes {m['false_wake_rate']:.1%} · recall obvious "
        f"{m['recall_obvious']:.3f} · small {m['recall_small_plume']:.3f} · "
        f"tiny {m['recall_tiny_plume']:.3f}")
    for c, row in m["confusion_pct"].items():
        log(f"    {c:6s} n={row['n']:4d}  woke {row['woke_pct']:5.1f}%  "
            f"asleep {row['asleep_pct']:5.1f}%")

    # The rule is checked against the whole sweep, not against whichever threshold
    # this run happened to default to. The threshold is a free deployment
    # parameter -- a model that satisfies the rule anywhere on its curve satisfies
    # the rule, and judging it at one arbitrary point measures the default rather
    # than the model.
    sw = sweep(p_scores, yte, tiny, small)
    ok = [r for r in sw if r["false_wake_rate"] <= 0.05
          and r["recall_small"] is not None and r["recall_small"] >= 0.70]
    best = max(ok, key=lambda r: r["recall_small"]) if ok else None
    passes = best is not None
    if passes:
        log(f"  decision rule PASS at threshold {best['threshold']}: "
            f"{best['false_wake_rate']:.1%} false wakes, "
            f"{best['recall_small']:.1%} small-plume recall -> port it")
    else:
        tight = min(sw, key=lambda r: abs(r["false_wake_rate"] - 0.05))
        log(f"  decision rule FAIL: best near a 5% budget is threshold "
            f"{tight['threshold']} at {tight['recall_small']:.1%} small recall")

    rec = {"experiment": "xp15_gate_training", "input_res": args.res,
           "width_mult": args.width, "epochs": args.epochs,
           "params": n_par, "train_minutes": round((time.perf_counter() - t0) / 60, 2),
           "device": dev, "metrics": m, "decision_rule_passed": passes,
           "threshold_sweep": sw, "operating_point": best,
           # Per-frame scores, so a confusion matrix at any threshold can be built
           # afterwards without a rescore. 4306 floats is a rounding error next to
           # the checkpoint, and aggregates alone cannot be re-cut later.
           "per_frame": {"score": [round(float(v), 5) for v in p_scores],
                         "label": [int(v) for v in yte],
                         "content": [str(c) for c in content],
                         "tiny": [bool(v) for v in tiny],
                         "small": [bool(v) for v in small]},
           # The confusion at the threshold the gate would actually ship with,
           # which is rarely the one a run happens to default to.
           "metrics_at_operating_point": (
               evaluate(p_scores, yte, tiny, small, content, best["threshold"])
               if best else None),
           "score_histogram": {
               c: np.histogram(p_scores[content == c], bins=20, range=(0, 1))[0].tolist()
               for c in CATEGORIES if (content == c).any()},
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

    if not args.no_onnx:
        onnx_path = WEIGHTS / f"{tag}.onnx"
        # opset 17, not 13. At 13 this graph needs a version conversion that fails
        # on the global-average-pool axes, and the exporter then writes a file with
        # no initializers in it at all -- a 40 KB graph for a 520 KB model that
        # loads without complaint and computes nothing. Checked below rather than
        # trusted, because a silently weightless export is exactly the kind of file
        # that would be flashed to a board and debugged as a hardware problem.
        torch.onnx.export(model, torch.randn(1, 1, args.res, args.res),
                          str(onnx_path), input_names=["frame"],
                          output_names=["score"], opset_version=17)
        kb = onnx_path.stat().st_size / 1024
        expected_kb = n_par * 4 / 1024
        if kb < expected_kb * 0.5:
            raise RuntimeError(f"{onnx_path.name} is {kb:.0f} KB but the model holds "
                               f"{n_par} float32 weights (~{expected_kb:.0f} KB): the "
                               f"export dropped its initializers")
        log(f"  wrote {onnx_path.name} ({kb:.0f} KB, float32)")




if __name__ == "__main__":
    main()
