#!/usr/bin/env python3
"""XP15 stage B, step 1 — cut two small frame sets and score them off-device.

The board test does not use the camera. Pointing a lens out of a window produces
no fire, so it can only measure false alarms, and it tangles "can this chip run
the model" together with "does the field look like the training set". Those are
different questions and the compute one comes first. So the board is fed frames
from the D-Fire test split instead, and what comes back is checkable against a
known answer.

Two sets come out of here, and they must not be the same frames:

* **test** -- a stratified subset of the *test* split, embedded in the firmware and
  scored on-device. Stratified rather than random because 'fire' is 5% of the split
  and a random 200 would carry about ten of them, which is too few to see anything.
* **calib** -- frames from the *training* split, used only to pick int8 ranges
  during quantization. Calibrating on test data would tune the quantizer on the
  frames it is about to be judged on, which is the standing rule everywhere else in
  this repo and is not relaxed here.

Also writes the float score this repo's checkpoint gives for every test frame, so
the on-device int8 numbers have something to be compared against frame by frame.
Quantization drift then arrives as a measured delta rather than an assumption --
XP6 withdrew published numbers twice over exactly that kind of assumption.

Run on the machine holding the dataset, which is the Jetson.

Usage
    python export_frames.py --per-class 50
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
OUT = HERE / "board_data"
TAG = "xp15export"

CATEGORIES = ("none", "smoke", "fire", "both")


def log(m: str) -> None:
    print(f"[{TAG}] {m}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=96)
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--per-class", type=int, default=50,
                    help="test frames per content category")
    ap.add_argument("--calib", type=int, default=200,
                    help="training frames for the int8 quantizer")
    args = ap.parse_args()

    import cv2
    import numpy as np
    import torch
    from lib import data as dataset
    from train_gate import build_model, scores

    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(42)      # fixed: the subset must not move between runs

    # ---- test subset, stratified by what is actually in the frame ----------
    samples = dataset.load_samples("test")
    by_cat = {c: [s for s in samples if s.content == c] for c in CATEGORIES}
    picked = []
    for c in CATEGORIES:
        pool = by_cat[c]
        take = min(args.per_class, len(pool))
        idx = rng.choice(len(pool), size=take, replace=False)
        picked += [pool[i] for i in sorted(idx)]
        log(f"  {c:6s} {take} of {len(pool)}")

    frames, labels, content, stems = [], [], [], []
    for s in picked:
        im = cv2.imread(str(s.image), cv2.IMREAD_GRAYSCALE)
        frames.append(cv2.resize(im, (args.res, args.res), interpolation=cv2.INTER_AREA))
        labels.append(0 if s.is_background else 1)
        content.append(s.content)
        stems.append(s.stem)
    frames = np.stack(frames).astype("uint8")

    # ---- the reference the board is checked against ------------------------
    model = build_model(args.res, args.width)
    ck = WEIGHTS / f"gate_{args.res}px_w{args.width}.pt"
    model.load_state_dict(torch.load(ck, map_location="cpu",
                                     weights_only=False)["state_dict"])
    ref = scores(model, frames)
    log(f"  scored {len(frames)} frames with {ck.name} (float32, off-device)")

    np.savez_compressed(OUT / "frames_test.npz", frames=frames,
                        labels=np.array(labels), content=np.array(content),
                        stems=np.array(stems), ref_score=ref.astype("float32"))

    # ---- calibration frames, training split only ---------------------------
    train = dataset.load_samples("train")
    idx = rng.choice(len(train), size=min(args.calib, len(train)), replace=False)
    calib = []
    for i in sorted(idx):
        im = cv2.imread(str(train[i].image), cv2.IMREAD_GRAYSCALE)
        calib.append(cv2.resize(im, (args.res, args.res), interpolation=cv2.INTER_AREA))
    np.savez_compressed(OUT / "frames_calib.npz",
                        frames=np.stack(calib).astype("uint8"))

    kb = sum(f.stat().st_size for f in OUT.glob("*.npz")) / 1024
    log(f"wrote {OUT.name}/frames_test.npz and frames_calib.npz ({kb:.0f} KB total)")
    log(f"  {len(frames)} test frames -> {frames.nbytes/1024:.0f} KB raw in firmware")


if __name__ == "__main__":
    main()
