#!/usr/bin/env python3
"""XP15 stage B, step 4 — read the board's serial log and say whether the port held.

The sketch already prints a per-frame verdict, which is enough to see that
something is wrong. This turns the log into the numbers the experiment actually
needs: how far the board drifted from the reference, whether any of that drift
changed a decision, and what it cost in time.

**The distinction this exists to make.** A mean absolute error is not a result. The
gate emits one bit, and int8 arithmetic that moves a score by 0.02 is harmless
almost everywhere and fatal next to a threshold of 0.99, where the scores are
bunched. The off-device conversion already found six frames in two hundred that
cross the line under quantization, so the board's log is checked the same way:
error first, then decisions, and the decisions are what count.

Save the serial output to a file and pass it in.

Usage
    python check_board.py board_log.txt
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "board_data"
TAG = "xp15board"

ROW = re.compile(r"^(\d+),(\w+),([\d.]+),([\d.]+),([\d.]+),(\d+)\s*$")


def log(m: str) -> None:
    print(f"[{TAG}] {m}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("logfile", type=Path)
    ap.add_argument("--threshold", type=float, default=0.99)
    args = ap.parse_args()

    import numpy as np

    rows = []
    for line in args.logfile.read_text(errors="ignore").splitlines():
        mt = ROW.match(line.strip())
        if mt:
            rows.append((int(mt.group(1)), mt.group(2), float(mt.group(3)),
                         float(mt.group(4)), float(mt.group(5)), int(mt.group(6))))
    if not rows:
        raise SystemExit("no data rows found -- expected lines like "
                         "'0,none,0.00123,0.00125,0.00002,41234'")

    idx = np.array([r[0] for r in rows])
    cat = np.array([r[1] for r in rows])
    ref = np.array([r[2] for r in rows])
    got = np.array([r[3] for r in rows])
    us = np.array([r[5] for r in rows], float)
    log(f"{len(rows)} frames from {args.logfile.name}")

    # --- did it compute the same thing? ------------------------------------
    err = np.abs(got - ref)
    log(f"  |board - reference|: mean {err.mean():.5f}, worst {err.max():.5f}")
    flips = (ref >= args.threshold) != (got >= args.threshold)
    log(f"  decisions changed at threshold {args.threshold}: {int(flips.sum())} of "
        f"{len(rows)} ({flips.mean():.1%})")
    for i in np.where(flips)[0][:5]:
        log(f"    frame {idx[i]} ({cat[i]}): reference {ref[i]:.4f} -> board {got[i]:.4f}")

    # --- what did it cost? -------------------------------------------------
    ms = us / 1000
    log(f"  latency: mean {ms.mean():.1f} ms, median {np.median(ms):.1f}, "
        f"worst {ms.max():.1f}")
    log(f"  at one frame every 5 s that is a {ms.mean()/5000:.2%} duty cycle")

    # --- and did it agree per category? ------------------------------------
    log("  wake rate by category, board against reference:")
    for c in ("none", "smoke", "fire", "both"):
        m = cat == c
        if not m.any():
            continue
        log(f"    {c:6s} n={int(m.sum()):3d}  reference {(ref[m] >= args.threshold).mean():5.1%}"
            f"  board {(got[m] >= args.threshold).mean():5.1%}")

    ok = err.max() <= 0.01 and not flips.any()
    log("PORT OK -- the board reproduces the off-device scores" if ok else
        "PORT DIFFERS -- see the frames listed above")

    # A check on the check: the log's own 'ref' column should match the subset
    # make_headers.py compiled into the firmware. If it does not, the board was
    # flashed from stale headers and every number above compares two unrelated
    # things. Read from board_subset.npz rather than re-deriving the selection,
    # since a second derivation is a second thing that can disagree.
    subset = DATA / "board_subset.npz"
    if subset.exists():
        stored = np.load(subset, allow_pickle=True)["ref_score"]
        if idx.max() < len(stored):
            drift = float(np.abs(stored[idx] - ref).max())
            if drift > 1e-4:
                log(f"  WARNING: the log's reference column differs from the compiled "
                    f"subset by {drift:.2e} -- the board was probably flashed from "
                    f"stale headers, rerun make_headers.py and reflash")
        else:
            log(f"  WARNING: log has frame {idx.max()} but the compiled subset holds "
                f"{len(stored)} -- log and headers disagree")


if __name__ == "__main__":
    main()
