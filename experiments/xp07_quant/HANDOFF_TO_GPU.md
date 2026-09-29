# Handoff: the XP7 arms that still need training time

**Read this with [`README.md`](README.md)**, which records what has been measured. This file
records what has not, why, and exactly what to run.

Everything in XP7 that does not train has been run on the Jetson Orin Nano Super. What is left is
the **training**, and the cost is now measured rather than estimated —
`experiments/xp07_quant/e7_cost.py` times the real QAT loop (real dataloader, real augmentation,
fake-quant hooks live) and writes `results/raw/xp07e7_cost.json`.

**Measured on this board: 17.16 img/s forward+backward at 512 px, batch 8 — 15.1 minutes per
epoch, 3.0 hours per 12-epoch arm, 9.0 hours for E7 as specified.**

That is roughly **five times cheaper than the estimate this page carried before measuring**, which
assumed ~3 img/s. **E7 is reachable on the Jetson**: an overnight run covers the round-trip control
and both trained arms. A desktop GPU is a convenience here, not a prerequisite — which is the
opposite of what the plan assumed, and the reason the probe exists.

## What is blocked, and what it costs

Every "recovered" column, plus E7's trained arms and E10's second arm, costs **12 epochs over the
15,500-image training split** — the same budget every XP6 recovery got, which is what makes the
comparison fair and is not negotiable without breaking it.

| arm | what it needs | why it is blocked |
|---|---|---|
| **E7** — PTQ vs QAT | round-trip control + 2 trained arms = 3 x 12 epochs | **9.0 h** — an overnight run |
| **E10 arm 2** — quantize *then* recover | 1 x 12 epochs | **3.0 h** |
| **E3 / E5 / E6 recovered columns** | 1 x 12 epochs per arm | **3.0 h** each |

The damage numbers for E3, E5 and E6 are already measured and on the page. **`recovered: null` in
those JSONs means the arm was not run, never that it was zero** — the flag exists so the number is
produced deliberately.

## What to run

The scripts are unchanged; they take the epoch budget as a flag. On a desktop GPU:

```bash
# The guardrail first, alone. If it fails, nothing below is believable.
python experiments/xp07_quant/e7_qat.py --prove-loop --skip-arms --post-epochs 12

# Then the arms.
python experiments/xp07_quant/e7_qat.py --prove-loop --post-epochs 12
python experiments/xp07_quant/e3_granularity.py --recover-epochs 12
python experiments/xp07_quant/e5_targets.py     --recover-epochs 12
python experiments/xp07_quant/e6_mixed.py --from-e1 --recover-epochs 12
python experiments/xp07_quant/e10_compose.py --recover-epochs 12   # arm 2 only needs this
```

## Three things that must not change

1. **The frozen calibration set.** `data/splits/xp07_calib.txt`, checksum `61bb308c46715e06`, 512
   images, nested subsets. `_calib.calib_paths` re-verifies the hash on every call and raises if it
   moved. Every XP7 number is scored against this file.

2. **The guardrail comes first.** `--prove-loop` fine-tunes the *unquantized* model and checks it
   returns to its starting accuracy. XP6's bug list is explicit about this: a recovery loop that
   quietly degrades the baseline makes QAT look bad, or from a degraded baseline look good, for
   reasons that have nothing to do with quantization. The script **aborts** if the round trip loses
   more than the tolerance. Do not raise the tolerance to make it pass.

3. **`rebase()`, never `restore()`, after training.** This was a real bug and it was silent:
   `restore()` writes back the weights stashed at `enable()` — from *before* the fine-tune — so
   every epoch is discarded and the "recovered" number comes out as the damage number with fresh
   scales. Nothing in the output looks wrong. `experiments/xp07_quant/test_quant.py` pins it;
   **run the tests before trusting any recovered number** (CPU, seconds, no dataset).

## Speed numbers stay on the Jetson

An engine is built for one GPU and will not load on another. Anything measured on a desktop card
carries **accuracy only** and must be labelled that way — the same rule XP6 used. The board numbers
on the XP7 page all came from the Orin, and E9's chart uses engine rows only for exactly this
reason.

## What a desktop GPU would also make cheap

Not blocked, just slow here, and worth doing if the machine is available:

- **E8's error-compensated arms.** GPTQ needs one calibration pass *per layer* — 60 passes per
  bit-width. The RTN and codebook arms have run; `--skip-gptq` is what the board version used.
- **E4-QDQ (path A).** `onnxruntime.quantization` is CPU-bound and slow on aarch64. Path A is the
  only route by which E2's percentile-99.99 choice and E3's per-channel choice reach an engine at
  all — TensorRT's own calibrator offers only entropy and min-max, and **E2 measured that both of
  those are the wrong answer**. Until path A runs, every INT8 engine on the page is a *lower bound*
  on what INT8 can do on this detector.
