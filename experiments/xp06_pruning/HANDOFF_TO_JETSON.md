# Handing the XP6 extension back to the Jetson

Everything on this page was screened on an RTX 3090. **None of it has a speed number, and two
of the experiments have no verdict at all until the board runs them.** This file says exactly
what to carry over, what to build, and what to measure.

Read it with [`README.md`](README.md), which has the accuracy results and the reasoning.

## Why anything has to come back at all

A TensorRT engine is compiled for one GPU architecture and will not load on another. Beyond
that, the Orin has 8 GB shared between CPU and GPU, different memory bandwidth, and thermal
behaviour that only appears under sustained load. A throughput figure from a 350 W desktop card
says nothing about a 15 W board, so the desktop produced weights and accuracy and the board
produces every number about speed, power and memory.

**One accuracy caveat.** mAP50 measured on the 3090 matches the Jetson's to about three decimal
places but not bit exactly, because FP16 kernels differ between architectures. The gate check
run at the start of this work reproduced the board's unpruned 0.7775 as **0.7764**, a delta of
0.0011, which is the expected size. `prediction_fingerprint()` asserts bit reproducibility
*within* a machine, never across. Quote the 3090 numbers as screening, quote the board's as the
result.

## Environment the screening used

Record these next to any comparison, since they differ from the board:

| | 3090 (screening) | Jetson (authority) |
|---|---|---|
| torch | 2.13.0+cu126 | 2.11.0 |
| ultralytics | 8.4.118 | 8.4.118 |
| torch-pruning | 1.6.0 | 1.6.0 |
| batch (recovery) | 32 | 8 |
| epochs | 12 | 12 |

The two pins that matter are identical. Batch differs, but `lib/finetune.py` accumulates
gradients to a nominal batch of 64, so a larger real batch changes the number of accumulation
steps rather than the effective update.

## Priority 1. The two experiments that have no verdict yet

### E3, regularity: does `round_to` recover the missing speed?

This is the direct test of XP6's most interesting observation, that its iterative model kept
*more* parameters than its one-shot model and still ran 18% faster. The hypothesis is
channel-count regularity. `round_to` forces surviving channel counts onto a multiple, so the
same nominal cut lands on aligned widths.

Carry over, per arm:

```
weights/yolov5s_round{1,8,16,32}.pt      whole model objects, not state_dicts
weights/yolov5s_round{1,8,16,32}.onnx    architecture neutral, transfers cleanly
results/raw/xp06e3_*.json                the screening accuracy for each
```

On the board, for each arm:

```bash
python -c "
from lib.trt_export import build_fp16_engine
from pathlib import Path
build_fp16_engine(Path('weights/yolov5s_round32.onnx'),
                  Path('weights/yolov5s_round32_fp16_512.engine'), res=512)"
```

then measure through the frozen harness exactly as XP9 did, and record
`fps_batched`, `latency_ms_median`, `power_w_mean` and `energy_j_per_1000_frames`.

**What would confirm the hypothesis:** throughput rising with `round_to` at roughly constant
accuracy. `round_to=32` should be the fastest. If throughput is flat across all four, the
regularity explanation is wrong and the speed difference XP6 saw came from something else,
which is equally worth publishing.

The screening records already carry `widths_divisible_by_8/16/32` per arm, so the throughput
numbers can be read directly against how aligned each model actually is.

### E4, 2:4 sparsity: does the hardware path exist on this SKU?

Orin's GPU is Ampere class, so sparse tensor cores exist on paper. Whether TensorRT selects
them for this network is the question.

```
weights/yolov5s_sparse24.pt
weights/yolov5s_sparse24.onnx
results/raw/xp06e4_*.json
```

Build with sparsity **enabled**, which is not the default:

```bash
/usr/src/tensorrt/bin/trtexec --onnx=weights/yolov5s_sparse24.onnx \
    --saveEngine=weights/yolov5s_sparse24_fp16_512.engine \
    --fp16 --sparsity=enable \
    --minShapes=images:1x3x512x512 --optShapes=images:8x3x512x512 \
    --maxShapes=images:16x3x512x512 \
    --verbose 2>&1 | tee weights/sparse24_build.log
```

**Report three things separately and do not blur them together:**

1. **Accuracy** after masking and recovery. Already measured, in the screening record.
2. **Whether the compiler actually chose sparse kernels.** `--verbose` logs this. Grep the
   build log for `sparse` and record what it says. A build that succeeds with sparsity
   *enabled* is not the same as a build that *used* it, and reporting the first as the second
   would be the exact class of mistake this repo keeps catching.
3. **Throughput and energy**, against the dense FP16 engine at the same resolution.

It is entirely possible that accuracy holds and the speed-up never materialises. That is a
publishable result and fits the study's recurring theme, which is that hardware does not do what
the arithmetic promises.

Also build a **dense** engine from the same checkpoint as the control, so the comparison isolates
sparse kernel selection rather than mixing it with whatever the 2:4 masking cost in accuracy.

## Priority 2. The finalists worth re-measuring

These have accuracy verdicts already; what they lack is speed, and speed is the reason to prune.

| checkpoint | why it is interesting | screening mAP50 |
|---|---|---|
| `yolov5s_pruned25_lamp_recovered.pt` | best accuracy per parameter: 3.91 M, the smallest model here | 0.7543 |
| `yolov5s_pruned25_l1_recovered.pt` | 4.21 M, and cuts the most arithmetic of the good criteria (38.7%) | 0.7531 |
| `yolov5s_alloc_sensitivity.pt` | restores correct silence to the unpruned 97.4% | 0.7522 |

**Watch for a trap in the LAMP arm.** It reaches the smallest parameter count (3.91 M) while
cutting the *least* arithmetic (20.1% of MACs). Parameters and MACs diverge sharply here, and
XP6 already established that neither predicts speed on this hardware. LAMP could easily be the
smallest and the slowest at once. That would be a good finding, not a disappointment.

For each: export ONNX on the desktop or the board, build an FP16 engine at 512 px, and measure
through `lib/evaluator`. The line every one of them has to beat is unchanged:

**YOLOv5s TensorRT FP16 @512: 0.7776 mAP50, 474 img/s, 52 J per 1000 frames, 17 MB.**

## Priority 3. Nothing to build

Two arms deliberately produce no engine and should not get one.

**Fine-grained (E5) has no speed path on this hardware and must not be given one.** The weights
are masked, not removed, so the tensors keep their original shape. Irregular zeros have no
matching kernels; building an engine would produce a number identical to the dense model and
inviting someone to read it as a pruning result. Its purpose was to separate a capacity limit
from a structural one, and it did.

**The sensitivity sweep (E1) is a diagnostic on the validation split.** It selected the
allocation used in E6. It never touches test and produces no deployable model.

## When you have the numbers

Write them through `evaluator.results_record()` as usual, with `notes` recording that the
accuracy figure alongside is off-device screening. Commit the raw JSON from both machines:
they are small, they are the evidence, and `analysis/make_figures.py` and
`analysis/xp06_tables.py` rebuild every figure and table on the README from them.

Then update these three places in [`README.md`](README.md), which currently say the answer is
not known:

- Result 5, which still quotes only XP6's original speed measurements.
- The verdict table, which has no throughput column for the new models.
- The limitations list, whose first bullet says speed is unresolved for everything new.

If a pruned engine finally beats 474 img/s at 0.7776 mAP50, that is the first time anything in
this study has beaten simply feeding the network smaller pictures, and it should lead the page.
