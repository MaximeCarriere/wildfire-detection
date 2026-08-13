# Handoff: pruning work on the RTX 3090

**Read this together with the repository** at
<https://github.com/MaximeCarriere/wildfire-detection>. The repo tells you what has been
measured. This file tells you what to do next, why, and what must not change.

The goal of the next block of work has three parts, in order:

1. **Understand** every pruning technique in the MIT 6.5940 pruning lectures.
2. **Test** each of them on this detector, on real data, with real numbers.
3. **Explain** them in `experiments/xp06_pruning/README.md` well enough that the page
   serves two audiences at once: it teaches the techniques, and it demonstrates competence
   to someone evaluating the work commercially.

Those two audiences pull in different directions and the resolution is not to split the
page. It is to lead with the plain finding and keep the precision one line below it, which
is how every other README in this repo is written. Section 10 gives the required shape.

---

## 1. The project in sixty seconds

A fire and smoke detector running on a $249 fanless NVIDIA Jetson Orin Nano Super, with no
cloud connection. The question the repo answers is: **how small and how fast can this
detector get before it stops being useful?** Every technique is measured on the device,
including the ones that fail.

| | |
|---|---|
| Model | YOLOv5s, 7.03 M parameters, `0 = smoke`, `1 = fire` |
| Weights | The D-Fire authors' published detectors, not ours |
| Data | D-Fire, 21,527 images, ground level cameras and web photos |
| Splits | train 15,500 / val 1,721 / test 4,306, frozen and checksummed |
| Best measured | YOLOv5s + TensorRT FP16 @512 px: **0.7776 mAP50, 474 img/s, 52 J per 1000 frames, 17 MB** |

That last line is the frontier. **Every compression technique has to beat it or it loses.**
Pruning has not beaten it yet. Neither has INT8. Simply feeding the network 512 px images
instead of 640 did beat it, which is the running joke of the study: the cheap knob is
ahead of the clever ones.

Reported metrics are never a single number. Accuracy is always broken out as: overall
mAP50, fire, smoke, **small plumes** (ground truth box under 1% of frame area), **tiny
plumes** (under 0.1%, roughly 20x20 px, which is distant smoke and the thing an early
warning system exists for), and **correctly silent rate** on the 2,005 empty frames.

---

## 2. Rule zero: the 3090 makes weights, the Jetson makes numbers

This is the single most important constraint and it is not negotiable.

**A TensorRT engine is compiled for one GPU architecture and will not load on another.**
An engine built on a 3090 will not run on Orin, and a latency number from a 3090 says
nothing about a 15 W embedded board. On top of that, the Jetson has 8 GB shared between
CPU and GPU, a completely different memory bandwidth, and thermal behaviour that only
shows up over ten minutes of sustained load.

So the division of labour is:

| The 3090 produces | The Jetson produces |
|---|---|
| Pruned and fine-tuned `.pt` checkpoints | TensorRT engines |
| Accuracy screening (mAP50 and the size tiers) | **All** latency, throughput, fps |
| Sensitivity curves, criterion sweeps, ablations | **All** power and energy per 1000 frames |
| Anything that needs many training runs | **All** memory and endurance |
| ONNX files (portable, architecture neutral) | The number that goes in the README |

**Practical consequence:** screen widely on the 3090, then ship the two or three finalist
checkpoints back and re-measure them on the board. A technique that wins on 3090 accuracy
but is not re-measured on Orin has not been tested, it has been guessed at.

**One accuracy caveat.** mAP50 computed on the 3090 will match the Jetson's to about three
decimal places but not bit exactly, because FP16 kernels differ between architectures. That
is fine and expected. The repo already accepts this: `prediction_fingerprint()` asserts
bit reproducibility **within** a machine, not across machines. Quote 3090 accuracy as
screening evidence; quote the Jetson's re-measurement as the result.

---

## 3. Setting the machine up

### 3.1 Repository

```bash
git clone https://github.com/MaximeCarriere/wildfire-detection.git
cd wildfire-detection
```

### 3.2 Python environment

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install ultralytics==8.4.118 torch-pruning==1.6.0 \
            onnx onnxslim pycocotools numpy matplotlib pandas
```

Pin `ultralytics==8.4.118` and `torch-pruning==1.6.0` to match the board exactly. Same
training and inference code path on both machines, or the frozen protocol is not frozen.
The torch version itself is not load bearing off device (the board runs 2.11.0), but
**record whatever you use in the XP6 README**.

### 3.3 The dataset (about 3 GB)

`data/` is gitignored. Download "D-Fire dataset (only images and labels)" from
<https://github.com/gaiasd/DFireDataset>. The OneDrive link needs a browser session;
scripted GETs get 401/403.

```bash
unzip -q D-Fire.zip -d data/dfire
```

Expected layout and integrity:

```
data/dfire/train/images/   17,221 jpg
data/dfire/train/labels/   17,221 txt      YOLO: <class> <cx> <cy> <w> <h>, normalized
data/dfire/test/images/     4,306 jpg
data/dfire/test/labels/     4,306 txt
```

`D-Fire.zip` is 3,036,222,313 bytes,
sha256 `3824fb3ce32cfa8b538792dfd460603648d271072ac6ae34f1af0c713f60260c`.

**Do not re-run `experiments/xp00_foundation/run.py`.** The split files in `data/splits/`
are committed and frozen. Regenerating them would invalidate every number in the repo.

### 3.4 The YOLOv5 repository

The models are original YOLOv5, not Ultralytics-loadable (see the trap in section 9.1).
Clone the original repo and point the code at it:

```bash
git clone https://github.com/ultralytics/yolov5 ~/yolov5
export YOLOV5_REPO=~/yolov5      # lib/detectors.py reads this, default is ~/yolov5
```

### 3.5 Baseline weights

`yolov5s.pt` (14.4 MB) and `yolov5l.pt` (92.8 MB) from
<https://github.com/pedbrgs/Fire-Detection>, trained by the D-Fire authors on D-Fire's
official train split. Behind OneDrive links, browser required. Put them in `weights/`.

Verify you have the right file before anything else:

```python
from lib.prune_utils import load_yolov5
from pathlib import Path
m = load_yolov5("weights/yolov5s.pt", Path.home()/"yolov5")
print(m.names)          # MUST print ['smoke', 'fire'], not 80 COCO classes
```

### 3.6 Verify the environment before running anything

```bash
python -c "
from lib import data as d
print(d.verify_splits())          # raises if the splits drifted
print({k: len(d.load_split(k)) for k in ['train','val','test']})
"
```

Expected: `train 15500, val 1721, test 4306`, and checksums
`train 8dea80450f1d1611`, `val 48c7552154b86a49`, `test 7cc103e8cbb706f5`.

Then reproduce a known number as a smoke test. Evaluate the unpruned `yolov5s.pt` in
PyTorch FP16 at 512 px through `lib/evaluator.py`. You should get mAP50 close to
**0.7775**. If you do not, stop and find out why before doing any pruning work.

---

## 4. The measurement protocol you must not change

Every number in this repo goes through one harness, `lib/evaluator.py`, stamped with
`PROTOCOL_VERSION = "1.1"`. If the harness changes, **all** prior results are invalid and
must be re-run. `analysis/make_figures.py` refuses to draw a figure that mixes protocol
versions, so this is enforced rather than remembered.

Things that are load bearing:

- **`dataset.verify_splits()` is called inside `lib/finetune.py:build_loader`.** Training
  refuses to start if the split lists do not match `data/splits/manifest.json`. This exists
  precisely because work is moving to another machine. Weights can be made anywhere; if the
  split membership changes on the other machine, test images leak into training and every
  published number silently becomes a score on an exam the model has already seen. **Do not
  disable this check.** If it fires, your dataset copy is wrong.
- **Accuracy is COCO style mAP via pycocotools**, with the size tiers sliced using the
  `iscrowd` trick, and mAP50-95 computed directly from the precision array (COCOeval's
  `summarize()` hardcodes `maxDets=100` and we use 300, so calling it returns -1).
- **Empty frames are scored separately.** `background_false_alarms()` returns both
  `false_alarm_rate` and `correctly_silent_rate`. 47% of the test set contains no fire.
  Never fold this into an aggregate.
- **Results are one JSON per run** in `results/raw/`, written by
  `evaluator.results_record()` and `evaluator.write_results()`. Figures are generated only,
  never hand edited.
- **Report the full test set.** There is a `--quick` path in some runners; it is for
  debugging only and its results never go in a README.

Reuse the existing entry points rather than writing new ones:

| Purpose | Entry point |
|---|---|
| Load an original-YOLOv5 checkpoint as a trainable model | `lib.prune_utils.load_yolov5` |
| Parameters and MACs at a given resolution | `lib.prune_utils.model_stats` |
| One-shot structured channel prune | `lib.prune_utils.prune_channels` |
| Progressive prune with a recovery callback | `lib.prune_utils.prune_iterative` |
| Save a pruned model (whole object, not a state_dict) | `lib.prune_utils.save_pruned` |
| Recovery fine-tune | `lib.finetune.finetune` |
| Accuracy, latency, throughput, records | `lib.evaluator` |
| ONNX export and engine building | `lib.trt_export` (engine building is Jetson only) |
| The full damage-to-deploy chain | `experiments/xp06_pruning/recover_and_deploy.py` |

Current defaults in `recover_and_deploy.py`: `BASE = "yolov5s"`, `RES = 512`, batch 8.
On a 3090 raise the batch (32 or 64) and keep everything else identical. `lib/finetune.py`
already accumulates gradients to a nominal batch of 64, so raising the real batch reduces
accumulation steps rather than changing the effective update. That is the correct behaviour
and is why the loop is written that way.

---

## 5. What is already known

| # | Experiment | Result |
|---|---|---|
| XP0 | Data, splits, harness | Splits frozen and checksummed; class ids proven from box counts, not assumed |
| XP1 | Baselines | A 6.6x bigger model buys 1.4 accuracy points at 3.3x the energy |
| XP2 | Resolution | 512 px beats 640 px outright; overall accuracy hides a 77% collapse on tiny plumes |
| XP6 | Pruning | Loses on accuracy and speed; cutting 2% of channels costs 9 accuracy points |
| XP9 | TensorRT FP16 | Up to 5x faster, accuracy free; exposed that all earlier speed numbers measured the software |
| XP10 | INT8 | One default setting cost 67% of the accuracy; fixed, the real cost is about 8% |
| XP12 | Endurance | 10 min flat out, 280k images, -1.3% drift, no throttling |

### 5.1 XP6 in detail, because that is what you are extending

**Damage with no retraining at all**, full test set, 512 px:

| channels cut | params | MACs cut | mAP50 | small plumes | tiny plumes |
|---:|---:|---:|---:|---:|---:|
| none | 7.03 M | none | **0.7775** | 0.6062 | 0.1386 |
| 2% | 6.84 M | 4.1% | 0.6872 | 0.4441 | 0.0650 |
| 5% | 6.53 M | 10.4% | 0.0956 | 0.0327 | 0.0036 |
| 10% | 5.97 M | 19.8% | 0.0052 | 0.0000 | 0.0000 |
| 25% | 4.24 M | 43.1% | 0.0000 | 0.0000 | 0.0000 |
| 50% | 1.86 M | 73.0% | 0.0000 | 0.0000 | 0.0000 |
| 70% | 0.62 M | 88.9% | 0.0000 | 0.0000 | 0.0000 |

This collapse is far steeper than the textbook curves. **The confidence head fails first:**
maximum objectness falls from 0.72 to 0.0045 by the 5% cut, so the model still produces
boxes internally but none survive any sensible threshold. That is why the table shows exact
zeros rather than a gentle slide, and it is why recovery training here is not a refinement,
it is the experiment.

**After 25% pruning and 12 epochs of recovery, exported to TensorRT FP16 @512:**

| | unpruned | one-shot | iterative |
|---|---:|---:|---:|
| parameters | 7.03 M | 4.24 M | 4.51 M |
| **mAP50** | **0.7776** | 0.7297 | 0.6771 |
| small plumes | **0.6061** | 0.5516 | 0.4386 |
| tiny plumes | **0.1376** | 0.0960 | 0.0653 |
| correctly silent | **97.4%** | 95.2% | 92.7% |
| throughput | **473.7 img/s** | 381.1 | 450.2 |
| energy per 1000 frames | 52.1 J | 54.3 | **46.5 J** |

Three findings worth carrying forward:

1. **Neither arm beats the unpruned model.** One-shot gives up 4.8 accuracy points and runs
   20% slower.
2. **Arithmetic removed is not speed gained.** Removing 88.9% of the multiply-adds bought
   1.7x the throughput, not the ~9x the arithmetic implies. Pruned channel counts are
   irregular (47 instead of 64) and GPU kernels are tuned for regular tile sizes.
3. **The iterative model is faster while being larger** (4.51 M params, 450 img/s versus
   4.24 M and 381 img/s). Removing channels gradually appears to leave more regular channel
   counts. **How you prune changes speed more than how much you prune.** Finding 3 is the
   most interesting unexplored thread in the whole experiment and section 7.3 turns it into
   a controlled test.

### 5.2 The known confound in XP6, fix it early

Both arms got 12 epochs total, but the iterative model's **final architecture** only
existed for the last 4 of them, while the one-shot model trained all 12 in its final shape.
Equal total budget is not equal recovery budget, and that plausibly explains part of
iterative's deficit. A fair rerun gives both the same number of epochs **after the final
cut**. On a 3090 this is cheap. Do it (experiment E7), because until it is done the
one-shot versus iterative comparison in the README is not a clean result and is flagged as
such in its limitations.

---

## 6. The lecture, mapped to this project

`Lec03-Pruning-I.pdf` (MIT 6.5940, Song Han) frames pruning as four decisions. Lecture 3
covers the first two in depth; ratio and fine-tuning are Lecture 4, though XP6 already
touches both.

```
minimize  L(x; W_P)   subject to   ||W_P||_0 <= N
```

Everything below is a different answer to "which entries of W do we zero, in what pattern,
how many per layer, and what training do we do afterwards".

### 6.1 Axis A: granularity. In what pattern do we prune?

The lecture lays these on a spectrum from irregular to regular. Regular means easier to
accelerate; irregular means a higher achievable compression ratio.

| Granularity | What is removed | Accelerates on | Status here |
|---|---|---|---|
| **Fine-grained / unstructured** | Individual weights, anywhere | Custom hardware (EIE), not GPUs easily | Not tested |
| **Pattern-based (N:M, e.g. 2:4)** | N of every M contiguous weights | **Ampere sparse tensor cores, ~2x peak** | Not tested |
| **Vector-level** | Rows within a kernel | Partially | Not tested |
| **Kernel-level** | Whole k_h x k_w kernels | Partially | Not tested |
| **Channel-level** | Whole input or output channels | **Any hardware, it is just a smaller matrix** | **XP6, done** |

Two points from the lecture that matter directly here:

- Fine-grained pruning reaches much larger compression ratios (AlexNet 9x, VGG-16 12x) but
  "can deliver speed up on some custom hardware (e.g. EIE) but not GPU (easily)". On this
  project it is therefore an **accuracy ceiling experiment**, not a deployment candidate.
  That is still worth running, because it separates "this network cannot lose 25% of its
  capacity" from "this network cannot lose 25% of its channels". Those are very different
  conclusions and XP6 currently cannot tell them apart.
- 2:4 sparsity is 50% sparsity, is supported by Ampere, and NVIDIA's published table shows
  it "usually maintains accuracy": ResNet-50 76.1 to 76.2 top-1, **SSD-RN50 24.8 to 24.8
  bbAP on COCO**. That detection row is the reason 2:4 is the highest value untested
  technique in this whole plan. Orin's GPU is Ampere class, so the hardware path exists,
  but whether TensorRT delivers a real speedup on this SKU has to be **measured on the
  board**, not assumed from a datasheet.

The lecture also distinguishes **uniform shrink** (same sparsity every layer) from
**channel prune** (different sparsity per layer), and shows the non-uniform version
dominating uniform scaling on the accuracy versus latency curve (AMC, He et al. ECCV 2018).
XP6 used global pruning, which is a third thing again: one global threshold, letting the
ratio fall where it may. All three deserve a row in the table.

### 6.2 Axis B: criterion. Which synapses or neurons do we prune?

The principle: the less important the removed parameters, the better the pruned network.
The lecture gives five families.

**1. Magnitude based.** Larger absolute value means more important.

- Element-wise: `importance = |W|`
- Structural (row-wise or channel-wise), L1: `importance = sum_{i in S} |w_i|`
- Structural, L2: `importance = sqrt(sum_{i in S} |w_i|^2)`
- General L_p norm: `||W_S||_p = (sum |w_i|^p)^(1/p)`

XP6 used L2 structural (`GroupMagnitudeImportance(p=2)`). L1 is one character away and
untested.

**2. Scaling based (Network Slimming, Liu et al. ICCV 2017).** A trainable scaling factor
per output channel decides its fate; the factors can be **reused directly from the batch
norm gamma**, since BN already computes `z_o = gamma * (z_i - mu)/sqrt(var + eps) + beta`.
Cheap, and YOLOv5 is BN-dense so this is nearly free to try. In torch-pruning:
`BNScaleImportance()`.

**3. Second order (Optimal Brain Damage, LeCun et al. 1989).** Approximate the loss change
from removing a weight with a Taylor series and drop the terms OBD argues away:

```
dL_i = L(x; W) - L(x; W_P | w_i = 0)  ~=  (1/2) h_ii w_i^2
importance = |dL_i| = (1/2) h_ii w_i^2
```

The three assumptions are stated explicitly by the lecture and are worth quoting in the
README, because two of them are visibly questionable for a fine-tuned detector: the
objective is nearly quadratic (last term dropped), training has converged (first order
terms dropped), and per-parameter errors are independent (cross terms dropped). The lecture
then says the quiet part out loud: **"Hessian Matrix H is difficult to compute."** In
torch-pruning: `GroupHessianImportance()`, which needs accumulated gradients over a data
pass. There is also **first order Taylor** (Molchanov et al., ICLR 2017 and CVPR 2019),
which keeps the gradient term instead of the Hessian and is far cheaper:
`GroupTaylorImportance()`.

**4. Percentage of zero activations (APoZ, Hu et al. 2017).** ReLU produces zeros; a
channel that is almost always zero is doing little. Importance is the inverse of the
average percentage of zeros across a data pass. **Caveat specific to this model:** YOLOv5
uses SiLU, not ReLU, so exact zeros are rare and literal APoZ will be near-degenerate. The
honest adaptation is an activation-magnitude criterion (mean absolute activation, or
fraction below a small threshold), and **the adaptation must be stated as an adaptation in
the README**. torch-pruning has `ActivationImportance` for this shape of criterion. This is
a good teaching moment: a criterion from a ReLU-era paper does not transfer unexamined to a
SiLU network, and saying so is worth more than a number.

**5. Regression based (He et al. ICCV 2017).** Instead of minimizing the change in the
loss, minimize the **reconstruction error of the layer's own output**:

```
argmin_{W, beta}  || Z - Z_hat ||_F^2  =  || Z - sum_c beta_c X_c W_c^T ||_F^2
subject to ||beta||_0 <= N_c
```

Solved by alternating: fix W and solve beta for channel selection (a LASSO problem), then
fix beta and solve W by least squares to minimize reconstruction error. This is the most
implementation-heavy item on the list and has no ready torch-pruning equivalent. It is the
right thing to attempt **last**, and an honest "attempted, here is where it got hard" is a
perfectly good outcome for it.

Two more that torch-pruning offers and that are worth a row even though the lecture only
gestures at them: **FPGM** (geometric median, `FPGMImportance`, the idea being that the
channel closest to the geometric median of its layer is the most redundant rather than the
smallest) and **LAMP** (layer-adaptive magnitude, `LAMPImportance`, which normalizes
magnitudes per layer and so implicitly answers the ratio question too). And **random**
pruning (`RandomImportance`), which sounds like a joke and is not: it is the control that
tells you how much of your criterion's benefit is really coming from the criterion. Given
how badly this model responds to any pruning at all, the random control could be the most
informative single run in the set.

### 6.3 Axis C: ratio. What sparsity per layer?

Three positions:

- **Uniform:** same ratio everywhere. Simple, and the lecture shows it is beaten.
- **Global:** one threshold across the network, ratios fall out. **This is what XP6 did**
  (`global_pruning=True`).
- **Sensitivity driven:** measure each layer's tolerance separately, then allocate.

Sensitivity analysis is the missing diagnostic, and given XP6's collapse it may be the most
valuable single experiment in this plan. The procedure: for each prunable layer, prune only
that layer at a sweep of ratios, measure accuracy with no retraining, and plot one curve per
layer. Layers whose curve stays flat can absorb heavy pruning; layers whose curve falls off
a cliff must be protected. XP6's global pruner had no such map and may well have been
spending its budget in exactly the wrong place, which would explain why 2% of channels cost
9 accuracy points. **This experiment can either rescue pruning on this model or prove that
nothing can.** Either outcome is a result.

Automated methods (AMC, reinforcement learning over the ratio space; NetAdapt) are Lecture
4 material. They are legitimate future work; do not start there.

### 6.4 Axis D: fine-tune. How do we repair the damage?

The lecture's own progression, in one figure: pruning alone falls off past ~50% sparsity;
pruning plus fine-tuning holds to ~80%; **iterative** prune-and-fine-tune holds past 90%.

XP6 tested one-shot and iterative and found the opposite ordering, which is a real
disagreement with the literature and is currently confounded (section 5.2). Untested
variants worth a run each:

- **Equal epochs after the final cut** (the confound fix, E7).
- **Longer recovery.** XP6 used 12 epochs because that is what the board allows overnight.
  On a 3090, 50 epochs is affordable. This alone may change the verdict, and if it does,
  that is a finding about **who can afford to prune**, not about pruning.
- **Learning rate rewinding** versus continued decay.
- **Teacher supervised recovery**, recovering the pruned student under supervision from
  YOLOv5l. Cheap to describe, unpromising here: XP1 showed the larger model is only 1.4
  mAP points better, so there is very little it knows that the small one does not.

### 6.5 Coverage scorecard

Put this table in the README, updated. It is the honest one-glance answer to "did you
actually cover the material".

| Axis | Covered by XP6 | Still open |
|---|---|---|
| Granularity | Channel level | Fine-grained, **N:M / 2:4**, vector, kernel |
| Criterion | L2 structural magnitude | L1, BN scale, Taylor, Hessian/OBD, FPGM, LAMP, activation/APoZ, regression, **random control** |
| Ratio | Global automatic | Uniform, **sensitivity driven**, per-layer dict |
| Fine-tune | One-shot and iterative, 12 epochs | Equal-post-cut epochs, longer budget, LR rewind, teacher supervised |

---

## 7. The experiment plan

Ordered by information gained per hour. Do them in this order. Each one produces a
`results/raw/*.json` through the frozen harness and at least one row in a table.

### E1. Sensitivity analysis (do this first)

**Question:** which layers of this detector can absorb pruning and which cannot?

For each prunable layer, prune that layer alone at ratios {10, 20, 30, 50, 70}%, evaluate on
the **val** split (not test) with no retraining, record mAP50. Output: one curve per layer,
and a ranked list of fragile layers.

Use `pruning_ratio_dict` on `MetaPruner` to target a single layer. Val split, not test,
because this is a diagnostic that will inform later choices and test must stay clean.

**Why first:** it either explains XP6's collapse or rules out the obvious explanation, and
its output is the input to E6. Expect a few hours of pure inference, no training.

**Deliverable:** a heatmap or small-multiples figure, layer index versus ratio, coloured by
accuracy retained. This is also the most visually striking figure in the whole plan.

### E2. Criterion sweep at fixed ratio

**Question:** does the choice of importance criterion matter on this model, and does any of
it beat random?

Fix ratio at 25% one-shot (matching the existing XP6 arm so it is directly comparable), and
sweep: L2 magnitude (the existing result, free), L1 magnitude, BN scale, Taylor, Hessian,
FPGM, LAMP, and **random**. Measure damage with no retraining first, which is cheap, then
fine-tune only the top three and random.

**Why:** it is the largest block of untested lecture material and the single fixed ratio
keeps it honest. The random control is what turns this from a leaderboard into a result.

### E3. Regularity: does `round_to` recover the missing speed?

**Question:** XP6 found the iterative model was faster while larger, hypothesised to be
channel-count regularity. Test it directly.

`MetaPruner` takes a `round_to` argument that rounds surviving channel counts to a multiple.
Prune 25% one-shot with `round_to` in {1 (the XP6 baseline), 8, 16, 32}, fine-tune each
identically, export, and **measure throughput on the Jetson**.

**Why:** this converts XP6's most interesting observation from a hypothesis into a
controlled result, and it is the kind of hardware-aware finding that distinguishes a
measurement study from a tutorial. If regularity is the mechanism, `round_to=32` should
recover throughput at equal or better accuracy, and that is a genuinely useful practical
rule.

**This one absolutely requires the board for its headline number.** The 3090 produces the
four checkpoints; Orin produces the four throughput figures.

### E4. 2:4 structured sparsity

**Question:** the one granularity with dedicated hardware behind it. Does it hold accuracy
here the way it holds it for SSD-RN50 on COCO?

Path: apply a 2:4 mask (NVIDIA's ASP recipe, or `torch.ao.pruning`'s semi-structured
sparsity, or a hand-written mask keeping the two largest of every four contiguous weights),
fine-tune to recover, export to ONNX, then **on the Jetson** build the engine with
TensorRT's sparse weights builder flag and measure.

Report three things separately and do not conflate them: accuracy after masking and
recovery, whether TensorRT actually selected sparse kernels (it logs this), and measured
throughput. It is entirely possible that accuracy holds and the speedup does not
materialise on this SKU. **That is a publishable result and fits the repo's character
exactly**, which is that hardware does not do what the arithmetic promises.

### E5. Fine-grained pruning as the accuracy ceiling

**Question:** is XP6's collapse a capacity problem or a channel-structure problem?

Unstructured magnitude pruning at {25, 50, 70, 90}% sparsity, fine-tune, measure accuracy
only. Do not measure speed, and say clearly in the README that no speedup is expected on
this hardware, and why (irregular sparsity, no matching kernels).

**Why:** if the network tolerates 90% unstructured sparsity but dies at 5% channel pruning,
the story is "the capacity is there, the structure is the constraint", which is a far more
interesting and more accurate conclusion than "pruning does not work here". This experiment
is what licenses that sentence.

### E6. Ratio allocation: uniform versus global versus sensitivity driven

Using E1's output, build a `pruning_ratio_dict` that protects the fragile layers and cuts
hard where there is slack. Compare against uniform at the same overall ratio and against
XP6's global result. Same total parameter reduction across all three, so the comparison is
about allocation and nothing else.

### E7. The fair one-shot versus iterative rerun

Fix the confound in section 5.2: equal epochs **after the final cut**, on both arms. Then,
separately, extend recovery to 50 epochs on the winner to see whether the verdict against
pruning survives a training budget the board cannot afford.

### E8. Optional, if there is time

Regression-based channel selection (section 6.2 item 5), and teacher-supervised recovery
from YOLOv5l. Both are honest to attempt and honest to abandon with a written reason.

---

## 8. Code you will need to add

`lib/prune_utils.py` currently hardcodes two criteria. Widen the registry. These class
names and signatures are **verified against torch-pruning 1.6.0 as installed on the board**:

```python
# lib/prune_utils.py
def _make_importance(name: str):
    """The lecture's criteria, as torch-pruning importance objects.

    Names map to Lec03 sections: magnitude (L1/L2/Lp), scaling (BN gamma,
    Network Slimming), second order (Taylor, Hessian/OBD). `random` is the
    control that says how much of a criterion's benefit is the criterion.
    """
    import torch_pruning as tp
    return {
        "l2":      tp.importance.GroupMagnitudeImportance(p=2),   # XP6 baseline
        "l1":      tp.importance.GroupMagnitudeImportance(p=1),
        "bn":      tp.importance.BNScaleImportance(),             # Liu et al. ICCV 2017
        "taylor":  tp.importance.GroupTaylorImportance(),         # Molchanov et al.
        "hessian": tp.importance.GroupHessianImportance(),        # OBD-flavoured
        "fpgm":    tp.importance.FPGMImportance(p=2),
        "lamp":    tp.importance.LAMPImportance(p=2),
        "random":  tp.importance.RandomImportance(),              # the control
    }[name]
```

Three things to know before wiring these up:

1. **Taylor and Hessian need gradients.** They are not pure weight statistics. Run a
   backward pass over a batch (or a few hundred images) with the real YOLOv5 loss
   accumulated into `.grad` before calling `pruner.step()`. torch-pruning's own examples
   show the pattern: zero grads, forward, `loss.backward()`, then step the pruner. Use
   `lib.finetune.build_loader` to get a loader with the correct hyper-parameters, and
   `ComputeLoss` from the YOLOv5 repo for the loss.
2. **`RandomImportance` takes no useful arguments**, hence the bare call.
3. **`ActivationImportance` needs forward hooks** on the target modules, so it does not fit
   the one-line registry. Give it its own small helper if you run the APoZ adaptation.

Also expose the two `MetaPruner` arguments the plan needs, both already supported:

```python
pruner = tp.pruner.MetaPruner(
    model, example,
    importance=imp,
    pruning_ratio=ratio,
    pruning_ratio_dict=per_layer,     # E1 sensitivity, E6 allocation
    round_to=round_to,                # E3 regularity, try 8 / 16 / 32
    ignored_layers=detect_head_convs(model),
    global_pruning=global_pruning,    # make this a parameter: E6 needs it False
)
```

**Never prune the detect head.** Each of YOLOv5's three head convolutions emits exactly
`anchors * (5 + num_classes)` channels and that number is fixed by the output format. Prune
it and the decode arithmetic no longer describes the tensor it is given.
`detect_head_convs()` already finds them; keep passing it.

---

## 9. Traps already hit, do not rediscover them

Every one of these cost real hours. They are written up here so they cost you none.

**9.1 Ultralytics silently substitutes a different model.** Given `yolov5s.pt`, the
Ultralytics API downloaded `yolov5su.pt` instead: a different, COCO-trained, 80-class model.
It ran, it produced plausible-looking boxes, and it was measuring the wrong thing entirely.
Caught only by printing class names. **Always assert `model.names == ['smoke', 'fire']`.**
`lib/detectors.py:Yolov5Detector` does this for you; use it rather than loading models
yourself.

**9.2 COCOeval returns -1 for mAP50-95.** Its `summarize()` hardcodes `maxDets=100`; this
harness uses 300. Compute from the precision array directly. Already handled in
`lib/evaluator.py:_mean_precision`, do not "fix" it back.

**9.3 The training loop can destroy the model, and did.** A first fine-tune implementation
took the unpruned model from 0.8184 to **0.0079** in one epoch. Two recovery numbers had to
be publicly withdrawn. Three independent omissions, each sufficient alone:

- No gradient accumulation to YOLOv5's nominal batch of 64.
- Unscaled loss gains: the classification gain must be `cls * nc/80 * 3/nl`, which is
  **0.0125** for 2 classes, not the raw 0.5. It was 40x too heavy, drowning out objectness,
  and objectness is exactly what collapsed.
- No EMA. YOLOv5's published accuracy comes from an exponential moving average of the
  weights.

All three are fixed in `lib/finetune.py`. **Use that loop.** And before you believe any
recovery number, run the control: fine-tune the **unpruned** model for one epoch and check
it still scores around 0.79 to 0.82. If the control fails, every recovery number from that
session is a measurement of the training loop, not of pruning.

**9.4 ONNX export can silently write a stub.** torch 2.11's dynamo exporter fails on
YOLOv5's `Resize` and wrote a 0.5 MB file for a 4.24 M parameter model. Export with
`dynamo=False` and assert on the output file size. Already handled in
`lib/trt_export.py:export_onnx_from_model`.

**9.5 Repeated dataloader creation gets workers OOM-killed.** On the board, iterative
pruning created a fresh loader (4 workers) and a fresh EMA per increment and never freed
them; the third round died. Fixed with `_LOADER_CACHE`, workers reduced to 2, and explicit
`del` / `gc.collect()` / `empty_cache()`. On a 3090 with 24 GB you can raise workers, but
keep the cache: it is also just faster.

**9.6 Defaults are not neutral.** The INT8 story (XP10) is the standing reminder: TensorRT's
default entropy calibration decided the useful input range was 0 to 0.4475 out of 0 to 1,
flattening the bright sky where faint smoke lives, and cost 67% of the accuracy. It looked
exactly like "INT8 is bad for detection", a tidy quotable wrong conclusion. It was one
option, and MinMax fixed it. When a pruning criterion produces a catastrophic result,
suspect the setting before you write the conclusion.

**9.7 Parameter count and MAC count are both poor predictors of speed** on this hardware.
Report them as structural facts, never as performance claims. The verdict always comes from
measured throughput on the board.

**9.8 A pruned model cannot be saved as a state_dict.** Its channel counts no longer match
the architecture file, so it will not load back into a fresh `Model(cfg)`. Save the whole
object, which is what `save_pruned()` does.

---

## 10. What the pruning README must deliver

`experiments/xp06_pruning/README.md` is the deliverable. It has to work as **formation**
(teaching the techniques to their author) and as **demonstration** (evidence of competence
to a client or investor) at the same time.

House style, non-negotiable because the rest of the repo follows it:

- **No em dashes.** Use commas, colons, parentheses, or a full stop.
- **Each README reads in isolation.** No references to "the plan", no "as we saw in the
  previous experiment" as the load-bearing link. A reader landing on this page from a search
  result must get the whole story. Cross-links are fine as pointers, never as prerequisites.
- **Lead accessible, stay precise.** Plain-language claim first, the number that supports it
  immediately after. A non-technical reader should get the finding from the bold sentences
  alone; a technical reader should find the rigour in the same paragraph.
- **Every figure clean and near publication ready**, readable by someone who is not a
  specialist. Generated by `analysis/make_figures.py` from committed JSON, never hand
  edited, never orphaned (a figure not referenced by a README gets deleted).
- **Bug stories condensed inline, about three lines each.** They are part of the evidence,
  not an appendix.
- **State the models, sizes and provenance up front**, as bullets or a table, so the page
  is self-contained from the first screen.
- Length: more than one page is fine, still short. Cut anything that is not a finding, a
  limitation, or a definition the reader needs.
- Because the page contains the word "plume", it needs the plume definition box and
  `results/figures/plume_definition.png`, like every other page that uses the term.

Required content, over and above what is there now:

1. **A short conceptual section per axis**, in the reader's language. What granularity
   means, what a criterion is, what the ratio decision is, why fine-tuning is not optional.
   Two or three sentences each, with the formula where the formula is genuinely clearer than
   prose (magnitude and OBD qualify; regression-based probably does not).
2. **The coverage scorecard** from section 6.5, updated to reflect what you actually ran.
   Honesty about what was not tested is part of the demonstration, not a weakness in it.
3. **One results table per axis**, all numbers through the frozen harness, all six accuracy
   metrics present, with the unpruned frontier as the top row every time so the reader never
   loses the comparison.
4. **The verdict, stated plainly**, whatever it turns out to be. If pruning still loses to
   simply running the unpruned model at 512 px, say so in the first three lines. The repo's
   credibility comes from publishing the techniques that failed.
5. **Limitations**, in the existing style: single dataset, single architecture, recovery
   budget, whatever confounds remain.

Suggested figures, at most one or two per axis so the page stays readable:

- **Sensitivity heatmap** (E1): layer versus pruning ratio, coloured by accuracy retained.
  Likely the strongest figure in the repo.
- **Criterion comparison** (E2): grouped bars, all criteria including random, with the
  unpruned line drawn across.
- **Regularity versus throughput** (E3): `round_to` on x, throughput and mAP50 as two
  panels sharing the x axis. Never a dual y axis.
- **The granularity spectrum** (E4, E5): accuracy against sparsity, one line per
  granularity, with a clear marker for which points have hardware support.

---

## 11. Handing results back

For each finalist configuration, send back:

1. The pruned and fine-tuned `.pt` (whole model object, from `save_pruned`).
2. The `.onnx` (architecture neutral, so it transfers cleanly).
3. The `results/raw/*.json` from the 3090 side, with `notes` recording that it is a 3090
   screening measurement, the torch version, batch size, and epoch count.
4. The `prune_meta` and `train_meta` dictionaries, which `recover_and_deploy.py` already
   attaches to the record.

Then on the Jetson: build the engine, re-measure through the same harness, and **the
Jetson's numbers are the ones that go in the README**. Keep the 3090 accuracy figure
alongside as corroboration if it is useful; label it clearly as off-device.

Commit the raw JSON from both machines. They are small, they are the evidence, and
`analysis/make_figures.py` rebuilds every figure from them.

---

## 12. References from the lecture

Worth reading in this order if you want the primary sources behind section 6.

1. Han et al., *Learning Both Weights and Connections for Efficient Neural Networks*,
   NeurIPS 2015. The train / prune / fine-tune loop and the iterative curve.
2. LeCun et al., *Optimal Brain Damage*, NeurIPS 1989. The second-order criterion.
3. Mao et al., *Exploring the Granularity of Sparsity in Convolutional Neural Networks*,
   CVPR-W. The granularity spectrum figure.
4. NVIDIA, *Accelerating Inference with Sparsity Using the Ampere Architecture and
   TensorRT*. 2:4 sparsity, and the accuracy table that includes detection.
5. Liu et al., *Learning Efficient Convolutional Networks through Network Slimming*,
   ICCV 2017. BN-scale criterion.
6. Wen et al., *Learning Structured Sparsity in Deep Neural Networks*, NeurIPS 2016.
   Structural L_p norms.
7. Molchanov et al., *Pruning Convolutional Neural Networks for Resource Efficient
   Inference*, ICLR 2017, and *Importance Estimation for Neural Network Pruning*, CVPR 2019.
   Taylor criteria.
8. Hu et al., *Network Trimming*, arXiv 2017. APoZ.
9. He et al., *Channel Pruning for Accelerating Very Deep Neural Networks*, ICCV 2017.
   Regression-based selection.
10. He et al., *AMC: AutoML for Model Compression*, ECCV 2018. Automated ratio search.
11. Luo et al., *ThiNet*, ICCV 2017.
12. Frantar and Alistarh, *SparseGPT*, arXiv 2023. Where one-shot pruning went next.

Course materials: <https://efficientml.ai>. Ratio selection, fine-tuning strategy, automated
ratio search and system support for each granularity are covered in the **Pruning II**
lecture, which is the natural next document to work through after this block.
