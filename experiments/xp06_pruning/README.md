# XP6. Pruning: how much of this detector can you delete?

**Question:** pruning deletes parts of a trained network to make it smaller and faster. Does
any version of it beat the far simpler option of feeding the same network a smaller image?

**Outcome:** still no, but the earlier verdict was right for the wrong reason. The best pruned
model here keeps **97% of the accuracy at 56% of the parameters**, and the collapse this page
originally reported turned out to be mostly an artifact of one badly chosen setting.

**The line to beat**, and nothing on this page beats it: YOLOv5s at 512 px, TensorRT FP16,
**0.7776 mAP50 at 474 img/s and 52 J per 1000 frames** on the Jetson Orin Nano Super.

## What was measured, and on what

| | |
|---|---|
| Model | YOLOv5s, 7.03 M parameters, classes `0 = smoke`, `1 = fire` |
| Weights | The D-Fire authors' published detectors ([pedbrgs/Fire-Detection](https://github.com/pedbrgs/Fire-Detection)), not ours |
| Data | [D-Fire](https://github.com/gaiasd/DFireDataset), 21,527 images, ground-level cameras and web photos |
| Splits | train 15,500 / val 1,721 / test 4,306, frozen and checksummed |
| Resolution | 512 px, the deployment resolution established in [XP2](../xp02_resolution/) |
| Accuracy | Full 4,306-image test set unless a table says otherwise |
| Speed and energy | **Jetson Orin Nano Super only.** See the note below |

**Why some rows say "not measured here".** A TensorRT engine is compiled for one GPU
architecture and will not load on another, so a throughput number from a desktop GPU says
nothing about a 15 W embedded board. The extension experiments on this page were screened on
an RTX 3090 (torch 2.13.0+cu126, ultralytics 8.4.118, torch-pruning 1.6.0, batch 32) and their
accuracy is quoted as screening evidence. Every speed, power and memory figure comes from the
board. Where an experiment's headline question is about speed, this page says so and leaves the
answer open rather than filling it with a number from the wrong machine.

> **What "plume" means here.** A plume is the visible smoke or flame region the detector has
> to find. Accuracy is reported separately for **small plumes** (under 1% of the frame) and
> **tiny plumes** (under 0.1%, roughly 20x20 pixels), which is distant smoke, and what early
> detection actually depends on.

![What small and tiny plume mean](../../results/figures/plume_definition.png)

## The four decisions

Pruning looks like dozens of competing techniques. It is one problem with four independent
choices, and almost every named method is a particular combination of them. Keeping them
separate is what makes the results below comparable.

**Granularity: in what pattern do you delete?** Individual weights anywhere, or whole channels,
or something in between. The more freely you choose, the more you can delete before accuracy
suffers, because you can always find the genuinely useless weights. But scattered zeros do not
make a matrix smaller, they make a matrix with holes, and a GPU multiplies it at exactly the
same speed. Regular patterns delete less and actually run faster.

**Criterion: which parts do you delete?** The principle is that the less important what you
remove, the less the network notices. Every criterion is a different theory of "important":
the size of the weights (`||W||_1` or `||W||_2`), the batch-norm scale the network already
learned, how far the loss would move if the channel vanished (Taylor, Hessian), or how
redundant a channel is with its neighbours (FPGM).

**Ratio: how much comes out of each layer?** The same average can be spread evenly across
layers (uniform), decided by one threshold across the whole network (global), or allocated by
measuring what each layer can survive (sensitivity driven).

**Fine-tuning: how do you repair the damage?** Cutting always hurts; retraining is how the
surviving channels take up the slack. Removing everything at once and then retraining is
one-shot; alternating small cuts with recovery is iterative.

## Coverage

| Axis | Tested | Still open |
|---|---|---|
| Granularity | Channel, **fine-grained**, **2:4** | vector, kernel |
| Criterion | L2, **L1, BN scale, Taylor, Hessian, FPGM, LAMP, random control** | regression-based (He et al.), APoZ |
| Ratio | Global, **uniform, sensitivity driven** | automated search (AMC, NetAdapt) |
| Fine-tune | One-shot, iterative, **equal post-cut budget** | LR rewinding, teacher-supervised |

Bold is new on this page. Two of the new granularities produce an accuracy answer here and a
speed answer only on the board, and are marked as such where they appear.

## Result 1. The collapse was mostly a bad default

This page used to report that cutting 5% of channels costs 88% of the accuracy, and treated
that as a fact about the detector. It is mostly a fact about **L2 magnitude**, the one
criterion that had been tried.

Same 5% cut, same code path, no retraining, only the importance criterion changed:

![The importance criterion decides whether pruning is survivable](../../results/figures/xp06e2_criteria.png)

**L1 keeps 99% of the accuracy where L2 keeps 12%.** In code that is
`GroupMagnitudeImportance(p=2)` against `p=1`. L2 squares the weights before summing, so one
large weight can carry a channel whose others are near zero; L1 does not, and on this detector
that difference is the whole result.

The **random control** is what makes this a result rather than a leaderboard. Random pruning
collapses to 0.20 at a 2% cut, far below the good criteria, so the winners are genuinely doing
work rather than benefiting from any cut plus retraining.

**Two criteria did worse than random, and one of them is a warning about prerequisites.**
BN scale scores 0.0034 at a 2% cut. It is not that the method is bad: Network Slimming assumes
the network was trained with an L1 penalty on the batch-norm scales so they spread apart and
mean something. These are the D-Fire authors' weights, trained with no such term, so the scales
carry no signal and the criterion selects almost arbitrarily. **A criterion with an unmet
training-time prerequisite is not a criterion that failed, it is one that was never applicable**,
and reporting it as the former would be wrong.

## Result 2. Capacity was never the problem, structure was

The strongest evidence on this page comes from removing capacity the *other* way. Fine-grained
pruning deletes individual weights anywhere, with no structural constraint, so it removes
capacity without removing shape.

![The same network survives losing half its weights and dies losing 5% of its channels](../../results/figures/xp06e5_granularity.png)

**Ninety percent of every weight in the network can be deleted and the model still scores
higher than one with 5% of its channels removed.** 0.1447 against 0.0956, neither retrained.
Half the weights can go for a 2% accuracy cost.

That settles a question this page could not previously answer. The detector is not short of
capacity, and pruning is not futile here. **What it cannot absorb is having whole channels
taken away**, because a channel is a unit the surrounding architecture depends on, and every
residual add and concatenation downstream is shaped by it.

**No speed number appears for fine-grained pruning, on any machine, deliberately.** Irregular
zeros sit inside a full-size tensor with no matching kernels on this hardware, so the model is
not smaller in memory and not faster in time. It is an accuracy ceiling experiment and quoting
a speed figure for it would imply a deployment claim that does not exist.

## Result 3. Where you cut matters, but less than what you cut

Pruning one layer at a time, with everything else left alone, maps where the damage is cheap
and where it is fatal. This runs on the **validation** split, never test, because its output
selects a configuration and test has to stay clean.

![Every layer pruned on its own](../../results/figures/xp06e1_sensitivity.png)

Fragility tracks depth almost perfectly, and the useful part is the trade rather than the
ranking: **the layers that break first are the layers that free the least.**

| layer | accuracy kept at a 50% cut | parameters freed |
|---|---:|---:|
| `model.2.cv1.conv` | **2.9%** | 0.10% |
| `model.0.conv` | **15.8%** | 0.16% |
| `model.1.conv` | 30.0% | 0.16% |
| … | | |
| `model.6.m.0.cv1.conv` | 99.6% | 1.17% |
| `model.21.conv` | **99.1%** | **5.13%** |

Halving the first convolution costs 84% of the accuracy to free 0.16% of the model. Halving
`model.21.conv` costs 0.9% to free 5.13%: fifty times the saving for a hundredth of the damage.
Stages 0 to 4 keep 58.6% of accuracy on average under a 50% cut; stages 6 to 23 keep 94.3%.

A single global threshold ranks channels by weight size and has no way to see any of this, so
part of its budget lands where the trade is catastrophic. That is the mechanism behind the
original collapse.

Acting on the map helps, and helps less than you would hope. All three allocations below are
matched to the same measured parameter reduction by search, so the only thing that varies is
where the cut lands:

![Where the cut lands](../../results/figures/xp06e6_allocation.png)

**Sensitivity-driven allocation wins by 0.4 accuracy points. Choosing a better criterion was
worth 4.5.** The map is correct and acting on it is second-order. It does buy one thing worth
having: correct silence on empty frames returns to the unpruned 97.4%, undoing the extra false
alarms that pruning otherwise introduces.

## Result 4. Arithmetic removed is still not speed gained

This has not changed and is the most portable lesson here. On the board, removing **88.9% of
the multiply-adds bought 1.7x the throughput**, not the roughly 9x the arithmetic implies.
Pruned layers land on awkward widths, 47 channels instead of 64, and GPU kernels are written
for regular tile sizes, so a layer with 27% fewer channels often takes exactly as long.

The sharpest version of it: XP6's iterative model kept **more** parameters than its one-shot
model (4.51 M against 4.24 M) and ran **18% faster** (450 against 381 img/s). How you prune
changed speed more than how much you pruned.

**Parameter count and MAC count are structural facts, never performance claims.** Every speed
verdict on this page comes from a measured engine on the board.

## The verdict

**Pruning still loses.** The best model on this page is 44% smaller and gives up 2.2 accuracy
points, and simply running the unpruned network at 512 px remains the better deployment.

What changed is the size of the loss and the reason for it. The deficit went from 4.8 points to
**2.2**, and the remaining question is speed, which this machine cannot answer.

| | params | mAP50 | small plumes | tiny plumes | correctly silent |
|---|---:|---:|---:|---:|---:|
| **unpruned at 512 px** | 7.03 M | **0.7764** | **0.6038** | **0.1380** | 0.9736 |
| best pruned (LAMP, 12 epochs) | **3.91 M** | 0.7543 | 0.5783 | 0.1294 | 0.9771 |
| as originally published (L2) | 4.24 M | 0.7298 | 0.5525 | 0.0963 | - |

The honest summary is that **this detector can be pruned much harder than this page previously
claimed, and it still should not be**, because the cheap knob (feed it smaller pictures) is
still ahead. That has been the running result of the whole study.

## Bugs that produced wrong numbers

**The library swapped the model.** Given `yolov5s.pt`, the Ultralytics API recognised the
filename and silently downloaded its own COCO-trained 80-class model instead. It ran, it drew
plausible boxes, it measured the wrong network. Every load now asserts
`model.names == ['smoke', 'fire']`.

**The training loop destroyed the model.** A first recovery loop took the *unpruned* model from
0.818 to 0.008 in one epoch and two published numbers had to be withdrawn. Three causes, each
sufficient: no gradient accumulation to YOLOv5's nominal batch of 64, an unscaled
classification gain 40x too heavy, and no EMA. The standing rule is that no recovery number is
believed until retraining the unpruned model returns it to roughly where it started.

**The sparse models were not sparse.** The masked granularities came back 0% sparse after
training. `lib/finetune.py` ends by loading EMA weights, the EMA never runs a forward pass, so
the hooks enforcing the mask never fired on it and it averaged the raw non-zero parameters. The
mask is now re-imposed on the EMA and the sparsity is verified rather than assumed.

**Scoring a model broke training it.** The evaluator halves the model in place, so measuring
damage before recovery handed the optimizer FP16 parameters and torch refused the gradients.
Only the experiments that measure damage *before* retraining ever hit it. Evaluation now runs
on a copy.

## Limitations

- **Speed is unresolved for everything new on this page.** The extension was screened on a
  desktop GPU. The regularity experiment (`round_to`) and 2:4 sparsity both have throughput as
  their headline question, and both hand back checkpoints and ONNX rather than an answer.
- **Fixed channel ratio is not a fixed operating point.** At a 25% ratio, LAMP lands at 3.91 M
  parameters while cutting 20.1% of the arithmetic, and FPGM lands at 4.55 M while cutting
  33.5%. The criterion table compares equal *ratios*, not equal size or equal compute.
- **The allocation arms are matched on parameters, not on arithmetic.** Sensitivity-driven
  pruning cuts 33.6% of MACs where global cuts 38.7% at the same parameter count, so it may be
  slower despite being the same size.
- **Recovery is 12 epochs.** It is what the board allows overnight, and longer training would
  likely recover more.
- **Single dataset, single architecture.** Everything here is YOLOv5s on D-Fire. The claim that
  early layers are fragile is a claim about this network.
- **The sensitivity map is single-layer.** It measures each layer alone; the interactions when
  many are cut together are not additive, and the allocation experiment is what actually tests
  them.
- Two criteria from the lecture are untested: regression-based channel selection (He et al.
  ICCV 2017), and APoZ, which counts exact zero activations and does not transfer to this
  network because YOLOv5 uses SiLU rather than ReLU and almost never emits an exact zero.

## Reproduce

```bash
# the original arms
python experiments/xp06_pruning/run.py --stage damage --ratios 0.02 0.05 0.10 0.25 0.50 0.70
python experiments/xp06_pruning/recover_and_deploy.py --ratio 0.25 --epochs 12

# the extension
python experiments/xp06_pruning/e1_sensitivity.py
python experiments/xp06_pruning/e2_criteria.py --stage damage
python experiments/xp06_pruning/e2_criteria.py --stage recover --criteria lamp l1 fpgm random
python experiments/xp06_pruning/e5_finegrained.py --sparsity 0.90
python experiments/xp06_pruning/e6_allocation.py --plan
python experiments/xp06_pruning/e6_allocation.py --arm sensitivity
python experiments/xp06_pruning/e7_fair_rerun.py --mode iterative --post-epochs 12
python experiments/xp06_pruning/e3_regularity.py --round-to 32
python experiments/xp06_pruning/e4_sparsity24.py

python analysis/make_figures.py     # every figure, from the committed JSON
python analysis/xp06_tables.py      # every table on this page, from the same JSON
```

An interactive walkthrough of the techniques and how they differ is in
[`course/`](course/pruning-course.html).
