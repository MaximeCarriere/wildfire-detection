# XP6. Pruning: how much of this detector can you delete?

**Question:** pruning deletes parts of a trained network to make it smaller and faster. Does it
beat the far simpler option of feeding the same network a smaller image?

**Outcome:** no, but not for the reason this page used to give. The detector turns out to need
only **about a tenth of its weights**, and the collapse reported here originally was mostly the
fault of one badly chosen setting.

**The line to beat**, which nothing here beats: YOLOv5s at 512 px, TensorRT FP16,
**0.7776 mAP50 at 474 img/s** on the Jetson Orin Nano Super.

| | |
|---|---|
| Model | YOLOv5s, 7.03 M parameters, `0 = smoke`, `1 = fire` |
| Weights | The D-Fire authors' detectors ([pedbrgs/Fire-Detection](https://github.com/pedbrgs/Fire-Detection)), not ours |
| Data | [D-Fire](https://github.com/gaiasd/DFireDataset), 21,527 images; splits frozen at 15,500 / 1,721 / 4,306 |
| Accuracy | Full 4,306-image test set at 512 px |
| Speed | **Jetson only.** An engine is built for one GPU and will not load on another, so speed measured on a desktop card says nothing about a 15 W board. New results here were screened on an RTX 3090 and carry accuracy only |

> **What "plume" means here.** A plume is the visible smoke or flame the detector has to find.
> Accuracy is reported separately for **small plumes** (under 1% of the frame) and **tiny
> plumes** (under 0.1%, roughly 20x20 pixels), which is distant smoke, and what early detection
> actually depends on.

![What small and tiny plume mean](../../results/figures/plume_definition.png)

## Pruning is four choices, not one technique

**What shape** you delete in (single weights, or whole channels), **which parts** you pick,
**how much** comes out of each layer, and **what retraining** repairs the damage. Almost every
named method is one combination of those four. Each experiment below changes exactly one.

> **What a "channel" is, since most of this page turns on it.** A layer is one processing step;
> a **channel is one of that layer's outputs**, a single 2D map of "how strongly does my pattern
> appear here". The image arrives with 3 channels (red, green, blue), layer 0 turns those into
> 32 learned pattern detectors, and by layer 8 there are 512 of them. So a channel is not a
> colour and not a whole layer, it is one detector inside a layer. **Pruning channels makes
> layers narrower; no layer was ever deleted here.** This network has no fully-connected layers
> at all, so the channel is also what a "neuron" would be.
>
> The vocabulary is specific to **convolutional** networks. The four choices are not: a
> transformer has the same problem with attention heads and feed-forward widths.

## The experiments

| | axis it changes | question | outcome |
|---|---|---|---|
| **E1** | ratio | which layers can be cut at all? | ✅ fragile layers are the ones that free the least |
| **E2** | criterion | which channels to pick, and does it beat random? | ✅ decides everything: 99% kept vs 12% |
| **E3** | shape regularity | does rounding channel counts recover the missing speed? | ❌ **not run**, and needs the board |
| **E4** | granularity | does 2:4 sparsity, the pattern hardware understands, hold up? | ✅ accuracy holds; **speed needs the board** |
| **E5** | granularity | is the collapse a capacity limit or a structural one? | ✅ structural, decisively |
| **E6** | ratio | same cut, spread three ways. Does allocation rescue it? | ✅ helps, but far less than E2 |
| **E7** | retraining | does iterative still lose when both arms train equally? | ✅ yes, it still loses |
| **E8** | criterion | pick channels by reconstructing the layer's output | ❌ **not attempted** |

Each section below is one experiment. The unpruned model is the top row of every table.

## E1. Which layers can be cut

Prune one layer, leave the other 56 alone, measure, put it back, move on. 57 layers x 5 depths
of cut, on the **validation** split, since the result chooses a configuration and test has to
stay clean.

![Every layer pruned on its own](../../results/figures/xp06e1_sensitivity.png)

**How to read it.** On the left, every column is one layer and every row is how hard that single
layer was cut. Dark green means the network barely noticed, pale means it fell over. The black
line separates the early layers from the deeper ones. On the right, each dot is one layer
halved: how much of the model that freed, against how much accuracy survived. **Good cuts are
top-right. Bottom-left means you destroyed the detector and saved nothing.**

Two things jump out, and the second is the useful one.

**Fragility follows depth.** Early layers go pale as soon as you cut them; deeper layers stay
dark green even at 70%. Stages 0 to 4 keep 59% of accuracy under a 50% cut, stages 6 to 23 keep
94%.

**The fragile layers are also the ones with nothing to give.** Every red dot is jammed against
the left edge of the right-hand panel: halving the first convolution frees **0.16%** of the
model and costs **84%** of the accuracy. Halving `model.21.conv` costs **0.9%** and frees
**5.13%**. That is fifty times the saving for a hundredth of the damage.

**Conclusion.** There is no reason to ever cut the early layers of this detector. They are the
most expensive place to take damage and the least profitable place to save. A single global
threshold ranks channels by weight size and knows none of this, so part of its budget lands
exactly there, which is what produced the original collapse. E6 tests whether acting on this map
is enough to fix it.

## E2. Which channels to pick

This page used to say a 5% cut costs 88% of the accuracy. That is mostly a fact about **L2**,
the one selection rule that had ever been tried.

![The importance criterion decides whether pruning is survivable](../../results/figures/xp06e2_criteria.png)

**At a 5% cut, L1 keeps 99% of the accuracy where L2 keeps 12%.** In code the difference is
`p=2` against `p=1`.

| selection rule (all at a **25% channel cut**, 12 epochs) | params left | mAP50 | small plumes | tiny plumes |
|---|---:|---:|---:|---:|
| **none (unpruned)** | 7.03 M | **0.7764** | 0.6038 | 0.1380 |
| LAMP | **3.91 M** | 0.7543 | 0.5783 | 0.1294 |
| L1 | 4.21 M | 0.7531 | 0.5720 | 0.1294 |
| Taylor | 4.45 M | 0.7460 | 0.5784 | 0.1102 |
| FPGM | 4.55 M | 0.7438 | 0.5701 | 0.1190 |
| **L2, as first published** | 4.24 M | 0.7298 | 0.5525 | 0.0963 |
| BN scale | 4.15 M | 0.7153 | 0.5160 | 0.0739 |
| Hessian | 3.83 M | 0.7111 | 0.5152 | 0.0810 |
| random (the control) | 4.45 M | 0.7093 | 0.4812 | 0.0676 |

**Before retraining these rules range from 0.94 to 0.00; after it, from 0.754 to 0.709.**
Retraining is a great leveller, which is why damage measured without it is a poor guide to a
deployed model.

**BN scale scored near random, and the reason is measurable.** It ranks channels by the scale
batch norm already learned, which works only if training pushed some of those scales toward zero
to mark channels as dead. In these weights **not one of the 9,504 channels has a scale below
0.1**. The signal does not exist, so it selects almost arbitrarily. An unmet prerequisite, not a
failed method.

**Random is in the table on purpose.** Without it you cannot tell whether a rule is clever or
whether any cut plus retraining lands in the same place. On overall accuracy the good rules beat
it by 4.5 points, but **on tiny plumes by nearly double**. The choice barely matters for easy
cases and matters enormously for distant smoke.

## E5. Channels versus individual weights

Deleting individual weights removes the same capacity without removing any structure.

![The same network survives losing half its weights and dies losing 5% of its channels](../../results/figures/xp06e5_granularity.png)

| what was removed | weights left | mAP50 |
|---|---:|---:|
| **nothing** | 7.03 M | **0.7764** |
| 25% of individual weights | 5.28 M | 0.7552 |
| **90% of individual weights** | **0.74 M** | **0.7425** |
| 5% of *channels*, no retraining | 6.53 M | 0.0956 |

**Nine tenths of this network can go.** At 90% sparsity it holds 96% of its accuracy on 0.74 M
weights. The detector was never short of capacity; what it cannot survive is losing whole
**channels**, because everything downstream is shaped around them.

That is also why pruning keeps losing here. The redundancy is real and provable, and no kernel
on this board can turn it into a single extra frame per second.

## E4. The one pattern the hardware understands

**2:4 sparsity** is the version of that idea Ampere GPUs can accelerate: of every four
neighbouring weights, exactly two must be zero. The Orin's GPU is Ampere class.

| | non-zero weights | mAP50 | tiny plumes |
|---|---:|---:|---:|
| **unpruned** | 7.03 M | **0.7764** | 0.1380 |
| 2:4, no retraining | 3.53 M | **0.0000** | 0.0000 |
| 2:4, after 12 epochs | 3.53 M | **0.7527** | 0.1249 |

Retraining wins it all back, to 97% of the unpruned model, matching what NVIDIA reports for
detection. But note the untrained row: free choice at the same 50% sparsity scores 0.7622, and
forcing the identical amount into a fixed pattern scores zero. **It is not how much you remove,
it is whether the removal has to follow a shape.**

Whether the board's compiler actually uses sparse kernels is a separate question with a separate
answer. If it does, this is the most promising candidate in the study: channel-pruning accuracy
at half the weights. See [`HANDOFF_TO_JETSON.md`](HANDOFF_TO_JETSON.md).

## E6. How to spread the cut

Using E1's map, protect the fragile layers and cut hard where there is slack. All three arms are
matched to the same measured size by search, so only the distribution varies.

![Where the cut lands](../../results/figures/xp06e6_allocation.png)

**Sensitivity-driven allocation wins by 0.4 accuracy points. Choosing a better rule (E2) was
worth 4.5.** The map is correct and acting on it is second-order. It does buy one thing: correct
silence on empty frames returns to the unpruned 97.4%, undoing the extra false alarms pruning
otherwise causes.

## E7. One-shot versus iterative, fairly

The original comparison was confounded: both arms got 12 epochs, but the iterative model's final
shape existed for only 4 of them while one-shot trained all 12 in its final shape. Here both get
**12 epochs after their final cut**, and iterative additionally keeps its 8 between-step epochs,
so it gets *more* total training, not less.

![Iterative pruning still loses once both arms train equally](../../results/figures/xp06e7_fair_rerun.png)

| | params | epochs after final cut | total epochs | mAP50 | small plumes | tiny plumes |
|---|---:|---:|---:|---:|---:|---:|
| **none (unpruned)** | 7.03 M | - | - | **0.7764** | 0.6038 | 0.1380 |
| one-shot | 4.24 M | 12 | 12 | **0.7403** | 0.5617 | 0.0923 |
| iterative | 4.53 M | 12 | **20** | 0.7262 | 0.5316 | 0.0812 |

**The confound was real, and fixing it does not change the answer.** The published gap between
the two arms was 5.4 accuracy points (0.7298 against 0.6763); on equal footing it is **1.4**. So
roughly three quarters of iterative's apparent deficit was the shorter training in its final
shape, exactly as the limitation on this page always suspected.

**But iterative still loses**, and it loses while holding every advantage: more total training
(20 epochs against 12) and a larger model (4.53 M against 4.24 M). The textbook expectation is
that gradual pruning preserves more accuracy at the same sparsity. On this detector it does not,
and that is now a clean result rather than a budgeting artefact.

## Speed: removing arithmetic is still not gaining it

![Pruning: the damage is immediate, the speed-up is not](../../results/figures/xp06_pruning.png)

On the board, removing **88.9% of the multiply-adds bought 1.7x the throughput**, not the ~9x
the arithmetic implies. Pruned layers land on awkward widths (47 channels instead of 64) and GPU
kernels are written for regular sizes. Parameter counts and MAC counts are structural facts
here, never performance claims.

## Verdict

**Pruning still loses.** The best model is 44% smaller and gives up 2.2 accuracy points, and
simply running the unpruned network at 512 px is still the better deployment. What changed is
the size of the loss (4.8 points to 2.2) and the reason for it.

The honest summary: **this detector can be pruned far harder than this page used to claim, and
it still should not be**, because the cheap knob (smaller pictures) is still ahead.

## What was not finished

- **E3, the regularity test, never ran.** [`e3_regularity.py`](e3_regularity.py) is ready. It is
  the only direct test of why a larger pruned model ran faster than a smaller one.
- **E8, regression-based selection, was not attempted.** The most implementation-heavy item, and
  deliberately last.
- **Two sparsity levels in E5 have damage numbers only** (50% and 70%), after two runs were lost
  to a GPU out-of-memory error.

## Limitations

- **No speed number exists for anything new here.** It all needs the board.
- **A 25% cut does not mean the same size for every rule**, which is why E2's parameter column is
  not constant. The cut is a *channel* target, and channels are not equal in cost: one in the
  first layer carries about 100 weights, one deep in the network about 2,300. A rule that
  concentrates its cuts in the wide late layers strips far more parameters than one nibbling at
  narrow early ones. LAMP removes 44.4% of the parameters and FPGM 35.3% from the identical
  setting. **Compare rules at similar sizes, not by the label they share**, which is why E6
  matches its arms by measured size.
- **Nothing is matched on arithmetic.** LAMP is the smallest model but cuts only 20.1% of the
  MACs where L1 cuts 38.7%. On hardware where speed tracks neither, that is a third axis nobody
  here controlled for.
- **Damage is a poor guide to a retrained model.** It separates the good tier from the bad tier
  reliably, but not the order within a tier: BN scale has the worst damage of all eight and
  still finishes above Hessian.
- **Recovery is 12 epochs**, what the board allows overnight; **one dataset, one architecture.**
  "Early layers are fragile" is a claim about this network.

## Bugs that produced wrong numbers

**The library swapped the model.** Given `yolov5s.pt`, Ultralytics quietly downloaded its own
80-class COCO model instead. It ran and drew plausible boxes. Every load now asserts the class
names are `['smoke', 'fire']`.

**The training loop destroyed the model.** An early version took the *unpruned* model from 0.818
to 0.008 in one epoch and two published numbers were withdrawn. No recovery number is believed
now until retraining the unpruned model returns it to where it started.

**The sparse models were not sparse.** They came back 0% sparse, because the loop ends by
loading averaged weights and that average never runs a forward pass, so the masks enforcing
sparsity never reached it. Sparsity is now verified after training rather than assumed.

## Reproduce

```bash
python experiments/xp06_pruning/e1_sensitivity.py                       # E1 layer map
python experiments/xp06_pruning/e2_criteria.py --stage damage           # E2 selection rules
python experiments/xp06_pruning/e2_criteria.py --stage recover --criteria lamp l1 fpgm random
python experiments/xp06_pruning/e4_sparsity24.py                        # E4 2:4
python experiments/xp06_pruning/e5_finegrained.py --sparsity 0.90       # E5 single weights
python experiments/xp06_pruning/e6_allocation.py --plan                 # E6, then --arm <name>
python experiments/xp06_pruning/e7_fair_rerun.py --mode iterative --post-epochs 12  # E7

python analysis/make_figures.py     # every figure, from the committed JSON
python analysis/xp06_tables.py      # every table, from the same JSON
```

An interactive walkthrough of the techniques is in [`course/`](course/).
