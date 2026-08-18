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
| **E4** | granularity | does 2:4 sparsity, the pattern hardware understands, hold up? | ✅ accuracy holds; **the compiler refuses the sparse kernels** |
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

**Every rule is a different theory of what makes a channel worth keeping.** Each scores every
channel, and the lowest scores are deleted. $w$ is a channel's weights, $g$ its gradients,
$\gamma$ its batch-norm scale.

| rule | score | why you would use it | why you might not |
|---|---|---|---|
| **L1** | $\sum_i \lvert w_i \rvert$ | free, needs no data, treats every weight equally | blind to layer scale, so it over-cuts layers whose weights are globally small |
| **L2** | $\sqrt{\sum_i w_i^2}$ | the usual default, equally free | squaring lets **one** large weight rescue an otherwise-dead channel. On this model that was catastrophic |
| **LAMP** | $w_i^2 \big/ \sum_{j\ge i} w_j^2$ | layer-adaptive for free: layers compete on their own terms, so narrow critical layers survive | still pure magnitude, still ignores the data |
| **FPGM** | $\sum_j \lVert W_i - W_j \rVert_2$ | catches **redundancy** magnitude cannot: two large but near-identical channels | costs a pairwise distance per layer, the slowest of the cheap rules |
| **Taylor** | $\lvert g_i w_i \rvert$ | actually estimates the loss change, and is the only family that looks at the data | needs a backward pass over calibration data, and the loss is dominated by common cases, so rare ones get under-weighted |
| **Hessian** | $\tfrac{1}{2} h_{ii} w_i^2$ | the most principled on paper (Optimal Brain Damage) | assumes training converged and that weights are independent, both shaky for a fine-tuned detector. Worst real rule here |
| **BN scale** | $\lvert \gamma_c \rvert$ | free, reuses a number the network already learned | **requires** training with a penalty that spreads the scales apart. Without it the signal does not exist |
| **random** | $\sim U(0,1)$ | the control that proves the others are doing work | no signal, by construction |

The short version of the trade-off: **magnitude rules are free but data-blind, gradient rules see
the data but cost a backward pass and over-weight the common case, and FPGM is the only one
asking a different question entirely.** On this detector the free layer-adaptive rule beat every
expensive one.

![The importance criterion decides whether pruning is survivable](../../results/figures/xp06e2_criteria.png)

**At a 5% cut, L1 keeps 99% of the accuracy where L2 keeps 12%.** In code the difference is
`p=2` against `p=1`.

### Where each rule actually cut

The cut is one **global** threshold across the whole network, not a per-layer quota, so each rule
decides for itself where the damage lands. That turns out to explain the ranking.

| rule | early (0-4) | mid (5-9) | deep (10-23) | mAP50 |
|---|---:|---:|---:|---:|
| **LAMP** | **0.1%** | 28.8% | 22.9% | **0.7543** |
| L1 | 18.3% | 16.1% | 32.6% | 0.7531 |
| Taylor | 22.1% | 11.5% | 35.5% | 0.7460 |
| FPGM | 12.8% | 8.5% | 40.4% | 0.7438 |
| BN scale | 33.2% | 18.5% | 29.0% | 0.7153 |
| Hessian | 22.3% | 33.2% | 15.8% | 0.7111 |
| **random** | **48.5%** | 18.6% | 25.6% | **0.7093** |

**LAMP barely touches the early layers and wins; random hammers them and loses.** Across the
seven rules, the correlation between how much of the early network they cut and their final
accuracy is **r = -0.79**.

That is E1's prediction confirmed by a completely separate experiment. E1 measured layers one at
a time and concluded the early ones must be protected; E2 never used that map, yet the rules that
happened to protect them are the rules that won. Two independent routes to the same mechanism,
which is much stronger than either alone. (Seven points, so treat the coefficient as direction
rather than a precise effect size.)

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

## E4. The one pattern the hardware understands

**2:4 sparsity is a middle ground between deleting whole channels and deleting scattered
weights**, and the only one current hardware can actually exploit. The rule: of every four
neighbouring weights, exactly two must be zero. Half the numbers go, but *where* they go is
constrained.

That constraint is the point. Ampere GPUs have circuitry that skips those zeros for up to twice
the throughput, and the Orin's GPU is Ampere class. Scattered zeros (E5 below) get no such
support; whole channels need none.

Like E5 and unlike channel pruning, the weights here are **masked, not removed**: the model
still stores 7.03 M numbers, of which 3.53 M are non-zero. On this hardware it is only faster if
the compiler selects sparse kernels, which is exactly the open question.

![What 2:4 sparsity is, and what constraining the pattern costs](../../results/figures/xp06e4_sparsity24.png)

**Read the two grids on the left.** Both delete exactly half the weights. The top one may put
its zeros anywhere. The bottom one must place exactly two in every group of four, marked by the
black lines. Same quantity, different freedom, and the bars on the right show what that freedom
was worth.

**Half the weights removed, three ways.** Every row below deletes the same number of weights
(3.49 M of 6.99 M) from the same 57 layers, chosen by magnitude. Only the rule about *where* the
zeros may sit changes.

| 50% of the weights deleted | non-zero | mAP50 | tiny plumes |
|---|---:|---:|---:|
| **none (unpruned)** | 7.03 M | **0.7764** | 0.1380 |
| free choice of where ([E5](#e5-whole-channels-versus-individual-weights)) | 3.53 M | 0.7622 | 0.1562 |
| **forced into a 2:4 pattern** | 3.53 M | **0.0000** | 0.0000 |
| forced into a 2:4 pattern, then 12 epochs | 3.53 M | 0.7527 | 0.1249 |

Same quantity deleted, same layers, same criterion. Choosing freely costs **1.4 points**.
Following the pattern costs **everything**. **It is not how much you remove, it is whether the
removal has to follow a shape.** Retraining then wins it back to 97% of unpruned, matching what
NVIDIA reports for detection.

> **"Free" here means free to choose *where*, never free of charge.** Free-form pruning is
> unconstrained in **placement**, and it still costs: 1.4 accuracy points at 50% before
> retraining, and a file that does not shrink by a single byte. It buys no speed either, for a
> reason worth stating plainly: the kernels still multiply every position in the grid, and
> multiplying by zero costs exactly what multiplying by anything else costs (measured below).
> What it is free of is the pattern constraint. This experiment exists to price that constraint,
> and the price turns out to be everything.

### Why the patterned half collapses

**Free-form pruning is adaptive; 2:4 cannot be.** One global threshold (0.0069) lets the network
decide where the budget lands, and it lands very unevenly: measured per-layer sparsity runs from
**8.4% to 87.6%**. Layers full of small weights are stripped bare, layers full of large ones are
barely touched. 2:4 takes exactly 50% from every one of the 57 layers, because the quota is
enforced inside every group of four and a group cannot know it sits in an important layer.

| what actually got deleted | free choice | 2:4 pattern |
|---|---:|---:|
| largest single weight removed | 0.0069 | **0.4365** |
| share of the network's weight energy removed | 2.1% | **12.6%** |
| weights removed from the network's strongest 10% | **0** | **64,990** |
| per-layer sparsity | 8.4% to 87.6% | 50.0% everywhere |

At identical sparsity the patterned cut destroys **six times the weight energy** and deletes
65,000 weights from the strongest tenth of the network. Free-form deletes none of them, ever, by
definition of the threshold.

It also lands hardest exactly where this network can least afford it. The first convolution holds
the largest weights in the model; free-form removes 9.0% of it, 2:4 removes 50%, including one
weight of magnitude 0.4365. [E1](#e1-which-layers-can-be-cut) already measured that bill:
halving the first convolution alone costs **84%** of the accuracy.

**That is also why the score is exactly 0.0000 rather than merely low.** Maximum objectness over
a 48-image batch: unpruned **0.911**, free choice **0.895**, 2:4 **0.00022**. The patterned model
still draws boxes internally, but at confidences roughly 4,000x too low to clear any sensible
threshold, so mAP falls off a cliff instead of sliding. Twelve epochs of retraining rescale the
head and the accuracy comes back, which tells you the capacity was never lost. The calibration
was.

### Does it actually run faster?

Accuracy was screened on a desktop GPU. Speed cannot be, because a TensorRT engine is compiled
per GPU architecture, so this is measured on the Jetson. Four engines, same network, same shape,
same arithmetic: only which weights are zero, and whether the compiler was allowed to exploit
them. Each arm is timed twice, once down the list and once up it, because four engines timed back
to back would alias any drift in clocks or die temperature onto arm identity.

![2:4 sparsity on the board](../../results/figures/xp06e4b_sparsity_speed.png)

**The answer is in the build log, not the bars.**

```
(Sparsity) Found 39 layer(s) eligible to use sparse tactics
(Sparsity) Chose 0 layer(s) using sparse tactics
```

TensorRT identified 39 layers it could run with sparse kernels, timed them against the dense
ones, and **picked dense every single time**. The sparse engine is a dense engine. The hardware
path exists on this board and the compiler declined to take it, so the 2x that Ampere's sparse
tensor cores promise is simply not on offer for this network at this size.

**Then what is the 3.5%?** Not sparsity. The giveaway is the second row: free-form 50% also comes
out 1.6% faster than dense, and free-form 50% runs *identical arithmetic through identical
kernels*. A difference that appears where no difference can physically exist is the noise floor
between independently built engines, and TensorRT autotunes each build separately. Reversing the
measurement order reproduced the same ranking, so it is a property of the engine rather than of
when it was measured, but "reproducible" and "caused by the sparsity" are different claims and
only the first one is supported.

Energy tracks the same tiny spread: 52.4 J per 1000 frames dense against 49.3 to 51.1 for the
2:4 engines. Nothing here changes the deployment answer.

**So the verdict for the whole of XP6 stands.** The unpruned model at 512 px remains the line to
beat, and 2:4 was the last candidate with a hardware story behind it.

## E5. Whole channels versus individual weights

E1, E2, E6 and E7 all removed **whole channels**. E4 was the first to remove **individual
weights**, but under a rule about where the zeros may sit. This removes individual weights with
no rule at all, and the channel-versus-weight distinction is the most important one on the page.

A convolution's weights are a grid: `[output channels, input channels, height, width]`.

- **Channel pruning deletes an entire slice of that grid.** One whole detector disappears, every
  layer downstream loses its matching input, and the network genuinely becomes narrower. 7.03 M
  parameters really do become 4.2 M. It gets smaller on disk, in memory, and in arithmetic.
- **Weight pruning sets scattered individual numbers to zero.** Every channel is still there,
  still computed, still producing its feature map. The grid keeps its exact shape and only has
  holes in it.

**This is why the two columns below are not comparable as sizes.** A 90% weight-pruned model
still stores and loads 7.03 M numbers; 0.74 M of them are non-zero. It is *not* a 0.74 M model.

> **Why zeroing 90% of the weights leaves the file the same size.** Setting a weight to zero does
> not delete it. The weights sit in a dense rectangular array, and a zero occupies exactly the
> same two bytes as any other number in it. Shrinking the file would need a **sparse storage
> format**, which keeps only the non-zeros plus indices recording where each one belongs. Such
> formats exist and are decades old, but nothing in this pipeline uses one: PyTorch saves dense
> tensors, ONNX carries dense initialisers, and TensorRT compiles kernels that multiply every
> position in the grid. So the file is byte-for-byte the same size, and the GPU performs the same
> number of multiplications as before, most of them now by zero.
>
> Indices are not free either. At 50% sparsity you would store half the values plus a location for
> each survivor, and a naive index costs more than the value it points to. **2:4 is the exception
> that proves the rule**: because exactly two of every four survive, their positions fit in a
> couple of bits per group, and Ampere has circuitry that reads that encoding directly. That is
> the whole reason [E4](#e4-the-one-pattern-the-hardware-understands) is a separate experiment.

So this experiment cannot produce a deployable model. It exists to answer one question that
channel pruning cannot: **is this detector short of capacity, or short of structure?**

![The same network survives losing half its weights and dies losing 5% of its channels](../../results/figures/xp06e5_granularity.png)

| what was removed | model size | non-zero weights | mAP50 |
|---|---|---:|---:|
| **nothing** | 7.03 M | 7.03 M | **0.7764** |
| 25% of individual weights | still 7.03 M | 5.28 M | 0.7552 |
| **90% of individual weights** | **still 7.03 M** | **0.74 M** | **0.7425** |
| 5% of *channels* | genuinely 6.53 M | 6.53 M | 0.0956 |

**Nine tenths of the numbers in this network can be set to zero and it still works.** Meanwhile
deleting 5% of its channels destroys it.

Same network, same amount of capacity taken away, opposite outcome. **The capacity was never the
constraint; the structure is.** A channel is a unit everything downstream is built around, so
removing one forces a change on every layer that follows. A weight is not.

That is also the reason pruning keeps losing here. The redundancy is real and provable, and no
kernel on this board can convert it into a single extra frame per second.

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

- **E4 now has its speed number; E3 still does not.** Everything else new on this page is
  accuracy only and still needs the board.
- **The 3.5% spread between the four E4 engines is unexplained.** It is reproducible under
  reversed measurement order, so it belongs to the engine, and it cannot be sparsity because the
  compiler selected no sparse kernels and because free-form 50% shows the same effect while
  running identical arithmetic. The obvious control, rebuilding one ONNX twice to size TensorRT's
  own build-to-build variance, was started and lost when the board dropped off the network. Until
  it runs, "this is autotuner noise" is the best-supported reading rather than a measured one.
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
python experiments/xp06_pruning/e4_sparsity24.py                        # E4 2:4 accuracy
python experiments/xp06_pruning/e4_why.py                               # E4 why the pattern collapses
python experiments/xp06_pruning/e4_speed.py                             # E4 speed, ON THE BOARD
python experiments/xp06_pruning/e4_speed.py --skip-build --reverse      # ... and the order control
python experiments/xp06_pruning/e5_finegrained.py --sparsity 0.90       # E5 single weights
python experiments/xp06_pruning/e6_allocation.py --plan                 # E6, then --arm <name>
python experiments/xp06_pruning/e7_fair_rerun.py --mode iterative --post-epochs 12  # E7

python analysis/make_figures.py     # every figure, from the committed JSON
python analysis/xp06_tables.py      # every table, from the same JSON
```

An interactive walkthrough of the techniques is in [`course/`](course/).
