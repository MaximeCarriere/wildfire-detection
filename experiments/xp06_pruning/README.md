# XP6. Pruning: cutting channels out of the model

**Question:** pruning removes whole channels from a network, making it genuinely smaller.
Does that beat the far simpler option of feeding the same network a smaller image?

**Outcome:** no. Pruning loses on accuracy, and mostly on speed too. It also destroys this
detector almost immediately if you do not retrain afterwards.

![Pruning: the damage is immediate, the speed-up is not](../../results/figures/xp06_pruning.png)

> **What "plume" means here.** A plume is the visible smoke or flame region the detector has
> to find. Accuracy is reported separately for **small plumes** (under 1% of the frame) and
> **tiny plumes** (under 0.1%, roughly 20x20 pixels), which is distant smoke, and what
> early detection actually depends on.

![What small and tiny plume mean](../../results/figures/plume_definition.png)

## Result 1. The damage is immediate

Pruning with no retraining at all. Full 4,306-image test set, 512 px:

| channels cut | params | arithmetic cut | mAP50 | small plumes | tiny plumes |
|---:|---:|---:|---:|---:|---:|
| none (baseline) | 7.03 M | none | **0.7775** | 0.6062 | 0.1386 |
| 2% | 6.84 M | 4.1% | 0.6872 | 0.4441 | 0.0650 |
| 5% | 6.53 M | 10.4% | **0.0956** | 0.0327 | 0.0036 |
| 10% | 5.97 M | 19.8% | 0.0052 | 0.0000 | 0.0000 |
| 25% | 4.24 M | 43.1% | 0.0000 | 0.0000 | 0.0000 |
| 50% | 1.86 M | 73.0% | 0.0000 | 0.0000 | 0.0000 |
| 70% | 0.62 M | 88.9% | 0.0000 | 0.0000 | 0.0000 |

Removing **2%** of channels already costs 9 accuracy points. By 5% the model has lost 88% of
its accuracy, and by 10% it is dead.

The confidence head fails first. Maximum objectness falls from 0.72 to 0.0045 by 5% pruning,
so the model still produces boxes internally but none survive any sensible threshold. That
is why the table shows exact zeros rather than a gentle slide.

**Recovery training is therefore not a refinement here, it is the experiment.** An
"accuracy versus sparsity" curve without retraining, which is how pruning is often
illustrated, would be a curve of zeros on this model.

## Result 2. Arithmetic removed is not speed gained

Removing **88.9% of the multiply-adds buys 1.7× the throughput**, not the roughly 9× the
arithmetic implies.

Pruned channel counts are irregular, 47 channels instead of 64, and GPU kernels are tuned
for regular tile sizes, so a layer with 27% fewer channels frequently takes the same time as
before. Parameter count and arithmetic are both poor predictors of speed on this hardware.

## Result 3. Pruned, recovered and deployed, it still loses

25% of channels removed two different ways, each given 12 epochs of recovery training, each
exported to a TensorRT engine so the speed numbers are comparable:

- **One-shot:** remove 25% in a single operation, then train for 12 epochs.
- **Iterative:** remove about 7%, train 2 epochs, repeat four times, then train 4 more.

| | unpruned | one-shot | iterative |
|---|---:|---:|---:|
| parameters | 7.03 M | 4.24 M | 4.51 M |
| **mAP50** | **0.7776** | 0.7297 | 0.6771 |
| small plumes | **0.6061** | 0.5516 | 0.4386 |
| tiny plumes | **0.1376** | 0.0960 | 0.0653 |
| correctly silent on empty frames | **97.4%** | 95.2% | 92.7% |
| throughput | **473.7 img/s** | 381.1 | 450.2 |
| energy per 1000 frames | 52.1 J | 54.3 | **46.5 J** |

**Neither arm beats simply running the unpruned model at 512 px.** One-shot gives up 4.8
accuracy points and runs 20% slower. Iterative gives up 10 points and is still slower.

Two things in that table matter more than the headline:

**Pruning makes the detector noisier on empty frames.** Correct silence falls from 97.4% to
95.2% and then 92.7%. For a camera that watches nothing almost all the time, nearly tripling
the false-alarm rate is a bigger practical cost than the accuracy points, and it does not
appear in mAP at all.

**The iterative model is faster while being larger.** It keeps 4.51 M parameters against
one-shot's 4.24 M, yet runs 18% faster (450 against 381 img/s) and uses 14% less energy.
Removing channels gradually appears to leave more regular channel counts, which the GPU's
kernels handle better. That is Result 2 seen from the other side, and it suggests *how* you
prune changes speed more than *how much* you prune.

## Limitations

- **The two arms are not perfectly matched.** Both got 12 epochs in total, but the iterative
  model's final architecture only existed for the last 4 of them, while the one-shot model
  trained for all 12 in its final shape. Equal total budget is not equal recovery budget,
  and that plausibly explains part of iterative's accuracy deficit. A fairer rerun would
  give both the same number of epochs *after* the final cut.
- **Recovery is 12 epochs, not 50.** Longer training would likely recover more. This is the
  budget the board allows in a night, and it is stated rather than implied.
- **One pruning method**: global magnitude-based structured pruning. Other importance
  criteria (Taylor, Hessian) may cut differently. Untested.
- **No teacher-supervised recovery.** Recovering under supervision from a larger model is a
  known variant, but the larger model available here is only 1.4 mAP points better
  ([XP1](../xp01_baselines/)), so it is expensive and unpromising.
- Pruning ratios are *channel* targets. The arithmetic actually removed is measured
  separately, because the two diverge substantially.

## Reproduce

```bash
python experiments/xp06_pruning/run.py --stage damage --ratios 0.02 0.05 0.10 0.25 0.50 0.70
python experiments/xp06_pruning/recover_and_deploy.py --ratio 0.25 --epochs 12
python experiments/xp06_pruning/recover_and_deploy.py --ratio 0.25 --epochs 12 \
    --mode iterative --steps 4
```
