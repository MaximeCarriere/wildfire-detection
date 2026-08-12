# XP6 — Pruning: cutting channels out of the model

**Question:** pruning removes whole channels from the network, making it genuinely smaller.
Does that beat the far simpler option of just feeding it a smaller image?

**Outcome so far:** two negative results before the interesting part. Pruning **destroys
this detector almost immediately** without retraining, and even when it does shrink the
model, **the speed-up is nowhere near the arithmetic saved.**

![Pruning: the damage is immediate, the speed-up is not](../../results/figures/xp06_pruning.png)

> **What "plume" means here.** A plume is the visible smoke or flame region the detector has
> to find. Accuracy is reported separately for **small plumes** (under 1% of the frame) and
> **tiny plumes** (under 0.1%, roughly 20x20 pixels) — distant smoke, which is what early
> detection actually depends on.

![What small and tiny plume mean](../../results/figures/plume_definition.png)

## Result 1 — the damage is immediate

Pruning with no retraining at all, full 4,306-image test set, 512 px:

| channels cut | params | arithmetic cut | mAP50 | small plumes | tiny plumes |
|---:|---:|---:|---:|---:|---:|
| — (baseline) | 7.03 M | — | **0.7775** | 0.6062 | 0.1386 |
| 2% | 6.84 M | 4.1% | 0.6872 | 0.4441 | 0.0650 |
| 5% | 6.53 M | 10.4% | **0.0956** | 0.0327 | 0.0036 |
| 10% | 5.97 M | 19.8% | 0.0052 | 0.0000 | 0.0000 |
| 25% | 4.24 M | 43.1% | 0.0000 | 0.0000 | 0.0000 |
| 50% | 1.86 M | 73.0% | 0.0000 | 0.0000 | 0.0000 |
| 70% | 0.62 M | 88.9% | 0.0000 | 0.0000 | 0.0000 |

Removing **2%** of channels already costs 9 accuracy points. By 5% the model has lost 88% of
its accuracy; by 10% it is dead.

The confidence head fails first: the model's maximum objectness score falls from 0.72 to
0.0045 by 5% pruning — it still produces boxes internally, but none survive any sensible
threshold. That is why the table shows exact zeros rather than a gentle slide.

**So recovery training is not a refinement here, it is the experiment.** An
"accuracy vs sparsity" curve without retraining, which is how pruning is often illustrated,
would be a curve of zeros on this model.

## Result 2 — arithmetic removed ≠ speed gained

The right-hand panel is the one worth arguing about. Removing **88.9% of the multiply-adds
buys 1.7× the throughput**, not the ~9× the arithmetic implies.

Pruned channel counts are irregular — 47 channels instead of 64 — and GPU kernels are
tuned for regular tile sizes, so a layer with 27% fewer channels frequently takes the same
time as before. This is the finding PLAN.md scheduled as a separate experiment ("FLOPs lie,
FPS doesn't"); it fell out of XP6 for free.

It also means **parameter count is the wrong axis to judge pruning on.** The 70%-pruned
model is 11× smaller and 1.7× faster.

## In progress

Recovery fine-tuning at 25% and 50% pruning, then export to TensorRT and measurement against
the frontier. Recovery is ~20 min/epoch on this board (measured), so ~4 h per level.

The verdict this experiment exists to deliver — **does a pruned, recovered, deployed model
beat YOLOv5s at 512 px (0.7776 mAP50, 474 img/s, 52 J/1000 frames)?** — needs those runs.

## Limitations

- **Recovery is 10–12 epochs, not 50.** Longer training would likely recover more; this is
  the budget the board allows in a night, and it is stated rather than implied.
- **No teacher-supervised recovery arm.** PLAN.md asks for one, but it needs XP3's
  distillation machinery, and XP1 found the teacher only 1.4 mAP points ahead of the
  student — expensive and unpromising. Not run, not hidden.
- **One pruning method**, global magnitude-based structured pruning. Different importance
  criteria (Taylor, Hessian) may cut differently; untested.
- Pruning ratios are *channel* targets. The arithmetic actually removed is measured and
  reported separately because the two diverge substantially.

## Reproduce

```bash
python experiments/xp06_pruning/run.py --stage damage --ratios 0.02 0.05 0.10 0.25 0.50 0.70
python experiments/xp06_pruning/recover_and_deploy.py --ratio 0.25 --epochs 12
```
