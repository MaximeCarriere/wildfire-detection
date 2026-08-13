# XP1 — Baselines: what a good fire detector costs today

**Question:** what accuracy is achievable on this dataset, and what does the bigger model
buy over the smaller one?

**Outcome:** the bigger model buys almost nothing — which undermines the project's plan to
use it as a teacher for model distillation.

## The models

**We did not train these.** They are the D-Fire dataset authors' own published detectors,
used exactly as released.

| | YOLOv5s | YOLOv5l |
|---|---|---|
| Architecture | YOLOv5 small | YOLOv5 large |
| Parameters | 7.03 M | 46.14 M |
| File size | 14.4 MB | 92.8 MB |
| Input resolution | 640 × 640 | 640 × 640 |
| Classes | `0 = smoke`, `1 = fire` | `0 = smoke`, `1 = fire` |
| Trained on | D-Fire official train split (17,221 images) | same |

- **Who made them:** Pedro Vinícius Almeida Borges de Venâncio and colleagues — the same
  group that built the D-Fire dataset (Gaia, solutions on demand).
- **Where to get them:** [github.com/pedbrgs/Fire-Detection](https://github.com/pedbrgs/Fire-Detection).
  The weights sit behind OneDrive links in that repo's README and need a browser to
  download; scripted requests get a 403.
- **Licence:** their code is MIT; the YOLOv5 architecture itself is Ultralytics AGPL-3.0, so
  the AGPL note in the top-level README applies to any derived weights here too.
- **Training recipe:** not published — epochs, augmentation and training resolution are all
  unknown. That is a real limitation of using them, noted at the bottom of this page.

**Why use someone else's weights at all?** They were trained on D-Fire's *official* training
split, and XP0 deliberately preserved that split rather than reshuffling. So our 4,306-image
test set is provably outside their training data, and they can be scored honestly on day
one — no GPU rental, no waiting. It also means the numbers below are comparable to what the
dataset's own authors report.

![A 6.6x bigger model buys almost nothing](../../results/figures/xp01_baselines.png)

## What was measured

Jetson Orin Nano Super · PyTorch FP16 · 640×640 · full 4,306-image test set.

| | YOLOv5s | YOLOv5l |
|---|---:|---:|
| **Accuracy (mAP50)** | 0.7708 | **0.7847** |
| fire / smoke | 0.7206 / 0.8210 | 0.7211 / 0.8484 |
| tiny plumes | 0.1654 | 0.1974 |
| Correctly silent on empty frames | 96.9% | **97.8%** |
| Speed | 43.8 fps | 25.1 fps |
| Energy per 1000 frames | 232 J | 761 J |

> **What "plume" means here.** A plume is the visible smoke or flame region the detector has
> to find. Accuracy is reported separately for **small plumes** (under 1% of the frame) and
> **tiny plumes** (under 0.1%, roughly 20x20 pixels) — distant smoke, which is what early
> detection actually depends on.

![What small and tiny plume mean](../../results/figures/plume_definition.png)

## What this means

**6.6× the parameters for 1.4 accuracy points**, at 3.3× the energy. Distillation works by
transferring what a large model knows that a small one doesn't; here that surplus is thin,
so the planned distillation phase starts from a much weaker premise than assumed. (Caveat:
a small accuracy gap makes a large distillation gain unlikely, not impossible — soft targets
carry information beyond accuracy alone.)

**Fire is the harder class, not smoke** (0.72 vs 0.82). The project plan asserted the
opposite as a core assumption. It is now an open question.

**What the big model actually buys is restraint** — a 30% lower false-alarm rate on empty
landscape. For a camera watching nothing all day that may be worth more than the accuracy
points, and it is invisible in every metric the plan originally specified.

## What nearly went wrong

The inference library **silently substituted a different model.** Given the weights file, it
recognised the filename as a standard model name, downloaded a generic pre-trained model
from its own servers, and used that instead — 80 everyday object classes, none of them fire.
It logged a cheerful tip rather than an error. Caught only by printing the class names; the
loader now refuses to proceed unless the classes are exactly `smoke, fire`.

## Limitations

- These came from the authors with their training recipe, epochs and resolution unknown, so
  the small-vs-large comparison mixes architecture with training differences.
- No smaller "just use a tiny model" control yet — that needs one training run.

## Next

XP2 asks whether simply shrinking the input beats any of this.
