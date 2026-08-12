# Wildfire detection on a $249 edge computer

**How small and how fast can a fire/smoke detector get before it stops being useful?** This
repo measures that on one NVIDIA Jetson Orin Nano Super — every technique's real cost and
real gain, measured on the device, including the results that didn't work.

> Systems and measurement study, **not** a certified fire-safety product.
> Sister project to [jetson-xray-panel](https://github.com/MaximeCarriere/jetson-xray-panel-)
> — same box, same measurement discipline, new domain.

**What this is, in three lines:**

- A **fire and smoke detector** running on a $249 fanless computer the size of a paperback,
  with no cloud connection — the kind of box you would bolt to a mast in a forest.
- A **measured study of what makes it smaller and faster**, and what each step actually
  costs in accuracy: resolution, runtime conversion, quantization, and (planned) pruning
  and distillation.
- Every number is measured **on the device**, with the failures and wrong turns kept in.

## Where it stands

**Best configuration measured: YOLOv5s + TensorRT FP16 at 512 pixels — 0.778 mAP50,
474 images/second, 52 joules per 1000 frames, 17 MB.**

That runs on a fanless $249 computer drawing about 11 watts.

![What the detector is looking at](results/figures/dataset_examples.png)

| # | Experiment | Result |
|---|---|---|
| [XP0](experiments/xp00_foundation/) | Data, splits, measuring harness | Splits frozen and checksummed; class labels *proven* from the data, not assumed |
| [XP1](experiments/xp01_baselines/) | Baselines | **A 6.6× bigger model buys 1.4 accuracy points** at 3.3× the energy |
| [XP2](experiments/xp02_resolution/) | Resolution — the null hypothesis | **512px beats 640px outright**; overall accuracy hides a 77% collapse in distant-smoke detection |
| [XP9](experiments/xp09_tensorrt_fp16/) | TensorRT FP16 | **Up to 5× faster, accuracy free** — and it exposed that all earlier speed numbers were measuring the software |
| [XP10](experiments/xp10_int8_slices/) | INT8 quantization | **One default setting cost 67% of the accuracy**; fixed, the real cost is ~8% |
| [XP12](experiments/xp12_endurance/) | Endurance | 10 min flat out, 280k images, −1.3% drift, no throttling |

Planned and not yet run: distillation (XP3–5), pruning and sparsity (XP6–8),
quantization-aware training (XP11), full-stack demo (XP13), and a cascade "sleeping
detector" (XP14). See [PLAN.md](PLAN.md).

## Three findings worth the click

**The cheap knob beats the clever ones — so far.** Simply feeding the network 512-pixel
images instead of 640 made it *more* accurate and 1.6× faster. It also comes within 0.7
accuracy points of a model 6.6× its size, at 5.4× the speed. Every sophisticated compression
technique now has to beat that line to justify itself.

**Headline accuracy hides what matters.** Distant smoke — the thing an early-warning system
exists to catch — behaves nothing like the average. Dropping resolution cost 6% of overall
accuracy and **77%** of accuracy on the smallest targets. Every experiment here reports both.

**Defaults are not neutral.** TensorRT's standard quantization setting decided the useful
brightness range of an image was 0–0.45 out of 0–1, flattening the bright sky where faint
smoke lives. Accuracy fell 67%. It looked exactly like "INT8 is bad for detection" — a tidy,
quotable, wrong conclusion. It was one option.

> **What "plume" means here.** A plume is the visible smoke or flame region the detector has
> to find. Accuracy is reported separately for **small plumes** (under 1% of the frame) and
> **tiny plumes** (under 0.1%, roughly 20x20 pixels) — distant smoke, which is what early
> detection actually depends on.

![What small and tiny plume mean](results/figures/plume_definition.png)

## How the measurements are kept honest

Every number goes through one shared harness ([`lib/evaluator.py`](lib/evaluator.py)) with a
version stamp; if the harness changes, prior results are invalid and re-run. Beyond that:

- **Accuracy is reported per class and per target size**, never as a single number.
- **False alarms on empty frames** are counted separately — 47% of the test set contains no
  fire, and that is what a real deployment mostly sees.
- **Speed is reported two ways.** Per-frame latency and batched throughput measure different
  things on this hardware, and reporting only one is how XP9's error nearly happened.
- **Accuracy is bit-reproducible**, asserted by re-running and comparing a hash of every
  prediction.
- **No hand-made figures** — [`analysis/make_figures.py`](analysis/make_figures.py) rebuilds
  all of them from the committed result files.

## Honest limitations

- **No fire-safety claims.** This measures capability; it certifies nothing.
- **One dataset.** D-Fire is ground-level surveillance and web imagery from Brazil.
  Generalisation to other cameras, biomes or aerial views is untested.
- **Detection of genuinely distant smoke is weak in every configuration tested**
  (0.14–0.20 mAP50). A 6.6× larger model barely helps, so this looks like neither a capacity
  nor a resolution problem — and no compression technique can fix it. Tiled inference or
  higher-resolution input are the real levers, and neither is in the plan yet.
- **The baselines are the dataset authors' published models**, not ours, so their training
  recipe is unknown. Fair for measuring compression; not a clean architecture comparison.
- **Models are AGPL-3.0** (YOLOv5). Fine for open research; a commercial deployment would
  need a different architecture or a licence.

## Repository layout

```
lib/              shared harness: data · evaluator · detectors · power logging · TensorRT
experiments/      one folder per experiment — README, runnable script, results
results/raw/      every measurement as JSON, one file per run
results/figures/  generated only, never edited by hand
analysis/         figure generation
scripts/          engine building and ablations
PLAN.md           the original plan, plus §6 reconciling it against what was measured
```

Hardware: NVIDIA Jetson Orin Nano Super (8 GB), JetPack R36.4.3, CUDA 12.6, TensorRT 10.3.
Data: [D-Fire](https://github.com/gaiasd/DFireDataset) (Venâncio et al., 2022).

---

*Maxime Carriere*
