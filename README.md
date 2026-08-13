# Wildfire detection on a $249 edge computer

**How small and how fast can a fire/smoke detector get before it stops being useful?** This
repo measures that on one NVIDIA Jetson Orin Nano Super: every technique's real cost and
real gain, measured on the device, including the results that didn't work.

> Systems and measurement study, **not** a certified fire-safety product.
> Sister project to [jetson-xray-panel](https://github.com/MaximeCarriere/jetson-xray-panel-),
> using the same box and the same measurement discipline on a different problem.

**What this is, in three lines:**

- A **fire and smoke detector** running on a $249 fanless computer the size of a paperback,
  with no cloud connection. The kind of box you would bolt to a mast in a forest.
- A **measured study of what makes it smaller and faster**, and what each step actually
  costs in accuracy: resolution, runtime conversion, quantization, pruning.
- Every number is measured **on the device**, with the failures and wrong turns kept in.

## Where it stands

**Best configuration measured: YOLOv5s + TensorRT FP16 at 512 pixels. 0.778 mAP50,
474 images/second, 52 joules per 1000 frames, 17 MB.**

That runs on a fanless $249 computer drawing about 11 watts.

![What the detector is looking at](results/figures/dataset_examples.png)

| # | Experiment | Result |
|---|---|---|
| [XP0](experiments/xp00_foundation/) | Data, splits, measuring harness | Splits frozen and checksummed; class labels *proven* from the data, not assumed |
| [XP1](experiments/xp01_baselines/) | Baselines | **A 6.6× bigger model buys 1.4 accuracy points** at 3.3× the energy |
| [XP2](experiments/xp02_resolution/) | Resolution, the cheapest knob | **512px beats 640px outright**; overall accuracy hides a 77% collapse in distant-smoke detection |
| [XP6](experiments/xp06_pruning/) | Pruning | **Loses on accuracy and speed** against the unpruned model; cutting 2% of channels costs 9 accuracy points |
| [XP9](experiments/xp09_tensorrt_fp16/) | TensorRT FP16 | **Up to 5× faster, accuracy free**, and it exposed that all earlier speed numbers were measuring the software |
| [XP10](experiments/xp10_int8_slices/) | INT8 quantization | **One default setting cost 67% of the accuracy**; fixed, the real cost is about 8% |
| [XP12](experiments/xp12_endurance/) | Endurance | 10 min flat out, 280k images, −1.3% drift, no throttling |

Not yet run: knowledge distillation, 2:4 sparsity, quantization-aware training, a live demo,
and a two-stage cascade that keeps the expensive detector asleep. See [PLAN.md](PLAN.md).

## Three findings worth the click

**The cheap knob beats the clever ones, so far.** Simply feeding the network 512-pixel
images instead of 640 made it *more* accurate and 1.6× faster. It also comes within 0.7
accuracy points of a model 6.6× its size, at 5.4× the speed. Every sophisticated compression
technique has to beat that line to justify itself, and pruning already fails to.

**Headline accuracy hides what matters.** Distant smoke, the thing an early-warning system
exists to catch, behaves nothing like the average. Dropping resolution cost 6% of overall
accuracy and **77%** of accuracy on the smallest targets. Every experiment here reports both.

**Defaults are not neutral.** TensorRT's standard quantization setting decided the useful
brightness range of an image was 0 to 0.45 out of 0 to 1, flattening the bright sky where
faint smoke lives. Accuracy fell 67%. It looked exactly like "INT8 is bad for detection":
a tidy, quotable, wrong conclusion. It was one option.

> **What "plume" means here.** A plume is the visible smoke or flame region the detector has
> to find. Accuracy is reported separately for **small plumes** (under 1% of the frame) and
> **tiny plumes** (under 0.1%, roughly 20x20 pixels), which is distant smoke, and what early
> detection actually depends on.

![What small and tiny plume mean](results/figures/plume_definition.png)

## How the measurements are kept honest

Every number goes through one shared harness ([`lib/evaluator.py`](lib/evaluator.py)) with a
version stamp. If the harness changes, prior results are invalid and get re-run. Beyond that:

- **Accuracy is reported per class and per target size**, never as a single number.
- **Empty frames are scored separately.** 47% of the test set contains no fire at all, which
  is what a real deployment mostly sees, so "how often does it correctly stay silent" is its
  own metric rather than being folded into an aggregate.
- **Speed is reported two ways.** Per-frame latency and batched throughput measure different
  things on this hardware, and reporting only one is how XP9's error nearly happened.
- **Accuracy is bit-reproducible**, asserted by re-running and comparing a hash of every
  prediction.
- **Training refuses to start** if the dataset splits do not match their frozen checksums, so
  test images cannot leak into training when work moves to another machine.
- **No hand-made figures.** [`analysis/make_figures.py`](analysis/make_figures.py) rebuilds
  all of them from the committed result files.

## Honest limitations

- **No fire-safety claims.** This measures capability; it certifies nothing.
- **One dataset.** D-Fire is ground-level surveillance and web imagery from Brazil.
  Generalisation to other cameras, biomes or aerial views is untested.
- **Detection of genuinely distant smoke is weak in every configuration tested**
  (0.14 to 0.20 mAP50). A 6.6× larger model barely helps, so this looks like neither a
  capacity nor a resolution problem, and no compression technique can fix it. Tiled
  inference or a higher-resolution input are the real levers, and neither is tested here.
- **The baselines are the dataset authors' published models**, not ours, so their training
  recipe is unknown. Fair for measuring compression; not a clean architecture comparison.
- **Models are AGPL-3.0** (YOLOv5). Fine for open research; a commercial deployment would
  need a different architecture or a licence.

## Repository layout

```
lib/              shared harness: data, evaluator, detectors, power logging, TensorRT
experiments/      one folder per experiment, each with a README, script and results
results/raw/      every measurement as JSON, one file per run
results/figures/  generated only, never edited by hand
analysis/         figure generation
scripts/          engine building and ablations
PLAN.md           the original plan, plus a section reconciling it against what was measured
HANDOFF_RTX3090.md  setup, protocol and experiment plan for continuing the pruning work
                    on a desktop GPU, and what has to stay on the Jetson
```

Hardware: NVIDIA Jetson Orin Nano Super (8 GB), JetPack R36.4.3, CUDA 12.6, TensorRT 10.3.
Data: [D-Fire](https://github.com/gaiasd/DFireDataset) (Venâncio et al., 2022).

---

*Maxime Carriere*
