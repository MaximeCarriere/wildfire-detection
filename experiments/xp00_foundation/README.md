# XP0. Foundation: data, splits, and the measuring instrument

**Question:** before measuring anything, is the data understood and the measuring
apparatus trustworthy?

**Outcome:** yes, and two things surfaced that changed how everything after it is measured.

![What the detector is looking at](../../results/figures/dataset_examples.png)

![What a fire detector is asked to find](../../results/figures/xp00_dataset.png)

## What was frozen

D-Fire (21,527 images, ground-level surveillance cameras and web photos, **not** drone
footage). The published train/test split is preserved so results stay comparable to the
literature; only a validation set is carved out, at seed 42.

| Split | Images | fire boxes | smoke boxes | checksum |
|---|---:|---:|---:|---|
| train | 15,500 | 10,593 | 8,600 | `8dea80450f1d1611` |
| val | 1,721 | 1,221 | 950 | `48c7552154b86a49` |
| test | 4,306 | 2,878 | 2,315 | `7cc103e8cbb706f5` |

Also frozen: a 500-image calibration set for later compression work, and the definition of
a "small plume".

## Two findings that shaped everything after

**1. Half the targets are tiny.**

- The median box covers **1.34%** of the image.
- Defining a "small plume" as anything under 1% therefore sits almost exactly *on* that
  median, selecting the smaller **half** of all targets rather than the hard cases.
- A second tier was added at **0.1%** (≈20×20 pixels, 10.6% of boxes).
- The two behave completely differently under compression, so the distinction mattered:
  dropping resolution later cost 30% of the first tier and **77%** of the second.

**2. Nearly half the frames contain nothing.**

- **2,005 of 4,306** test images are empty landscape, containing no fire and no smoke.
- Standard detection metrics fold false alarms on those into one aggregate score.
- So a separate **false-alarm rate** was added: what fraction of empty frames raise an alarm.
- For a camera that watches nothing almost all the time, that is arguably the number that
  decides whether the system is deployable at all.

> **What "plume" means here.** A plume is the visible smoke or flame region the detector has
> to find. Accuracy is reported separately for **small plumes** (under 1% of the frame) and
> **tiny plumes** (under 0.1%, roughly 20×20 pixels), which is distant smoke, and what
> early detection actually depends on.

![What small and tiny plume mean](../../results/figures/plume_definition.png)

## Limitations

- **Condition labels don't exist in D-Fire.** Night, fog and backlight are approximated from
  image brightness statistics, and are stated as proxies wherever they are used.
- **The splits are pinned to the checksums above.** Each split is a list of image paths, and
  the checksum is a fingerprint of that list, so moving one image between splits changes it.
  Every accuracy number in this repo was measured on one specific set of 4,306 test images,
  so if that set silently changed, comparing a new model against an old number would be
  comparing scores on different exam papers. A checksum mismatch makes that visible instead
  of invisible, and invalidates every prior result until they are re-run.

## Next

Measuring published fire detectors through this harness
([XP1](../xp01_baselines/)).
