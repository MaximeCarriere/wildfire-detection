# Data

`data/` holds the datasets themselves and is **gitignored** — do not commit images,
labels, or weights. What *is* committed is `data/splits/`: the frozen file lists that
define every split, the small-plume view, and the INT8 calibration set. This file records
what should be on disk and how to get it back.

## D-Fire (present on disk) — train / val / in-distribution test

Supplied by the repo owner from the official OneDrive link on 2026-08-12, transferred to
the Jetson and unpacked to `data/dfire/`.

> Pedro Vinícius Almeida Borges de Venâncio, Adriano Chaves Lisboa, Adriano Vilela
> Barbosa. *An automatic fire detection system based on deep convolutional neural networks
> for low-power, resource-constrained devices.* Neural Computing and Applications, 2022.
> <https://github.com/gaiasd/DFireDataset>

```
data/dfire/
├── train/images/   17,221 jpg
├── train/labels/   17,221 txt      YOLO: <class> <cx> <cy> <w> <h>, normalized
├── test/images/     4,306 jpg
└── test/labels/     4,306 txt
```

`D-Fire.zip` — 3,036,222,313 bytes, `sha256 3824fb3ce32cfa8b538792dfd460603648d271072ac6ae34f1af0c713f60260c`

Restore with:

```bash
# The 1drv.ms link needs a browser session — scripted GETs get 401/403.
# Download "D-Fire dataset (only images and labels)" from the repo above, then:
unzip -q D-Fire.zip -d data/dfire
python experiments/xp00_foundation/run.py --stages scan splits calib
```

### Class ids are verified, not assumed

The D-Fire README numbers the classes *Fire (1), Smoke (2)* while the label files use
`0` and `1`. XP0 resolves the ambiguity from the data itself: the published box counts
(fire 14,692 / smoke 11,865) are far enough apart to act as a fingerprint.

Observed across all 21,527 images: **class 0 → 11,865 boxes, class 1 → 14,692 boxes** —
an exact match. So `0 = smoke, 1 = fire`, asserted on every run by
`lib.data.verify_class_map`, which halts XP0 if the data ever disagrees.

### Splits

D-Fire's **published train/test split is preserved** — it is the split every paper reports
against, and re-shuffling it would silently cost that comparison. Only *val* is carved out,
from the training pool, stratified by content category (fire / smoke / both / none) at
seed 42.

| Split | Images | fire boxes | smoke boxes | small-plume boxes |
|---|---:|---:|---:|---:|
| train | 15,500 | 10,593 | 8,600 | 8,550 |
| val | 1,721 | 1,221 | 950 | 947 |
| test | 4,306 | 2,878 | 2,315 | 2,359 |

Checksums (`data/splits/manifest.json`): train `8dea80450f1d1611`, val `48c7552154b86a49`,
test `7cc103e8cbb706f5`. Any change to these invalidates every prior result.

**Small plumes are 45% of all boxes**, not a rare tail — 1,229 of the 4,306 test images
carry at least one box under 1% of image area. `map50_small_plume` is therefore a
well-populated metric rather than a noisy one, which is good news for the early-detection
story the series is built on.

## OOD sets — test only, quarantined (NOT yet on disk)

These are evaluated on and **never trained on**. `lib.data.assert_trainable` reads every
`data/splits/ood_*.txt` and raises if a training list touches one, so the quarantine is
enforced by the harness rather than by memory.

| Dataset | Role | Status | Source |
|---|---|---|---|
| FLAME / FLAME 2 (~2k sampled aerial frames) | primary OOD — aerial drone domain shift | ⬜ needed | IEEE DataPort, open access (account required) |
| Boreal Forest Fire (~1k sampled) | optional 2nd OOD — European vegetation | ⬜ optional | Scientific Data release |
| FLAME 2 video clips (incl. no-fire segments) | XP14 cascade evaluation | ⬜ needed for XP14 | IEEE DataPort |

Until these land, `map50_ood_flame` and `map50_ood_boreal` stay `null` in every
results.json — absent rather than faked.

## INT8 calibration set

500 images drawn from the **training split only** (never val, never test, never OOD),
frozen in `data/splits/calib.txt` and described by `data/splits/calib_manifest.json`.

PLAN.md §1 requires coverage of night, fog, sunset/backlight and small distant plumes —
the conditions INT8 is most likely to quietly break. D-Fire carries no condition labels,
so conditions are approximated from luminance statistics, quantile-based so thresholds
adapt to the dataset instead of being magic numbers:

| Category | Proxy | Quota |
|---|---|---:|
| small plume | has a GT box < 1% of image area | 150 |
| night | lowest 15% mean luminance | 100 |
| fog / haze | lowest 15% luminance std (low contrast) | 100 |
| backlight | highest 15% of (p95 − mean) luminance | 100 |
| random | topped up from the remainder | 50 |

These are proxies and they are stated as proxies. XP10's slice analysis is where they get
tested: if INT8 breaks on a condition the proxy mislabels, that shows up there, and the
proxy gets revisited — with a `PROTOCOL_VERSION` bump, because it would change numbers.

## Licensing

D-Fire is distributed from the authors' repository for research use; cite the 2022 paper
in any publication. Check the repository's LICENSE before republishing images or derived
weights. Trained weights in this repo are derived works of an AGPL-3.0 codebase
(Ultralytics) — see the license note in the top-level README.
