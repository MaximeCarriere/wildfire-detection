# PLAN.md — Wildfire Detection Compression Study on a $249 Jetson Orin Nano Super

**Project codename:** `jetson-fire-watch`
**Question:** How small and how fast can a fire/smoke detector get — via distillation,
pruning, sparsity, quantization, resolution, and system-level gating — before it stops
being useful, and what does each step actually buy on a $249 edge box?

> Systems / capability study, **not** a certified fire-safety product. The value is the
> *measured compression arc*: every technique's real cost and real gain on-device,
> including negative results. Sister project to
> [jetson-xray-panel](https://github.com/MaximeCarriere/jetson-xray-panel-) — same box,
> same measurement discipline, new domain (detection instead of classification).

> ⚠️ **This plan was written before anything was measured. Five of its assumptions did not
> survive contact with the board.** The original text below is preserved unedited so the
> changes stay auditable; **§6 (Reconciliation) at the end records what was measured, what
> it contradicts, and what changed as a result.** Read §6 before following §2.

---

## 0. Ground rules (apply to every XP)

These mirror the X-ray repo conventions. Enforce them in code, not by hope.

- **One folder per experiment**: `experiments/xpNN_name/` with its own `README.md`,
  scripts, and a `results.json`. Each XP must be runnable standalone
  (`python run.py`), pulling shared code from `lib/`.
- **Frozen protocol**: all evaluation goes through ONE shared harness
  (`lib/evaluator.py`). No XP implements its own mAP computation or its own timing
  loop. If the harness changes, every prior XP result is invalidated — version it
  (`protocol_version` field in every results.json).
- **Repeats and error bars**: throughput/latency = 3 runs minimum, report mean ± 1 SE.
  Latency = median + p95 over ≥ 1000 frames after ≥ 50 warm-up frames.
  Accuracy (mAP) is deterministic given fixed weights + fixed data — assert
  bit-reproducibility across 2 runs instead.
- **On-device numbers only count on-device**: anything labeled "Jetson" must be
  measured on the Jetson Orin Nano Super (8 GB), power mode stated (default MAXN,
  also record 25 W where noted), via `lib/power_logger.py` (tegrastats wrapper —
  port from the X-ray repo).
- **Negative results are results.** If a technique doesn't help, the XP README says so
  in its headline line.
- **Every figure regenerable**: `analysis/make_figures.py` rebuilds all plots from
  `results/raw/`. No hand-made figures.
- **Fixed hyper-choices, one per knob** (to keep the series comparable):
  input 640×640 unless the XP's explicit variable is resolution, NMS IoU 0.45,
  conf threshold 0.25 for qualitative demos, mAP computed threshold-free
  (COCO-style) via the harness. Seed = 42 everywhere.
  Any deviation must be an explicit XP variable, never a silent change.

### Metrics schema (identical keys in every results.json)

```json
{
  "protocol_version": "1.0",
  "model_id": "yolov8s_distilled_w050",
  "params_m": 0.0,
  "size_disk_mb": 0.0,
  "format": "pt | onnx | trt_fp16 | trt_int8 | trt_int8_24sparse",
  "input_res": 640,
  "map50_dfire_test": 0.0,
  "map5095_dfire_test": 0.0,
  "map50_fire_class": 0.0,
  "map50_smoke_class": 0.0,
  "map50_small_plume": 0.0,
  "map50_ood_flame": 0.0,
  "map50_ood_boreal": null,
  "jetson": {
    "power_mode": "MAXN",
    "latency_ms_median": 0.0,
    "latency_ms_p95": 0.0,
    "fps_batch1": 0.0,
    "mem_mb": 0.0,
    "power_w_mean": 0.0,
    "runs": 3,
    "se": 0.0
  }
}
```

Per-class mAP is mandatory: smoke degrades before fire under compression — that
asymmetry is a core finding of the series, don't average it away. `map50_small_plume`
(boxes < 1% of image area, subset tagged in XP0) is the early-detection metric —
the one that matters most for a real fire watch and the first casualty of most
compression techniques. Track it from XP1, not just in the INT8 analysis.

---

## 1. Assets

### Models (Ultralytics YOLOv8 family)

| Role | Model | Params | Disk (FP32) |
|---|---|---|---|
| Teacher | YOLOv8m | 25.9 M | ~50 MB |
| Baseline / student | YOLOv8s | 11.2 M | ~22 MB |
| Control + student | YOLOv8n | 3.2 M | ~6 MB |
| Sub-nano students | YOLOv8n @ width 0.75 / 0.5 / 0.25 | <3.2 M | <6 MB |

**License note (state in README + final post):** Ultralytics YOLOv8 is AGPL-3.0.
Fine for this open research repo (all code published). A commercial deployment
would swap to an Apache-2.0 architecture (YOLOX-s / RT-DETR) or license Ultralytics.
This is a deliberate, documented trade-off — say it before HN says it for us.

### Datasets

| Dataset | Role | Size | Notes |
|---|---|---|---|
| D-Fire (~21k images, fire+smoke boxes, YOLO format) | train / val / in-distribution test | ~3–4 GB | Verify license terms in repo before first commit; cite the paper |
| FLAME / FLAME 2 aerial frames (~2k sampled) | OOD test only — never trained on | ~1–2 GB sampled | IEEE DataPort open access; aerial drone domain shift |
| Boreal Forest Fire (~1k sampled) | optional 2nd OOD test | ~1 GB sampled | European vegetation; Scientific Data release |
| FLAME 2 video clips (a few, incl. no-fire segments) | XP14 cascade evaluation | ~1 GB | temporal stream with long idle stretches |
| Calibration set (500 imgs from D-Fire train) | INT8 calibration, frozen in XP0 | — | MUST include: night, fog, sunset/backlight, small distant plumes |

Splits are frozen in XP0 as explicit file lists committed to `data/splits/`.
OOD sets are quarantined: the harness refuses to train on any file in an OOD list.

### Hardware

- Training/distillation: rented GPU (4090 / A10 class), a few days total.
- Everything else (pruning fine-tune verification, all evaluation, all timing): Jetson.
- Note: the Orin **Nano** has NO DLA units (that's Orin NX / AGX) — DLA offload is
  out of scope by hardware, not by choice. One line in the final post.

---

## 2. Experiments

### Phase 0 — Foundation

**XP0 — Data, splits, calibration set, harness**
Download D-Fire; build frozen train/val/test splits + OOD samples + calibration set
+ small-plume test subset tags; implement `lib/evaluator.py` (mAP, timing loop,
power logging); smoke-test end to end with COCO-pretrained YOLOv8n (no fine-tune).
*Deliverable:* splits committed, harness unit-tested, one dry-run results.json.

**XP1 — Baselines & control**
Fine-tune YOLOv8m (teacher), YOLOv8s, YOLOv8n on D-Fire (same recipe, same epochs,
same augmentation — recipe frozen here). Evaluate all three, full schema, on Jetson
(baseline runtime: PyTorch FP16 — stated once, used everywhere until Phase 4).
*Headline to fill:* the teacher ceiling, the n-control floor, per-class + OOD +
small-plume gaps. *This is the table every later XP is judged against.*

**XP2 — Resolution sweep (the cheapest knob, measured first)**
All three XP1 baselines at 640 / 512 / 416 / 320 input. No retraining arm first
(pure inference-resolution drop), then one retrained-at-resolution arm for YOLOv8s
to quantify the train/test resolution mismatch cost.
*Deliverable:* the resolution frontier — mAP vs FPS per resolution per model, with
`map50_small_plume` as the counter-metric (distant plumes are what low res kills).
*Why first:* every later technique must beat this frontier to justify its
complexity. If plain 448px YOLOv8s dominates a pruned 640px model, the series
says so. Resolution is the null hypothesis of model compression.

### Phase 1 — Distillation (how small can a student get?)

**XP3 — Distillation machinery + sanity check**
Implement output distillation (teacher soft targets + ground truth) and one
feature-map distillation method at the neck (CWD or MGD — pick ONE, justify in
README, do not implement both). Sanity check: distill YOLOv8s from YOLOv8m; must
beat the XP1 label-only YOLOv8s or the machinery is broken.
*Deliverable:* distilled-s vs plain-s, same-architecture pair, the attribution test.

**XP4 — The student ladder**
Distill down the ladder: s → n → width-scaled n (0.75×, 0.5×, 0.25×). For EVERY
rung also train the same architecture label-only (no teacher). 2 runs each if
budget allows, else 1 run + note.
*Deliverable:* THE curve — params (x, log) vs mAP50 (y), two lines
(with / without teacher), in-distribution AND the OOD twin plot.
*Questions answered:* where do the lines diverge (distillation gain vs size)?
Where is the cliff? Does the teacher transfer generalization (OOD) or only
in-distribution accuracy? Which class dies first?
*Expected honest outcome:* distillation buys ~one size class, a few mAP points.
If it buys nothing — that's the headline, publish it.

**XP5 — Pick S\* + Jetson pass**
Select S\* = best size/accuracy trade-off from XP4 (decision rule stated in README
*before* looking: e.g. smallest student within 2 mAP50 points of distilled-s on
in-distribution AND within 3 on OOD). Full Jetson measurement of the whole ladder
(latency, FPS, memory, power) — first time size translates to measured speed.
Overlay the ladder on the XP2 resolution frontier: does distillation actually
beat the dumb knob?
*Deliverable:* the size→speed reality table; blog post 1 ships after this XP.

### Phase 2 — Pruning & sparsity (cutting S\*)

**XP6 — Structured channel pruning sweep**
torch-pruning (use their Ultralytics examples as the starting point — the detection
head plumbing is solved there) on S\*: target 25% / 50% / 70% FLOPs reduction.
After each: recovery fine-tune, TWO arms — (a) labels only, (b) labels + teacher
(distillation-aware recovery, reusing XP3 machinery).
*Deliverable:* accuracy-vs-sparsity curve, both recovery arms, cliff located,
per-class + OOD + small-plume throughout.

**XP7 — 2:4 structured sparsity (the hardware-native rival)**
The Orin's Ampere GPU has sparse tensor cores: 2-of-4 fine-grained sparsity
(50% weights zeroed) with native TensorRT acceleration. Apply 2:4 sparsification
to S\* (NVIDIA ASP / apex.contrib.sparsity workflow), fine-tune to recover, and
run head-to-head against the XP6 channel-pruned model at matched ~50% sparsity.
*Deliverable:* the sparsity showdown — channel pruning (generic, tile-luck speedup)
vs 2:4 (hardware-guaranteed speedup, fixed 50%) at equal weight reduction:
accuracy, and later (XP10) measured TensorRT speed. This head-to-head on a Jetson
is genuinely underpublished — likely the most cited artifact of Phase 2.
*Note:* 2:4 speedup only materializes inside TensorRT sparse engines — PyTorch
numbers here are accuracy-only; speed verdict lands in XP10.

**XP8 — FLOPs vs reality — CANCELLED, superseded by XP6**
Was: measure every pruned variant on the Jetson and plot theoretical FLOPs
reduction against measured FPS gain, the gap being the finding (GPU tile-size
friendliness ≠ FLOPs).

*Why cancelled:* the measurement it depended on could not have meant anything when
it was scheduled. XP2 found PyTorch eager inference on this board is
kernel-launch-bound at batch 1, so model size barely moves measured speed at all —
the honesty plot would have shown ~0% FPS gain from any FLOPs cut, a flat line
that looks exactly like the intended finding while having an entirely different
cause. Phase 3 moved ahead of Phases 1–2 for this reason (see "What XP2 changed"
below), and by the time TensorRT removed the bottleneck the question had a better
home.

*Where the question was answered instead:* XP6's extension, on TensorRT engines
rather than PyTorch timings, and with the mechanism measured rather than assumed.
XP6 E3 is the honesty plot in its sharpest form — 4.21 M parameters at 363 img/s
against 3.52 M at 642, so fewer parameters and 1.77x the speed. XP6 E5 carries the
gap itself: 88.9% of the multiply-adds removed bought 1.55x, not the ~9x the
arithmetic implies. XP6 E9 found the cause XP8 could only have hypothesised, by
timing one convolution across widths: four of fifteen run *slower* than the
unpruned layer, and the clean sequence runs through the multiples of 32.

*If it is ever wanted as an artifact*, it is now a synthesis figure over data that
already exists, not a run.

### Phase 3 — Quantization (the deployment squeeze)

**XP9 — TensorRT FP16**
Export the XP6 winner (pruned+recovered S\*, call it P\*) → ONNX → TensorRT FP16 on
the Jetson. Also convert the unpruned S\* and YOLOv8n control for comparison.
*Expected:* accuracy ~free, large speedup (X-ray repo saw 25× over naive PyTorch —
detection will differ, measure it).

**XP10 — INT8 PTQ + sparse engines + failure analysis**
TensorRT INT8 with the frozen XP0 calibration set, on P\* AND on the XP7 2:4 model
(sparse INT8 engine — the maximum-squeeze configuration this GPU supports).
This is where the XP7 showdown resolves: measured FPS of channel-pruned vs 2:4
sparse engines. THEN the differentiating analysis: slice errors by condition —
night / fog / backlight / small-plume. Does INT8 quietly kill early detection
(small faint plumes) while aggregate mAP looks fine?
*Deliverable:* aggregate + sliced accuracy, speed, power, sparse-vs-pruned verdict.
The slice table is the most shareable artifact of the series.

**XP11 — QAT rescue (conditional)**
Run ONLY if XP10 drops > ~1–2 mAP50: quantization-aware training on P\*, optionally
teacher-supervised (closing the loop with Phase 1). If XP10 is clean, skip and
say so in a one-line README.

### Phase 4 — System level & the stack

**XP12 — Endurance + power envelope**
10-minute sustained inference on the final INT8 model: throughput drift,
temperature, throttling check (X-ray repo format: held 508 img/s, −0.2%, 69 °C).
Repeat at 25 W power mode — the solar-mast scenario.

**XP13 — Final stack table + demo**
One table: YOLOv8m teacher → distilled S\* → pruned/sparse P\* → INT8, each row
full schema, vs the YOLOv8n control line AND the best XP2 resolution-only
baseline. Total compression factor, total speedup, total mAP cost — and whether
the fancy stack beats the dumb knobs. Live demo: camera feed on the Jetson,
WiFi off, fire/smoke boxes on screen — `demos/` gets a 20-second capture.
Blog post 3 + finale ship after this.

**XP15 — The 5-euro sensor: can the gate leave the Jetson entirely?**
XP14 puts a cheap gate and the full detector on the *same* board. This asks
whether the gate belongs on separate silicon: a XIAO ESP32-S3 Sense (240 MHz
Xtensa LX7 with SIMD, 8 MB PSRAM, OV2640) running a tiny binary "anything there?"
classifier, waking an Orin only on suspicion.

**Classification, not detection, and the reason is already measured.** XP2's
resolution sweep shows a *full* YOLOv5s scoring 0.0000 on tiny plumes at 160 px
while still reaching 0.458 mAP50 overall. An ESP32 runs at ~96 px with a model
orders of magnitude smaller, so boxes are hopeless and distant plumes are gone
before the microcontroller is even involved. What may survive is "there is obvious
fire or smoke in view", which is all a wake-up gate needs.

*Two stages, and the first can kill the second:* train the classifier on a
workstation and measure recall stratified by plume size; only if it clears the bar
does the port happen. Then deploy, and measure what actually matters for a device
that watches nothing 99.9% of the time — **average power, not FPS**.
*Deliverable:* the recall-vs-plume-size table, and an average-watts comparison
against an always-on Orin. A negative result is publishable and cheap.

**XP14 — Cascade gate: the idle-sky experiment**
A fire watch stares at nothing 99.9% of the time — per-frame FPS is the wrong
metric; average watts is the right one. Build a two-stage cascade: a cheap gate
(arm A: frame-differencing / motion trigger; arm B: the 0.25× sub-nano student
from XP4 as a binary "anything there?" screener) runs every frame; the full P\*
INT8 detector wakes only on suspicion. Evaluate on FLAME 2 video clips including
long no-fire stretches: detection recall vs gate threshold, false-wake rate,
and measured average power vs always-on.
*Deliverable:* average-watts comparison (always-on vs gated), the recall/power
trade-off curve, and time-to-first-detection when fire does appear.
*Why it matters:* this is the system-level technique no per-model compression
can touch, it reuses the distillation ladder's smallest student, and average
power is THE metric for the solar-powered-mast story. Strong standalone post
("the detector that sleeps").

### Stretch (only if the series lands)

- **XP15 — See-then-say transplant:** detector sees → fixed rule maps confidence to
  alert level → on-device LLM writes the dispatch line ("smoke plume, NE bearing,
  confidence 0.87…"). The X-ray XP13 pattern, third domain — the platform proof.
- **XP16 — Sub-nano on-device limits:** the 0.25× student on INT8 — how low is
  still useful? Bridge toward the ESP32 fixed-mast story.
- **Road not taken (document, don't run):** low-rank factorization (dominated by
  pruning+quant on modern CNNs), NAS (a research program, not an XP),
  binarization (detection falls off a cliff), DLA offload (no DLA on this SKU).

---

## 3. Repository layout

```
lib/            shared: data.py (splits, quarantine), evaluator.py (mAP + timing,
                versioned), distill.py, prune_utils.py, sparsity_24.py,
                cascade.py, trt_export.py, power_logger.py (port from
                jetson-xray-panel)
data/           splits/ (committed file lists) · README with download instructions
                (datasets themselves are NOT committed — .gitignore'd)
experiments/    xp00_foundation … xp14_cascade_gate, each: README.md + run.py +
                results.json
demos/          live camera demo (final model) · captured clips
results/        raw/ (per-run JSON) · figures/ (generated only)
analysis/       make_figures.py — regenerates every figure from results/raw/
PLAN.md         this file
requirements.txt  desktop-GPU env + Jetson env (two sections, like the X-ray repo)
```

---

## 4. Honesty guardrails

- No fire-safety claims. This measures capability, not certifies a product.
- Datasets are prescribed burns and curated web images — real deployment is harder;
  say so in every post.
- AGPL-3.0 (Ultralytics) stated openly + the production swap named.
- The $249 is a hook, not a hard constraint.
- The YOLOv8n control ("just use a smaller model") AND the resolution frontier
  ("just use a smaller input") are reported next to every compressed result.
  If the fancy techniques never beat the dumb knobs, the series says so — that
  too is a publishable finding, arguably the most useful one.
- Negative results in headlines, not footnotes.

## 5. Publishing checkpoints

| After | Blog post |
|---|---|
| XP5 | Post 1 — Distillation: how small can a fire detector get? (the ladder curve, vs the resolution null hypothesis) |
| ~~XP8~~ XP6 | Post 2 — Pruning: FLOPs lie, FPS doesn't (XP8 cancelled; XP6 E3/E5/E9 carry it) |
| XP10/11 | Post 3 — INT8 + 2:4 sparsity: fast, frugal, and where it fails (the slice analysis + the sparsity showdown) |
| XP13 | Finale — The full stack, one table, live demo (Show HN candidate) |
| XP14 | Bonus post — The detector that sleeps: watts, not FPS (solar-mast story) |

Each post reuses the X-ray post format: measured on-board, variance stated,
endurance run, boundary section, design-partner CTA.

---

## 6. Reconciliation — what the board actually measured

*Added after XP0–XP2. §0–§5 above are the original plan, preserved unedited. This section
records where measurement contradicted it. Every claim here is backed by a committed
`results/raw/*.json` under `protocol_version` 1.1.*

### 6.1 The five assumptions that broke

**① "Baseline runtime: PyTorch FP16, used everywhere until Phase 4" (§2, XP1).**
PyTorch eager inference on this board is **kernel-launch-bound at batch 1**. A YOLOv5s
forward pass costs ~22 ms whether the input is 640×640 or 320×320 — 4× fewer pixels, zero
speedup — and at 320px *eight* images complete in the same wall-clock as one (21.40 vs
21.70 ms). The ~200 kernel launches per forward, not the arithmetic, set the floor.

Consequence: in that regime model size barely moves measured speed. **XP8's "FLOPs vs FPS
honesty plot" would have shown ~0% FPS gain from any FLOPs cut** — which is why XP8 was
cancelled outright rather than rescheduled — a flat line resembling
the intended finding while having an entirely different cause. XP5's size→speed table would
have been equally hollow.

*Changed:* protocol 1.1 records **both** batch-1 latency (real, and the right metric for
XP14's single camera feed) and **batched throughput** (the compute-bound axis compression
must be judged on). **Phase 3 (TensorRT) moved ahead of Phases 1–2**, because it is what
actually removes the bottleneck.

**② "Smoke degrades before fire under compression — a core finding" (§0).**
At baseline the asymmetry runs the *other way*: fire 0.7206 vs smoke 0.8210 (YOLOv5s), and
scaling to YOLOv5l improves smoke by 2.7 points and fire by 0.05. Fire is the weaker class
by ~10 points before any compression is applied.
*Changed:* treated as an open question, not an assumption. Per-class reporting stays
mandatory — the reason for it is unaffected.

**③ "Teacher YOLOv8m → student ladder" (§2, Phase 1).**
The prospective teacher beats the prospective student by **1.4 mAP50 points** (0.7847 vs
0.7708) for 6.6× the parameters, 1.75× the latency and 3.3× the energy. Distillation
transfers the teacher's surplus; here the surplus is thin. It is not zero — soft targets
carry information beyond the teacher's own mAP — but XP3–XP5 start from a much smaller
margin than the plan assumed.
*Changed:* teacher choice should be revisited before spending training budget. XP3's sanity
check remains the right first test.

**④ "map50_small_plume (boxes < 1% of image area)" (§0).**
Measured: D-Fire's **median box covers 1.34%** of the image, so the 1% threshold sits
essentially *on the median* and selects the smaller half (45.4% of boxes) rather than the
hard tail its name implies.
*Changed:* protocol 1.1 adds a second **tiny-plume tier at <0.1%** (~20×20 px in a 640 frame,
10.6% of boxes). It immediately justified itself: dropping 640→320 costs 6% of aggregate
mAP50, 30% of small-plume, and **77% of tiny-plume**. The 1% tier alone would have
understated the collapse by half.

**⑤ "map50_ood_flame" (§1).**
FLAME and FLAME 2 ship **frame-level classification labels and segmentation masks, not
bounding boxes**. The mAP the schema asks for is not directly obtainable from them.
*Changed:* OOD deprioritised to a later robustness check. FLAME 2's real value is **XP14's
cascade gate**, which needs exactly the frame-level fire/no-fire labels it has. If an OOD
mAP is ever wanted, FLAME 1 masks can be converted to boxes — with every resulting number
labelled as *derived* ground truth.

### 6.1b The INT8 trap — quantizing the decode destroys detection

Not a broken assumption in the plan so much as a hole in it. §2's XP10 says "TensorRT INT8
with the frozen XP0 calibration set" as though that were one unambiguous operation. It is
not, and the wrong reading of it produces a result that looks like a finding.

**Measured, first attempt:** YOLOv5s INT8 scored **mAP50 0.2554** against FP16's 0.7776 — a
67% collapse, with small-plume down 93% and tiny-plume down 99% (0.1376 -> 0.0016), while
classification partly survived (the night slice held 0.4823 against 0.8184).

**Cause**, visible in the exported graph: YOLOv5's Detect layer decodes the raw head outputs
into a single tensor concatenating **box coordinates in pixels (0-640)** with **objectness
and class probabilities in [0,1]**. INT8 carries one scale per tensor, so 256 levels are
stretched over 0-640 — roughly 2.5 px of granularity on every box edge — while the
probabilities are compressed into a fraction of one quantization step. Box regression is
destroyed and the classifier limps on. That asymmetry is the fingerprint, and it is why the
per-slice reporting mattered: an aggregate-only view would have shown "INT8 costs 67%" with
no clue as to why.

*Changed:* `lib.trt_export.build_int8_engine` now pins the decode tail to FP16 by default.
Those layers are elementwise arithmetic, reshapes and concats — a negligible share of the
FLOPs — so the fix should be nearly free in speed. The broken engines and their results are
archived under `experiments/xp10_int8_slices/evidence_no_fp16_head/` rather than deleted:
the before/after is a stronger artifact than a clean INT8 number alone.

**Publishing note.** "INT8 costs a fire detector 67% of its accuracy" would have been
striking, quotable and wrong — a claim about quantization when it is really a claim about
*where you let the quantizer reach*. Any INT8 number in this series must state which layers
were quantized.

### 6.1c On-device training: recovery fine-tunes yes, the distillation ladder no

*Measured* on this board, compute only (no dataloading, augmentation or mosaic — treat every
epoch figure as a floor):

| model | res | batch | ms/iter | epoch over 15,500 images | peak memory |
|---|---:|---:|---:|---:|---:|
| YOLOv5s | 640 | 4 | 181.8 | 11.7 min | 1.31 GB |
| YOLOv5s | 640 | 8 | 345.6 | 11.2 min | 2.57 GB |
| YOLOv5s | 640 | 16 | — | — | **process SIGKILLed** |
| YOLOv5s | 512 | 4 | 116.5 | 7.5 min | 0.85 GB |
| YOLOv5s | 512 | 8 | 219.5 | **7.1 min** | 1.65 GB |
| YOLOv5l | 640 | 2 | 392.7 | 50.7 min | 2.28 GB |
| YOLOv5l | 640 | 4 | 696.7 | 45.0 min | 4.13 GB |

Batch 16 at 640 does not raise CUDA OOM — the Linux OOM killer takes the whole process,
because the 8 GB is shared between CPU and GPU. **Batch 8 is the practical ceiling at 640.**

**Distillation overhead, measured separately:** a frozen YOLOv5l teacher forward costs
**+48.5 ms per image** at 640. Added to the student's step that is 345.6 + 387.8 = 733 ms
per batch-8 iteration, or **~23.7 min per epoch**.

What that implies for §2:

- **Pruning recovery fine-tune (XP6)** — one run from an already-trained model. The
  compute-only figure above says ~2.5 h at 512 for 20 epochs. **Measured in XP6 with the
  real dataloader and augmentation: ~20 min/epoch at 512 px, batch 8 — roughly 3x the
  compute-only estimate**, because the Orin's CPU, not its GPU, is the bottleneck for JPEG
  decode and mosaic. So ~4 h for 12 epochs. Still an overnight job, still **no GPU rental
  needed** — but every wall-clock estimate derived from the compute-only table above should
  be multiplied by about three.
- **XP4's distillation ladder** — ~5 rungs x 2 arms = 10 runs at ~24 min/epoch:
  **8+ days of board time for 50 epochs each.** Not feasible here. It needs a rented GPU, or
  the ladder needs shrinking (fewer rungs, fewer epochs, or 512px training — which is 37%
  cheaper per epoch and, per §6.3, the resolution you would deploy at anyway).

### 6.2 Other deviations, deliberate

- **Model family is YOLOv5, not YOLOv8.** The D-Fire authors published trained YOLOv5s/l
  (MIT) on D-Fire's official train split. XP0 preserved that split, so our test set is
  provably outside their training data — which bought a real baseline on day one with no
  GPU rental. Cost: XP1's "same recipe, same epochs" symmetry is lost (their recipe,
  unknown epochs and training resolution), and there is no published YOLOv5n, so the
  control floor still needs one training run.
- **Background false-alarm rate added to the schema.** 46.6% of the test split is empty
  landscape; mAP folds those false positives into one aggregate. For a detector that stares
  at nothing almost all the time, "what fraction of empty frames raise an alarm" is closer
  to the deployability metric — and it is the axis where YOLOv5l's extra capacity most
  clearly pays (0.022 vs 0.031).
- **No camera.** XP13's live demo becomes a file-based inference capture. No scientific cost.
- **XP2's retrain-at-resolution arm** is not done; it needs a GPU and is the arm that would
  confirm the train/test mismatch explanation for 512px beating 640px.

### 6.3 The line every technique must now beat

Not the 640px baseline. **YOLOv5s @ 512 — mAP50 0.7775, 178.7 img/s (batch 16), 8.48 W.**
It is *more* accurate than the same model at 640 while being 1.55× faster on 17% less
power, and it comes within 0.7 mAP points of YOLOv5l@640 at **5.4× the throughput and 2.3×
less power**. Any distilled, pruned or quantized model that lands below this line has not
earned its complexity — which is exactly the test §2's XP2 was written to impose, now with
numbers attached.

### 6.4 Open questions raised by measurement, not in the original plan

- **Tiny plumes barely work at any setting.** Best tiny-plume score in the whole sweep is
  0.1974 (YOLOv5l@640); YOLOv5s@640 gets 0.1654. A 6.6× larger model recovers 0.03.
  Distant-plume detection appears to be **neither a capacity nor a resolution problem** at
  these scales. Compression cannot fix this; tiling / sliced inference (SAHI-style) or a
  genuinely higher-resolution input are the levers that would, and neither is in this plan.
- **Are the fire/smoke and small/large asymmetries the same effect?** Fire boxes skew
  smaller than smoke boxes, so "fire is harder" and "small boxes are harder" may be one
  finding seen twice. Testable directly from the existing per-class, per-tier data.
