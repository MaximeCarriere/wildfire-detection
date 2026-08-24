# XP15 — The 5-euro sensor: can the gate leave the Jetson entirely?

> **Status:** stage A done — the gate **passes** the port rule. Nothing has run on the ESP32 yet.
> The question, method and decision rule below were written *before* the run, which is the
> convention XP5 set and the reason XP6's early numbers had to be withdrawn twice.

## Why

A fire watch stares at nothing almost all the time. [XP14](../../PLAN.md) makes that observation
and puts a cheap gate in front of the real detector on the *same* board. This asks the next
question: does the gate belong on the same board at all?

The target is a **XIAO ESP32-S3 Sense** — 240 MHz dual Xtensa LX7 with vector instructions,
8 MB PSRAM, 8 MB flash, OV2640 camera, about €27 and roughly a hundredth of an Orin's idle power.
If a device like that can say "something is happening, wake up", the architecture changes: a
handful of cheap always-on eyes and one Orin that sleeps.

## The question this actually asks

**Can a model small enough for a microcontroller catch a fire early enough to be worth waking
anything for?**

Not "does a CNN fit on an ESP32" — it does, and that part is not interesting. The interesting
part is whether anything useful survives at the resolution such a device can process.

## What is already known, before any work

[XP2](../xp02_resolution/) swept resolution with a **full** YOLOv5s:

| resolution | mAP50 | tiny plumes |
|---:|---:|---:|
| 416 px | 0.764 | 0.094 |
| 320 px | 0.725 | 0.038 |
| 256 px | 0.659 | 0.009 |
| 160 px | 0.458 | **0.0000** |

At 160 px a model orders of magnitude larger than anything that fits here already scores
**exactly zero** on tiny plumes. This board would run at ~96 px.

**Two conclusions follow, and both shape the design:**

- **Detection is off the table.** No boxes, no anchors, no NMS. The gate outputs one number.
- **Distant smoke is off the table too.** Early detection of a far-away plume is what a wildfire
  system is *for*, and this device cannot do it. What may survive is "there is obvious fire or
  smoke in view", which is a legitimate but much weaker capability, and the experiment has to be
  honest that this is what it is testing.

## Stage A — train it, on a workstation

`train_gate.py`. A depthwise-separable CNN at 96×96 grayscale, binary label derived from D-Fire
(any box → positive, background → negative). `--width` sets the channel multiplier: 0.25 gives
10k parameters (~10 KB int8), 1.0 gives roughly 160k (~160 KB), and the board's 8 MB of PSRAM
means size is not the binding constraint — recall is. Start wide and shrink only if the device
complains.

**Recall is reported stratified by plume size and never pooled.** A gate scoring 90% overall by
catching every wall of flame and no distant smoke is worse than useless: it fires when the fire
is already obvious. `lib.data` carries `has_tiny_plume` and `has_small_plume` per sample, so the
breakdown is free.

**Decision rule, fixed before the run:**

> Port to the device only if, at a false-wake rate ≤ 5% on background frames, recall on frames
> containing a **small** plume is ≥ 0.70.

Tiny plumes are deliberately *not* in the rule — at 96 px they are almost certainly unreachable,
and a bar nobody can clear proves nothing. They are measured and reported anyway, because that
number is the honest limit of the whole idea.

```bash
python experiments/xp15_esp32_gate/train_gate.py --epochs 30
python experiments/xp15_esp32_gate/train_gate.py --epochs 30 --res 128   # if 96 misses
```

**Where to run it.** On the Jetson, which is the only machine here holding the D-Fire images —
`data/` is gitignored, and this repo's Mac checkout carries `data/splits/` alone. The script picks
its device automatically (`cuda` → `mps` → `cpu`), so it also runs on an Apple laptop once the
3 GB dataset is unpacked locally; the model is far too small for compute to be the constraint.
Training the whole thing on the board is fine here precisely *because* the model is tiny — unlike
[XP7b](../xp06_pruning/HANDOFF_TO_RTX3090.md), where detector training on the Orin was ten times
slower than the 3090 and not batch-comparable.

## Stage A results

![What the gate wakes for](../../results/figures/xp15_gate.png)

**It passes, and the premise this page was built on was wrong.** I predicted distant smoke would
be unreachable, because XP2 showed a full YOLOv5s scoring 0.0000 mAP50 on tiny plumes at 160 px.
The gate reaches **96% recall on tiny-plume frames** at a loose threshold. The reason is that
classification is a far easier task than detection: mAP demands the box be localised, while the
gate only has to notice that something in the frame is smoky.

| | threshold 0.30 | **threshold 0.99** |
|---|---:|---:|
| false wakes on empty frames | 13.9% | **3.4%** |
| smoke | 93.7% | 68.9% |
| fire | 94.1% | 75.5% |
| both | 95.9% | 76.1% |
| recall, small plumes | 96.3% | **77.4%** |
| recall, tiny plumes | 96.0% | 69.5% |

**Threshold 0.99 is the shipping configuration** and clears the rule: ≤5% false wakes with ≥70%
recall on small plumes. Those are two different products and the figure shows both — one wakes on
nearly every fire and cries wolf on one empty frame in seven, the other is quiet enough to deploy
and sleeps through a quarter of real fires.

**One caveat that bounds the tiny-plume number.** `has_tiny_plume` means the frame contains at
least one box under the size threshold — it does not mean that is the *only* thing in frame. Many
of those frames also carry larger smoke, so 96% is recall on *frames containing* a tiny plume,
not on frames where a tiny plume is all there is. The honest version of that measurement needs a
tiny-only subset and has not been run.

## Stage B — put it on the device

Only if stage A passes. Two viable toolchains, and the choice is not obvious:

| | **ESP-DL** (Espressif) | **TFLite Micro + ESP-NN** |
|---|---|---|
| input | ONNX, which this repo already exports everywhere | needs ONNX → TF → TFLite conversion |
| quantization | its own int8 post-training flow | mature, well documented |
| S3 SIMD | native, it is Espressif's own | via the ESP-NN kernels |
| risk | thinner documentation | the conversion hop is the fragile part |

**Start with ESP-DL**, because `train_gate.py` already emits ONNX and the extra framework hop is
where this kind of port usually dies. Fall back to TFLite Micro if an op fails to lower.

Every operation in the model — 3×3 conv, depthwise conv, 1×1 conv, batch norm, ReLU, global
average pool, one linear — was chosen because it has an int8 kernel in **both**. An elegant
architecture that lowers to an unsupported op is worth nothing on this target.

**What to measure on the board**, in this order:

1. **It runs at all**, and the int8 model agrees with the float model on the test set. Quantization
   drift is measured, never assumed — this repo has withdrawn numbers over exactly that.
2. **Inference latency and peak RAM.** A few hundred ms is fine; a fire watch can sample every
   few seconds. RAM is the harder constraint despite the 8 MB PSRAM, since PSRAM bandwidth is far
   below internal SRAM and a layer that spills is much slower than its op count suggests.
3. **Average power over a duty cycle** — the number that decides whether the architecture is
   worth anything, against an always-on Orin at ~12 W. Reviews of this board note it runs hot
   under sustained load, so thermals are a real variable and not a footnote.

## What would count as a result

- **Pass, useful recall:** the architecture is validated — cheap always-on eyes, one sleeping
  Orin — and XP14's premise gets much stronger, since the gate no longer competes for the
  detector's silicon.
- **Pass on obvious fire, fail on small plumes:** the honest outcome, and probably the most
  likely one. It says this is a *confirmation* sensor, not an early-warning one, which is a
  different product and worth stating plainly rather than dressing up.
- **Fail outright:** also fine, and cheap. It bounds the whole idea with one afternoon of
  training and no firmware, which is precisely why stage A runs first.

## Notes

- The gate threshold defaults to **0.30**, deliberately low. A false wake costs one Orin
  inference; a missed fire costs the entire point of the system.
- Grayscale input, not RGB: smoke is grey by definition, and it cuts the first layer's work by
  three.
- Nothing here writes a `model_id`. This is not a detector and must not appear in the detector
  tables — the comparison that matters is against the *gate's* job, not against 0.7776 mAP50.
