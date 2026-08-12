# XP2 — Resolution: the null hypothesis

**Question:** input resolution is the cheapest knob available — no retraining, no tooling.
How much of the compression story does it already explain?

**Outcome:** more than expected. **512 pixels is both more accurate and 1.6× faster than
640.** A free win, before any sophisticated technique was tried.

![Shrinking the input is free speed](../../results/figures/xp02_resolution.png)

### The same thing, on one frame

![The same frame at four resolutions](../../results/figures/xp02_resolution_visual.png)

The distant plume is found at 640 and 512 pixels, weakly at 416, and missed entirely at 320
— while the overall accuracy score barely moves. That gap is the whole finding.

## The frontier

Published weights, evaluated at resolutions they were never trained for.

| model | resolution | accuracy | small plumes | tiny plumes | speed | power |
|---|---:|---:|---:|---:|---:|---:|
| YOLOv5s | 640 | 0.7708 | 0.6365 | 0.1654 | 115 img/s | 10.3 W |
| **YOLOv5s** | **512** | **0.7775** | 0.6062 | 0.1386 | **179 img/s** | **8.5 W** |
| YOLOv5s | 416 | 0.7635 | 0.5466 | 0.0942 | 246 img/s | 7.7 W |
| YOLOv5s | 320 | 0.7246 | 0.4459 | 0.0384 | 404 img/s | 7.0 W |
| YOLOv5l | 640 | 0.7847 | 0.6410 | 0.1974 | 33 img/s | 19.2 W |

> **What "plume" means here.** A plume is the visible smoke or flame region the detector has
> to find. Accuracy is reported separately for **small plumes** (under 1% of the frame) and
> **tiny plumes** (under 0.1%, roughly 20x20 pixels) — distant smoke, which is what early
> detection actually depends on.

![What small and tiny plume mean](../../results/figures/plume_definition.png)

## What this means

**The cheap knob beats the expensive model.** YOLOv5s at 512 comes within 0.7 accuracy
points of the 6.6×-larger YOLOv5l at 640, while running **5.4× faster on half the power**.
Every technique in the rest of the project now has to beat *this* line, not the 640px
baseline.

**Overall accuracy hides the damage.** Dropping 640 → 320 costs 6% of overall accuracy but
**77% of tiny-plume accuracy**. Judged on the headline number alone, the honest conclusion
would be "resolution is nearly free" — which is exactly backwards for a system whose job is
spotting distant smoke early.

**Bigger doesn't fix small targets.** Tiny plumes score 0.17–0.20 at *every* setting
measured. A 6.6× larger model recovers almost nothing. Distant-plume detection appears to
be neither a capacity nor a resolution problem at these scales — which means no compression
technique can address it either.

## Limitations

- Inference-resolution only. The models were trained at one resolution and tested at
  others; retraining at each resolution would likely widen the 512px win, and needs a GPU.
- The 512 > 640 result is probably train/test resolution mismatch, but that is a hypothesis
  until the retraining arm is run.

## Next

XP9 shows this entire frontier was measured on a runtime that was hiding the real speeds.
