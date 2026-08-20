# Handing E9 to the RTX 3090

This is one experiment, `e9_frontier.py`, and it needs about **7 hours of GPU time**. The
board cannot run it: the Orin trains at roughly 20 img/s against the 3090's 172, and it caps at
batch 8 in its 7 GB where every published XP6 number used batch 32. A board-trained result would
be ten times slower to get *and* not comparable to the table it has to join.

Read it with [`README.md`](README.md), and with E7, which is the experiment this fixes.

## What this is for

E7 compared one-shot against iterative channel pruning and found iterative losing by 1.4 points.
That comparison is sound, but it was taken at **one ratio**: 25% of channels, which is 40.1% of
the parameters.

The reference result everyone cites for iterative pruning (Han et al., *Deep Compression*) plots
that same comparison as a curve from 40% to 95% of parameters removed. On it, the one-shot and
iterative curves lie **on top of each other until roughly 90%** and separate only past it. E7's
single point sits at 40% — in the flat region, where the literature predicts no difference at
all.

So E7 did not contradict the textbook. It sampled the one part of the axis least able to
distinguish the two arms. **E9 sweeps the whole axis** so the claim can be made properly, in
either direction.

There is a second reason to want this curve. E5 and E6 both found that 12 epochs of recovery
erase large structural differences on this detector — E6's three allocations span 0.158 mAP50 as
damage and 0.004 after retraining. If iterative pruning helps anywhere, it helps where recovery
*cannot* absorb the damage, and that is the far right of this curve. Nothing on this page has
looked there yet.

## Environment

Match the existing screening records, since every number this produces has to sit in the same
tables:

| | value | why it matters |
|---|---|---|
| torch | 2.13.0+cu126 | as recorded in the existing `train_meta` |
| torch-pruning | 1.6.0 | pinned; the pruning API is version-sensitive |
| batch (recovery) | **32** | every published XP6 number used 32. Do not change it. |
| resolution | 512 | `_screen.RES` |
| criterion | L1 | set in the script, see below |

`lib/finetune.py` accumulates gradients to a nominal batch of 64, so a different *real* batch
changes the number of accumulation steps rather than the effective update — but it still changes
BatchNorm statistics, which is why 32 is not negotiable here.

## Running it

Confirm the plan first. It cuts at every ratio and reports the achieved size without training
anything, which takes a couple of minutes and catches a broken environment before you spend
seven hours on one:

```bash
python experiments/xp06_pruning/e9_frontier.py --plan
```

Expected output, already verified on the Jetson:

```
 channel cut    params  params cut  MACs cut
        15%     5.293      24.66%    24.35%
        25%     4.205      40.14%    38.73%
        35%     3.129      55.46%    53.44%
        45%     2.223      68.36%    65.92%
        55%     1.469      79.08%    76.67%
        65%     0.890      87.33%    85.40%
        75%     0.465      93.39%    91.88%
        80%     0.315      95.52%    93.69%
```

If the parameter column differs, stop — the pruning library is behaving differently and nothing
downstream will be comparable.

Then the three arms, in this order:

```bash
python experiments/xp06_pruning/e9_frontier.py --arm damage      # ~25 min
python experiments/xp06_pruning/e9_frontier.py --arm oneshot     # ~2.7 h
python experiments/xp06_pruning/e9_frontier.py --arm iterative   # ~4.4 h
```

Damage first, deliberately. It is cheap, it needs no training, and it tells you where the curve
falls off — if channel pruning is already at 0.0000 by 55% removed, that is worth knowing before
committing four hours to the iterative arm.

Each arm writes one JSON per ratio and is safe to interrupt. Resume with an explicit subset:

```bash
python experiments/xp06_pruning/e9_frontier.py --arm oneshot --ratios 0.55 0.65 0.75 0.80
```

## Choices already made, and why

Do not change these without a reason, because each one is holding a confound still:

- **Criterion is L1, not E7's L2.** E7 held L2 because it was rerunning a specific published
  comparison and changing the rule would have fixed a different experiment. E9 has no such
  obligation, and E2 established L2 is a poor rule on this detector. An iterative arm applies
  the criterion once per step, so a bad rule is applied four times instead of once — running the
  frontier on L2 would confound "iterative does not help" with "the criterion was wrong, more
  often".
- **`round_to=1`, no width rounding.** E3's rounding is free accuracy and 1.77x the speed, but
  it changes the achieved size, and this experiment's x-axis *is* achieved size. Apply rounding
  after picking a point on this curve, not while measuring it.
- **Post-cut epochs held equal at 12.** This is E7's fix and it is the thing that makes the
  comparison fair. Iterative additionally keeps its 8 between-step epochs, so it receives 20
  total epochs against one-shot's 12 — more training, deliberately, so that a loss cannot be
  blamed on budget.
- **Plotted against measured parameter reduction, never the channel ratio.** A 25% channel cut
  removes 40.1% of the parameters. E5 and E6 both had to make this correction before their
  comparisons meant anything.

## Handing the results back

Commit the raw JSONs — that is the whole handover, there are no weights worth moving:

```
results/raw/xp06e9_damage.json                      the unretrained series
results/raw/xp06e9_dfire_yolov5s_frontier_*.json    one per (arm, ratio)
```

`analysis/make_figures.py` already has `fig_xp06e9` and builds
`results/figures/xp06e9_frontier.png` from them. It renders as soon as the damage arm exists and
fills in as the trained arms land, so you can look at the curve before the run finishes. It
marks E7's single measurement point on the x-axis, which is the whole reason the sweep exists.

**One accuracy caveat, unchanged from the Jetson handoff.** mAP50 on the 3090 matches the
board's to about three decimal places, not bit exactly: the gate check reproduced the board's
unpruned 0.7775 as 0.7764. The damage arm re-scores the unpruned model in its own run and stores
it as `unpruned_here`, so its deltas hold on whichever machine ran it. The trained arms are read
against the 0.7764 screening reference.

## What the answer would mean

- **The curves separate above 90%, iterative on top.** The literature is right and E7 simply
  measured in the flat region. E7's section needs rewriting to say "no difference where none was
  predicted" instead of "iterative still loses", and iterative pruning becomes the right choice
  for aggressive compression on this detector.
- **The curves stay together to 95%.** Then E7's finding generalises and is much stronger than
  it currently is: iterative pruning does not pay for itself on this network *anywhere*, and the
  extra training it costs is wasted. That is a genuine disagreement with the literature and
  should be stated as one.
- **Both collapse before 90%.** Then this detector cannot be channel-pruned into the regime
  where the question is even meaningful, which is its own result and consistent with E5 — where
  7% of the model removed as channels already cost most of the accuracy.

All three are publishable. The one outcome that would be a problem is not running it, because
E7's section currently reads as a contradiction of the literature on the strength of a single
point taken where no contradiction was available.
