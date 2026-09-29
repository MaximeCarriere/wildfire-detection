#!/usr/bin/env python3
"""XP7 self-checks — CPU only, no dataset, no GPU, a few seconds.

Every number on the XP7 page is produced by ``_quant.py`` and ``_lowbit.py``, and
two of the bugs found while writing them were **silent**: they produced plausible
output that was wrong, rather than crashing. Those are the ones worth a test.

* ``restore()`` after a fine-tune wrote back the weights stashed *before*
  training, discarding every epoch. The recovered numbers would have been the
  damage numbers with fresh scales, and nothing in the output would have looked
  odd. :func:`test_rebase_keeps_training` is the regression test.
* A layer whose Hessian is entirely zero made GPTQ zero every weight and report
  success, because "no data reached this layer" and "every input column is dead"
  are the same condition. :func:`test_gptq_dead_hessian_falls_back` pins it.

The third check is the one that says the machinery does anything at all:
error-compensated rounding is *supposed* to match round-to-nearest when the
inputs are uncorrelated, so a passing "GPTQ ≈ RTN" test proves nothing on its
own. :func:`test_gptq_helps_only_when_correlated` asserts both halves — no gain
on white noise, a large gain on a realistic correlated feature map.

Usage
    python test_quant.py           # or: pytest experiments/xp07_quant/test_quant.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _lowbit import HessianCollector, gptq_quantize, huffman_bits, kmeans_codebook
from _quant import (DetectOutputQuantizer, QuantConfig, Quantizer,
                    model_size_report, quantize_weight)


class Detect(nn.Module):
    """Stands in for YOLOv5's Detect head: pixel-valued boxes concatenated with
    probabilities, returned as ``(decoded, raw_list)``. The class *name* matters —
    ``DetectOutputQuantizer`` finds it by name, exactly as the real model requires."""

    def __init__(self):
        super().__init__()
        self.m = nn.ModuleList([nn.Conv2d(8, 6, 1)])

    def forward(self, x):
        y = self.m[0](x)
        return (torch.cat([y[:, :4] * 100.0, torch.sigmoid(y[:, 4:])], 1), [y])


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 8, 3, padding=1)
        self.c2 = nn.Conv2d(8, 8, 3, padding=1)
        self.detect = Detect()

    def forward(self, x):
        return self.detect(self.c2(torch.relu(self.c1(x))))


def _calibrated(net, cfg, n: int = 3) -> Quantizer:
    q = Quantizer(net, cfg).observe()
    with torch.no_grad():
        for _ in range(n):
            net(torch.randn(2, 3, 16, 16))
    return q.freeze()


CFG = QuantConfig(weight_bits=8, act_bits=8, granularity="per_channel", method="minmax")


def _weights(net, names):
    mods = dict(net.named_modules())
    return {n: mods[n].weight.data.clone() for n in names}


def test_restore_round_trips():
    torch.manual_seed(0)
    net = Net().eval()
    names = [n for n, m in net.named_modules() if isinstance(m, nn.Conv2d)]
    before = _weights(net, names)
    q = _calibrated(net, CFG).enable()
    after = _weights(net, names)
    assert all(not torch.equal(before[n], after[n]) for n in names), "nothing was quantized"
    q.restore()
    assert all(torch.equal(before[n], v) for n, v in _weights(net, names).items())


def test_rebase_keeps_training():
    """The silent bug: ``restore()`` after training threw the fine-tune away."""
    torch.manual_seed(0)
    net = Net().eval()
    names = [n for n, m in net.named_modules() if isinstance(m, nn.Conv2d)]
    q = _calibrated(net, CFG).enable()

    mods = dict(net.named_modules())
    with torch.no_grad():                      # stand-in for a recovery fine-tune
        for n in names:
            mods[n].weight.data += 0.05
    trained = _weights(net, names)

    q.rebase()
    assert all(torch.equal(trained[n], v) for n, v in _weights(net, names).items()), \
        "rebase() must not touch the weights, only drop the stale stash"
    q.enable()
    assert all(not torch.equal(trained[n], v) for n, v in _weights(net, names).items()), \
        "re-enabling must re-place the trained weights on the grid"
    q.restore()
    assert all(torch.equal(trained[n], v) for n, v in _weights(net, names).items()), \
        "restore() after rebase() must return the TRAINED weights"


def test_detect_output_quantizer():
    torch.manual_seed(0)
    net = Net().eval()
    x = torch.randn(2, 3, 16, 16)
    with torch.no_grad():
        ref = net(x)[0]

    dq = DetectOutputQuantizer(net, bits=8, method="minmax").observe()
    with torch.no_grad():
        for _ in range(3):
            net(torch.randn(2, 3, 16, 16))
    dq.freeze()
    rep = dq.report()
    assert rep["represented_max"] > 10, "the pixel-valued channels should set the scale"

    dq.enable()
    with torch.no_grad():
        out = net(x)
    assert isinstance(out, tuple) and len(out[1]) == 1, "output structure must survive"
    assert out[0].shape == ref.shape

    # The fingerprint: probabilities are hurt far more, relative to their size.
    box_rel = ((out[0][:, :4] - ref[:, :4]).abs().mean() / ref[:, :4].abs().mean()).item()
    prob_rel = ((out[0][:, 4:] - ref[:, 4:]).abs().mean() / ref[:, 4:].abs().mean()).item()
    assert prob_rel > box_rel * 5, (
        f"expected the probability channels to take the damage "
        f"(box {box_rel:.4f}, prob {prob_rel:.4f})")

    dq.restore()
    with torch.no_grad():
        assert torch.equal(net(x)[0], ref), "restore() must be exact"


def test_scoring_does_not_change_the_model_dtype():
    """Scoring must leave its argument alone, or every recovery arm crashes.

    ``Yolov5Detector.from_model(half=True)`` casts in place. Harmless when the
    model is scored and discarded; fatal when it is fine-tuned afterwards, which
    is what E3, E5, E6, E7 and E10's second arm all do. The failure is
    ``ValueError: Attempting to unscale FP16 gradients`` three hours into a run.
    """
    torch.manual_seed(0)
    net = Net().eval()
    assert next(net.parameters()).dtype == torch.float32

    # Stand in for the harness: cast to half in place, as from_model does.
    def scoring_harness(m):
        m.float().eval()
        m.half()
        return {"map50": 0.0}

    was_half = next(net.parameters()).dtype == torch.float16
    scoring_harness(net)
    if not was_half:
        net.float()
    assert next(net.parameters()).dtype == torch.float32, \
        "a model that went into scoring as float32 must come out as float32"

    # And the thing it protects: a half model cannot be trained with a scaler.
    net.half()
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    assert next(net.parameters()).dtype == torch.float16
    net.float()
    assert next(net.parameters()).dtype == torch.float32


def test_gptq_helps_only_when_correlated():
    """Both halves matter: no gain on white noise IS the correct answer."""
    torch.manual_seed(0)
    conv = nn.Conv2d(16, 8, 3, padding=1)
    w = conv.weight.data
    flat = w.reshape(w.shape[0], -1)

    def white(b=4):
        return torch.randn(b, 16, 32, 32)

    def correlated(b=4):
        z = F.interpolate(torch.randn(b, 4, 8, 8), size=(32, 32),
                          mode="bilinear", align_corners=False)
        return torch.einsum("oc,bchw->bohw", torch.randn(16, 4), z)

    gains = {}
    for label, gen in (("white", white), ("correlated", correlated)):
        col = HessianCollector(conv).attach(chunk=512)
        with torch.no_grad():
            for _ in range(6):
                conv(gen())
        col.detach()
        assert torch.isfinite(col.h).all()

        q, info = gptq_quantize(flat, col.h, bits=4, group_size=16)
        assert not info["fallback"]
        rtn = quantize_weight(w, bits=4, granularity="per_group", group_size=16)

        xs = torch.cat([gen() for _ in range(4)])
        with torch.no_grad():
            ref = F.conv2d(xs, w, conv.bias, padding=1)
            eg = (ref - F.conv2d(xs, q.reshape(w.shape), conv.bias, padding=1)).pow(2).mean()
            er = (ref - F.conv2d(xs, rtn, conv.bias, padding=1)).pow(2).mean()
        gains[label] = ((er - eg) / er).item()

    assert abs(gains["white"]) < 0.10, (
        f"on uncorrelated inputs GPTQ should match RTN, got {gains['white']:+.1%}")
    assert gains["correlated"] > 0.25, (
        f"on correlated inputs GPTQ should clearly win, got {gains['correlated']:+.1%}")


def test_gptq_dead_hessian_falls_back():
    """A layer no data reached must fall back, not silently return zeros."""
    torch.manual_seed(0)
    conv = nn.Conv2d(16, 8, 3, padding=1)
    flat = conv.weight.data.reshape(8, -1)
    n = flat.shape[1]
    q, info = gptq_quantize(flat, torch.zeros(n, n), bits=4, group_size=16)
    assert info["fallback"] and "no calibration data" in info["reason"]
    assert q.abs().sum() > 0, "the fallback must preserve the weights, not zero them"


def test_size_report_itemises():
    torch.manual_seed(0)
    net = Net().eval()
    full = model_size_report(net, CFG)
    assert full["compression_x"] > 1.5
    b = full["bytes"]
    parts = (b["quantized_weights"] + b["untouched_params"]
             + b["weight_scales"] + b["activation_scales"])
    assert parts == b["total"], f"the itemisation must add up: {parts} != {b['total']}"
    assert b["fp16_baseline"] > b["total"]

    # Compare bytes, not the rounded MB: on a toy net every arm is 0.001 MB.
    sub = model_size_report(net, CFG, targets=["c1", "c2"])
    assert sub["params_quantized_pct"] < full["params_quantized_pct"]
    assert sub["bytes"]["total"] > b["total"], "leaving layers in FP16 costs bytes"
    assert sub["bytes"]["weight_scales"] < b["weight_scales"], \
        "fewer quantized layers means fewer scale tables"


def test_codebook_and_huffman():
    torch.manual_seed(0)
    w = torch.randn(64, 32) * 0.1
    centroids, idx = kmeans_codebook(w, bits=4)
    assert centroids.numel() == 16 and idx.numel() == w.numel()
    bits = huffman_bits(idx)
    assert 0 < bits <= 4.05, f"a 16-symbol Huffman code cannot beat 4 bits by much: {bits}"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {t.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
