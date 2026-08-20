#!/usr/bin/env python3
"""XP6 E9 — can the pruning ratio be searched automatically against measured latency?

Every allocation this study has tried is hand-designed. E6 compared three of them
(uniform, global, sensitivity-driven) and found them 0.4 accuracy points apart
after recovery. The lecture's answer to "stop guessing" is automatic search, and
its rule-based method is **NetAdapt**: pick a per-layer pruning ratio by measuring
what each candidate cut actually saves, using a pre-built table of layer latency
versus channel count.

NetAdapt is the right method to want here. XP6's recurring finding is that
parameters and MACs do not predict speed on this board -- E3 measured a 3.52 M
model running 1.77x faster than a 4.21 M one -- and NetAdapt is the one method in
the lecture that optimises measured latency instead of a proxy.

**This experiment tests the cost model, not the search.** The search is
straightforward to implement; it is worthless if the table it consumes cannot
represent what the hardware does. Three stages, cheapest first, each able to kill
the method on its own:

1. **Shape.** Sweep one real layer across output-channel counts. If latency is flat
   in channel count there is nothing to optimise.
2. **Noise.** Build the same layer three times. The search ranks candidates whose
   predicted savings differ by a few percent, so the table's own repeatability
   bounds how fine a width grid can be searched at all.
3. **Slope.** The one that matters. NetAdapt never uses absolute latency, only the
   *difference* a cut makes, so a constant per-layer overhead cancels and a wrong
   total is survivable. A wrong slope is not. E3 already built four real engines
   from one 25% cut at different width-rounding and measured each on this board,
   which is ground truth this experiment can be scored against without training
   anything.

``round_to=1`` is the decisive case. It deletes 40% of the parameters and the real
engine gets **slower**, 472.6 to 362.9 img/s. A per-layer table can only ever say
that fewer channels cost less time. If it predicts a saving where the hardware
delivers a loss, NetAdapt's search would walk confidently toward the slowest model
on offer -- not because the search is wrong, but because the cost model it consumes
cannot express what this hardware does.

Board-only, and no training anywhere. Writes one side-data JSON with no
``model_id``.

Usage
    python e9_netadapt.py --stage shape     # ~5 min
    python e9_netadapt.py --stage noise     # ~2 min
    python e9_netadapt.py --stage slope     # ~45 min, includes the baseline table
    python e9_netadapt.py --stage all
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import onnx                                     # noqa: E402
import torch                                    # noqa: E402
import torch.nn as nn                           # noqa: E402

RAW = REPO / "results" / "raw"
WEIGHTS = REPO / "weights"
CACHE = Path("/tmp/xp06e9_lut")
TRTEXEC = "/usr/src/tensorrt/bin/trtexec"
RES = 512
BATCH = 16
TAG = "e9"

#: Measured on this board by E3b and XP9, images per second at batch 16. The
#: ground truth the LUT is scored against; nothing here re-measures them.
REAL_FPS = {"unpruned": 472.6, "round1": 362.9, "round8": 532.3,
            "round16": 622.4, "round32": 641.9}


def log(m: str) -> None:
    print(f"[{TAG}] {m}", flush=True)


class OneConv(nn.Module):
    """Conv + BN + SiLU: the unit TensorRT fuses into a single kernel here.

    Timing a bare Conv2d would measure something the real engine never runs, since
    every convolution in this detector is followed by BN and SiLU and the builder
    folds all three together.
    """

    def __init__(self, cin, cout, k, stride):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride, k // 2, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


def time_layer(cin, cout, k, stride, hw, rep=0):
    """Mean GPU ms for one fused conv, built and measured alone. None on failure."""
    CACHE.mkdir(exist_ok=True)
    tag = f"c{cin}_{cout}_k{k}s{stride}_{hw}_r{rep}"
    onnx_p, eng = CACHE / f"{tag}.onnx", CACHE / f"{tag}.engine"
    if not eng.exists():
        m = OneConv(cin, cout, k, stride).eval()
        torch.onnx.export(m, torch.randn(BATCH, cin, hw, hw), str(onnx_p),
                          input_names=["x"], output_names=["y"], opset_version=17)
        subprocess.run([TRTEXEC, f"--onnx={onnx_p}", f"--saveEngine={eng}",
                        "--fp16", "--skipInference"], capture_output=True, text=True)
        if not eng.exists():
            return None
    r = subprocess.run([TRTEXEC, f"--loadEngine={eng}", "--iterations=200",
                        "--warmUp=200", "--avgRuns=100", "--noDataTransfers",
                        "--useSpinWait"], capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if "GPU Compute Time" in line and "mean" in line:
            for part in line.split(","):
                if "mean" in part:
                    return float(part.split("=")[1].strip().split()[0])
    return None


def conv_shapes(path: Path, res: int = RES):
    """Every Conv in an exported arm, with the spatial size it actually sees.

    The arms were exported with dynamic axes, so the graph carries no spatial sizes
    and inference returns zeros. Pinning the input to one concrete shape first is
    what makes the rest of the graph resolvable.
    """
    m = onnx.load(str(path))
    d = m.graph.input[0].type.tensor_type.shape.dim
    for i, v in enumerate((1, 3, res, res)):
        d[i].ClearField("dim_param")
        d[i].dim_value = v
    del m.graph.value_info[:]
    m = onnx.shape_inference.infer_shapes(m)

    dims = {}
    for vi in list(m.graph.value_info) + list(m.graph.input) + list(m.graph.output):
        dd = vi.type.tensor_type.shape.dim
        if len(dd) == 4:
            dims[vi.name] = [x.dim_value for x in dd]
    init = {i.name: i for i in m.graph.initializer}

    out = []
    for node in m.graph.node:
        if node.op_type != "Conv" or node.input[1] not in init:
            continue
        w = init[node.input[1]].dims                  # [cout, cin/groups, kh, kw]
        hw = dims.get(node.input[0], [0, 0, 0, 0])[-1]
        stride = next((a.ints[0] for a in node.attribute if a.name == "strides"), 1)
        if hw and w[1] > 0:
            out.append({"cin": w[1], "cout": w[0], "k": w[2],
                        "stride": stride, "hw": hw})
    return out


def stage_shape() -> dict:
    """Is layer latency informative in channel count at all?"""
    CIN, K, S, HW = 128, 1, 1, 64          # model.4.cv3.conv, mid-network, 512 px
    widths = [16, 24, 32, 40, 48, 52, 56, 64, 72, 80, 96, 97, 104, 112, 128]
    log(f"shape: conv{K}x{K} s{S}, {CIN} in, {HW}x{HW}, batch {BATCH}")
    pts = []
    for w in widths:
        ms = time_layer(CIN, w, K, S, HW)
        if ms is None:
            continue
        pts.append({"out_ch": w, "ms": ms})
        log(f"  {w:4d} ch -> {ms:7.4f} ms ({ms / w * 1000:6.2f} us/ch)")
    return {"cin": CIN, "k": K, "stride": S, "hw": HW, "points": pts}


def stage_drift(prev: dict | None) -> dict:
    """Re-time the *same cached engines* and compare against an earlier pass.

    Stage 2 rebuilds a layer to measure build-to-build variation, which turned out
    to be small. This measures something different and much larger: the same engine
    file, already built, timed again later. No compilation is involved, so whatever
    moves is the board -- clocks, thermals, whatever else was scheduled.

    It matters more than the build noise because of how NetAdapt consumes the table.
    The search does not use the total; it compares one layer's entry against
    another's to decide where to spend the next cut. Entry-level instability is
    therefore the number that bounds the method, and the total being steady is no
    consolation.
    """
    if not prev or not prev.get("points"):
        log("drift: no earlier shape sweep stored, run --stage shape first")
        return {}
    log("drift: re-timing the same engines from the earlier sweep")
    rows, worst = [], 0.0
    for pt in prev["points"]:
        ms = time_layer(prev["cin"], pt["out_ch"], prev["k"], prev["stride"], prev["hw"])
        if ms is None:
            continue
        pct = (ms / pt["ms"] - 1) * 100
        worst = max(worst, abs(pct))
        rows.append({"out_ch": pt["out_ch"], "first_ms": pt["ms"],
                     "again_ms": ms, "drift_pct": round(pct, 2)})
        log(f"  {pt['out_ch']:4d} ch: {pt['ms']:.4f} -> {ms:.4f} ms ({pct:+.1f}%)")
    log(f"  worst entry drift {worst:.0f}% with no rebuild")
    return {"worst_drift_pct": round(worst, 1), "entries": rows}


def stage_noise() -> dict:
    """How repeatable is one table entry? This bounds the usable width grid."""
    log("noise: same layer, three independent builds")
    out = []
    for cout in (56, 64):
        ts = [t for t in (time_layer(128, cout, 1, 1, 64, rep=r) for r in range(3)) if t]
        if len(ts) < 2:
            continue
        spread = (max(ts) / min(ts) - 1) * 100
        out.append({"out_ch": cout, "ms": ts, "spread_pct": round(spread, 2)})
        log(f"  {cout} ch: {['%.4f' % t for t in ts]} -> spread {spread:.1f}%")
    return {"repeats": out}


def lut_sum(path: Path, label: str) -> tuple[float, int]:
    shapes = conv_shapes(path)
    total, miss = 0.0, 0
    for s in shapes:
        ms = time_layer(s["cin"], s["cout"], s["k"], s["stride"], s["hw"])
        if ms is None:
            miss += 1
            continue
        total += ms
    log(f"  {label:10s} {len(shapes):3d} convs ({miss} failed) -> LUT {total:7.2f} ms")
    return total, len(shapes)


def stage_slope() -> dict:
    """Does the table predict the saving a real cut delivers? The decisive stage."""
    log("slope: LUT-predicted saving vs the saving E3's real engines delivered")

    # The unpruned baseline comes from the live model rather than an export,
    # because the board keeps a .onnx per pruned arm but none for the original.
    from lib.detectors import YOLOV5_REPO
    from lib.prune_utils import load_yolov5
    model = load_yolov5(WEIGHTS / "yolov5s.pt", YOLOV5_REPO, device="cpu")
    seen, hooks = [], []

    def hook(mod, inp, _out):
        seen.append({"cin": mod.in_channels, "cout": mod.out_channels,
                     "k": mod.kernel_size[0], "stride": mod.stride[0],
                     "hw": inp[0].shape[-1]})

    for mod in model.modules():
        if isinstance(mod, nn.Conv2d):
            hooks.append(mod.register_forward_hook(hook))
    with torch.no_grad():
        model(torch.randn(1, 3, RES, RES))
    for h in hooks:
        h.remove()
    del model

    base_lut = sum(t for t in (time_layer(s["cin"], s["cout"], s["k"], s["stride"],
                                          s["hw"]) for s in seen) if t)
    n = len(seen)
    log(f"  {'unpruned':10s} {n:3d} convs -> LUT {base_lut:7.2f} ms")

    base_real = BATCH / REAL_FPS["unpruned"] * 1000
    arms = []
    for name in ("round1", "round32"):
        p = WEIGHTS / f"dfire_yolov5s_{name}.onnx"
        if not p.exists():
            log(f"  missing {p.name}")
            continue
        lut, _ = lut_sum(p, name)
        real = BATCH / REAL_FPS[name] * 1000
        pred, actual = base_lut - lut, base_real - real
        ok = (pred > 0) == (actual > 0)
        arms.append({"arm": name, "lut_ms": round(lut, 3), "real_ms": round(real, 3),
                     "predicted_saving_ms": round(pred, 3),
                     "actual_saving_ms": round(actual, 3), "sign_agrees": ok})
        log(f"  {name:10s} predicted {pred:+7.2f} ms · actual {actual:+7.2f} ms · "
            f"{'sign OK' if ok else 'SIGN WRONG'}")
    return {"unpruned_lut_ms": round(base_lut, 3),
            "unpruned_real_ms": round(base_real, 3),
            "n_convs": n, "ratio_lut_over_real": round(base_lut / base_real, 3),
            "arms": arms}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["shape", "noise", "drift", "slope", "all"],
                    default="all")
    args = ap.parse_args()

    path = RAW / "xp06e9_netadapt_lut.json"
    rec = json.loads(path.read_text()) if path.exists() else {}
    rec |= {"experiment": "xp06e9_netadapt_lut", "input_res": RES, "batch": BATCH,
            "machine": "jetson_orin_nano",
            "notes": (
                "XP6 E9: whether NetAdapt's per-layer latency lookup table is a valid cost "
                "model on this target. Three stages: the shape of layer latency in channel "
                "count, the table's own build-to-build repeatability, and whether the table "
                "predicts the saving E3's real engines actually delivered. No search is "
                "implemented and no model is trained: the search is only worth writing if "
                "the table it consumes can represent what this hardware does. Scored against "
                "E3b's measured throughputs rather than re-measuring them. No model_id: side "
                "data for the E9 figure, not a results record.")}

    if args.stage in ("shape", "all"):
        rec["shape"] = stage_shape()
    if args.stage in ("noise", "all"):
        rec["noise"] = stage_noise()
    if args.stage in ("drift", "all"):
        rec["drift"] = stage_drift(rec.get("shape"))
    if args.stage in ("slope", "all"):
        rec["slope"] = stage_slope()

    path.write_text(json.dumps(rec, indent=2) + "\n")
    log(f"wrote {path.name}")


if __name__ == "__main__":
    main()
