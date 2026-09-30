#!/usr/bin/env python3
"""What numeric formats can this board actually multiply? **Run on the Jetson.**

"Can the board do FP8?" hides three different questions, and they have different
answers:

1. **Can it represent it?**  Storage only. ``torch.float8_e4m3fn`` exists on any
   recent PyTorch regardless of hardware; it says nothing about arithmetic.
2. **Can it multiply it?**   Is there a tensor-core instruction in *this* GPU's
   ISA? This is the hardware question, and it is decidable: assemble a kernel
   that uses the instruction and see whether ``ptxas`` accepts it for this
   architecture.
3. **Will the runtime use it?**  TensorRT and cuDNN expose a subset of what the
   silicon can do. XP6-E4 is the cautionary tale: 2:4 sparsity was supported by
   the hardware, requested by us, and the compiler picked dense kernels anyway.

A format is only useful to this study when all three are true. This script
answers 1 and 2 directly and reports what the runtime advertises for 3; the only
honest answer to 3 is to build an engine and read back which kernels ran, which
is what ``e4_precision.py`` does.

**The control that makes the ISA probe trustworthy.** A ``ptxas`` failure can mean
"this architecture lacks the instruction" *or* "the operand shape in this probe is
wrong". To tell them apart, every instruction is assembled for this board **and**
for an architecture known to support it. Fails-here-passes-there is a hardware
answer; fails-everywhere is a bug in the probe and is reported as `INCONCLUSIVE`
rather than as absence.

Usage
    python probe_precision.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _calib                                                      # noqa: E402
from _calib import log                                             # noqa: E402

#: name -> (PTX instruction, operand template, an arch known to support it)
#:
#: Operand widths follow the PTX ISA's mma shapes. They differ per instruction,
#: and getting one wrong produces a syntax error that looks exactly like an
#: unsupported architecture — hence the control arch on every row.
PROBES = {
    "fp16 (f16 x f16 -> f32)": (
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32",
        "{%f0,%f1,%f2,%f3}, {%r0,%r1,%r2,%r3}, {%r4,%r5}, {%f4,%f5,%f6,%f7}",
        "sm_80"),
    "int8 (s8 x s8 -> s32)": (
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32",
        "{%r0,%r1,%r2,%r3}, {%r4,%r5,%r6,%r7}, {%r8,%r9}, {%r10,%r11,%r12,%r13}",
        "sm_80"),
    "int4 (s4 x s4 -> s32)": (
        "mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32",
        "{%r0,%r1}, {%r2}, {%r3}, {%r4,%r5}",
        "sm_80"),
    "binary (b1 x b1 -> s32, and.popc)": (
        "mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.and.popc",
        "{%r0,%r1}, {%r2}, {%r3}, {%r4,%r5}",
        "sm_80"),
    "fp8 E4M3 (e4m3 x e4m3 -> f32)": (
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32",
        "{%f0,%f1,%f2,%f3}, {%r0,%r1,%r2,%r3}, {%r4,%r5}, {%f4,%f5,%f6,%f7}",
        "sm_89"),
    "fp8 E5M2 (e5m2 x e5m2 -> f32)": (
        "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32",
        "{%f0,%f1,%f2,%f3}, {%r0,%r1,%r2,%r3}, {%r4,%r5}, {%f4,%f5,%f6,%f7}",
        "sm_89"),
}

PTX_TEMPLATE = """//
.version {ptx}
.target {arch}
.address_size 64

.visible .entry probe()
{{
  .reg .b32  %r<16>;
  .reg .f32  %f<16>;
  {instr} {operands};
  ret;
}}
"""


def ptxas() -> str | None:
    for c in ("ptxas", "/usr/local/cuda/bin/ptxas"):
        p = shutil.which(c) or (c if Path(c).exists() else None)
        if p:
            return p
    return None


def max_ptx_version(tool: str) -> str:
    """Highest PTX ISA this ``ptxas`` accepts.

    Not cosmetic: FP8 ``mma`` requires **PTX ISA 8.4 or later**, and a template
    pinned to an older version fails with a *feature* error that looks exactly
    like an unsupported architecture. The first run of this probe reported FP8 as
    INCONCLUSIVE for that reason — correctly, because the control arch failed too.
    """
    for v in ("8.7", "8.6", "8.5", "8.4", "8.3", "8.2", "8.0", "7.8"):
        ptx = PTX_TEMPLATE.format(ptx=v, arch="sm_80", instr="ret", operands="")
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "v.ptx"
            src.write_text(ptx.replace("ret ;", "ret;"))
            r = subprocess.run([tool, "--gpu-name=sm_80", str(src), "-o", str(Path(d) / "o")],
                               capture_output=True, text=True)
            if r.returncode == 0:
                return v
    return "7.8"


def assembles(tool: str, instr: str, operands: str, arch: str, ptx_ver: str) -> tuple[bool, str]:
    ptx = PTX_TEMPLATE.format(ptx=ptx_ver, arch=arch, instr=instr, operands=operands)
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "probe.ptx"
        src.write_text(ptx)
        r = subprocess.run([tool, f"--gpu-name={arch}", str(src), "-o", str(Path(d) / "o.cubin")],
                           capture_output=True, text=True)
        return r.returncode == 0, (r.stderr or "").strip().splitlines()[-1] if r.stderr else ""


def main() -> None:
    import torch

    p = torch.cuda.get_device_properties(0)
    arch = f"sm_{p.major}{p.minor}"
    log("probe", f"{p.name} — compute capability {arch}")

    tool = ptxas()
    if not tool:
        raise SystemExit("ptxas not found; install the CUDA toolkit or add it to PATH")

    ptx_ver = max_ptx_version(tool)
    log("probe", f"ptxas accepts PTX ISA up to .version {ptx_ver}")
    rows = {}
    log("probe", "--- 2. can the silicon multiply it? (PTX ISA, decisive) ---")
    for name, (instr, ops, control) in PROBES.items():
        here, err_here = assembles(tool, instr, ops, arch, ptx_ver)
        there, _ = assembles(tool, instr, ops, control, ptx_ver)
        if here:
            verdict, note = "YES", ""
        elif there:
            verdict, note = "NO", f"assembles for {control}, not for {arch}"
        else:
            verdict, note = "INCONCLUSIVE", f"fails for {control} too — probe bug: {err_here[:70]}"
        rows[name] = {"instruction": instr, "arch": arch, "ptx_version": ptx_ver,
                      "supported": verdict,
                      "control_arch": control, "control_assembles": there, "note": note}
        log("probe", f"  {verdict:12s} {name}" + (f"   ({note})" if note else ""))

    log("probe", "--- 1. can torch represent it? (storage only, proves nothing) ---")
    storage = {}
    for dt in ("float8_e4m3fn", "float8_e5m2", "int8", "quint4x2", "bfloat16"):
        storage[dt] = hasattr(torch, dt)
        log("probe", f"  {'yes' if storage[dt] else 'no ':3s}  torch.{dt}")

    log("probe", "--- 3. what does the runtime advertise? ---")
    import tensorrt as trt
    b = trt.Builder(trt.Logger(trt.Logger.ERROR))
    rt = {"tensorrt": trt.__version__,
          "platform_has_fast_fp16": bool(b.platform_has_fast_fp16),
          "platform_has_fast_int8": bool(b.platform_has_fast_int8)}
    for n in ("FP8", "INT4", "BF16"):
        rt[f"BuilderFlag.{n}"] = hasattr(trt.BuilderFlag, n)
        rt[f"DataType.{n}"] = hasattr(trt.DataType, n)
    for k, v in rt.items():
        log("probe", f"  {k} = {v}")
    log("probe", "  (advertised != used — only reading an engine's tactics settles it)")

    _calib.write_json("xp07_precision_support.json", {
        "experiment": "xp07_precision_support",
        "question": "which numeric formats can this board represent, multiply, and be made to use?",
        "gpu": p.name, "compute_capability": arch, "ptx_isa_version": ptx_ver,
        "isa_probe": rows, "torch_storage_dtypes": storage, "tensorrt": rt,
        "caveat": "ISA support is necessary, not sufficient. A format the silicon "
                  "multiplies but the runtime exposes no kernel for is unusable "
                  "without hand-written CUTLASS — XP6-E4's lesson, one level down.",
    })


if __name__ == "__main__":
    main()
