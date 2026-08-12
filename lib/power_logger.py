"""Power / thermal logging via ``tegrastats`` (PLAN.md §0: on-device numbers only).

Ported from jetson-xray-panel/lib/power_logger.py — same board (Orin Nano Super,
JetPack R36.4.3), same telemetry, so the parsing is unchanged and power numbers
stay comparable across the two repos. ``tegrastats`` is spawned as a subprocess,
read in a background thread, and each sample is stamped with
``time.perf_counter()`` so a benchmark can align the power window to the exact
steady-state inference window it measured.

Sample line (Orin Nano Super, JetPack R36.4.3)::

    ... GR3D_FREQ 0% ... gpu@54.0C tj@54.0C ... VDD_IN 6357mW/6449mW
    VDD_CPU_GPU_CV 1458mW/1497mW VDD_SOC 2250mW/2262mW

Field meanings:
  * ``VDD_IN``          total module power draw (the honest "what the box pulls").
  * ``VDD_CPU_GPU_CV``  the combined CPU+GPU+CV compute rail (isolates compute).
  * ``GR3D_FREQ``       GPU utilization %.
  * ``tj``              junction temperature (watch for thermal throttling).

Each rail is reported as ``<instant>mW/<running_avg>mW``; we keep the instant
value and do our own mean/peak aggregation over the aligned window.

XP14 (the cascade gate) turns on ``energy_joules`` rather than mean power: a fire
watch that sleeps is judged on average watts over a long idle stream, not on the
power drawn while it happens to be inferring.
"""
from __future__ import annotations

import re
import subprocess
import threading
import time
from dataclasses import dataclass, field

# Defensive regexes — fields can vary across JetPack versions, so every parse is
# optional and missing fields simply stay None for that sample.
_RE_GR3D = re.compile(r"GR3D_FREQ\s+(\d+)%")
_RE_VDD_IN = re.compile(r"VDD_IN\s+(\d+)mW")
_RE_VDD_CGC = re.compile(r"VDD_CPU_GPU_CV\s+(\d+)mW")
_RE_TJ = re.compile(r"tj@([\d.]+)C")


@dataclass
class Sample:
    t: float                       # perf_counter timestamp
    vdd_in_mw: float | None = None
    vdd_cgc_mw: float | None = None
    gr3d_pct: float | None = None
    tj_c: float | None = None


@dataclass
class PowerLogger:
    """Context manager that records tegrastats samples for the duration of a run.

    Usage::

        with PowerLogger() as p:
            t0 = time.perf_counter()
            ...run inference...
            t1 = time.perf_counter()
        stats = p.summary(t0, t1)   # aggregate over just the measured window
    """
    interval_ms: int = 100
    samples: list[Sample] = field(default_factory=list)
    _proc: subprocess.Popen | None = None
    _thread: threading.Thread | None = None
    _stop: threading.Event = field(default_factory=threading.Event)

    def _reader(self) -> None:
        assert self._proc and self._proc.stdout
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            s = Sample(t=time.perf_counter())
            if (m := _RE_VDD_IN.search(line)):
                s.vdd_in_mw = float(m.group(1))
            if (m := _RE_VDD_CGC.search(line)):
                s.vdd_cgc_mw = float(m.group(1))
            if (m := _RE_GR3D.search(line)):
                s.gr3d_pct = float(m.group(1))
            if (m := _RE_TJ.search(line)):
                s.tj_c = float(m.group(1))
            self.samples.append(s)

    def __enter__(self) -> "PowerLogger":
        self._proc = subprocess.Popen(
            ["tegrastats", "--interval", str(self.interval_ms)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        # Give tegrastats a moment to emit its first sample before the caller times.
        time.sleep(0.3)
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._thread:
            self._thread.join(timeout=2)

    def energy_joules(self, t_start: float | None = None, t_end: float | None = None) -> float:
        """Integrate total board power (VDD_IN) over [t_start, t_end] -> Joules.
        Trapezoidal over the samples (energy = mean-power × elapsed)."""
        win = [
            s for s in self.samples
            if s.vdd_in_mw is not None
            and (t_start is None or s.t >= t_start) and (t_end is None or s.t <= t_end)
        ]
        if len(win) < 2:
            return 0.0
        e = 0.0
        for a, b in zip(win, win[1:]):
            e += (a.vdd_in_mw + b.vdd_in_mw) / 2 / 1000.0 * (b.t - a.t)   # W·s = J
        return e

    def latest(self) -> dict:
        """Most recent sample as plain values, for live readouts (None if no data)."""
        if not self.samples:
            return {"power_w": None, "gpu_util_pct": None, "temp_c": None}
        s = self.samples[-1]
        return {
            "power_w": None if s.vdd_in_mw is None else round(s.vdd_in_mw / 1000, 1),
            "gpu_util_pct": s.gr3d_pct,
            "temp_c": s.tj_c,
        }

    def summary(self, t_start: float | None = None, t_end: float | None = None) -> dict:
        """Aggregate power/util/temp over samples in [t_start, t_end].

        Returns a dict ready to drop into the results JSON schema:
        ``power_w`` (total, from VDD_IN), ``compute_power_w`` (VDD_CPU_GPU_CV),
        ``gpu_util_pct``, ``temp_c_peak``, and the raw ``n_samples``.
        """
        win = [
            s for s in self.samples
            if (t_start is None or s.t >= t_start) and (t_end is None or s.t <= t_end)
        ]

        def agg(vals):
            vals = [v for v in vals if v is not None]
            if not vals:
                return {"mean": None, "peak": None}
            return {"mean": sum(vals) / len(vals), "peak": max(vals)}

        vdd_in = agg([s.vdd_in_mw for s in win])
        vdd_cgc = agg([s.vdd_cgc_mw for s in win])
        gr3d = [s.gr3d_pct for s in win if s.gr3d_pct is not None]
        tj = [s.tj_c for s in win if s.tj_c is not None]

        return {
            "power_w": {
                "mean": None if vdd_in["mean"] is None else round(vdd_in["mean"] / 1000, 3),
                "peak": None if vdd_in["peak"] is None else round(vdd_in["peak"] / 1000, 3),
            },
            "compute_power_w": {
                "mean": None if vdd_cgc["mean"] is None else round(vdd_cgc["mean"] / 1000, 3),
                "peak": None if vdd_cgc["peak"] is None else round(vdd_cgc["peak"] / 1000, 3),
            },
            "gpu_util_pct": round(sum(gr3d) / len(gr3d), 1) if gr3d else None,
            "temp_c_peak": max(tj) if tj else None,
            "n_samples": len(win),
        }


def power_mode() -> str:
    """Current nvpmodel mode name, recorded in every Jetson result block.

    PLAN.md §0 requires the power mode to be stated with every on-device number;
    reading it here means no XP can forget to.
    """
    try:
        out = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return "unknown"
    for line in out.splitlines():
        if "NV Power Mode" in line:
            return line.split(":")[-1].strip()
    return "unknown"


if __name__ == "__main__":
    # Self-test: log ~4s of idle/light load and print the summary.
    import json
    import torch
    print("power mode:", power_mode())
    with PowerLogger(interval_ms=100) as p:
        t0 = time.perf_counter()
        if torch.cuda.is_available():
            for _ in range(200):
                x = torch.randn(2000, 2000, device="cuda")
                (x @ x).sum().item()
        else:
            time.sleep(4)
        t1 = time.perf_counter()
    print(json.dumps(p.summary(t0, t1), indent=2))
    print("energy_J:", round(p.energy_joules(t0, t1), 2))
