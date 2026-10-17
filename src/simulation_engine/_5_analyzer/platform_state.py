"""State of the machine during a measurement: clocks, power mode, temperatures.

Timing and energy depend on how the platform is set up: the frequency
governors, whether the clocks are pinned (``jetson_clocks``), the Jetson power
mode, the temperature reached. The batch runner and the re-measure worker
record this state at the start and at the end of a run, under
``platform_state`` in the report metadata, so a run can be checked afterwards
(clocks pinned throughout, temperatures, the same power mode). ``cpu_threading``
holds the CPU thread pools at the time of the snapshot (see ``cpu_threads``);
each CPU pass also records the pools it ran with.

Everything is read without root, from sysfs, ``/proc``, ``nvpmodel -q`` and
``nvidia-smi``; whatever is missing is left out. The memory (EMC) clock needs
root and is not read.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

CPUFREQ = Path("/sys/devices/system/cpu/cpufreq")
DEVFREQ = Path("/sys/class/devfreq")
THERMAL = Path("/sys/class/thermal")
HWMON = Path("/sys/class/hwmon")


def _read(path: Path) -> Optional[str]:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _read_int(path: Path) -> Optional[int]:
    text = _read(path)
    try:
        return int(text) if text is not None else None
    except ValueError:
        return None


def _run(cmd: list[str]) -> Optional[str]:
    if shutil.which(cmd[0]) is None:
        return None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def cpu_state() -> dict[str, Any]:
    """Online CPUs and, per frequency policy (one per cluster), governor and kHz."""
    policies = []
    for pol in sorted(CPUFREQ.glob("policy*"), key=lambda p: int(p.name[6:] or 0)):
        policies.append({
            "cpus": _read(pol / "affected_cpus"),
            "governor": _read(pol / "scaling_governor"),
            "min_khz": _read_int(pol / "scaling_min_freq"),
            "max_khz": _read_int(pol / "scaling_max_freq"),
            "cur_khz": _read_int(pol / "scaling_cur_freq"),
            "hw_max_khz": _read_int(pol / "cpuinfo_max_freq"),
        })
    out: dict[str, Any] = {"online": _read(Path("/sys/devices/system/cpu/online")),
                           "count": os.cpu_count(), "policies": policies}
    if policies:
        out["pinned"] = all(p["min_khz"] is not None and p["min_khz"] == p["max_khz"] for p in policies)
    return out


def jetson_state() -> Optional[dict[str, Any]]:
    """Model, L4T release, power mode, GPU clock and fan of a Jetson; None elsewhere."""
    release = _read(Path("/etc/nv_tegra_release"))
    if release is None:
        return None
    out: dict[str, Any] = {"model": (_read(Path("/proc/device-tree/model")) or "").rstrip("\x00"),
                           "l4t": release.splitlines()[0]}
    mode = _run(["nvpmodel", "-q"])
    if mode:
        lines = mode.splitlines()
        out["nvpmodel"] = {"name": lines[0].split(":", 1)[-1].strip(),
                           "id": int(lines[1]) if len(lines) > 1 and lines[1].strip().isdigit() else None}
    for dev in sorted(DEVFREQ.glob("*.gpu")):
        gpu = {"device": dev.name, "governor": _read(dev / "governor"),
               "min_hz": _read_int(dev / "min_freq"), "max_hz": _read_int(dev / "max_freq"),
               "cur_hz": _read_int(dev / "cur_freq")}
        gpu["pinned"] = gpu["min_hz"] is not None and gpu["min_hz"] == gpu["max_hz"]
        out["gpu"] = gpu
        break
    for hw in sorted(HWMON.glob("hwmon*")):
        name = _read(hw / "name")
        if name == "pwmfan" and (hw / "pwm1").exists():
            out["fan_pwm"] = _read_int(hw / "pwm1")
        elif name == "pwm_tach" and (hw / "rpm").exists():
            out["fan_rpm"] = _read_int(hw / "rpm")
    return out


def nvidia_smi_state() -> Optional[dict[str, Any]]:
    """Clocks, performance state and temperature of a desktop NVIDIA GPU."""
    fields = ["name", "driver_version", "pstate", "clocks.sm", "clocks.mem",
              "temperature.gpu", "power.limit"]
    out = _run(["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"])
    if not out:
        return None
    values = [v.strip() for v in out.splitlines()[0].split(",")]
    return dict(zip(fields, values, strict=False))


def temperatures_c() -> dict[str, float]:
    temps = {}
    for zone in sorted(THERMAL.glob("thermal_zone*")):
        kind, milli = _read(zone / "type"), _read_int(zone / "temp")
        if kind and milli is not None:
            temps[kind] = milli / 1000.0
    return temps


def platform_state() -> dict[str, Any]:
    """Snapshot of the platform, for the report metadata (see module docstring)."""
    state: dict[str, Any] = {
        "time": datetime.now().isoformat(),
        "hostname": platform.node(),
        "machine": platform.machine(),
        "kernel": platform.release(),
        "python": platform.python_version(),
    }
    try:
        import torch
        state["torch"] = torch.__version__
        state["torch_cuda"] = torch.version.cuda
        state["torch_threads"] = torch.get_num_threads()
    except ImportError:
        pass
    from simulation_engine._5_analyzer.cpu_threads import cpu_threading_state
    state["cpu_threading"] = cpu_threading_state()
    state["cpu"] = cpu_state()
    jetson = jetson_state()
    if jetson is not None:
        state["jetson"] = jetson
    else:
        gpu = nvidia_smi_state()
        if gpu is not None:
            state["nvidia_gpu"] = gpu
    state["temperatures_c"] = temperatures_c()
    return state
