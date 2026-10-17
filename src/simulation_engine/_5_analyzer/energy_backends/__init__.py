"""Energy measurement backends with auto-detection for different hardware platforms.

Supported backends:
- :class:`NVMLBackend` — NVIDIA GPUs (desktop/workstation) via pynvml
- :class:`JetsonSysfsBackend` — NVIDIA Jetson module input rail (INA3221) sampled from sysfs at 200 Hz
- :class:`JtopEnergyBackend` — NVIDIA Jetson via jtop, the fallback (refreshes about once a second)
- :class:`RAPLBackend` — Intel CPUs via RAPL (Running Average Power Limit)

The :class:`EnergyMonitor` class auto-detects available backends and provides a
unified interface for energy measurement during inference.
"""
from simulation_engine._5_analyzer.energy_backends._base import (
    DeviceType,
    EnergyBackend,
    EnergyMeasurementResult,
    EnergyReading,
)
from simulation_engine._5_analyzer.energy_backends._jetson import JetsonSysfsBackend
from simulation_engine._5_analyzer.energy_backends._jtop_energy import (
    JtopEnergyBackend,
)
from simulation_engine._5_analyzer.energy_backends._monitor import (
    EnergyMonitor,
    measure_energy,
)
from simulation_engine._5_analyzer.energy_backends._nvml import NVMLBackend
from simulation_engine._5_analyzer.energy_backends._rapl import RAPLBackend

__all__ = [
    "DeviceType",
    "EnergyBackend",
    "EnergyMeasurementResult",
    "EnergyMonitor",
    "EnergyReading",
    "JetsonSysfsBackend",
    "JtopEnergyBackend",
    "NVMLBackend",
    "RAPLBackend",
    "measure_energy",
]
