"""Jetson energy backend that reuses ``jtop`` (jetson-stats).

The raw sysfs layout under ``/sys/bus/i2c/drivers/ina3221/`` is per-board
specific and shifts between JetPack releases; jtop normalises all of it
into a single dict, so this backend just reads instantaneous power from
jtop on start/stop and integrates trapezoidally — same idea as
:class:`JetsonSysfsBackend`, but without the per-rail label bookkeeping.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

from simulation_engine._5_analyzer.energy_backends._base import (
    DeviceType,
    EnergyBackend,
    EnergyReading,
)

try:
    from jtop import jtop
    HAS_JTOP = True
except ImportError:
    HAS_JTOP = False


def _detect_jetson_type() -> Optional[DeviceType]:
    """Map ``/proc/device-tree/model`` to a :class:`DeviceType`."""
    if not Path("/etc/nv_tegra_release").exists():
        return None
    model_path = Path("/proc/device-tree/model")
    if not model_path.exists():
        return DeviceType.JETSON_ORIN  # generic fallback
    try:
        model = model_path.read_text().lower()
    except Exception:
        return DeviceType.JETSON_ORIN
    if "orin" in model:
        return DeviceType.JETSON_ORIN
    if "xavier" in model:
        return DeviceType.JETSON_XAVIER
    if "tx2" in model:
        return DeviceType.JETSON_TX2
    if "nano" in model:
        return DeviceType.JETSON_NANO
    return DeviceType.JETSON_ORIN


def _read_jtop_power_mw(jt) -> float:
    """Best-effort total instantaneous power reader across jtop API revisions.

    jtop has shipped several accessors for power; try the structured ones
    first and fall back to the flat ``jt.stats`` dict. Returns 0.0 if none
    match (daemon not yet reporting, or API surface unknown).
    """
    # jtop >= 4.x — structured accessor.
    power = getattr(jt, "power", None)
    if isinstance(power, dict):
        total = power.get("tot") or power.get("total")
        if isinstance(total, dict):
            value = total.get("power") or total.get("avg")
            if value is not None:
                return float(value)
        # If there is no aggregate, sum the rails we can see.
        rails = power.get("rail") if "rail" in power else None
        if isinstance(rails, dict):
            acc = 0.0
            for entry in rails.values():
                if isinstance(entry, dict):
                    val = entry.get("power") or entry.get("avg")
                    if val is not None:
                        acc += float(val)
            if acc > 0:
                return acc
    # Older jtop / different release — flat ``stats`` dict.
    stats = getattr(jt, "stats", None)
    if isinstance(stats, dict):
        for key in ("Power TOT", "Power POM_5V_IN", "Power VDD_IN",
                    "Power VIN_SYS_5V0"):
            if key in stats:
                try:
                    return float(stats[key])
                except (TypeError, ValueError):
                    pass
    return 0.0


class JtopEnergyBackend(EnergyBackend):
    """Jetson energy backend that delegates sensor reading to jtop."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        super().__init__(logger)
        self._jt = None
        self._t_start: float = 0.0
        self._p_start_mw: float = 0.0
        self._device_type = DeviceType.JETSON_ORIN  # refined in initialize()

    def initialize(self) -> bool:
        if not HAS_JTOP:
            self.logger.debug("jetson-stats not installed; skipping jtop energy backend")
            return False
        detected = _detect_jetson_type()
        if detected is None:
            self.logger.debug("Not a Jetson; skipping jtop energy backend")
            return False
        self._device_type = detected

        try:
            self._jt = jtop()
            self._jt.start()
        except Exception as e:
            # Common cause: the jtop.service daemon isn't running. Guide the
            # user to the fix instead of failing silently.
            self.logger.warning(
                "Could not start jtop for energy monitoring (%s). "
                "On Jetson install and start the jtop service:\n"
                "  sudo pip install -U jetson-stats\n"
                "  sudo systemctl enable --now jtop",
                e,
            )
            self._jt = None
            return False

        # Sanity check: make sure the daemon is actually reporting power.
        # The first update can take up to ~1 s to arrive after start().
        deadline = time.time() + 2.0
        while time.time() < deadline:
            if self._jt.ok(spin=True) and _read_jtop_power_mw(self._jt) > 0:
                break
            time.sleep(0.05)

        power_mw = _read_jtop_power_mw(self._jt)
        if power_mw <= 0:
            self.logger.warning(
                "jtop started but reports 0 W — the daemon may still be "
                "warming up or this board doesn't expose a total power rail."
            )
            # We still register the backend; stop_measurement will just
            # return 0 W rather than raise.

        self._device_name = "Jetson (jtop)"
        self._is_initialized = True
        self.logger.info("jtop energy backend initialized: %s", self._device_name)
        return True

    def get_current_power(self) -> float:
        """Instantaneous total power in Watts."""
        if self._jt is None or not self._jt.ok():
            return 0.0
        return _read_jtop_power_mw(self._jt) / 1000.0

    def start_measurement(self) -> None:
        if not self._is_initialized:
            raise RuntimeError("Backend not initialized")
        self._t_start = time.perf_counter()
        self._p_start_mw = _read_jtop_power_mw(self._jt)

    def stop_measurement(self) -> EnergyReading:
        if not self._is_initialized:
            raise RuntimeError("Backend not initialized")
        t_end = time.perf_counter()
        p_end_mw = _read_jtop_power_mw(self._jt)
        duration = t_end - self._t_start
        avg_mw = (self._p_start_mw + p_end_mw) / 2.0
        energy_mj = avg_mw * duration
        return EnergyReading(
            energy_joules=energy_mj / 1000.0,
            avg_power_watts=avg_mw / 1000.0,
            duration_seconds=duration,
            device_type=self._device_type,
            device_name=self._device_name,
            gpu_energy_joules=energy_mj / 1000.0,  # Jetson GPU shares the rail
        )

    def shutdown(self) -> None:
        if self._jt is not None:
            try:
                self._jt.close()
            except Exception:
                pass
            self._jt = None
        self._is_initialized = False
