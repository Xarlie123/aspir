"""NVIDIA Jetson energy backend: the on-module INA3221 sensor, read from sysfs.

The module input rail (``VDD_IN`` on Orin, ``POM_5V_IN`` on Nano, TX2 and
Xavier NX) feeds the CPU, the GPU, the memory and the rest of the SoC. It is
the one rail that covers the whole module, so it is the one integrated; the
other channels (``VDD_CPU_GPU_CV``, ``VDD_SOC``, ...) lie inside it and are
never added to it. A board without a known input rail falls back to the sum of
its channels, and says so in the log.

While a measurement is open a background thread reads the rail's voltage and
current at ``sampling_hz`` (200 Hz by default; the Orin driver refreshes the
registers every millisecond) and integrates the power with the trapezoidal
rule, from a sample taken by ``start_measurement`` to one taken by
``stop_measurement``. Each reading carries its sample count and the rate
achieved. Outside measurements the thread is paused, so it does not run during
the timing pass.

``start_sampling`` keeps the thread running between measurements; then
``get_current_power`` returns the mean power since its previous call, so the
1 Hz idle-baseline poll yields consecutive 1 s means.

jtop refreshes its power reading about once a second, and the jtop backend
reads it only at the two ends of a measurement: a GPU block of 80 inferences
(about 0.6 s) gets no reading of its own. This backend is therefore tried
first on a Jetson.
"""
from __future__ import annotations

import contextlib
import logging
import os
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from simulation_engine._5_analyzer.energy_backends._base import (
    DeviceType,
    EnergyBackend,
    EnergyReading,
)

# Rails that carry the whole module's input power, in order of preference.
MODULE_INPUT_RAILS = ("VDD_IN", "POM_5V_IN")
DRIVER_PATHS = ("/sys/bus/i2c/drivers/ina3221", "/sys/bus/i2c/drivers/ina3221x")
DEFAULT_SAMPLING_HZ = 200.0


@dataclass
class _Channel:
    """One INA3221 channel: either a power file, or a voltage and a current file."""

    label: str
    power: Optional[Path] = None     # microwatts
    voltage: Optional[Path] = None   # millivolts
    current: Optional[Path] = None   # milliamps

    @property
    def readable(self) -> bool:
        return self.power is not None or (self.voltage is not None and self.current is not None)


def _scan_device_dir(directory: Path, channels: dict[str, _Channel]) -> None:
    """Add the channels found in one hwmon (or iio) directory."""
    # hwmon layout (JetPack 5 and 6): inN_label, inN_input (mV), currN_input (mA)
    for label_file in sorted(directory.glob("in*_label")):
        idx = label_file.name[2:-len("_label")]
        try:
            label = label_file.read_text().strip()
        except OSError:
            continue
        ch = channels.setdefault(label, _Channel(label))
        if (directory / f"in{idx}_input").exists():
            ch.voltage = directory / f"in{idx}_input"
        if (directory / f"curr{idx}_input").exists():
            ch.current = directory / f"curr{idx}_input"
        if (directory / f"power{idx}_input").exists():
            ch.power = directory / f"power{idx}_input"
    # iio layout (older L4T): rail_name_N, in_powerN_input (mW), in_voltageN_input, in_currentN_input
    for rail_file in sorted(directory.glob("rail_name_*")):
        idx = rail_file.name.split("_")[-1]
        try:
            label = rail_file.read_text().strip()
        except OSError:
            continue
        ch = channels.setdefault(label, _Channel(label))
        if (directory / f"in_voltage{idx}_input").exists():
            ch.voltage = directory / f"in_voltage{idx}_input"
        if (directory / f"in_current{idx}_input").exists():
            ch.current = directory / f"in_current{idx}_input"


def find_ina3221_channels(driver_paths: Sequence[str | Path] = DRIVER_PATHS) -> dict[str, _Channel]:
    """Every readable INA3221 channel under the given driver directories, by label."""
    channels: dict[str, _Channel] = {}
    for driver in map(Path, driver_paths):
        if not driver.is_dir():
            continue
        for device in sorted(driver.iterdir()):
            if not device.is_dir():
                continue
            dirs = sorted((device / "hwmon").glob("hwmon*")) if (device / "hwmon").is_dir() else []
            dirs += [d for d in sorted(device.glob("iio:device*")) if d.is_dir()]
            for d in dirs or [device]:
                _scan_device_dir(d, channels)
    return {k: v for k, v in channels.items() if v.readable}


class _Reader:
    """Reads one channel's power in mW through open file descriptors."""

    def __init__(self, channel: _Channel):
        self.channel = channel
        paths = [channel.power] if channel.power is not None else [channel.voltage, channel.current]
        self._fds = [os.open(str(p), os.O_RDONLY) for p in paths]

    def read_mw(self) -> Optional[float]:
        try:
            values = [int(os.pread(fd, 64, 0)) for fd in self._fds]
        except (OSError, ValueError):
            return None
        if len(values) == 1:
            return values[0] / 1000.0          # uW -> mW
        return values[0] * values[1] / 1000.0  # mV * mA -> mW

    def close(self) -> None:
        for fd in self._fds:
            with contextlib.suppress(OSError):
                os.close(fd)
        self._fds = []


class _Window:
    """Trapezoidal integral of the power over one open measurement."""

    def __init__(self, t: float, p_mw: float):
        self.t0 = self.t_last = t
        self.p_last = p_mw
        self.energy_mj = 0.0
        self.n = 1

    def add(self, t: float, p_mw: float) -> None:
        if t <= self.t_last:   # a sample read before the window opened
            return
        self.energy_mj += 0.5 * (p_mw + self.p_last) * (t - self.t_last)
        self.t_last, self.p_last = t, p_mw
        self.n += 1


class JetsonSysfsBackend(EnergyBackend):
    """Energy of the Jetson module input rail, sampled from sysfs (see module docstring)."""

    def __init__(self, logger: Optional[logging.Logger] = None,
                 sampling_hz: float = DEFAULT_SAMPLING_HZ,
                 driver_paths: Sequence[str | Path] = DRIVER_PATHS,
                 require_jetson: bool = True):
        super().__init__(logger)
        self._sampling_hz = float(sampling_hz)
        self._driver_paths = tuple(driver_paths)
        self._require_jetson = require_jetson
        self._readers: list[_Reader] = []
        self.rail_label = ""
        self.rails_available: list[str] = []
        self._lock = threading.Lock()
        self._run = threading.Event()
        self._shutdown = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._window: Optional[_Window] = None
        self._sampling_requested = False
        self._since_sum_mw = 0.0
        self._since_n = 0

    # -- discovery ---------------------------------------------------------

    @staticmethod
    def _detect_jetson_type() -> Optional[DeviceType]:
        if not Path("/etc/nv_tegra_release").exists():
            return None
        try:
            model = Path("/proc/device-tree/model").read_text().lower()
        except OSError:
            return DeviceType.JETSON_ORIN
        for key, kind in (("orin", DeviceType.JETSON_ORIN), ("xavier", DeviceType.JETSON_XAVIER),
                          ("tx2", DeviceType.JETSON_TX2), ("nano", DeviceType.JETSON_NANO)):
            if key in model:
                return kind
        return DeviceType.JETSON_ORIN

    def initialize(self) -> bool:
        detected = self._detect_jetson_type()
        if detected is None and self._require_jetson:
            self.logger.debug("Not a Jetson; skipping the INA3221 energy backend")
            return False
        self._device_type = detected or DeviceType.JETSON_ORIN

        channels = find_ina3221_channels(self._driver_paths)
        if not channels:
            self.logger.warning("No INA3221 channel found under %s", ", ".join(map(str, self._driver_paths)))
            return False
        self.rails_available = sorted(channels)
        rail = next((r for r in MODULE_INPUT_RAILS if r in channels), None)
        if rail is not None:
            chosen = [channels[rail]]
            self.rail_label = rail
        else:
            chosen = list(channels.values())
            self.rail_label = "+".join(c.label for c in chosen)
            self.logger.warning(
                "No module input rail (%s) among %s: integrating the sum of the channels",
                "/".join(MODULE_INPUT_RAILS), self.rails_available)
        try:
            self._readers = [_Reader(c) for c in chosen]
        except OSError as exc:
            self.logger.warning("Cannot open the INA3221 files: %s", exc)
            self._close_readers()
            return False
        if self._read_mw() is None:
            self.logger.warning("The INA3221 files of %s cannot be read", self.rail_label)
            self._close_readers()
            return False

        self._device_name = f"Jetson INA3221 {self.rail_label}"
        self._shutdown.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="ina3221-sampler")
        self._thread.start()
        self._is_initialized = True
        self.logger.info("Jetson energy backend: %s at %.0f Hz (channels: %s)",
                         self._device_name, self._sampling_hz, ", ".join(self.rails_available))
        return True

    # -- sampling ----------------------------------------------------------

    @property
    def sampling_hz(self) -> float:
        return self._sampling_hz

    def _read_mw(self) -> Optional[float]:
        total = 0.0
        for reader in self._readers:
            value = reader.read_mw()
            if value is None:
                return None
            total += value
        return total

    def _sample(self) -> Optional[tuple[float, float]]:
        # A read goes over I2C (about 0.5 ms on an Orin): time it at its midpoint
        t_before = time.perf_counter()
        p = self._read_mw()
        return None if p is None else (0.5 * (t_before + time.perf_counter()), p)

    def _loop(self) -> None:
        period = 1.0 / self._sampling_hz
        while not self._shutdown.is_set():
            if not self._run.wait(timeout=0.5):
                continue
            next_t = time.perf_counter()
            while self._run.is_set() and not self._shutdown.is_set():
                sample = self._sample()
                if sample is not None:
                    with self._lock:
                        if self._window is not None:
                            self._window.add(*sample)
                        if self._sampling_requested:
                            self._since_sum_mw += sample[1]
                            self._since_n += 1
                next_t += period
                delay = next_t - time.perf_counter()
                if delay > 0:
                    self._shutdown.wait(delay)
                else:
                    next_t = time.perf_counter()   # fell behind: no burst to catch up

    def _update_run_state(self) -> None:
        if self._window is not None or self._sampling_requested:
            self._run.set()
        else:
            self._run.clear()

    def _sample_retrying(self) -> tuple[float, float]:
        for _ in range(5):
            sample = self._sample()
            if sample is not None:
                return sample
            time.sleep(0.001)
        raise RuntimeError(f"Cannot read the INA3221 rail {self.rail_label}")

    # -- EnergyBackend interface ---------------------------------------------

    def start_sampling(self) -> bool:
        if not self._is_initialized:
            return False
        with self._lock:
            self._sampling_requested = True
            self._since_sum_mw, self._since_n = 0.0, 0
            self._update_run_state()
        return True

    def stop_sampling(self) -> None:
        with self._lock:
            self._sampling_requested = False
            self._update_run_state()

    def get_current_power(self) -> float:
        """Mean power (W) since the previous call while sampling, else one reading."""
        if not self._is_initialized:
            return 0.0
        with self._lock:
            if self._sampling_requested and self._since_n > 0:
                mean_mw = self._since_sum_mw / self._since_n
                self._since_sum_mw, self._since_n = 0.0, 0
                return mean_mw / 1000.0
        p = self._read_mw()
        return 0.0 if p is None else p / 1000.0

    def start_measurement(self) -> None:
        if not self._is_initialized:
            raise RuntimeError("Backend not initialized")
        sample = self._sample_retrying()
        with self._lock:
            self._window = _Window(*sample)
            self._update_run_state()

    def stop_measurement(self) -> EnergyReading:
        if not self._is_initialized:
            raise RuntimeError("Backend not initialized")
        sample = self._sample()
        with self._lock:
            window, self._window = self._window, None
            self._update_run_state()
        if window is None:
            raise RuntimeError("stop_measurement without start_measurement")
        if sample is not None:
            window.add(*sample)
        duration = window.t_last - window.t0
        energy_j = window.energy_mj / 1000.0
        # The rail is shared by CPU and GPU: the total goes into both
        # component fields, as the jtop backend does.
        return EnergyReading(
            energy_joules=energy_j,
            avg_power_watts=energy_j / duration if duration > 0 else 0.0,
            duration_seconds=duration,
            device_type=self._device_type,
            device_name=self._device_name,
            cpu_energy_joules=energy_j,
            gpu_energy_joules=energy_j,
            n_samples=window.n,
            sampling_hz=(window.n - 1) / duration if duration > 0 else None,
        )

    def _close_readers(self) -> None:
        for reader in self._readers:
            reader.close()
        self._readers = []

    def shutdown(self) -> None:
        self._shutdown.set()
        self._run.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._run.clear()
        self._close_readers()
        self._is_initialized = False
