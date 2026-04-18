"""Intel CPU energy backend via Linux RAPL sysfs interface."""
from __future__ import annotations

import logging
import platform
import time
from pathlib import Path
from typing import Optional

from simulation_engine._5_analyzer.energy_backends._base import (
    DeviceType,
    EnergyBackend,
    EnergyReading,
)


class RAPLBackend(EnergyBackend):
    """
    Energy measurement backend for Intel CPUs using RAPL (Running Average Power Limit).

    Reads energy counters from /sys/class/powercap/intel-rapl/ on Linux.
    Requires read access to the RAPL sysfs interface.
    """

    RAPL_PATH = Path("/sys/class/powercap/intel-rapl")

    def __init__(self, logger: Optional[logging.Logger] = None):
        super().__init__(logger)
        self._domains: dict[str, Path] = {}  # domain_name -> energy_uj path
        self._start_readings: dict[str, int] = {}
        self._max_energy: dict[str, int] = {}
        self._measurement_start_time = 0

    def initialize(self) -> bool:
        """Initialize RAPL by discovering available power domains."""
        if platform.system() != "Linux":
            self.logger.info("RAPL is only available on Linux")
            return False

        if not self.RAPL_PATH.exists():
            self.logger.info("RAPL sysfs interface not found")
            return False

        # Discover RAPL domains
        try:
            for domain_dir in self.RAPL_PATH.iterdir():
                if domain_dir.name.startswith("intel-rapl:"):
                    name_path = domain_dir / "name"
                    energy_path = domain_dir / "energy_uj"
                    max_energy_path = domain_dir / "max_energy_range_uj"

                    if energy_path.exists() and name_path.exists():
                        try:
                            domain_name = name_path.read_text().strip()
                            # Test if we can read the energy file
                            test_read = energy_path.read_text().strip()
                            int(test_read)  # Verify it's a valid number

                            self._domains[domain_name] = energy_path

                            # Get max energy range for overflow handling
                            if max_energy_path.exists():
                                self._max_energy[domain_name] = int(max_energy_path.read_text().strip())
                            else:
                                self._max_energy[domain_name] = 2**32  # Default assumption

                            self.logger.debug(f"Found RAPL domain: {domain_name}")
                        except PermissionError:
                            self.logger.warning(f"No permission to read {energy_path}. Try: sudo chmod +r {energy_path}")
                            continue
                        except ValueError as e:
                            self.logger.warning(f"Invalid value in {energy_path}: {e}")
                            continue

            if not self._domains:
                self.logger.warning("No readable RAPL domains found (check permissions)")
                return False

            # Detect CPU type
            self._device_name = self._detect_cpu_name()
            self._device_type = DeviceType.INTEL_CPU
            self._is_initialized = True

            self.logger.info(f"RAPL initialized with domains: {list(self._domains.keys())}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize RAPL: {e}")
            return False

    def _detect_cpu_name(self) -> str:
        """Detect CPU model name."""
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":")[1].strip()
        except Exception:
            pass
        return "Intel CPU"

    def _read_energy_uj(self, domain: str) -> int:
        """Read energy counter in microjoules."""
        try:
            return int(self._domains[domain].read_text().strip())
        except (PermissionError, FileNotFoundError, ValueError) as e:
            self.logger.warning(f"Failed to read RAPL energy for {domain}: {e}")
            return 0

    def start_measurement(self) -> None:
        """Record starting energy values for all domains."""
        if not self._is_initialized:
            raise RuntimeError("Backend not initialized")

        self._start_readings = {
            domain: self._read_energy_uj(domain)
            for domain in self._domains
        }
        self._measurement_start_time = time.perf_counter()

    def stop_measurement(self) -> EnergyReading:
        """Calculate energy consumed since start."""
        if not self._is_initialized:
            raise RuntimeError("Backend not initialized")

        t_end = time.perf_counter()
        duration = t_end - self._measurement_start_time

        total_energy_uj = 0
        for domain in self._domains:
            end_val = self._read_energy_uj(domain)
            start_val = self._start_readings.get(domain, 0)

            # Handle counter overflow
            if end_val < start_val:
                energy_uj = (self._max_energy[domain] - start_val) + end_val
            else:
                energy_uj = end_val - start_val

            total_energy_uj += energy_uj

        energy_joules = total_energy_uj / 1_000_000.0
        avg_power = energy_joules / duration if duration > 0 else 0

        return EnergyReading(
            energy_joules=energy_joules,
            avg_power_watts=avg_power,
            duration_seconds=duration,
            device_type=self._device_type,
            device_name=self._device_name,
            cpu_energy_joules=energy_joules
        )

    def get_current_power(self) -> float:
        """Estimate current power by sampling over a short interval."""
        if not self._is_initialized:
            return 0.0

        # Sample for 100ms
        start = {d: self._read_energy_uj(d) for d in self._domains}
        time.sleep(0.1)
        end = {d: self._read_energy_uj(d) for d in self._domains}

        total_uj = sum(end[d] - start[d] for d in self._domains)
        return (total_uj / 1_000_000.0) / 0.1  # Convert to Watts

    def shutdown(self) -> None:
        """No cleanup needed for RAPL."""
        self._is_initialized = False
