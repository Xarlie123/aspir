"""NVIDIA desktop/workstation GPU energy backend via NVML (pynvml)."""
from __future__ import annotations

import logging
import time
from typing import Optional

from simulation_engine._5_analyzer.energy_backends._base import (
    DeviceType,
    EnergyBackend,
    EnergyReading,
)


class NVMLBackend(EnergyBackend):
    """
    Energy measurement backend for NVIDIA desktop/workstation GPUs using NVML.

    Uses pynvml to query GPU power consumption directly from the GPU.
    This is the preferred method for desktop NVIDIA GPUs.
    """

    def __init__(self, gpu_index: int = 0, logger: Optional[logging.Logger] = None):
        super().__init__(logger)
        self._gpu_index = gpu_index
        self._handle = None
        self._pynvml = None
        self._measurement_start_time = 0
        self._measurement_start_energy = 0
        self._power_samples: list[tuple[float, float]] = []  # (timestamp, power_mw)
        self._measuring = False

    def initialize(self) -> bool:
        """Initialize NVML and get GPU handle."""
        try:
            # Suppress FutureWarning about pynvml deprecation
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                import pynvml
                self._pynvml = pynvml

            pynvml.nvmlInit()

            device_count = pynvml.nvmlDeviceGetCount()
            if self._gpu_index >= device_count:
                self.logger.error(f"GPU index {self._gpu_index} out of range (found {device_count} GPUs)")
                return False

            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self._gpu_index)
            name = pynvml.nvmlDeviceGetName(self._handle)
            self._device_name = name.decode('utf-8') if isinstance(name, bytes) else str(name)

            self._device_type = DeviceType.NVIDIA_GPU
            self._is_initialized = True

            self.logger.info(f"NVML initialized for GPU {self._gpu_index}: {self._device_name}")
            return True

        except ImportError as e:
            self.logger.warning(f"pynvml not installed: {e}. Install with: pip install pynvml")
            return False
        except Exception as e:
            self.logger.error(f"Failed to initialize NVML: {e}")
            return False

    def start_measurement(self) -> None:
        """Start energy measurement by recording initial state."""
        if not self._is_initialized:
            raise RuntimeError("Backend not initialized")

        self._power_samples = []
        self._measurement_start_time = time.perf_counter()
        self._measuring = True

        # Try to get total energy counter if available
        try:
            self._measurement_start_energy = self._pynvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)
        except self._pynvml.NVMLError:
            # Energy counter not available on this GPU, use power sampling
            self._measurement_start_energy = None
            self.logger.debug("Energy counter not available, using power sampling")

    def stop_measurement(self) -> EnergyReading:
        """Stop measurement and calculate energy consumed."""
        if not self._is_initialized:
            raise RuntimeError("Backend not initialized")

        self._measuring = False
        t_end = time.perf_counter()
        duration = t_end - self._measurement_start_time

        # Try to use energy counter if available
        if self._measurement_start_energy is not None:
            try:
                end_energy = self._pynvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)
                # Energy is in millijoules
                energy_joules = (end_energy - self._measurement_start_energy) / 1000.0

                # Warn if energy counter didn't register a change (measurement too short)
                if energy_joules == 0:
                    self.logger.warning(
                        f"Energy counter returned 0 (measurement too short: {duration*1000:.1f}ms). "
                        "Consider increasing measurement_runs for accurate readings."
                    )

                avg_power = energy_joules / duration if duration > 0 else 0
                temp = self._get_temperature()

                return EnergyReading(
                    energy_joules=energy_joules,
                    avg_power_watts=avg_power,
                    duration_seconds=duration,
                    device_type=self._device_type,
                    device_name=self._device_name,
                    gpu_energy_joules=energy_joules,
                    temperature_celsius=temp
                )
            except self._pynvml.NVMLError:
                pass

        # Fallback: estimate from instantaneous power reading
        # This is less accurate but works on all NVIDIA GPUs
        try:
            power_mw = self._pynvml.nvmlDeviceGetPowerUsage(self._handle)
            power_w = power_mw / 1000.0
            energy_joules = power_w * duration

            temp = self._get_temperature()

            return EnergyReading(
                energy_joules=energy_joules,
                avg_power_watts=power_w,
                duration_seconds=duration,
                device_type=self._device_type,
                device_name=self._device_name,
                gpu_energy_joules=energy_joules,
                temperature_celsius=temp
            )
        except self._pynvml.NVMLError as e:
            self.logger.error(f"Failed to get power reading: {e}")
            return EnergyReading(
                energy_joules=0,
                avg_power_watts=0,
                duration_seconds=duration,
                device_type=self._device_type,
                device_name=self._device_name
            )

    def _get_temperature(self) -> Optional[float]:
        """Get GPU temperature safely."""
        try:
            return self._pynvml.nvmlDeviceGetTemperature(
                self._handle,
                self._pynvml.NVML_TEMPERATURE_GPU
            )
        except Exception:
            return None

    def get_current_power(self) -> float:
        """Get current GPU power consumption in Watts."""
        if not self._is_initialized:
            return 0.0
        try:
            power_mw = self._pynvml.nvmlDeviceGetPowerUsage(self._handle)
            return power_mw / 1000.0
        except self._pynvml.NVMLError:
            return 0.0

    def get_temperature(self) -> Optional[float]:
        """Get current GPU temperature in Celsius."""
        return self._get_temperature()

    def shutdown(self) -> None:
        """Shutdown NVML."""
        if self._is_initialized and self._pynvml:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass
            self._is_initialized = False
            self.logger.debug("NVML shutdown complete")
