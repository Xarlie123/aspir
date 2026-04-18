"""Data types and the abstract backend base class for energy measurement."""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class DeviceType(Enum):
    """Supported device types for energy measurement."""
    NVIDIA_GPU = "nvidia_gpu"         # Desktop/workstation NVIDIA GPU
    JETSON_ORIN = "jetson_orin"       # NVIDIA Jetson Orin family
    JETSON_NANO = "jetson_nano"       # NVIDIA Jetson Nano
    JETSON_XAVIER = "jetson_xavier"   # NVIDIA Jetson Xavier
    JETSON_TX2 = "jetson_tx2"         # NVIDIA Jetson TX2
    INTEL_CPU = "intel_cpu"           # Intel CPU with RAPL
    AMD_CPU = "amd_cpu"               # AMD CPU (limited RAPL support)
    UNKNOWN = "unknown"


@dataclass
class EnergyReading:
    """Single energy reading with metadata."""
    energy_joules: float           # Total energy consumed in Joules
    avg_power_watts: float         # Average power in Watts
    duration_seconds: float        # Measurement duration
    device_type: DeviceType
    device_name: str
    timestamp: float = field(default_factory=time.time)

    # Optional per-component breakdown (if available)
    gpu_energy_joules: Optional[float] = None
    cpu_energy_joules: Optional[float] = None
    memory_energy_joules: Optional[float] = None

    # Temperature data (if available)
    temperature_celsius: Optional[float] = None

    @property
    def energy_mj(self) -> float:
        """Energy in millijoules."""
        return self.energy_joules * 1000.0

    @property
    def power_mw(self) -> float:
        """Power in milliwatts."""
        return self.avg_power_watts * 1000.0


@dataclass
class EnergyMeasurementResult:
    """Result from measuring energy during a function execution."""
    readings: list[EnergyReading]
    execution_time_ms: float
    n_iterations: int

    @property
    def total_energy_joules(self) -> float:
        """Total energy across all devices."""
        return sum(r.energy_joules for r in self.readings)

    @property
    def avg_energy_per_iteration_joules(self) -> float:
        """Average energy per iteration."""
        return self.total_energy_joules / self.n_iterations if self.n_iterations > 0 else 0

    @property
    def avg_power_watts(self) -> float:
        """Average power during measurement."""
        duration = self.execution_time_ms / 1000.0
        return self.total_energy_joules / duration if duration > 0 else 0

    def get_by_device_type(self, device_type: DeviceType) -> Optional[EnergyReading]:
        """Get reading for a specific device type."""
        for r in self.readings:
            if r.device_type == device_type:
                return r
        return None


class EnergyBackend(ABC):
    """Abstract base class for energy measurement backends."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self._is_initialized = False
        self._device_type = DeviceType.UNKNOWN
        self._device_name = "Unknown"

    @property
    def device_type(self) -> DeviceType:
        return self._device_type

    @property
    def device_name(self) -> str:
        return self._device_name

    @property
    def is_available(self) -> bool:
        """Check if this backend is available on the current system."""
        return self._is_initialized

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the backend. Returns True if successful."""
        pass

    @abstractmethod
    def start_measurement(self) -> None:
        """Start energy measurement."""
        pass

    @abstractmethod
    def stop_measurement(self) -> EnergyReading:
        """Stop measurement and return the reading."""
        pass

    @abstractmethod
    def get_current_power(self) -> float:
        """Get instantaneous power reading in Watts."""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Clean up resources."""
        pass

    def measure_function(
        self,
        func: Callable,
        *args,
        n_iterations: int = 1,
        warmup_iterations: int = 0,
        pre_sync: Optional[Callable] = None,
        post_sync: Optional[Callable] = None,
        **kwargs
    ) -> tuple[Any, EnergyReading]:
        """
        Measure energy consumption while executing a function.

        Args:
            func: Function to measure
            *args: Arguments to pass to function
            n_iterations: Number of times to run the function
            warmup_iterations: Warmup runs (not measured)
            pre_sync: Synchronization function to call before timing
            post_sync: Synchronization function to call after timing
            **kwargs: Keyword arguments to pass to function

        Returns:
            Tuple of (function result, EnergyReading)
        """
        # Warmup
        for _ in range(warmup_iterations):
            result = func(*args, **kwargs)
            if post_sync:
                post_sync()

        # Stabilization delay
        time.sleep(0.1)

        if pre_sync:
            pre_sync()

        self.start_measurement()
        t_start = time.perf_counter()

        for _ in range(n_iterations):
            result = func(*args, **kwargs)
            if post_sync:
                post_sync()

        t_end = time.perf_counter()
        reading = self.stop_measurement()

        # Update reading with actual measured duration
        reading.duration_seconds = t_end - t_start
        reading.avg_power_watts = reading.energy_joules / reading.duration_seconds if reading.duration_seconds > 0 else 0

        return result, reading
