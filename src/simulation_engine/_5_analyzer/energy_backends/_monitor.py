"""Unified energy monitoring — auto-detects and orchestrates all backends."""
from __future__ import annotations

import logging
import platform
import time
from pathlib import Path
from typing import Any, Callable, Optional

from simulation_engine._5_analyzer.energy_backends._base import (
    DeviceType,
    EnergyBackend,
    EnergyMeasurementResult,
    EnergyReading,
)
from simulation_engine._5_analyzer.energy_backends._jetson import JetsonSysfsBackend
from simulation_engine._5_analyzer.energy_backends._nvml import NVMLBackend
from simulation_engine._5_analyzer.energy_backends._rapl import RAPLBackend


class EnergyMonitor:
    """
    Unified energy monitoring interface with auto-detection.

    Automatically detects available energy measurement backends and
    provides a unified interface for measuring energy consumption.

    Usage:
        monitor = EnergyMonitor()
        monitor.initialize()

        # Get available backends
        print(monitor.available_backends)

        # Measure a function
        result = monitor.measure(my_inference_function, input_tensor)
        print(f"Energy: {result.total_energy_joules:.3f} J")

        monitor.shutdown()
    """

    def __init__(
        self,
        enable_gpu: bool = True,
        enable_cpu: bool = True,
        enable_jetson: bool = True,
        pmlib_server_ip: str = "127.0.0.1",
        pmlib_server_port: int = 6526,
        logger: Optional[logging.Logger] = None
    ):
        self.logger = logger or logging.getLogger("EnergyMonitor")

        self._enable_gpu = enable_gpu
        self._enable_cpu = enable_cpu
        self._enable_jetson = enable_jetson
        # PMLib params kept for backwards compatibility but not used
        self._pmlib_server_ip = pmlib_server_ip
        self._pmlib_server_port = pmlib_server_port

        self._backends: list[EnergyBackend] = []
        self._is_initialized = False

    @property
    def available_backends(self) -> list[str]:
        """List of available backend names."""
        return [b.device_name for b in self._backends if b.is_available]

    @property
    def device_types(self) -> list[DeviceType]:
        """List of available device types."""
        return [b.device_type for b in self._backends if b.is_available]

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    def detect_platform(self) -> dict[str, Any]:
        """
        Detect the current platform and available energy measurement capabilities.

        Returns:
            Dictionary with platform information
        """
        info = {
            "platform": platform.system(),
            "processor": platform.processor(),
            "is_jetson": False,
            "jetson_type": None,
            "has_nvidia_gpu": False,
            "gpu_name": None,
            "has_rapl": False,
            "rapl_domains": []
        }

        # Check for Jetson
        if Path("/etc/nv_tegra_release").exists():
            info["is_jetson"] = True
            model_path = Path("/proc/device-tree/model")
            if model_path.exists():
                try:
                    model = model_path.read_text()
                    info["jetson_type"] = model.strip('\x00')
                except Exception:
                    pass

        # Check for NVIDIA GPU (non-Jetson)
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                import pynvml
                pynvml.nvmlInit()
                count = pynvml.nvmlDeviceGetCount()
                if count > 0:
                    info["has_nvidia_gpu"] = True
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    name = pynvml.nvmlDeviceGetName(handle)
                    info["gpu_name"] = name.decode('utf-8') if isinstance(name, bytes) else str(name)
                pynvml.nvmlShutdown()
        except ImportError:
            self.logger.debug("pynvml not installed")
        except Exception as e:
            self.logger.debug(f"NVML detection failed: {e}")

        # Check for RAPL
        rapl_path = Path("/sys/class/powercap/intel-rapl")
        if rapl_path.exists():
            info["has_rapl"] = True
            try:
                for domain_dir in rapl_path.iterdir():
                    if domain_dir.name.startswith("intel-rapl:"):
                        name_path = domain_dir / "name"
                        if name_path.exists():
                            try:
                                info["rapl_domains"].append(name_path.read_text().strip())
                            except PermissionError:
                                pass
            except Exception:
                pass

        return info

    def initialize(self) -> bool:
        """
        Initialize available energy measurement backends.

        Returns:
            True if at least one backend was initialized
        """
        platform_info = self.detect_platform()
        self.logger.info(f"Detected platform: {platform_info}")

        # Try Jetson first (pure Python sysfs backend)
        if self._enable_jetson and platform_info["is_jetson"]:
            backend = JetsonSysfsBackend(logger=self.logger)
            if backend.initialize():
                self._backends.append(backend)
                self.logger.info(f"Jetson energy monitoring enabled: {backend.device_name}")

        # Try NVIDIA GPU (NVML) - only if not on Jetson (to avoid conflicts)
        if self._enable_gpu and platform_info["has_nvidia_gpu"] and not platform_info["is_jetson"]:
            backend = NVMLBackend(logger=self.logger)
            if backend.initialize():
                self._backends.append(backend)
                self.logger.info(f"NVIDIA GPU energy monitoring enabled: {backend.device_name}")

        # Try Intel CPU RAPL
        if self._enable_cpu and platform_info["has_rapl"]:
            backend = RAPLBackend(logger=self.logger)
            if backend.initialize():
                self._backends.append(backend)
                self.logger.info(f"CPU RAPL energy monitoring enabled: {backend.device_name}")

        self._is_initialized = len(self._backends) > 0

        if not self._is_initialized:
            self.logger.warning("No energy measurement backends available")

        return self._is_initialized

    def start_measurement(self) -> None:
        """Start measurement on all backends."""
        for backend in self._backends:
            backend.start_measurement()

    def stop_measurement(self) -> list[EnergyReading]:
        """Stop measurement and return readings from all backends."""
        readings = []
        for backend in self._backends:
            try:
                readings.append(backend.stop_measurement())
            except Exception as e:
                self.logger.error(f"Error stopping {backend.device_name}: {e}")
        return readings

    def get_current_power(self) -> dict[str, float]:
        """Get current power from all backends."""
        return {
            backend.device_name: backend.get_current_power()
            for backend in self._backends
        }

    def measure(
        self,
        func: Callable,
        *args,
        n_iterations: int = 1,
        warmup_iterations: int = 0,
        cuda_sync: bool = True,
        **kwargs
    ) -> EnergyMeasurementResult:
        """
        Measure energy consumption while executing a function.

        Args:
            func: Function to measure
            *args: Arguments to pass to function
            n_iterations: Number of times to run the function
            warmup_iterations: Warmup runs (not measured)
            cuda_sync: Whether to synchronize CUDA before/after measurements
            **kwargs: Keyword arguments to pass to function

        Returns:
            EnergyMeasurementResult with readings from all backends
        """
        if not self._is_initialized:
            raise RuntimeError("EnergyMonitor not initialized")

        # Setup CUDA sync if requested
        pre_sync = None
        post_sync = None
        if cuda_sync:
            try:
                import torch
                if torch.cuda.is_available():
                    pre_sync = torch.cuda.synchronize
                    post_sync = torch.cuda.synchronize
            except ImportError:
                pass

        # Warmup
        for _ in range(warmup_iterations):
            result = func(*args, **kwargs)
            if post_sync:
                post_sync()

        # Stabilization delay
        time.sleep(0.1)

        # Start all backends
        if pre_sync:
            pre_sync()

        self.start_measurement()
        t_start = time.perf_counter()

        # Run measured iterations
        for _ in range(n_iterations):
            result = func(*args, **kwargs)
            if post_sync:
                post_sync()

        t_end = time.perf_counter()
        readings = self.stop_measurement()

        execution_time_ms = (t_end - t_start) * 1000.0

        return EnergyMeasurementResult(
            readings=readings,
            execution_time_ms=execution_time_ms,
            n_iterations=n_iterations
        )

    def shutdown(self) -> None:
        """Shutdown all backends."""
        for backend in self._backends:
            try:
                backend.shutdown()
            except Exception as e:
                self.logger.error(f"Error shutting down {backend.device_name}: {e}")

        self._backends = []
        self._is_initialized = False


# Convenience function for quick measurements
def measure_energy(
    func: Callable,
    *args,
    n_iterations: int = 1,
    warmup_iterations: int = 5,
    **kwargs
) -> EnergyMeasurementResult:
    """
    Quick energy measurement helper.

    Initializes an EnergyMonitor, measures the function, and cleans up.

    Args:
        func: Function to measure
        *args: Arguments to pass to function
        n_iterations: Number of iterations
        warmup_iterations: Warmup iterations
        **kwargs: Keyword arguments

    Returns:
        EnergyMeasurementResult
    """
    monitor = EnergyMonitor()
    if not monitor.initialize():
        raise RuntimeError("No energy measurement backends available")

    try:
        return monitor.measure(
            func, *args,
            n_iterations=n_iterations,
            warmup_iterations=warmup_iterations,
            **kwargs
        )
    finally:
        monitor.shutdown()
