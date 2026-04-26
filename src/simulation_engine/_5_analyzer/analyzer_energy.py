"""
Energy analysis module for measuring power consumption during inference.

Provides EnergyAnalyzer class that uses the energy_backends module to measure
energy consumption on various hardware platforms (NVIDIA GPU, Jetson, Intel CPU).
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Callable, Union

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

from .energy_backends import (
    EnergyMonitor,
    EnergyMeasurementResult,
    EnergyReading,
    DeviceType
)


@dataclass
class EnergyAnalysisResult:
    """Complete results from energy analysis."""

    # Per-image energy measurements
    energy_per_image_joules: List[float]
    power_per_image_watts: List[float]
    time_per_image_ms: List[float]

    # Aggregate statistics
    mean_energy_joules: float
    std_energy_joules: float
    mean_power_watts: float
    std_power_watts: float
    mean_time_ms: float
    std_time_ms: float

    # Device information
    device_type: DeviceType
    device_name: str

    # Temperature data (if available)
    temperature_per_image: Optional[List[float]] = None
    mean_temperature: Optional[float] = None

    # Breakdown by component (if available)
    gpu_energy_joules: Optional[float] = None
    cpu_energy_joules: Optional[float] = None

    @property
    def energy_per_image_mj(self) -> List[float]:
        """Energy per image in millijoules."""
        return [e * 1000 for e in self.energy_per_image_joules]

    @property
    def mean_energy_mj(self) -> float:
        """Mean energy in millijoules."""
        return self.mean_energy_joules * 1000

    @property
    def efficiency_images_per_joule(self) -> float:
        """Processing efficiency in images per Joule."""
        return 1.0 / self.mean_energy_joules if self.mean_energy_joules > 0 else 0


class EnergyAnalyzer:
    """
    Analyzer for measuring energy consumption during model inference.

    Supports automatic hardware detection and provides detailed energy
    statistics per image and overall.

    Usage:
        analyzer = EnergyAnalyzer(model, device='cuda')
        analyzer.initialize()

        result = analyzer.analyze_inference(test_images, n_runs=10)
        print(f"Mean energy: {result.mean_energy_mj:.2f} mJ/image")
        print(f"Mean power: {result.mean_power_watts:.2f} W")

        analyzer.shutdown()
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        device: str = 'cpu',
        warmup_runs: int = 5,
        measurement_runs: int = 10,
        enable_gpu_energy: bool = True,
        enable_cpu_energy: bool = True,
        pmlib_server_ip: str = "127.0.0.1",
        pmlib_server_port: int = 6526,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the EnergyAnalyzer.

        Args:
            model: PyTorch model for inference (optional, can be set later)
            device: Device for inference ('cpu' or 'cuda')
            warmup_runs: Number of warmup iterations before measurement
            measurement_runs: Number of measurement runs per image
            enable_gpu_energy: Enable GPU energy measurement
            enable_cpu_energy: Enable CPU energy measurement
            pmlib_server_ip: PMLib server IP (for Jetson)
            pmlib_server_port: PMLib server port (for Jetson)
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger("ASPIR.EnergyAnalyzer")

        self.model = model
        self.device = device
        self.warmup_runs = warmup_runs
        self.measurement_runs = measurement_runs

        self._energy_monitor = EnergyMonitor(
            enable_gpu=enable_gpu_energy,
            enable_cpu=enable_cpu_energy,
            enable_jetson=enable_gpu_energy,  # Jetson uses GPU measurement path
            pmlib_server_ip=pmlib_server_ip,
            pmlib_server_port=pmlib_server_port,
            logger=self.logger
        )

        self._is_initialized = False
        self._last_result: Optional[EnergyAnalysisResult] = None

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    @property
    def available_backends(self) -> List[str]:
        """List available energy measurement backends."""
        return self._energy_monitor.available_backends

    @property
    def platform_info(self) -> Dict[str, Any]:
        """Get platform detection information."""
        return self._energy_monitor.detect_platform()

    def initialize(self) -> bool:
        """
        Initialize energy monitoring backends.

        Returns:
            True if at least one backend is available
        """
        if self._is_initialized:
            return True

        self._is_initialized = self._energy_monitor.initialize()

        if self._is_initialized:
            self.logger.info(f"Energy analyzer initialized with backends: {self.available_backends}")
        else:
            self.logger.warning("No energy measurement backends available")

        return self._is_initialized

    def set_model(self, model: Any, device: str = 'cpu') -> None:
        """Set or update the model for inference."""
        self.model = model
        self.device = device

    def analyze_inference(
        self,
        input_tensors: Union[Any, List[Any]],
        n_runs: Optional[int] = None,
        warmup_runs: Optional[int] = None,
        return_per_image: bool = True
    ) -> EnergyAnalysisResult:
        """
        Analyze energy consumption during model inference.

        Args:
            input_tensors: Single tensor or list of tensors to process
            n_runs: Number of measurement runs per image (overrides default)
            warmup_runs: Number of warmup runs (overrides default)
            return_per_image: If True, measure each image separately

        Returns:
            EnergyAnalysisResult with detailed measurements
        """
        if not self._is_initialized:
            if not self.initialize():
                raise RuntimeError("Energy analyzer could not be initialized")

        if self.model is None:
            raise RuntimeError("No model set for inference")

        n_runs = n_runs or self.measurement_runs
        warmup_runs = warmup_runs or self.warmup_runs

        # Prepare model
        self.model.eval()

        # Determine device
        device = torch.device(self.device) if TORCH_AVAILABLE else None

        # Normalize input to list
        if not isinstance(input_tensors, (list, tuple)):
            input_tensors = [input_tensors]

        energy_readings = []
        power_readings = []
        time_readings = []
        temp_readings = []
        # Accumulate every backend reading across all images / sub-runs
        # so the CPU/GPU component aggregation below sees the full set
        # rather than just the final loop iteration. (Earlier code
        # relied on ``result.readings`` from the last call and
        # mis-attributed energy when callers passed multiple images
        # for variance.)
        all_readings = []

        # Define inference function
        def run_inference(tensor):
            with torch.no_grad():
                return self.model(tensor)

        # Setup CUDA sync
        def cuda_sync():
            if TORCH_AVAILABLE and torch.cuda.is_available() and self.device != 'cpu':
                torch.cuda.synchronize()

        self.logger.info(f"Starting energy analysis: {len(input_tensors)} images, {n_runs} runs each")

        for idx, tensor in enumerate(input_tensors):
            # Ensure tensor is on correct device
            if TORCH_AVAILABLE and device is not None:
                if hasattr(tensor, 'to'):
                    tensor = tensor.to(device)

                # Add batch dimension if needed
                if tensor.dim() == 3:
                    tensor = tensor.unsqueeze(0)

            # Warmup for first image or if return_per_image
            if idx == 0 or return_per_image:
                self.logger.debug(f"Warmup: {warmup_runs} runs")
                with torch.no_grad():
                    for _ in range(warmup_runs):
                        _ = self.model(tensor)
                        cuda_sync()
                time.sleep(0.1)  # Stabilization

            # Measure energy
            cuda_sync()
            result = self._energy_monitor.measure(
                run_inference,
                tensor,
                n_iterations=n_runs,
                warmup_iterations=0,  # Already did warmup
                cuda_sync=True
            )

            # Extract measurements
            avg_energy = result.avg_energy_per_iteration_joules
            avg_time = result.execution_time_ms / n_runs
            avg_power = result.avg_power_watts

            energy_readings.append(avg_energy)
            time_readings.append(avg_time)
            power_readings.append(avg_power)
            all_readings.extend(result.readings)

            # Get temperature if available
            for reading in result.readings:
                if reading.temperature_celsius is not None:
                    temp_readings.append(reading.temperature_celsius)
                    break

            self.logger.debug(
                f"Image {idx + 1}/{len(input_tensors)}: "
                f"energy={avg_energy * 1000:.3f} mJ, "
                f"power={avg_power:.2f} W, "
                f"time={avg_time:.2f} ms"
            )

        # Calculate statistics
        energy_arr = np.array(energy_readings)
        power_arr = np.array(power_readings)
        time_arr = np.array(time_readings)

        # Get device info from first reading
        device_type = DeviceType.UNKNOWN
        device_name = "Unknown"
        gpu_energy = None
        cpu_energy = None

        if all_readings:
            first_reading = all_readings[0]
            device_type = first_reading.device_type
            device_name = first_reading.device_name

            # Sum component energies across every sub-run / image and
            # divide by the total number of inference iterations so the
            # result is energy-per-iteration regardless of how the
            # caller chose to slice the workload.
            total_gpu = sum(r.gpu_energy_joules or 0 for r in all_readings)
            total_cpu = sum(r.cpu_energy_joules or 0 for r in all_readings)
            total_iterations = n_runs * len(input_tensors)
            if total_gpu > 0:
                gpu_energy = total_gpu / total_iterations
            if total_cpu > 0:
                cpu_energy = total_cpu / total_iterations

        analysis_result = EnergyAnalysisResult(
            energy_per_image_joules=energy_readings,
            power_per_image_watts=power_readings,
            time_per_image_ms=time_readings,
            mean_energy_joules=float(np.mean(energy_arr)),
            std_energy_joules=float(np.std(energy_arr)),
            mean_power_watts=float(np.mean(power_arr)),
            std_power_watts=float(np.std(power_arr)),
            mean_time_ms=float(np.mean(time_arr)),
            std_time_ms=float(np.std(time_arr)),
            device_type=device_type,
            device_name=device_name,
            temperature_per_image=temp_readings if temp_readings else None,
            mean_temperature=float(np.mean(temp_readings)) if temp_readings else None,
            gpu_energy_joules=gpu_energy,
            cpu_energy_joules=cpu_energy
        )

        self._last_result = analysis_result
        self.logger.info(
            f"Energy analysis complete: "
            f"mean={analysis_result.mean_energy_mj:.3f} mJ, "
            f"power={analysis_result.mean_power_watts:.2f} W"
        )

        return analysis_result

    def measure_reconstruction_energy(
        self,
        reconstruction_func: Callable,
        *args,
        n_iterations: int = 1,
        **kwargs
    ) -> EnergyMeasurementResult:
        """
        Measure energy during reconstruction (non-neural network operations).

        Args:
            reconstruction_func: Function to measure
            *args: Arguments to pass to function
            n_iterations: Number of iterations
            **kwargs: Keyword arguments

        Returns:
            EnergyMeasurementResult
        """
        if not self._is_initialized:
            if not self.initialize():
                raise RuntimeError("Energy analyzer could not be initialized")

        return self._energy_monitor.measure(
            reconstruction_func,
            *args,
            n_iterations=n_iterations,
            warmup_iterations=self.warmup_runs,
            **kwargs
        )

    def get_current_power(self) -> Dict[str, float]:
        """Get current power readings from all backends."""
        if not self._is_initialized:
            return {}
        return self._energy_monitor.get_current_power()

    def shutdown(self) -> None:
        """Clean up resources."""
        if self._is_initialized:
            self._energy_monitor.shutdown()
            self._is_initialized = False
            self.logger.debug("Energy analyzer shutdown complete")

    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
        return False
