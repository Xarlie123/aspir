"""
Worker for background energy measurement.

Runs energy analysis in a separate thread to prevent GUI freezing.
"""
import logging
import time
from typing import Optional, List, Dict, Any

from PyQt5.QtCore import QObject, pyqtSignal

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False


class EnergyMeasurementWorker(QObject):
    """
    Worker for running energy measurements in a background thread.

    Signals:
        progress: Emitted with (current, total, message) during measurement
        finished: Emitted when measurement is complete
        error: Emitted with exception if an error occurs
        result: Emitted with the energy_data dictionary
        platform_detected: Emitted with (platform_info, backends) after detection
    """

    progress = pyqtSignal(int, int, str)  # current, total, message
    finished = pyqtSignal()
    error = pyqtSignal(Exception)
    result = pyqtSignal(dict)  # energy_data dictionary
    platform_detected = pyqtSignal(dict, list)  # platform_info, backend_names

    def __init__(
        self,
        mode: str = "measure",
        model=None,
        test_images: Optional[List] = None,
        device: str = "cpu",
        warmup_runs: int = 5,
        measurement_runs: int = 10,
        enable_gpu_energy: bool = True,
        enable_cpu_energy: bool = False,
        pmlib_server_ip: str = "127.0.0.1",
        pmlib_server_port: int = 6526,
        logger: Optional[logging.Logger] = None,
        parent=None
    ):
        """
        Initialize the energy measurement worker.

        Args:
            mode: "detect" for platform detection, "measure" for energy measurement
            model: PyTorch model for inference (required for "measure" mode)
            test_images: List of test images/tensors (required for "measure" mode)
            device: Device for inference ('cpu' or 'cuda')
            warmup_runs: Number of warmup iterations
            measurement_runs: Number of measurement runs per image
            enable_gpu_energy: Enable GPU energy measurement
            enable_cpu_energy: Enable CPU energy measurement
            pmlib_server_ip: PMLib server IP
            pmlib_server_port: PMLib server port
            logger: Logger instance
            parent: Parent QObject
        """
        super().__init__(parent)

        self.mode = mode
        self.model = model
        self.test_images = test_images
        self.device = device
        self.warmup_runs = warmup_runs
        self.measurement_runs = measurement_runs
        self.enable_gpu_energy = enable_gpu_energy
        self.enable_cpu_energy = enable_cpu_energy
        self.pmlib_server_ip = pmlib_server_ip
        self.pmlib_server_port = pmlib_server_port

        if logger is not None:
            self.logger = logger.getChild("EnergyWorker")
        else:
            self.logger = logging.getLogger("ASPIR.EnergyWorker")
        self.logger.setLevel(logging.DEBUG)

        self._is_cancelled = False

    def cancel(self):
        """Request cancellation of the measurement."""
        self._is_cancelled = True
        self.logger.info("Cancellation requested")

    def run(self):
        """Execute the measurement or detection."""
        self.logger.debug(f"EnergyWorker run() started in mode: {self.mode}")

        try:
            if self.mode == "detect":
                self._run_detection()
            elif self.mode == "measure":
                self._run_measurement()
            else:
                raise ValueError(f"Unknown mode: {self.mode}")

        except Exception as e:
            self.logger.error(f"Exception in run(): {e}", exc_info=True)
            self.error.emit(e)

        self.logger.debug("run() exiting")

    def _run_detection(self):
        """Detect available energy measurement backends."""
        self.progress.emit(0, 100, "Detecting energy backends...")

        from simulation_engine._5_analyzer.energy_backends import EnergyMonitor

        monitor = EnergyMonitor(
            enable_gpu=self.enable_gpu_energy,
            enable_cpu=self.enable_cpu_energy,
            enable_jetson=self.enable_gpu_energy,
            pmlib_server_ip=self.pmlib_server_ip,
            pmlib_server_port=self.pmlib_server_port,
            logger=self.logger
        )

        self.progress.emit(30, 100, "Scanning hardware...")
        platform_info = monitor.detect_platform()

        self.progress.emit(60, 100, "Initializing backends...")
        monitor.initialize()

        backends = monitor.available_backends
        self.progress.emit(90, 100, "Detection complete")

        # Clean up
        monitor.shutdown()

        self.progress.emit(100, 100, f"Found {len(backends)} backend(s)")
        self.platform_detected.emit(platform_info, backends)
        self.finished.emit()

    def _run_measurement(self):
        """Run the energy measurement."""
        if self.model is None:
            raise RuntimeError("No model provided for energy measurement")

        if self.test_images is None or len(self.test_images) == 0:
            raise RuntimeError("No test images provided for energy measurement")

        from simulation_engine._5_analyzer.analyzer_energy import EnergyAnalyzer

        total_images = len(self.test_images)
        self.progress.emit(0, total_images + 2, "Initializing energy analyzer...")

        # Create energy analyzer
        analyzer = EnergyAnalyzer(
            model=self.model,
            device=self.device,
            warmup_runs=self.warmup_runs,
            measurement_runs=self.measurement_runs,
            enable_gpu_energy=self.enable_gpu_energy,
            enable_cpu_energy=self.enable_cpu_energy,
            pmlib_server_ip=self.pmlib_server_ip,
            pmlib_server_port=self.pmlib_server_port,
            logger=self.logger
        )

        if not analyzer.initialize():
            raise RuntimeError("Failed to initialize energy analyzer. No backends available.")

        self.progress.emit(1, total_images + 2, "Backends initialized")

        if self._is_cancelled:
            analyzer.shutdown()
            return

        # Prepare model
        self.model.eval()
        torch_device = torch.device(self.device) if TORCH_AVAILABLE else None

        # Get list of active backends for per-backend tracking
        active_backends = analyzer._energy_monitor.available_backends
        self.logger.debug(f"Active backends for measurement: {active_backends}")

        # Measure energy for each image - track per backend
        # Structure: {backend_name: {'energy': [], 'power': [], 'temp': []}}
        per_backend_data = {name: {'energy': [], 'power': [], 'temp': []} for name in active_backends}
        time_readings = []  # Time is shared across backends

        self.progress.emit(1, total_images + 2, "Starting measurements...")

        for idx, image in enumerate(self.test_images):
            if self._is_cancelled:
                self.logger.info("Measurement cancelled")
                analyzer.shutdown()
                return

            self.progress.emit(idx + 2, total_images + 2, f"Measuring image {idx + 1}/{total_images}...")

            # Prepare tensor
            if TORCH_AVAILABLE and torch_device is not None:
                if hasattr(image, 'to'):
                    tensor = image.to(torch_device)
                else:
                    tensor = torch.tensor(image, device=torch_device, dtype=torch.float32)

                if tensor.dim() == 2:
                    tensor = tensor.unsqueeze(0).unsqueeze(0)
                elif tensor.dim() == 3:
                    tensor = tensor.unsqueeze(0)
            else:
                tensor = image

            # Measure energy
            try:
                result = analyzer._energy_monitor.measure(
                    lambda t=tensor: self._inference_func(t),
                    n_iterations=self.measurement_runs,
                    warmup_iterations=self.warmup_runs if idx == 0 else 0,
                    cuda_sync=True
                )

                avg_time = result.execution_time_ms / self.measurement_runs
                time_readings.append(avg_time)

                # Store per-backend readings
                for reading in result.readings:
                    backend_name = reading.device_name
                    if backend_name in per_backend_data:
                        # Energy per iteration for this backend
                        energy_per_iter = reading.energy_joules / self.measurement_runs
                        per_backend_data[backend_name]['energy'].append(energy_per_iter)
                        per_backend_data[backend_name]['power'].append(reading.avg_power_watts)
                        if reading.temperature_celsius is not None:
                            per_backend_data[backend_name]['temp'].append(reading.temperature_celsius)

                # Log total energy
                total_energy = result.avg_energy_per_iteration_joules
                self.logger.debug(
                    f"Image {idx + 1}: total_energy={total_energy * 1000:.3f} mJ, "
                    f"time={avg_time:.2f} ms"
                )

            except Exception as e:
                self.logger.warning(f"Failed to measure image {idx + 1}: {e}")
                time_readings.append(0.0)
                # Add zeros for all backends
                for backend_name in per_backend_data:
                    per_backend_data[backend_name]['energy'].append(0.0)
                    per_backend_data[backend_name]['power'].append(0.0)

        # Calculate statistics
        import numpy as np
        time_arr = np.array(time_readings)

        # Build per-backend statistics
        backends_stats = {}
        total_energy_per_image = np.zeros(total_images)

        for backend_name, data in per_backend_data.items():
            energy_arr = np.array(data['energy'])
            power_arr = np.array(data['power'])
            temp_arr = np.array(data['temp']) if data['temp'] else None

            # Accumulate total energy
            total_energy_per_image += energy_arr

            # Determine backend type (GPU or CPU)
            is_gpu = any(kw in backend_name.lower() for kw in ['nvidia', 'geforce', 'rtx', 'gtx', 'quadro', 'tesla'])
            backend_type = 'gpu' if is_gpu else 'cpu'

            backends_stats[backend_name] = {
                'type': backend_type,
                'energy_per_image_mj': [e * 1000 for e in energy_arr],
                'power_per_image_watts': power_arr.tolist(),
                'mean_energy_mj': float(np.mean(energy_arr)) * 1000,
                'std_energy_mj': float(np.std(energy_arr)) * 1000,
                'mean_power_watts': float(np.mean(power_arr)),
                'std_power_watts': float(np.std(power_arr)),
                'mean_temperature': float(np.mean(temp_arr)) if temp_arr is not None and len(temp_arr) > 0 else None,
            }

        # Count zero readings (measurement too short)
        zero_count = sum(1 for e in total_energy_per_image if e == 0)
        has_zero_readings = zero_count > 0

        # Calculate totals
        mean_total_energy_j = float(np.mean(total_energy_per_image))
        total_power_per_image = sum(
            np.array(data['power']) for data in per_backend_data.values()
        )

        # Build result dictionary with both total and per-backend data
        energy_data = {
            # Total (combined) metrics for backward compatibility
            'energy_per_image_mj': [e * 1000 for e in total_energy_per_image],
            'power_per_image_watts': total_power_per_image.tolist() if isinstance(total_power_per_image, np.ndarray) else [],
            'time_per_image_ms': time_readings,
            'mean_energy_mj': mean_total_energy_j * 1000,
            'std_energy_mj': float(np.std(total_energy_per_image)) * 1000,
            'mean_power_watts': float(np.mean(total_power_per_image)) if len(total_power_per_image) > 0 else 0,
            'std_power_watts': float(np.std(total_power_per_image)) if len(total_power_per_image) > 0 else 0,
            'mean_time_ms': float(np.mean(time_arr)),
            'std_time_ms': float(np.std(time_arr)),
            'device_name': ', '.join(active_backends),
            'efficiency_images_per_joule': 1.0 / mean_total_energy_j if mean_total_energy_j > 0 else 0,
            'n_images': total_images,
            'measurement_runs': self.measurement_runs,
            'warmup_runs': self.warmup_runs,
            'zero_readings_count': zero_count,
            'has_warning': has_zero_readings,
            # NEW: Per-backend breakdown
            'backends': backends_stats,
            'backend_names': active_backends,
        }

        if has_zero_readings:
            self.logger.warning(
                f"{zero_count}/{total_images} images had 0 energy reading. "
                "Measurements are too short - increase 'measurement_runs' parameter."
            )

        # Clean up
        analyzer.shutdown()

        self.progress.emit(total_images + 2, total_images + 2, "Measurement complete")
        self.result.emit(energy_data)
        self.finished.emit()

    def _inference_func(self, tensor):
        """Run model inference with proper synchronization."""
        with torch.no_grad():
            output = self.model(tensor)
            if TORCH_AVAILABLE and torch.cuda.is_available() and self.device != 'cpu':
                torch.cuda.synchronize()
            return output
