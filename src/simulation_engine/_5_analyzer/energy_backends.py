"""
Energy measurement backends with auto-detection for different hardware platforms.

Supported backends:
- NVMLBackend: NVIDIA GPUs (desktop/workstation) via pynvml
- JetsonSysfsBackend: NVIDIA Jetson devices via sysfs (pure Python, no PMLib needed)
- RAPLBackend: Intel CPUs via RAPL (Running Average Power Limit)

The EnergyMonitor class auto-detects available backends and provides a unified
interface for energy measurement during inference.
"""

import logging
import os
import platform
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Callable, Any, Tuple

import numpy as np


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
    readings: List[EnergyReading]
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
    ) -> Tuple[Any, EnergyReading]:
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
        self._power_samples: List[Tuple[float, float]] = []  # (timestamp, power_mw)
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
        except:
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
            except:
                pass
            self._is_initialized = False
            self.logger.debug("NVML shutdown complete")


class RAPLBackend(EnergyBackend):
    """
    Energy measurement backend for Intel CPUs using RAPL (Running Average Power Limit).

    Reads energy counters from /sys/class/powercap/intel-rapl/ on Linux.
    Requires read access to the RAPL sysfs interface.
    """

    RAPL_PATH = Path("/sys/class/powercap/intel-rapl")

    def __init__(self, logger: Optional[logging.Logger] = None):
        super().__init__(logger)
        self._domains: Dict[str, Path] = {}  # domain_name -> energy_uj path
        self._start_readings: Dict[str, int] = {}
        self._max_energy: Dict[str, int] = {}
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
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":")[1].strip()
        except:
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


class JetsonSysfsBackend(EnergyBackend):
    """
    Pure Python energy measurement backend for NVIDIA Jetson devices.

    Reads power sensors directly from sysfs without requiring PMLib.
    Supports Orin, Nano, Xavier, and TX2 devices.

    Power rails are read from /sys/bus/i2c/drivers/ina3221[x]/
    """

    # Jetson device configurations
    JETSON_CONFIGS = {
        DeviceType.JETSON_ORIN: {
            "driver_path": "/sys/bus/i2c/drivers/ina3221/",
            "rails": ["VDD_GPU_SOC", "VDD_CPU_CV", "VIN_SYS_5V0"],
            "device_name": "Jetson-Orin"
        },
        DeviceType.JETSON_NANO: {
            "driver_path": "/sys/bus/i2c/drivers/ina3221x/",
            "rails": ["POM_5V_IN", "POM_5V_GPU", "POM_5V_CPU"],
            "device_name": "Jetson-Nano"
        },
        DeviceType.JETSON_XAVIER: {
            "driver_path": "/sys/bus/i2c/drivers/ina3221x/",
            "rails": ["GPU", "CPU", "SOC", "CV", "VDDRQ", "SYS5V"],
            "device_name": "Jetson-Xavier"
        },
        DeviceType.JETSON_TX2: {
            "driver_path": "/sys/bus/i2c/drivers/ina3221x/",
            "rails": ["VDD_SYS_GPU", "VDD_SYS_SOC", "VDD_4V0_WIFI",
                      "VDD_IN", "VDD_SYS_CPU", "VDD_SYS_DDR"],
            "device_name": "Jetson-TX2"
        }
    }

    def __init__(self, logger: Optional[logging.Logger] = None):
        super().__init__(logger)
        self._power_paths: Dict[str, Path] = {}  # rail_name -> power_path
        self._voltage_paths: Dict[str, Path] = {}  # rail_name -> voltage_path
        self._current_paths: Dict[str, Path] = {}  # rail_name -> current_path
        self._measurement_start_time = 0
        self._power_samples: List[Tuple[float, Dict[str, float]]] = []
        self._sampling_interval = 0.01  # 10ms sampling

    def _detect_jetson_type(self) -> Optional[DeviceType]:
        """Auto-detect Jetson device type from system info."""
        # Check if this is a Jetson device
        if not Path("/etc/nv_tegra_release").exists():
            return None

        # Check device tree model
        model_path = Path("/proc/device-tree/model")
        if model_path.exists():
            try:
                model = model_path.read_text().lower()
                if "orin" in model:
                    return DeviceType.JETSON_ORIN
                elif "xavier" in model:
                    return DeviceType.JETSON_XAVIER
                elif "tx2" in model:
                    return DeviceType.JETSON_TX2
                elif "nano" in model:
                    return DeviceType.JETSON_NANO
            except:
                pass

        # Fallback: check which i2c driver exists
        if Path("/sys/bus/i2c/drivers/ina3221/").exists():
            return DeviceType.JETSON_ORIN
        elif Path("/sys/bus/i2c/drivers/ina3221x/").exists():
            return DeviceType.JETSON_NANO  # Default to Nano for older driver

        return None

    def _find_power_sensors(self, driver_path: str) -> Dict[str, Dict[str, Path]]:
        """
        Find all power sensor paths under the INA3221 driver.

        Returns dict: {rail_name: {'power': Path, 'voltage': Path, 'current': Path}}
        """
        sensors = {}
        driver_dir = Path(driver_path)

        if not driver_dir.exists():
            return sensors

        # Find all i2c device directories
        for device_dir in driver_dir.iterdir():
            if not device_dir.is_dir():
                continue

            # Look for hwmon subdirectory
            hwmon_dir = device_dir / "hwmon"
            if not hwmon_dir.exists():
                # Try iio subdirectory for some Jetson models
                iio_dir = device_dir / "iio:device0"
                if iio_dir.exists():
                    hwmon_dir = iio_dir

            if hwmon_dir.exists():
                # Find the hwmon subdirectory (e.g., hwmon0, hwmon1, etc.)
                for hwmon_subdir in hwmon_dir.iterdir():
                    if hwmon_subdir.name.startswith("hwmon"):
                        self._scan_hwmon_dir(hwmon_subdir, sensors)
                        break
            else:
                # Direct sensor files in device directory
                self._scan_hwmon_dir(device_dir, sensors)

        return sensors

    def _scan_hwmon_dir(self, hwmon_dir: Path, sensors: Dict):
        """Scan a hwmon directory for power sensor files."""
        # Look for power/voltage/current files with labels
        # Pattern: in[0-9]_input, curr[0-9]_input, power[0-9]_input
        # Labels: in[0-9]_label, curr[0-9]_label, power[0-9]_label

        for label_file in hwmon_dir.glob("*_label"):
            try:
                label = label_file.read_text().strip()
                base_name = label_file.name.replace("_label", "")

                # Find corresponding input file
                input_file = hwmon_dir / f"{base_name}_input"
                if input_file.exists():
                    if label not in sensors:
                        sensors[label] = {}

                    if base_name.startswith("in"):
                        sensors[label]['voltage'] = input_file
                    elif base_name.startswith("curr"):
                        sensors[label]['current'] = input_file
                    elif base_name.startswith("power"):
                        sensors[label]['power'] = input_file
            except:
                continue

        # Also look for rail_name files (alternative format)
        for rail_file in hwmon_dir.glob("rail_name_*"):
            try:
                idx = rail_file.name.split("_")[-1]
                label = rail_file.read_text().strip()

                power_file = hwmon_dir / f"in_power{idx}_input"
                voltage_file = hwmon_dir / f"in_voltage{idx}_input"
                current_file = hwmon_dir / f"in_current{idx}_input"

                if label not in sensors:
                    sensors[label] = {}

                if power_file.exists():
                    sensors[label]['power'] = power_file
                if voltage_file.exists():
                    sensors[label]['voltage'] = voltage_file
                if current_file.exists():
                    sensors[label]['current'] = current_file
            except:
                continue

    def initialize(self) -> bool:
        """Initialize by finding power sensor sysfs paths."""
        detected_type = self._detect_jetson_type()

        if detected_type is None:
            self.logger.info("Not running on a Jetson device")
            return False

        self._device_type = detected_type
        config = self.JETSON_CONFIGS.get(detected_type)

        if config is None:
            self.logger.error(f"No configuration for device type: {detected_type}")
            return False

        # Find power sensors
        sensors = self._find_power_sensors(config["driver_path"])

        if not sensors:
            # Try alternative path for some Jetson models
            alt_paths = [
                "/sys/bus/i2c/devices/",
                "/sys/class/hwmon/"
            ]
            for alt_path in alt_paths:
                sensors = self._find_power_sensors(alt_path)
                if sensors:
                    break

        if not sensors:
            self.logger.warning(f"No power sensors found for {config['device_name']}")
            return False

        # Store paths for later use
        for rail_name, paths in sensors.items():
            if 'power' in paths:
                self._power_paths[rail_name] = paths['power']
            if 'voltage' in paths:
                self._voltage_paths[rail_name] = paths['voltage']
            if 'current' in paths:
                self._current_paths[rail_name] = paths['current']

        self._device_name = config["device_name"]
        self._is_initialized = True

        self.logger.info(f"Jetson sysfs backend initialized: {self._device_name}")
        self.logger.info(f"Found {len(self._power_paths)} power rails: {list(self._power_paths.keys())}")

        return True

    def _read_power_mw(self, rail: str) -> float:
        """Read power in milliwatts from a rail."""
        try:
            if rail in self._power_paths:
                # Direct power reading (usually in microwatts)
                value = int(self._power_paths[rail].read_text().strip())
                return value / 1000.0  # Convert uW to mW
            elif rail in self._voltage_paths and rail in self._current_paths:
                # Calculate power from voltage and current
                voltage_mv = int(self._voltage_paths[rail].read_text().strip())
                current_ma = int(self._current_paths[rail].read_text().strip())
                return (voltage_mv * current_ma) / 1000.0  # mV * mA / 1000 = mW
        except (ValueError, FileNotFoundError, PermissionError) as e:
            self.logger.debug(f"Failed to read power for {rail}: {e}")
        return 0.0

    def get_current_power(self) -> float:
        """Get total current power in Watts."""
        if not self._is_initialized:
            return 0.0

        total_mw = 0.0
        for rail in self._power_paths:
            total_mw += self._read_power_mw(rail)

        # If no direct power readings, try voltage * current
        if total_mw == 0:
            for rail in self._voltage_paths:
                if rail in self._current_paths:
                    total_mw += self._read_power_mw(rail)

        return total_mw / 1000.0  # Convert mW to W

    def start_measurement(self) -> None:
        """Start energy measurement by recording initial power samples."""
        if not self._is_initialized:
            raise RuntimeError("Backend not initialized")

        self._power_samples = []
        self._measurement_start_time = time.perf_counter()

        # Take initial sample
        self._power_samples.append((
            time.perf_counter(),
            {rail: self._read_power_mw(rail) for rail in self._power_paths}
        ))

    def stop_measurement(self) -> EnergyReading:
        """Stop measurement and calculate energy using trapezoidal integration."""
        if not self._is_initialized:
            raise RuntimeError("Backend not initialized")

        t_end = time.perf_counter()

        # Take final sample
        self._power_samples.append((
            t_end,
            {rail: self._read_power_mw(rail) for rail in self._power_paths}
        ))

        # Calculate energy using trapezoidal rule
        # For just two samples (start and end), this is:
        # E = (P_start + P_end) / 2 * duration
        if len(self._power_samples) >= 2:
            start_time, start_powers = self._power_samples[0]
            end_time, end_powers = self._power_samples[-1]

            duration = end_time - start_time

            # Sum power across all rails
            start_total_mw = sum(start_powers.values())
            end_total_mw = sum(end_powers.values())

            avg_power_mw = (start_total_mw + end_total_mw) / 2
            energy_mj = avg_power_mw * duration  # mW * s = mJ

            energy_joules = energy_mj / 1000.0
            avg_power_watts = avg_power_mw / 1000.0
        else:
            duration = t_end - self._measurement_start_time
            energy_joules = 0.0
            avg_power_watts = 0.0

        return EnergyReading(
            energy_joules=energy_joules,
            avg_power_watts=avg_power_watts,
            duration_seconds=duration,
            device_type=self._device_type,
            device_name=self._device_name,
            gpu_energy_joules=energy_joules  # Jetson GPU is integrated
        )

    def shutdown(self) -> None:
        """No cleanup needed for sysfs backend."""
        self._is_initialized = False


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

        self._backends: List[EnergyBackend] = []
        self._is_initialized = False

    @property
    def available_backends(self) -> List[str]:
        """List of available backend names."""
        return [b.device_name for b in self._backends if b.is_available]

    @property
    def device_types(self) -> List[DeviceType]:
        """List of available device types."""
        return [b.device_type for b in self._backends if b.is_available]

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    def detect_platform(self) -> Dict[str, Any]:
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
                except:
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
            except:
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

    def stop_measurement(self) -> List[EnergyReading]:
        """Stop measurement and return readings from all backends."""
        readings = []
        for backend in self._backends:
            try:
                readings.append(backend.stop_measurement())
            except Exception as e:
                self.logger.error(f"Error stopping {backend.device_name}: {e}")
        return readings

    def get_current_power(self) -> Dict[str, float]:
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
