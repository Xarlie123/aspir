"""NVIDIA Jetson energy backend via sysfs INA3221 power sensors."""
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
        self._power_paths: dict[str, Path] = {}  # rail_name -> power_path
        self._voltage_paths: dict[str, Path] = {}  # rail_name -> voltage_path
        self._current_paths: dict[str, Path] = {}  # rail_name -> current_path
        self._measurement_start_time = 0
        self._power_samples: list[tuple[float, dict[str, float]]] = []
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
            except Exception:
                pass

        # Fallback: check which i2c driver exists
        if Path("/sys/bus/i2c/drivers/ina3221/").exists():
            return DeviceType.JETSON_ORIN
        elif Path("/sys/bus/i2c/drivers/ina3221x/").exists():
            return DeviceType.JETSON_NANO  # Default to Nano for older driver

        return None

    def _find_power_sensors(self, driver_path: str) -> dict[str, dict[str, Path]]:
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

    def _scan_hwmon_dir(self, hwmon_dir: Path, sensors: dict):
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
            except Exception:
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
            except Exception:
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
