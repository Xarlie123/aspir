"""
Data models for Batch Test configuration.
"""
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from enum import Enum, auto
from pathlib import Path
import json
import logging
import yaml  # For legacy file support
import os
from datetime import datetime

from ui.utils.file_formats import FileExtensions, BATCH_TESTS_DIR
from simulation_engine._3_applicator.applicator_fista import (
    DEFAULT_ITERATIONS as FISTA_DEFAULT_ITERATIONS,
    DEFAULT_LAMBDA as FISTA_DEFAULT_LAMBDA,
)
from simulation_engine._3_applicator.applicator_tv_norm import (
    DEFAULT_ITERATIONS as TV_DEFAULT_ITERATIONS,
    DEFAULT_LAMBDA as TV_DEFAULT_LAMBDA,
)


class TestStatus(Enum):
    """Status of a test in the batch."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExportLevel(Enum):
    """Export level options for batch test results."""
    REPORTS_ONLY = auto()           # Only CSV reports
    REPORTS_AND_MODELS = auto()     # Reports + trained models (.pt, ONNX)
    ALL_DATA = auto()               # Reports + models + datasets (images, masks)


# Current .batch_config format. 2.2: FISTA / TV lambda and iteration values
# are applied by the solvers. Before 2.2 they were stored but never used.
BATCH_CONFIG_FORMAT_VERSION = "2.2"

_log = logging.getLogger("BatchTestConfig")


def _solver_params_never_applied(version) -> bool:
    """True for configs written before 2.2 (or unversioned legacy YAML)."""
    try:
        major, minor = (int(p) for p in str(version).split(".")[:2])
    except (TypeError, ValueError):
        return True
    return (major, minor) < (2, 2)


def _apply_historical_solver_params(test) -> int:
    """Replace stored FISTA/TV values with those that actually ran; 1 if any changed."""
    before = (test.fista_lambda, test.fista_iterations, test.tv_lambda, test.tv_iterations)
    test.fista_lambda = FISTA_DEFAULT_LAMBDA
    test.fista_iterations = FISTA_DEFAULT_ITERATIONS
    test.tv_lambda = TV_DEFAULT_LAMBDA
    test.tv_iterations = TV_DEFAULT_ITERATIONS
    after = (test.fista_lambda, test.fista_iterations, test.tv_lambda, test.tv_iterations)
    return int(before != after)


@dataclass
class TestConfiguration:
    """Configuration for a single test in the batch."""
    # Identification
    name: str = "Test 1"

    # Mask configuration
    mask_type: str = "scatter"  # scatter, hadamard, sweep, etc.
    mask_seed: int = 42

    # Scatter-specific parameters
    scatter_point_density: float = 10.0  # Percentage of points (0-100)
    scatter_num_patterns: int = 100

    # Hadamard-specific parameters
    hadamard_min_idx: int = 0
    hadamard_max_idx: int = 1024  # Will be clamped to img_size²

    # Fourier (FSPI) parameters. sampling_ratio is M/N, i.e. patterns over
    # pixels; each coefficient costs fourier_phase_steps patterns.
    fourier_sampling_ratio: float = 0.25
    fourier_phase_steps: int = 4           # 4 = four-step, 3 = three-step
    fourier_spectrum_ordering: str = "circular"   # or "square"
    # Binary illumination for DMD projection: "none", "floyd_steinberg", "bayer".
    fourier_binarization: str = "none"
    fourier_dmd_upsampling: int = 1

    # Sweep-specific parameters
    sweep_bar_widths: List[int] = field(default_factory=lambda: [2, 2, 2, 2])
    sweep_strides: List[int] = field(default_factory=lambda: [4, 4, 4, 4])
    sweep_angles: List[float] = field(default_factory=lambda: [0.0, 45.0, 90.0, 135.0])

    # Reconstruction method
    reconstruction_method: str = "conventional"  # conventional, pseudoinverse, fista, tv_norm
    # Where the reconstruction runs: "auto" (CUDA if available), "cpu" or "cuda".
    reconstruction_device: str = "auto"

    # FISTA-specific parameters
    # Defaults are the values the solvers have always run with (see R1.3a).
    fista_lambda: float = FISTA_DEFAULT_LAMBDA
    fista_iterations: int = FISTA_DEFAULT_ITERATIONS

    # TV-norm specific parameters
    tv_lambda: float = TV_DEFAULT_LAMBDA
    tv_iterations: int = TV_DEFAULT_ITERATIONS

    # DNN configuration
    model_name: str = "u-net"
    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    dropout: float = 0.0
    loss_function: str = "MSE"  # MSE, L1, SmoothL1, Huber
    optimizer: str = "Adam"  # Adam, AdamW, SGD, RMSprop
    use_gpu: bool = True  # Use GPU if available

    # Model-specific architecture parameters (e.g. depth, base_channels, ...)
    architecture_config: Dict[str, Any] = field(default_factory=dict)

    # Dataset split configuration (percentages, must sum to 100)
    train_split: int = 80  # Train percentage
    val_split: int = 10    # Validation percentage
    test_split: int = 10   # Test percentage

    # Reports to generate (all enabled by default)
    reports: List[str] = field(default_factory=lambda: ["training_curves", "quality", "timing", "energy"])

    # Whether to include datasets (training, validation, test images) in export
    include_datasets: bool = True

    # Timing analysis parameters — 20 / 800 is the unified default
    # used across Single Test → Timing, Batch Test, and Re-measure.
    # Both numbers are deliberately on the high side: 20 warmup
    # iterations cover CUDA kernel JIT and cache warming, and 800
    # measurement runs give the energy backend a long enough window
    # to escape jtop / RAPL / NVML counter quantisation.
    timing_warmup_runs: int = 20
    timing_measurement_runs: int = 800
    timing_sampling_rate_khz: float = 10.752

    # Runtime state (not saved to config)
    status: TestStatus = field(default=TestStatus.PENDING, compare=False)
    progress: int = field(default=0, compare=False)
    results: Dict[str, Any] = field(default_factory=dict, compare=False)
    error_message: str = field(default="", compare=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for YAML serialization (excluding runtime state)."""
        return {
            "name": self.name,
            "mask_type": self.mask_type,
            "mask_seed": self.mask_seed,
            # Scatter params
            "scatter_point_density": self.scatter_point_density,
            "scatter_num_patterns": self.scatter_num_patterns,
            # Hadamard params
            "hadamard_min_idx": self.hadamard_min_idx,
            "hadamard_max_idx": self.hadamard_max_idx,
            # Fourier params
            "fourier_sampling_ratio": self.fourier_sampling_ratio,
            "fourier_phase_steps": self.fourier_phase_steps,
            "fourier_spectrum_ordering": self.fourier_spectrum_ordering,
            "fourier_binarization": self.fourier_binarization,
            "fourier_dmd_upsampling": self.fourier_dmd_upsampling,
            # Sweep params
            "sweep_bar_widths": self.sweep_bar_widths.copy(),
            "sweep_strides": self.sweep_strides.copy(),
            "sweep_angles": self.sweep_angles.copy(),
            # Reconstruction
            "reconstruction_method": self.reconstruction_method,
            "reconstruction_device": self.reconstruction_device,
            "fista_lambda": self.fista_lambda,
            "fista_iterations": self.fista_iterations,
            "tv_lambda": self.tv_lambda,
            "tv_iterations": self.tv_iterations,
            # DNN
            "model_name": self.model_name,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "dropout": self.dropout,
            "loss_function": self.loss_function,
            "optimizer": self.optimizer,
            "use_gpu": self.use_gpu,
            "architecture_config": dict(self.architecture_config),
            # Dataset split
            "train_split": self.train_split,
            "val_split": self.val_split,
            "test_split": self.test_split,
            "reports": self.reports.copy(),
            "include_datasets": self.include_datasets,
            # Timing analysis
            "timing_warmup_runs": self.timing_warmup_runs,
            "timing_measurement_runs": self.timing_measurement_runs,
            "timing_sampling_rate_khz": self.timing_sampling_rate_khz,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TestConfiguration":
        """Create from dictionary."""
        return cls(
            name=data.get("name", "Test"),
            mask_type=data.get("mask_type", "scatter"),
            mask_seed=data.get("mask_seed", 42),
            # Scatter
            scatter_point_density=data.get("scatter_point_density", 10.0),
            scatter_num_patterns=data.get("scatter_num_patterns", 100),
            # Hadamard
            hadamard_min_idx=data.get("hadamard_min_idx", 0),
            hadamard_max_idx=data.get("hadamard_max_idx", 1024),
            # Fourier
            fourier_sampling_ratio=data.get("fourier_sampling_ratio", 0.25),
            fourier_phase_steps=data.get("fourier_phase_steps", 4),
            fourier_spectrum_ordering=data.get("fourier_spectrum_ordering", "circular"),
            fourier_binarization=data.get("fourier_binarization", "none"),
            fourier_dmd_upsampling=data.get("fourier_dmd_upsampling", 1),
            # Sweep - support both new list format and legacy single-value format
            sweep_bar_widths=data.get("sweep_bar_widths",
                                      [data["sweep_bar_width"]] * len(data.get("sweep_angles", [0.0, 45.0, 90.0, 135.0]))
                                      if "sweep_bar_width" in data
                                      else [2] * len(data.get("sweep_angles", [0.0, 45.0, 90.0, 135.0]))),
            sweep_strides=data.get("sweep_strides",
                                   [data["sweep_stride"]] * len(data.get("sweep_angles", [0.0, 45.0, 90.0, 135.0]))
                                   if "sweep_stride" in data
                                   else [4] * len(data.get("sweep_angles", [0.0, 45.0, 90.0, 135.0]))),
            sweep_angles=data.get("sweep_angles", [0.0, 45.0, 90.0, 135.0]),
            # Reconstruction
            reconstruction_method=data.get("reconstruction_method", "pseudoinverse"),
            reconstruction_device=data.get("reconstruction_device", "auto"),
            fista_lambda=data.get("fista_lambda", FISTA_DEFAULT_LAMBDA),
            fista_iterations=data.get("fista_iterations", FISTA_DEFAULT_ITERATIONS),
            tv_lambda=data.get("tv_lambda", TV_DEFAULT_LAMBDA),
            tv_iterations=data.get("tv_iterations", TV_DEFAULT_ITERATIONS),
            # DNN
            model_name=data.get("model_name", "u-net"),
            epochs=data.get("epochs", 50),
            batch_size=data.get("batch_size", 16),
            learning_rate=data.get("learning_rate", 0.001),
            weight_decay=data.get("weight_decay", 0.0001),
            dropout=data.get("dropout", 0.0),
            loss_function=data.get("loss_function", "MSE"),
            optimizer=data.get("optimizer", "Adam"),
            use_gpu=data.get("use_gpu", True),
            architecture_config=dict(data.get("architecture_config", {})),
            # Dataset split
            train_split=data.get("train_split", 80),
            val_split=data.get("val_split", 10),
            test_split=data.get("test_split", 10),
            reports=data.get("reports", ["training_curves", "quality", "timing", "energy"]),
            include_datasets=data.get("include_datasets", True),
            # Timing analysis
            timing_warmup_runs=data.get("timing_warmup_runs", 5),
            timing_measurement_runs=data.get("timing_measurement_runs", 800),
            timing_sampling_rate_khz=data.get("timing_sampling_rate_khz", 10.752),
        )

    def copy(self) -> "TestConfiguration":
        """Create a copy of this configuration."""
        new_config = TestConfiguration.from_dict(self.to_dict())
        return new_config


@dataclass
class BatchTestConfig:
    """Configuration for a batch of tests."""
    name: str = "Batch Test"
    description: str = ""
    tests: List[TestConfiguration] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Execution options
    parallel_execution: bool = False  # Run tests in parallel (faster but uses more resources)
    parallel_threads: int = 2  # Number of parallel threads when parallel_execution is True
    # Note: Timing and energy analysis always run sequentially for accurate measurements

    # Idle-baseline capture (energy backend) — measures system idle
    # power for ``baseline_duration_s`` before the first test, then
    # subtracts that pedestal from each test's energy to derive
    # "dynamic" power/energy/efficiency. Off → totals only, dynamic
    # columns blank. See ``simulation_engine/_5_analyzer/baseline_capture.py``.
    capture_baseline: bool = True
    baseline_duration_s: int = 60

    # Available options for UI dropdowns
    MASK_TYPES = [
        "scatter",
        "hadamard_natural",
        "hadamard_cake_cutting",
        "hadamard_walsh_paley",
        "sweep",
        "cal_sal",
        "fourier"
    ]
    RECONSTRUCTION_METHODS = ["conventional", "pseudoinverse", "fista", "tv_norm"]
    MODEL_NAMES = ["autoencoder", "dncnn", "u-net", "u-net-residual", "u-net-residual-attention",
                   "residual-cnn", "noise2void", "mobilenet-denoising", "dilated-cnn", "cgan-denoising"]
    LOSS_FUNCTIONS = ["MSE", "L1", "SmoothL1", "Huber"]
    OPTIMIZERS = ["Adam", "AdamW", "SGD", "RMSprop"]
    REPORT_TYPES = ["training_curves", "quality", "timing", "energy"]

    def add_test(self, config: Optional[TestConfiguration] = None) -> TestConfiguration:
        """Add a new test configuration."""
        if config is None:
            config = TestConfiguration(name=f"Test {len(self.tests) + 1}")
        self.tests.append(config)
        return config

    def remove_test(self, index: int) -> bool:
        """Remove test at index."""
        if 0 <= index < len(self.tests):
            del self.tests[index]
            return True
        return False

    def duplicate_test(self, index: int) -> Optional[TestConfiguration]:
        """Duplicate test at index."""
        if 0 <= index < len(self.tests):
            new_config = self.tests[index].copy()
            new_config.name = f"{new_config.name} (copy)"
            self.tests.insert(index + 1, new_config)
            return new_config
        return None

    def move_test(self, from_index: int, to_index: int) -> bool:
        """Move test from one position to another."""
        if 0 <= from_index < len(self.tests) and 0 <= to_index < len(self.tests):
            test = self.tests.pop(from_index)
            self.tests.insert(to_index, test)
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "parallel_execution": self.parallel_execution,
            "parallel_threads": self.parallel_threads,
            "capture_baseline": self.capture_baseline,
            "baseline_duration_s": self.baseline_duration_s,
            "tests": [t.to_dict() for t in self.tests],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BatchTestConfig":
        """Create from dictionary."""
        config = cls(
            name=data.get("name", "Batch Test"),
            description=data.get("description", ""),
            created_at=data.get("created_at", datetime.now().isoformat()),
            parallel_execution=data.get("parallel_execution", False),
            parallel_threads=data.get("parallel_threads", 2),
            capture_baseline=data.get("capture_baseline", True),
            baseline_duration_s=int(data.get("baseline_duration_s", 60)),
        )
        migrate = _solver_params_never_applied(data.get("format_version"))
        replaced = 0
        for test_data in data.get("tests", []):
            test = TestConfiguration.from_dict(test_data)
            if migrate:
                replaced += _apply_historical_solver_params(test)
            config.tests.append(test)
        if replaced:
            _log.warning(
                "Batch config format %s predates 2.2: its FISTA/TV lambda and iteration "
                "values were never applied by the solvers. Using the values that actually "
                "ran instead (FISTA lambda=%g, %d it; TV lambda=%g, %d it) in %d test(s).",
                data.get("format_version", "unversioned"),
                FISTA_DEFAULT_LAMBDA, FISTA_DEFAULT_ITERATIONS,
                TV_DEFAULT_LAMBDA, TV_DEFAULT_ITERATIONS, replaced)
        return config

    def save(self, filepath: str, dataset_info: Optional[Dict[str, Any]] = None) -> bool:
        """
        Save configuration to JSON file with .batch_config extension.

        Args:
            filepath: Path to save the configuration
            dataset_info: Optional dict with dataset information to include

        Returns:
            True if successful, False otherwise
        """
        try:
            path = Path(filepath)
            # Ensure correct extension
            if path.suffix != FileExtensions.BATCH_CONFIG:
                path = path.with_suffix(FileExtensions.BATCH_CONFIG)

            path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "format_version": BATCH_CONFIG_FORMAT_VERSION,
                **self.to_dict()
            }

            # Include dataset info if provided
            if dataset_info:
                data["dataset_info"] = dataset_info

            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            return True
        except Exception as e:
            print(f"Error saving batch config: {e}")
            return False

    @classmethod
    def load_with_dataset_info(cls, filepath: str) -> tuple:
        """
        Load configuration from JSON or legacy YAML file, returning dataset info.

        Args:
            filepath: Path to the configuration file

        Returns:
            Tuple of (BatchTestConfig, dataset_info dict or None)
        """
        try:
            path = Path(filepath)

            # Determine format based on extension
            if path.suffix == FileExtensions.BATCH_CONFIG or path.suffix == '.json':
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            elif path.suffix in ('.yaml', '.yml'):
                # Legacy YAML format
                with open(path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
            else:
                # Try JSON first, fallback to YAML
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                except json.JSONDecodeError:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = yaml.safe_load(f)

            # Extract dataset info if present
            dataset_info = data.pop("dataset_info", None)

            return cls.from_dict(data), dataset_info
        except Exception as e:
            print(f"Error loading batch config: {e}")
            return None, None

    @classmethod
    def load(cls, filepath: str) -> Optional["BatchTestConfig"]:
        """
        Load configuration from JSON or legacy YAML file.

        Args:
            filepath: Path to the configuration file

        Returns:
            BatchTestConfig instance or None if loading failed
        """
        try:
            path = Path(filepath)

            # Determine format based on extension
            if path.suffix == FileExtensions.BATCH_CONFIG or path.suffix == '.json':
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            elif path.suffix in ('.yaml', '.yml'):
                # Legacy YAML format
                with open(path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
            else:
                # Try JSON first, fallback to YAML
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                except json.JSONDecodeError:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = yaml.safe_load(f)

            return cls.from_dict(data)
        except Exception as e:
            print(f"Error loading batch config: {e}")
            return None

    @staticmethod
    def get_default_directory() -> Path:
        """Get the default directory for batch configs."""
        BATCH_TESTS_DIR.mkdir(parents=True, exist_ok=True)
        return BATCH_TESTS_DIR

    @staticmethod
    def get_file_filter() -> str:
        """Get file filter for open/save dialogs."""
        return (
            f"Batch Config (*{FileExtensions.BATCH_CONFIG});;"
            "Legacy YAML (*.yaml *.yml);;"
            "All Files (*.*)"
        )

    def get_completed_count(self) -> int:
        """Get number of completed tests."""
        return sum(1 for t in self.tests if t.status == TestStatus.COMPLETED)

    def get_pending_count(self) -> int:
        """Get number of pending tests."""
        return sum(1 for t in self.tests if t.status == TestStatus.PENDING)

    def reset_all_status(self):
        """Reset all test statuses to pending."""
        for test in self.tests:
            test.status = TestStatus.PENDING
            test.progress = 0
            test.results = {}
            test.error_message = ""
