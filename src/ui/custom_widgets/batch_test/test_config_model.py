"""
Data models for Batch Test configuration.
"""
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from enum import Enum, auto
from pathlib import Path
import json
import yaml  # For legacy file support
import os
from datetime import datetime

from ui.utils.file_formats import FileExtensions, BATCH_TESTS_DIR


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

    # Sweep-specific parameters
    sweep_bar_width: int = 2
    sweep_stride: int = 4
    sweep_angles: List[float] = field(default_factory=lambda: [0.0, 45.0, 90.0, 135.0])

    # Reconstruction method
    reconstruction_method: str = "conventional"  # conventional, pseudoinverse, fista, tv_norm

    # FISTA-specific parameters
    fista_lambda: float = 0.01
    fista_iterations: int = 100

    # TV-norm specific parameters
    tv_lambda: float = 0.1
    tv_iterations: int = 50

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

    # Dataset split configuration (percentages, must sum to 100)
    train_split: int = 80  # Train percentage
    val_split: int = 10    # Validation percentage
    test_split: int = 10   # Test percentage

    # Reports to generate (all enabled by default)
    reports: List[str] = field(default_factory=lambda: ["training_curves", "quality", "timing", "energy"])

    # Whether to include datasets (training, validation, test images) in export
    include_datasets: bool = True

    # Timing analysis parameters
    timing_warmup_runs: int = 5
    timing_measurement_runs: int = 800  # High value for accurate energy measurement
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
            # Sweep params
            "sweep_bar_width": self.sweep_bar_width,
            "sweep_stride": self.sweep_stride,
            "sweep_angles": self.sweep_angles.copy(),
            # Reconstruction
            "reconstruction_method": self.reconstruction_method,
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
            # Sweep
            sweep_bar_width=data.get("sweep_bar_width", 2),
            sweep_stride=data.get("sweep_stride", 4),
            sweep_angles=data.get("sweep_angles", [0.0, 45.0, 90.0, 135.0]),
            # Reconstruction
            reconstruction_method=data.get("reconstruction_method", "pseudoinverse"),
            fista_lambda=data.get("fista_lambda", 0.01),
            fista_iterations=data.get("fista_iterations", 100),
            tv_lambda=data.get("tv_lambda", 0.1),
            tv_iterations=data.get("tv_iterations", 50),
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

    # Available options for UI dropdowns
    MASK_TYPES = [
        "scatter",
        "hadamard_natural",
        "hadamard_scramble",
        "hadamard_cake_cutting",
        "hadamard_walsh_paley",
        "sweep",
        "cal_sal"
    ]
    RECONSTRUCTION_METHODS = ["conventional", "pseudoinverse", "fista", "tv_norm"]
    MODEL_NAMES = ["autoencoder", "dncnn", "u-net", "u-net-residual-attention", "residual_cnn",
                   "noise2void", "mobilenet_denoising", "dilatedcnn", "cgan denoising"]
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
        )
        for test_data in data.get("tests", []):
            config.tests.append(TestConfiguration.from_dict(test_data))
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
                "format_version": "2.1",
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
