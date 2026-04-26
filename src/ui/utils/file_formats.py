"""
Centralized file format definitions and output directory management.

This module defines all custom file extensions and provides utilities
for managing the experiments directory structure.
"""
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


# =============================================================================
# Directory Structure
# =============================================================================

def get_project_root() -> Path:
    """Get the project root directory (where .git is located)."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / ".git").exists():
            return parent
    # Fallback: assume src/ui/utils/file_formats.py structure
    return current.parent.parent.parent.parent


# Base output directory
OUTPUT_DIR = get_project_root() / "experiments"

# Subdirectories
SINGLE_TESTS_DIR = OUTPUT_DIR / "single_tests"
BATCH_TESTS_DIR = OUTPUT_DIR / "batch_tests"
MODELS_DIR = OUTPUT_DIR / "models"


# =============================================================================
# File Extensions
# =============================================================================

class FileExtensions:
    """Custom file extensions for the application."""

    # Configuration files (JSON format)
    SINGLE_TEST_CONFIG = ".single_test_config"
    BATCH_CONFIG = ".batch_config"

    # Report files (JSON format)
    BATCH_ANALYSIS_REPORT = ".batch_analysis_report"

    # Experiment files (contains datasets + config)
    SINGLE_TEST_EXPERIMENT = ".single_test_experiment"

    # Neural network exports
    PYTORCH_MODEL = ".pt"
    ONNX_MODEL = ".onnx"

    # Data files
    NUMPY_COMPRESSED = ".npz"


# =============================================================================
# File Format Handlers
# =============================================================================

class SingleTestConfig:
    """
    Handler for .single_test_config files.

    JSON format containing widget configurations for single test mode.
    """

    EXTENSION = FileExtensions.SINGLE_TEST_CONFIG

    @staticmethod
    def get_default_path(name: Optional[str] = None) -> Path:
        """Get default save path for a single test config."""
        if name is None:
            name = f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return SINGLE_TESTS_DIR / f"{name}{FileExtensions.SINGLE_TEST_CONFIG}"

    @staticmethod
    def save(data: Dict[str, Any], path: Optional[Path] = None, name: Optional[str] = None) -> Path:
        """
        Save single test configuration to JSON file.

        Args:
            data: Configuration dictionary
            path: Optional explicit path (overrides name)
            name: Optional name for auto-generated path

        Returns:
            Path where file was saved
        """
        if path is None:
            path = SingleTestConfig.get_default_path(name)

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

        return path

    @staticmethod
    def load(path: Path) -> Dict[str, Any]:
        """Load single test configuration from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)


class BatchConfig:
    """
    Handler for .batch_config files.

    JSON format containing batch test configurations.
    """

    EXTENSION = FileExtensions.BATCH_CONFIG

    @staticmethod
    def get_default_path(name: Optional[str] = None) -> Path:
        """Get default save path for a batch config."""
        if name is None:
            name = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return BATCH_TESTS_DIR / f"{name}{FileExtensions.BATCH_CONFIG}"

    @staticmethod
    def save(data: Dict[str, Any], path: Optional[Path] = None, name: Optional[str] = None) -> Path:
        """Save batch configuration to JSON file."""
        if path is None:
            path = BatchConfig.get_default_path(name)

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

        return path

    @staticmethod
    def load(path: Path) -> Dict[str, Any]:
        """Load batch configuration from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)


class BatchAnalysisReport:
    """
    Handler for .batch_analysis_report files.

    JSON format containing batch test results and metrics.
    """

    EXTENSION = FileExtensions.BATCH_ANALYSIS_REPORT

    @staticmethod
    def get_default_path(name: Optional[str] = None) -> Path:
        """Get default save path for a batch analysis report."""
        if name is None:
            name = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return BATCH_TESTS_DIR / f"{name}{FileExtensions.BATCH_ANALYSIS_REPORT}"

    @staticmethod
    def save(results: list, metadata: Optional[Dict[str, Any]] = None,
             path: Optional[Path] = None, name: Optional[str] = None) -> Path:
        """
        Save batch analysis report to JSON file.

        Args:
            results: List of test result dictionaries
            metadata: Optional metadata (timestamp, config info, etc.)
            path: Optional explicit path
            name: Optional name for auto-generated path

        Returns:
            Path where file was saved
        """
        if path is None:
            path = BatchAnalysisReport.get_default_path(name)

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "metadata": metadata or {
                "created_at": datetime.now().isoformat(),
                "version": "1.0",
            },
            "summary": {
                "total_tests": len(results),
                "completed": sum(1 for r in results if r.get("status") == "completed"),
                "failed": sum(1 for r in results if r.get("status") == "failed"),
                "cancelled": sum(1 for r in results if r.get("status") == "cancelled"),
            },
            "results": results,
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        return path

    @staticmethod
    def load(path: Path) -> Dict[str, Any]:
        """Load batch analysis report from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)


class SingleTestExperiment:
    """
    Handler for .single_test_experiment files.

    This is a directory-based format containing:
    - config.json: Widget configurations
    - datasets/: Training, validation, test datasets (NPZ)
    - masks/: Generated mask patterns (NPZ)
    - model/: Trained neural network (.pt, .onnx)
    - results/: Test results and metrics
    """

    EXTENSION = FileExtensions.SINGLE_TEST_EXPERIMENT

    @staticmethod
    def get_default_path(name: Optional[str] = None) -> Path:
        """Get default path for a single test experiment."""
        if name is None:
            name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return SINGLE_TESTS_DIR / f"{name}{FileExtensions.SINGLE_TEST_EXPERIMENT}"

    @staticmethod
    def create_structure(path: Path) -> Dict[str, Path]:
        """
        Create the experiment directory structure.

        Returns dict with paths to subdirectories.
        """
        path = Path(path)

        subdirs = {
            "root": path,
            "datasets": path / "datasets",
            "masks": path / "masks",
            "model": path / "model",
            "results": path / "results",
        }

        for subdir in subdirs.values():
            subdir.mkdir(parents=True, exist_ok=True)

        return subdirs

    @staticmethod
    def save_config(experiment_path: Path, config: Dict[str, Any]) -> Path:
        """Save experiment configuration."""
        config_path = Path(experiment_path) / "config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False, default=str)
        return config_path

    @staticmethod
    def load_config(experiment_path: Path) -> Dict[str, Any]:
        """Load experiment configuration."""
        config_path = Path(experiment_path) / "config.json"
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def get_manifest(experiment_path: Path) -> Dict[str, Any]:
        """Get experiment manifest with file inventory."""
        path = Path(experiment_path)

        manifest = {
            "path": str(path),
            "created_at": datetime.fromtimestamp(path.stat().st_ctime).isoformat(),
            "files": {},
        }

        # Check for config
        config_path = path / "config.json"
        if config_path.exists():
            manifest["files"]["config"] = str(config_path)

        # Check datasets
        datasets_dir = path / "datasets"
        if datasets_dir.exists():
            manifest["files"]["datasets"] = [
                str(f) for f in datasets_dir.glob("*.npz")
            ]

        # Check masks
        masks_dir = path / "masks"
        if masks_dir.exists():
            manifest["files"]["masks"] = [
                str(f) for f in masks_dir.glob("*.npz")
            ]

        # Check model
        model_dir = path / "model"
        if model_dir.exists():
            manifest["files"]["model"] = {
                "pt": [str(f) for f in model_dir.glob("*.pt")],
                "onnx": [str(f) for f in model_dir.glob("*.onnx")],
            }

        # Check results
        results_dir = path / "results"
        if results_dir.exists():
            manifest["files"]["results"] = [
                str(f) for f in results_dir.iterdir() if f.is_file()
            ]

        return manifest


class ModelExport:
    """
    Handler for neural network model exports (.pt, .onnx).
    """

    @staticmethod
    def get_default_path(model_name: str, extension: str = ".pt") -> Path:
        """Get default path for model export."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return MODELS_DIR / f"{model_name}_{timestamp}{extension}"

    @staticmethod
    def get_experiment_model_path(experiment_path: Path, model_name: str,
                                   extension: str = ".pt") -> Path:
        """Get path for model within an experiment."""
        return Path(experiment_path) / "model" / f"{model_name}{extension}"


# =============================================================================
# Utility Functions
# =============================================================================

def safe_test_dirname(name: str) -> str:
    """Map a test display name to its on-disk folder name.

    Both the batch exporter and the report viewers must agree on this
    transformation, otherwise a test saved as ``Celeb - Cake Cutting 4%``
    is written to ``Celeb_-_Cake_Cutting_4%/`` but looked up as
    ``Celeb_-_Cake_Cutting_4_/`` and the data is "missing". Keeping the
    rule here in one place prevents that drift.

    The historical convention only collapses spaces and forward slashes,
    so we preserve any other characters the user typed (``%``, ``[``,
    ``+``, …). It works on Linux/macOS and on Windows for the characters
    we have observed in real test names.
    """
    return name.replace(" ", "_").replace("/", "-")


def ensure_output_dirs():
    """Ensure all output directories exist."""
    for dir_path in [OUTPUT_DIR, SINGLE_TESTS_DIR, BATCH_TESTS_DIR, MODELS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)


def get_file_filter(extension: str) -> str:
    """Get Qt file dialog filter for an extension."""
    ext_names = {
        FileExtensions.SINGLE_TEST_CONFIG: "Single Test Config",
        FileExtensions.BATCH_CONFIG: "Batch Config",
        FileExtensions.BATCH_ANALYSIS_REPORT: "Batch Analysis Report",
        FileExtensions.SINGLE_TEST_EXPERIMENT: "Single Test Experiment",
        FileExtensions.PYTORCH_MODEL: "PyTorch Model",
        FileExtensions.ONNX_MODEL: "ONNX Model",
    }
    name = ext_names.get(extension, "File")
    return f"{name} (*{extension});;All Files (*.*)"


def list_files_by_extension(directory: Path, extension: str) -> list:
    """List all files with given extension in directory."""
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted(directory.glob(f"*{extension}"), key=lambda p: p.stat().st_mtime, reverse=True)
