"""
Data model for managing loaded batch experiments in Batch Reports mode.
"""
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any

from PyQt5.QtCore import QObject, pyqtSignal

from ui.utils.file_formats import FileExtensions


@dataclass
class LoadedExperiment:
    """Data class representing a loaded batch experiment."""
    path: Path
    name: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)
    results: List[Dict[str, Any]] = field(default_factory=list)
    export_level: str = "REPORTS_ONLY"

    @property
    def test_count(self) -> int:
        """Get the number of tests in this experiment."""
        return len(self.results)

    @property
    def completed_count(self) -> int:
        """Get the number of completed tests."""
        return self.summary.get('completed_tests', 0)

    @property
    def timestamp(self) -> str:
        """Get the experiment timestamp."""
        return self.metadata.get('timestamp', '')

    def get_test_names(self) -> List[str]:
        """Get list of test names in this experiment."""
        return [r.get('test_name', f'Test {i}') for i, r in enumerate(self.results)]


class BatchReportModel(QObject):
    """
    Model for managing multiple loaded batch experiments.

    Provides methods for:
    - Loading experiments from .batch_analysis_report files
    - Removing experiments
    - Getting combined data for comparison
    """

    # Signals
    experiments_changed = pyqtSignal()  # Emitted when experiments list changes
    experiment_loaded = pyqtSignal(int)  # Emitted with index of newly loaded experiment
    experiment_removed = pyqtSignal(int)  # Emitted with index of removed experiment

    def __init__(self, logger=None, parent=None):
        super().__init__(parent)

        if logger:
            self.logger = logger.getChild("BatchReportModel")
        else:
            self.logger = logging.getLogger("BatchReportModel")

        self._experiments: List[LoadedExperiment] = []

    @property
    def experiments(self) -> List[LoadedExperiment]:
        """Get the list of loaded experiments."""
        return self._experiments

    def load_experiment(self, path: Path) -> Optional[LoadedExperiment]:
        """
        Load a batch analysis report file.

        Args:
            path: Path to the .batch_analysis_report file

        Returns:
            LoadedExperiment if successful, None otherwise
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Extract fields from report
            metadata = data.get('metadata', {})
            summary = data.get('summary', {})
            results = data.get('results', [])
            export_level = metadata.get('export_level', 'REPORTS_ONLY')

            experiment = LoadedExperiment(
                path=path,
                name=metadata.get('batch_name', path.stem),
                metadata=metadata,
                summary=summary,
                results=results,
                export_level=export_level,
            )

            self._experiments.append(experiment)
            index = len(self._experiments) - 1

            self.logger.info(
                "Loaded experiment '%s' with %d tests from %s",
                experiment.name, len(results), path
            )

            self.experiment_loaded.emit(index)
            self.experiments_changed.emit()

            return experiment

        except json.JSONDecodeError as e:
            self.logger.error("Invalid JSON in %s: %s", path, e)
            return None
        except Exception as e:
            self.logger.error("Failed to load experiment from %s: %s", path, e)
            return None

    def remove_experiment(self, index: int) -> bool:
        """
        Remove an experiment by index.

        Args:
            index: Index of the experiment to remove

        Returns:
            True if removed, False if index invalid
        """
        if 0 <= index < len(self._experiments):
            removed = self._experiments.pop(index)
            self.logger.info("Removed experiment '%s'", removed.name)
            self.experiment_removed.emit(index)
            self.experiments_changed.emit()
            return True
        return False

    def clear_all(self):
        """Remove all loaded experiments."""
        count = len(self._experiments)
        self._experiments.clear()
        self.logger.info("Cleared %d experiments", count)
        self.experiments_changed.emit()

    def move_experiment(self, from_index: int, to_index: int) -> bool:
        """
        Move an experiment from one position to another.

        Args:
            from_index: Current index of the experiment
            to_index: Target index for the experiment

        Returns:
            True if moved successfully, False otherwise
        """
        if not (0 <= from_index < len(self._experiments)):
            return False
        if not (0 <= to_index < len(self._experiments)):
            return False
        if from_index == to_index:
            return False

        experiment = self._experiments.pop(from_index)
        self._experiments.insert(to_index, experiment)
        self.logger.info("Moved experiment '%s' from index %d to %d",
                        experiment.name, from_index, to_index)
        self.experiments_changed.emit()
        return True

    def rename_experiment(self, index: int, new_name: str) -> bool:
        """
        Rename an experiment.

        Args:
            index: Index of the experiment to rename
            new_name: New name for the experiment

        Returns:
            True if renamed successfully, False otherwise
        """
        if not (0 <= index < len(self._experiments)):
            return False

        old_name = self._experiments[index].name
        self._experiments[index].name = new_name
        self.logger.info("Renamed experiment '%s' to '%s'", old_name, new_name)
        self.experiments_changed.emit()
        return True

    def get_experiment(self, index: int) -> Optional[LoadedExperiment]:
        """Get an experiment by index."""
        if 0 <= index < len(self._experiments):
            return self._experiments[index]
        return None

    def get_all_tests(self) -> List[Dict[str, Any]]:
        """
        Get a flattened list of all tests from all experiments.

        Each test dict includes additional fields:
        - _experiment_name: Name of the parent experiment
        - _experiment_index: Index of the parent experiment
        - _experiment_path: Path to the experiment file
        - _batch_dir: Path to the batch directory containing the experiment
        - _original_name: Original test name (for file path lookup after rename)
        """
        all_tests = []
        for exp_idx, exp in enumerate(self._experiments):
            for test in exp.results:
                test_copy = test.copy()
                test_copy['_experiment_name'] = exp.name
                test_copy['_experiment_index'] = exp_idx
                test_copy['_experiment_path'] = str(exp.path)
                test_copy['_batch_dir'] = str(exp.path.parent)
                # Expose the experiment metadata so views can read, e.g.,
                # ``dataset_info.img_size`` when computing the sampling ratio.
                test_copy['_experiment_metadata'] = exp.metadata
                # Store original name for file path lookup (survives renames)
                test_copy['_original_name'] = test.get('name', '')
                all_tests.append(test_copy)
        return all_tests

    def get_comparison_data(self) -> Dict[str, Any]:
        """
        Get structured data for comparison views.

        Returns:
            Dict with keys:
            - experiments: List of experiment summaries
            - all_tests: Flattened test list
            - metrics: Aggregated metrics across all tests
        """
        all_tests = self.get_all_tests()

        # Aggregate metrics
        metrics = {
            'total_tests': len(all_tests),
            'total_experiments': len(self._experiments),
            'psnr_values': [],
            'ssim_values': [],
            'lpips_values': [],
            'inference_times': [],
        }

        for test in all_tests:
            quality = test.get('quality_metrics', {})
            timing = test.get('timing_metrics', {})

            if 'psnr' in quality:
                metrics['psnr_values'].append(quality['psnr'])
            if 'ssim' in quality:
                metrics['ssim_values'].append(quality['ssim'])
            if 'lpips' in quality:
                metrics['lpips_values'].append(quality['lpips'])
            if 'inference_time_ms' in timing:
                metrics['inference_times'].append(timing['inference_time_ms'])

        # Calculate statistics if we have data
        if metrics['psnr_values']:
            metrics['psnr_mean'] = sum(metrics['psnr_values']) / len(metrics['psnr_values'])
            metrics['psnr_max'] = max(metrics['psnr_values'])
            metrics['psnr_min'] = min(metrics['psnr_values'])

        if metrics['ssim_values']:
            metrics['ssim_mean'] = sum(metrics['ssim_values']) / len(metrics['ssim_values'])
            metrics['ssim_max'] = max(metrics['ssim_values'])
            metrics['ssim_min'] = min(metrics['ssim_values'])

        if metrics['lpips_values']:
            metrics['lpips_mean'] = sum(metrics['lpips_values']) / len(metrics['lpips_values'])
            metrics['lpips_max'] = max(metrics['lpips_values'])
            metrics['lpips_min'] = min(metrics['lpips_values'])

        if metrics['inference_times']:
            metrics['inference_time_mean'] = sum(metrics['inference_times']) / len(metrics['inference_times'])
            metrics['inference_time_max'] = max(metrics['inference_times'])
            metrics['inference_time_min'] = min(metrics['inference_times'])

        return {
            'experiments': [
                {
                    'name': exp.name,
                    'test_count': exp.test_count,
                    'timestamp': exp.timestamp,
                    'export_level': exp.export_level,
                }
                for exp in self._experiments
            ],
            'all_tests': all_tests,
            'metrics': metrics,
        }

    def is_empty(self) -> bool:
        """Check if no experiments are loaded."""
        return len(self._experiments) == 0

    def __len__(self) -> int:
        """Return the number of loaded experiments."""
        return len(self._experiments)
