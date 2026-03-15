"""
Batch Test Runner - Executes batch test configurations.
Supports both sequential and parallel execution with progress updates and cancellation.
Timing and energy measurements always run sequentially for accuracy.
"""
import os
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Dict, Any, Optional, List

from PyQt5.QtCore import QThread, pyqtSignal

from ui.custom_widgets.batch_test.test_config_model import (
    TestConfiguration, BatchTestConfig, TestStatus, ExportLevel
)
from ui.utils.file_formats import (
    FileExtensions, BatchAnalysisReport, BATCH_TESTS_DIR, MODELS_DIR
)

# Import simulation components
from simulation_engine._2_mask_gen.mask_scatter import MaskScatter
from simulation_engine._2_mask_gen.mask_hadamard import MaskHadamard
from simulation_engine._2_mask_gen.mask_hadamard_scramble import MaskHadamardScramble
from simulation_engine._2_mask_gen.mask_hadamard_cake_cutting import MaskHadamardCakeCutting
from simulation_engine._2_mask_gen.mask_hadamard_walsh_paley import MaskHadamardWalshPaley
from simulation_engine._2_mask_gen.mask_sweep import MaskSweep
from simulation_engine._2_mask_gen.mask_cal_sal import MaskCalSal
from simulation_engine._3_applicator.applicator_scatter import ApplicatorScatter
from simulation_engine._3_applicator.applicator_scatter_pseudoinverse import ApplicatorScatterPseudoinverse
from simulation_engine._3_applicator.applicator_scatter_fista import ApplicatorScatterFISTA
from simulation_engine._3_applicator.applicator_scatter_tv_norm import ApplicatorScatterTV
from simulation_engine._3_applicator.applicator_sweep import ApplicatorSweep
from simulation_engine._3_applicator.applicator_hadamard import ApplicatorHadamard
from simulation_engine._4_postprocessor.postprocessor_nn import PostprocessorNN
from simulation_engine._5_analyzer.analyzer import Analyzer
from simulation_engine._5_analyzer.analyzer_energy import EnergyAnalyzer

# PyTorch profiler imports
try:
    from torch.profiler import profile, record_function, ProfilerActivity
    PROFILER_AVAILABLE = True
except ImportError:
    PROFILER_AVAILABLE = False


class BatchTestRunner(QThread):
    """
    Worker thread for executing batch tests.

    Executes tests sequentially (parallel option exists but timing/energy
    measurements always run sequentially for accuracy).

    Signals:
        test_started(int): Emitted when a test starts (index)
        test_progress(int, int): Emitted during test (index, progress 0-100)
        test_completed(int, dict): Emitted when test completes (index, results)
        test_failed(int, str): Emitted when test fails (index, error message)
        test_cancelled(int): Emitted when test is cancelled
        batch_completed(str): Emitted when all tests complete (results CSV path)
        batch_cancelled(): Emitted when batch is cancelled
        status_update(str): General status message updates
    """

    # Test lifecycle signals
    test_started = pyqtSignal(int)
    test_progress = pyqtSignal(int, int)  # index, progress %
    test_completed = pyqtSignal(int, dict)  # index, results dict
    test_failed = pyqtSignal(int, str)  # index, error message
    test_cancelled = pyqtSignal(int)
    batch_completed = pyqtSignal(str)  # results path
    batch_cancelled = pyqtSignal()
    status_update = pyqtSignal(str)

    # Phase-specific progress signals (index, phase_name, progress 0-100)
    phase_started = pyqtSignal(int, str)        # Test index, Phase name started
    phase_progress = pyqtSignal(int, str, int)  # Test index, Phase name, progress 0-100
    phase_completed = pyqtSignal(int, str)      # Test index, Phase name completed

    # Phase names constants
    PHASE_MASKS = "Masks"
    PHASE_RECONSTRUCTION = "Reconstruction"
    PHASE_MODEL_SETUP = "Model Setup"
    PHASE_TRAINING = "Training"
    PHASE_ANALYSIS = "Analysis"
    PHASE_EXPORT = "Export"

    def __init__(
        self,
        batch_config: BatchTestConfig,
        dataset,
        export_level: ExportLevel = ExportLevel.REPORTS_ONLY,
        batch_name: str = "",
        logger=None,
        parent=None
    ):
        super().__init__(parent)

        if logger:
            self.logger = logger.getChild("BatchTestRunner")
        else:
            self.logger = logging.getLogger("BatchTestRunner")

        self.batch_config = batch_config
        self.dataset = dataset
        self.export_level = export_level
        self.batch_name = batch_name
        self._cancel_requested = False
        self._cancel_test_indices = set()  # Tests to cancel
        self._all_results: List[Dict[str, Any]] = []
        self._trained_models: Dict[int, Any] = {}  # Store models for export
        self._test_data: Dict[int, Dict] = {}  # Store test data for export
        self._results_lock = Lock()  # Thread safety for results

    def run(self):
        """Execute all tests in the batch (sequential or parallel based on config)."""
        self.logger.info("Starting batch test run with %d tests", len(self.batch_config.tests))
        self._cancel_requested = False
        self._all_results = []

        try:
            if self.batch_config.parallel_execution:
                self._run_parallel()
            else:
                self._run_sequential()

            # Export results based on export level
            results_dir = self._export_all_results()
            self.logger.info("Batch completed, results saved to: %s", results_dir)
            self.batch_completed.emit(results_dir)

        except Exception as e:
            self.logger.error("Batch run failed with exception: %s", e, exc_info=True)
            self.status_update.emit(f"Batch failed: {e}")
            self.batch_cancelled.emit()

    def _run_sequential(self):
        """Run all tests sequentially (original behavior)."""
        for i, test_config in enumerate(self.batch_config.tests):
            # Check for batch cancellation
            if self._cancel_requested:
                self.logger.info("Batch cancelled before test %d", i)
                self._mark_remaining_cancelled(i)
                self.batch_cancelled.emit()
                return

            # Check for individual test cancellation
            if i in self._cancel_test_indices:
                self.logger.info("Test %d was cancelled before execution", i)
                test_config.status = TestStatus.CANCELLED
                self.test_cancelled.emit(i)
                continue

            # Run the test
            self._run_single_test(i, test_config)

    def _run_parallel(self):
        """
        Run tests in parallel with hybrid approach:
        1. Training phase (masks, reconstruction, model setup, training) runs in parallel
        2. Analysis phase (timing/energy) runs sequentially for accuracy

        This allows heavy computation to be parallelized while maintaining
        accurate timing/energy measurements.
        """
        num_threads = self.batch_config.parallel_threads
        self.logger.info("Running in parallel mode with %d threads (hybrid approach)", num_threads)

        # Collect all tests to run
        tests_to_run = []
        for i, test_config in enumerate(self.batch_config.tests):
            if i in self._cancel_test_indices:
                self.logger.info("Test %d was cancelled before execution", i)
                test_config.status = TestStatus.CANCELLED
                self.test_cancelled.emit(i)
                continue
            tests_to_run.append((i, test_config))

        if not tests_to_run:
            return

        # Phase 1: Run training in parallel for all tests
        self.status_update.emit(f"Phase 1: Training {len(tests_to_run)} tests in parallel...")
        self.logger.info("Starting parallel training phase for %d tests", len(tests_to_run))
        self._execute_parallel_training(tests_to_run, num_threads)

        # Check for cancellation before analysis phase
        if self._cancel_requested:
            self.logger.info("Batch cancelled before analysis phase")
            self.batch_cancelled.emit()
            return

        # Phase 2: Run analysis sequentially for tests that need timing/energy
        tests_needing_analysis = [
            (i, config) for i, config in tests_to_run
            if config.status == TestStatus.RUNNING and
               any(r in config.reports for r in ["timing", "energy"])
        ]

        if tests_needing_analysis:
            self.status_update.emit(f"Phase 2: Analyzing {len(tests_needing_analysis)} tests sequentially...")
            self.logger.info("Starting sequential analysis phase for %d tests", len(tests_needing_analysis))

            for i, test_config in tests_needing_analysis:
                if self._cancel_requested:
                    self._mark_remaining_cancelled(i)
                    self.batch_cancelled.emit()
                    return

                self._run_analysis_phase(i, test_config)

    def _execute_parallel_training(self, tests: List[tuple], num_threads: int):
        """Execute training phase in parallel for all tests."""
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit all training tasks
            future_to_test = {
                executor.submit(self._run_training_phase_safe, i, config): (i, config)
                for i, config in tests
            }

            # Wait for completion
            for future in as_completed(future_to_test):
                if self._cancel_requested:
                    for f in future_to_test:
                        f.cancel()
                    break

                i, config = future_to_test[future]
                try:
                    future.result()
                except Exception as e:
                    self.logger.error("Training phase for test %d failed: %s", i, e)

    def _run_training_phase_safe(self, index: int, config: TestConfiguration):
        """Thread-safe wrapper for _run_training_phase."""
        try:
            self._run_training_phase(index, config)
        except Exception as e:
            self.logger.error("Test %d training failed: %s", index, e, exc_info=True)
            config.status = TestStatus.FAILED
            config.error_message = str(e)
            with self._results_lock:
                self._all_results.append({
                    "name": config.name,
                    "status": "failed",
                    "error": str(e),
                    "end_time": datetime.now().isoformat(),
                })
            self.test_failed.emit(index, str(e))

    def _run_training_phase(self, index: int, config: TestConfiguration):
        """
        Run the training phase of a test (masks, reconstruction, model setup, training).
        This phase can run in parallel with other tests.
        """
        self.logger.info("Starting training phase for test %d: %s", index, config.name)
        config.status = TestStatus.RUNNING
        config.progress = 0
        self.test_started.emit(index)
        self.status_update.emit(f"Training: {config.name}")

        # Initialize results dict with all training parameters (always included)
        results = {
            "name": config.name,
            "test_name": config.name,
            "mask_type": config.mask_type,
            "reconstruction_method": config.reconstruction_method,
            "model_name": config.model_name,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "dropout": config.dropout,
            "loss_function": config.loss_function,
            "optimizer": config.optimizer,
            "use_gpu": config.use_gpu,
            "train_split": config.train_split,
            "val_split": config.val_split,
            "test_split": config.test_split,
            "start_time": datetime.now().isoformat(),
        }

        # Phase 1: Create mask
        self.phase_started.emit(index, self.PHASE_MASKS)
        self.status_update.emit(f"Generating masks: {config.name}")
        self.phase_progress.emit(index, self.PHASE_MASKS, 0)
        mask = self._create_mask(config)
        self.phase_progress.emit(index, self.PHASE_MASKS, 100)
        self.phase_completed.emit(index, self.PHASE_MASKS)
        self.test_progress.emit(index, 10)

        if self._should_cancel(index):
            return self._handle_test_cancel(index, config)

        # Phase 2: Create applicator and reconstruct
        self.phase_started.emit(index, self.PHASE_RECONSTRUCTION)
        self.status_update.emit(f"Creating applicator: {config.name}")
        self.phase_progress.emit(index, self.PHASE_RECONSTRUCTION, 0)
        applicator = self._create_applicator(config, mask)

        # Update reconstruction method from applicator (more accurate than config)
        if hasattr(applicator, 'RECONSTRUCTION_METHOD'):
            results["reconstruction_method"] = applicator.RECONSTRUCTION_METHOD

        self.phase_progress.emit(index, self.PHASE_RECONSTRUCTION, 10)

        self.status_update.emit(f"Reconstructing images: {config.name}")
        self.phase_progress.emit(index, self.PHASE_RECONSTRUCTION, 20)
        reconstructed_df = applicator.process_dataset()
        reconstructed_data = reconstructed_df.to_numpy()
        self.phase_progress.emit(index, self.PHASE_RECONSTRUCTION, 100)
        self.phase_completed.emit(index, self.PHASE_RECONSTRUCTION)
        self.test_progress.emit(index, 30)

        if self._should_cancel(index):
            return self._handle_test_cancel(index, config)

        # Phase 3: Model Setup
        self.phase_started.emit(index, self.PHASE_MODEL_SETUP)
        self.status_update.emit(f"Setting up model: {config.name}")
        self.phase_progress.emit(index, self.PHASE_MODEL_SETUP, 0)

        # Phase 4: Train DNN
        postprocessor = self._create_and_train_postprocessor(
            config, mask, applicator, index, reconstructed_data
        )
        self.test_progress.emit(index, 75)

        # Add training curves to results if collected
        if "training_curves" in config.reports and hasattr(postprocessor, '_training_curves'):
            results["training_curves"] = postprocessor._training_curves

        if self._should_cancel(index):
            return self._handle_test_cancel(index, config)

        # Run basic analysis (PSNR, SSIM, LPIPS) - these are fast and can run in parallel
        self.phase_started.emit(index, self.PHASE_ANALYSIS)
        self.status_update.emit(f"Basic analysis: {config.name}")
        self.phase_progress.emit(index, self.PHASE_ANALYSIS, 0)

        # Get test outputs
        orig, recons, denoised = postprocessor.test_dataset()
        analyzer = Analyzer(orig, recons, denoised)
        analyzer.analyze_noise()

        # Collect quality metrics (PSNR, SSIM, LPIPS)
        if "quality" in config.reports:
            results["quality_metrics"] = {}

            # PSNR (mean values)
            results["psnr_recons"] = analyzer.noise.psnr_noise_mean
            results["psnr_denoised"] = analyzer.noise.psnr_rec_mean
            results["quality_metrics"]["psnr"] = analyzer.noise.psnr_rec_mean

            # SSIM (mean values)
            results["ssim_recons"] = analyzer.noise.ssim_noise_mean
            results["ssim_denoised"] = analyzer.noise.ssim_rec_mean
            results["quality_metrics"]["ssim"] = analyzer.noise.ssim_rec_mean

            # LPIPS (mean values)
            try:
                results["lpips_recons"] = analyzer.noise.lpips_noise_mean
                results["lpips_denoised"] = analyzer.noise.lpips_rec_mean
                results["quality_metrics"]["lpips"] = analyzer.noise.lpips_rec_mean
            except Exception as e:
                self.logger.warning("LPIPS computation failed: %s", e)
                results["lpips_error"] = str(e)

            # Per-image metrics for detailed charts
            results["quality_per_image"] = {
                "psnr_noisy": list(analyzer.noise.psnr_noise),
                "psnr_denoised": list(analyzer.noise.psnr_rec),
                "ssim_noisy": list(analyzer.noise.ssim_noise),
                "ssim_denoised": list(analyzer.noise.ssim_rec),
                "lpips_noisy": list(analyzer.noise.lpips_noise),
                "lpips_denoised": list(analyzer.noise.lpips_rec),
            }

        # Check if we need timing/energy analysis later
        needs_timing_energy = any(r in config.reports for r in ["timing", "energy"])

        if needs_timing_energy:
            # Store postprocessor and applicator for later analysis phase
            with self._results_lock:
                self._trained_models[index] = {
                    "postprocessor": postprocessor,
                    "applicator": applicator,
                    "config": config,
                    "results": results,
                }
            self.phase_progress.emit(index, self.PHASE_ANALYSIS, 50)
            # Don't mark as completed yet - analysis phase will continue
            self.logger.info("Training phase complete for test %d, pending timing/energy analysis", index)
        else:
            # No timing/energy needed - complete the test now
            self.phase_progress.emit(index, self.PHASE_ANALYSIS, 100)
            self.phase_completed.emit(index, self.PHASE_ANALYSIS)
            self.test_progress.emit(index, 100)

            results["end_time"] = datetime.now().isoformat()
            results["status"] = "completed"
            config.status = TestStatus.COMPLETED
            config.progress = 100
            config.results = results

            # Store for export
            with self._results_lock:
                if self.export_level in (ExportLevel.REPORTS_AND_MODELS, ExportLevel.ALL_DATA):
                    self._trained_models[index] = {
                        "postprocessor": postprocessor,
                        "applicator": applicator,
                        "config": config,
                    }
                if self.export_level == ExportLevel.ALL_DATA:
                    self._test_data[index] = {
                        "mask": mask,
                        "applicator": applicator,
                        "originals": orig,
                        "reconstructions": recons,
                        "denoised": denoised,
                        "config": config,
                    }
                self._all_results.append(results)

            self.test_completed.emit(index, results)
            self.logger.info("Test %d completed (no timing/energy): %s", index, config.name)

    def _run_analysis_phase(self, index: int, config: TestConfiguration):
        """
        Run the analysis phase for timing/energy measurements.
        This phase runs sequentially for accurate measurements.
        """
        self.logger.info("Starting analysis phase for test %d: %s", index, config.name)
        self.status_update.emit(f"Timing/Energy analysis: {config.name}")

        # Get stored data from training phase
        with self._results_lock:
            model_data = self._trained_models.get(index)
            if not model_data:
                self.logger.error("No trained model found for test %d", index)
                return

        postprocessor = model_data["postprocessor"]
        applicator = model_data.get("applicator")
        results = model_data.get("results", {})

        # Continue analysis phase
        self.phase_progress.emit(index, self.PHASE_ANALYSIS, 60)

        # Run timing analysis if requested
        if "timing" in config.reports:
            try:
                self.status_update.emit(f"Measuring timing: {config.name}")
                timing_results = self._measure_timing(
                    postprocessor, config,
                    applicator=applicator,
                    warmup_runs=config.timing_warmup_runs,
                    measurement_runs=config.timing_measurement_runs,
                    sampling_rate_khz=config.timing_sampling_rate_khz
                )
                results.update(timing_results)
                results["timing_metrics"] = results.get("timing_metrics", {})

                # Run PyTorch profiler and store results
                self.status_update.emit(f"Running profiler: {config.name}")
                profiler_results = self._profile_model(
                    postprocessor, config,
                    num_images=10,
                    warmup_runs=3
                )
                if profiler_results:
                    results["profiler_results"] = profiler_results
                results["timing_metrics"]["inference_time_ms"] = timing_results.get("timing_mean_ms", 0)
            except Exception as e:
                self.logger.warning("Timing analysis failed: %s", e)
                results["timing_error"] = str(e)

        self.phase_progress.emit(index, self.PHASE_ANALYSIS, 80)

        # Run energy analysis if requested
        if "energy" in config.reports:
            try:
                self.status_update.emit(f"Measuring energy: {config.name}")
                energy_results = self._measure_energy(
                    postprocessor, config,
                    warmup_runs=config.timing_warmup_runs,
                    measurement_runs=config.timing_measurement_runs
                )
                results.update(energy_results)
            except Exception as e:
                self.logger.warning("Energy analysis failed: %s", e)
                results["energy_error"] = str(e)

        self.phase_progress.emit(index, self.PHASE_ANALYSIS, 100)
        self.phase_completed.emit(index, self.PHASE_ANALYSIS)
        self.test_progress.emit(index, 100)

        # Complete the test
        results["end_time"] = datetime.now().isoformat()
        results["status"] = "completed"
        config.status = TestStatus.COMPLETED
        config.progress = 100
        config.results = results

        # Update stored data for export
        with self._results_lock:
            # Update model data
            self._trained_models[index]["results"] = results

            if self.export_level == ExportLevel.ALL_DATA:
                orig, recons, denoised = postprocessor.test_dataset()
                self._test_data[index] = {
                    "mask": getattr(postprocessor, 'mask', None),
                    "applicator": getattr(postprocessor, 'applicator', None),
                    "originals": orig,
                    "reconstructions": recons,
                    "denoised": denoised,
                    "config": config,
                }

            self._all_results.append(results)

        self.test_completed.emit(index, results)
        self.logger.info("Test %d analysis complete: %s", index, config.name)

    def _execute_parallel_tests(self, tests: List[tuple], num_threads: int):
        """Execute tests in parallel using ThreadPoolExecutor (legacy method)."""
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit all tests
            future_to_test = {
                executor.submit(self._run_single_test_safe, i, config): (i, config)
                for i, config in tests
            }

            # Wait for completion and handle results
            for future in as_completed(future_to_test):
                if self._cancel_requested:
                    # Cancel remaining futures
                    for f in future_to_test:
                        f.cancel()
                    break

                i, config = future_to_test[future]
                try:
                    future.result()  # Raises exception if test failed
                except Exception as e:
                    self.logger.error("Parallel test %d failed: %s", i, e)

    def _run_single_test_safe(self, index: int, config: TestConfiguration):
        """
        Thread-safe wrapper for _run_single_test.
        Used for parallel execution.
        """
        try:
            self._run_single_test(index, config)
        except Exception as e:
            self.logger.error("Test %d failed in parallel execution: %s", index, e, exc_info=True)
            config.status = TestStatus.FAILED
            config.error_message = str(e)
            with self._results_lock:
                self._all_results.append({
                    "name": config.name,
                    "status": "failed",
                    "error": str(e),
                    "end_time": datetime.now().isoformat(),
                })
            self.test_failed.emit(index, str(e))

    def _run_single_test(self, index: int, config: TestConfiguration):
        """Run a single test configuration."""
        self.logger.info("Starting test %d: %s", index, config.name)
        config.status = TestStatus.RUNNING
        config.progress = 0
        self.test_started.emit(index)
        self.status_update.emit(f"Running: {config.name}")

        # Initialize results dict with all training parameters (always included)
        results = {
            "name": config.name,
            "test_name": config.name,
            "mask_type": config.mask_type,
            "reconstruction_method": config.reconstruction_method,
            "model_name": config.model_name,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "dropout": config.dropout,
            "loss_function": config.loss_function,
            "optimizer": config.optimizer,
            "use_gpu": config.use_gpu,
            "train_split": config.train_split,
            "val_split": config.val_split,
            "test_split": config.test_split,
            "start_time": datetime.now().isoformat(),
        }

        try:
            # Phase 1: Create mask
            self.phase_started.emit(index, self.PHASE_MASKS)
            self.status_update.emit(f"Generating masks: {config.name}")
            self.phase_progress.emit(index, self.PHASE_MASKS, 0)
            mask = self._create_mask(config)
            self.phase_progress.emit(index, self.PHASE_MASKS, 100)
            self.phase_completed.emit(index, self.PHASE_MASKS)
            self.test_progress.emit(index, 10)

            if self._should_cancel(index):
                return self._handle_test_cancel(index, config)

            # Phase 2: Create applicator
            self.phase_started.emit(index, self.PHASE_RECONSTRUCTION)
            self.status_update.emit(f"Creating applicator: {config.name}")
            self.phase_progress.emit(index, self.PHASE_RECONSTRUCTION, 0)
            applicator = self._create_applicator(config, mask)

            # Update reconstruction method from applicator (more accurate than config)
            if hasattr(applicator, 'RECONSTRUCTION_METHOD'):
                results["reconstruction_method"] = applicator.RECONSTRUCTION_METHOD

            self.phase_progress.emit(index, self.PHASE_RECONSTRUCTION, 10)

            # Phase 2b: Run actual reconstruction (applying masks to all images)
            self.status_update.emit(f"Reconstructing images: {config.name}")
            self.phase_progress.emit(index, self.PHASE_RECONSTRUCTION, 20)
            reconstructed_df = applicator.process_dataset()
            reconstructed_data = reconstructed_df.to_numpy()
            self.phase_progress.emit(index, self.PHASE_RECONSTRUCTION, 100)
            self.phase_completed.emit(index, self.PHASE_RECONSTRUCTION)
            self.test_progress.emit(index, 30)

            if self._should_cancel(index):
                return self._handle_test_cancel(index, config)

            # Phase 3: Model Setup (creating model architecture, moving to GPU)
            self.phase_started.emit(index, self.PHASE_MODEL_SETUP)
            self.status_update.emit(f"Setting up model: {config.name}")
            self.phase_progress.emit(index, self.PHASE_MODEL_SETUP, 0)

            # Phase 4: Train DNN (pass pre-reconstructed data)
            postprocessor = self._create_and_train_postprocessor(
                config, mask, applicator, index, reconstructed_data
            )
            self.test_progress.emit(index, 75)

            # Add training curves to results if collected
            if "training_curves" in config.reports and hasattr(postprocessor, '_training_curves'):
                results["training_curves"] = postprocessor._training_curves

            if self._should_cancel(index):
                return self._handle_test_cancel(index, config)

            # Phase 5: Analyze results
            self.phase_started.emit(index, self.PHASE_ANALYSIS)
            self.status_update.emit(f"Analyzing: {config.name}")
            self.phase_progress.emit(index, self.PHASE_ANALYSIS, 0)
            analysis_results = self._analyze_results(config, postprocessor, applicator=applicator)
            results.update(analysis_results)
            self.phase_progress.emit(index, self.PHASE_ANALYSIS, 100)
            self.phase_completed.emit(index, self.PHASE_ANALYSIS)
            self.test_progress.emit(index, 100)

            # Mark completed
            results["end_time"] = datetime.now().isoformat()
            results["status"] = "completed"
            config.status = TestStatus.COMPLETED
            config.progress = 100
            config.results = results

            # Store data for export based on export level (thread-safe)
            with self._results_lock:
                if self.export_level in (ExportLevel.REPORTS_AND_MODELS, ExportLevel.ALL_DATA):
                    self._trained_models[index] = {
                        "postprocessor": postprocessor,
                        "applicator": applicator,
                        "config": config,
                    }

                if self.export_level == ExportLevel.ALL_DATA:
                    # Store test data for export (mask, applicator, results)
                    orig, recons, denoised = postprocessor.test_dataset()
                    self._test_data[index] = {
                        "mask": mask,
                        "applicator": applicator,
                        "originals": orig,
                        "reconstructions": recons,
                        "denoised": denoised,
                        "config": config,
                    }

                self._all_results.append(results)

            self.test_completed.emit(index, results)
            self.logger.info("Test %d completed: %s", index, config.name)

        except Exception as e:
            self.logger.error("Test %d failed: %s", index, e, exc_info=True)
            results["end_time"] = datetime.now().isoformat()
            results["status"] = "failed"
            results["error"] = str(e)

            config.status = TestStatus.FAILED
            config.error_message = str(e)

            with self._results_lock:
                self._all_results.append(results)
            self.test_failed.emit(index, str(e))

    def _create_mask(self, config: TestConfiguration):
        """Create mask based on configuration."""
        img_size = self.dataset.img_size

        if config.mask_type == "scatter":
            return MaskScatter(
                img_size=img_size,
                point_density=config.scatter_point_density,
                num_patterns=config.scatter_num_patterns,
                seed=config.mask_seed,
                logger=self.logger
            )
        elif config.mask_type == "hadamard_natural":
            max_idx = min(config.hadamard_max_idx, img_size * img_size)
            return MaskHadamard(
                img_size=img_size,
                min_idx=config.hadamard_min_idx,
                max_idx=max_idx,
                logger=self.logger
            )
        elif config.mask_type == "hadamard_scramble":
            max_idx = min(config.hadamard_max_idx, img_size * img_size)
            return MaskHadamardScramble(
                img_size=img_size,
                min_idx=config.hadamard_min_idx,
                max_idx=max_idx,
                logger=self.logger
            )
        elif config.mask_type == "hadamard_cake_cutting":
            max_idx = min(config.hadamard_max_idx, img_size * img_size)
            return MaskHadamardCakeCutting(
                img_size=img_size,
                min_idx=config.hadamard_min_idx,
                max_idx=max_idx,
                logger=self.logger
            )
        elif config.mask_type == "hadamard_walsh_paley":
            max_idx = min(config.hadamard_max_idx, img_size * img_size)
            return MaskHadamardWalshPaley(
                img_size=img_size,
                min_idx=config.hadamard_min_idx,
                max_idx=max_idx,
                logger=self.logger
            )
        elif config.mask_type == "sweep":
            # Convert sweep angles to parameters list format
            parametros = []
            for i, angle in enumerate(config.sweep_angles):
                parametros.append({
                    "angle": angle,
                    "bar_width": config.sweep_bar_widths[i] if i < len(config.sweep_bar_widths) else 2,
                    "stride": config.sweep_strides[i] if i < len(config.sweep_strides) else 4,
                })
            return MaskSweep(
                img_size=img_size,
                parametros=parametros,
                logger=self.logger
            )
        elif config.mask_type == "cal_sal":
            return MaskCalSal(
                img_size=img_size,
                logger=self.logger
            )
        else:
            raise ValueError(f"Unknown mask type: {config.mask_type}")

    def _create_applicator(self, config: TestConfiguration, mask):
        """Create applicator based on mask type and reconstruction method."""
        # Generate mask patterns
        mask.generate_masks()

        if isinstance(mask, MaskScatter):
            method = config.reconstruction_method
            if method == "conventional":
                # Direct scatter reconstruction (simple sampling)
                return ApplicatorScatter(self.dataset, mask)
            elif method == "pseudoinverse":
                return ApplicatorScatterPseudoinverse(self.dataset, mask)
            elif method == "fista":
                applicator = ApplicatorScatterFISTA(self.dataset, mask)
                applicator.lambda_val = config.fista_lambda
                applicator.max_iter = config.fista_iterations
                return applicator
            elif method == "tv_norm":
                applicator = ApplicatorScatterTV(self.dataset, mask)
                applicator.lambda_val = config.tv_lambda
                applicator.max_iter = config.tv_iterations
                return applicator
            else:
                # Fallback to conventional
                return ApplicatorScatter(self.dataset, mask)

        elif isinstance(mask, MaskSweep):
            return ApplicatorSweep(self.dataset, mask)

        elif isinstance(mask, (MaskHadamard, MaskHadamardScramble, MaskHadamardCakeCutting,
                               MaskHadamardWalshPaley, MaskCalSal)):
            return ApplicatorHadamard(self.dataset, mask)

        else:
            raise ValueError(f"Unsupported mask type for applicator: {type(mask)}")

    def _create_and_train_postprocessor(
        self, config: TestConfiguration, mask, applicator, test_index: int,
        reconstructed_data=None
    ) -> PostprocessorNN:
        """Create and train the postprocessor (DNN).

        Args:
            config: Test configuration
            mask: Generated mask
            applicator: Applicator object (for reference, data already reconstructed)
            test_index: Index of current test
            reconstructed_data: Pre-reconstructed data from applicator.process_dataset()

        Returns:
            Trained PostprocessorNN instance
        """
        # Storage for training curves if requested
        training_curves = {}

        def training_progress(epoch, total_epochs):
            """Callback for training progress."""
            # Calculate progress percentage for training phase (0-100%)
            phase_pct = int((epoch / total_epochs) * 100)
            self.phase_progress.emit(test_index, self.PHASE_TRAINING, phase_pct)

            # Overall test progress: training is 30% to 75% of total
            overall_progress = 30 + int((epoch / total_epochs) * 45)
            self.test_progress.emit(test_index, overall_progress)
            config.progress = overall_progress

            # Check for cancellation during training
            if self._should_cancel(test_index):
                raise InterruptedError("Training cancelled")

        def metrics_callback(val_losses, test_losses, val_psnr, val_ssim, val_lpips):
            """Callback to receive training metrics curves."""
            training_curves["val_losses"] = val_losses
            training_curves["test_losses"] = test_losses
            training_curves["val_psnr"] = val_psnr
            training_curves["val_ssim"] = val_ssim
            training_curves["val_lpips"] = val_lpips

        # Prepare model overrides from config
        model_overrides = {}
        if config.dropout > 0:
            model_overrides['dropout'] = config.dropout

        # Model setup phase - creating model architecture and moving to GPU
        self.phase_progress.emit(test_index, self.PHASE_MODEL_SETUP, 50)

        # Convert split percentages to ratios (0-1)
        train_ratio = config.train_split / 100.0
        val_ratio = config.val_split / 100.0
        test_ratio = config.test_split / 100.0

        postprocessor = PostprocessorNN(
            model_name=config.model_name,
            model_overrides=model_overrides,
            dataset=self.dataset,
            applicator=applicator,
            batch_size=config.batch_size,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            loss_function=config.loss_function,
            optimizer_name=config.optimizer,
            use_gpu=config.use_gpu,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            logger=self.logger,
            reconstructed_data=reconstructed_data  # Pass pre-reconstructed data
        )

        # Model setup complete
        self.phase_progress.emit(test_index, self.PHASE_MODEL_SETUP, 100)
        self.phase_completed.emit(test_index, self.PHASE_MODEL_SETUP)

        # Start training phase
        self.phase_started.emit(test_index, self.PHASE_TRAINING)
        self.status_update.emit(f"Training: {config.name}")

        # Train the model - use train_with_metrics if training curves are requested
        if "training_curves" in config.reports:
            postprocessor.train_with_metrics(
                num_epochs=config.epochs,
                progress_callback=training_progress,
                metrics_callback=metrics_callback
            )
            # Store training curves for later access
            postprocessor._training_curves = training_curves
        else:
            postprocessor.train(
                num_epochs=config.epochs,
                progress_callback=training_progress
            )

        # Training complete
        self.phase_completed.emit(test_index, self.PHASE_TRAINING)

        return postprocessor

    def _analyze_results(self, config: TestConfiguration, postprocessor: PostprocessorNN,
                         applicator=None) -> Dict[str, Any]:
        """Analyze test results based on configured reports."""
        results = {}

        # Get test outputs
        orig, recons, denoised = postprocessor.test_dataset()

        # Create analyzer for metrics (orig, noisy=recons, reconstructions=denoised)
        analyzer = Analyzer(orig, recons, denoised)

        # Run noise analysis to compute all metrics
        analyzer.analyze_noise()

        # Collect quality metrics (PSNR, SSIM, LPIPS)
        if "quality" in config.reports:
            results["quality_metrics"] = {}

            # PSNR: psnr_noise_mean = recons vs orig, psnr_rec_mean = denoised vs orig
            results["psnr_recons"] = analyzer.noise.psnr_noise_mean
            results["psnr_denoised"] = analyzer.noise.psnr_rec_mean
            results["quality_metrics"]["psnr"] = analyzer.noise.psnr_rec_mean

            # SSIM
            results["ssim_recons"] = analyzer.noise.ssim_noise_mean
            results["ssim_denoised"] = analyzer.noise.ssim_rec_mean
            results["quality_metrics"]["ssim"] = analyzer.noise.ssim_rec_mean

            # LPIPS
            try:
                results["lpips_recons"] = analyzer.noise.lpips_noise_mean
                results["lpips_denoised"] = analyzer.noise.lpips_rec_mean
                results["quality_metrics"]["lpips"] = analyzer.noise.lpips_rec_mean
            except Exception as e:
                self.logger.warning("LPIPS computation failed: %s", e)
                results["lpips_error"] = str(e)

            # Per-image metrics for detailed charts
            results["quality_per_image"] = {
                "psnr_noisy": list(analyzer.noise.psnr_noise),
                "psnr_denoised": list(analyzer.noise.psnr_rec),
                "ssim_noisy": list(analyzer.noise.ssim_noise),
                "ssim_denoised": list(analyzer.noise.ssim_rec),
                "lpips_noisy": list(analyzer.noise.lpips_noise),
                "lpips_denoised": list(analyzer.noise.lpips_rec),
            }

        if "timing" in config.reports:
            # Note: Timing analysis runs sequentially for accuracy
            try:
                timing_results = self._measure_timing(
                    postprocessor, config,
                    applicator=applicator,
                    warmup_runs=config.timing_warmup_runs,
                    measurement_runs=config.timing_measurement_runs,
                    sampling_rate_khz=config.timing_sampling_rate_khz
                )
                results.update(timing_results)

                # Run PyTorch profiler and store results
                profiler_results = self._profile_model(
                    postprocessor, config,
                    num_images=10,
                    warmup_runs=3
                )
                if profiler_results:
                    results["profiler_results"] = profiler_results
            except Exception as e:
                self.logger.warning("Timing analysis failed: %s", e)
                results["timing_error"] = str(e)

        if "energy" in config.reports:
            # Note: Energy analysis runs sequentially for accuracy
            try:
                self.status_update.emit(f"Measuring energy: {config.name}")
                energy_results = self._measure_energy(
                    postprocessor, config,
                    warmup_runs=config.timing_warmup_runs,
                    measurement_runs=config.timing_measurement_runs
                )
                results.update(energy_results)
            except Exception as e:
                self.logger.warning("Energy analysis failed: %s", e)
                results["energy_error"] = str(e)

        return results

    def _measure_timing(
        self, postprocessor: PostprocessorNN, config: TestConfiguration,
        applicator=None, warmup_runs: int = 5, measurement_runs: int = 20,
        sampling_rate_khz: float = 10.752
    ) -> Dict[str, Any]:
        """
        Measure inference timing with configurable parameters.

        Always measures CPU timing. If use_gpu is enabled and CUDA is available,
        also measures GPU timing (like Single Test behavior).
        Also measures reconstruction time if applicator is provided.
        """
        import torch
        import numpy as np

        model = postprocessor.model
        img_size = postprocessor.img_size
        is_conv = postprocessor.is_conv

        model.eval()

        # Calculate acquisition time based on sampling rate and number of patterns
        num_patterns = 1  # Default
        mask = None

        # Try to get mask from applicator
        if applicator is not None and hasattr(applicator, 'mask'):
            mask = applicator.mask
        elif hasattr(postprocessor, 'applicator') and postprocessor.applicator is not None:
            if hasattr(postprocessor.applicator, 'mask'):
                mask = postprocessor.applicator.mask

        if mask is not None:
            if hasattr(mask, 'num_patterns'):
                num_patterns = mask.num_patterns
            elif hasattr(mask, 'patterns') and mask.patterns is not None:
                num_patterns = len(mask.patterns)

        t_acquisition_ms = num_patterns / sampling_rate_khz if sampling_rate_khz > 0 else 0

        # Measure reconstruction time if applicator is available
        # Match Single Test behavior: measure all test images, with warmup
        t_reconstruction_ms = 0.0
        if applicator is not None:
            self.logger.debug("Measuring reconstruction timing...")
            try:
                dataset_size = len(getattr(self.dataset, 'data', []))
                # Use same number of samples as test set (from config split)
                test_ratio = config.test_split / 100.0
                n_recon_samples = max(1, min(dataset_size, int(dataset_size * test_ratio)))

                # Warmup runs to ensure fair comparison (like inference warmup)
                warmup_recon = min(2, n_recon_samples)
                for idx in range(warmup_recon):
                    _ = applicator.process_image(idx)

                # Measure reconstruction time for test images
                recon_times = []
                for idx in range(n_recon_samples):
                    t0 = time.perf_counter()
                    _ = applicator.process_image(idx)
                    t1 = time.perf_counter()
                    recon_times.append((t1 - t0) * 1000.0)  # ms

                if recon_times:
                    t_reconstruction_ms = float(np.mean(recon_times))
                    self.logger.debug("Reconstruction time: %.2f ms (mean of %d samples, after %d warmup)",
                                     t_reconstruction_ms, len(recon_times), warmup_recon)
            except Exception as e:
                self.logger.warning("Reconstruction timing failed: %s", e)

        # Helper function to measure timing on a specific device
        def measure_on_device(device: torch.device) -> List[float]:
            model.to(device)
            if is_conv:
                sample = torch.randn(1, 1, img_size, img_size, device=device)
            else:
                sample = torch.randn(1, img_size * img_size, device=device)

            # Warmup
            with torch.no_grad():
                for _ in range(warmup_runs):
                    _ = model(sample)
                    if device.type == "cuda":
                        torch.cuda.synchronize()

            # Measure
            times = []
            with torch.no_grad():
                for _ in range(measurement_runs):
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    start = time.perf_counter()
                    _ = model(sample)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    end = time.perf_counter()
                    times.append((end - start) * 1000)  # Convert to ms

            return times

        # Always measure CPU timing
        self.logger.debug("Measuring CPU inference timing...")
        cpu_device = torch.device('cpu')
        cpu_times = measure_on_device(cpu_device)
        cpu_times_np = np.array(cpu_times)

        results = {
            "timing_cpu_mean_ms": float(cpu_times_np.mean()),
            "timing_cpu_std_ms": float(cpu_times_np.std()),
            "timing_cpu_min_ms": float(cpu_times_np.min()),
            "timing_cpu_max_ms": float(cpu_times_np.max()),
            "timing_warmup_runs": warmup_runs,
            "timing_measurement_runs": measurement_runs,
            "timing_sampling_rate_khz": sampling_rate_khz,
            "timing_acquisition_ms": float(t_acquisition_ms),
            "timing_reconstruction_ms": float(t_reconstruction_ms),
            "timing_num_patterns": num_patterns,
            "use_gpu": config.use_gpu,
        }

        # Also measure GPU timing if requested and available
        if config.use_gpu and torch.cuda.is_available():
            self.logger.debug("Measuring GPU inference timing...")
            try:
                gpu_device = torch.device('cuda')
                gpu_times = measure_on_device(gpu_device)
                gpu_times_np = np.array(gpu_times)

                results["timing_gpu_mean_ms"] = float(gpu_times_np.mean())
                results["timing_gpu_std_ms"] = float(gpu_times_np.std())
                results["timing_gpu_min_ms"] = float(gpu_times_np.min())
                results["timing_gpu_max_ms"] = float(gpu_times_np.max())

                # Move model back to original device
                model.to(postprocessor.device)
            except Exception as e:
                self.logger.warning("GPU timing measurement failed: %s", e)
        else:
            # Ensure model is on CPU if GPU was not used
            model.to(cpu_device)

        # For backwards compatibility, also set timing_mean_ms to the primary device
        if config.use_gpu and "timing_gpu_mean_ms" in results:
            results["timing_mean_ms"] = results["timing_gpu_mean_ms"]
        else:
            results["timing_mean_ms"] = results["timing_cpu_mean_ms"]

        return results

    def _profile_model(
        self, postprocessor: PostprocessorNN, config: TestConfiguration,
        num_images: int = 10, warmup_runs: int = 3
    ) -> Optional[Dict[str, Any]]:
        """
        Profile model with PyTorch profiler and return serializable results.

        When GPU is enabled, profiles both CPU and GPU separately.
        When only CPU is used, profiles CPU only.

        Args:
            postprocessor: The trained postprocessor
            config: Test configuration
            num_images: Number of images to profile
            warmup_runs: Warmup iterations before profiling

        Returns:
            Dictionary with profiler results for JSON serialization:
            - If GPU enabled: {"cpu": cpu_results, "gpu": gpu_results}
            - If CPU only: {"cpu": cpu_results}
            Returns None if profiler unavailable.
        """
        if not PROFILER_AVAILABLE:
            self.logger.warning("PyTorch profiler not available")
            return None

        import torch
        import numpy as np

        model = postprocessor.model
        device = postprocessor.device
        img_size = postprocessor.img_size
        is_conv = postprocessor.is_conv
        has_gpu = device.type == "cuda"

        model.eval()

        # Create sample inputs on target device
        if is_conv:
            sample = torch.randn(num_images, 1, img_size, img_size, device=device)
        else:
            sample = torch.randn(num_images, img_size * img_size, device=device)

        # Warmup
        self.logger.debug("Profiler: Running %d warmup iterations", warmup_runs)
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(sample[:1])
                if has_gpu:
                    torch.cuda.synchronize()

        results = {}

        # Profile CPU (always)
        self.logger.info("Running PyTorch profiler (CPU)...")
        try:
            cpu_results = self._run_profiler_pass(
                model, sample, num_images, [ProfilerActivity.CPU], "cpu", has_gpu
            )
            if cpu_results:
                results["cpu"] = cpu_results
        except Exception as e:
            self.logger.error("CPU profiling failed: %s", e)

        # Profile GPU (if available)
        if has_gpu:
            self.logger.info("Running PyTorch profiler (GPU)...")
            try:
                gpu_results = self._run_profiler_pass(
                    model, sample, num_images,
                    [ProfilerActivity.CPU, ProfilerActivity.CUDA], "cuda", has_gpu
                )
                if gpu_results:
                    results["gpu"] = gpu_results
            except Exception as e:
                self.logger.error("GPU profiling failed: %s", e)

        if results:
            self.logger.info("Profiling complete (CPU: %s, GPU: %s)",
                           "yes" if "cpu" in results else "no",
                           "yes" if "gpu" in results else "no")
            return results
        return None

    def _run_profiler_pass(
        self, model, sample, num_images: int,
        activities: list, device_str: str, sync_cuda: bool
    ) -> Optional[Dict[str, Any]]:
        """Run a single profiler pass with specified activities."""
        import torch

        try:
            with profile(
                activities=activities,
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
                with_flops=True
            ) as prof:
                with torch.no_grad():
                    for i in range(num_images):
                        with record_function(f"inference_image_{i}"):
                            _ = model(sample[i:i+1])
                            if sync_cuda:
                                torch.cuda.synchronize()

            # Extract results (JSON-serializable)
            return self._extract_profiler_results(prof, device_str, num_images)

        except Exception as e:
            self.logger.error("Profiler pass failed for %s: %s", device_str, e)
            return None

    def _extract_profiler_results(
        self, prof, device: str, num_images: int
    ) -> Dict[str, Any]:
        """Extract JSON-serializable profiler results."""
        import numpy as np

        is_cuda = "cuda" in device

        # Get key averages
        if is_cuda:
            top_ops = prof.key_averages()
            top_ops = sorted(top_ops, key=lambda x: x.device_time_total, reverse=True)[:20]
        else:
            top_ops = prof.key_averages()
            top_ops = sorted(top_ops, key=lambda x: x.cpu_time_total, reverse=True)[:20]

        # Extract top operations (bottlenecks)
        bottlenecks = []
        for op in top_ops:
            device_time = op.device_time_total if is_cuda else 0
            bottlenecks.append({
                'name': op.key,
                'cpu_time_ms': float(op.cpu_time_total / 1000),
                'cuda_time_ms': float(device_time / 1000),
                'calls': int(op.count),
                'cpu_time_per_call_ms': float(op.cpu_time_total / op.count / 1000) if op.count > 0 else 0,
                'cuda_time_per_call_ms': float(device_time / op.count / 1000) if op.count > 0 and is_cuda else 0,
            })

        # Calculate totals
        total_cpu_time = sum(op.cpu_time_total for op in prof.key_averages()) / 1000
        total_device_time = sum(op.device_time_total for op in prof.key_averages()) / 1000 if is_cuda else 0

        # Group by layer type for pie chart
        layer_breakdown = self._categorize_profiler_ops(prof, is_cuda)

        # Generate summary text
        summary_lines = [
            f"Device: {'CUDA' if is_cuda else 'CPU'}",
            f"Images profiled: {num_images}",
            f"Total CPU time: {total_cpu_time:.2f} ms",
        ]
        if is_cuda:
            summary_lines.append(f"Total CUDA time: {total_device_time:.2f} ms")
        summary_lines.append(f"Avg time per image: {(total_device_time if is_cuda else total_cpu_time) / num_images:.2f} ms")

        return {
            'device': 'cuda' if is_cuda else 'cpu',
            'num_images': num_images,
            'total_cpu_time_ms': float(total_cpu_time),
            'total_cuda_time_ms': float(total_device_time),
            'avg_time_per_image_ms': float((total_device_time if is_cuda else total_cpu_time) / num_images),
            'bottlenecks': bottlenecks,
            'layer_breakdown': layer_breakdown,
            'summary': '\n'.join(summary_lines),
        }

    def _categorize_profiler_ops(self, prof, is_cuda: bool) -> List[Dict[str, Any]]:
        """Categorize profiler operations by type for pie chart."""
        layer_types = {}

        for op in prof.key_averages():
            name = op.key
            name_lower = name.lower()

            # Skip profiler markers
            if (name_lower.startswith('inference_image_') or
                name_lower.startswith('profiler') or
                name_lower.startswith('cudalaunch') or
                name_lower.startswith('enumerate')):
                continue

            # Categorize
            category = self._categorize_operation(name_lower)

            if category not in layer_types:
                layer_types[category] = {'category': category, 'total_time_ms': 0.0}

            # Use self time to avoid double-counting
            if is_cuda:
                time_ms = op.self_device_time_total / 1000
            else:
                time_ms = op.self_cpu_time_total / 1000

            layer_types[category]['total_time_ms'] += time_ms

        # Convert to list and sort
        result = [{'category': k, 'total_time_ms': float(v['total_time_ms'])}
                  for k, v in layer_types.items() if v['total_time_ms'] > 0]
        return sorted(result, key=lambda x: x['total_time_ms'], reverse=True)

    def _categorize_operation(self, name_lower: str) -> str:
        """Categorize a PyTorch operation by its name."""
        # Convolution
        if any(p in name_lower for p in ['conv', 'winograd', 'scudnn', 'cudnn_conv', 'implicit_convolve']):
            return 'Convolution'

        # BatchNorm
        if any(p in name_lower for p in ['batch_norm', 'batchnorm', '_bn', 'cudnn_batch_norm']):
            return 'BatchNorm'

        # Activations
        if any(p in name_lower for p in ['relu', 'leaky_relu', 'prelu', 'elu', 'sigmoid', 'tanh', 'softmax', 'gelu', 'silu']):
            return 'Activation'

        # Pooling
        if 'pool' in name_lower:
            return 'Pooling'

        # Linear/Dense layers
        if any(p in name_lower for p in ['linear', 'matmul', 'gemm', 'addmm', 'cublas', 'mm']):
            return 'Linear'

        # Skip connections
        if 'add_' in name_lower or name_lower.endswith('::add') or 'aten::add' in name_lower:
            return 'Add (Skip)'

        # Concatenation
        if 'cat' in name_lower or 'concat' in name_lower:
            return 'Concatenation'

        # Upsampling
        if any(p in name_lower for p in ['upsample', 'interpolate', 'nearest', 'bilinear']):
            return 'Upsample'

        # Memory operations
        if any(p in name_lower for p in ['copy', 'contiguous', 'clone', 'memcpy', 'memset', 'to']):
            return 'Memory'

        # Reshape operations
        if any(p in name_lower for p in ['view', 'reshape', 'flatten', 'squeeze', 'permute', 'transpose']):
            return 'Reshape'

        return 'Other'

    def _measure_energy(
        self, postprocessor: PostprocessorNN, config: TestConfiguration,
        warmup_runs: int = 5, measurement_runs: int = 10
    ) -> Dict[str, Any]:
        """Measure inference energy with configurable parameters."""
        import torch

        model = postprocessor.model
        device = postprocessor.device
        img_size = postprocessor.img_size
        is_conv = postprocessor.is_conv

        model.eval()

        # Create sample input tensor for measurement
        if is_conv:
            sample = torch.randn(1, 1, img_size, img_size, device=device)
        else:
            sample = torch.randn(1, img_size * img_size, device=device)

        # Create energy analyzer
        analyzer = EnergyAnalyzer(
            model=model,
            device=str(device),
            warmup_runs=warmup_runs,
            measurement_runs=measurement_runs,
            enable_gpu_energy=config.use_gpu,
            enable_cpu_energy=True,
            logger=self.logger
        )

        try:
            if not analyzer.initialize():
                self.logger.warning("Energy analyzer could not be initialized - no backends available")
                return {
                    "energy_error": "No energy measurement backends available"
                }

            self.logger.info("Measuring energy with backends: %s", analyzer.available_backends)

            # Run energy analysis on the sample tensor
            result = analyzer.analyze_inference(
                [sample],
                n_runs=measurement_runs,
                warmup_runs=warmup_runs
            )

            # Build per-backend energy data
            energy_data = {
                # Total/combined values (for backward compatibility)
                "energy_mean_mj": result.mean_energy_mj,
                "energy_std_mj": result.std_energy_joules * 1000,
                "energy_mean_watts": result.mean_power_watts,
                "energy_std_watts": result.std_power_watts,
                "energy_device_name": result.device_name,
                "energy_warmup_runs": warmup_runs,
                "energy_measurement_runs": measurement_runs,
                "efficiency_images_per_joule": result.efficiency_images_per_joule,
                # Per-backend breakdown
                "energy_backends": analyzer.available_backends,
            }

            # Add GPU-specific data if available
            if result.gpu_energy_joules is not None and result.gpu_energy_joules > 0:
                energy_data["energy_gpu_mj"] = result.gpu_energy_joules * 1000
                # Estimate GPU power from energy and time
                if result.mean_time_ms > 0:
                    energy_data["energy_gpu_watts"] = (result.gpu_energy_joules * 1000) / result.mean_time_ms

            # Add CPU-specific data if available
            if result.cpu_energy_joules is not None and result.cpu_energy_joules > 0:
                energy_data["energy_cpu_mj"] = result.cpu_energy_joules * 1000
                # Estimate CPU power from energy and time
                if result.mean_time_ms > 0:
                    energy_data["energy_cpu_watts"] = (result.cpu_energy_joules * 1000) / result.mean_time_ms

            return energy_data

        except Exception as e:
            self.logger.error("Energy measurement failed: %s", e)
            return {
                "energy_error": str(e)
            }
        finally:
            analyzer.shutdown()

    def _should_cancel(self, index: int) -> bool:
        """Check if this test or the batch should be cancelled."""
        return self._cancel_requested or index in self._cancel_test_indices

    def _handle_test_cancel(self, index: int, config: TestConfiguration):
        """Handle test cancellation."""
        self.logger.info("Test %d cancelled: %s", index, config.name)
        config.status = TestStatus.CANCELLED
        with self._results_lock:
            self._all_results.append({
                "name": config.name,
                "status": "cancelled",
                "end_time": datetime.now().isoformat(),
            })
        self.test_cancelled.emit(index)

    def _mark_remaining_cancelled(self, start_index: int):
        """Mark all remaining tests as cancelled."""
        with self._results_lock:
            for i in range(start_index, len(self.batch_config.tests)):
                self.batch_config.tests[i].status = TestStatus.CANCELLED
                self._all_results.append({
                    "name": self.batch_config.tests[i].name,
                    "status": "cancelled",
                })

    def _get_unique_output_dir(self, base_name: str) -> Path:
        """
        Get a unique output directory name, appending _1, _2, etc. if needed.

        Args:
            base_name: The desired folder name

        Returns:
            Path to a unique directory that doesn't exist yet
        """
        base_dir = BATCH_TESTS_DIR / base_name

        if not base_dir.exists():
            return base_dir

        # Directory exists, find a unique suffix
        counter = 1
        while True:
            new_dir = BATCH_TESTS_DIR / f"{base_name}_{counter}"
            if not new_dir.exists():
                return new_dir
            counter += 1

    def _export_all_results(self) -> str:
        """Export all results based on export level."""
        import numpy as np

        # Start export phase (use -1 to indicate batch-level operation)
        self.phase_started.emit(-1, self.PHASE_EXPORT)
        self.status_update.emit("Exporting results...")
        self.phase_progress.emit(-1, self.PHASE_EXPORT, 0)

        # Use batch name or fallback to timestamp
        if self.batch_name:
            base_name = self.batch_name
        else:
            base_name = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        results_dir = self._get_unique_output_dir(base_name)
        results_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("Output directory: %s", results_dir)
        self.phase_progress.emit(-1, self.PHASE_EXPORT, 20)

        # Always export JSON report with .batch_analysis_report extension
        report_path = self._export_results_json(results_dir)
        self.logger.info("Exported analysis report: %s", report_path)
        self.phase_progress.emit(-1, self.PHASE_EXPORT, 40)

        # Export models if requested
        if self.export_level in (ExportLevel.REPORTS_AND_MODELS, ExportLevel.ALL_DATA):
            self._export_models(str(results_dir))
            self.phase_progress.emit(-1, self.PHASE_EXPORT, 70)

        # Export all data if requested
        if self.export_level == ExportLevel.ALL_DATA:
            self._export_datasets(str(results_dir))
            self.phase_progress.emit(-1, self.PHASE_EXPORT, 90)

        self.phase_progress.emit(-1, self.PHASE_EXPORT, 100)
        self.phase_completed.emit(-1, self.PHASE_EXPORT)

        # Return the report file path (not directory) for "Load Executed Batch Test" feature
        return report_path if report_path else str(results_dir)

    def _export_results_json(self, output_dir: Path) -> str:
        """Export all results to JSON file with .batch_analysis_report extension."""
        filename = f"results{FileExtensions.BATCH_ANALYSIS_REPORT}"
        filepath = output_dir / filename

        if not self._all_results:
            self.logger.warning("No results to export")
            return ""

        # Build metadata
        metadata = {
            "created_at": datetime.now().isoformat(),
            "version": "2.0",
            "batch_name": self.batch_name or self.batch_config.name,  # Use export name from UI
            "batch_description": self.batch_config.description,
            "export_level": self.export_level.name,
            "dataset_info": {
                "name": getattr(self.dataset, 'name', 'unknown'),
                "img_size": getattr(self.dataset, 'img_size', 0),
                "num_images": len(self.dataset.data) if hasattr(self.dataset, 'data') else 0,
            },
        }

        # Use the BatchAnalysisReport handler
        BatchAnalysisReport.save(
            results=self._all_results,
            metadata=metadata,
            path=filepath
        )

        self.logger.info("Exported %d results to: %s", len(self._all_results), filepath)
        return str(filepath)

    def _export_models(self, output_dir: str):
        """Export trained models (.pt and ONNX)."""
        import torch

        models_dir = os.path.join(output_dir, "models")
        os.makedirs(models_dir, exist_ok=True)

        for idx, model_data in self._trained_models.items():
            postprocessor = model_data["postprocessor"]
            config = model_data["config"]

            # Safe name for files
            safe_name = config.name.replace(" ", "_").replace("/", "-")

            # Export PyTorch model
            pt_path = os.path.join(models_dir, f"{safe_name}.pt")
            try:
                torch.save(postprocessor.model.state_dict(), pt_path)
                self.logger.info("Exported model: %s", pt_path)
            except Exception as e:
                self.logger.error("Failed to export model %s: %s", safe_name, e)

            # Export ONNX model
            onnx_path = os.path.join(models_dir, f"{safe_name}.onnx")
            try:
                self._export_onnx(postprocessor, onnx_path)
                self.logger.info("Exported ONNX: %s", onnx_path)
            except Exception as e:
                self.logger.warning("Failed to export ONNX %s: %s", safe_name, e)

    def _export_onnx(self, postprocessor, onnx_path: str):
        """Export model to ONNX format."""
        import torch

        model = postprocessor.model
        device = postprocessor.device
        img_size = postprocessor.img_size
        is_conv = postprocessor.is_conv

        model.eval()

        # Create sample input
        if is_conv:
            sample = torch.randn(1, 1, img_size, img_size, device=device)
        else:
            sample = torch.randn(1, img_size * img_size, device=device)

        # Export to ONNX
        torch.onnx.export(
            model,
            sample,
            onnx_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )

    def _export_datasets(self, output_dir: str):
        """Export all test data including masks and inference results (test images).

        Note: We don't export the original training dataset as it's not needed for reports.
        We only export the test images (ground truth, noisy/reconstructed, denoised) per test.
        """
        import numpy as np

        data_dir = os.path.join(output_dir, "data")
        os.makedirs(data_dir, exist_ok=True)

        # Export per-test data
        for idx, test_data in self._test_data.items():
            config = test_data["config"]
            safe_name = config.name.replace(" ", "_").replace("/", "-")
            test_dir = os.path.join(data_dir, safe_name)
            os.makedirs(test_dir, exist_ok=True)

            try:
                # Export mask patterns
                mask = test_data["mask"]
                if hasattr(mask, "mascaras") and mask.mascaras is not None:
                    np.savez_compressed(
                        os.path.join(test_dir, "masks.npz"),
                        masks=mask.mascaras
                    )
                    self.logger.debug("Exported masks for %s", safe_name)

                # Export test images: originals, reconstructions (after mask), denoised (after DNN)
                np.savez_compressed(
                    os.path.join(test_dir, "test_images.npz"),
                    originals=test_data["originals"],
                    reconstructions=test_data["reconstructions"],
                    denoised=test_data["denoised"]
                )
                self.logger.debug("Exported test images for %s", safe_name)

                # Export test configuration as JSON for reference
                config_path = os.path.join(test_dir, "test_config.json")
                config_dict = {
                    "name": config.name,
                    "mask_type": config.mask_type,
                    "reconstruction_method": config.reconstruction_method,
                    "model_name": config.model_name,
                    "epochs": config.epochs,
                    "batch_size": config.batch_size,
                    "learning_rate": config.learning_rate,
                }
                with open(config_path, 'w') as f:
                    json.dump(config_dict, f, indent=2)

                self.logger.info("Exported data for test: %s", safe_name)

            except Exception as e:
                self.logger.error("Failed to export data for %s: %s", safe_name, e)

    # Public methods for cancellation control

    def cancel_all(self):
        """Request cancellation of all remaining tests."""
        self.logger.info("Cancel all requested")
        self._cancel_requested = True

    def cancel_test(self, index: int):
        """Request cancellation of a specific test."""
        self.logger.info("Cancel test %d requested", index)
        self._cancel_test_indices.add(index)
