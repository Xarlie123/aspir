"""
Batch Test Runner - Executes batch test configurations.
Supports both sequential and parallel execution with progress updates and cancellation.
Timing and energy measurements always run sequentially for accuracy.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from threading import Lock
from typing import Any

from PySide6.QtCore import QThread, Signal

from simulation_engine._4_postprocessor.postprocessor_nn import PostprocessorNN
from simulation_engine._5_analyzer.analyzer import Analyzer
from ui.custom_widgets.batch_test.batch_test_runner._energy import measure_energy
from ui.custom_widgets.batch_test.batch_test_runner._export import (
    export_datasets,
    export_models,
    export_results_json,
    get_unique_output_dir,
)
from ui.custom_widgets.batch_test.batch_test_runner._pipeline import (
    create_applicator,
    create_mask,
)
from ui.custom_widgets.batch_test.batch_test_runner._profiling import profile_model
from ui.custom_widgets.batch_test.batch_test_runner._timing import measure_timing
from ui.custom_widgets.batch_test.test_config_model import (
    BatchTestConfig,
    ExportLevel,
    TestConfiguration,
    TestStatus,
)

# Frames exported as INT8 quantization calibration set. A few hundred inputs are
# enough to observe activation ranges on the networks used here; None exports the
# whole train split.
CALIBRATION_LIMIT = 200


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
    test_started = Signal(int)
    test_progress = Signal(int, int)  # index, progress %
    test_completed = Signal(int, dict)  # index, results dict
    test_failed = Signal(int, str)  # index, error message
    test_cancelled = Signal(int)
    batch_completed = Signal(str)  # results path
    batch_cancelled = Signal()
    status_update = Signal(str)

    # Phase-specific progress signals (index, phase_name, progress 0-100)
    phase_started = Signal(int, str)        # Test index, Phase name started
    phase_progress = Signal(int, str, int)  # Test index, Phase name, progress 0-100
    phase_completed = Signal(int, str)      # Test index, Phase name completed

    # Phase names constants
    PHASE_BASELINE = "Idle Baseline"
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
        self._all_results: list[dict[str, Any]] = []
        self._trained_models: dict[int, Any] = {}  # Store models for export
        self._test_data: dict[int, dict] = {}  # Store test data for export
        self._results_lock = Lock()  # Thread safety for results
        self._baseline = None  # set by _capture_baseline_phase if enabled

    def run(self):
        """Execute all tests in the batch (sequential or parallel based on config)."""
        self.logger.info("Starting batch test run with %d tests", len(self.batch_config.tests))
        self._cancel_requested = False
        self._all_results = []

        try:
            # Capture system idle power before any test starts so the
            # serializer can derive dynamic-power columns. Skipped (and
            # the report ends up with ``baseline = null``) when the
            # batch config opts out, when no test in the batch enables
            # the energy report, or when no energy backend is
            # available on this host.
            self._capture_baseline_phase()

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

    def _merge_dynamic_metrics(self, results: dict, test_name: str) -> None:
        """Add ``dynamic_power_W`` / ``dynamic_energy_mj`` /
        ``dynamic_efficiency_imgs_per_J`` to ``results`` using the
        baseline captured at the start of the batch.

        No-op when the baseline is missing — those columns will be
        absent from the report (consumers detect that and render `-`).
        Negative dynamic power is *not* clipped: it's a real
        diagnostic signal that the host warmed up between baseline and
        test, so we surface a WARNING and persist the negative value.
        """
        if self._baseline is None:
            return
        from simulation_engine._5_analyzer.baseline_capture import (
            derive_dynamic_metrics,
        )
        dyn = derive_dynamic_metrics(
            energy_mean_mj=results.get("energy_mean_mj"),
            energy_mean_watts=results.get("energy_mean_watts"),
            baseline_power_W=self._baseline.total_power_W,
        )
        results.update(dyn)
        if dyn.get("dynamic_power_W") is not None and dyn["dynamic_power_W"] < 0:
            self.logger.warning(
                "Dynamic power negative for '%s': baseline %.2f W > test "
                "power %.2f W — check thermal state or sampling protocol.",
                test_name, self._baseline.total_power_W,
                float(results.get("energy_mean_watts") or 0.0),
            )

    def _capture_baseline_phase(self) -> None:
        """Sample idle power before the first test, store on
        ``self._baseline`` for the export step."""
        if not getattr(self.batch_config, "capture_baseline", True):
            self.logger.info("Idle-baseline capture disabled in batch config.")
            return
        # No point measuring idle power when no test even asks for the
        # energy report — the dynamic columns wouldn't have a numerator.
        any_energy = any(
            "energy" in (t.reports or [])
            for t in self.batch_config.tests
        )
        if not any_energy:
            self.logger.info(
                "Idle-baseline capture skipped: no test in this batch "
                "has the energy report enabled."
            )
            return

        from simulation_engine._5_analyzer.energy_backends._monitor import (
            EnergyMonitor,
        )
        from simulation_engine._5_analyzer.baseline_capture import (
            capture_idle_baseline,
        )

        duration = max(30, min(300, int(getattr(
            self.batch_config, "baseline_duration_s", 60))))
        # Decide GPU enable from the first test that wants energy —
        # close enough for the baseline (we just want all available
        # backends online for the same span as the test's own ones).
        first_energy_test = next(
            (t for t in self.batch_config.tests if "energy" in (t.reports or [])),
            None,
        )
        use_gpu = bool(getattr(first_energy_test, "use_gpu", True))

        monitor = EnergyMonitor(
            enable_gpu=use_gpu,
            enable_cpu=True,
            enable_jetson=True,
            logger=self.logger,
        )
        if not monitor.initialize():
            self.logger.warning(
                "No energy backend available — skipping idle-baseline capture."
            )
            return

        self.phase_started.emit(-1, self.PHASE_BASELINE)
        self.status_update.emit(
            f"Capturing idle baseline ({duration} s)…"
        )

        def _tick(elapsed_s: float, total_s: float) -> None:
            pct = int(min(100.0, 100.0 * elapsed_s / max(total_s, 1.0)))
            self.phase_progress.emit(-1, self.PHASE_BASELINE, pct)
            self.status_update.emit(
                f"Capturing idle baseline ({int(elapsed_s)} / {int(total_s)} s)"
            )

        try:
            self._baseline = capture_idle_baseline(
                monitor,
                duration_s=duration,
                logger=self.logger,
                progress_callback=_tick,
                cancel_check=lambda: self._cancel_requested,
            )
        finally:
            try:
                monitor.shutdown()
            except Exception:
                self.logger.debug("Baseline monitor shutdown raised", exc_info=True)
            self.phase_completed.emit(-1, self.PHASE_BASELINE)

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

    def _execute_parallel_training(self, tests: list[tuple], num_threads: int):
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
            "architecture_config": dict(config.architecture_config or {}),
            "train_split": config.train_split,
            "val_split": config.val_split,
            "test_split": config.test_split,
            "start_time": datetime.now().isoformat(),
        }

        # Phase 1: Create mask
        self.phase_started.emit(index, self.PHASE_MASKS)
        self.status_update.emit(f"Generating masks: {config.name}")
        self.phase_progress.emit(index, self.PHASE_MASKS, 0)
        mask = create_mask(config, self.dataset, self.logger)
        self.phase_progress.emit(index, self.PHASE_MASKS, 100)
        self.phase_completed.emit(index, self.PHASE_MASKS)
        self.test_progress.emit(index, 10)

        if self._should_cancel(index):
            return self._handle_test_cancel(index, config)

        # Phase 2: Create applicator and reconstruct
        self.phase_started.emit(index, self.PHASE_RECONSTRUCTION)
        self.status_update.emit(f"Creating applicator: {config.name}")
        self.phase_progress.emit(index, self.PHASE_RECONSTRUCTION, 0)
        applicator = create_applicator(config, mask, self.dataset)

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
                        **self._calibration_payload(postprocessor),
                    }
                self._all_results.append(results)

            self.test_completed.emit(index, results)
            self.logger.info("Test %d completed (no timing/energy): %s", index, config.name)

    def _calibration_payload(self, postprocessor: PostprocessorNN) -> dict:
        """Build the INT8 calibration entries for a test's export record.

        Returns train-split inputs (never test frames, which would leak the
        evaluation distribution into the quantized model) plus the normalization
        constants needed to reproduce the reconstruction downstream.

        Never raises: this is an additive export, and a failure here must not
        cost the caller the data it already collected.
        """
        try:
            calib_orig, calib_recons = postprocessor.train_dataset(limit=CALIBRATION_LIMIT)
            indices = list(getattr(postprocessor, "train_indices", []))[:len(calib_recons)]
            return {
                "calibration_originals": calib_orig,
                "calibration_reconstructions": calib_recons,
                "calibration_indices": indices,
                "norm_lo": getattr(postprocessor, "norm_lo", None),
                "norm_span": getattr(postprocessor, "norm_span", None),
            }
        except Exception as e:
            self.logger.warning("Failed to collect calibration data: %s", e)
            return {}

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
                timing_results = measure_timing(
                    postprocessor, config, self.dataset, self.logger,
                    applicator=applicator,
                    warmup_runs=config.timing_warmup_runs,
                    measurement_runs=config.timing_measurement_runs,
                    sampling_rate_khz=config.timing_sampling_rate_khz
                )
                results.update(timing_results)
                results["timing_metrics"] = results.get("timing_metrics", {})

                # Run PyTorch profiler and store results
                self.status_update.emit(f"Running profiler: {config.name}")
                profiler_results = profile_model(
                    postprocessor, config, self.logger,
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
                energy_results = measure_energy(
                    postprocessor, config, self.logger,
                    warmup_runs=config.timing_warmup_runs,
                    measurement_runs=config.timing_measurement_runs
                )
                results.update(energy_results)
                self._merge_dynamic_metrics(results, config.name)
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
                # PostprocessorNN does not expose .mask; pull it from the
                # applicator (applicator.mask is set in ApplicatorABC.__init__).
                mask_obj = getattr(applicator, 'mask', None) if applicator else None
                self._test_data[index] = {
                    "mask": mask_obj,
                    "applicator": applicator,
                    "originals": orig,
                    "reconstructions": recons,
                    "denoised": denoised,
                    "config": config,
                    **self._calibration_payload(postprocessor),
                }

            self._all_results.append(results)

        self.test_completed.emit(index, results)
        self.logger.info("Test %d analysis complete: %s", index, config.name)

    def _execute_parallel_tests(self, tests: list[tuple], num_threads: int):
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
            "architecture_config": dict(config.architecture_config or {}),
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
            mask = create_mask(config, self.dataset, self.logger)
            self.phase_progress.emit(index, self.PHASE_MASKS, 100)
            self.phase_completed.emit(index, self.PHASE_MASKS)
            self.test_progress.emit(index, 10)

            if self._should_cancel(index):
                return self._handle_test_cancel(index, config)

            # Phase 2: Create applicator
            self.phase_started.emit(index, self.PHASE_RECONSTRUCTION)
            self.status_update.emit(f"Creating applicator: {config.name}")
            self.phase_progress.emit(index, self.PHASE_RECONSTRUCTION, 0)
            applicator = create_applicator(config, mask, self.dataset)

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
                        **self._calibration_payload(postprocessor),
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
        # Start with per-model architecture parameters, then add training-level
        # overrides (e.g. dropout from the DNN panel takes precedence).
        model_overrides = dict(config.architecture_config or {})
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
                         applicator=None) -> dict[str, Any]:
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
                timing_results = measure_timing(
                    postprocessor, config, self.dataset, self.logger,
                    applicator=applicator,
                    warmup_runs=config.timing_warmup_runs,
                    measurement_runs=config.timing_measurement_runs,
                    sampling_rate_khz=config.timing_sampling_rate_khz
                )
                results.update(timing_results)

                # Run PyTorch profiler and store results
                profiler_results = profile_model(
                    postprocessor, config, self.logger,
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
                energy_results = measure_energy(
                    postprocessor, config, self.logger,
                    warmup_runs=config.timing_warmup_runs,
                    measurement_runs=config.timing_measurement_runs
                )
                results.update(energy_results)
                self._merge_dynamic_metrics(results, config.name)
            except Exception as e:
                self.logger.warning("Energy analysis failed: %s", e)
                results["energy_error"] = str(e)

        return results

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

    def _export_all_results(self) -> str:
        """Export all results based on export level."""
        # Start export phase (use -1 to indicate batch-level operation)
        self.phase_started.emit(-1, self.PHASE_EXPORT)
        self.status_update.emit("Exporting results...")
        self.phase_progress.emit(-1, self.PHASE_EXPORT, 0)

        # Use batch name or fallback to timestamp
        if self.batch_name:
            base_name = self.batch_name
        else:
            base_name = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        results_dir = get_unique_output_dir(base_name)
        results_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("Output directory: %s", results_dir)
        self.phase_progress.emit(-1, self.PHASE_EXPORT, 20)

        # Always export JSON report with .batch_analysis_report extension
        report_path = export_results_json(
            self._all_results, self.dataset, self.batch_name,
            self.batch_config, self.export_level, self.logger, results_dir,
            idle_baseline=(self._baseline.to_dict()
                           if self._baseline is not None else None),
        )
        self.logger.info("Exported analysis report: %s", report_path)
        self.phase_progress.emit(-1, self.PHASE_EXPORT, 40)

        # Export models if requested
        if self.export_level in (ExportLevel.REPORTS_AND_MODELS, ExportLevel.ALL_DATA):
            export_models(self._trained_models, self.logger, str(results_dir))
            self.phase_progress.emit(-1, self.PHASE_EXPORT, 70)

        # Export all data if requested
        if self.export_level == ExportLevel.ALL_DATA:
            export_datasets(self._test_data, self.logger, str(results_dir))
            self.phase_progress.emit(-1, self.PHASE_EXPORT, 90)

        self.phase_progress.emit(-1, self.PHASE_EXPORT, 100)
        self.phase_completed.emit(-1, self.PHASE_EXPORT)

        # Return the report file path (not directory) for "Load Executed Batch Test" feature
        return report_path if report_path else str(results_dir)

    # Public methods for cancellation control

    def cancel_all(self):
        """Request cancellation of all remaining tests."""
        self.logger.info("Cancel all requested")
        self._cancel_requested = True

    def cancel_test(self, index: int):
        """Request cancellation of a specific test."""
        self.logger.info("Cancel test %d requested", index)
        self._cancel_test_indices.add(index)
