"""Handler for the unified Reports tab (Quality Metrics + Timing Analysis + Energy Analysis)."""
import logging
import time
import numpy as np
import torch
from PyQt5.QtWidgets import (
    QMessageBox, QListWidget, QStackedWidget, QLabel,
    QVBoxLayout, QWidget, QScrollArea, QSizePolicy, QFrame
)
from PyQt5.QtCore import QObject, QThread

from ui.custom_widgets.quality_metrics.quality_metrics_page import QualityMetricsPage
from ui.custom_widgets.timing_analysis.timing_analysis_page import TimingAnalysisPage
from ui.custom_widgets.energy_analysis.energy_analysis_page import EnergyAnalysisPage
from ui.custom_widgets.energy_analysis.energy_worker import EnergyMeasurementWorker

from simulation_engine._5_analyzer.analyzer import Analyzer


class UIReportsHandler(QObject):
    """
    Unified handler for the Reports tab.
    Combines Quality Metrics, Timing Analysis, and Energy Analysis in a menu-based interface.
    """

    def __init__(self, ui, simulation, logger, status_manager=None):
        super().__init__()
        self.ui = ui
        self.simulation = simulation
        self.status_manager = status_manager

        self.logger = logger.getChild("UIReportsHandler")
        self.logger.setLevel(logging.DEBUG)
        self.logger.debug("Initializing UIReportsHandler")

        # Create widget for Quality Metrics (new design)
        self.quality_metrics_page = QualityMetricsPage(
            parent=None,
            logger=self.logger
        )
        self.quality_metrics_page.analysisRequested.connect(self._do_quality_analysis)
        self.logger.debug("Created QualityMetricsPage widget")

        # Create widget for Timing Analysis (new design)
        self.timing_page = TimingAnalysisPage(
            parent=None,
            logger=self.logger
        )
        self.timing_page.analysisRequested.connect(self._on_analyze_times)
        self.timing_page.profilingRequested.connect(self._on_profile_bottlenecks)
        self.timing_page.nsightProfilingRequested.connect(self._on_nsight_profiling)
        self.logger.debug("Created TimingAnalysisPage widget")

        # Create widget for Energy Analysis
        self.energy_page = EnergyAnalysisPage(
            parent=None,
            logger=self.logger
        )
        self.energy_page.analysisRequested.connect(self._on_analyze_energy)
        self.energy_page.detect_button.clicked.connect(self._on_detect_energy_backends)
        self.logger.debug("Created EnergyAnalysisPage widget")

        # Worker references for background tasks
        self._energy_worker = None
        self._energy_thread = None

        # Setup menu-based interface
        self._setup_menu_interface()
        self.logger.info("UIReportsHandler initialization complete")

    def _setup_menu_interface(self):
        """Setup menu-based interface with QListWidget and QStackedWidget."""
        self.logger.debug("Setting up menu-based interface for Reports")

        # Create QListWidget for menu
        self.reports_menu = QListWidget()
        self.reports_menu.addItems([
            "Quality Metrics",
            "Timing Analysis",
            "Energy Analysis"
        ])
        self.reports_menu.setCurrentRow(0)
        self.reports_menu.currentRowChanged.connect(self._on_menu_selection_changed)

        # Style the menu
        self.reports_menu.setStyleSheet("""
            QListWidget {
                border: 1px solid #ccc;
                border-radius: 4px;
                background-color: #f5f5f5;
            }
            QListWidget::item {
                padding: 10px;
                border-bottom: 1px solid #e0e0e0;
            }
            QListWidget::item:selected {
                background-color: #0078d7;
                color: white;
            }
            QListWidget::item:hover {
                background-color: #e5e5e5;
            }
        """)

        # Create QStackedWidget for content pages
        self.logger.debug("Creating QStackedWidget for Reports content")
        self.reports_stacked = QStackedWidget()

        # Common style for content panels
        panel_style = """
            QWidget#contentPanel {
                background-color: #f5f5f5;
                border: 1px solid #ccc;
                border-radius: 4px;
            }
        """

        # Page 0: Quality Metrics
        quality_page = self._create_quality_metrics_page(panel_style)
        self.reports_stacked.addWidget(quality_page)
        self.logger.debug("Added page 0: Quality Metrics")

        # Page 1: Timing Analysis
        timing_page = self._create_timing_analysis_page(panel_style)
        self.reports_stacked.addWidget(timing_page)
        self.logger.debug("Added page 1: Timing Analysis")

        # Page 2: Energy Analysis
        energy_page = self._create_energy_analysis_page(panel_style)
        self.reports_stacked.addWidget(energy_page)
        self.logger.debug("Added page 2: Energy Analysis")

        # Embed menu into placeholder
        self.logger.debug("Embedding menu into reports_menu_placeholder")
        menu_layout = QVBoxLayout(self.ui.reports_menu_placeholder)
        menu_layout.setContentsMargins(0, 0, 0, 0)
        menu_layout.addWidget(self.reports_menu)

        # Embed stacked widget into content placeholder
        self.logger.debug("Embedding stacked widget into reports_content_placeholder")
        content_layout = QVBoxLayout(self.ui.reports_content_placeholder)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.addWidget(self.reports_stacked)

        # Explicitly set the first page as current
        self.reports_stacked.setCurrentIndex(0)
        self.logger.debug("Set initial stacked widget page to index 0")

        self.logger.info("Menu-based interface setup complete for Reports")

    def _create_quality_metrics_page(self, panel_style):
        """Create the Quality Metrics page."""
        page = QWidget()
        page.setObjectName("contentPanel")
        page.setStyleSheet(panel_style)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # Add the quality metrics page widget
        self.quality_metrics_page.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.quality_metrics_page)

        return page

    def _create_timing_analysis_page(self, panel_style):
        """Create the Timing Analysis page."""
        page = QWidget()
        page.setObjectName("contentPanel")
        page.setStyleSheet(panel_style)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # Add the timing analysis page widget (similar to quality metrics)
        self.timing_page.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.timing_page)

        return page

    def _create_energy_analysis_page(self, panel_style):
        """Create the Energy Analysis page."""
        page = QWidget()
        page.setObjectName("contentPanel")
        page.setStyleSheet(panel_style)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # Add the energy analysis page widget
        self.energy_page.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.energy_page)

        return page

    def _on_menu_selection_changed(self, index):
        """Switch stacked widget page when menu selection changes."""
        self.logger.debug(f"Changing report type to index {index}")
        self.reports_stacked.setCurrentIndex(index)

    # -------------------------------------------------------------------------
    # Quality Metrics Analysis Methods
    # -------------------------------------------------------------------------

    def _do_quality_analysis(self):
        """
        Perform quality analysis using stored validation results,
        or synthesize them from the loaded model if possible.
        """
        self.logger.info("Starting quality metrics analysis")

        if self.status_manager:
            self.status_manager.start_task("Quality analysis")

        vr = getattr(self.simulation, 'validation_results', None)

        # Auto-synthesize preview if missing but a trained/loaded model exists
        if not vr:
            post = getattr(self.simulation, 'postprocessor', None)
            if post is not None and getattr(post, 'trained', False):
                self.logger.debug("No validation_results; generating from trained model...")
                try:
                    orig, recons, denoised = post.test_dataset()
                    vr = self.simulation.validation_results = {
                        'original': orig,
                        'recons': recons,
                        'denoised': denoised
                    }
                except Exception as e:
                    self.logger.error("Could not generate validation preview: %s", e, exc_info=True)
                    if self.status_manager:
                        self.status_manager.error_task("No validation data")
                    QMessageBox.warning(None, "Attention", "Run postprocessing first.")
                    return
            else:
                self.logger.warning("No validation_results found, aborting analysis")
                if self.status_manager:
                    self.status_manager.error_task("No validation data")
                QMessageBox.warning(None, "Attention", "Run postprocessing first.")
                return

        # Use validation_results
        orig, recons, denoised = vr['original'], vr['recons'], vr['denoised']
        self.logger.debug(
            "Validation results loaded: orig=%d, recons=%d, denoised=%d",
            len(orig), len(recons), len(denoised)
        )

        # Analyze quality metrics
        analyzer = Analyzer(orig, recons, denoised)
        try:
            # Unpack 6 values (PSNR, SSIM, LPIPS for noisy & reconstructed)
            pn, sn, ln, pr, sr, lr = analyzer.analyze_noise()
            self.logger.info("Quality analysis completed")
        except Exception as e:
            self.logger.error("Error during quality analysis: %s", e, exc_info=True)
            if self.status_manager:
                self.status_manager.error_task(str(e)[:50])
            QMessageBox.critical(None, "Analysis Error", str(e))
            return

        # Update the quality metrics page with data
        try:
            self.quality_metrics_page.set_data(
                orig_images=orig,
                noisy_images=recons,
                recon_images=denoised,
                psnr_noisy=pn,
                ssim_noisy=sn,
                lpips_noisy=ln,
                psnr_recon=pr,
                ssim_recon=sr,
                lpips_recon=lr
            )
            self.logger.debug("Quality metrics page updated with analysis results")
        except Exception as e:
            self.logger.error("Error updating quality metrics page: %s", e, exc_info=True)

        if self.status_manager:
            self.status_manager.finish_task()

        self.logger.info("Quality analysis results displayed")

    # -------------------------------------------------------------------------
    # Timing Analysis Methods
    # -------------------------------------------------------------------------

    def _ensure_post_ready(self) -> bool:
        """
        Ensure there is a postprocessor and it is 'trained' or at least has a loaded model.
        Returns True if ready, False otherwise.
        """
        post = getattr(self.simulation, 'postprocessor', None)
        if post is None:
            self.logger.warning("No postprocessor found in simulation.")
            return False

        # If trained flag is False but the model exists (loaded from disk), consider it ready.
        if not getattr(post, 'trained', False):
            has_model = hasattr(post, 'model') and (post.model is not None)
            if has_model:
                try:
                    # Promote to 'trained' state because weights are present
                    post.trained = True
                    self.logger.debug("Postprocessor had weights but trained=False; forcing trained=True.")
                except Exception:
                    pass

        if not getattr(post, 'trained', False):
            self.logger.warning("Postprocessor not trained (or no weights loaded).")
            return False

        return True

    def _measure_per_image_times(self, device, test_images, warmup_runs=5, measurement_runs=20):
        """
        Per-image measurement using the TEST/VALIDATION dataset:
          - Reconstruction on CPU (no GPU syncs)
          - Denoising on selected 'device'
        Returns two lists (ms): recon_times, denoise_times

        Args:
            device: torch device for denoising
            test_images: list of reconstructed images from validation set (noisy inputs to DNN)
            warmup_runs: number of warmup iterations before timing
            measurement_runs: number of measurement iterations per image
        """
        recon_times = []
        denoise_times = []

        aplic = getattr(self.simulation, 'applicator', None)
        post = getattr(self.simulation, 'postprocessor', None)

        if aplic is None or post is None:
            self.logger.error("Missing 'applicator' or 'postprocessor' in simulation.")
            return recon_times, denoise_times

        n_test_images = len(test_images)
        self.logger.debug(
            "Timing config: warmup_runs=%d, measurement_runs=%d, n_test_images=%d",
            warmup_runs, measurement_runs, n_test_images
        )
        self.logger.info("Using TEST dataset (%d images) for timing analysis", n_test_images)

        # 1) Reconstruction timing (CPU only)
        # Measure reconstruction time on a subset equal to the test set size
        dataset_size = len(getattr(self.simulation.dataset, 'data', []))
        n_recon_samples = min(n_test_images, dataset_size)

        # Warmup runs for reconstruction (ensures fair comparison with Batch Test)
        warmup_recon = min(2, n_recon_samples)
        for idx in range(warmup_recon):
            _ = aplic.process_image(idx)

        # Timed reconstruction runs
        for idx in range(n_recon_samples):
            t0 = time.perf_counter()
            _ = aplic.process_image(idx)
            t1 = time.perf_counter()
            recon_times.append((t1 - t0) * 1000.0)  # ms

        # 2) Denoising timing using TEST images (model on 'device')
        # Prepare sample tensor for warmup using first test image
        sample_img = np.array(test_images[0], dtype=np.float32)
        if getattr(post, 'is_conv', False):
            sample_tensor = torch.from_numpy(sample_img).unsqueeze(0).unsqueeze(0).to(device)
        else:
            sample_tensor = torch.from_numpy(sample_img.flatten()).unsqueeze(0).to(device)

        # Warmup runs
        self.logger.debug("Starting warmup runs (%d iterations)", warmup_runs)
        post.model.eval()
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = post.model(sample_tensor)
                if device.type == 'cuda':
                    torch.cuda.synchronize()

            # Measurement runs per TEST image
            for img in test_images:
                arr = np.array(img, dtype=np.float32)
                if getattr(post, 'is_conv', False):
                    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
                else:
                    tensor = torch.from_numpy(arr.flatten()).unsqueeze(0)

                tensor = tensor.to(device)

                # Average over multiple runs for more stable timing
                run_times = []
                for _ in range(measurement_runs):
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    _ = post.model(tensor)
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    run_times.append((t1 - t0) * 1000.0)

                # Use mean of measurement runs
                denoise_times.append(float(np.mean(run_times)))

        return recon_times, denoise_times

    def _on_analyze_times(self):
        """Called when the user clicks 'Analyze times'."""
        self.logger.info("Starting time analysis")

        if self.status_manager:
            self.status_manager.start_task("Timing analysis")

        # Ensure the postprocessor is present and marked as ready
        if not self._ensure_post_ready():
            if self.status_manager:
                self.status_manager.error_task("No trained model")
            QMessageBox.warning(None, "Attention", "Train or load a trained model first.")
            return

        post = self.simulation.postprocessor

        # Read UI parameters from timing page properties
        sampling_rate_khz = self.timing_page.sampling_rate_khz
        use_gpu = self.timing_page.use_gpu
        warmup_runs = self.timing_page.warmup_runs
        measurement_runs = self.timing_page.measurement_runs

        self.logger.debug(
            "Timing params: sampling_rate=%.3f kHz, use_gpu=%s, warmup=%d, runs=%d",
            sampling_rate_khz, use_gpu, warmup_runs, measurement_runs
        )

        # Build analyzer facade data (orig, recon, denoised) - use CPU first
        try:
            post.device = torch.device('cpu')
            post.model = post.model.to('cpu')
            origs, recons, denoised = post.test_dataset()
        except Exception as e:
            self.logger.error("Failed to run test_dataset(): %s", e, exc_info=True)
            if self.status_manager:
                self.status_manager.error_task("Test failed")
            QMessageBox.critical(None, "Error", f"Could not run validation inference: {e}")
            return

        # Keep a reference in simulation (handy for other analysis)
        self.simulation.validation_results = {
            'original': origs,
            'recons': recons,
            'denoised': denoised
        }

        # Calculate acquisition time
        num_masks = len(getattr(self.simulation.mask, 'masks', []))
        t_acq_single_s = 1.0 / (sampling_rate_khz * 1e3) if sampling_rate_khz > 0 else 0.0
        t_acq_ms = t_acq_single_s * num_masks * 1000.0

        # Measure CPU times
        self.logger.info("Measuring CPU inference times...")
        cpu_device = torch.device('cpu')
        post.model = post.model.to(cpu_device)
        recon_times_ms, denoise_times_cpu_ms = self._measure_per_image_times(
            device=cpu_device,
            test_images=recons,
            warmup_runs=warmup_runs,
            measurement_runs=measurement_runs
        )

        if len(recon_times_ms) == 0 or len(denoise_times_cpu_ms) == 0:
            self.logger.warning("No CPU timing data collected (empty lists).")
            if self.status_manager:
                self.status_manager.error_task("No timing data")
            QMessageBox.warning(None, "Attention", "Could not measure times (empty lists).")
            return

        mean_recon_ms = float(np.mean(recon_times_ms))
        mean_denoise_cpu_ms = float(np.mean(denoise_times_cpu_ms))
        self.logger.debug("CPU inference mean: %.3f ms", mean_denoise_cpu_ms)

        # Measure GPU times if requested and available
        denoise_times_gpu_ms = None
        mean_denoise_gpu_ms = None

        if use_gpu and torch.cuda.is_available():
            self.logger.info("Measuring GPU inference times...")
            gpu_device = torch.device('cuda')
            try:
                post.model = post.model.to(gpu_device)
                _, denoise_times_gpu_ms = self._measure_per_image_times(
                    device=gpu_device,
                    test_images=recons,
                    warmup_runs=warmup_runs,
                    measurement_runs=measurement_runs
                )
                if denoise_times_gpu_ms:
                    mean_denoise_gpu_ms = float(np.mean(denoise_times_gpu_ms))
                    self.logger.debug("GPU inference mean: %.3f ms", mean_denoise_gpu_ms)
            except Exception as e:
                self.logger.error("GPU measurement failed: %s", e, exc_info=True)

        # Prepare timing data dictionary for the page
        timing_data = {
            't_acq_ms': t_acq_ms,
            't_recon_ms': mean_recon_ms,
            't_inf_cpu_ms': mean_denoise_cpu_ms,
            't_inf_gpu_ms': mean_denoise_gpu_ms,
            'recon_times_ms': recon_times_ms,
            'denoise_times_cpu_ms': denoise_times_cpu_ms,
            'denoise_times_gpu_ms': denoise_times_gpu_ms if denoise_times_gpu_ms else [],
        }

        # Update the timing page with data
        try:
            self.timing_page.set_data(timing_data)
            self.logger.debug("Timing page updated with analysis results")
        except Exception as e:
            self.logger.error("Error updating timing page: %s", e, exc_info=True)

        if self.status_manager:
            self.status_manager.finish_task()

        self.logger.info("Time analysis completed")

    def _on_profile_bottlenecks(self):
        """Open the profiler popup to analyze performance bottlenecks."""
        self.logger.info("Opening profiler for bottleneck analysis")

        from ui.custom_widgets.timing_analysis.profiler_results_popup import ProfilerResultsPopup

        # Get GPU setting from timing page
        use_gpu = self.timing_page.use_gpu

        popup = ProfilerResultsPopup(
            parent=None,
            simulation=self.simulation,
            use_gpu=use_gpu,
            logger=self.logger
        )
        popup.exec_()

    def _on_nsight_profiling(self):
        """Open the Nsight Systems profiler popup."""
        self.logger.info("Opening Nsight Systems profiler")

        from ui.custom_widgets.timing_analysis.nsight_profiler_popup import NsightProfilerPopup

        popup = NsightProfilerPopup(
            parent=None,
            simulation=self.simulation,
            logger=self.logger
        )
        popup.exec_()

    # -------------------------------------------------------------------------
    # Energy Analysis Methods
    # -------------------------------------------------------------------------

    def _cleanup_energy_worker(self):
        """Clean up energy worker thread."""
        if self._energy_thread is not None:
            if self._energy_thread.isRunning():
                self._energy_worker.cancel()
                self._energy_thread.quit()
                self._energy_thread.wait(2000)
            self._energy_thread = None
            self._energy_worker = None

    def _on_detect_energy_backends(self):
        """Detect available energy measurement backends."""
        self.logger.info("Detecting energy backends...")

        if self.status_manager:
            self.status_manager.start_task("Detecting energy backends")

        # Clean up any existing worker
        self._cleanup_energy_worker()

        # Create worker for detection
        # Always enable both backends during detection to discover what's available
        # The checkboxes are used later for actual measurements, not detection
        self._energy_worker = EnergyMeasurementWorker(
            mode="detect",
            enable_gpu_energy=True,
            enable_cpu_energy=True,
            pmlib_server_ip=self.energy_page.pmlib_server_ip,
            pmlib_server_port=self.energy_page.pmlib_server_port,
            logger=self.logger
        )

        # Create thread
        self._energy_thread = QThread()
        self._energy_worker.moveToThread(self._energy_thread)

        # Connect signals
        self._energy_thread.started.connect(self._energy_worker.run)
        self._energy_worker.platform_detected.connect(self._on_platform_detected)
        self._energy_worker.error.connect(self._on_energy_error)
        self._energy_worker.finished.connect(self._on_energy_detection_finished)

        # Start thread
        self._energy_thread.start()

    def _on_platform_detected(self, platform_info, backends):
        """Handle platform detection results."""
        self.logger.info(f"Platform detected: {len(backends)} backend(s)")
        self.energy_page.set_platform_info(platform_info, backends)

        if self.status_manager:
            self.status_manager.finish_task()

    def _on_energy_detection_finished(self):
        """Handle energy detection completion."""
        self.logger.debug("Energy detection finished")
        self._cleanup_energy_worker()

    def _on_analyze_energy(self):
        """Run energy analysis on test images."""
        self.logger.info("Starting energy analysis")

        if self.status_manager:
            self.status_manager.start_task("Energy analysis")

        # Ensure the postprocessor is present and ready
        if not self._ensure_post_ready():
            if self.status_manager:
                self.status_manager.error_task("No trained model")
            QMessageBox.warning(None, "Attention", "Train or load a trained model first.")
            return

        post = self.simulation.postprocessor

        # Get test images
        vr = getattr(self.simulation, 'validation_results', None)
        if not vr:
            self.logger.debug("No validation_results; generating from trained model...")
            try:
                orig, recons, denoised = post.test_dataset()
                vr = self.simulation.validation_results = {
                    'original': orig,
                    'recons': recons,
                    'denoised': denoised
                }
            except Exception as e:
                self.logger.error("Could not generate validation preview: %s", e, exc_info=True)
                if self.status_manager:
                    self.status_manager.error_task("No validation data")
                QMessageBox.warning(None, "Attention", "Run postprocessing first.")
                return

        all_test_images = vr['recons']  # Use reconstructed (noisy) images as input to DNN

        # Update the energy page with available image count
        self.energy_page.set_max_test_images(len(all_test_images))

        # Apply image range selection
        start_idx = self.energy_page.image_start_index
        end_idx = self.energy_page.image_end_index
        test_images = all_test_images[start_idx:end_idx]

        self.logger.debug(f"Using {len(test_images)} test images for energy analysis (indices {start_idx} to {end_idx - 1})")

        # Determine device based on where the model is (not which energy backends are enabled)
        # This allows measuring CPU energy while running inference on GPU
        try:
            model_device = next(post.model.parameters()).device
            device = str(model_device)
        except StopIteration:
            # Model has no parameters, default to cuda if available
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Clean up any existing worker
        self._cleanup_energy_worker()

        # Create worker for measurement
        self._energy_worker = EnergyMeasurementWorker(
            mode="measure",
            model=post.model,
            test_images=test_images,
            device=device,
            warmup_runs=self.energy_page.warmup_runs,
            measurement_runs=self.energy_page.measurement_runs,
            enable_gpu_energy=self.energy_page.enable_gpu_energy,
            enable_cpu_energy=self.energy_page.enable_cpu_energy,
            pmlib_server_ip=self.energy_page.pmlib_server_ip,
            pmlib_server_port=self.energy_page.pmlib_server_port,
            logger=self.logger
        )

        # Create thread
        self._energy_thread = QThread()
        self._energy_worker.moveToThread(self._energy_thread)

        # Connect signals
        self._energy_thread.started.connect(self._energy_worker.run)
        self._energy_worker.progress.connect(self._on_energy_progress)
        self._energy_worker.result.connect(self._on_energy_result)
        self._energy_worker.error.connect(self._on_energy_error)
        self._energy_worker.finished.connect(self._on_energy_measurement_finished)

        # Start thread
        self._energy_thread.start()

    def _on_energy_progress(self, current, total, message):
        """Handle energy measurement progress."""
        self.logger.debug(f"Energy progress: {current}/{total} - {message}")

    def _on_energy_result(self, energy_data):
        """Handle energy measurement results."""
        self.logger.info(
            f"Energy analysis complete: mean={energy_data.get('mean_energy_mj', 0):.2f} mJ, "
            f"power={energy_data.get('mean_power_watts', 0):.2f} W"
        )

        # Update the energy page with data
        try:
            self.energy_page.set_data(energy_data)
            self.logger.debug("Energy page updated with analysis results")
        except Exception as e:
            self.logger.error("Error updating energy page: %s", e, exc_info=True)

        # Show warning if measurements were too short
        if energy_data.get('has_warning', False):
            zero_count = energy_data.get('zero_readings_count', 0)
            n_images = energy_data.get('n_images', 0)
            QMessageBox.warning(
                None,
                "Measurement Warning",
                f"{zero_count} of {n_images} measurements returned 0 energy.\n\n"
                "The inference time is too short for the energy counter resolution.\n\n"
                "Recommendation: Increase 'Measurement runs' parameter to accumulate "
                "more energy per measurement (e.g., 50-100 runs)."
            )

        if self.status_manager:
            self.status_manager.finish_task()

    def _on_energy_error(self, error):
        """Handle energy measurement error."""
        self.logger.error(f"Energy analysis error: {error}")
        if self.status_manager:
            self.status_manager.error_task(str(error)[:50])
        QMessageBox.critical(None, "Energy Analysis Error", str(error))
        self._cleanup_energy_worker()

    def _on_energy_measurement_finished(self):
        """Handle energy measurement completion."""
        self.logger.debug("Energy measurement finished")
        self._cleanup_energy_worker()

    def get_content_widget(self):
        """Return the main container widget for stepper integration."""
        return self.ui.reports_main_container
