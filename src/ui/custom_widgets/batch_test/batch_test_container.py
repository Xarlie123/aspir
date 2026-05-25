"""
Main container widget for Batch Test mode.
Provides UI for configuring and running multiple test configurations.
"""
import os
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QPushButton,
    QLabel, QProgressBar, QFileDialog, QMessageBox, QGroupBox,
    QSizePolicy, QFrame, QComboBox, QLineEdit,
    QRadioButton, QSpinBox, QButtonGroup, QCheckBox,
)
from PySide6.QtCore import Signal, Qt, QTimer

from ui.custom_widgets.batch_test.test_config_model import (
    TestConfiguration, BatchTestConfig, TestStatus, ExportLevel
)
from ui.custom_widgets.batch_test.test_list_widget import TestListWidget
from ui.custom_widgets.batch_test.test_config_widget import TestConfigWidget
from ui.custom_widgets.batch_test.batch_test_runner import BatchTestRunner
from ui.custom_widgets.common.button_styles import (
    BUTTON_STYLE_GREEN, BUTTON_STYLE_BLUE, BUTTON_STYLE_RED, apply_button_style
)
from ui.custom_widgets.common.multi_phase_progress import MultiPhaseProgressWidget
from ui.custom_widgets.batch_test.resource_monitor_widget import ResourceMonitorWidget


class BatchTestContainer(QWidget):
    """
    Main container for Batch Test mode.

    Allows users to:
    - Create/edit multiple test configurations
    - Save/load batch configurations (.batch_config)
    - Run all tests sequentially
    - Monitor progress with status indicators
    - Cancel individual tests or all tests
    - Export results to CSV
    """

    # Signals
    run_requested = Signal(object)  # Emits BatchTestConfig
    cancel_all_requested = Signal()
    cancel_test_requested = Signal(int)  # Test index
    dataset_load_requested = Signal(dict, bool)  # Emits (dataset_info, should_generate) when loading config
    batch_report_available = Signal(str)  # Emits path to last completed batch report

    def __init__(self, simulation=None, logger=None, parent=None):
        super().__init__(parent)

        if logger:
            self.logger = logger.getChild("BatchTestContainer")
        else:
            self.logger = logging.getLogger("BatchTestContainer")

        self.simulation = simulation
        self._batch_config = BatchTestConfig()
        self._is_running = False
        self._runner: Optional[BatchTestRunner] = None
        self._last_report_path: Optional[str] = None  # Path to last executed batch report
        self._suppress_img_size_reset = False  # Flag to prevent img_size reset during config loading

        self._setup_ui()
        self._connect_signals()

        # Add initial test
        self._on_add_test()

        self.logger.debug("BatchTestContainer initialized")

    def _setup_ui(self):
        """Setup the UI layout (single page, no stepper)."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Single page layout - configure tests directly
        self._setup_configure_page(main_layout)

    def _setup_configure_page(self, layout: QVBoxLayout):
        """Setup the test configuration UI directly in the given layout."""

        # Title
        title = QLabel("Batch Test Configuration")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #333;")
        layout.addWidget(title)

        # Description
        desc = QLabel("Configure multiple tests to run sequentially. Each test can have different mask, reconstruction, and DNN settings.")
        desc.setStyleSheet("color: #666; margin-bottom: 5px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Batch name input row
        name_layout = QHBoxLayout()
        name_layout.setSpacing(10)

        name_label = QLabel("Export Name:")
        name_label.setStyleSheet("font-weight: bold; color: #333;")
        name_layout.addWidget(name_label)

        self.batch_name_edit = QLineEdit()
        self.batch_name_edit.setPlaceholderText("Name for the results folder (e.g., 'scatter_comparison_results')")
        self.batch_name_edit.setText(f"batch_{datetime.now().strftime('%Y%m%d')}")
        self.batch_name_edit.setToolTip(
            "Name used to create the output folder.\n"
            "If a folder with this name exists, a suffix (_1, _2, ...) will be added."
        )
        self.batch_name_edit.setMinimumWidth(300)
        name_layout.addWidget(self.batch_name_edit, 1)

        layout.addLayout(name_layout)

        # Dataset notice banner
        self.dataset_notice = QLabel(
            "⚠️ A dataset must be loaded in <b>Single Test</b> mode before running batch tests."
        )
        self.dataset_notice.setStyleSheet("""
            QLabel {
                background-color: #fff3cd;
                border: 1px solid #ffc107;
                border-radius: 4px;
                padding: 8px 12px;
                color: #856404;
            }
        """)
        self.dataset_notice.setWordWrap(True)
        layout.addWidget(self.dataset_notice)

        # Execution options row
        exec_layout = QHBoxLayout()
        exec_layout.setSpacing(15)

        exec_label = QLabel("Execution Mode:")
        exec_label.setStyleSheet("font-weight: bold; color: #333;")
        exec_layout.addWidget(exec_label)

        # Radio buttons for sequential/parallel
        self.exec_mode_group = QButtonGroup(self)

        self.sequential_radio = QRadioButton("Sequential")
        self.sequential_radio.setToolTip("Run tests one after another")
        self.sequential_radio.setChecked(True)
        self.exec_mode_group.addButton(self.sequential_radio, 0)
        exec_layout.addWidget(self.sequential_radio)

        self.parallel_radio = QRadioButton("Parallel")
        self.parallel_radio.setToolTip(
            "Run multiple tests simultaneously.\n"
            "Note: Timing and energy reports always run sequentially for accuracy."
        )
        self.exec_mode_group.addButton(self.parallel_radio, 1)
        exec_layout.addWidget(self.parallel_radio)

        # Thread count spinbox with max threads detection
        import multiprocessing
        max_threads = multiprocessing.cpu_count()

        self.threads_label = QLabel("Threads:")
        self.threads_label.setStyleSheet("color: #666;")
        self.threads_label.setEnabled(False)
        exec_layout.addWidget(self.threads_label)

        self.threads_spin = QSpinBox()
        self.threads_spin.setRange(2, max_threads)
        self.threads_spin.setValue(min(2, max_threads))
        self.threads_spin.setToolTip("Number of tests to run in parallel")
        self.threads_spin.setEnabled(False)
        self.threads_spin.setFixedWidth(60)
        exec_layout.addWidget(self.threads_spin)

        # Max threads info label
        self.max_threads_label = QLabel(f"(max: {max_threads})")
        self.max_threads_label.setStyleSheet("color: #999; font-size: 11px;")
        self.max_threads_label.setEnabled(False)
        exec_layout.addWidget(self.max_threads_label)

        # Resource monitor (CPU/RAM/GPU usage) - updates every 2s when idle, 500ms when running
        self.resource_monitor = ResourceMonitorWidget(
            update_interval_ms=2000,
            logger=self.logger
        )
        exec_layout.addWidget(self.resource_monitor)

        # Connect radio buttons to enable/disable thread spinbox
        self.parallel_radio.toggled.connect(self._on_execution_mode_changed)

        exec_layout.addStretch()

        layout.addLayout(exec_layout)

        # Idle-baseline row — captures system idle power before the
        # first test so each test row can be reported with a "dynamic
        # power" column (= total − baseline). Disabled → totals only,
        # dynamic columns blank in the report.
        baseline_layout = QHBoxLayout()
        baseline_layout.setSpacing(10)

        self.baseline_check = QCheckBox("Capture idle baseline")
        self.baseline_check.setChecked(True)
        self.baseline_check.setToolTip(
            "Sample the energy backend's instantaneous power for the\n"
            "configured duration before the first test starts. The\n"
            "mean is subtracted from each test's average power to\n"
            "derive the dynamic-power / dynamic-energy / dynamic-\n"
            "efficiency columns in the batch report."
        )
        baseline_layout.addWidget(self.baseline_check)

        baseline_label = QLabel("Idle baseline duration:")
        baseline_label.setStyleSheet("color: #666;")
        baseline_layout.addWidget(baseline_label)

        self.baseline_spin = QSpinBox()
        self.baseline_spin.setRange(30, 300)
        self.baseline_spin.setValue(60)
        self.baseline_spin.setSuffix(" s")
        self.baseline_spin.setFixedWidth(72)
        self.baseline_spin.setToolTip(
            "Seconds of idle sampling at the start of the batch."
        )
        baseline_layout.addWidget(self.baseline_spin)

        self.baseline_check.toggled.connect(
            lambda on: self.baseline_spin.setEnabled(on)
        )

        baseline_layout.addStretch()
        layout.addLayout(baseline_layout)

        # Main content splitter
        splitter = QSplitter(Qt.Horizontal)

        # Left panel: Test list
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.test_list = TestListWidget(logger=self.logger)
        left_layout.addWidget(self.test_list)

        left_panel.setMinimumWidth(200)
        left_panel.setMaximumWidth(300)
        splitter.addWidget(left_panel)

        # Right panel: Test configuration
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.test_config = TestConfigWidget(logger=self.logger)
        right_layout.addWidget(self.test_config)

        splitter.addWidget(right_panel)

        # Set initial sizes
        splitter.setSizes([250, 550])

        layout.addWidget(splitter, 1)

        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setStyleSheet("background-color: #ddd;")
        layout.addWidget(separator)

        # Bottom section: Actions and progress
        bottom_layout = QVBoxLayout()
        bottom_layout.setSpacing(10)

        # Action buttons row
        actions_layout = QHBoxLayout()
        actions_layout.setSpacing(10)

        # Save/Load buttons
        self.save_btn = QPushButton("💾 Save Batch Config")
        self.save_btn.clicked.connect(self._on_save_config)
        self.save_btn.setToolTip("Save batch test configuration to .batch_config file")
        apply_button_style(self.save_btn, BUTTON_STYLE_BLUE)
        actions_layout.addWidget(self.save_btn)

        self.load_btn = QPushButton("📂 Load Batch Config")
        self.load_btn.clicked.connect(self._on_load_config)
        self.load_btn.setToolTip("Load batch test configuration from .batch_config file")
        apply_button_style(self.load_btn, BUTTON_STYLE_BLUE)
        actions_layout.addWidget(self.load_btn)

        actions_layout.addStretch()

        # Cancel all button (only visible during execution)
        self.cancel_all_btn = QPushButton("Cancel All")
        self.cancel_all_btn.clicked.connect(self._on_cancel_all)
        self.cancel_all_btn.setToolTip("Cancel all remaining tests")
        apply_button_style(self.cancel_all_btn, BUTTON_STYLE_RED)
        self.cancel_all_btn.hide()  # Hidden by default, shown only when running
        actions_layout.addWidget(self.cancel_all_btn)

        # Run button
        self.run_btn = QPushButton("▶ Run All Tests")
        self.run_btn.clicked.connect(self._on_run_all)
        self.run_btn.setMinimumWidth(150)
        self.run_btn.setToolTip("Start running all configured tests")
        apply_button_style(self.run_btn, BUTTON_STYLE_GREEN)
        actions_layout.addWidget(self.run_btn)

        bottom_layout.addLayout(actions_layout)

        # Progress section
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        progress_layout.setSpacing(8)

        # Overall batch progress
        overall_layout = QHBoxLayout()
        overall_label = QLabel("Overall:")
        overall_label.setFixedWidth(60)
        overall_label.setStyleSheet("font-weight: bold;")
        overall_layout.addWidget(overall_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%v% (%p% complete)")
        self.progress_bar.setFixedHeight(22)
        overall_layout.addWidget(self.progress_bar)
        progress_layout.addLayout(overall_layout)

        # Current test label
        self.current_test_label = QLabel("")
        self.current_test_label.setStyleSheet("color: #2196F3; font-weight: bold; font-size: 12px;")
        self.current_test_label.hide()
        progress_layout.addWidget(self.current_test_label)

        # Multi-phase progress for current test
        self.phase_progress = MultiPhaseProgressWidget(
            phases=[
                BatchTestRunner.PHASE_MASKS,
                BatchTestRunner.PHASE_RECONSTRUCTION,
                BatchTestRunner.PHASE_MODEL_SETUP,
                BatchTestRunner.PHASE_TRAINING,
                BatchTestRunner.PHASE_ANALYSIS,
                BatchTestRunner.PHASE_EXPORT,
            ],
            title="Current Test Progress"
        )
        self.phase_progress.hide()  # Hidden until tests start running
        progress_layout.addWidget(self.phase_progress)

        # Status label
        self.status_label = QLabel("Ready - Configure tests and click 'Run All Tests' to begin")
        self.status_label.setStyleSheet("color: #666;")
        progress_layout.addWidget(self.status_label)

        bottom_layout.addWidget(progress_group)

        layout.addLayout(bottom_layout)

    def _connect_signals(self):
        """Connect internal signals."""
        # Test list signals
        self.test_list.test_selected.connect(self._on_test_selected)
        self.test_list.test_added.connect(self._on_add_test)
        self.test_list.test_removed.connect(self._on_remove_test)
        self.test_list.test_duplicated.connect(self._on_duplicate_test)
        self.test_list.tests_reordered.connect(self._on_reorder_test)
        self.test_list.cancel_test_requested.connect(self._on_cancel_test)

        # Config widget signals
        self.test_config.config_changed.connect(self._on_config_changed)

    def _on_test_selected(self, index: int):
        """Handle test selection."""
        if 0 <= index < len(self._batch_config.tests):
            t = self._batch_config.tests[index]
            self.logger.debug("[DEBUG] _on_test_selected[%d]: name='%s', mask='%s', scatter_patterns=%s, sweep_bws=%s, epochs=%s",
                             index, t.name, t.mask_type, t.scatter_num_patterns, t.sweep_bar_widths, t.epochs)
            self.test_config.set_config(t)
        else:
            self.test_config.set_config(None)

    def _on_execution_mode_changed(self, is_parallel: bool):
        """Handle execution mode radio button change."""
        self.threads_label.setEnabled(is_parallel)
        self.threads_spin.setEnabled(is_parallel)
        self.max_threads_label.setEnabled(is_parallel)
        if is_parallel:
            self.threads_label.setStyleSheet("color: #333; font-weight: 500;")
            self.max_threads_label.setStyleSheet("color: #666; font-size: 11px;")
        else:
            self.threads_label.setStyleSheet("color: #999;")
            self.max_threads_label.setStyleSheet("color: #999; font-size: 11px;")

    def _on_add_test(self):
        """Add a new test."""
        config = self._batch_config.add_test()
        self.test_list.set_tests(self._batch_config.tests)
        self.test_list.select_test(len(self._batch_config.tests) - 1)
        self.logger.debug("Added test: %s", config.name)

    def _on_remove_test(self, index: int):
        """Remove a test."""
        if self._batch_config.remove_test(index):
            self.test_list.set_tests(self._batch_config.tests)
            # Select previous or first
            new_index = min(index, len(self._batch_config.tests) - 1)
            if new_index >= 0:
                self.test_list.select_test(new_index)
            else:
                self.test_config.set_config(None)
            self.logger.debug("Removed test at index %d", index)

    def _on_duplicate_test(self, index: int):
        """Duplicate a test."""
        src = self._batch_config.tests[index] if 0 <= index < len(self._batch_config.tests) else None
        if src:
            self.logger.debug("[DEBUG] Duplicate source[%d]: name='%s', mask='%s', scatter_patterns=%s, sweep_bws=%s, epochs=%s",
                             index, src.name, src.mask_type, src.scatter_num_patterns, src.sweep_bar_widths, src.epochs)
        new_config = self._batch_config.duplicate_test(index)
        if new_config:
            self.logger.debug("[DEBUG] Duplicate result: name='%s', mask='%s', scatter_patterns=%s, sweep_bws=%s, epochs=%s",
                             new_config.name, new_config.mask_type, new_config.scatter_num_patterns, new_config.sweep_bar_widths, new_config.epochs)
            self.test_list.set_tests(self._batch_config.tests)
            self.test_list.select_test(index + 1)
            self.logger.debug("Duplicated test: %s", new_config.name)

    def _on_reorder_test(self, from_index: int, to_index: int):
        """Handle test reordering (from drag-and-drop or context menu)."""
        if self._batch_config.move_test(from_index, to_index):
            self.test_list.set_tests(self._batch_config.tests)
            self.test_list.select_test(to_index)
            self.logger.debug("Moved test from %d to %d", from_index, to_index)

    def _on_config_changed(self):
        """Handle configuration change."""
        # Update list item display (name might have changed)
        index = self.test_list.get_selected_index()
        if index >= 0:
            self.test_list.update_test_status(index)

    def _on_save_config(self):
        """Save batch configuration to JSON file, including dataset info."""
        from ui.utils.file_formats import FileExtensions

        default_dir = BatchTestConfig.get_default_directory()
        default_name = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}{FileExtensions.BATCH_CONFIG}"

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Batch Test Configuration",
            str(default_dir / default_name),
            BatchTestConfig.get_file_filter()
        )

        if filepath:
            # Collect dataset info if available
            dataset_info = None
            if self.simulation and self.simulation.dataset:
                ds = self.simulation.dataset
                dataset_info = {
                    "type": ds.__class__.__name__,
                    "name": getattr(ds, 'name', 'unknown'),
                    "img_size": getattr(ds, 'img_size', 0),
                    "num_images": len(ds.data) if hasattr(ds, 'data') and ds.data else 0,
                }
                # Add source path if available
                if hasattr(ds, 'img_path'):
                    dataset_info["source_path"] = ds.img_path
                elif hasattr(ds, 'folder_path'):
                    dataset_info["source_path"] = ds.folder_path
                elif hasattr(ds, 'dataset_path'):
                    dataset_info["source_path"] = ds.dataset_path

                # Generator-specific parameters: persist enough so the Single
                # Test dataset panel can be re-populated exactly on reload.
                if hasattr(ds, 'seed'):
                    dataset_info["seed"] = ds.seed
                if hasattr(ds, 'data_format') and ds.data_format is not None:
                    dataset_info["data_format"] = ds.data_format
                if ds.__class__.__name__ == "DatasetFromIRBeam":
                    dataset_info["mode_distribution"] = dict(getattr(ds, 'mode_distribution', {}))
                    dataset_info["speckle_noise"] = float(getattr(ds, 'speckle_noise', 0.0))
                    dataset_info["max_mode_order"] = int(getattr(ds, 'max_mode_order', 3))

            if self._batch_config.save(filepath, dataset_info=dataset_info):
                self.logger.info("Saved batch config to: %s", filepath)
                msg = f"Configuration saved to:\n{filepath}"
                if dataset_info:
                    msg += f"\n\nDataset: {dataset_info['name']} ({dataset_info['num_images']} images)"
                QMessageBox.information(self, "Saved", msg)
            else:
                QMessageBox.critical(self, "Error", "Failed to save configuration")

    def _on_load_config(self):
        """Load batch configuration from JSON or legacy YAML file."""
        default_dir = BatchTestConfig.get_default_directory()

        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Batch Test Configuration",
            str(default_dir),
            BatchTestConfig.get_file_filter()
        )

        if filepath:
            config, dataset_info = BatchTestConfig.load_with_dataset_info(filepath)
            if config:
                self._batch_config = config

                # Set the image size from dataset_info BEFORE setting config values
                # This ensures spinbox/slider maximums are correct (img_size² patterns)
                if dataset_info:
                    saved_img_size = dataset_info.get('img_size')
                    if saved_img_size and isinstance(saved_img_size, int):
                        self.logger.info("Setting img_size from config: %d", saved_img_size)
                        self.test_config.set_img_size(saved_img_size)

                self.test_list.set_tests(self._batch_config.tests)
                if self._batch_config.tests:
                    self.test_list.select_test(0)
                self.logger.info("Loaded batch config from: %s", filepath)

                # Set batch name based on loaded config filename
                config_filename = Path(filepath).stem  # e.g., "beam_had_unet" from "beam_had_unet.batch_config"
                self.batch_name_edit.setText(f"{config_filename}_results")

                # Build message with dataset info
                msg = f"Loaded {len(self._batch_config.tests)} test configurations."

                if dataset_info:
                    ds_name = dataset_info.get('name', 'unknown')
                    ds_type = dataset_info.get('type', 'unknown')
                    ds_size = dataset_info.get('img_size', '?')
                    ds_num = dataset_info.get('num_images', '?')
                    ds_path = dataset_info.get('source_path', '')

                    # Show dialog asking what to do with dataset info
                    dataset_msg = f"This batch config was created with:\n\n"
                    dataset_msg += f"  • Dataset: {ds_name}\n"
                    dataset_msg += f"  • Type: {ds_type}\n"
                    dataset_msg += f"  • Size: {ds_size}×{ds_size}, {ds_num} images"
                    if ds_path:
                        dataset_msg += f"\n  • Source: {ds_path}"
                    if dataset_info.get("seed") is not None:
                        dataset_msg += f"\n  • Seed: {dataset_info['seed']}"
                    if "data_format" in dataset_info:
                        dataset_msg += f"\n  • Data format: {dataset_info['data_format']}"
                    if "mode_distribution" in dataset_info:
                        md = dataset_info["mode_distribution"]
                        mode_str = ", ".join(f"{k}={v:.0f}%" for k, v in md.items())
                        dataset_msg += f"\n  • Modes: {mode_str}"
                    if "speckle_noise" in dataset_info:
                        dataset_msg += f"\n  • Speckle noise: {dataset_info['speckle_noise']:.2f}"
                    if "max_mode_order" in dataset_info:
                        dataset_msg += f"\n  • Max mode order: {dataset_info['max_mode_order']}"

                    dataset_msg += "\n\nDo you want to load/generate this dataset now?"
                    dataset_msg += "\n\n• Yes: Switch to Single Test and generate the dataset"
                    dataset_msg += "\n• No: Just configure the widgets (you can generate later)"

                    reply = QMessageBox.question(
                        self, "Load Dataset?",
                        dataset_msg,
                        QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                        QMessageBox.Yes
                    )

                    if reply == QMessageBox.Cancel:
                        # User cancelled, still show the loaded config
                        QMessageBox.information(self, "Batch Config Loaded", msg)
                        return

                    # Emit signal with dataset info and whether to generate
                    should_generate = (reply == QMessageBox.Yes)

                    if should_generate:
                        # Suppress img_size reset to preserve loaded config values
                        self._suppress_img_size_reset = True

                    self.dataset_load_requested.emit(dataset_info, should_generate)

                    if should_generate:
                        msg += "\n\nSwitching to Single Test mode to generate dataset..."
                    else:
                        msg += "\n\nDataset widgets configured. Generate when ready."
                else:
                    # No dataset info in config
                    msg += "\n\n⚠️ No dataset info saved in this config."
                    if not (self.simulation and self.simulation.dataset):
                        msg += "\nPlease load a dataset in Single Test mode before running."

                QMessageBox.information(self, "Batch Config Loaded", msg)
            else:
                QMessageBox.critical(self, "Error", "Failed to load configuration")

    def _on_run_all(self):
        """Start running all tests."""
        if not self._batch_config.tests:
            QMessageBox.warning(self, "No Tests", "Please add at least one test configuration.")
            return

        # Check if dataset is loaded
        if self.simulation and (self.simulation.dataset is None or not self.simulation.dataset.data):
            QMessageBox.warning(
                self, "No Dataset",
                "Please load a dataset first in Single Test mode before running batch tests."
            )
            return

        self._set_running_state(True)
        self._batch_config.reset_all_status()
        self.test_list.set_tests(self._batch_config.tests)

        self.status_label.setText("Starting batch tests...")
        self.progress_bar.setValue(0)

        # Update batch config with execution settings
        self._batch_config.parallel_execution = self.parallel_radio.isChecked()
        self._batch_config.parallel_threads = self.threads_spin.value()
        self._batch_config.capture_baseline = self.baseline_check.isChecked()
        self._batch_config.baseline_duration_s = int(self.baseline_spin.value())

        # Get the selected export level and batch name
        export_level = self.get_export_level()
        batch_name = self.get_batch_name()
        exec_mode = "parallel" if self._batch_config.parallel_execution else "sequential"
        self.logger.info(
            "Export level: %s, batch name: %s, execution: %s (threads: %d)",
            export_level.name, batch_name, exec_mode, self._batch_config.parallel_threads
        )

        # Create and start the runner
        self._runner = BatchTestRunner(
            batch_config=self._batch_config,
            dataset=self.simulation.dataset,
            export_level=export_level,
            batch_name=batch_name,
            logger=self.logger,
            parent=self
        )

        # Connect runner signals
        self._runner.test_started.connect(self._on_runner_test_started)
        self._runner.test_progress.connect(self._on_runner_test_progress)
        self._runner.test_completed.connect(self._on_runner_test_completed)
        self._runner.test_failed.connect(self._on_runner_test_failed)
        self._runner.test_cancelled.connect(self._on_runner_test_cancelled)
        self._runner.batch_completed.connect(self._on_runner_batch_completed)
        self._runner.batch_cancelled.connect(self._on_runner_batch_cancelled)
        self._runner.status_update.connect(self._on_runner_status_update)

        # Connect phase-specific signals
        self._runner.phase_started.connect(self._on_phase_started)
        self._runner.phase_progress.connect(self._on_phase_progress)
        self._runner.phase_completed.connect(self._on_phase_completed)

        self._runner.start()
        self.logger.info("Batch test run started with %d tests", len(self._batch_config.tests))

    def _on_cancel_all(self):
        """Cancel all remaining tests."""
        reply = QMessageBox.question(
            self, "Cancel All Tests",
            "Are you sure you want to cancel all remaining tests?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            if self._runner:
                self._runner.cancel_all()
            self.logger.info("Cancel all tests requested")

    def _on_cancel_test(self, index: int):
        """Cancel a specific test."""
        if self._runner:
            self._runner.cancel_test(index)
        self.logger.info("Cancel requested for test %d", index)

    # Runner signal handlers

    def _on_runner_test_started(self, index: int):
        """Handle test started from runner."""
        self.test_list.update_test_status(index)

        # Reset phase progress for new test
        self.phase_progress.reset_all()

        # Update current test label
        if 0 <= index < len(self._batch_config.tests):
            test_name = self._batch_config.tests[index].name
            total = len(self._batch_config.tests)
            self.current_test_label.setText(f"Running test {index + 1}/{total}: {test_name}")
            self.current_test_label.show()

    def _on_runner_test_progress(self, index: int, progress: int):
        """Handle test progress update from runner."""
        if 0 <= index < len(self._batch_config.tests):
            self._batch_config.tests[index].progress = progress
            self.test_list.update_test_status(index)

    def _on_runner_test_completed(self, index: int, results: dict):
        """Handle test completion from runner."""
        # Reset per-test phase progress
        self.test_list.reset_phase_progress(index)
        self.test_list.update_test_status(index)
        self._update_overall_progress()

        # Mark all phases as completed for visual feedback
        self.phase_progress.set_all_completed()

    def _on_runner_test_failed(self, index: int, error_msg: str):
        """Handle test failure from runner."""
        # Reset per-test phase progress
        self.test_list.reset_phase_progress(index)
        self.test_list.update_test_status(index)
        self._update_overall_progress()
        self.status_label.setText(f"Failed: {self._batch_config.tests[index].name}")

    def _on_runner_test_cancelled(self, index: int):
        """Handle test cancellation from runner."""
        # Reset per-test phase progress
        self.test_list.reset_phase_progress(index)
        self.test_list.update_test_status(index)
        self._update_overall_progress()

    def _on_runner_batch_completed(self, results_path: str):
        """Handle batch completion from runner."""
        self.on_batch_completed(results_path)

    def _on_runner_batch_cancelled(self):
        """Handle batch cancellation from runner."""
        self.on_batch_cancelled()

    def _on_runner_status_update(self, message: str):
        """Handle status update from runner."""
        self.status_label.setText(message)

    # Phase-specific signal handlers

    def _on_phase_started(self, index: int, phase_name: str):
        """Handle phase started signal."""
        # Update batch-level progress for export phase or fallback
        if index < 0:
            self.phase_progress.start_phase(phase_name)
        else:
            # Update test item progress
            self.test_list.start_phase(index, phase_name)
            # Also update batch-level for visibility
            self.phase_progress.start_phase(phase_name)

    def _on_phase_progress(self, index: int, phase_name: str, progress: int):
        """Handle phase progress signal."""
        if index < 0:
            self.phase_progress.update_phase_progress(phase_name, progress)
        else:
            # Update test item progress
            self.test_list.update_phase_progress(index, phase_name, progress)
            # Also update batch-level for visibility
            self.phase_progress.update_phase_progress(phase_name, progress)

    def _on_phase_completed(self, index: int, phase_name: str):
        """Handle phase completed signal."""
        if index < 0:
            self.phase_progress.complete_phase(phase_name)
        else:
            # Update test item progress
            self.test_list.complete_phase(index, phase_name)
            # Also update batch-level for visibility
            self.phase_progress.complete_phase(phase_name)

    def _update_overall_progress(self):
        """Update the overall progress bar."""
        completed = sum(1 for t in self._batch_config.tests
                       if t.status in (TestStatus.COMPLETED, TestStatus.FAILED, TestStatus.CANCELLED))
        total = len(self._batch_config.tests)
        progress = int((completed / total) * 100) if total > 0 else 0
        self.progress_bar.setValue(progress)

    def _set_running_state(self, running: bool):
        """Set UI state for running/idle."""
        self._is_running = running

        # Disable editing during run
        self.test_list.set_read_only(running)
        self.test_config.set_read_only(running)
        self.batch_name_edit.setReadOnly(running)

        # Update buttons
        self.save_btn.setEnabled(not running)
        self.load_btn.setEnabled(not running)
        self.run_btn.setEnabled(not running)

        # Show/hide cancel button and phase progress based on running state
        if running:
            self.cancel_all_btn.show()
            self.phase_progress.show()
            self.phase_progress.reset_all()
            # Faster resource monitoring during execution
            self.resource_monitor.set_update_interval(500)
        else:
            self.cancel_all_btn.hide()
            self.phase_progress.hide()
            self.current_test_label.hide()
            # Slower resource monitoring when idle
            self.resource_monitor.set_update_interval(2000)

    # Public methods for external control (by runner/executor)

    def update_test_status(self, index: int, status: TestStatus, progress: int = 0, message: str = ""):
        """Update the status of a specific test."""
        if 0 <= index < len(self._batch_config.tests):
            test = self._batch_config.tests[index]
            test.status = status
            test.progress = progress
            if message:
                test.error_message = message

            self.test_list.update_test_status(index)

            # Update overall progress
            completed = self._batch_config.get_completed_count()
            total = len(self._batch_config.tests)
            overall_progress = int((completed / total) * 100) if total > 0 else 0
            self.progress_bar.setValue(overall_progress)

            # Update status label
            if status == TestStatus.RUNNING:
                self.status_label.setText(f"Running: {test.name} ({progress}%)")
            elif status == TestStatus.COMPLETED:
                self.status_label.setText(f"Completed: {test.name} - {completed}/{total} tests done")
            elif status == TestStatus.FAILED:
                self.status_label.setText(f"Failed: {test.name} - {message}")
            elif status == TestStatus.CANCELLED:
                self.status_label.setText(f"Cancelled: {test.name}")

    def on_batch_completed(self, results_path: str = ""):
        """Called when all tests are completed."""
        self._set_running_state(False)

        completed = self._batch_config.get_completed_count()
        total = len(self._batch_config.tests)

        self.progress_bar.setValue(100)
        self.status_label.setText(f"Batch completed: {completed}/{total} tests successful")

        if results_path:
            # Store the path for "Load Last Session" feature
            self._last_report_path = results_path
            self.logger.info(f"Batch report path stored: {results_path}")
            self.batch_report_available.emit(results_path)

            QMessageBox.information(
                self, "Batch Complete",
                f"Batch testing completed!\n\n"
                f"Completed: {completed}/{total} tests\n"
                f"Results saved to: {results_path}"
            )
        else:
            QMessageBox.information(
                self, "Batch Complete",
                f"Batch testing completed!\n\nCompleted: {completed}/{total} tests"
            )

    def on_batch_cancelled(self):
        """Called when batch is cancelled."""
        self._set_running_state(False)
        self.status_label.setText("Batch cancelled")

    def get_batch_config(self) -> BatchTestConfig:
        """Get the current batch configuration."""
        return self._batch_config

    def get_last_report_path(self) -> Optional[str]:
        """Get the path to the last executed batch report, if any."""
        return self._last_report_path

    def has_last_report(self) -> bool:
        """Check if there's a last report available from this session."""
        return self._last_report_path is not None

    def should_suppress_img_size_reset(self) -> bool:
        """Check if img_size reset should be suppressed (during config loading)."""
        return self._suppress_img_size_reset

    def clear_suppress_img_size_reset(self):
        """Clear the suppress flag after config loading is complete."""
        self._suppress_img_size_reset = False

    def get_export_level(self) -> ExportLevel:
        """
        Determine export level from test configurations.

        Returns ALL_DATA if any test has include_datasets=True,
        otherwise REPORTS_AND_MODELS (always export trained models).
        """
        # Check if any test wants datasets included
        for test in self._batch_config.tests:
            if test.include_datasets:
                return ExportLevel.ALL_DATA
        return ExportLevel.REPORTS_AND_MODELS

    def get_batch_name(self) -> str:
        """Get the batch test name, sanitized for use as folder name."""
        name = self.batch_name_edit.text().strip()
        if not name:
            name = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        # Sanitize: replace invalid characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            name = name.replace(char, '_')
        return name

    def update_dataset_notice(self):
        """Update the dataset notice visibility based on whether a dataset is loaded."""
        has_dataset = (
            self.simulation is not None and
            self.simulation.dataset is not None and
            self.simulation.dataset.data is not None and
            len(self.simulation.dataset.data) > 0
        )

        if has_dataset:
            # Show success notice
            self.dataset_notice.setText(
                f"✅ Dataset loaded: {len(self.simulation.dataset.data)} images "
                f"({self.simulation.dataset.img_size}x{self.simulation.dataset.img_size})"
            )
            self.dataset_notice.setStyleSheet("""
                QLabel {
                    background-color: #d4edda;
                    border: 1px solid #28a745;
                    border-radius: 4px;
                    padding: 8px 12px;
                    color: #155724;
                }
            """)
        else:
            # Show warning notice
            self.dataset_notice.setText(
                "⚠️ A dataset must be loaded in <b>Single Test</b> mode before running batch tests."
            )
            self.dataset_notice.setStyleSheet("""
                QLabel {
                    background-color: #fff3cd;
                    border: 1px solid #ffc107;
                    border-radius: 4px;
                    padding: 8px 12px;
                    color: #856404;
                }
            """)
