import logging
from ui.utils.colored_formatter import ColoredFormatter
from ui.utils.status_manager import StatusManager
from ui.utils.log_manager import get_log_manager
from ui.utils.log_settings_dialog import LogSettingsDialog
from ui.utils.log_viewer_dialog import LogViewerDialog
from ui.utils.external_apps_settings import (
    ExternalAppsSettingsDialog, get_external_apps_manager
)
from PyQt5.QtWidgets import (
    QMainWindow,
    QFileDialog,
    QMessageBox,
    QVBoxLayout,
    QHBoxLayout,
    QSizePolicy,
    QLabel,
    QWidget,
    QStackedWidget,
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QColor, QPainter, QBrush, QIcon, QPixmap
import os
import json
import time
import numpy as np

from ui.ui_main_window import Ui_MainWindow
from ui._1_dataset.ui_dataset_handler import UIDatasetHandler
from ui._2_masks.ui_mask_handler import UIMaskHandler
from ui._3_test_masks.ui_test_mask_handler import UITestMascaraHandler
from ui._4_postprocessor.ui_postprocessor_handler import UIPostprocessorHandler
from ui._5_reports.ui_reports_handler import UIReportsHandler
from ui._7_pipeline.ui_pipeline_handler import UIPipelineHandler
from ui.utils.config_yaml_handler import ConfigYamlHandler
from ui.utils.file_formats import (
    FileExtensions, SingleTestExperiment, SINGLE_TESTS_DIR
)
from simulation_engine.simulation import Simulacion

from ui.custom_widgets.mode_selector import ModeSelectorWidget
from ui.modes import SingleTestContainer
from ui.custom_widgets.batch_test import BatchTestContainer
from ui.custom_widgets.batch_reports import BatchReportsContainer


class StatusLED(QWidget):
    """A simple LED indicator widget that displays a colored circle."""

    def __init__(self, parent=None, size=16):
        super().__init__(parent)
        self._color = QColor("#00cc00")  # Green by default (ready)
        self._size = size
        self.setFixedSize(size, size)

    def set_color(self, color: str):
        """Set the LED color using a hex string (e.g., '#cc0000' for red)."""
        self._color = QColor(color)
        self.update()

    def set_ready(self):
        """Set LED to green (ready state)."""
        self.set_color("#00cc00")

    def set_busy(self):
        """Set LED to red (busy state)."""
        self.set_color("#cc0000")

    def set_error(self):
        """Set LED to orange (error state)."""
        self.set_color("#ff8800")

    def paintEvent(self, event):
        """Paint the LED as a filled circle."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QBrush(self._color))
        painter.setPen(Qt.NoPen)
        # Draw circle with small margin
        margin = 2
        painter.drawEllipse(margin, margin, self._size - 2 * margin, self._size - 2 * margin)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Initialize logging system using LogManager
        self.log_manager = get_log_manager()
        self.logger = self.log_manager.setup_logging("SPIm")
        self.logger.debug("Initializing MainWindow")

        # UI + simulation
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Set window icon
        self._assets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'assets')
        icon_path = os.path.join(self._assets_dir, 'icon_app.png')
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
            self.logger.debug("Window icon set from: %s", icon_path)

        self.simulation = Simulacion(logger=self.logger)

        # Status manager for task state
        self.status_manager = StatusManager(logger=self.logger)
        self._setup_status_bar()

        # Handlers (pass status_manager to each for task state management)
        self.ui_mask_handler = UIMaskHandler(
            self.ui, self.simulation, logger=self.logger,
            status_manager=self.status_manager
        )
        self.ui_test_mask_handler = UITestMascaraHandler(
            self.ui, self.simulation, self.ui_mask_handler, logger=self.logger,
            status_manager=self.status_manager
        )
        self.ui_dataset_handler = UIDatasetHandler(
            self.ui, self.simulation, self.logger,
            self.ui_test_mask_handler, self.ui_mask_handler,
            status_manager=self.status_manager
        )
        self.ui_postprocessor_handler = UIPostprocessorHandler(
            self.ui, self.simulation, logger=self.logger,
            status_manager=self.status_manager
        )
        self.ui_reports_handler = UIReportsHandler(
            self.ui, self.simulation, logger=self.logger,
            status_manager=self.status_manager
        )
        self.ui_pipeline_handler = UIPipelineHandler(
            self.ui, self.simulation, logger=self.logger
        )

        # YAML handler (includes dataset handler for data format persistence)
        self.config_yaml_handler = ConfigYamlHandler(
            self.ui, self.ui_mask_handler, logger=self.logger,
            dataset_handler=self.ui_dataset_handler
        )

        # Setup new mode-based layout (replaces tabs with stepper)
        self._setup_mode_layout()

        # Set initial window size to accommodate all content
        self.resize(1400, 900)
        self.setMinimumSize(1200, 800)

        # Menu connections
        self.ui.action_save_config.triggered.connect(self.save_config)
        self.ui.action_load_config.triggered.connect(self.load_config)
        self.ui.action_save_experiment.triggered.connect(self.save_experiment)
        self.ui.action_load_experiment.triggered.connect(self.load_experiment)
        self.ui.action_log_settings.triggered.connect(self.show_log_settings)
        self.ui.action_external_apps.triggered.connect(self.show_external_apps_settings)
        self.ui.action_view_log.triggered.connect(self.show_log_viewer)
        self.ui.action_about.triggered.connect(self.show_about_dialog)

        # Dataset re-emit signals
        self.ui_dataset_handler.dataset_updated.connect(
            self.ui_mask_handler.dataset_changed.emit
        )
        self.ui_dataset_handler.dataset_updated.connect(
            self.ui_test_mask_handler.dataset_changed.emit
        )
        self.ui_dataset_handler.dataset_updated.connect(
            lambda size: (
                self.ui_test_mask_handler.visual_applicator.set_data(
                    self.simulation.dataset,
                    self.simulation.mask,
                    self.simulation.applicator,
                )
            ) if getattr(self.simulation, 'mask', None) is not None else None
        )
        self.ui_dataset_handler.dataset_updated.connect(
            lambda size: self.ui_postprocessor_handler.update_dataset_info()
        )
        self.ui_mask_handler.mask_created.connect(
            self.ui_test_mask_handler.visual_applicator.set_data
        )

    def _setup_mode_layout(self):
        """
        Setup the new mode-based layout with stepper navigation.
        Replaces the tab-based interface with:
        - Top bar: Mode selector (left) + Stepper widget (right, only in Single Test mode)
        - Content area below (stepper wizard or pipeline)
        """
        self.logger.debug("Setting up mode-based layout")

        # Hide the original tabWidget
        self.ui.tabWidget.hide()

        # Create central layout
        central_layout = QVBoxLayout(self.ui.centralwidget)
        central_layout.setContentsMargins(5, 5, 5, 5)
        central_layout.setSpacing(5)

        # === Top Bar: Logo + Mode selector + Stepper ===
        self.top_bar = QWidget()
        self.top_bar.setFixedHeight(90)  # Match stepper widget height
        top_bar_layout = QHBoxLayout(self.top_bar)
        top_bar_layout.setContentsMargins(5, 0, 0, 0)
        top_bar_layout.setSpacing(10)

        # Left: Logo banner
        self.logo_label = QLabel()
        logo_path = os.path.join(self._assets_dir, 'logo_banner.png')
        if os.path.exists(logo_path):
            logo_pixmap = QPixmap(logo_path)
            # Scale to fit height of top bar (80px max height, keep aspect ratio)
            scaled_logo = logo_pixmap.scaledToHeight(70, Qt.SmoothTransformation)
            self.logo_label.setPixmap(scaled_logo)
            self.logger.debug("Logo banner set from: %s", logo_path)
        self.logo_label.setFixedSize(self.logo_label.pixmap().size() if self.logo_label.pixmap() else QSize(100, 70))
        top_bar_layout.addWidget(self.logo_label)

        # Mode selector (compact horizontal)
        self.mode_selector = ModeSelectorWidget()
        self.mode_selector.mode_changed.connect(self._on_mode_changed)
        top_bar_layout.addWidget(self.mode_selector)

        # Create Single Test Container (stepper wizard) - need it before getting stepper widget
        self.single_test_container = SingleTestContainer(
            simulation=self.simulation,
            logger=self.logger,
            status_manager=self.status_manager
        )

        # Create Batch Test Container early - need it before getting its stepper widget
        self.batch_test_container = BatchTestContainer(
            simulation=self.simulation,
            logger=self.logger
        )

        # Create Batch Reports Container (no stepper needed)
        self.batch_reports_container = BatchReportsContainer(
            logger=self.logger
        )

        # Right: Stacked widget to hold steppers for each mode
        self.stepper_stack = QStackedWidget()
        self.stepper_stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        # Page 0: Single Test stepper widget
        self.stepper_widget = self.single_test_container.get_stepper_widget()
        self.stepper_stack.addWidget(self.stepper_widget)

        # Page 1: Empty placeholder for Batch Test (no stepper)
        batch_test_stepper_placeholder = QWidget()
        self.stepper_stack.addWidget(batch_test_stepper_placeholder)

        # Page 2: Empty placeholder for Batch Reports (no stepper)
        batch_reports_stepper_placeholder = QWidget()
        self.stepper_stack.addWidget(batch_reports_stepper_placeholder)

        top_bar_layout.addWidget(self.stepper_stack, 1)  # stretch=1 to fill remaining space

        central_layout.addWidget(self.top_bar)

        # === Content Area: Stacked widget for different mode containers ===
        self.mode_content_stack = QStackedWidget()
        self.mode_content_stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Setup responsive layouts for the containers BEFORE reparenting
        self._setup_container_layouts()

        # Wire handlers to single test container
        self.single_test_container.set_handlers({
            'dataset': self.ui_dataset_handler,
            'masks': self.ui_mask_handler,
            'test': self.ui_test_mask_handler,
            'postprocessor': self.ui_postprocessor_handler,
            'reports': self.ui_reports_handler,
        })

        # Connect dataset updates to batch test container
        self.ui_dataset_handler.dataset_updated.connect(
            lambda size: self.batch_test_container.update_dataset_notice()
        )
        self.ui_dataset_handler.dataset_updated.connect(
            lambda size: self.batch_test_container.test_config.set_img_size(size)
            if not self.batch_test_container.should_suppress_img_size_reset() else None
        )

        # Connect batch config dataset loading request
        self.batch_test_container.dataset_load_requested.connect(
            self._on_batch_dataset_requested
        )

        # Connect batch report availability for "Load Last Session" button
        self.batch_test_container.batch_report_available.connect(
            self.batch_reports_container.set_last_session_path
        )

        # Add containers to stack
        self.mode_content_stack.addWidget(self.single_test_container)
        self.mode_content_stack.addWidget(self.batch_test_container)
        self.mode_content_stack.addWidget(self.batch_reports_container)

        central_layout.addWidget(self.mode_content_stack, 1)  # stretch=1 to fill space

        # Start with Single Test mode
        self.mode_content_stack.setCurrentIndex(0)

        self.logger.debug("Mode-based layout configured")

    def _setup_container_layouts(self):
        """
        Setup responsive layouts for the main containers.
        Called before reparenting widgets to the stepper.
        """
        # Fix container size policies
        self.ui.dataset_main_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ui.masks_main_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ui.postprocessor_main_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ui.reports_main_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Hide old fixed-geometry widgets in test_masks_tab
        self.ui.verticalLayoutWidget_6.hide()
        self.ui.verticalLayoutWidget_7.hide()

        self.logger.debug("Container layouts configured")

    def _on_mode_changed(self, mode: str):
        """Handle mode switch between Single Test, Batch Test, and Batch Reports."""
        self.logger.debug("Mode changed to: %s", mode)
        if mode == "single_test":
            self.mode_content_stack.setCurrentIndex(0)
            self.stepper_stack.setCurrentIndex(0)  # Show Single Test stepper
        elif mode == "batch_test":
            self.mode_content_stack.setCurrentIndex(1)
            self.stepper_stack.setCurrentIndex(1)  # Show Batch Test stepper
            # Update dataset notice when entering batch test mode
            self.batch_test_container.update_dataset_notice()
        elif mode == "batch_reports":
            self.mode_content_stack.setCurrentIndex(2)
            self.stepper_stack.setCurrentIndex(2)  # Show empty placeholder (no stepper)

    def _on_batch_dataset_requested(self, dataset_info: dict, should_generate: bool):
        """
        Handle dataset loading request from batch config.

        Args:
            dataset_info: Dictionary with dataset type, name, img_size, source_path
            should_generate: If True, switch to Single Test and trigger generation
        """
        self.logger.info("Batch dataset request: %s, generate=%s", dataset_info, should_generate)

        # Map dataset type to menu index
        ds_type = dataset_info.get('type', '')
        menu_index = self._get_dataset_menu_index(ds_type)

        if menu_index is None:
            self.logger.warning("Unknown dataset type: %s", ds_type)
            return

        # Switch to Single Test mode
        self.mode_selector.set_mode("single_test")

        # Navigate to Dataset step (step 0) in the stepper
        self.single_test_container.state_manager.set_current_step(0)

        # Select the appropriate dataset type in the menu
        self.ui_dataset_handler.dataset_menu.setCurrentRow(menu_index)

        # Populate widget values based on dataset type
        self._populate_dataset_widget(dataset_info, menu_index)

        # If should_generate, trigger dataset generation and return to Batch Test after
        if should_generate:
            # Connect to dataset_updated to return to Batch Test mode after generation
            def on_dataset_generated(img_size):
                # Disconnect this one-time handler
                try:
                    self.ui_dataset_handler.dataset_updated.disconnect(on_dataset_generated)
                except TypeError:
                    pass  # Already disconnected
                # Return to Batch Test mode
                self.logger.info("Dataset generated, returning to Batch Test mode")
                self.mode_selector.set_mode("batch_test")
                # Clear the suppress flag now that we're back in Batch Test mode
                self.batch_test_container.clear_suppress_img_size_reset()

            self.ui_dataset_handler.dataset_updated.connect(on_dataset_generated)

            # Use a timer to allow UI to update first
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(100, lambda: self._trigger_dataset_generation(menu_index))

    def _get_dataset_menu_index(self, dataset_type: str) -> int:
        """Map dataset type class name to menu index."""
        # Menu indices:
        # 0 = Generate IR Profile (DatasetFromIRBeam)
        # 1 = Load Single Image (DatasetFromImage)
        # 2 = Load Multiple Images (DatasetFromFolder)
        # 3 = Download from Internet (DatasetFromCelebrities, DatasetFromSVHN)
        type_map = {
            'DatasetFromIRBeam': 0,
            'DatasetFromImage': 1,
            'DatasetFromFolder': 2,
            'DatasetFromCelebrities': 3,
            'DatasetFromSVHN': 3,
        }
        return type_map.get(dataset_type)

    def _populate_dataset_widget(self, dataset_info: dict, menu_index: int):
        """Populate the dataset widget with values from batch config."""
        ds_handler = self.ui_dataset_handler
        img_size = dataset_info.get('img_size', 64)
        source_path = dataset_info.get('source_path', '')
        num_images = dataset_info.get('num_images', 100)

        if menu_index == 0:  # IR Profile
            widget = ds_handler.ir_widget
            # Set image dimension
            idx = widget.image_dimension_value.findText(str(img_size))
            if idx >= 0:
                widget.image_dimension_value.setCurrentIndex(idx)
            # Set number of images
            widget.dataset_size_value.setValue(num_images)

        elif menu_index == 1:  # Single Image
            widget = ds_handler.selecciona_imagen_widget
            if source_path:
                widget.image_path_input.setText(source_path)

        elif menu_index == 2:  # Folder
            widget = ds_handler.selecciona_directorio_imagen_widget
            if source_path:
                widget.folder_path_input.setText(source_path)

        elif menu_index == 3:  # Internet (SVHN/Celebrities)
            widget = ds_handler.internet_widget
            ds_type = dataset_info.get('type', '')

            # Select correct dataset type
            if 'SVHN' in ds_type:
                widget.select_dataset_comboBox.setCurrentIndex(0)
            elif 'Celebrities' in ds_type:
                widget.select_dataset_comboBox.setCurrentIndex(1)

            # Set image dimension
            idx = widget.image_dimension_value.findText(str(img_size))
            if idx >= 0:
                widget.image_dimension_value.setCurrentIndex(idx)
            # Set number of images
            widget.dataset_size_value.setValue(num_images)

        self.logger.debug("Populated dataset widget for menu_index=%d", menu_index)

    def _trigger_dataset_generation(self, menu_index: int):
        """Trigger dataset generation by clicking the generate button."""
        ds_handler = self.ui_dataset_handler

        if menu_index == 0:
            ds_handler.ir_widget.generate_dataset_button.click()
        elif menu_index == 1:
            ds_handler.selecciona_imagen_widget.generate_dataset_button.click()
        elif menu_index == 2:
            ds_handler.selecciona_directorio_imagen_widget.generate_dataset_button.click()
        elif menu_index == 3:
            ds_handler.internet_widget.generate_dataset_button.click()

        self.logger.info("Triggered dataset generation for menu_index=%d", menu_index)

    def _setup_status_bar(self):
        """
        Setup the status bar with LED indicator and status text.
        Shows green LED + "Ready" when idle, red LED + task name when busy.
        """
        # Create LED indicator
        self.status_led = StatusLED(size=16)
        self.status_led.set_ready()

        # Create status text label
        self.status_text = QLabel("Ready")
        self.status_text.setStyleSheet("padding-left: 5px;")

        # Add widgets to status bar
        self.ui.statusbar.addWidget(self.status_led)
        self.ui.statusbar.addWidget(self.status_text)

        # Connect status manager signals
        self.status_manager.task_started.connect(self._on_task_started)
        self.status_manager.task_finished.connect(self._on_task_finished)
        self.status_manager.task_error.connect(self._on_task_error)

        self.logger.debug("Status bar configured")

    def _on_task_started(self, task_name: str):
        """Update status bar when a task starts."""
        self.status_led.set_busy()
        self.status_text.setText(f"{task_name}...")

    def _on_task_finished(self):
        """Update status bar when a task finishes."""
        self.status_led.set_ready()
        self.status_text.setText("Ready")

    def _on_task_error(self, error_message: str):
        """Update status bar when a task errors."""
        self.status_led.set_error()
        self.status_text.setText(f"Error: {error_message}")

    def show_about_dialog(self):
        """Show the About dialog with application information and logo."""
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton

        dialog = QDialog(self)
        dialog.setWindowTitle("About ASPIR")
        dialog.setMinimumWidth(500)

        layout = QVBoxLayout(dialog)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Logo at top center
        logo_path = os.path.join(self._assets_dir, 'logo_banner.png')
        if os.path.exists(logo_path):
            logo_label = QLabel()
            logo_pixmap = QPixmap(logo_path)
            scaled_logo = logo_pixmap.scaledToHeight(80, Qt.SmoothTransformation)
            logo_label.setPixmap(scaled_logo)
            logo_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(logo_label)

        # About text
        about_text = """<h2 style="text-align: center;">ASPIR</h2>
<p style="text-align: center;"><b>A Single-Pixel Imaging Research platform for energy-efficient benchmarking</b></p>
<p style="text-align: center;">Version 1.0</p>
<hr>
<p>A comprehensive platform for simulating and analyzing Single-Pixel Imaging (SPI)
systems with various mask patterns, reconstruction algorithms, and neural network
post-processing.</p>
<p><b>Features:</b></p>
<ul>
<li>Multiple mask types: Hadamard, Sweep, Scatter</li>
<li>Classical reconstruction: Conventional, Pseudoinverse, FISTA, TV-norm</li>
<li>Deep learning post-processing with 8+ model architectures</li>
<li>Performance analysis: timing, energy, image quality metrics</li>
</ul>
<hr>
<p><b>Repository:</b> <a href="https://github.com/Xarlie123/ir_beam">github.com/Xarlie123/ir_beam</a></p>
<p style="text-align: center;">PhD Research Project - Spectral Photon Imaging</p>
"""
        text_label = QLabel(about_text)
        text_label.setTextFormat(Qt.RichText)
        text_label.setWordWrap(True)
        text_label.setOpenExternalLinks(True)
        layout.addWidget(text_label)

        # Close button
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        close_btn.setFixedWidth(100)
        button_layout.addWidget(close_btn)
        button_layout.addStretch()
        layout.addLayout(button_layout)

        dialog.exec_()

    def show_log_settings(self):
        """Show the Log Settings dialog."""
        dialog = LogSettingsDialog(self)
        dialog.exec_()

    def show_external_apps_settings(self):
        """Show the External Applications Settings dialog."""
        manager = get_external_apps_manager(self.logger)
        dialog = ExternalAppsSettingsDialog(manager, self)
        dialog.exec_()

    def show_log_viewer(self):
        """Show the Log Viewer dialog."""
        dialog = LogViewerDialog(self)
        dialog.exec_()

    def save_config(self):
        """Save UI configuration as JSON with .single_test_config extension."""
        default_dir = self.config_yaml_handler.get_default_directory()
        default_name = f"config_{time.strftime('%Y%m%d_%H%M%S')}{FileExtensions.SINGLE_TEST_CONFIG}"

        archivo, _ = QFileDialog.getSaveFileName(
            self, "Save Configuration",
            str(default_dir / default_name),
            self.config_yaml_handler.get_file_filter()
        )
        if archivo:
            self.config_yaml_handler.save_config(archivo)
            self.logger.info(f"Configuration saved to {archivo}")

    def load_config(self):
        """Load UI configuration from JSON or legacy YAML."""
        default_dir = self.config_yaml_handler.get_default_directory()

        archivo, _ = QFileDialog.getOpenFileName(
            self, "Load Configuration",
            str(default_dir),
            self.config_yaml_handler.get_file_filter()
        )
        if archivo:
            self.config_yaml_handler.load_config(archivo)
            self.logger.info(f"Configuration loaded from {archivo}")

    def save_experiment(self):
        """
        Save a complete experiment to a .single_test_experiment directory.

        Structure:
        <name>.single_test_experiment/
        ├── config.json              # Widget configurations
        ├── datasets/
        │   └── dataset.npz          # Training/test/validation data
        ├── masks/
        │   └── masks.npz            # Generated mask patterns
        ├── model/
        │   ├── model.pt             # PyTorch weights
        │   ├── model.onnx           # ONNX export (if available)
        │   └── manifest.json        # Model metadata
        └── results/
            ├── test_results.npz     # Validation triplets
            └── metrics.json         # Training curves
        """
        from pathlib import Path

        # Get experiment name from user
        default_name = f"experiment_{time.strftime('%Y%m%d_%H%M%S')}"
        name, ok = QFileDialog.getSaveFileName(
            self,
            "Save Experiment",
            str(SINGLE_TESTS_DIR / f"{default_name}{FileExtensions.SINGLE_TEST_EXPERIMENT}"),
            f"Single Test Experiment (*{FileExtensions.SINGLE_TEST_EXPERIMENT});;All Files (*.*)"
        )

        if not name or not ok:
            return

        # Ensure correct extension
        exp_path = Path(name)
        if exp_path.suffix != FileExtensions.SINGLE_TEST_EXPERIMENT:
            exp_path = exp_path.with_suffix(FileExtensions.SINGLE_TEST_EXPERIMENT)

        self.logger.info("Saving experiment to: %s", exp_path)

        # Create directory structure
        dirs = SingleTestExperiment.create_structure(exp_path)

        ok_config = ok_dataset = ok_masks = False
        ok_model = ok_onnx = ok_test_results = ok_metrics = False

        # ---- Save Config (JSON) ----
        try:
            config_data = self.config_yaml_handler._collect_config_data()
            SingleTestExperiment.save_config(exp_path, config_data)
            ok_config = True
            self.logger.info("Config saved to %s", dirs["root"] / "config.json")
        except Exception as e:
            self.logger.exception("Error saving config: %s", e)

        # ---- Save Dataset ----
        ds = getattr(self.simulation, "dataset", None)
        try:
            if ds is not None and getattr(ds, "data", None) is not None and len(ds.data) > 0:
                ds_path = dirs["datasets"] / "dataset.npz"
                old_path = getattr(ds, "dataset_path", None)
                ds.dataset_path = str(ds_path)
                ds.save_dataset()
                if old_path is not None:
                    ds.dataset_path = old_path
                ok_dataset = True
                self.logger.info("Dataset saved to %s", ds_path)
            else:
                self.logger.warning("No dataset in memory to save.")
        except Exception as e:
            self.logger.exception("Error saving dataset: %s", e)

        # ---- Save Masks ----
        mk = getattr(self.simulation, "mask", None)
        try:
            if mk is not None and getattr(mk, "mascaras", None) is not None:
                mk_path = dirs["masks"] / "masks.npz"
                if hasattr(mk, "save_masks"):
                    ok_masks = bool(mk.save_masks(path=str(mk_path), compress=True))
                    if ok_masks:
                        self.logger.info("Masks saved to %s", mk_path)
                else:
                    self.logger.error("Mask object does not implement save_masks(path=...).")
            else:
                self.logger.warning("No masks in memory to save.")
        except Exception as e:
            self.logger.exception("Error saving masks: %s", e)

        # ---- Save Model + Results + Metrics ----
        pp = getattr(self.simulation, "postprocessor", None)
        if pp is not None and getattr(pp, "model", None) is not None:
            # Save PyTorch weights
            model_path = dirs["model"] / "model.pt"
            try:
                pp.save_model(str(model_path))
                ok_model = True
                self.logger.info("Model saved to %s", model_path)
            except Exception as e:
                self.logger.exception("Could not save model: %s", e)

            # Export to ONNX
            onnx_path = dirs["model"] / "model.onnx"
            try:
                import torch
                model = pp.model
                device = pp.device
                img_size = pp.img_size
                is_conv = pp.is_conv
                model.eval()

                if is_conv:
                    sample = torch.randn(1, 1, img_size, img_size, device=device)
                else:
                    sample = torch.randn(1, img_size * img_size, device=device)

                torch.onnx.export(
                    model, sample, str(onnx_path),
                    export_params=True, opset_version=17,
                    do_constant_folding=True,
                    input_names=['input'], output_names=['output'],
                    dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
                )
                ok_onnx = True
                self.logger.info("ONNX model saved to %s", onnx_path)
            except Exception as e:
                self.logger.warning("Could not export ONNX: %s", e)

            # Save model manifest
            manifest_path = dirs["model"] / "manifest.json"
            try:
                model_name = ""
                if hasattr(self.ui_postprocessor_handler, "get_current_model"):
                    model_name = self.ui_postprocessor_handler.get_current_model().lower()
                elif hasattr(pp, "model_name"):
                    model_name = str(pp.model_name).lower()

                model_manifest = {
                    "model_name": model_name,
                    "img_size": getattr(self.simulation.dataset, "img_size", None),
                    "is_conv": getattr(pp, "is_conv", None),
                    "n_params": int(pp.n_params) if hasattr(pp, "n_params") else None,
                    "has_onnx": ok_onnx,
                }
                with open(manifest_path, "w", encoding="utf-8") as f:
                    json.dump(model_manifest, f, indent=2, ensure_ascii=False)
            except Exception as e:
                self.logger.warning("Could not write model manifest: %s", e)

            # Save validation preview triplets
            results_path = dirs["results"] / "test_results.npz"
            try:
                vr = getattr(self.simulation, "validation_results", None)
                if vr and all(k in vr for k in ("original", "recons", "denoised")):
                    np.savez(
                        str(results_path),
                        original=np.asarray(vr["original"], dtype=np.float32),
                        recons=np.asarray(vr["recons"], dtype=np.float32),
                        denoised=np.asarray(vr["denoised"], dtype=np.float32),
                    )
                    ok_test_results = True
                    self.logger.info("Test results saved to %s", results_path)
            except Exception as e:
                self.logger.warning("Could not save test_results: %s", e)

            # Save training curves
            metrics_path = dirs["results"] / "metrics.json"
            try:
                viz = getattr(self.ui_postprocessor_handler, "visual_pp", None)
                if viz is not None and hasattr(viz, "val_losses") and hasattr(viz, "test_losses"):
                    metrics = {
                        "val_losses": list(getattr(viz, "val_losses", [])),
                        "test_losses": list(getattr(viz, "test_losses", [])),
                    }
                    with open(metrics_path, "w", encoding="utf-8") as f:
                        json.dump(metrics, f, indent=2, ensure_ascii=False)
                    ok_metrics = True
                    self.logger.info("Training metrics saved to %s", metrics_path)
            except Exception as e:
                self.logger.warning("Could not save metrics: %s", e)
        else:
            self.logger.info("No trained/loaded postprocessor to save model/results.")

        # ---- Summary dialog ----
        msg = [
            f"Experiment saved to: {exp_path.name}",
            "",
            f"Config: {'OK' if ok_config else 'NO'}",
            f"Dataset: {'OK' if ok_dataset else 'NO'}",
            f"Masks: {'OK' if ok_masks else 'NO'}",
            f"Model (.pt): {'OK' if ok_model else 'NO'}",
            f"Model (.onnx): {'OK' if ok_onnx else 'NO'}",
            f"Test results: {'OK' if ok_test_results else 'NO'}",
            f"Metrics: {'OK' if ok_metrics else 'NO'}",
        ]
        QMessageBox.information(self, "Save Experiment", "\n".join(msg))

    def load_experiment(self):
        """
        English comment:
        Load a complete experiment from a folder.
        Ensures dataset/mask/postprocessor objects exist, loads artifacts,
        synthesizes preview if needed, and refreshes UI.
        """
        import importlib

        in_dir = QFileDialog.getExistingDirectory(
            self, "Select experiment folder to load", ""
        )
        if not in_dir:
            return

        self.logger.info("Loading experiment from folder: %s", in_dir)

        yaml_path           = os.path.join(in_dir, "config.yaml")
        ds_path             = os.path.join(in_dir, "dataset.npz")
        mk_path             = os.path.join(in_dir, "mascaras.npz")
        manifest_path       = os.path.join(in_dir, "manifest.json")
        model_path          = os.path.join(in_dir, "model.pth")
        model_manifest_path = os.path.join(in_dir, "model_manifest.json")
        test_results_path   = os.path.join(in_dir, "test_results.npz")
        metrics_path        = os.path.join(in_dir, "metrics.json")

        # ---- Read manifest (optional) ----
        manifest = {}
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f) or {}
                self.logger.debug("Manifest: %s", manifest)
            except Exception as e:
                self.logger.warning("Invalid manifest: %s", e)

        # ---- Helper to instantiate dataset/mask if missing ----
        def _instantiate_from_manifest(kind: str, fallback_img_size: int | None):
            """
            English comment:
            kind: 'dataset' or 'mask'. Creates instance using module/class if present
            or falls back to lightweight holders compatible with our load methods.
            """
            cls_name = manifest.get(f"{kind}_cls")
            mod_name = manifest.get(f"{kind}_module")
            img_size = manifest.get(f"{kind}_img_size", fallback_img_size)

            if cls_name and mod_name:
                try:
                    mod = importlib.import_module(mod_name)
                    cls = getattr(mod, cls_name)
                    if kind == "dataset":
                        try:
                            return cls(getattr(self, "name", "LoadedDataset"), img_size, logger=self.logger)
                        except Exception:
                            return cls(img_size, logger=self.logger)
                    else:
                        return cls(img_size=img_size, logger=self.logger)
                except Exception as e:
                    self.logger.warning("Could not instantiate %s %s.%s: %s", kind, mod_name, cls_name, e)

            # Fallback holders
            if kind == "dataset":
                class _DatasetNPZHolder:
                    """English comment: Minimal dataset holder for NPZ loading."""
                    def __init__(self, img_size, logger):
                        self.name = "LoadedDataset"
                        self.img_size = int(img_size) if img_size is not None else None
                        self.data = []
                        self.dataset_type = "NPZHolder"
                        self.dataset_path = ""
                        self.logger = logger

                    def save_dataset(self):
                        if self.dataset_path and self.data:
                            images_array = np.array(self.data)
                            np.savez(self.dataset_path, images=images_array)
                            self.logger.info("Dataset holder saved to %s", self.dataset_path)

                    def load_data(self, progress_callback=None):
                        # English comment: Load images from NPZ
                        if not self.dataset_path or not os.path.exists(self.dataset_path):
                            self.logger.error("Dataset file not found: %s", self.dataset_path)
                            return False
                        with np.load(self.dataset_path, allow_pickle=False) as npz:
                            if "images" not in npz.files:
                                self.logger.error("Key 'images' not found in NPZ: %s", self.dataset_path)
                                return False
                            images = npz["images"]
                        if images.ndim < 3:
                            self.logger.error("Invalid images shape: %s", images.shape)
                            return False
                        if self.img_size is None:
                            self.img_size = int(images.shape[1])
                        self.data = list(images)
                        if progress_callback:
                            try:
                                progress_callback(1, 1)
                            except Exception:
                                pass
                        return True

                if img_size is None:
                    try:
                        with np.load(ds_path, allow_pickle=False) as _npz:
                            im = _npz["images"]
                            img_size = int(im.shape[1])
                    except Exception:
                        img_size = 0
                return _DatasetNPZHolder(img_size, logger=self.logger)

            else:
                from simulation_engine._2_mask_gen.mask import MascaraABC
                class _MascaraHolder(MascaraABC):
                    def generate_masks(self, progress_callback=None):
                        raise RuntimeError("Holder cannot generate masks; use load_masks().")
                if img_size is None:
                    try:
                        with np.load(mk_path, allow_pickle=False) as _npz:
                            m = _npz["mascaras"]
                            img_size = int(m.shape[1])
                    except Exception:
                        img_size = 0
                return _MascaraHolder(img_size=img_size, logger=self.logger)

        # ---- Load YAML (UI state) ----
        ok_yaml = True
        try:
            if os.path.exists(yaml_path):
                self.config_yaml_handler.load_from_yaml(yaml_path)
            else:
                ok_yaml = False
                self.logger.warning("config.yaml not found in selected folder.")
        except Exception as e:
            ok_yaml = False
            self.logger.exception("Error loading YAML: %s", e)

        # ---- Ensure dataset exists, then load NPZ ----
        ok_dataset = True
        ds = getattr(self.simulation, "dataset", None)
        if ds is None and os.path.exists(ds_path):
            ds = _instantiate_from_manifest("dataset", fallback_img_size=None)
            if hasattr(self.simulation, "set_dataset"):
                try:
                    self.simulation.set_dataset(ds)
                except Exception:
                    self.simulation.dataset = ds
            else:
                self.simulation.dataset = ds

        try:
            if ds is not None and os.path.exists(ds_path):
                old_path = getattr(ds, "dataset_path", None)
                ds.dataset_path = ds_path
                loaded = bool(ds.load_data(progress_callback=None))
                if old_path is not None:
                    ds.dataset_path = old_path
                ok_dataset = loaded
                if loaded:
                    size = getattr(ds, "img_size", None) or 0
                    try:
                        self.ui_dataset_handler.dataset_updated.emit(size)
                    except Exception:
                        pass
                    self.logger.info("Dataset loaded from %s", ds_path)
                else:
                    self.logger.error("Failed to load dataset from %s", ds_path)
            else:
                ok_dataset = False
                self.logger.warning("No dataset.npz or could not create dataset.")
        except Exception as e:
            ok_dataset = False
            self.logger.exception("Error loading dataset: %s", e)

        # ---- Ensure mask exists, then load NPZ ----
        ok_masks = True
        mk = getattr(self.simulation, "mask", None)
        if mk is None and os.path.exists(mk_path):
            mk = _instantiate_from_manifest("mask", fallback_img_size=getattr(ds, "img_size", None))
            if hasattr(self.simulation, "set_mascara"):
                try:
                    self.simulation.set_mask(mk)
                except Exception:
                    self.simulation.mask = mk
            else:
                self.simulation.mask = mk

        try:
            if mk is not None and os.path.exists(mk_path):
                if hasattr(mk, "load_masks"):
                    loaded = bool(mk.load_masks(path=mk_path, progress_callback=None, mmap_mode=None))
                    ok_masks = loaded
                    if loaded:
                        try:
                            self.ui_mask_handler.mask_created.emit(
                                self.simulation.dataset,
                                self.simulation.mask,
                                getattr(self.simulation, "applicator", None)
                            )
                        except Exception:
                            pass
                        self.logger.info("Masks loaded from %s", mk_path)
                    else:
                        self.logger.error("Failed to load masks from %s", mk_path)
                else:
                    ok_masks = False
                    self.logger.error("Mask object does not expose load_masks(path=...).")
            else:
                ok_masks = False
                self.logger.warning("No mascaras.npz or could not create mask.")
        except Exception as e:
            ok_masks = False
            self.logger.exception("Error loading masks: %s", e)

        # ---- Load model + preview + metrics (optional) ----
        try:
            from simulation_engine._4_postprocessor.postprocessor_nn import PostprocessorNN

            model_manifest = {}
            if os.path.exists(model_manifest_path):
                try:
                    with open(model_manifest_path, "r", encoding="utf-8") as f:
                        model_manifest = json.load(f) or {}
                    self.logger.debug("Model manifest: %s", model_manifest)
                except Exception as e:
                    self.logger.warning("Could not read model_manifest.json: %s", e)

            # Create engine if model exists
            if os.path.exists(model_path):
                pp = getattr(self.simulation, "postprocessor", None)
                if pp is None:
                    model_name = (model_manifest.get("model_name") or "").lower()
                    overrides  = model_manifest.get("overrides") or {}
                    if "img_size" not in overrides or overrides["img_size"] is None:
                        overrides["img_size"] = getattr(self.simulation.dataset, "img_size", None)
                    try:
                        self.simulation.set_postprocessor(
                            self.simulation.dataset,
                            self.simulation.mask,
                            getattr(self.simulation, "applicator", None),
                            postprocesador_cls=PostprocessorNN,
                            model_name=model_name,
                            model_overrides=overrides,
                            batch_size=16,
                            lr=1e-3,
                            weight_decay=1e-5
                        )
                    except Exception as e:
                        self.logger.exception("Could not create PostprocessorNN: %s", e)
                        self.simulation.postprocessor = PostprocessorNN(
                            model_name=model_name,
                            model_overrides=overrides,
                            dataset=self.simulation.dataset,
                            aplicador=getattr(self.simulation, "applicator", None),
                            batch_size=16,
                            lr=1e-3,
                            weight_decay=1e-5,
                            logger=self.logger
                        )

                # Load weights and mark as trained
                try:
                    self.simulation.postprocessor.load_model(model_path)
                    self.simulation.postprocessor.trained = True
                    # Friendly type for other tabs
                    self.simulation.postprocessor.postproc_type = model_manifest.get("model_name", "NN")
                    self.logger.info("Model loaded from %s", model_path)
                except Exception as e:
                    self.logger.exception("Error loading model: %s", e)

            # Load saved preview if present
            if os.path.exists(test_results_path):
                try:
                    with np.load(test_results_path, allow_pickle=False) as npz:
                        orig = npz["original"]
                        rec  = npz["recons"]
                        den  = npz["denoised"]
                    self.simulation.validation_results = {
                        "original": list(orig),
                        "recons":   list(rec),
                        "denoised": list(den)
                    }
                    viz = getattr(self.ui_postprocessor_handler, "visual_pp", None)
                    if viz is not None:
                        viz.set_images(self.simulation.validation_results["original"],
                                       self.simulation.validation_results["recons"],
                                       self.simulation.validation_results["denoised"])
                        model_name_for_info = model_manifest.get("model_name", "")
                        if not model_name_for_info and hasattr(self.ui_postprocessor_handler, "get_current_model"):
                            model_name_for_info = self.ui_postprocessor_handler.get_current_model()
                        viz.update_info(
                            num_images=len(self.simulation.validation_results["denoised"]),
                            img_size=getattr(self.simulation.dataset, "img_size", 0),
                            tipo_dataset=getattr(self.simulation.dataset, "dataset_type", ""),
                            tipo_mascara=type(getattr(self.simulation, "mask", object())).__name__,
                            tipo_postprocesado=model_name_for_info,
                            n_params=getattr(self.simulation.postprocessor, "n_params", None)
                        )
                        viz.image_slider_value.setValue(0)
                    self.logger.info("Test results loaded from %s", test_results_path)
                except Exception as e:
                    self.logger.warning("Could not load test_results: %s", e)

            # Load metrics if present
            if os.path.exists(metrics_path):
                try:
                    with open(metrics_path, "r", encoding="utf-8") as f:
                        metrics = json.load(f) or {}
                    viz = getattr(self.ui_postprocessor_handler, "visual_pp", None)
                    if viz is not None:
                        viz.val_losses = metrics.get("val_losses", [])
                        viz.test_losses = metrics.get("test_losses", [])
                        viz.plot_losses()
                    self.logger.info("Metrics loaded from %s", metrics_path)
                except Exception as e:
                    self.logger.warning("Could not load metrics: %s", e)

            # ---- If no preview saved, synthesize from the loaded model ----
            try:
                if not getattr(self.simulation, "validation_results", None):
                    pp = getattr(self.simulation, "postprocessor", None)
                    if pp is not None and getattr(pp, "trained", False):
                        orig, recons, denoised = pp.test_dataset()
                        self.simulation.validation_results = {
                            "original": orig,
                            "recons":   recons,
                            "denoised": denoised
                        }
                        viz = getattr(self.ui_postprocessor_handler, "visual_pp", None)
                        if viz is not None:
                            viz.set_images(orig, recons, denoised)
                            model_name_for_info = ""
                            if os.path.exists(model_manifest_path):
                                try:
                                    with open(model_manifest_path, "r", encoding="utf-8") as f:
                                        _mm = json.load(f) or {}
                                    model_name_for_info = _mm.get("model_name", "")
                                except Exception:
                                    pass
                            if not model_name_for_info and hasattr(self.ui_postprocessor_handler, "get_current_model"):
                                model_name_for_info = self.ui_postprocessor_handler.get_current_model()
                            viz.update_info(
                                num_images=len(denoised),
                                img_size=getattr(self.simulation.dataset, "img_size", 0),
                                tipo_dataset=getattr(self.simulation.dataset, "dataset_type", ""),
                                tipo_mascara=type(getattr(self.simulation, "mask", object())).__name__,
                                tipo_postprocesado=model_name_for_info,
                                n_params=getattr(pp, "n_params", None)
                            )
                            viz.image_slider_value.setValue(0)
                        setattr(pp, "postproc_type", model_name_for_info or "NN")
                        self.logger.info("Preview synthesized from loaded model (no test_results.npz found)")
            except Exception as e:
                self.logger.warning("Could not synthesize preview from model: %s", e)

        except Exception as e:
            self.logger.exception("General error loading model/results: %s", e)

        # ---- Final safety: ask handler to refresh preview if possible ----
        try:
            self.ui_postprocessor_handler.refresh_preview_from_state()
        except Exception:
            pass

        # ---- Summary dialog ----
        msg = [
            f"YAML: {'OK' if ok_yaml else 'NO'}",
            f"Dataset: {'OK' if ok_dataset else 'NO'}",
            f"Masks: {'OK' if ok_masks else 'NO'}",
        ]
        QMessageBox.information(self, "Load experiment", "\n".join(msg))
