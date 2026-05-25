"""Main application window — top-level QMainWindow wiring UI handlers together."""
from __future__ import annotations

import os
import time

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from simulation_engine.simulation import Simulation
from ui._1_dataset.ui_dataset_handler import UIDatasetHandler
from ui._2_masks.ui_mask_handler import UIMaskHandler
from ui._3_test_masks.ui_test_mask_handler import UITestMaskHandler
from ui._4_postprocessor.ui_postprocessor_handler import UIPostprocessorHandler
from ui._5_reports.ui_reports_handler import UIReportsHandler
from ui._7_pipeline.ui_pipeline_handler import UIPipelineHandler
from ui.custom_widgets.batch_reports import BatchReportsContainer
from ui.custom_widgets.batch_test import BatchTestContainer
from ui.custom_widgets.mode_selector import ModeSelectorWidget
from ui.main_window._about_dialog import show_about_dialog
from ui.main_window._load_experiment import load_experiment
from ui.main_window._save_experiment import save_experiment
from ui.main_window._status_led import StatusLED
from ui.modes import SingleTestContainer
from ui.ui_main_window import Ui_MainWindow
from ui.utils.config_yaml_handler import ConfigYamlHandler
from ui.utils.external_apps_settings import (
    ExternalAppsSettingsDialog,
    get_external_apps_manager,
)
from ui.utils.file_formats import FileExtensions
from ui.utils.log_manager import get_log_manager
from ui.utils.log_settings_dialog import LogSettingsDialog
from ui.utils.log_viewer_dialog import LogViewerDialog
from ui.utils.status_manager import StatusManager


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Initialize logging system using LogManager
        self.log_manager = get_log_manager()
        self.logger = self.log_manager.setup_logging("ASPIR")
        self.logger.debug("Initializing MainWindow")

        # UI + simulation
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Set window icon.
        # __file__ is src/ui/main_window/window.py — go up 3 levels to reach src/,
        # then one more to land in the repo root, then append 'assets'.
        self._assets_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '..', 'assets'
        )
        icon_path = os.path.join(self._assets_dir, 'icon_app.png')
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
            self.logger.debug("Window icon set from: %s", icon_path)

        self.simulation = Simulation(logger=self.logger)

        # Status manager for task state
        self.status_manager = StatusManager(logger=self.logger)
        self._setup_status_bar()

        # Handlers (pass status_manager to each for task state management)
        self.ui_mask_handler = UIMaskHandler(
            self.ui, self.simulation, logger=self.logger,
            status_manager=self.status_manager
        )
        self.ui_test_mask_handler = UITestMaskHandler(
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
            from PySide6.QtCore import QTimer
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
            # Forward the whole dataset_info through the widget's set_settings
            # so the user sees exactly which beam parameters were used. Fields
            # absent from dataset_info keep their current UI values.
            settings = {
                "img_size": img_size,
                "num_images": num_images,
            }
            if "seed" in dataset_info and dataset_info["seed"] is not None:
                settings["seed"] = int(dataset_info["seed"])
            if "data_format" in dataset_info:
                settings["data_format"] = dataset_info["data_format"]
            mode_settings = {}
            if "mode_distribution" in dataset_info:
                mode_settings["mode_distribution"] = dataset_info["mode_distribution"]
            if "speckle_noise" in dataset_info:
                mode_settings["speckle_noise"] = dataset_info["speckle_noise"]
            if "max_mode_order" in dataset_info:
                mode_settings["max_mode_order"] = dataset_info["max_mode_order"]
            if mode_settings:
                settings["mode_settings"] = mode_settings
            widget.set_settings(settings)

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
            # Restore seed and data_format if they were saved
            if dataset_info.get("seed") is not None:
                widget.random_seed_value.setValue(int(dataset_info["seed"]))
            if "data_format" in dataset_info and hasattr(widget, "data_format_selector"):
                widget.data_format_selector.set_format(dataset_info["data_format"])

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
        show_about_dialog(self, self._assets_dir)

    def show_log_settings(self):
        """Show the Log Settings dialog."""
        dialog = LogSettingsDialog(self)
        dialog.exec()

    def show_external_apps_settings(self):
        """Show the External Applications Settings dialog."""
        manager = get_external_apps_manager(self.logger)
        dialog = ExternalAppsSettingsDialog(manager, self)
        dialog.exec()

    def show_log_viewer(self):
        """Show the Log Viewer dialog."""
        dialog = LogViewerDialog(self)
        dialog.exec()

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
        """Save a complete experiment to a .single_test_experiment directory."""
        save_experiment(self)

    def load_experiment(self):
        """Load a complete experiment from a folder."""
        load_experiment(self)
