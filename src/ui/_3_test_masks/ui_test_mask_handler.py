import logging
import numpy as np
import matplotlib.cm as cm  # For the jet colormap
from PyQt5.QtWidgets import (QMessageBox, QSizePolicy, QVBoxLayout, QHBoxLayout,
                              QApplication, QWidget, QLabel, QFrame)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, pyqtSignal, QObject, QTimer

from ui.custom_widgets.visualizers.visual_dataset.visual_dataset_widget import VisualDatasetWidget
from ui.custom_widgets.visualizers.visual_applicator.visual_applicator_widget import VisualApplicatorWidget
from PIL import Image, ImageDraw, ImageFont


class UITestMascaraHandler(QObject):
    """
    Handler for mask testing: visualizes dataset and applicator.
    Updates previews when the Test Masks tab becomes visible.
    """
    dataset_changed = pyqtSignal(int)

    def __init__(self, ui, simulation, ui_mask_handler, logger=None, status_manager=None):
        super().__init__()
        self.ui = ui
        self.simulation = simulation
        self.ui_mask_handler = ui_mask_handler
        self.status_manager = status_manager
        # Configure logger
        if logger is None:
            self.logger = logging.getLogger(self.__class__.__name__)
        else:
            self.logger = logger.getChild(self.__class__.__name__)
        self.logger.debug("Initializing UITestMascaraHandler")

        self.img_size = 64  # default image size

        # Track if data has changed and needs refresh when tab becomes visible
        self._dataset_needs_refresh = False
        self._applicator_needs_refresh = False

        # Setup the new menu-based interface (similar to dataset/mask tabs)
        self._setup_test_masks_interface()

        # Connect dataset_changed signals
        self.dataset_changed.connect(self._on_dataset_changed)

        # Connect internal slider signals
        self.visual_widget.select_image_slider_value.valueChanged.connect(
            self.visual_widget._on_slider_moved
        )

        # Connect mask creation to reload applicator data
        self.ui_mask_handler.mask_created.connect(self._on_mask_created)

        # Sync dataset slider to applicator index
        self.visual_widget.select_image_slider_value.valueChanged.connect(
            self.visual_applicator.set_image_index
        )

    def _setup_test_masks_interface(self):
        """Setup the Test Masks interface with consistent styling."""
        self.logger.debug("Setting up Test Masks interface")

        # Common panel style (same as dataset/mask tabs)
        panel_style = """
            QWidget#previewPanel {
                background-color: #f5f5f5;
                border: 1px solid #ccc;
                border-radius: 4px;
            }
        """

        # Create main container
        self.test_masks_container = QWidget()
        self.test_masks_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Main horizontal layout
        main_layout = QHBoxLayout(self.test_masks_container)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(10)

        # Left panel: Dataset Preview
        left_panel = QWidget()
        left_panel.setObjectName("previewPanel")
        left_panel.setStyleSheet(panel_style)
        left_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(5)

        left_title = QLabel("<h3>Dataset preview:</h3>")
        left_title.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(left_title)

        # Create VisualDatasetWidget
        self.visual_widget = VisualDatasetWidget(parent=left_panel, logger=self.logger)
        self.visual_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # Hide progress bars - dataset is already loaded at this step
        self.visual_widget.dataset_progress_bar.hide()
        self.visual_widget.phase_progress.hide()
        # Reduce controls height since progress bar is hidden (120 instead of 170)
        self.visual_widget.set_controls_height(120)
        left_layout.addWidget(self.visual_widget, 1)

        main_layout.addWidget(left_panel, 1)
        self.logger.info("VisualDatasetWidget integrated")

        # Right panel: Applicator Preview
        right_panel = QWidget()
        right_panel.setObjectName("previewPanel")
        right_panel.setStyleSheet(panel_style)
        right_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(5)

        right_title = QLabel("<h3>Reconstruction Preview:</h3>")
        right_title.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(right_title)

        # Create VisualApplicatorWidget
        self.visual_applicator = VisualApplicatorWidget(
            self.simulation, parent=right_panel, logger=self.logger,
            status_manager=self.status_manager
        )
        self.visual_applicator.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_layout.addWidget(self.visual_applicator, 1)

        main_layout.addWidget(right_panel, 1)
        self.logger.info("VisualApplicatorWidget integrated")

        # Add the container to the tab (will be done in main_window.py via _setup_responsive_layouts)
        # Store reference for main_window to access
        self.main_container = self.test_masks_container

        self.logger.info("Test Masks interface setup complete")

    def on_tab_visible(self):
        """
        Called when the Test Masks tab becomes visible.
        Refreshes the previews if data has changed since last view.
        """
        self.logger.debug("Test Masks tab became visible")

        # Small delay to let layout settle before rendering
        QTimer.singleShot(50, self._refresh_previews)

    def _refresh_previews(self):
        """Refresh both dataset and applicator previews."""
        if self._dataset_needs_refresh:
            self.logger.debug("Refreshing dataset preview")
            self._dataset_needs_refresh = False
            current_idx = self.visual_widget.select_image_slider_value.value()
            self.visual_widget._on_slider_moved(current_idx)

        if self._applicator_needs_refresh:
            self.logger.debug("Refreshing applicator preview")
            self._applicator_needs_refresh = False
            current_mask_idx = self.visual_applicator.select_image_slider_value.value()
            self.visual_applicator._on_slider_moved(current_mask_idx)

    def _on_dataset_changed(self, size):
        """
        Callback when dataset changes: updates widget data and marks for refresh.
        """
        self.logger.debug(f"Dataset changed: image size = {size}")
        ds = self.simulation.dataset

        # Update dataset view data
        self.visual_widget.set_data(ds.data, getattr(ds, 'data_format', None))
        self.visual_widget.update_info(
            len(ds.data),
            size,
            ds.dataset_type
        )

        # Mark that we need to refresh when tab becomes visible
        self._dataset_needs_refresh = True

        # If tab is already visible, refresh immediately
        if self.ui.test_masks_tab.isVisible():
            QTimer.singleShot(50, self._refresh_previews)

    def _on_mask_created(self, dataset, mask, applicator):
        """
        Callback when a mask is created: updates applicator visual data and marks for refresh.
        """
        self.logger.info("Mask created, updating VisualApplicatorWidget")
        self.visual_applicator.set_data(dataset, mask, applicator)

        # Mark that we need to refresh when tab becomes visible
        self._applicator_needs_refresh = True

        # If tab is already visible, refresh immediately
        if self.ui.test_masks_tab.isVisible():
            QTimer.singleShot(50, self._refresh_previews)

    def get_content_widget(self):
        """Return the main container widget for stepper integration."""
        return self.main_container
