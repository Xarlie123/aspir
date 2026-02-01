# File: ui/_2_masks/ui_mask_handler.py

import logging
import numpy as np
from PyQt5.QtWidgets import (QMessageBox, QListWidget, QStackedWidget, QLabel,
                               QVBoxLayout, QWidget, QScrollArea, QSizePolicy, QFrame)
from PyQt5.QtCore import QObject, pyqtSignal, Qt
from PyQt5.QtGui import QFont

from simulation_engine._2_mask_gen.mask_cal_sal import MaskCalSal
from simulation_engine._2_mask_gen.mask_hadamard_walsh_paley import MaskHadamardWalshPaley
from simulation_engine._2_mask_gen.mask_hadamard import MaskHadamard
from simulation_engine._2_mask_gen.mask_hadamard_cake_cutting import MaskHadamardCakeCutting
from simulation_engine._2_mask_gen.mask_hadamard_scramble import MaskHadamardScramble
from simulation_engine._2_mask_gen.mask_scatter import MaskScatter

from ui.custom_widgets.mask_control.sweep_control.sweep_control_widget import SweepControlWidget
from ui.custom_widgets.mask_control.scatter_control.scatter_control_widget import ScatterControlWidget
from ui.custom_widgets.mask_control.hadamard_control.hadamard_control_widget import HadamardControlWidget
from ui._2_masks.mask_worker import MascaraWorker
from ui.custom_widgets.visualizers.visual_mask.visual_mask_widget import VisualMaskWidget
# from ui.utils.widget_helpers import embed_widget  # No longer needed
from ui.utils.worker_launcher import WorkerLauncher

class UIMaskHandler(QObject):
    """Manages the masks tab: controls, generation and preview."""
    mask_created    = pyqtSignal(object, object, object)
    dataset_changed = pyqtSignal(int)

    def __init__(self, ui, simulation, logger=None, status_manager=None):
        super().__init__()
        # Logger
        if logger is None:
            self.logger = logging.getLogger(self.__class__.__name__)
        else:
            self.logger = logger.getChild(self.__class__.__name__)
        self.logger.debug("Initializing UIMaskHandler")

        self.ui = ui
        self.simulation = simulation
        self.status_manager = status_manager

        # React to dataset changes
        self.dataset_changed.connect(self._on_dataset_changed)
        self.logger.debug("Connected signal dataset_changed to method _on_dataset_changed")

        # Create mask control widgets first (before embedding in menu interface)
        self._setup_sweep_control()
        self._setup_scatter_control()
        self._setup_hadamard_controls()

        # Create mask preview widget (before menu interface setup)
        self.visual_widget = VisualMaskWidget(parent=None, logger=self.logger)
        self.logger.debug("VisualMaskWidget created")

        # Setup menu-based interface (uses the widgets created above)
        self.logger.debug("Setting up menu-based interface for masks")
        self._setup_menu_interface()

        # Connect post-mask signals
        self.mask_created.connect(
            lambda dataset, mask, applicator: self.visual_widget.set_masks(mask.mascaras)
        )
        self.mask_created.connect(
            lambda dataset, mask, applicator: self.visual_widget.update_info(
                len(mask.mascaras),
                dataset.img_size,
                type(mask).__name__
            )
        )
        self.logger.debug("Signals mask_created connected to visual_widget")

    def _setup_menu_interface(self):
        """Setup menu-based interface with QListWidget and QStackedWidget."""
        self.logger.debug("Creating QListWidget menu for mask selection")

        # Create QListWidget for menu
        self.masks_menu = QListWidget()
        self.masks_menu.addItems([
            "Sweep",
            "Scatter",
            "Hadamard (Natural)",
            "Hadamard (Scramble)",
            "Hadamard (Cake Cutting)",
            "Hadamard (Walsh-Paley)",
            "Cal/Sal"
        ])
        self.masks_menu.setCurrentRow(0)
        self.masks_menu.currentRowChanged.connect(self._on_menu_selection_changed)

        # Style the menu
        self.masks_menu.setStyleSheet("""
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
        self.logger.debug("Creating QStackedWidget for mask controls")
        self.masks_stacked = QStackedWidget()

        # Common style for content panels
        panel_style = """
            QWidget#contentPanel {
                background-color: #f5f5f5;
                border: 1px solid #ccc;
                border-radius: 4px;
            }
        """

        # Helper function to create a page with proper sizing
        def create_page(title, widget):
            page = QWidget()
            page.setObjectName("contentPanel")
            page.setStyleSheet(panel_style)
            layout = QVBoxLayout(page)
            layout.setContentsMargins(10, 10, 10, 10)
            layout.setSpacing(10)

            # Title label
            title_label = QLabel(f"<h3>{title}</h3>")
            layout.addWidget(title_label)

            # Ensure widget has proper size policy
            widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            widget.setMinimumHeight(100)
            layout.addWidget(widget)

            # Add stretch to push content to top
            layout.addStretch()

            return page

        # Create pages
        page0 = create_page("Combined Sweep", self.sweep_control)
        self.masks_stacked.addWidget(page0)
        self.logger.debug("Added page 0: Sweep")

        page1 = create_page("Gaussian Scatter", self.scatter_control)
        self.masks_stacked.addWidget(page1)
        self.logger.debug("Added page 1: Scatter")

        # Hadamard variants and Cal/Sal
        hadamard_titles = [
            "Hadamard Natural",
            "Hadamard Scramble",
            "Hadamard Cake Cutting",
            "Hadamard Walsh-Paley",
            "Cal/Sal Transform"
        ]
        for idx, (title, hadamard_ctrl) in enumerate(zip(hadamard_titles, self.hadamard_controls)):
            page = create_page(title, hadamard_ctrl)
            self.masks_stacked.addWidget(page)
            self.logger.debug("Added page %d: %s", idx + 2, title)

        # Embed menu into placeholder
        self.logger.debug("Embedding menu into masks_menu_placeholder")
        menu_layout = QVBoxLayout(self.ui.masks_menu_placeholder)
        menu_layout.setContentsMargins(0, 0, 0, 0)
        menu_layout.addWidget(self.masks_menu)

        # Embed stacked widget into content placeholder with scroll area
        self.logger.debug("Embedding stacked widget into masks_content_placeholder")
        content_layout = QVBoxLayout(self.ui.masks_content_placeholder)
        content_layout.setContentsMargins(0, 0, 0, 0)

        # Wrap stacked widget in scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setWidget(self.masks_stacked)
        content_layout.addWidget(scroll_area)

        # Embed preview widget into masks_preview_placeholder
        self.logger.debug("Embedding preview widget into masks_preview_placeholder")

        # Style the preview placeholder with same panel style
        self.ui.masks_preview_placeholder.setStyleSheet("""
            QWidget#masks_preview_placeholder {
                background-color: #f5f5f5;
                border: 1px solid #ccc;
                border-radius: 4px;
            }
        """)
        self.ui.masks_preview_placeholder.setObjectName("masks_preview_placeholder")

        preview_layout = QVBoxLayout(self.ui.masks_preview_placeholder)
        preview_layout.setContentsMargins(10, 10, 10, 10)

        preview_label = QLabel("<h3>Masks preview:</h3>")
        preview_label.setAlignment(Qt.AlignCenter)
        preview_layout.addWidget(preview_label)

        # Ensure visual widget expands
        self.visual_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        preview_layout.addWidget(self.visual_widget, 1)

        # Explicitly set the first page as current
        self.masks_stacked.setCurrentIndex(0)
        self.logger.debug("Set initial stacked widget page to index 0")

        self.logger.info("Menu-based interface setup complete")

    def _on_menu_selection_changed(self, index):
        """Switch stacked widget page when menu selection changes."""
        self.logger.debug(f"Changing mask generation method to index {index}")
        self.masks_stacked.setCurrentIndex(index)

    def _setup_sweep_control(self):
        self.logger.debug("Configuring SweepControlWidget")
        self.sweep_control = SweepControlWidget(
            parent=None,
            logger=self.logger
        )
        self.sweep_control.maskReady.connect(self.create_mask)
        self.logger.debug("SweepControlWidget created and signal maskReady connected")

    def _setup_scatter_control(self):
        self.logger.debug("Configuring ScatterControlWidget")
        self.scatter_control = ScatterControlWidget(
            parent=None,
            logger=self.logger
        )
        self.scatter_control.maskReady.connect(self.create_mask)
        self.logger.debug("ScatterControlWidget created and signal maskReady connected")

    def _setup_hadamard_controls(self):
        self.logger.debug("Configuring HadamardControlWidgets")
        self.hadamard_controls = []
        mask_classes = [
            MaskHadamard,
            MaskHadamardScramble,
            MaskHadamardCakeCutting,
            MaskHadamardWalshPaley,
            MaskCalSal,
        ]
        for mask_cls in mask_classes:
            w = HadamardControlWidget(
                parent=None,
                mask_cls=mask_cls,
                logger=self.logger
            )
            w.maskReady.connect(self.create_mask)
            self.hadamard_controls.append(w)
            self.logger.debug("%s created", mask_cls.__name__)

    def create_mask(self, mascara_obj):
        """Manages mask creation and worker for selected mask."""
        self.logger.info("Mask creation requested: %s", type(mascara_obj).__name__)
        # Evita workers concurrentes
        if getattr(self, 'mascara_thread', None) and self.mascara_thread.isRunning():
            self.logger.warning("Mask worker already running, ignoring new request")
            QMessageBox.warning(None, "Attention", "A mask is already being generated.")
            return

        # Configure in simulation
        self.simulation.set_mask(mascara_obj)
        self.logger.debug("Mask assigned in Simulation: %s", type(mascara_obj).__name__)

        # Lanza el worker
        self._start_worker(mascara_obj)

    def _start_worker(self, mascara):
        """Launches the MascaraWorker with progress and callbacks."""
        self.logger.debug("Starting MascaraWorker for %s", type(mascara).__name__)

        # Notify status manager that task is starting
        if self.status_manager:
            self.status_manager.start_task("Mask generation")

        self.mascara_worker = MascaraWorker(mascara, logger=self.logger)
        thread = WorkerLauncher.launch(
            self.mascara_worker,
            on_progress=self.visual_widget.set_progress,
            on_finished=self._on_mask_finished,
            on_error=lambda e: self._on_mask_error(e)
        )
        self.mascara_thread = thread
        thread.finished.connect(lambda: setattr(self, 'mascara_thread', None))
        thread.finished.connect(thread.deleteLater)
        self.logger.debug("MascaraWorker launched and references saved")

    def _on_mask_error(self, e):
        """Handle mask worker error."""
        self.logger.error("Error in MascaraWorker: %s", e, exc_info=True)
        QMessageBox.critical(None, "Error", str(e))
        if self.status_manager:
            self.status_manager.error_task(str(e)[:50])

    def _on_mask_finished(self):
        """Callback when worker finishes: emits mask_created signal."""
        self.logger.info("MascaraWorker finished, emitting mask_created")

        # Notify status manager that task is finished
        if self.status_manager:
            self.status_manager.finish_task()

        self.mask_created.emit(
            self.simulation.dataset,
            self.simulation.mask,
            self.simulation.applicator
        )

    def _on_dataset_changed(self, img_size: int):
        """Updates controls when dataset image size changes."""
        self.logger.debug("Dataset changed, new img_size=%d", img_size)
        self.sweep_control.set_img_size(img_size)
        self.scatter_control.set_img_size(img_size)
        max_pat = img_size * img_size
        for ctrl in self.hadamard_controls:
            ctrl.set_img_size(img_size)
            ctrl.number_patterns_max_hadamard_value.setText(str(max_pat))
            # Use set_range to properly update slider and emit signals
            ctrl.hadamard_slider.set_range(0, max_pat)
        self.logger.debug("Controls updated for img_size=%d", img_size)

    def get_content_widget(self):
        """Return the main container widget for stepper integration."""
        return self.ui.masks_main_container
