import numpy as np
from PyQt5.QtWidgets import (QMessageBox, QListWidget, QStackedWidget, QLabel,
                               QVBoxLayout, QWidget, QScrollArea, QSizePolicy, QFrame)
from PyQt5.QtCore import pyqtSignal, QObject, Qt
from PyQt5.QtGui import QFont

from ui._1_dataset.dataset_worker import DatasetWorker
from ui.custom_widgets.dataset_control.select_image.select_image_widget import SeleccionaImagenWidget
from ui.custom_widgets.dataset_control.select_image_directory.select_image_directory_widget import SeleccionaCarpetaImagenWidget
from ui.custom_widgets.dataset_control.generate_dataset_ir_profile.generate_dataset_ir_profile_widget import GeneratesDatasetPerfilIR
from ui.custom_widgets.dataset_control.generate_dataset_internet.generate_dataset_from_internet_widget import GeneratesDatasetInternetWidget
from ui.custom_widgets.visualizers.visual_dataset.visual_dataset_widget import VisualDatasetWidget
# from ui.utils.widget_helpers import embed_widget  # No longer needed
from ui.utils.worker_launcher import WorkerLauncher

class UIDatasetHandler(QObject):
    dataset_updated = pyqtSignal(int)  # notify new img_size

    def __init__(self, ui, simulation, logger, ui_test_mask=None, ui_mask=None, status_manager=None):
        super().__init__()
        self.ui = ui
        self.simulation = simulation
        self.logger = logger.getChild("UIDatasetHandler")
        self.logger.debug("Initializing UIDatasetHandler")
        self.ui_test_mask = ui_test_mask
        self.ui_mask = ui_mask
        self.status_manager = status_manager

        # Internal flag to prevent duplicate runs
        self._dataset_running = False
        self.worker = None

        # 1) Selector de archivo
        self.logger.debug("Configuring SeleccionaImagenWidget")
        self.selecciona_imagen_widget = SeleccionaImagenWidget(logger=self.logger)
        self.selecciona_imagen_widget.datasetReady.connect(self.create_dataset)

        # 2) Selector de carpeta
        self.logger.debug("Configuring SeleccionaCarpetaImagenWidget")
        self.selecciona_directorio_imagen_widget = SeleccionaCarpetaImagenWidget(logger=self.logger)
        self.selecciona_directorio_imagen_widget.datasetReady.connect(self.create_dataset)

        # 3) Internet dataset generator (SVHN / Celebrities)
        self.logger.debug("Configuring GeneratesDatasetInternetWidget (SVHN/Celebrities)")
        self.internet_widget = GeneratesDatasetInternetWidget(logger=self.logger)
        # When the internet widget emits a dataset, run the common creation pipeline
        self.internet_widget.datasetReady.connect(self.create_dataset)

        # 4) Generator IR beam
        self.logger.debug("Configuring GeneratesDatasetPerfilIR")
        self.ir_widget = GeneratesDatasetPerfilIR(logger=self.logger)
        self.ir_widget.datasetReady.connect(self.create_dataset)

        # 5) Create VisualDatasetWidget (will be embedded in _setup_menu_interface)
        self.logger.debug("Configuring VisualDatasetWidget")
        self.visual_widget = VisualDatasetWidget(logger=self.logger)

        # 6) Setup menu-based interface
        self.logger.debug("Setting up menu-based interface")
        self._setup_menu_interface()

        # 7) When the dataset changes, update the visualizer
        self.dataset_updated.connect(
            lambda size: self.visual_widget.set_data(
                self.simulation.dataset.data,
                getattr(self.simulation.dataset, 'data_format', None)
            )
        )

    def _setup_menu_interface(self):
        """Setup menu-based interface with QListWidget and QStackedWidget."""
        self.logger.debug("Creating QListWidget menu")

        # Create QListWidget for menu
        self.dataset_menu = QListWidget()
        self.dataset_menu.addItems([
            "Generate IR Profile",
            "Load Single Image",
            "Load Multiple Images",
            "Download from Internet"
        ])
        self.dataset_menu.setCurrentRow(0)
        self.dataset_menu.currentRowChanged.connect(self._on_menu_selection_changed)

        # Style the menu
        self.dataset_menu.setStyleSheet("""
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
        self.logger.debug("Creating QStackedWidget for content")
        self.dataset_stacked = QStackedWidget()

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

        # Create pages for each dataset type (order matches menu)
        page0 = create_page("Create IR Beam Profile Dataset", self.ir_widget)
        self.dataset_stacked.addWidget(page0)
        self.logger.debug("Added page 0: Generate IR Profile")

        page1 = create_page("Load a Single Image", self.selecciona_imagen_widget)
        self.dataset_stacked.addWidget(page1)
        self.logger.debug("Added page 1: Load Single Image")

        page2 = create_page("Load Images from Folder", self.selecciona_directorio_imagen_widget)
        self.dataset_stacked.addWidget(page2)
        self.logger.debug("Added page 2: Load Multiple Images")

        page3 = create_page("Create Dataset from Internet", self.internet_widget)
        self.dataset_stacked.addWidget(page3)
        self.logger.debug("Added page 3: Download from Internet")

        # Embed menu into placeholder
        self.logger.debug("Embedding menu into dataset_menu_placeholder")
        menu_layout = QVBoxLayout(self.ui.dataset_menu_placeholder)
        menu_layout.setContentsMargins(0, 0, 0, 0)
        menu_layout.addWidget(self.dataset_menu)

        # Embed stacked widget into content placeholder with scroll area
        self.logger.debug("Embedding stacked widget into dataset_content_placeholder")
        content_layout = QVBoxLayout(self.ui.dataset_content_placeholder)
        content_layout.setContentsMargins(0, 0, 0, 0)

        # Wrap stacked widget in scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setWidget(self.dataset_stacked)
        content_layout.addWidget(scroll_area)

        # Embed preview widget into preview placeholder
        self.logger.debug("Embedding preview widget into dataset_preview_placeholder")

        # Style the preview placeholder with same panel style
        self.ui.dataset_preview_placeholder.setStyleSheet("""
            QWidget#dataset_preview_placeholder {
                background-color: #f5f5f5;
                border: 1px solid #ccc;
                border-radius: 4px;
            }
        """)
        self.ui.dataset_preview_placeholder.setObjectName("dataset_preview_placeholder")

        preview_layout = QVBoxLayout(self.ui.dataset_preview_placeholder)
        preview_layout.setContentsMargins(10, 10, 10, 10)

        preview_label = QLabel("<h3>Dataset preview:</h3>")
        preview_label.setAlignment(Qt.AlignCenter)
        preview_layout.addWidget(preview_label)

        # Ensure visual widget expands
        self.visual_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        preview_layout.addWidget(self.visual_widget, 1)

        # Explicitly set the first page as current
        self.dataset_stacked.setCurrentIndex(0)
        self.logger.debug("Set initial stacked widget page to index 0")

        self.logger.info("Menu-based interface setup complete")

    def _on_menu_selection_changed(self, index):
        """Switch stacked widget page when menu selection changes."""
        self.logger.debug(f"Changing dataset method to index {index}")
        self.dataset_stacked.setCurrentIndex(index)

    def create_dataset(self, dataset):
        """
        Launches dataset generation in a worker thread.
        """
        if self._dataset_running:
            self.logger.warning("Dataset generation already in progress, ignoring request")
            QMessageBox.warning(None, "Attention", "A dataset is already being generated.")
            return

        self.logger.info("Starting dataset creation: %s", getattr(dataset, 'name', str(dataset)))
        self._dataset_running = True

        # Notify status manager that task is starting
        if self.status_manager:
            self.status_manager.start_task("Dataset generation")

        # Configure the simulation
        self.simulation.set_dataset(dataset)

        # Configure and launch the worker
        self.worker = DatasetWorker(dataset, logger=self.logger)
        self.thread = WorkerLauncher.launch(
            self.worker,
            on_progress=self.visual_widget.set_progress,
            on_finished=self._handle_finished,
            on_error=lambda e: self._handle_error(e)
        )
        self.thread.finished.connect(lambda: setattr(self, 'thread', None))

    def _handle_error(self, e):
        """Handle worker error."""
        self.logger.error("Error generating dataset: %s", e, exc_info=True)
        QMessageBox.critical(None, "Error", str(e))
        self._dataset_running = False
        if self.status_manager:
            self.status_manager.error_task(str(e)[:50])

    def _handle_finished(self):
        """
        Callback when the worker finishes.
        """
        self.logger.info("Dataset worker finished")
        self._dataset_running = False

        # Notify status manager that task is finished
        if self.status_manager:
            self.status_manager.finish_task()

        self._on_dataset_finished()

    def _on_dataset_finished(self):
        """
        Process the dataset once generated: update UI and emit signal.
        """
        ds = self.simulation.dataset
        self.logger.info("Dataset '%s' ready with %d images", ds.name, len(ds.data))

        # Update general info in UI
        self.visual_widget.dataset_size_info_value.setText(str(len(ds.data)))
        self.visual_widget.image_dimension_info_value.setText(f"{ds.img_size}×{ds.img_size}")
        self.visual_widget.dataset_type_info_value.setText(ds.dataset_type)

        # Emit update signal
        self.logger.debug("Emitting dataset_updated with size %d", ds.img_size)
        self.dataset_updated.emit(ds.img_size)

    def get_content_widget(self):
        """Return the main container widget for stepper integration."""
        return self.ui.dataset_main_container
