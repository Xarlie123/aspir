# File: ui/custom_widgets/dataset_control/select_image/select_image_widget.py

import os
import logging
from PySide6 import QtWidgets
from PySide6.QtCore import Signal
from PySide6.QtGui import QImageReader, QImage
from PySide6.QtWidgets import QMessageBox, QFileDialog
from ui.custom_widgets.dataset_control.select_image.ui_select_image_widget import Ui_Selecciona_imagen
from ui.custom_widgets.common.data_format_selector import DataFormatSelector
from ui.custom_widgets.common.speckle_noise_widget import SpeckleNoiseWidget
from ui.custom_widgets.common.button_styles import BUTTON_STYLE_GREEN, BUTTON_STYLE_BLUE, apply_button_style
from simulation_engine._1_dataset_gen.DatasetFromImage import DatasetFromImage

class SeleccionaImagenWidget(QtWidgets.QWidget, Ui_Selecciona_imagen):
    """
    Widget to select, validate an image,
    and emit a DatasetFromImage ready for generation.
    """
    datasetReady = Signal(object)

    def __init__(self, parent=None, logger=None):
        super().__init__(parent)
        self.setupUi(self)

        # Logger
        if logger is None:
            self.logger = logging.getLogger(self.__class__.__name__)
        else:
            self.logger = logger.getChild(self.__class__.__name__)
        self.logger.debug("Initializing SeleccionaImagenWidget")

        # Add data format selector
        self.data_format_selector = DataFormatSelector(logger=self.logger)
        # Insert before the generate button
        self.main_layout.insertWidget(self.main_layout.count() - 1, self.data_format_selector)

        # Add speckle noise widget (default 0 = disabled)
        self.speckle_noise_widget = SpeckleNoiseWidget(logger=self.logger)
        self.main_layout.insertWidget(self.main_layout.count() - 1, self.speckle_noise_widget)

        # Connect internal buttons
        self.select_image_button.clicked.connect(self._select_and_validate)
        self.generate_dataset_button.clicked.connect(self._on_generate_dataset)

        # Apply button styles
        apply_button_style(self.select_image_button, BUTTON_STYLE_BLUE)
        apply_button_style(self.generate_dataset_button, BUTTON_STYLE_GREEN)

        # Storage for the last validated image
        self._image: QImage | None = None
        self._image_path: str = ""

        # Use preferred size from Designer
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Preferred
        )
        self.adjustSize()

    def _select_and_validate(self):
        """
        Opens file dialog, loads and validates the image.
        """
        self.logger.debug("Opening dialog to select image")
        # Start file dialog from datasets folder
        import os
        datasets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))), "datasets")
        path, _ = QFileDialog.getOpenFileName(
            self, "Select image file", datasets_dir,
            "Image files (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        if not path:
            self.logger.debug("Selection cancelled by user")
            return

        self.logger.info("Path selected: %s", path)
        self.image_path_input.setText(path)
        image = self._load_and_validate_image(path)
        if image is None:
            self.logger.warning("Invalid image: %s", path)
            self._image = None
            self._image_path = ""
            return

        # Store valid image for later generation
        self.logger.info("Image validated successfully: %s", path)
        self._image = image
        self._image_path = path

    def _on_generate_dataset(self):
        """
        Called on 'Generate dataset' button click: builds and emits DatasetFromImage.
        """
        self.logger.debug("Generate dataset button pressed")
        ds = self.dataset
        if ds is None:
            self.logger.warning("Attempted to generate dataset without valid image")
            QMessageBox.warning(
                self, "Warning",
                "Please select and validate an image first."
            )
            return

        self.logger.info("Emitting datasetReady for %s", ds.name)
        self.datasetReady.emit(ds)

    def _load_and_validate_image(self, path: str) -> QImage | None:
        """
        Try to load and validate the image.
        Shows QMessageBox on failure.
        """
        self.logger.debug("Validating image at path: %s", path)
        if not os.path.isfile(path):
            self.logger.error("File not found: %s", path)
            QMessageBox.critical(self, "Error", f"File not found:\n{path}")
            return None

        reader = QImageReader(path)
        if reader.format() == b'':
            self.logger.error("Unsupported format: %s", path)
            QMessageBox.critical(self, "Error", "Unsupported image format.")
            return None

        image = reader.read()
        if image is None or image.isNull():
            self.logger.error("Failed to load image or invalid image: %s", path)
            QMessageBox.critical(
                self, "Error",
                "Could not load image or image is corrupted."
            )
            return None

        if image.width() != image.height():
            self.logger.warning("Image is not square: %dx%d", image.width(), image.height())
            QMessageBox.critical(
                self, "Error",
                "Image must be square."
            )
            return None

        return image

    @property
    def dataset(self) -> DatasetFromImage | None:
        """
        Build and return a DatasetFromImage
        from the last validated image.
        """
        if self._image is None or not self._image_path:
            return None
        data_format = self.data_format_selector.get_format()
        speckle_noise = self.speckle_noise_widget.get_value()
        self.logger.debug("Creating DatasetFromImage with format: %s, speckle: %.2f",
                         data_format, speckle_noise)
        return DatasetFromImage(
            self._image.width(),
            self._image_path,
            logger=self.logger,
            data_format=data_format,
            speckle_noise=speckle_noise
        )
