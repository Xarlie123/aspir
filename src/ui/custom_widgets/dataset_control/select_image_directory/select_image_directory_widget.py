# File: ui/custom_widgets/selecciona_directorio_imagen_widget.py

import os
import logging
from PySide6 import QtWidgets
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QMessageBox, QFileDialog
from PySide6.QtGui import QImageReader
from ui.custom_widgets.dataset_control.select_image_directory.ui_select_image_directory_widget import Ui_Selecciona_directorio_imagen
from ui.custom_widgets.common.data_format_selector import DataFormatSelector
from ui.custom_widgets.common.speckle_noise_widget import SpeckleNoiseWidget
from ui.custom_widgets.common.button_styles import BUTTON_STYLE_GREEN, BUTTON_STYLE_BLUE, apply_button_style
from simulation_engine._1_dataset_gen.DatasetFromFolder import DatasetFromFolder

class SeleccionaCarpetaImagenWidget(QtWidgets.QWidget, Ui_Selecciona_directorio_imagen):
    """
    Widget para seleccionar un directorio de imágenes
    y emitir un DatasetFromFolder listo para generar.
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
        self.logger.debug("Initializing SeleccionaCarpetaImagenWidget")

        # Add data format selector
        self.data_format_selector = DataFormatSelector(logger=self.logger)
        # Insert before the generate button
        self.main_layout.insertWidget(self.main_layout.count() - 1, self.data_format_selector)

        # Add speckle noise widget (default 0 = disabled)
        self.speckle_noise_widget = SpeckleNoiseWidget(logger=self.logger)
        self.main_layout.insertWidget(self.main_layout.count() - 1, self.speckle_noise_widget)

        # Conectar botones
        self.select_folder_button.clicked.connect(self._select_and_validate_directory)
        self.generate_dataset_button.clicked.connect(self._on_generate_dataset)

        # Apply button styles
        apply_button_style(self.select_folder_button, BUTTON_STYLE_BLUE)
        apply_button_style(self.generate_dataset_button, BUTTON_STYLE_GREEN)

        # Directorio seleccionado válido
        self._dir_path: str = ""

        # Permitir tamaño preferido
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Preferred
        )
        self.adjustSize()

    def _select_and_validate_directory(self):
        """
        Opens folder dialog, validates and stores the path.
        """
        self.logger.debug("Opening dialog to select image folder")
        # Start file dialog from datasets folder
        datasets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))), "datasets")
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select image directory", datasets_dir
        )
        if not dir_path:
            self.logger.debug("Selección de carpeta cancelada")
            return

        self.logger.info("Carpeta seleccionada: %s", dir_path)
        self.folder_path_input.setText(dir_path)

        if not os.path.isdir(dir_path):
            self.logger.error("Ruta no válida: %s", dir_path)
            QMessageBox.critical(self, "Error", f"Directorio no válido:\n{dir_path}")
            self._dir_path = ""
            return

        self._dir_path = dir_path
        self.logger.debug("Directorio válido almacenado: %s", dir_path)

    def _on_generate_dataset(self):
        """
        Construye y emite DatasetFromFolder al pulsar 'Generatesr'.
        """
        self.logger.debug("Button Generatesr pulsado")
        if not self._dir_path:
            self.logger.warning("Intento sin carpeta válida")
            QMessageBox.warning(
                self, "Advertencia",
                "Primero selecciona un directorio de imágenes válido."
            )
            return

        # Listar archivos de imagen
        files = sorted(
            f for f in os.listdir(self._dir_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
        )
        if not files:
            self.logger.warning("No se encontraron imágenes en %s", self._dir_path)
            QMessageBox.warning(
                self, "Advertencia",
                "No se encontraron archivos de imagen en el directorio."
            )
            return

        # Leer primer archivo para tamaño
        first_path = os.path.join(self._dir_path, files[0])
        reader = QImageReader(first_path)
        image = reader.read()
        if image is None or image.isNull():
            self.logger.error("Error cargando imagen de prueba: %s", first_path)
            QMessageBox.critical(
                self, "Error",
                f"No se pudo cargar la imagen:\n{first_path}"
            )
            return

        if image.width() != image.height():
            self.logger.error(
                "Imagen inicial no cuadrada: %dx%d",
                image.width(), image.height()
            )
            QMessageBox.critical(
                self, "Error",
                "Las imágenes deben ser cuadradas y de tamaño uniforme."
            )
            return

        img_size = image.width()
        self.logger.info("Size detected: %d", img_size)

        # Get selected data format
        data_format = self.data_format_selector.get_format()
        self.logger.info("Data format selected: %s", data_format)

        # Get speckle noise level
        speckle_noise = self.speckle_noise_widget.get_value()

        # Crear y emitir el dataset
        ds = DatasetFromFolder(img_size, self._dir_path, logger=self.logger,
                              data_format=data_format, speckle_noise=speckle_noise)
        self.logger.info("Emitting datasetReady para %s with format %s, speckle %.2f",
                        ds.name, data_format, speckle_noise)
        self.datasetReady.emit(ds)
