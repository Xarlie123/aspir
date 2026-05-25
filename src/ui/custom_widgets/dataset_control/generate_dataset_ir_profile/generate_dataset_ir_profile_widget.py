# File: ui/custom_widgets/genera_dataset_perfil_ir_widget.py

import logging
from PySide6 import QtWidgets
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QMessageBox
from ui.custom_widgets.dataset_control.generate_dataset_ir_profile.ui_generate_dataset_ir_profile_widget import Ui_Generates_dataset_perfil_ir
from ui.custom_widgets.common.data_format_selector import DataFormatSelector
from ui.custom_widgets.common.mode_distribution_widget import ModeDistributionWidget
from ui.custom_widgets.common.button_styles import BUTTON_STYLE_GREEN, apply_button_style
from simulation_engine._1_dataset_gen.DatasetFromIRBeam import DatasetFromIRBeam


class GeneratesDatasetPerfilIR(QtWidgets.QWidget, Ui_Generates_dataset_perfil_ir):
    """
    Widget to configure and generate an IR beam profile dataset.

    Features:
    - Configure image dimension, dataset size, random seed
    - Select data format (FP32, INT8, INT4)
    - Configure beam mode distribution (Gaussian, Hermite-Gauss, Laguerre-Gauss, Doughnut)
    - Add speckle noise to simulate IR sensor behavior
    - Visualize mode distribution with pie chart
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
        self.logger.debug("Initializing GeneratesDatasetPerfilIR")

        # Add data format selector to the form layout
        self.data_format_selector = DataFormatSelector(logger=self.logger)
        self.formLayout.addRow("Data Format:", self.data_format_selector)

        # Add mode distribution widget
        self.mode_distribution_widget = ModeDistributionWidget(logger=self.logger)
        self.main_layout.insertWidget(self.main_layout.count() - 1, self.mode_distribution_widget)

        # Connect and style the generate button
        self.generate_dataset_button.clicked.connect(self._on_generate_dataset)
        apply_button_style(self.generate_dataset_button, BUTTON_STYLE_GREEN)

    def _on_generate_dataset(self):
        """
        Called when 'Generate' button is pressed: builds and emits DatasetFromIRBeam.
        """
        self.logger.debug("Generate IR dataset button pressed")

        try:
            img_size = int(self.image_dimension_value.currentText())
            self.logger.info("Image size selected: %d", img_size)
        except Exception as e:
            self.logger.error("Error reading image size: %s", e, exc_info=True)
            QMessageBox.warning(self, "Error", "Please select a valid image size.")
            return

        num_images = self.dataset_size_value.value()
        seed = self.random_seed_value.value()

        # Get selected data format
        data_format = self.data_format_selector.get_format()

        # Get mode distribution settings
        mode_settings = self.mode_distribution_widget.get_all_settings()
        mode_distribution = mode_settings["mode_distribution"]
        speckle_noise = mode_settings["speckle_noise"]
        max_mode_order = mode_settings["max_mode_order"]

        # Validate mode distribution
        total_pct = sum(mode_distribution.values())
        if total_pct == 0:
            QMessageBox.warning(self, "Warning",
                "No beam modes selected. Please select at least one mode.")
            return

        self.logger.info(
            "IRBeam parameters -> img_size: %d, num_images: %d, seed: %d, format: %s",
            img_size, num_images, seed, data_format
        )
        self.logger.info(
            "Mode distribution: %s, noise: %.2f, max_order: %d",
            mode_distribution, speckle_noise, max_mode_order
        )

        # Instantiate DatasetFromIRBeam with all parameters
        ds = DatasetFromIRBeam(
            name="IRBeam",
            img_size=img_size,
            num_images=num_images,
            seed=seed,
            logger=self.logger,
            data_format=data_format,
            mode_distribution=mode_distribution,
            speckle_noise=speckle_noise,
            max_mode_order=max_mode_order
        )
        self.logger.debug("DatasetFromIRBeam instantiated: %s with format %s", ds.name, data_format)

        # Emit for the handler to process
        self.logger.info("Emitting datasetReady with %s", ds.name)
        self.datasetReady.emit(ds)

    def get_settings(self) -> dict:
        """Get all current settings as a dictionary for saving."""
        return {
            "img_size": self.image_dimension_value.currentText(),
            "num_images": self.dataset_size_value.value(),
            "seed": self.random_seed_value.value(),
            "data_format": self.data_format_selector.get_format(),
            "mode_settings": self.mode_distribution_widget.get_all_settings(),
        }

    def set_settings(self, settings: dict):
        """Restore settings from a dictionary."""
        if "img_size" in settings:
            idx = self.image_dimension_value.findText(str(settings["img_size"]))
            if idx >= 0:
                self.image_dimension_value.setCurrentIndex(idx)
        if "num_images" in settings:
            self.dataset_size_value.setValue(settings["num_images"])
        if "seed" in settings:
            self.random_seed_value.setValue(settings["seed"])
        if "data_format" in settings:
            self.data_format_selector.set_format(settings["data_format"])
        if "mode_settings" in settings:
            self.mode_distribution_widget.set_all_settings(settings["mode_settings"])
