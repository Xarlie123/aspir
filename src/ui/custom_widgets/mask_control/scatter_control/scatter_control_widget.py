# File: ui/custom_widgets/mascara_control/scatter_control/scatter_control_widget.py
import logging
from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QMessageBox
from ui.custom_widgets.mask_control.scatter_control.ui_scatter_control import Ui_Scatter_Control
from ui.custom_widgets.common.button_styles import BUTTON_STYLE_GREEN, apply_button_style
from simulation_engine._2_mask_gen.mask_scatter import MaskScatter

class ScatterControlWidget(QtWidgets.QWidget, Ui_Scatter_Control):
    """
    Custom widget to configure and generate a scatter mask, with logging.
    Emits:
        maskReady(mask: MaskScatter) when the mask is created successfully.
    """
    maskReady = pyqtSignal(object)

    def __init__(self, parent=None, logger=None):
        super().__init__(parent)
        self.setupUi(self)
        # Initialize logger
        if logger is None:
            self.logger = logging.getLogger(self.__class__.__name__)
        else:
            self.logger = logger.getChild(self.__class__.__name__)
        self.logger.debug("Initializing ScatterControlWidget")

        # internal image size, set externally via set_img_size
        self._img_size = None

        # connect internal "Generate" button
        self.generate_masks_button.clicked.connect(self._on_generate_clicked)
        apply_button_style(self.generate_masks_button, BUTTON_STYLE_GREEN)
        self.logger.debug("Connected generate_masks_button to _on_generate_clicked")

        # allow widget to use its preferred size
        self.setSizePolicy(QtWidgets.QSizePolicy.Preferred,
                           QtWidgets.QSizePolicy.Preferred)
        self.adjustSize()

    def set_img_size(self, img_size: int):
        """Set the image size (dimension) for mask creation."""
        self._img_size = img_size
        # Default number of patterns to total pixels (img_size²)
        num_pixels = img_size * img_size
        # Update maximum first, then set value (order matters for Qt spinbox)
        self.number_patterns_scatter_value.setMaximum(num_pixels)
        self.number_patterns_scatter_value.setValue(num_pixels)
        self.logger.info("Image size set to %d, default patterns=%d", img_size, num_pixels)

    def _on_generate_clicked(self):
        """Slot called when user clicks the generate-mask button."""
        self.logger.debug("_on_generate_clicked called; img_size=%s", self._img_size)
        if self._img_size is None:
            QMessageBox.warning(self, "Error", "Create a dataset first.")
            self.logger.warning("Attempt to generate mask without img_size set")
            return

        # read parameters from UI controls
        d = self.point_density_value.value()
        n = self.number_patterns_scatter_value.value()
        s = self.random_seed_scatter_value.value()
        self.logger.info("Generating MaskScatter with img_size=%d, density=%d%%, patterns=%d, seed=%d",
                         self._img_size, d, n, s)

        # attempt to build the mask
        try:
            mask = MaskScatter(self._img_size, d, n, s)
            mask.applicator_type_scatter = self.select_applicator_scatter_list.currentText()
            self.logger.info("MaskScatter created: %s", type(mask).__name__)
        except Exception as e:
            QMessageBox.critical(self, "Error creating scatter mask", str(e))
            self.logger.error("Error creating MaskScatter: %s", e, exc_info=True)
            return

        # emit the ready signal
        self.maskReady.emit(mask)
        self.logger.debug("maskReady emitted with %s", type(mask).__name__)
