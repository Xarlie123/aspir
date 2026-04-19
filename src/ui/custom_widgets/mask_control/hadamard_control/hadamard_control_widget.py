import logging
from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import QSizePolicy, QMessageBox, QHBoxLayout, QLabel, QDoubleSpinBox
from ui.custom_widgets.mask_control.hadamard_control.ui_hadamard_control import Ui_Hadamard_Control
from ui.custom_widgets.mask_control.hadamard_control.qrange_slider import QRangeSlider
from ui.custom_widgets.common.button_styles import BUTTON_STYLE_GREEN, apply_button_style


class HadamardControlWidget(QtWidgets.QWidget, Ui_Hadamard_Control):
    """
    Generic Hadamard control widget with logging.
    Parameterized by a mask class (e.g. MaskHadamard or MaskHadamardCakeCutting).
    Emits maskReady(mask) when a mask object is created.

    Features:
    - Range slider for selecting Hadamard pattern indices
    - Percentage spinbox to select by percentage of total patterns
    - Click on slider track (between handles) drags the entire range
    """
    maskReady = pyqtSignal(object)

    def __init__(self, parent=None, mask_cls=None, logger=None):
        super().__init__(parent)
        self.setupUi(self)
        # Initialize logger
        if logger is None:
            self.logger = logging.getLogger(self.__class__.__name__)
        else:
            self.logger = logger.getChild(self.__class__.__name__)
        self.logger.debug("Initializing HadamardControlWidget for %s", mask_cls.__name__ if mask_cls else 'None')

        self._mask_cls = mask_cls
        self._mask = None
        self._img_size = None
        self._updating_percentage = False  # Prevent signal loops

        # 1) Create QRangeSlider
        old_slider = self.hadamard_slider
        range_label = self.range_patterns_hadamard_value
        max_label = self.number_patterns_max_hadamard_value
        try:
            max_val = int(max_label.text())
        except ValueError:
            max_val = old_slider.maximum()
        self.logger.debug("Configuring QRangeSlider with max_val=%d", max_val)
        new_slider = QRangeSlider(min_val=0, max_val=max_val, value_label=range_label)

        # 2) Replace in the vertical layout from Designer
        self.main_layout.replaceWidget(old_slider, new_slider)

        # 3) Delete the old slider and update the reference
        old_slider.deleteLater()
        self.hadamard_slider = new_slider
        max_label.setText(str(max_val))

        # 4) Add percentage spinbox row
        self._add_percentage_controls()

        # 5) Connect slider percentage signal to spinbox
        self.hadamard_slider.percentageChanged.connect(self._on_slider_percentage_changed)

        # 6) Connect and style the button
        self.generate_masks_button.clicked.connect(self._on_generate_mask)
        apply_button_style(self.generate_masks_button, BUTTON_STYLE_GREEN)
        self.logger.debug("Button generate_masks_button connected to slot _on_generate_mask")

        # 7) Let the layout measure the widget properly
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.adjustSize()

        # 8) Initialize percentage display
        self._on_slider_percentage_changed(self.hadamard_slider.get_percentage())

    def _add_percentage_controls(self):
        """Add percentage spinbox and selected count display."""
        # Create horizontal layout for percentage controls
        pct_layout = QHBoxLayout()
        pct_layout.setSpacing(10)

        # Percentage label and spinbox
        pct_label = QLabel("Percentage:")
        pct_label.setStyleSheet("font-size: 11px;")
        pct_layout.addWidget(pct_label)

        self.percentage_spinbox = QDoubleSpinBox()
        self.percentage_spinbox.setRange(0.1, 100.0)
        self.percentage_spinbox.setDecimals(1)
        self.percentage_spinbox.setSingleStep(5.0)
        self.percentage_spinbox.setValue(100.0)  # Default 100% (all patterns)
        self.percentage_spinbox.setSuffix(" %")
        self.percentage_spinbox.setFixedWidth(80)
        self.percentage_spinbox.setToolTip("Set the percentage of total patterns to select")
        self.percentage_spinbox.valueChanged.connect(self._on_percentage_spinbox_changed)
        pct_layout.addWidget(self.percentage_spinbox)

        # Selected count label
        self.selected_count_label = QLabel("(0 / 0 patterns)")
        self.selected_count_label.setStyleSheet("font-size: 11px; color: #666;")
        pct_layout.addWidget(self.selected_count_label)

        pct_layout.addStretch()

        # Insert before the generate button
        button_index = self.main_layout.indexOf(self.generate_masks_button)
        self.main_layout.insertLayout(button_index, pct_layout)

    def _on_slider_percentage_changed(self, percentage: float):
        """Update spinbox when slider changes."""
        if self._updating_percentage:
            return
        self._updating_percentage = True
        self.percentage_spinbox.setValue(percentage)
        self._update_count_label()
        self._updating_percentage = False

    def _on_percentage_spinbox_changed(self, percentage: float):
        """Update slider when spinbox changes."""
        if self._updating_percentage:
            return
        self._updating_percentage = True
        self.hadamard_slider.set_percentage(percentage)
        self._update_count_label()
        self._updating_percentage = False

    def _update_count_label(self):
        """Update the selected/total patterns count label."""
        selected = self.hadamard_slider.get_selected_count()
        total = self.hadamard_slider.get_total_count()
        self.selected_count_label.setText(f"({selected} / {total} patterns)")

    def set_img_size(self, img_size: int):
        """Set the image size for mask instantiation."""
        self._img_size = img_size
        self.logger.info("Image size set to %d", img_size)

    def _on_generate_mask(self):
        """
        Slot to generate the mask when the button is clicked.
        """
        self.logger.debug("_on_generate_mask called; img_size=%s", self._img_size)
        if self._img_size is None:
            QMessageBox.warning(self, "Error", "Create a dataset first.")
            self.logger.warning("Attempted to generate mask without img_size set")
            return
        low = self.hadamard_slider.low_value
        high = self.hadamard_slider.high_value
        self.logger.info("Generating mask with range [%d, %d]", low, high)
        try:
            mask_obj = self._mask_cls(self._img_size, low, high, logger=self.logger)
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))
            self.logger.error("Error instantiating mask: %s", e)
            return
        self._mask = mask_obj
        self.logger.info("Mask generated: %s", type(mask_obj).__name__)
        self.maskReady.emit(mask_obj)

    @property
    def mask(self):
        """Last generated mask object."""
        return self._mask
