# File: ui/custom_widgets/common/speckle_noise_widget.py
"""
Simple widget for configuring speckle noise level.
Reusable across different dataset types.
"""

import logging
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QLabel, QSlider, QDoubleSpinBox
)
from PyQt5.QtCore import Qt, pyqtSignal


class SpeckleNoiseWidget(QWidget):
    """
    Widget for configuring speckle noise level (0.0 to 1.0).
    Contains a slider and spinbox for precise control.
    Default value is 0 (no noise).
    """
    valueChanged = pyqtSignal(float)  # Emits noise level when changed

    def __init__(self, parent=None, logger=None):
        super().__init__(parent)

        if logger is None:
            self.logger = logging.getLogger(self.__class__.__name__)
        else:
            self.logger = logger.getChild(self.__class__.__name__)

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Setup the widget UI."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 5, 0, 5)
        layout.setSpacing(10)

        # Label
        label = QLabel("Speckle Noise:")
        label.setToolTip("Add speckle noise to simulate sensor behavior (0 = no noise)")
        layout.addWidget(label)

        # Slider (0-100, maps to 0.0-1.0)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(0)
        self.slider.setTickPosition(QSlider.NoTicks)
        layout.addWidget(self.slider, 1)

        # Spinbox (0.0-1.0 with 0.05 step)
        self.spinbox = QDoubleSpinBox()
        self.spinbox.setRange(0.0, 1.0)
        self.spinbox.setSingleStep(0.05)
        self.spinbox.setDecimals(2)
        self.spinbox.setValue(0.0)
        self.spinbox.setFixedWidth(70)
        self.spinbox.setToolTip("Noise level: 0.0 = none, 1.0 = maximum")
        layout.addWidget(self.spinbox)

    def _connect_signals(self):
        """Connect internal signals."""
        self.slider.valueChanged.connect(self._on_slider_changed)
        self.spinbox.valueChanged.connect(self._on_spinbox_changed)

    def _on_slider_changed(self, value: int):
        """Handle slider change."""
        float_value = value / 100.0
        self.spinbox.blockSignals(True)
        self.spinbox.setValue(float_value)
        self.spinbox.blockSignals(False)
        self.valueChanged.emit(float_value)

    def _on_spinbox_changed(self, value: float):
        """Handle spinbox change."""
        int_value = int(value * 100)
        self.slider.blockSignals(True)
        self.slider.setValue(int_value)
        self.slider.blockSignals(False)
        self.valueChanged.emit(value)

    def get_value(self) -> float:
        """Get the current speckle noise level (0.0 to 1.0)."""
        return self.spinbox.value()

    def set_value(self, value: float):
        """Set the speckle noise level (0.0 to 1.0)."""
        value = max(0.0, min(1.0, value))
        self.spinbox.blockSignals(True)
        self.slider.blockSignals(True)
        self.spinbox.setValue(value)
        self.slider.setValue(int(value * 100))
        self.spinbox.blockSignals(False)
        self.slider.blockSignals(False)
