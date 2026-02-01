# File: ui/custom_widgets/common/mode_distribution_widget.py
"""
Widget for configuring and visualizing beam mode distribution.
Includes sliders for each mode percentage and a pie chart visualization.
"""

import logging
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                             QLabel, QSlider, QSpinBox, QGroupBox, QDoubleSpinBox)
import numpy as np


class PieChartWidget(QWidget):
    """
    Custom widget that draws a pie chart showing mode distribution.
    """
    # Colors for each mode type
    MODE_COLORS = {
        "gaussian": QtGui.QColor(65, 105, 225),       # Royal Blue
        "hermite_gauss": QtGui.QColor(50, 205, 50),   # Lime Green
        "laguerre_gauss": QtGui.QColor(255, 165, 0),  # Orange
        "doughnut": QtGui.QColor(220, 20, 60),        # Crimson
    }

    MODE_LABELS = {
        "gaussian": "Gaussian",
        "hermite_gauss": "Hermite-Gauss",
        "laguerre_gauss": "Laguerre-Gauss",
        "doughnut": "Doughnut",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(150, 150)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self._distribution = {}

    def set_distribution(self, distribution: dict):
        """Set the mode distribution to display."""
        self._distribution = distribution
        self.update()

    def paintEvent(self, event):
        """Draw the pie chart."""
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        # Calculate the square area for the pie chart
        rect = self.rect()
        size = min(rect.width(), rect.height()) - 20  # Margin
        x = (rect.width() - size) // 2
        y = (rect.height() - size) // 2
        pie_rect = QtCore.QRectF(x, y, size, size)

        # Draw background
        painter.fillRect(rect, QtGui.QColor(245, 245, 245))

        # Calculate total and filter out zero values
        total = sum(self._distribution.values())
        if total <= 0:
            # Draw empty circle
            painter.setPen(QtGui.QPen(QtGui.QColor(200, 200, 200), 2))
            painter.setBrush(QtGui.QColor(230, 230, 230))
            painter.drawEllipse(pie_rect)
            painter.drawText(pie_rect, Qt.AlignCenter, "No modes\nselected")
            return

        # Draw pie slices
        start_angle = 90 * 16  # Start from top (Qt uses 1/16 of a degree)
        for mode, value in self._distribution.items():
            if value <= 0:
                continue

            # Calculate span angle (negative for clockwise)
            span_angle = -int((value / total) * 360 * 16)

            # Get color for this mode
            color = self.MODE_COLORS.get(mode, QtGui.QColor(128, 128, 128))

            # Draw slice
            painter.setPen(QtGui.QPen(Qt.white, 1))
            painter.setBrush(color)
            painter.drawPie(pie_rect, start_angle, span_angle)

            # Move to next slice
            start_angle += span_angle

        # Draw center circle (to make it look like a donut chart - optional)
        # center_size = size * 0.3
        # center_rect = QtCore.QRectF(x + (size - center_size) / 2,
        #                              y + (size - center_size) / 2,
        #                              center_size, center_size)
        # painter.setBrush(QtGui.QColor(245, 245, 245))
        # painter.drawEllipse(center_rect)

        painter.end()


class ModeSlider(QWidget):
    """
    A single mode slider with label and percentage display.
    """
    valueChanged = pyqtSignal(str, int)  # mode_name, value

    def __init__(self, mode_name: str, display_name: str, color: QtGui.QColor, parent=None):
        super().__init__(parent)
        self.mode_name = mode_name

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)

        # Color indicator
        self.color_label = QLabel()
        self.color_label.setFixedSize(16, 16)
        self.color_label.setStyleSheet(f"background-color: {color.name()}; border-radius: 3px;")
        layout.addWidget(self.color_label)

        # Mode name
        self.name_label = QLabel(display_name)
        self.name_label.setFixedWidth(100)
        layout.addWidget(self.name_label)

        # Slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(0)
        self.slider.setTickPosition(QSlider.NoTicks)
        layout.addWidget(self.slider, 1)

        # Percentage spinbox
        self.spinbox = QSpinBox()
        self.spinbox.setRange(0, 100)
        self.spinbox.setSuffix("%")
        self.spinbox.setFixedWidth(60)
        layout.addWidget(self.spinbox)

        # Connect signals
        self.slider.valueChanged.connect(self._on_slider_changed)
        self.spinbox.valueChanged.connect(self._on_spinbox_changed)

    def _on_slider_changed(self, value):
        self.spinbox.blockSignals(True)
        self.spinbox.setValue(value)
        self.spinbox.blockSignals(False)
        self.valueChanged.emit(self.mode_name, value)

    def _on_spinbox_changed(self, value):
        self.slider.blockSignals(True)
        self.slider.setValue(value)
        self.slider.blockSignals(False)
        self.valueChanged.emit(self.mode_name, value)

    def get_value(self) -> int:
        return self.slider.value()

    def set_value(self, value: int):
        self.slider.blockSignals(True)
        self.spinbox.blockSignals(True)
        self.slider.setValue(value)
        self.spinbox.setValue(value)
        self.slider.blockSignals(False)
        self.spinbox.blockSignals(False)


class ModeDistributionWidget(QWidget):
    """
    Complete widget for configuring beam mode distribution.
    Includes sliders for each mode and a pie chart visualization.
    """
    distributionChanged = pyqtSignal(dict)  # Emits the mode distribution dict

    def __init__(self, parent=None, logger=None):
        super().__init__(parent)

        if logger is None:
            self.logger = logging.getLogger(self.__class__.__name__)
        else:
            self.logger = logger.getChild(self.__class__.__name__)

        self._setup_ui()
        self._connect_signals()

        # Set default distribution (100% Gaussian)
        self.sliders["gaussian"].set_value(100)

        # Manually update pie chart and total label since signals are blocked during set_value
        self._update_pie_chart()
        self._update_total_label()

    def _setup_ui(self):
        """Set up the widget UI."""
        # Set minimum width on the widget itself to ensure label is visible
        self.setMinimumWidth(450)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Group box for mode distribution
        group_box = QGroupBox("Beam Mode Distribution  ")
        group_layout = QHBoxLayout(group_box)

        # Left side: Sliders
        sliders_widget = QWidget()
        sliders_layout = QVBoxLayout(sliders_widget)
        sliders_layout.setContentsMargins(0, 0, 0, 0)
        sliders_layout.setSpacing(4)

        self.sliders = {}
        modes = [
            ("gaussian", "Gaussian", PieChartWidget.MODE_COLORS["gaussian"]),
            ("hermite_gauss", "Hermite-Gauss", PieChartWidget.MODE_COLORS["hermite_gauss"]),
            ("laguerre_gauss", "Laguerre-Gauss", PieChartWidget.MODE_COLORS["laguerre_gauss"]),
            ("doughnut", "Doughnut", PieChartWidget.MODE_COLORS["doughnut"]),
        ]

        for mode_name, display_name, color in modes:
            slider = ModeSlider(mode_name, display_name, color)
            self.sliders[mode_name] = slider
            sliders_layout.addWidget(slider)

        sliders_layout.addStretch()

        # Total percentage label
        self.total_label = QLabel("Total: 0%")
        self.total_label.setAlignment(Qt.AlignCenter)
        sliders_layout.addWidget(self.total_label)

        group_layout.addWidget(sliders_widget, 2)

        # Right side: Pie chart
        self.pie_chart = PieChartWidget()
        group_layout.addWidget(self.pie_chart, 1)

        main_layout.addWidget(group_box)

        # Additional settings
        settings_layout = QHBoxLayout()

        # Speckle noise slider
        noise_label = QLabel("Speckle Noise:")
        settings_layout.addWidget(noise_label)

        self.noise_slider = QSlider(Qt.Horizontal)
        self.noise_slider.setRange(0, 100)
        self.noise_slider.setValue(0)
        settings_layout.addWidget(self.noise_slider)

        self.noise_spinbox = QDoubleSpinBox()
        self.noise_spinbox.setRange(0.0, 1.0)
        self.noise_spinbox.setSingleStep(0.05)
        self.noise_spinbox.setDecimals(2)
        self.noise_spinbox.setValue(0.0)
        self.noise_spinbox.setFixedWidth(70)
        settings_layout.addWidget(self.noise_spinbox)

        settings_layout.addSpacing(20)

        # Max mode order
        order_label = QLabel("Max Mode Order:")
        settings_layout.addWidget(order_label)

        self.order_spinbox = QSpinBox()
        self.order_spinbox.setRange(1, 10)
        self.order_spinbox.setValue(3)
        self.order_spinbox.setFixedWidth(50)
        self.order_spinbox.setToolTip("Maximum n,m values for HG/LG modes")
        settings_layout.addWidget(self.order_spinbox)

        settings_layout.addStretch()

        main_layout.addLayout(settings_layout)

    def _connect_signals(self):
        """Connect all signals."""
        for slider in self.sliders.values():
            slider.valueChanged.connect(self._on_slider_changed)

        self.noise_slider.valueChanged.connect(self._on_noise_slider_changed)
        self.noise_spinbox.valueChanged.connect(self._on_noise_spinbox_changed)

    def _on_slider_changed(self, mode_name: str, value: int):
        """Handle slider value change."""
        self._update_pie_chart()
        self._update_total_label()
        self.distributionChanged.emit(self.get_distribution())

    def _on_noise_slider_changed(self, value: int):
        """Handle noise slider change."""
        self.noise_spinbox.blockSignals(True)
        self.noise_spinbox.setValue(value / 100.0)
        self.noise_spinbox.blockSignals(False)

    def _on_noise_spinbox_changed(self, value: float):
        """Handle noise spinbox change."""
        self.noise_slider.blockSignals(True)
        self.noise_slider.setValue(int(value * 100))
        self.noise_slider.blockSignals(False)

    def _update_pie_chart(self):
        """Update the pie chart with current distribution."""
        distribution = self.get_distribution()
        self.pie_chart.set_distribution(distribution)

    def _update_total_label(self):
        """Update the total percentage label."""
        total = sum(slider.get_value() for slider in self.sliders.values())
        if total == 100:
            self.total_label.setText("Total: 100%")
            self.total_label.setStyleSheet("color: green; font-weight: bold;")
        elif total == 0:
            self.total_label.setText("Total: 0%")
            self.total_label.setStyleSheet("color: red; font-weight: bold;")
        else:
            self.total_label.setText(f"Total: {total}%")
            self.total_label.setStyleSheet("color: orange; font-weight: bold;")

    def get_distribution(self) -> dict:
        """Get the current mode distribution as a dictionary."""
        return {name: slider.get_value() for name, slider in self.sliders.items()}

    def set_distribution(self, distribution: dict):
        """Set the mode distribution."""
        for name, value in distribution.items():
            if name in self.sliders:
                self.sliders[name].set_value(value)
        self._update_pie_chart()
        self._update_total_label()

    def get_speckle_noise(self) -> float:
        """Get the speckle noise level (0.0 to 1.0)."""
        return self.noise_spinbox.value()

    def set_speckle_noise(self, value: float):
        """Set the speckle noise level."""
        self.noise_spinbox.setValue(max(0.0, min(1.0, value)))

    def get_max_mode_order(self) -> int:
        """Get the maximum mode order."""
        return self.order_spinbox.value()

    def set_max_mode_order(self, value: int):
        """Set the maximum mode order."""
        self.order_spinbox.setValue(max(1, min(10, value)))

    def get_all_settings(self) -> dict:
        """Get all settings as a dictionary."""
        return {
            "mode_distribution": self.get_distribution(),
            "speckle_noise": self.get_speckle_noise(),
            "max_mode_order": self.get_max_mode_order(),
        }

    def set_all_settings(self, settings: dict):
        """Set all settings from a dictionary."""
        if "mode_distribution" in settings:
            self.set_distribution(settings["mode_distribution"])
        if "speckle_noise" in settings:
            self.set_speckle_noise(settings["speckle_noise"])
        if "max_mode_order" in settings:
            self.set_max_mode_order(settings["max_mode_order"])
