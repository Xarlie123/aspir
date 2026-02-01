# File: ui/custom_widgets/common/dataset_split_widget.py
"""
Widget for configuring and visualizing dataset split (train/validation/test).
Includes sliders for each split percentage and a horizontal stacked bar visualization.
"""

import logging
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QSlider, QSpinBox, QGroupBox)


class StackedBarWidget(QWidget):
    """
    Custom widget that draws a horizontal stacked bar showing dataset split.
    """
    # Colors for each split type
    SPLIT_COLORS = {
        "train": QtGui.QColor(76, 175, 80),       # Green
        "validation": QtGui.QColor(255, 152, 0),  # Orange
        "test": QtGui.QColor(33, 150, 243),       # Blue
    }

    SPLIT_LABELS = {
        "train": "Train",
        "validation": "Val",
        "test": "Test",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(40)
        self.setMaximumHeight(50)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self._distribution = {"train": 70, "validation": 15, "test": 15}
        self._total_images = 100

    def set_distribution(self, distribution: dict):
        """Set the split distribution to display."""
        self._distribution = distribution
        self.update()

    def set_total_images(self, total: int):
        """Set the total number of images."""
        self._total_images = total
        self.update()

    def paintEvent(self, event):
        """Draw the stacked bar."""
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        rect = self.rect()
        bar_height = 30
        bar_y = (rect.height() - bar_height) // 2
        bar_width = rect.width() - 20  # Margin
        bar_x = 10

        # Draw background
        painter.fillRect(rect, QtGui.QColor(250, 250, 250))

        # Calculate total percentage
        total_pct = sum(self._distribution.values())
        if total_pct <= 0:
            # Draw empty bar
            painter.setPen(QtGui.QPen(QtGui.QColor(200, 200, 200), 1))
            painter.setBrush(QtGui.QColor(230, 230, 230))
            painter.drawRoundedRect(bar_x, bar_y, bar_width, bar_height, 4, 4)
            painter.drawText(QtCore.QRectF(bar_x, bar_y, bar_width, bar_height),
                           Qt.AlignCenter, "No split configured")
            return

        # Draw stacked bar sections
        current_x = bar_x
        order = ["train", "validation", "test"]

        for split_name in order:
            pct = self._distribution.get(split_name, 0)
            if pct <= 0:
                continue

            section_width = int((pct / 100.0) * bar_width)
            if split_name == order[-1]:  # Last section fills remaining space
                section_width = bar_x + bar_width - current_x

            color = self.SPLIT_COLORS.get(split_name, QtGui.QColor(128, 128, 128))

            # Draw section
            painter.setPen(Qt.NoPen)
            painter.setBrush(color)

            # Rounded corners only on edges
            if split_name == order[0] and pct > 0:
                # First section - round left corners
                path = QtGui.QPainterPath()
                path.addRoundedRect(current_x, bar_y, section_width + 4, bar_height, 4, 4)
                path2 = QtGui.QPainterPath()
                path2.addRect(current_x + section_width - 4, bar_y, 8, bar_height)
                painter.drawPath(path.subtracted(path2) if section_width > 8 else path)
                painter.drawRect(current_x + 4, bar_y, section_width - 4, bar_height)
            elif split_name == order[-1]:
                # Last section - round right corners
                painter.drawRoundedRect(current_x, bar_y, section_width, bar_height, 4, 4)
                painter.drawRect(current_x, bar_y, 4, bar_height)
            else:
                # Middle section - no rounded corners
                painter.drawRect(current_x, bar_y, section_width, bar_height)

            # Draw label and count inside section if wide enough
            n_images = int(self._total_images * pct / 100.0)
            label = self.SPLIT_LABELS.get(split_name, split_name)
            text = f"{label}: {n_images}"

            painter.setPen(Qt.white)
            font = painter.font()
            font.setBold(True)
            font.setPointSize(9)
            painter.setFont(font)

            text_rect = QtCore.QRectF(current_x, bar_y, section_width, bar_height)
            if section_width > 50:
                painter.drawText(text_rect, Qt.AlignCenter, text)

            current_x += section_width

        # Draw border around entire bar
        painter.setPen(QtGui.QPen(QtGui.QColor(180, 180, 180), 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(bar_x, bar_y, bar_width, bar_height, 4, 4)

        painter.end()


class SplitSlider(QWidget):
    """
    A single split slider with label, percentage display, and image count.
    """
    valueChanged = pyqtSignal(str, int)  # split_name, value

    def __init__(self, split_name: str, display_name: str, color: QtGui.QColor,
                 initial_value: int = 0, parent=None):
        super().__init__(parent)
        self.split_name = split_name
        self._total_images = 100

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)

        # Color indicator
        self.color_label = QLabel()
        self.color_label.setFixedSize(16, 16)
        self.color_label.setStyleSheet(
            f"background-color: {color.name()}; border-radius: 3px;"
        )
        layout.addWidget(self.color_label)

        # Split name
        self.name_label = QLabel(display_name)
        self.name_label.setFixedWidth(70)
        layout.addWidget(self.name_label)

        # Slider (minimum 1% to avoid empty splits)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(1, 98)
        self.slider.setValue(initial_value)
        self.slider.setTickPosition(QSlider.NoTicks)
        layout.addWidget(self.slider, 1)

        # Percentage spinbox (minimum 1% to avoid empty splits)
        self.spinbox = QSpinBox()
        self.spinbox.setRange(1, 98)
        self.spinbox.setSuffix("%")
        self.spinbox.setValue(initial_value)
        self.spinbox.setFixedWidth(55)
        layout.addWidget(self.spinbox)

        # Image count label
        self.count_label = QLabel("(0 img)")
        self.count_label.setFixedWidth(60)
        self.count_label.setStyleSheet("color: #666;")
        layout.addWidget(self.count_label)

        # Connect signals
        self.slider.valueChanged.connect(self._on_slider_changed)
        self.spinbox.valueChanged.connect(self._on_spinbox_changed)

    def _on_slider_changed(self, value):
        self.spinbox.blockSignals(True)
        self.spinbox.setValue(value)
        self.spinbox.blockSignals(False)
        self._update_count_label()
        self.valueChanged.emit(self.split_name, value)

    def _on_spinbox_changed(self, value):
        self.slider.blockSignals(True)
        self.slider.setValue(value)
        self.slider.blockSignals(False)
        self._update_count_label()
        self.valueChanged.emit(self.split_name, value)

    def _update_count_label(self):
        n_images = int(self._total_images * self.slider.value() / 100.0)
        self.count_label.setText(f"({n_images} img)")

    def set_total_images(self, total: int):
        self._total_images = total
        self._update_count_label()

    def get_value(self) -> int:
        return self.slider.value()

    def set_value(self, value: int, emit_signal: bool = True):
        self.slider.blockSignals(True)
        self.spinbox.blockSignals(True)
        self.slider.setValue(value)
        self.spinbox.setValue(value)
        self.slider.blockSignals(False)
        self.spinbox.blockSignals(False)
        self._update_count_label()
        if emit_signal:
            self.valueChanged.emit(self.split_name, value)

    def set_maximum(self, max_val: int):
        """Set maximum value for this slider."""
        self.slider.setMaximum(max_val)
        self.spinbox.setMaximum(max_val)


class DatasetSplitWidget(QWidget):
    """
    Complete widget for configuring dataset split (train/validation/test).
    Ensures the total always equals 100%.
    """
    splitChanged = pyqtSignal(dict)  # Emits the split distribution dict

    def __init__(self, parent=None, logger=None):
        super().__init__(parent)

        if logger is None:
            self.logger = logging.getLogger(self.__class__.__name__)
        else:
            self.logger = logger.getChild(self.__class__.__name__)

        self._total_images = 100
        self._adjusting = False  # Prevent recursive adjustments

        self._setup_ui()
        self._connect_signals()

        # Set default distribution
        self._set_initial_distribution()

    def _setup_ui(self):
        """Set up the widget UI."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Group box with larger title
        group_box = QGroupBox()
        group_box.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 14px;
                border: none;
                margin-top: 0px;
                padding-top: 0px;
            }
        """)
        group_layout = QVBoxLayout(group_box)
        group_layout.setSpacing(6)
        group_layout.setContentsMargins(0, 0, 0, 0)

        # Title label (matching "Training Parameters" style)
        title_label = QLabel("<h3>Dataset Split</h3>")
        group_layout.addWidget(title_label)

        # Total images label
        self.total_label = QLabel("Total: 100 images available")
        self.total_label.setStyleSheet("color: #666; font-size: 11px;")
        group_layout.addWidget(self.total_label)

        # Stacked bar visualization
        self.stacked_bar = StackedBarWidget()
        group_layout.addWidget(self.stacked_bar)

        # Sliders
        self.sliders = {}
        splits = [
            ("train", "Train", StackedBarWidget.SPLIT_COLORS["train"], 80),
            ("validation", "Validation", StackedBarWidget.SPLIT_COLORS["validation"], 10),
            ("test", "Test", StackedBarWidget.SPLIT_COLORS["test"], 10),
        ]

        for split_name, display_name, color, initial in splits:
            slider = SplitSlider(split_name, display_name, color, initial)
            self.sliders[split_name] = slider
            group_layout.addWidget(slider)

        main_layout.addWidget(group_box)

    def _connect_signals(self):
        """Connect all signals."""
        for slider in self.sliders.values():
            slider.valueChanged.connect(self._on_slider_changed)

    def _set_initial_distribution(self):
        """Set initial distribution without triggering adjustments."""
        self._adjusting = True
        self.sliders["train"].set_value(80, emit_signal=False)
        self.sliders["validation"].set_value(10, emit_signal=False)
        self.sliders["test"].set_value(10, emit_signal=False)
        self._adjusting = False
        self._update_bar()

    def _on_slider_changed(self, split_name: str, value: int):
        """Handle slider value change - adjust others to maintain 100%."""
        if self._adjusting:
            return

        self._adjusting = True

        # Get current values
        train_val = self.sliders["train"].get_value()
        val_val = self.sliders["validation"].get_value()
        test_val = self.sliders["test"].get_value()

        total = train_val + val_val + test_val
        diff = total - 100

        # Minimum 1% for each split to avoid empty datasets
        MIN_SPLIT = 1

        if diff != 0:
            # Adjust the other sliders proportionally (respecting minimum)
            if split_name == "train":
                # Adjust validation and test
                remaining = val_val + test_val
                new_remaining = 100 - train_val
                if remaining > 0:
                    val_ratio = val_val / remaining
                    new_val = max(MIN_SPLIT, int(new_remaining * val_ratio))
                    new_test = max(MIN_SPLIT, new_remaining - new_val)
                else:
                    new_val = max(MIN_SPLIT, new_remaining // 2)
                    new_test = max(MIN_SPLIT, new_remaining - new_val)

                # Ensure we don't exceed 100%
                if train_val + new_val + new_test > 100:
                    new_test = 100 - train_val - new_val

                self.sliders["validation"].set_value(new_val, emit_signal=False)
                self.sliders["test"].set_value(new_test, emit_signal=False)

            elif split_name == "validation":
                # Adjust test primarily, then train if needed
                new_test = 100 - train_val - val_val
                if new_test < MIN_SPLIT:
                    new_test = MIN_SPLIT
                    new_train = 100 - val_val - new_test
                    self.sliders["train"].set_value(max(MIN_SPLIT, new_train), emit_signal=False)
                self.sliders["test"].set_value(new_test, emit_signal=False)

            elif split_name == "test":
                # Adjust validation primarily, then train if needed
                new_val = 100 - train_val - test_val
                if new_val < MIN_SPLIT:
                    new_val = MIN_SPLIT
                    new_train = 100 - test_val - new_val
                    self.sliders["train"].set_value(max(MIN_SPLIT, new_train), emit_signal=False)
                self.sliders["validation"].set_value(new_val, emit_signal=False)

        self._adjusting = False
        self._update_bar()
        self.splitChanged.emit(self.get_split())

    def _update_bar(self):
        """Update the stacked bar with current distribution."""
        distribution = self.get_split()
        self.stacked_bar.set_distribution(distribution)

    def set_total_images(self, total: int):
        """Set the total number of images available."""
        self._total_images = total
        self.total_label.setText(f"Total: {total} images available")
        self.stacked_bar.set_total_images(total)
        for slider in self.sliders.values():
            slider.set_total_images(total)

    def get_split(self) -> dict:
        """Get the current split as percentages."""
        return {name: slider.get_value() for name, slider in self.sliders.items()}

    def get_split_counts(self) -> dict:
        """Get the current split as image counts."""
        split_pct = self.get_split()
        return {
            name: int(self._total_images * pct / 100.0)
            for name, pct in split_pct.items()
        }

    def set_split(self, split: dict):
        """Set the split distribution (percentages)."""
        self._adjusting = True
        for name, value in split.items():
            if name in self.sliders:
                self.sliders[name].set_value(value, emit_signal=False)
        self._adjusting = False
        self._update_bar()

    def get_train_ratio(self) -> float:
        """Get train ratio (0.0 to 1.0)."""
        return self.sliders["train"].get_value() / 100.0

    def get_validation_ratio(self) -> float:
        """Get validation ratio (0.0 to 1.0)."""
        return self.sliders["validation"].get_value() / 100.0

    def get_test_ratio(self) -> float:
        """Get test ratio (0.0 to 1.0)."""
        return self.sliders["test"].get_value() / 100.0
