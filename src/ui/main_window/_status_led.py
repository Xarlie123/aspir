"""Simple LED-style status indicator widget."""
from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QBrush, QColor, QPainter
from PyQt5.QtWidgets import QWidget


class StatusLED(QWidget):
    """A simple LED indicator widget that displays a colored circle."""

    def __init__(self, parent=None, size=16):
        super().__init__(parent)
        self._color = QColor("#00cc00")  # Green by default (ready)
        self._size = size
        self.setFixedSize(size, size)

    def set_color(self, color: str):
        """Set the LED color using a hex string (e.g., '#cc0000' for red)."""
        self._color = QColor(color)
        self.update()

    def set_ready(self):
        """Set LED to green (ready state)."""
        self.set_color("#00cc00")

    def set_busy(self):
        """Set LED to red (busy state)."""
        self.set_color("#cc0000")

    def set_error(self):
        """Set LED to orange (error state)."""
        self.set_color("#ff8800")

    def paintEvent(self, event):
        """Paint the LED as a filled circle."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QBrush(self._color))
        painter.setPen(Qt.NoPen)
        # Draw circle with small margin
        margin = 2
        painter.drawEllipse(margin, margin, self._size - 2 * margin, self._size - 2 * margin)
