"""Hoverable image label that displays pixel coordinates and value on mouse hover."""
import numpy as np
from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QPixmap


class HoverableImageLabel(QLabel):
    """
    A QLabel that displays an image and shows pixel coordinates (X, Y) and value
    as a tooltip when the mouse hovers over the image.

    Coordinates use standard image convention: X=0, Y=0 at bottom-left corner.

    Attributes:
        _original_data: The original numpy array (grayscale or RGB)
        _pixmap_size: The size of the displayed pixmap (for coordinate mapping)
        _data_format: Optional data format string ("FP32", "INT8", "INT4")
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._original_data = None
        self._pixmap_size = None
        self._image_offset = QPoint(0, 0)  # Offset if image is centered
        self._data_format = None  # Optional data format for formatted display
        self.setMouseTracking(True)

    def setPixmapWithData(self, pixmap: QPixmap, original_data: np.ndarray,
                          data_format: str = None):
        """
        Set the pixmap and store the original image data for hover info.

        Args:
            pixmap: The QPixmap to display
            original_data: The original numpy array (H, W) or (H, W, 3)
            data_format: Optional data format string ("FP32", "INT8", "INT4")
        """
        self.setPixmap(pixmap)
        self._original_data = original_data
        self._pixmap_size = pixmap.size()
        self._data_format = data_format

        # Calculate offset for centered alignment
        self._update_image_offset()

    def set_data_format(self, data_format: str):
        """Set the data format for display in tooltips."""
        self._data_format = data_format

    def _update_image_offset(self):
        """Calculate the offset if the pixmap is centered within the label."""
        if self._pixmap_size is None:
            self._image_offset = QPoint(0, 0)
            return

        # Calculate offset based on alignment
        label_w = self.width()
        label_h = self.height()
        pix_w = self._pixmap_size.width()
        pix_h = self._pixmap_size.height()

        align = self.alignment()

        offset_x = 0
        offset_y = 0

        if align & Qt.AlignHCenter:
            offset_x = (label_w - pix_w) // 2
        elif align & Qt.AlignRight:
            offset_x = label_w - pix_w

        if align & Qt.AlignVCenter:
            offset_y = (label_h - pix_h) // 2
        elif align & Qt.AlignBottom:
            offset_y = label_h - pix_h

        self._image_offset = QPoint(max(0, offset_x), max(0, offset_y))

    def resizeEvent(self, event):
        """Update image offset when label is resized."""
        super().resizeEvent(event)
        self._update_image_offset()

    def _format_value_for_precision(self, value: float, min_val: float, max_val: float,
                                     data_format: str) -> str:
        """
        Format a value according to the specified data format precision.

        Args:
            value: The raw pixel value
            min_val: Minimum value in the data range
            max_val: Maximum value in the data range
            data_format: "FP32", "INT8", or "INT4"

        Returns:
            Formatted string representation
        """
        # Normalize value to [0, 1] based on actual data range
        if max_val > min_val:
            normalized = (value - min_val) / (max_val - min_val)
        else:
            normalized = 0.0

        # Clamp to [0, 1]
        normalized = max(0.0, min(1.0, normalized))

        if data_format == "INT4":
            # 4-bit: 16 levels (0-15)
            quantized = int(round(normalized * 15))
            return f"{quantized} (0-15)"
        elif data_format == "INT8":
            # 8-bit: 256 levels (0-255)
            quantized = int(round(normalized * 255))
            return f"{quantized} (0-255)"
        else:  # FP32 or unknown
            return f"{value:.6f}"

    def mouseMoveEvent(self, event):
        """Show tooltip with pixel coordinates and value."""
        super().mouseMoveEvent(event)

        if self._original_data is None or self._pixmap_size is None:
            self.setToolTip("")
            return

        # Get mouse position relative to label
        mouse_x = event.x() - self._image_offset.x()
        mouse_y = event.y() - self._image_offset.y()

        # Check if mouse is within the pixmap area
        pix_w = self._pixmap_size.width()
        pix_h = self._pixmap_size.height()

        if mouse_x < 0 or mouse_y < 0 or mouse_x >= pix_w or mouse_y >= pix_h:
            self.setToolTip("")
            return

        # Map to original image coordinates
        orig_h, orig_w = self._original_data.shape[:2]

        # Calculate scale factors
        scale_x = orig_w / pix_w
        scale_y = orig_h / pix_h

        # Get original pixel coordinates (array indices)
        array_x = int(mouse_x * scale_x)
        array_y = int(mouse_y * scale_y)

        # Clamp to valid range
        array_x = max(0, min(array_x, orig_w - 1))
        array_y = max(0, min(array_y, orig_h - 1))

        # Convert to display coordinates with Y=0 at bottom-left
        display_x = array_x
        display_y = orig_h - 1 - array_y

        # Get pixel value (using array coordinates)
        pixel_val = self._original_data[array_y, array_x]

        # Get data range for proper normalization
        data_min = float(self._original_data.min())
        data_max = float(self._original_data.max())

        # Format value based on data type
        if isinstance(pixel_val, np.ndarray):
            # RGB or multi-channel
            val_str = f"[{', '.join(f'{v:.2f}' if np.issubdtype(type(v), np.floating) else str(v) for v in pixel_val)}]"
            formatted_str = None  # Skip formatted for multi-channel
        elif np.issubdtype(type(pixel_val), np.floating):
            val_str = f"{pixel_val:.4f}"
            # Get formatted value if data format is set
            if self._data_format:
                formatted_str = self._format_value_for_precision(
                    float(pixel_val), data_min, data_max, self._data_format
                )
            else:
                formatted_str = None
        else:
            val_str = str(pixel_val)
            # For integer types, format based on actual range
            if self._data_format:
                formatted_str = self._format_value_for_precision(
                    float(pixel_val), data_min, data_max, self._data_format
                )
            else:
                formatted_str = None

        # Build tooltip
        tooltip_lines = [
            f"X: {display_x}, Y: {display_y}",
            f"Value: {val_str}"
        ]

        if formatted_str and self._data_format:
            tooltip_lines.append(f"{self._data_format}: {formatted_str}")

        self.setToolTip("\n".join(tooltip_lines))

    def leaveEvent(self, event):
        """Clear tooltip when mouse leaves the label."""
        super().leaveEvent(event)
        self.setToolTip("")
