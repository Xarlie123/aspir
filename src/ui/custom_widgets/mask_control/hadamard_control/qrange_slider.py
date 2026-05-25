# File: ui/custom_widgets/mask_control/hadamard_control/qrange_slider.py
"""
Custom dual-handle range slider widget with modern styling.
Used for selecting a range of Hadamard patterns.

Features:
- Dual handles for range selection
- Click on track between handles moves the entire range
- Percentage display support via get_percentage() method
"""
import logging
from PySide6.QtWidgets import QWidget, QLabel
from PySide6.QtCore import Qt, QRect, QRectF, Signal
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QPainterPath


class QRangeSlider(QWidget):
    """
    A custom slider with two handles for selecting a range.

    Features:
    - Modern flat design with subtle handles
    - Smooth rounded track
    - Hover effects on handles
    - Click between handles moves the entire range
    - Connected to external label for value display
    """
    valueChanged = Signal(int, int)  # Signal emitting (low, high) values
    percentageChanged = Signal(float)  # Signal emitting percentage of total

    # Color palette (matching application style)
    COLOR_TRACK_BG = QColor(220, 220, 220)       # Light gray track background
    COLOR_TRACK_ACTIVE = QColor(76, 175, 80)     # Green active range (matches button style)
    COLOR_HANDLE = QColor(255, 255, 255)         # White handle fill
    COLOR_HANDLE_BORDER = QColor(76, 175, 80)    # Green handle border
    COLOR_HANDLE_HOVER = QColor(200, 230, 201)   # Light green on hover
    COLOR_HANDLE_ACTIVE = QColor(76, 175, 80)    # Green when dragging
    COLOR_RANGE_HOVER = QColor(180, 220, 180)    # Hover color for range area

    def __init__(self, min_val=0, max_val=100, parent=None, value_label: QLabel = None):
        super().__init__(parent)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug("Initializing QRangeSlider with min=%d, max=%d", min_val, max_val)

        self.min_val = min_val
        self.max_val = max_val

        # Default to 100% of patterns selected (full range)
        self.low_value = min_val
        self.high_value = max_val

        # Styling dimensions
        self.track_height = 6           # Height of the track
        self.handle_width = 14          # Width of handle (pill shape)
        self.handle_height = 22         # Height of handle
        self.handle_radius = 4          # Corner radius for rounded handles
        self.padding = 10               # Padding on sides

        # Interaction state
        self.active_handle = None       # "low", "high", "range", or None
        self.hover_handle = None        # "low", "high", "range", or None
        self._drag_start_x = 0          # For range dragging
        self._drag_start_low = 0
        self._drag_start_high = 0

        # Label connection
        self.value_label = value_label
        if self.value_label is not None:
            self.valueChanged.connect(self.update_label)
            self.update_label(self.low_value, self.high_value)

        self.setMinimumHeight(36)
        self.setMaximumHeight(36)
        self.setMouseTracking(True)
        self.setCursor(Qt.PointingHandCursor)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Calculate track position
        track_y = self.height() // 2 - self.track_height // 2
        track_left = self.padding + self.handle_width // 2
        track_right = self.width() - self.padding - self.handle_width // 2
        track_width = track_right - track_left

        # Draw background track (rounded rectangle)
        track_rect = QRectF(track_left, track_y, track_width, self.track_height)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(self.COLOR_TRACK_BG))
        painter.drawRoundedRect(track_rect, self.track_height // 2, self.track_height // 2)

        # Calculate handle positions
        left_x = self._value_to_pos(self.low_value)
        right_x = self._value_to_pos(self.high_value)

        # Draw active range on track
        active_rect = QRectF(left_x, track_y, right_x - left_x, self.track_height)
        painter.setBrush(QBrush(self.COLOR_TRACK_ACTIVE))
        painter.drawRect(active_rect)

        # Draw handles (rounded pill shapes)
        self._draw_handle(painter, left_x, "low")
        self._draw_handle(painter, right_x, "high")

    def _draw_handle(self, painter: QPainter, center_x: float, handle_id: str):
        """Draw a single handle at the given x position."""
        handle_y = self.height() // 2 - self.handle_height // 2
        handle_rect = QRectF(
            center_x - self.handle_width // 2,
            handle_y,
            self.handle_width,
            self.handle_height
        )

        # Determine colors based on state
        if self.active_handle == handle_id:
            fill_color = self.COLOR_HANDLE_ACTIVE
            border_color = self.COLOR_HANDLE_ACTIVE.darker(110)
        elif self.hover_handle == handle_id:
            fill_color = self.COLOR_HANDLE_HOVER
            border_color = self.COLOR_HANDLE_BORDER
        else:
            fill_color = self.COLOR_HANDLE
            border_color = self.COLOR_HANDLE_BORDER

        # Draw handle with border
        painter.setPen(QPen(border_color, 2))
        painter.setBrush(QBrush(fill_color))
        painter.drawRoundedRect(handle_rect, self.handle_radius, self.handle_radius)

        # Draw subtle grip lines inside handle
        if self.active_handle != handle_id:
            line_color = QColor(180, 180, 180) if self.hover_handle != handle_id else QColor(120, 160, 120)
            painter.setPen(QPen(line_color, 1))
            line_y1 = handle_y + self.handle_height // 3
            line_y2 = handle_y + 2 * self.handle_height // 3
            for offset in [-2, 2]:
                painter.drawLine(int(center_x + offset), int(line_y1),
                               int(center_x + offset), int(line_y2))

    def mousePressEvent(self, event):
        left_x = self._value_to_pos(self.low_value)
        right_x = self._value_to_pos(self.high_value)

        # Check if clicking on handles first
        if self._is_on_handle(event.x(), event.y(), left_x):
            self.active_handle = "low"
            self.logger.debug("Pressed low handle at x=%d", event.x())
        elif self._is_on_handle(event.x(), event.y(), right_x):
            self.active_handle = "high"
            self.logger.debug("Pressed high handle at x=%d", event.x())
        elif self._is_on_range(event.x(), event.y(), left_x, right_x):
            # Click between handles - drag the entire range
            self.active_handle = "range"
            self._drag_start_x = event.x()
            self._drag_start_low = self.low_value
            self._drag_start_high = self.high_value
            self.logger.debug("Started range drag at x=%d", event.x())
        else:
            # Click outside range - move nearest handle to that position
            if event.x() < left_x:
                self.active_handle = "low"
                self._update_handle_value(event.x())
            elif event.x() > right_x:
                self.active_handle = "high"
                self._update_handle_value(event.x())

        self.update()

    def mouseMoveEvent(self, event):
        if self.active_handle:
            if self.active_handle == "range":
                self._update_range_drag(event.x())
            else:
                self._update_handle_value(event.x())
        else:
            # Update hover state
            left_x = self._value_to_pos(self.low_value)
            right_x = self._value_to_pos(self.high_value)

            old_hover = self.hover_handle
            if self._is_on_handle(event.x(), event.y(), left_x):
                self.hover_handle = "low"
            elif self._is_on_handle(event.x(), event.y(), right_x):
                self.hover_handle = "high"
            elif self._is_on_range(event.x(), event.y(), left_x, right_x):
                self.hover_handle = "range"
            else:
                self.hover_handle = None

            if old_hover != self.hover_handle:
                # Update cursor based on hover state
                if self.hover_handle == "range":
                    self.setCursor(Qt.SizeHorCursor)
                else:
                    self.setCursor(Qt.PointingHandCursor)
                self.update()

    def mouseReleaseEvent(self, event):
        self.logger.debug("Released handle %s", self.active_handle)
        self.active_handle = None
        self.update()

    def leaveEvent(self, event):
        self.hover_handle = None
        self.update()

    def _is_on_handle(self, x: int, y: int, handle_center_x: float) -> bool:
        """Check if point (x, y) is within a handle."""
        handle_y = self.height() // 2 - self.handle_height // 2
        return (abs(x - handle_center_x) <= self.handle_width // 2 + 2 and
                handle_y <= y <= handle_y + self.handle_height)

    def _is_on_range(self, x: int, y: int, left_x: float, right_x: float) -> bool:
        """Check if point (x, y) is within the active range (between handles)."""
        track_y = self.height() // 2 - self.track_height // 2
        # Expand the clickable area for better usability
        expanded_height = max(self.track_height, 16)
        track_y_expanded = self.height() // 2 - expanded_height // 2
        return (left_x + self.handle_width // 2 < x < right_x - self.handle_width // 2 and
                track_y_expanded <= y <= track_y_expanded + expanded_height)

    def _update_range_drag(self, pos_x: int):
        """Update both handles while dragging the range."""
        delta_x = pos_x - self._drag_start_x
        # Convert delta to value change
        track_left = self.padding + self.handle_width // 2
        track_right = self.width() - self.padding - self.handle_width // 2
        track_width = track_right - track_left

        if track_width <= 0:
            return

        delta_value = int(delta_x / track_width * (self.max_val - self.min_val))

        # Calculate new values
        new_low = self._drag_start_low + delta_value
        new_high = self._drag_start_high + delta_value
        range_size = self._drag_start_high - self._drag_start_low

        # Clamp to boundaries while keeping range size
        if new_low < self.min_val:
            new_low = self.min_val
            new_high = self.min_val + range_size
        if new_high > self.max_val:
            new_high = self.max_val
            new_low = self.max_val - range_size

        if new_low != self.low_value or new_high != self.high_value:
            self.low_value = new_low
            self.high_value = new_high
            self.valueChanged.emit(self.low_value, self.high_value)
            self._emit_percentage()
            self.logger.debug("Range dragged to [%d, %d]", new_low, new_high)

        self.update()

    def _update_handle_value(self, pos_x: int):
        """Update the active handle's value based on mouse position."""
        new_value = self._pos_to_value(pos_x)

        if self.active_handle == "low":
            new_value = max(self.min_val, min(new_value, self.high_value - 1))
            if new_value != self.low_value:
                self.low_value = new_value
                self.valueChanged.emit(self.low_value, self.high_value)
                self._emit_percentage()
                self.logger.debug("Low handle moved to %d", new_value)
        elif self.active_handle == "high":
            new_value = min(self.max_val, max(new_value, self.low_value + 1))
            if new_value != self.high_value:
                self.high_value = new_value
                self.valueChanged.emit(self.low_value, self.high_value)
                self._emit_percentage()
                self.logger.debug("High handle moved to %d", new_value)

        self.update()

    def _emit_percentage(self):
        """Emit the current percentage of total range selected."""
        pct = self.get_percentage()
        self.percentageChanged.emit(pct)

    def _value_to_pos(self, value: int) -> float:
        """Convert a value to x position."""
        track_left = self.padding + self.handle_width // 2
        track_right = self.width() - self.padding - self.handle_width // 2
        track_width = track_right - track_left

        if self.max_val == self.min_val:
            return track_left

        ratio = (value - self.min_val) / (self.max_val - self.min_val)
        return track_left + ratio * track_width

    def _pos_to_value(self, pos: int) -> int:
        """Convert x position to a value."""
        track_left = self.padding + self.handle_width // 2
        track_right = self.width() - self.padding - self.handle_width // 2
        track_width = track_right - track_left

        if track_width <= 0:
            return self.min_val

        ratio = (pos - track_left) / track_width
        ratio = max(0, min(1, ratio))
        return int(self.min_val + ratio * (self.max_val - self.min_val))

    def update_label(self, low: int, high: int):
        """Update the connected label with current range."""
        text = f"{low} - {high - 1}"
        self.logger.debug("Updating label to '%s'", text)
        self.value_label.setText(text)

    def set_range(self, min_val: int, max_val: int, reset_to_full: bool = True):
        """
        Set the slider range.

        Args:
            min_val: Minimum value
            max_val: Maximum value
            reset_to_full: If True, reset selection to full range (100%).
                          If False, keep current selection clamped to new range.
        """
        self.min_val = min_val
        self.max_val = max_val

        if reset_to_full:
            # Default to 100% selection when range changes
            self.low_value = min_val
            self.high_value = max_val
        else:
            # Clamp current values to new range
            self.low_value = max(min_val, min(self.low_value, max_val - 1))
            self.high_value = max(self.low_value + 1, min(self.high_value, max_val))

        self.valueChanged.emit(self.low_value, self.high_value)
        self._emit_percentage()
        self.update()

    def set_values(self, low: int, high: int):
        """Set the current low and high values."""
        self.low_value = max(self.min_val, min(low, self.max_val - 1))
        self.high_value = max(self.low_value + 1, min(high, self.max_val))
        self.valueChanged.emit(self.low_value, self.high_value)
        self._emit_percentage()
        self.update()

    def get_percentage(self) -> float:
        """Get the percentage of total range currently selected."""
        total = self.max_val - self.min_val
        if total <= 0:
            return 0.0
        selected = self.high_value - self.low_value
        return (selected / total) * 100.0

    def set_percentage(self, percentage: float, anchor: str = "center"):
        """
        Set the range to cover a specific percentage of the total range.

        Args:
            percentage: Target percentage (0-100)
            anchor: Where to anchor the range - "center", "low", or "high"
        """
        total = self.max_val - self.min_val
        if total <= 0:
            return

        # Calculate new range size
        new_size = int((percentage / 100.0) * total)
        new_size = max(1, min(new_size, total))  # At least 1, at most total

        if anchor == "low":
            # Keep low value, adjust high
            new_high = min(self.low_value + new_size, self.max_val)
            new_low = self.low_value
        elif anchor == "high":
            # Keep high value, adjust low
            new_low = max(self.high_value - new_size, self.min_val)
            new_high = self.high_value
        else:  # center
            # Keep center position, adjust both
            current_center = (self.low_value + self.high_value) // 2
            half_size = new_size // 2
            new_low = max(self.min_val, current_center - half_size)
            new_high = min(self.max_val, new_low + new_size)
            # Adjust if we hit max boundary
            if new_high == self.max_val:
                new_low = max(self.min_val, new_high - new_size)

        self.low_value = new_low
        self.high_value = new_high
        self.valueChanged.emit(self.low_value, self.high_value)
        self._emit_percentage()
        self.update()

    def get_selected_count(self) -> int:
        """Get the number of patterns currently selected."""
        return self.high_value - self.low_value

    def get_total_count(self) -> int:
        """Get the total number of patterns available."""
        return self.max_val - self.min_val
