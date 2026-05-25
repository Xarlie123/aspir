"""
Horizontal stepper widget showing step progress in a wizard-style interface.
"""
import logging
from enum import Enum
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QFrame, QSizePolicy
)
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QFont, QPainter, QColor, QPen, QBrush


class StepState(Enum):
    """Possible states for a step."""
    PENDING = "pending"       # Not yet reachable
    ACTIVE = "active"         # Current step
    COMPLETED = "completed"   # Done, can revisit
    INVALIDATED = "invalidated"  # Was done but needs redo


class StepIndicator(QWidget):
    """
    Individual step indicator showing number, name, and state.
    Clickable when completed or active.
    """
    clicked = Signal(int)  # Emits step index when clicked

    # Colors for different states
    COLORS = {
        StepState.PENDING: {
            'bg': QColor(200, 200, 200),      # Gray
            'border': QColor(180, 180, 180),
            'text': QColor(120, 120, 120),
            'number': QColor(255, 255, 255),
        },
        StepState.ACTIVE: {
            'bg': QColor(33, 150, 243),       # Blue
            'border': QColor(25, 118, 210),
            'text': QColor(33, 150, 243),
            'number': QColor(255, 255, 255),
        },
        StepState.COMPLETED: {
            'bg': QColor(76, 175, 80),        # Green
            'border': QColor(56, 142, 60),
            'text': QColor(76, 175, 80),
            'number': QColor(255, 255, 255),
        },
        StepState.INVALIDATED: {
            'bg': QColor(255, 152, 0),        # Orange
            'border': QColor(245, 124, 0),
            'text': QColor(255, 152, 0),
            'number': QColor(255, 255, 255),
        },
    }

    # State icons
    ICONS = {
        StepState.PENDING: "○",
        StepState.ACTIVE: "●",
        StepState.COMPLETED: "✓",
        StepState.INVALIDATED: "⚠",
    }

    def __init__(self, index: int, name: str, parent=None):
        super().__init__(parent)
        self.index = index
        self.name = name
        self._state = StepState.PENDING
        self._is_current = False  # Whether this is the currently viewed step

        self.setFixedSize(80, 74)  # Slightly taller to accommodate underline
        self.setCursor(Qt.PointingHandCursor)
        self._setup_ui()

    def _setup_ui(self):
        """Setup the indicator UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.setAlignment(Qt.AlignCenter)

        # Circle with number will be drawn in paintEvent
        self.circle_widget = QWidget()
        self.circle_widget.setFixedSize(36, 36)
        layout.addWidget(self.circle_widget, alignment=Qt.AlignCenter)

        # Step name
        self.name_label = QLabel(self.name)
        self.name_label.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(9)
        self.name_label.setFont(font)
        layout.addWidget(self.name_label)

        # State icon
        self.state_label = QLabel(self.ICONS[self._state])
        self.state_label.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(10)
        self.state_label.setFont(font)
        layout.addWidget(self.state_label)

    def set_state(self, state: StepState):
        """Set the step state and update appearance."""
        self._state = state
        colors = self.COLORS[state]

        # Update name label color
        self.name_label.setStyleSheet(f"color: {colors['text'].name()};")

        # Update state icon
        self.state_label.setText(self.ICONS[state])
        self.state_label.setStyleSheet(f"color: {colors['text'].name()};")

        # Update cursor based on clickability
        if state in (StepState.COMPLETED, StepState.ACTIVE, StepState.INVALIDATED):
            self.setCursor(Qt.PointingHandCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

        self.update()

    def get_state(self) -> StepState:
        """Get current state."""
        return self._state

    def set_current(self, is_current: bool):
        """Set whether this step is currently being viewed."""
        self._is_current = is_current
        self.update()

    def is_current(self) -> bool:
        """Check if this step is currently being viewed."""
        return self._is_current

    def paintEvent(self, event):
        """Draw the circle with step number and current indicator."""
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        colors = self.COLORS[self._state]

        # Draw circle
        circle_rect = self.circle_widget.geometry()
        center_x = circle_rect.center().x()
        center_y = circle_rect.center().y()
        radius = 16

        # Fill
        painter.setBrush(QBrush(colors['bg']))
        painter.setPen(QPen(colors['border'], 2))
        painter.drawEllipse(center_x - radius, center_y - radius,
                           radius * 2, radius * 2)

        # Number text
        painter.setPen(colors['number'])
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(center_x - radius, center_y - radius,
                        radius * 2, radius * 2,
                        Qt.AlignCenter, str(self.index + 1))

        # Draw underline if this is the current step being viewed
        if self._is_current:
            underline_color = QColor(33, 150, 243)  # Blue color for visibility
            painter.setPen(QPen(underline_color, 3))
            # Draw line at the very bottom of the widget (with small offset from edge)
            y = self.height() - 1
            margin = 10
            painter.drawLine(margin, y, self.width() - margin, y)

    def mousePressEvent(self, event):
        """Handle click - only emit if clickable."""
        if self._state in (StepState.COMPLETED, StepState.ACTIVE, StepState.INVALIDATED):
            self.clicked.emit(self.index)
        super().mousePressEvent(event)


class StepConnector(QWidget):
    """Horizontal line connecting two steps."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(36)
        self.setMinimumWidth(30)
        self._active = False

    def set_active(self, active: bool):
        """Set whether this connector is active (between completed steps)."""
        self._active = active
        self.update()

    def paintEvent(self, event):
        """Draw the connector line."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        color = QColor(76, 175, 80) if self._active else QColor(200, 200, 200)
        painter.setPen(QPen(color, 2))

        y = self.height() // 2
        painter.drawLine(0, y, self.width(), y)


class StepperWidget(QWidget):
    """
    Horizontal stepper showing 5 steps with their states.
    Emits step_clicked when a completed/active step is clicked.
    """
    step_clicked = Signal(int)  # Emits step index

    # Step definitions
    STEPS = [
        "Dataset",
        "Masks",
        "Test",
        "DNN",
        "Reports"
    ]

    def __init__(self, parent=None, logger=None):
        super().__init__(parent)

        if logger is None:
            self.logger = logging.getLogger(self.__class__.__name__)
        else:
            self.logger = logger.getChild(self.__class__.__name__)

        self._indicators = []
        self._connectors = []
        self._current_step = 0

        self._setup_ui()
        self.logger.debug("StepperWidget initialized with %d steps", len(self.STEPS))

    def _setup_ui(self):
        """Setup the stepper UI."""
        self.setFixedHeight(94)  # Height for step indicators (74px) + margins

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 5, 20, 5)  # Less vertical margin
        main_layout.setSpacing(0)

        # Add "Experiment Steps:" label at the start (matching "Select Mode:" style)
        steps_label = QLabel("Experiment Steps:")
        steps_label.setStyleSheet("""
            QLabel {
                font-size: 13px;
                font-weight: bold;
                color: #333;
            }
        """)
        steps_label.setAlignment(Qt.AlignVCenter)
        main_layout.addWidget(steps_label)
        main_layout.addSpacing(15)

        for i, step_name in enumerate(self.STEPS):
            # Add step indicator
            indicator = StepIndicator(i, step_name)
            indicator.clicked.connect(self._on_step_clicked)
            self._indicators.append(indicator)
            main_layout.addWidget(indicator)

            # Add connector between steps (not after last)
            if i < len(self.STEPS) - 1:
                connector = StepConnector()
                self._connectors.append(connector)
                main_layout.addWidget(connector)

        # Add stretch at end for centering
        main_layout.addStretch()

        # Set initial state
        self._indicators[0].set_state(StepState.ACTIVE)
        self._indicators[0].set_current(True)  # Mark first step as currently viewed

    def _on_step_clicked(self, index: int):
        """Handle step indicator click."""
        self.logger.debug("Step %d clicked", index)
        self.step_clicked.emit(index)

    def set_current_step(self, index: int):
        """Set the current active step (the one being viewed)."""
        if 0 <= index < len(self._indicators):
            # Clear previous current indicator
            for indicator in self._indicators:
                indicator.set_current(False)
            # Set new current indicator
            self._current_step = index
            self._indicators[index].set_current(True)
            self._update_visual_states()
            self.logger.debug("Current step set to %d", index)

    def get_current_step(self) -> int:
        """Get current step index."""
        return self._current_step

    def set_step_state(self, index: int, state: StepState):
        """Set state for a specific step."""
        if 0 <= index < len(self._indicators):
            self._indicators[index].set_state(state)
            self._update_connectors()
            self.logger.debug("Step %d state set to %s", index, state.value)

    def get_step_state(self, index: int) -> StepState:
        """Get state of a specific step."""
        if 0 <= index < len(self._indicators):
            return self._indicators[index].get_state()
        return StepState.PENDING

    def complete_step(self, index: int):
        """Mark a step as completed."""
        self.set_step_state(index, StepState.COMPLETED)

    def invalidate_from(self, start_index: int):
        """Invalidate all steps from start_index onwards."""
        for i in range(start_index, len(self._indicators)):
            current_state = self._indicators[i].get_state()
            if current_state in (StepState.COMPLETED, StepState.ACTIVE):
                self._indicators[i].set_state(StepState.INVALIDATED)
        self._update_connectors()
        self.logger.debug("Invalidated steps from %d", start_index)

    def _update_visual_states(self):
        """Update visual states based on current step."""
        for i, indicator in enumerate(self._indicators):
            if i == self._current_step:
                if indicator.get_state() != StepState.COMPLETED:
                    indicator.set_state(StepState.ACTIVE)
        self._update_connectors()

    def _update_connectors(self):
        """Update connector colors based on step states."""
        for i, connector in enumerate(self._connectors):
            # Connector is active if the step before it is completed
            left_state = self._indicators[i].get_state()
            connector.set_active(left_state == StepState.COMPLETED)

    def reset(self):
        """Reset all steps to initial state."""
        for i, indicator in enumerate(self._indicators):
            indicator.set_current(i == 0)  # Only first step is current
            if i == 0:
                indicator.set_state(StepState.ACTIVE)
            else:
                indicator.set_state(StepState.PENDING)
        self._current_step = 0
        self._update_connectors()
        self.logger.debug("Stepper reset to initial state")
