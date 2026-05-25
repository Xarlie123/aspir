"""
Navigation bar for stepper widget with Back and Next buttons.
"""
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QPushButton, QSizePolicy
)
from PySide6.QtCore import Signal


class StepperNavigation(QWidget):
    """
    Horizontal navigation bar with Back and Next buttons.
    Back is disabled on step 1.
    Next is disabled until current step is complete.
    """
    back_clicked = Signal()
    next_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_step = 0
        self._total_steps = 5
        self._next_enabled = False

        self._setup_ui()

    def _setup_ui(self):
        """Setup the navigation UI."""
        self.setFixedHeight(50)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 5, 20, 10)

        # Back button (left side)
        self.back_button = QPushButton("← Back")
        self.back_button.setFixedWidth(100)
        self.back_button.clicked.connect(self._on_back_clicked)
        self.back_button.setStyleSheet("""
            QPushButton {
                background-color: #e0e0e0;
                border: 1px solid #bdbdbd;
                border-radius: 4px;
                padding: 8px 16px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #d0d0d0;
            }
            QPushButton:disabled {
                background-color: #f5f5f5;
                color: #bdbdbd;
            }
        """)
        layout.addWidget(self.back_button)

        # Spacer
        layout.addStretch()

        # Next button (right side)
        self.next_button = QPushButton("Next →")
        self.next_button.setFixedWidth(100)
        self.next_button.clicked.connect(self._on_next_clicked)
        self.next_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                border: 1px solid #1976D2;
                border-radius: 4px;
                padding: 8px 16px;
                font-size: 12px;
                color: white;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #BBDEFB;
                border: 1px solid #90CAF9;
                color: #90CAF9;
            }
        """)
        layout.addWidget(self.next_button)

        # Initial state
        self._update_button_states()

    def _on_back_clicked(self):
        """Handle back button click."""
        self.back_clicked.emit()

    def _on_next_clicked(self):
        """Handle next button click."""
        self.next_clicked.emit()

    def set_current_step(self, step: int):
        """Set current step index (0-based)."""
        self._current_step = step
        self._update_button_states()

    def set_total_steps(self, total: int):
        """Set total number of steps."""
        self._total_steps = total
        self._update_button_states()

    def set_next_enabled(self, enabled: bool):
        """Enable or disable the Next button based on step completion."""
        self._next_enabled = enabled
        self._update_button_states()

    def _update_button_states(self):
        """Update button enabled/disabled states."""
        # Back disabled on first step
        self.back_button.setEnabled(self._current_step > 0)

        # Next disabled on last step or if step not complete
        on_last_step = self._current_step >= self._total_steps - 1
        self.next_button.setEnabled(self._next_enabled and not on_last_step)

        # Change Next text on last step
        if on_last_step:
            self.next_button.setText("Finish")
        else:
            self.next_button.setText("Next →")
