"""
Mode selector widget for choosing between Single Test, Batch Test, and Batch Reports modes.
Compact layout for top bar placement with descriptions.
"""
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QRadioButton, QButtonGroup, QLabel, QFrame,
    QSizePolicy
)
from PyQt5.QtCore import pyqtSignal, Qt


class ModeSelectorWidget(QWidget):
    """
    Compact selector for choosing application mode.
    Three modes: Single Test, Batch Test, and Batch Reports.
    Includes descriptions for each mode.
    """
    mode_changed = pyqtSignal(str)  # Emits "single_test", "batch_test", or "batch_reports"

    MODES = {
        "single_test": {
            "title": "Single Test",
            "description": "Step-by-step experiment wizard",
        },
        "batch_test": {
            "title": "Batch Test",
            "description": "Run multiple test configurations",
        },
        "batch_reports": {
            "title": "Batch Reports",
            "description": "Explore executed batch tests",
        }
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_mode = "single_test"
        self._setup_ui()

    def _setup_ui(self):
        """Setup the mode selector UI with descriptions."""
        # Fixed size to prevent expansion when stepper hides
        self.setFixedWidth(580)
        self.setFixedHeight(90)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(15, 10, 20, 10)
        layout.setSpacing(20)

        # Mode label
        mode_label = QLabel("Select Mode:")
        mode_label.setStyleSheet("""
            QLabel {
                font-size: 13px;
                font-weight: bold;
                color: #333;
            }
        """)
        mode_label.setAlignment(Qt.AlignTop)
        layout.addWidget(mode_label)

        # Button group for radio buttons
        self.button_group = QButtonGroup(self)

        # Create radio buttons with descriptions for each mode
        for i, (mode_key, mode_info) in enumerate(self.MODES.items()):
            # Container for radio + description
            mode_container = QWidget()
            mode_container.setMinimumWidth(120)
            mode_layout = QVBoxLayout(mode_container)
            mode_layout.setContentsMargins(0, 0, 0, 0)
            mode_layout.setSpacing(2)

            # Radio button
            radio = QRadioButton(mode_info["title"])
            radio.setProperty("mode_key", mode_key)
            radio.setStyleSheet("""
                QRadioButton {
                    spacing: 6px;
                    font-size: 12px;
                    font-weight: bold;
                }
                QRadioButton::indicator {
                    width: 14px;
                    height: 14px;
                }
                QRadioButton::indicator:checked {
                    background-color: #2196F3;
                    border: 2px solid #1976D2;
                    border-radius: 8px;
                }
                QRadioButton::indicator:unchecked {
                    background-color: white;
                    border: 2px solid #bdbdbd;
                    border-radius: 8px;
                }
            """)
            mode_layout.addWidget(radio)

            # Description label
            desc_label = QLabel(mode_info["description"])
            desc_label.setStyleSheet("""
                QLabel {
                    font-size: 10px;
                    color: #666;
                    padding-left: 20px;
                }
            """)
            desc_label.setWordWrap(True)
            mode_layout.addWidget(desc_label)

            mode_layout.addStretch()

            self.button_group.addButton(radio, i)
            layout.addWidget(mode_container)

            # Set first as default
            if i == 0:
                radio.setChecked(True)

        # Connect button group signal
        self.button_group.buttonClicked.connect(self._on_button_clicked)

        # Separator line at the end
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setStyleSheet("background-color: #ccc;")
        separator.setFixedWidth(1)
        layout.addWidget(separator)

    def _on_button_clicked(self, button):
        """Handle radio button selection."""
        mode_key = button.property("mode_key")
        if mode_key and mode_key != self._current_mode:
            self._current_mode = mode_key
            self.mode_changed.emit(mode_key)

    def get_current_mode(self) -> str:
        """Get the currently selected mode."""
        return self._current_mode

    def set_mode(self, mode: str):
        """Set the current mode programmatically."""
        if mode in self.MODES:
            for button in self.button_group.buttons():
                if button.property("mode_key") == mode:
                    button.setChecked(True)
                    self._current_mode = mode
                    break
