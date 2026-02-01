"""
Log Settings Dialog - Configure logging settings.
"""
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QComboBox, QSpinBox, QCheckBox, QPushButton, QGroupBox, QLabel
)
from PyQt5.QtCore import Qt

from ui.utils.log_manager import get_log_manager, LOG_LEVELS


class LogSettingsDialog(QDialog):
    """
    Dialog for configuring logging settings.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Log Settings")
        self.setMinimumWidth(350)

        self.log_manager = get_log_manager()
        self._setup_ui()
        self._load_current_settings()

    def _setup_ui(self):
        """Setup the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        # Log Level Group
        level_group = QGroupBox("Log Level")
        level_layout = QFormLayout(level_group)

        self.level_combo = QComboBox()
        self.level_combo.addItems(list(LOG_LEVELS.keys()))
        level_layout.addRow("Level:", self.level_combo)

        level_help = QLabel(
            "DEBUG: All messages (verbose)\n"
            "INFO: Normal operation messages\n"
            "WARNING: Potential issues\n"
            "ERROR: Errors only"
        )
        level_help.setStyleSheet("color: #666; font-size: 10px;")
        level_layout.addRow(level_help)

        layout.addWidget(level_group)

        # File Settings Group
        file_group = QGroupBox("File Settings")
        file_layout = QFormLayout(file_group)

        self.max_files_spin = QSpinBox()
        self.max_files_spin.setRange(1, 100)
        self.max_files_spin.setSuffix(" files")
        file_layout.addRow("Keep last:", self.max_files_spin)

        self.max_size_spin = QSpinBox()
        self.max_size_spin.setRange(1, 100)
        self.max_size_spin.setSuffix(" MB")
        file_layout.addRow("Max file size:", self.max_size_spin)

        self.log_to_file_check = QCheckBox("Enable file logging")
        file_layout.addRow(self.log_to_file_check)

        layout.addWidget(file_group)

        # Console Settings Group
        console_group = QGroupBox("Console Settings")
        console_layout = QFormLayout(console_group)

        self.log_to_console_check = QCheckBox("Enable console logging")
        console_layout.addRow(self.log_to_console_check)

        layout.addWidget(console_group)

        # Buttons
        button_layout = QHBoxLayout()

        self.apply_btn = QPushButton("Apply")
        self.apply_btn.clicked.connect(self._apply_settings)

        self.save_btn = QPushButton("Save && Close")
        self.save_btn.clicked.connect(self._save_and_close)
        self.save_btn.setDefault(True)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)

        button_layout.addStretch()
        button_layout.addWidget(self.apply_btn)
        button_layout.addWidget(self.save_btn)
        button_layout.addWidget(self.cancel_btn)

        layout.addLayout(button_layout)

    def _load_current_settings(self):
        """Load current settings from log manager."""
        config = self.log_manager.get_config()

        # Log level
        level_index = self.level_combo.findText(config.get("log_level", "INFO"))
        if level_index >= 0:
            self.level_combo.setCurrentIndex(level_index)

        # File settings
        self.max_files_spin.setValue(config.get("max_log_files", 10))
        self.max_size_spin.setValue(config.get("max_file_size_mb", 5))
        self.log_to_file_check.setChecked(config.get("log_to_file", True))

        # Console settings
        self.log_to_console_check.setChecked(config.get("log_to_console", True))

    def _apply_settings(self):
        """Apply settings without closing."""
        self.log_manager.set_log_level(self.level_combo.currentText())
        self.log_manager.set_max_log_files(self.max_files_spin.value())
        self.log_manager.set_log_to_file(self.log_to_file_check.isChecked())
        self.log_manager.set_log_to_console(self.log_to_console_check.isChecked())

        # Note: max_file_size_mb change requires restart of logging to take effect
        config = self.log_manager.get_config()
        config["max_file_size_mb"] = self.max_size_spin.value()

    def _save_and_close(self):
        """Apply settings, save to file, and close."""
        self._apply_settings()
        self.log_manager.save_config()
        self.accept()
