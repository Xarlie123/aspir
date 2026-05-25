"""
Log Viewer Dialog - View application log files.
"""
import os
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTextEdit, QComboBox,
    QPushButton, QLabel, QFileDialog, QCheckBox, QSplitter
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QTextCursor

from ui.utils.log_manager import get_log_manager


class LogViewerDialog(QDialog):
    """
    Dialog for viewing log files with auto-refresh capability.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("View Log")
        self.setMinimumSize(800, 600)

        self.log_manager = get_log_manager()
        self._auto_refresh = True
        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self._refresh_log)
        self._last_position = 0

        self._setup_ui()
        self._populate_log_files()
        self._refresh_log()

        # Start auto-refresh
        self._refresh_timer.start(2000)  # Refresh every 2 seconds

    def _setup_ui(self):
        """Setup the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Top controls
        controls_layout = QHBoxLayout()

        controls_layout.addWidget(QLabel("Log file:"))

        self.file_combo = QComboBox()
        self.file_combo.setMinimumWidth(300)
        self.file_combo.currentIndexChanged.connect(self._on_file_changed)
        controls_layout.addWidget(self.file_combo, 1)

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self._refresh_log)
        controls_layout.addWidget(self.refresh_btn)

        self.auto_refresh_check = QCheckBox("Auto-refresh")
        self.auto_refresh_check.setChecked(True)
        self.auto_refresh_check.toggled.connect(self._on_auto_refresh_toggled)
        controls_layout.addWidget(self.auto_refresh_check)

        layout.addLayout(controls_layout)

        # Filter controls
        filter_layout = QHBoxLayout()

        filter_layout.addWidget(QLabel("Filter:"))

        self.filter_debug = QCheckBox("DEBUG")
        self.filter_debug.setChecked(True)
        self.filter_debug.setStyleSheet("color: #666;")
        self.filter_debug.toggled.connect(self._refresh_log)
        filter_layout.addWidget(self.filter_debug)

        self.filter_info = QCheckBox("INFO")
        self.filter_info.setChecked(True)
        self.filter_info.setStyleSheet("color: #0066cc;")
        self.filter_info.toggled.connect(self._refresh_log)
        filter_layout.addWidget(self.filter_info)

        self.filter_warning = QCheckBox("WARNING")
        self.filter_warning.setChecked(True)
        self.filter_warning.setStyleSheet("color: #cc6600;")
        self.filter_warning.toggled.connect(self._refresh_log)
        filter_layout.addWidget(self.filter_warning)

        self.filter_error = QCheckBox("ERROR")
        self.filter_error.setChecked(True)
        self.filter_error.setStyleSheet("color: #cc0000;")
        self.filter_error.toggled.connect(self._refresh_log)
        filter_layout.addWidget(self.filter_error)

        filter_layout.addStretch()

        self.scroll_to_end_check = QCheckBox("Scroll to end")
        self.scroll_to_end_check.setChecked(True)
        filter_layout.addWidget(self.scroll_to_end_check)

        layout.addLayout(filter_layout)

        # Log content
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Monospace", 9))
        self.log_text.setLineWrapMode(QTextEdit.NoWrap)
        layout.addWidget(self.log_text, 1)

        # Bottom controls
        bottom_layout = QHBoxLayout()

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #666;")
        bottom_layout.addWidget(self.status_label, 1)

        self.open_folder_btn = QPushButton("Open Log Folder")
        self.open_folder_btn.clicked.connect(self._open_log_folder)
        bottom_layout.addWidget(self.open_folder_btn)

        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        bottom_layout.addWidget(self.close_btn)

        layout.addLayout(bottom_layout)

    def _populate_log_files(self):
        """Populate the file combo with available log files."""
        self.file_combo.clear()

        log_files = self.log_manager.get_log_files()
        current_log = self.log_manager.get_current_log_file()

        for filepath in log_files:
            filename = os.path.basename(filepath)
            if filepath == current_log:
                filename += " (current)"
            self.file_combo.addItem(filename, filepath)

        if not log_files:
            self.file_combo.addItem("No log files found", None)

    def _on_file_changed(self):
        """Handle log file selection change."""
        self._last_position = 0
        self._refresh_log()

    def _on_auto_refresh_toggled(self, checked):
        """Handle auto-refresh toggle."""
        self._auto_refresh = checked
        if checked:
            self._refresh_timer.start(2000)
        else:
            self._refresh_timer.stop()

    def _refresh_log(self):
        """Refresh the log content."""
        filepath = self.file_combo.currentData()
        if not filepath:
            self.log_text.setText("No log file selected.")
            self.status_label.setText("")
            return

        if not os.path.exists(filepath):
            self.log_text.setText(f"Log file not found: {filepath}")
            self.status_label.setText("")
            return

        try:
            content = self.log_manager.read_log_file(filepath, max_lines=5000)
            filtered_content = self._filter_content(content)

            # Only update if content changed
            current_content = self.log_text.toPlainText()
            if filtered_content != current_content:
                self.log_text.setText(filtered_content)

                if self.scroll_to_end_check.isChecked():
                    cursor = self.log_text.textCursor()
                    cursor.movePosition(QTextCursor.End)
                    self.log_text.setTextCursor(cursor)

            # Update status
            file_size = os.path.getsize(filepath)
            if file_size < 1024:
                size_str = f"{file_size} B"
            elif file_size < 1024 * 1024:
                size_str = f"{file_size / 1024:.1f} KB"
            else:
                size_str = f"{file_size / (1024 * 1024):.2f} MB"

            line_count = filtered_content.count('\n') + 1 if filtered_content else 0
            self.status_label.setText(f"{line_count} lines | {size_str}")

        except Exception as e:
            self.log_text.setText(f"Error reading log file: {e}")
            self.status_label.setText("")

    def _filter_content(self, content: str) -> str:
        """Filter log content based on selected levels."""
        if not content:
            return content

        # If all filters are on, return as-is
        if all([
            self.filter_debug.isChecked(),
            self.filter_info.isChecked(),
            self.filter_warning.isChecked(),
            self.filter_error.isChecked()
        ]):
            return content

        lines = content.split('\n')
        filtered_lines = []

        for line in lines:
            # Check each log level
            if " - DEBUG - " in line and not self.filter_debug.isChecked():
                continue
            if " - INFO - " in line and not self.filter_info.isChecked():
                continue
            if " - WARNING - " in line and not self.filter_warning.isChecked():
                continue
            if " - ERROR - " in line and not self.filter_error.isChecked():
                continue
            if " - CRITICAL - " in line and not self.filter_error.isChecked():
                continue

            filtered_lines.append(line)

        return '\n'.join(filtered_lines)

    def _open_log_folder(self):
        """Open the log folder in file manager."""
        from ui.utils.log_manager import LOG_DIR
        import subprocess
        import sys

        if sys.platform == "linux":
            subprocess.Popen(["xdg-open", str(LOG_DIR)])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(LOG_DIR)])
        elif sys.platform == "win32":
            subprocess.Popen(["explorer", str(LOG_DIR)])

    def closeEvent(self, event):
        """Stop timer when closing."""
        self._refresh_timer.stop()
        super().closeEvent(event)
