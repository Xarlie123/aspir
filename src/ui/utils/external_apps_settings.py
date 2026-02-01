"""
External Applications Settings Dialog and Manager.

Manages configuration for external tools required by the application:
- pdflatex: For DNN architecture preview (PlotNeuralNet)
- nsys: NVIDIA Nsight Systems for profiling
- kaggle: Kaggle CLI for dataset downloads
"""
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QPushButton, QLineEdit, QGroupBox, QFileDialog, QMessageBox,
    QFrame, QSizePolicy
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont


class ExternalApp:
    """Represents an external application configuration."""

    def __init__(self, name: str, display_name: str, description: str,
                 check_command: list, install_hint: str):
        self.name = name
        self.display_name = display_name
        self.description = description
        self.check_command = check_command  # Command to check availability
        self.install_hint = install_hint
        self.custom_path: str = ""
        self.is_available: bool = False
        self.version: str = ""


class ExternalAppsManager:
    """
    Manager for external application paths and availability.

    Stores settings in ~/.config/ir_beam/external_apps.json
    """

    CONFIG_DIR = Path.home() / ".config" / "ir_beam"
    CONFIG_FILE = CONFIG_DIR / "external_apps.json"

    # Define supported external applications
    APPS = {
        "pdflatex": ExternalApp(
            name="pdflatex",
            display_name="pdflatex (LaTeX)",
            description="Required for DNN architecture preview diagrams",
            check_command=["pdflatex", "--version"],
            install_hint="Install with: sudo apt install texlive-latex-base texlive-latex-extra"
        ),
        "nsys": ExternalApp(
            name="nsys",
            display_name="nsys (NVIDIA Nsight Systems)",
            description="Required for GPU profiling and timeline analysis",
            check_command=["nsys", "--version"],
            install_hint="Install NVIDIA Nsight Systems from NVIDIA Developer website"
        ),
        "kaggle": ExternalApp(
            name="kaggle",
            display_name="kaggle (Python package)",
            description="Required for downloading datasets from Kaggle",
            check_command=[sys.executable, "-m", "kaggle.cli", "--version"],
            install_hint="Install with: pip install kaggle"
        ),
    }

    def __init__(self, logger=None):
        if logger:
            self.logger = logger.getChild("ExternalAppsManager")
        else:
            self.logger = logging.getLogger("ExternalAppsManager")

        self._ensure_config_dir()
        self._load_config()
        self.check_all_apps()

    def _ensure_config_dir(self):
        """Create config directory if it doesn't exist."""
        self.CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    def _load_config(self):
        """Load saved configuration from file."""
        if self.CONFIG_FILE.exists():
            try:
                with open(self.CONFIG_FILE, 'r') as f:
                    data = json.load(f)
                    for app_name, app_config in data.items():
                        if app_name in self.APPS:
                            self.APPS[app_name].custom_path = app_config.get("custom_path", "")
                self.logger.info("Loaded external apps config from %s", self.CONFIG_FILE)
            except Exception as e:
                self.logger.warning("Failed to load config: %s", e)

    def save_config(self):
        """Save configuration to file."""
        try:
            data = {}
            for app_name, app in self.APPS.items():
                data[app_name] = {
                    "custom_path": app.custom_path
                }
            with open(self.CONFIG_FILE, 'w') as f:
                json.dump(data, f, indent=2)
            self.logger.info("Saved external apps config to %s", self.CONFIG_FILE)
        except Exception as e:
            self.logger.error("Failed to save config: %s", e)

    def check_app(self, app_name: str) -> Tuple[bool, str]:
        """
        Check if an application is available.

        Returns:
            Tuple of (is_available, version_or_error)
        """
        if app_name not in self.APPS:
            return False, "Unknown application"

        app = self.APPS[app_name]

        # Build command with custom path if set
        if app.custom_path:
            if app_name == "kaggle":
                # Kaggle uses Python module, custom_path would be python executable
                cmd = [app.custom_path, "-m", "kaggle", "--version"]
            else:
                cmd = [app.custom_path] + app.check_command[1:]
        else:
            cmd = app.check_command

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                # Extract version from output
                version = result.stdout.strip().split('\n')[0][:50]
                app.is_available = True
                app.version = version
                return True, version
            else:
                app.is_available = False
                app.version = ""
                return False, result.stderr.strip()[:100] if result.stderr else "Command failed"
        except FileNotFoundError:
            app.is_available = False
            app.version = ""
            return False, "Not found in PATH"
        except subprocess.TimeoutExpired:
            app.is_available = False
            app.version = ""
            return False, "Timeout checking application"
        except Exception as e:
            app.is_available = False
            app.version = ""
            return False, str(e)

    def check_all_apps(self):
        """Check availability of all applications."""
        for app_name in self.APPS:
            self.check_app(app_name)

    def get_executable_path(self, app_name: str) -> Optional[str]:
        """
        Get the executable path for an application.

        Returns custom path if set, otherwise returns the default command.
        Returns None if app is not available.
        """
        if app_name not in self.APPS:
            return None

        app = self.APPS[app_name]

        if app.custom_path:
            return app.custom_path

        # Return default command name (will use PATH)
        if app.is_available:
            return app.check_command[0]

        return None

    def set_custom_path(self, app_name: str, path: str):
        """Set custom path for an application."""
        if app_name in self.APPS:
            self.APPS[app_name].custom_path = path
            self.check_app(app_name)


class ExternalAppsSettingsDialog(QDialog):
    """Dialog for configuring external application paths."""

    def __init__(self, manager: ExternalAppsManager, parent=None):
        super().__init__(parent)
        self.manager = manager
        self._path_edits: Dict[str, QLineEdit] = {}
        self._status_labels: Dict[str, QLabel] = {}
        self._setup_ui()
        self._refresh_status()

    def _setup_ui(self):
        self.setWindowTitle("External Applications Settings")
        self.setMinimumWidth(600)
        self.setMinimumHeight(400)

        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Title
        title = QLabel("External Applications")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(title)

        desc = QLabel(
            "Configure paths to external applications. If left empty, "
            "the application will be searched in the system PATH."
        )
        desc.setStyleSheet("color: #666;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Create a group for each application
        for app_name, app in self.manager.APPS.items():
            group = self._create_app_group(app_name, app)
            layout.addWidget(group)

        layout.addStretch()

        # Buttons
        btn_layout = QHBoxLayout()

        check_all_btn = QPushButton("Check All")
        check_all_btn.clicked.connect(self._on_check_all)
        btn_layout.addWidget(check_all_btn)

        btn_layout.addStretch()

        save_btn = QPushButton("Save")
        save_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 8px 20px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        save_btn.clicked.connect(self._on_save)
        btn_layout.addWidget(save_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        btn_layout.addWidget(close_btn)

        layout.addLayout(btn_layout)

    def _create_app_group(self, app_name: str, app: ExternalApp) -> QGroupBox:
        """Create a group box for an application configuration."""
        group = QGroupBox(app.display_name)
        group_layout = QVBoxLayout(group)
        group_layout.setSpacing(8)

        # Description
        desc_label = QLabel(app.description)
        desc_label.setStyleSheet("color: #555; font-size: 11px;")
        group_layout.addWidget(desc_label)

        # Path configuration row
        path_layout = QHBoxLayout()
        path_layout.setSpacing(8)

        path_label = QLabel("Path:")
        path_label.setFixedWidth(40)
        path_layout.addWidget(path_label)

        path_edit = QLineEdit()
        path_edit.setPlaceholderText("Leave empty to use system PATH")
        path_edit.setText(app.custom_path)
        path_edit.textChanged.connect(lambda text, n=app_name: self._on_path_changed(n, text))
        self._path_edits[app_name] = path_edit
        path_layout.addWidget(path_edit)

        browse_btn = QPushButton("Browse...")
        browse_btn.setFixedWidth(80)
        browse_btn.clicked.connect(lambda checked, n=app_name: self._on_browse(n))
        path_layout.addWidget(browse_btn)

        check_btn = QPushButton("Check")
        check_btn.setFixedWidth(60)
        check_btn.clicked.connect(lambda checked, n=app_name: self._on_check(n))
        path_layout.addWidget(check_btn)

        group_layout.addLayout(path_layout)

        # Status row
        status_layout = QHBoxLayout()
        status_layout.setSpacing(8)

        status_icon = QLabel()
        status_icon.setFixedWidth(40)
        status_layout.addWidget(status_icon)

        status_label = QLabel()
        status_label.setWordWrap(True)
        self._status_labels[app_name] = status_label
        status_layout.addWidget(status_label, 1)

        group_layout.addLayout(status_layout)

        # Install hint
        hint_label = QLabel(f"<i>{app.install_hint}</i>")
        hint_label.setStyleSheet("color: #888; font-size: 10px;")
        hint_label.setWordWrap(True)
        group_layout.addWidget(hint_label)

        return group

    def _on_path_changed(self, app_name: str, text: str):
        """Handle path text change."""
        self.manager.set_custom_path(app_name, text)

    def _on_browse(self, app_name: str):
        """Open file browser to select executable."""
        app = self.manager.APPS[app_name]

        if app_name == "kaggle":
            # Kaggle uses Python, so browse for Python executable
            file_filter = "Python Executable (python python3 python.exe);;All Files (*)"
            title = "Select Python Executable with Kaggle installed"
        else:
            file_filter = "Executable Files (*);;All Files (*)"
            title = f"Select {app.display_name} Executable"

        path, _ = QFileDialog.getOpenFileName(
            self, title, "", file_filter
        )

        if path:
            self._path_edits[app_name].setText(path)
            self.manager.set_custom_path(app_name, path)
            self._update_status(app_name)

    def _on_check(self, app_name: str):
        """Check availability of a single application."""
        self._update_status(app_name)

    def _on_check_all(self):
        """Check all applications."""
        self._refresh_status()

    def _update_status(self, app_name: str):
        """Update status display for an application."""
        is_available, info = self.manager.check_app(app_name)

        if is_available:
            self._status_labels[app_name].setText(
                f"<span style='color: green;'>✓ Available</span> - {info}"
            )
        else:
            self._status_labels[app_name].setText(
                f"<span style='color: red;'>✗ Not available</span> - {info}"
            )

    def _refresh_status(self):
        """Refresh status for all applications."""
        for app_name in self.manager.APPS:
            self._update_status(app_name)

    def _on_save(self):
        """Save configuration and close."""
        self.manager.save_config()
        QMessageBox.information(
            self, "Settings Saved",
            "External applications settings have been saved."
        )
        self.accept()


# Global manager instance (lazy initialization)
_manager_instance: Optional[ExternalAppsManager] = None


def get_external_apps_manager(logger=None) -> ExternalAppsManager:
    """Get or create the global ExternalAppsManager instance."""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = ExternalAppsManager(logger)
    return _manager_instance
