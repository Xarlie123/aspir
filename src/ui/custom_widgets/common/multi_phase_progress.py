"""
Multi-phase progress widget for showing progress of multiple sequential tasks.
"""
from typing import List, Dict, Optional
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar, QFrame, QSizePolicy
)
from PyQt5.QtCore import Qt


class PhaseProgressBar(QWidget):
    """A single phase progress bar with label and status (vertical layout for horizontal display)."""

    STATUS_COLORS = {
        "pending": "#9e9e9e",      # Gray
        "running": "#2196F3",      # Blue
        "completed": "#4CAF50",    # Green
        "failed": "#f44336",       # Red
    }

    STATUS_ICONS = {
        "pending": "○",
        "running": "◉",
        "completed": "✓",
        "failed": "✗",
    }

    def __init__(self, name: str, compact: bool = False, parent=None):
        super().__init__(parent)
        self.name = name
        self._status = "pending"
        self._compact = compact
        self._setup_ui()

    def _setup_ui(self):
        # Vertical layout for horizontal arrangement (icon + name on top, progress below)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        # Top row: status icon + phase name
        top_layout = QHBoxLayout()
        top_layout.setSpacing(4)

        self.status_icon = QLabel(self.STATUS_ICONS["pending"])
        self.status_icon.setFixedWidth(14)
        self.status_icon.setAlignment(Qt.AlignCenter)
        top_layout.addWidget(self.status_icon)

        self.name_label = QLabel(self.name)
        self.name_label.setStyleSheet("font-size: 10px;")
        self.name_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        top_layout.addWidget(self.name_label, 1)

        layout.addLayout(top_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setFixedHeight(14)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #ddd;
                border-radius: 3px;
                background-color: #f5f5f5;
                text-align: center;
                font-size: 9px;
            }
            QProgressBar::chunk {
                background-color: #2196F3;
                border-radius: 2px;
            }
        """)
        layout.addWidget(self.progress_bar)

        # Set size policy
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self._update_style()

    def _update_style(self):
        """Update visual style based on status."""
        color = self.STATUS_COLORS.get(self._status, "#9e9e9e")
        icon = self.STATUS_ICONS.get(self._status, "○")

        self.status_icon.setText(icon)
        self.status_icon.setStyleSheet(f"color: {color}; font-weight: bold;")

        if self._status == "pending":
            self.name_label.setStyleSheet("color: #999; font-size: 11px;")
            self.progress_bar.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #ddd;
                    border-radius: 3px;
                    background-color: #f5f5f5;
                    text-align: center;
                    font-size: 10px;
                }
                QProgressBar::chunk {
                    background-color: #bdbdbd;
                    border-radius: 2px;
                }
            """)
        elif self._status == "running":
            self.name_label.setStyleSheet("color: #2196F3; font-size: 11px; font-weight: bold;")
            self.progress_bar.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #2196F3;
                    border-radius: 3px;
                    background-color: #e3f2fd;
                    text-align: center;
                    font-size: 10px;
                }
                QProgressBar::chunk {
                    background-color: #2196F3;
                    border-radius: 2px;
                }
            """)
        elif self._status == "completed":
            self.name_label.setStyleSheet("color: #4CAF50; font-size: 11px;")
            self.progress_bar.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #4CAF50;
                    border-radius: 3px;
                    background-color: #e8f5e9;
                    text-align: center;
                    font-size: 10px;
                }
                QProgressBar::chunk {
                    background-color: #4CAF50;
                    border-radius: 2px;
                }
            """)
        elif self._status == "failed":
            self.name_label.setStyleSheet("color: #f44336; font-size: 11px;")
            self.progress_bar.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #f44336;
                    border-radius: 3px;
                    background-color: #ffebee;
                    text-align: center;
                    font-size: 10px;
                }
                QProgressBar::chunk {
                    background-color: #f44336;
                    border-radius: 2px;
                }
            """)

    def set_status(self, status: str):
        """Set phase status: pending, running, completed, failed."""
        self._status = status
        self._update_style()

    def set_progress(self, value: int):
        """Set progress value (0-100)."""
        self.progress_bar.setValue(value)

    def reset(self):
        """Reset to initial state."""
        self._status = "pending"
        self.progress_bar.setValue(0)
        self._update_style()


class MultiPhaseProgressWidget(QWidget):
    """
    Widget showing multiple progress bars for sequential phases in horizontal layout.

    Each phase has its own progress bar that shows:
    - Status icon (pending, running, completed, failed)
    - Phase name
    - Progress bar (0-100%)

    Layout: [Phase1] [Phase2] [Phase3] ... (side by side)
    """

    def __init__(self, phases: List[str], title: str = "Progress", show_title: bool = True, parent=None):
        """
        Args:
            phases: List of phase names (e.g., ["Masks", "Reconstruction", "Training", "Analysis"])
            title: Title for the progress section
            show_title: Whether to show the title label
        """
        super().__init__(parent)
        self.phases = phases
        self.title = title
        self.show_title = show_title
        self._phase_bars: Dict[str, PhaseProgressBar] = {}
        self._current_phase_index = -1
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Title (optional)
        if self.show_title:
            title_label = QLabel(self.title)
            title_label.setStyleSheet("font-weight: bold; font-size: 11px; color: #333;")
            layout.addWidget(title_label)

        # Horizontal layout for phase progress bars
        phases_layout = QHBoxLayout()
        phases_layout.setSpacing(8)
        phases_layout.setContentsMargins(0, 0, 0, 0)

        # Phase progress bars (side by side)
        for phase_name in self.phases:
            phase_bar = PhaseProgressBar(phase_name, compact=True)
            self._phase_bars[phase_name] = phase_bar
            phases_layout.addWidget(phase_bar, 1)  # stretch=1 for equal width

        layout.addLayout(phases_layout)

    def start_phase(self, phase_name: str):
        """Mark a phase as started/running."""
        if phase_name in self._phase_bars:
            # Mark previous phases as completed if not already
            found = False
            for name, bar in self._phase_bars.items():
                if name == phase_name:
                    bar.set_status("running")
                    bar.set_progress(0)
                    found = True
                elif not found and bar._status == "pending":
                    bar.set_status("completed")
                    bar.set_progress(100)

    def update_phase_progress(self, phase_name: str, progress: int):
        """Update progress for a specific phase."""
        if phase_name in self._phase_bars:
            bar = self._phase_bars[phase_name]
            if bar._status != "running":
                bar.set_status("running")
            bar.set_progress(progress)

    def complete_phase(self, phase_name: str):
        """Mark a phase as completed."""
        if phase_name in self._phase_bars:
            bar = self._phase_bars[phase_name]
            bar.set_status("completed")
            bar.set_progress(100)

    def fail_phase(self, phase_name: str):
        """Mark a phase as failed."""
        if phase_name in self._phase_bars:
            bar = self._phase_bars[phase_name]
            bar.set_status("failed")

    def reset_all(self):
        """Reset all phases to pending state."""
        for bar in self._phase_bars.values():
            bar.reset()

    def set_all_completed(self):
        """Mark all phases as completed."""
        for bar in self._phase_bars.values():
            bar.set_status("completed")
            bar.set_progress(100)
