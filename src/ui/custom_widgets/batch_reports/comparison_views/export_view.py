"""
Export view for Batch Reports - allows generating publication-quality figures and interactive HTML reports.
"""
import logging
import io
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox,
    QPushButton, QFileDialog, QMessageBox, QScrollArea,
    QGridLayout, QFrame, QDialog, QComboBox, QSpinBox,
    QCheckBox, QLineEdit, QSizePolicy, QSplitter, QTextEdit
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from ui.utils.file_formats import BATCH_TESTS_DIR


class ExportView(QWidget):
    """
    Export view for generating publication-quality figures and reports.

    Features:
    - Generate Visual Comparison Figure (multiple reconstruction methods)
    - Generate Quality per Sampling Ratio Figure (table format)
    - Generate Samples Grid Figure (multiple samples at different ratios)
    - Export Interactive HTML Report with all charts
    """

    def __init__(self, logger=None, parent=None):
        super().__init__(parent)

        if logger:
            self.logger = logger.getChild("ExportView")
        else:
            self.logger = logging.getLogger("ExportView")

        self._tests: List[Dict[str, Any]] = []
        self._setup_ui()

    def _setup_ui(self):
        """Setup the export view UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # Header
        header = QLabel("Export Publication Figures & Reports")
        header.setStyleSheet("font-size: 18px; font-weight: bold; color: #333;")
        layout.addWidget(header)

        desc = QLabel(
            "Generate publication-quality figures and interactive reports from your batch test results. "
            "Each button opens a configuration dialog to customize the output."
        )
        desc.setStyleSheet("color: #666; font-size: 12px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Buttons grid
        buttons_group = QGroupBox("Figure Generation")
        buttons_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 13px;
                border: 1px solid #ccc;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 8px;
            }
        """)
        buttons_layout = QGridLayout(buttons_group)
        buttons_layout.setSpacing(15)
        buttons_layout.setContentsMargins(15, 20, 15, 15)

        # Button 1: Visual Comparison (Fig 9)
        self.fig9_btn = self._create_export_button(
            "Visual Comparison\n(Reconstruction Methods)",
            "Compare Ground Truth, Linear Recon, Iterative CS, and DNN output\nwith PSNR and timing metrics",
            "#4CAF50"
        )
        self.fig9_btn.clicked.connect(self._on_generate_fig9)
        buttons_layout.addWidget(self.fig9_btn, 0, 0)

        # Button 2: Quality per Sampling Ratio (Fig 8)
        self.fig8_btn = self._create_export_button(
            "Quality vs Sampling Ratio\n(Single Image Analysis)",
            "Show reconstructed and denoised images at different\nsampling ratios with quality metrics",
            "#2196F3"
        )
        self.fig8_btn.clicked.connect(self._on_generate_fig8)
        buttons_layout.addWidget(self.fig8_btn, 0, 1)

        # Button 3: Samples Grid (Fig 2)
        self.fig2_btn = self._create_export_button(
            "Samples Grid\n(Multiple Images)",
            "Grid of multiple samples showing reconstructions\nat different sampling ratios",
            "#FF9800"
        )
        self.fig2_btn.clicked.connect(self._on_generate_fig2)
        buttons_layout.addWidget(self.fig2_btn, 1, 0)

        # Button 4: Interactive HTML
        self.html_btn = self._create_export_button(
            "Interactive HTML Report\n(All Charts)",
            "Export all comparison charts to an interactive\nHTML report using Plotly",
            "#9C27B0"
        )
        self.html_btn.clicked.connect(self._on_generate_html)
        buttons_layout.addWidget(self.html_btn, 1, 1)

        layout.addWidget(buttons_group)

        # Info panel
        info_frame = QFrame()
        info_frame.setStyleSheet("""
            QFrame {
                background-color: #e3f2fd;
                border-radius: 6px;
                padding: 15px;
            }
        """)
        info_layout = QVBoxLayout(info_frame)
        info_layout.setContentsMargins(15, 12, 15, 12)

        self.info_label = QLabel("Load experiments to enable figure generation")
        self.info_label.setStyleSheet("color: #1976d2; font-size: 12px;")
        info_layout.addWidget(self.info_label)

        layout.addWidget(info_frame)

        layout.addStretch()

    def _create_export_button(self, title: str, description: str, color: str) -> QPushButton:
        """Create a styled export button."""
        btn = QPushButton()
        btn.setMinimumSize(280, 120)
        btn.setEnabled(False)
        btn.setCursor(Qt.PointingHandCursor)

        # Use HTML for multi-line button with title and description
        btn.setText(f"{title}")
        btn.setToolTip(description)

        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 13px;
                font-weight: bold;
                text-align: center;
                padding: 15px;
            }}
            QPushButton:hover:enabled {{
                background-color: {self._darken_color(color)};
            }}
            QPushButton:disabled {{
                background-color: #ccc;
                color: #888;
            }}
        """)

        return btn

    def _darken_color(self, hex_color: str) -> str:
        """Darken a hex color by 15%."""
        hex_color = hex_color.lstrip('#')
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        r = max(0, int(r * 0.85))
        g = max(0, int(g * 0.85))
        b = max(0, int(b * 0.85))
        return f"#{r:02x}{g:02x}{b:02x}"

    def set_tests(self, tests: List[Dict[str, Any]]):
        """Set the tests to export."""
        self._tests = tests
        self._update_state()

    def _update_state(self):
        """Update UI state based on current data."""
        has_tests = bool(self._tests)

        self.fig9_btn.setEnabled(has_tests)
        self.fig8_btn.setEnabled(has_tests)
        self.fig2_btn.setEnabled(has_tests)
        self.html_btn.setEnabled(has_tests)

        if has_tests:
            experiment_count = len(set(t.get("_experiment_name", "") for t in self._tests))
            self.info_label.setText(
                f"Ready to generate figures from {len(self._tests)} tests "
                f"({experiment_count} experiment(s))"
            )
        else:
            self.info_label.setText("Load experiments to enable figure generation")

    def _on_generate_fig9(self):
        """Open dialog for Visual Comparison figure (Fig 9)."""
        from ui.custom_widgets.batch_reports.comparison_views.figure_export_popups import (
            VisualComparisonPopup
        )
        popup = VisualComparisonPopup(self._tests, logger=self.logger, parent=self)
        popup.exec_()

    def _on_generate_fig8(self):
        """Open dialog for Quality per Sampling Ratio figure (Fig 8)."""
        from ui.custom_widgets.batch_reports.comparison_views.figure_export_popups import (
            QualitySamplingRatioPopup
        )
        popup = QualitySamplingRatioPopup(self._tests, logger=self.logger, parent=self)
        popup.exec_()

    def _on_generate_fig2(self):
        """Open dialog for Samples Grid figure (Fig 2)."""
        from ui.custom_widgets.batch_reports.comparison_views.figure_export_popups import (
            SamplesGridPopup
        )
        popup = SamplesGridPopup(self._tests, logger=self.logger, parent=self)
        popup.exec_()

    def _on_generate_html(self):
        """Open dialog for Interactive HTML export."""
        from ui.custom_widgets.batch_reports.comparison_views.figure_export_popups import (
            InteractiveHTMLPopup
        )
        popup = InteractiveHTMLPopup(self._tests, logger=self.logger, parent=self)
        popup.exec_()

    def clear(self):
        """Clear all data from the view."""
        self._tests = []
        self._update_state()
