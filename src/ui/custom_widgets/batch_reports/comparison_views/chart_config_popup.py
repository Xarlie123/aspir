"""
Chart Configuration Popup for Batch Reports views (Quality, Training, etc.).

Provides tabbed configuration for axes, legend, colors, and line/marker settings.
Includes a custom NavigationToolbar with extended settings button.
Reusable across all chart views in Batch Reports.
"""
import logging
from typing import Dict, Any, Callable, Optional

from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget,
    QPushButton, QGroupBox, QSpinBox, QComboBox, QLabel,
    QCheckBox, QColorDialog, QGridLayout, QFrame, QLineEdit,
    QDoubleSpinBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QIcon


class ColorButton(QPushButton):
    """Button that displays and allows selecting a color."""

    def __init__(self, color: str = "#1f77b4", parent=None):
        super().__init__(parent)
        self._color = color
        self.setFixedSize(60, 24)
        self._update_style()
        self.clicked.connect(self._on_click)

    def _update_style(self):
        """Update button style to show the color."""
        # Calculate contrasting text color
        qcolor = QColor(self._color)
        brightness = (qcolor.red() * 299 + qcolor.green() * 587 + qcolor.blue() * 114) / 1000
        text_color = "#000000" if brightness > 128 else "#ffffff"

        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {self._color};
                color: {text_color};
                border: 1px solid #888;
                border-radius: 3px;
                font-size: 10px;
            }}
            QPushButton:hover {{
                border: 2px solid #333;
            }}
        """)
        self.setText(self._color.upper())

    def _on_click(self):
        """Open color dialog."""
        color = QColorDialog.getColor(QColor(self._color), self, "Select Color")
        if color.isValid():
            self._color = color.name()
            self._update_style()

    def get_color(self) -> str:
        """Return the current color."""
        return self._color

    def set_color(self, color: str):
        """Set the color."""
        self._color = color
        self._update_style()


class LegendConfigTab(QWidget):
    """Tab for legend configuration."""

    def __init__(self, logger=None, parent=None):
        super().__init__(parent)
        self.logger = logger or logging.getLogger(__name__)
        self._setup_ui()

    def _setup_ui(self):
        """Setup the legend configuration UI with 2-column layout."""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)

        # Two-column layout
        columns_layout = QHBoxLayout()
        columns_layout.setSpacing(15)

        # === LEFT COLUMN ===
        left_column = QVBoxLayout()
        left_column.setSpacing(10)

        # Position group
        pos_group = QGroupBox("Position")
        pos_layout = QGridLayout(pos_group)
        pos_layout.setSpacing(8)

        pos_layout.addWidget(QLabel("Location:"), 0, 0)
        self.position_combo = QComboBox()
        self.position_combo.addItems([
            "Inside (upper right)",
            "Inside (upper left)",
            "Inside (lower right)",
            "Inside (lower left)",
            "Right side (outside)",
            "Below (outside)"
        ])
        self.position_combo.setCurrentIndex(0)
        pos_layout.addWidget(self.position_combo, 0, 1)

        left_column.addWidget(pos_group)

        # Font group
        font_group = QGroupBox("Font")
        font_layout = QGridLayout(font_group)
        font_layout.setSpacing(8)

        font_layout.addWidget(QLabel("Size:"), 0, 0)
        self.fontsize_spin = QSpinBox()
        self.fontsize_spin.setMinimum(6)
        self.fontsize_spin.setMaximum(18)
        self.fontsize_spin.setValue(9)
        self.fontsize_spin.setSuffix(" pt")
        font_layout.addWidget(self.fontsize_spin, 0, 1)

        font_layout.addWidget(QLabel("Columns:"), 1, 0)
        self.ncol_spin = QSpinBox()
        self.ncol_spin.setMinimum(1)
        self.ncol_spin.setMaximum(10)
        self.ncol_spin.setValue(1)
        self.ncol_spin.setToolTip("Number of columns in the legend (1 = vertical list)")
        font_layout.addWidget(self.ncol_spin, 1, 1)

        left_column.addWidget(font_group)
        left_column.addStretch()

        # === RIGHT COLUMN ===
        right_column = QVBoxLayout()
        right_column.setSpacing(10)

        # Style group
        style_group = QGroupBox("Style")
        style_layout = QGridLayout(style_group)
        style_layout.setSpacing(8)

        self.frameon_check = QCheckBox("Show frame")
        self.frameon_check.setChecked(True)
        style_layout.addWidget(self.frameon_check, 0, 0)

        self.shadow_check = QCheckBox("Show shadow")
        self.shadow_check.setChecked(False)
        style_layout.addWidget(self.shadow_check, 0, 1)

        self.fancybox_check = QCheckBox("Rounded corners")
        self.fancybox_check.setChecked(True)
        style_layout.addWidget(self.fancybox_check, 1, 0)

        # Frame alpha
        style_layout.addWidget(QLabel("Opacity:"), 2, 0)
        self.framealpha_spin = QSpinBox()
        self.framealpha_spin.setMinimum(0)
        self.framealpha_spin.setMaximum(100)
        self.framealpha_spin.setValue(80)
        self.framealpha_spin.setSuffix(" %")
        style_layout.addWidget(self.framealpha_spin, 2, 1)

        right_column.addWidget(style_group)
        right_column.addStretch()

        # Add columns to main layout
        columns_layout.addLayout(left_column)
        columns_layout.addLayout(right_column)
        main_layout.addLayout(columns_layout)

    def get_config(self) -> Dict[str, Any]:
        """Return the legend configuration."""
        return {
            'position': self.position_combo.currentIndex(),
            'position_text': self.position_combo.currentText(),
            'fontsize': self.fontsize_spin.value(),
            'frameon': self.frameon_check.isChecked(),
            'shadow': self.shadow_check.isChecked(),
            'fancybox': self.fancybox_check.isChecked(),
            'framealpha': self.framealpha_spin.value() / 100.0,
            'ncol': self.ncol_spin.value(),
        }

    def set_config(self, config: Dict[str, Any]):
        """Set the legend configuration."""
        if 'position' in config:
            self.position_combo.setCurrentIndex(config['position'])
        if 'fontsize' in config:
            self.fontsize_spin.setValue(config['fontsize'])
        if 'frameon' in config:
            self.frameon_check.setChecked(config['frameon'])
        if 'shadow' in config:
            self.shadow_check.setChecked(config['shadow'])
        if 'fancybox' in config:
            self.fancybox_check.setChecked(config['fancybox'])
        if 'framealpha' in config:
            self.framealpha_spin.setValue(int(config['framealpha'] * 100))
        if 'ncol' in config:
            self.ncol_spin.setValue(config['ncol'])


class ColorsConfigTab(QWidget):
    """Tab for colors configuration."""

    def __init__(self, logger=None, parent=None):
        super().__init__(parent)
        self.logger = logger or logging.getLogger(__name__)
        self._setup_ui()

    def _setup_ui(self):
        """Setup the colors configuration UI with 2-column layout."""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)

        # Two-column layout
        columns_layout = QHBoxLayout()
        columns_layout.setSpacing(15)

        # === LEFT COLUMN ===
        left_column = QVBoxLayout()
        left_column.setSpacing(10)

        # Metric colors group
        metrics_group = QGroupBox("Metric Colors")
        metrics_layout = QGridLayout(metrics_group)
        metrics_layout.setSpacing(8)

        metrics_layout.addWidget(QLabel("PSNR:"), 0, 0)
        self.psnr_color = ColorButton("#1f77b4")
        metrics_layout.addWidget(self.psnr_color, 0, 1)

        metrics_layout.addWidget(QLabel("SSIM:"), 1, 0)
        self.ssim_color = ColorButton("#2ca02c")
        metrics_layout.addWidget(self.ssim_color, 1, 1)

        metrics_layout.addWidget(QLabel("LPIPS:"), 2, 0)
        self.lpips_color = ColorButton("#d62728")
        metrics_layout.addWidget(self.lpips_color, 2, 1)

        # Reset button
        reset_metrics_btn = QPushButton("Reset")
        reset_metrics_btn.clicked.connect(self._reset_metric_colors)
        metrics_layout.addWidget(reset_metrics_btn, 3, 0, 1, 2)

        left_column.addWidget(metrics_group)

        # Opacity group
        opacity_group = QGroupBox("Opacity")
        opacity_layout = QGridLayout(opacity_group)
        opacity_layout.setSpacing(8)

        opacity_layout.addWidget(QLabel("Bars:"), 0, 0)
        self.bar_alpha_spin = QSpinBox()
        self.bar_alpha_spin.setMinimum(10)
        self.bar_alpha_spin.setMaximum(100)
        self.bar_alpha_spin.setValue(80)
        self.bar_alpha_spin.setSuffix(" %")
        opacity_layout.addWidget(self.bar_alpha_spin, 0, 1)

        opacity_layout.addWidget(QLabel("Histogram:"), 1, 0)
        self.hist_alpha_spin = QSpinBox()
        self.hist_alpha_spin.setMinimum(10)
        self.hist_alpha_spin.setMaximum(100)
        self.hist_alpha_spin.setValue(70)
        self.hist_alpha_spin.setSuffix(" %")
        opacity_layout.addWidget(self.hist_alpha_spin, 1, 1)

        left_column.addWidget(opacity_group)
        left_column.addStretch()

        # === RIGHT COLUMN ===
        right_column = QVBoxLayout()
        right_column.setSpacing(10)

        # Histogram colors group
        histogram_group = QGroupBox("Histogram Colors")
        histogram_layout = QGridLayout(histogram_group)
        histogram_layout.setSpacing(8)

        histogram_layout.addWidget(QLabel("Noisy:"), 0, 0)
        self.noisy_color = ColorButton("#1f77b4")
        histogram_layout.addWidget(self.noisy_color, 0, 1)

        histogram_layout.addWidget(QLabel("Denoised:"), 1, 0)
        self.denoised_color = ColorButton("#2ca02c")
        histogram_layout.addWidget(self.denoised_color, 1, 1)

        # Reset button
        reset_hist_btn = QPushButton("Reset")
        reset_hist_btn.clicked.connect(self._reset_histogram_colors)
        histogram_layout.addWidget(reset_hist_btn, 2, 0, 1, 2)

        right_column.addWidget(histogram_group)

        # Line-chart series colors — used by "PSNR vs Sampling Ratio".
        # Defaults are matplotlib's classic tab:blue / tab:orange so the
        # plot ships with the Tableau look out of the box.
        lines_group = QGroupBox("Lines (PSNR vs M/N)")
        lines_layout = QGridLayout(lines_group)
        lines_layout.setSpacing(8)

        lines_layout.addWidget(QLabel("Reconstructed:"), 0, 0)
        self.recon_line_color = ColorButton("#1f77b4")
        lines_layout.addWidget(self.recon_line_color, 0, 1)

        lines_layout.addWidget(QLabel("Denoised:"), 1, 0)
        self.denoised_line_color = ColorButton("#ff7f0e")
        lines_layout.addWidget(self.denoised_line_color, 1, 1)

        reset_lines_btn = QPushButton("Reset")
        reset_lines_btn.clicked.connect(self._reset_line_colors)
        lines_layout.addWidget(reset_lines_btn, 2, 0, 1, 2)

        right_column.addWidget(lines_group)
        right_column.addStretch()

        # Add columns to main layout
        columns_layout.addLayout(left_column)
        columns_layout.addLayout(right_column)
        main_layout.addLayout(columns_layout)

    def _reset_metric_colors(self):
        """Reset metric colors to defaults."""
        self.psnr_color.set_color("#1f77b4")
        self.ssim_color.set_color("#2ca02c")
        self.lpips_color.set_color("#d62728")

    def _reset_histogram_colors(self):
        """Reset histogram colors to defaults."""
        self.noisy_color.set_color("#1f77b4")
        self.denoised_color.set_color("#2ca02c")

    def _reset_line_colors(self):
        """Reset line-chart series colors to the Tableau-style defaults."""
        self.recon_line_color.set_color("#1f77b4")
        self.denoised_line_color.set_color("#ff7f0e")

    def get_config(self) -> Dict[str, Any]:
        """Return the colors configuration."""
        return {
            'psnr': self.psnr_color.get_color(),
            'ssim': self.ssim_color.get_color(),
            'lpips': self.lpips_color.get_color(),
            'noisy': self.noisy_color.get_color(),
            'denoised': self.denoised_color.get_color(),
            'recon_line': self.recon_line_color.get_color(),
            'denoised_line': self.denoised_line_color.get_color(),
            'bar_alpha': self.bar_alpha_spin.value() / 100.0,
            'hist_alpha': self.hist_alpha_spin.value() / 100.0,
        }

    def set_config(self, config: Dict[str, Any]):
        """Set the colors configuration."""
        if 'psnr' in config:
            self.psnr_color.set_color(config['psnr'])
        if 'ssim' in config:
            self.ssim_color.set_color(config['ssim'])
        if 'lpips' in config:
            self.lpips_color.set_color(config['lpips'])
        if 'noisy' in config:
            self.noisy_color.set_color(config['noisy'])
        if 'denoised' in config:
            self.denoised_color.set_color(config['denoised'])
        if 'recon_line' in config:
            self.recon_line_color.set_color(config['recon_line'])
        if 'denoised_line' in config:
            self.denoised_line_color.set_color(config['denoised_line'])
        if 'bar_alpha' in config:
            self.bar_alpha_spin.setValue(int(config['bar_alpha'] * 100))
        if 'hist_alpha' in config:
            self.hist_alpha_spin.setValue(int(config['hist_alpha'] * 100))


class AxesConfigTab(QWidget):
    """Tab for axes configuration (title, labels, fonts, scales)."""

    def __init__(self, logger=None, parent=None):
        super().__init__(parent)
        self.logger = logger or logging.getLogger(__name__)
        self._setup_ui()

    def _setup_ui(self):
        """Setup the axes configuration UI with 2-column layout."""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)

        # Two-column layout
        columns_layout = QHBoxLayout()
        columns_layout.setSpacing(15)

        # === LEFT COLUMN ===
        left_column = QVBoxLayout()
        left_column.setSpacing(10)

        # Title group
        title_group = QGroupBox("Title")
        title_layout = QGridLayout(title_group)
        title_layout.setSpacing(8)

        title_layout.addWidget(QLabel("Text:"), 0, 0)
        self.title_edit = QLineEdit()
        self.title_edit.setPlaceholderText("(auto)")
        self.title_edit.setToolTip("Leave empty to use automatic title")
        title_layout.addWidget(self.title_edit, 0, 1)

        title_layout.addWidget(QLabel("Font size:"), 1, 0)
        self.title_fontsize_spin = QSpinBox()
        self.title_fontsize_spin.setMinimum(8)
        self.title_fontsize_spin.setMaximum(24)
        self.title_fontsize_spin.setValue(13)
        self.title_fontsize_spin.setSuffix(" pt")
        title_layout.addWidget(self.title_fontsize_spin, 1, 1)

        left_column.addWidget(title_group)

        # X-Axis group
        xaxis_group = QGroupBox("X-Axis")
        xaxis_layout = QGridLayout(xaxis_group)
        xaxis_layout.setSpacing(8)

        xaxis_layout.addWidget(QLabel("Label:"), 0, 0)
        self.xlabel_edit = QLineEdit()
        self.xlabel_edit.setPlaceholderText("(auto)")
        xaxis_layout.addWidget(self.xlabel_edit, 0, 1)

        xaxis_layout.addWidget(QLabel("Label font:"), 1, 0)
        self.xlabel_fontsize_spin = QSpinBox()
        self.xlabel_fontsize_spin.setMinimum(6)
        self.xlabel_fontsize_spin.setMaximum(18)
        self.xlabel_fontsize_spin.setValue(11)
        self.xlabel_fontsize_spin.setSuffix(" pt")
        xaxis_layout.addWidget(self.xlabel_fontsize_spin, 1, 1)

        xaxis_layout.addWidget(QLabel("Tick font:"), 2, 0)
        self.xtick_fontsize_spin = QSpinBox()
        self.xtick_fontsize_spin.setMinimum(6)
        self.xtick_fontsize_spin.setMaximum(14)
        self.xtick_fontsize_spin.setValue(8)
        self.xtick_fontsize_spin.setSuffix(" pt")
        xaxis_layout.addWidget(self.xtick_fontsize_spin, 2, 1)

        # Padding between the X label ("Sampling Ratio", "Time (ms)" …)
        # and the axis itself. matplotlib calls this ``labelpad`` and
        # measures it in points; sensible range 0–40.
        xaxis_layout.addWidget(QLabel("Label pad:"), 3, 0)
        self.xlabel_pad_spin = QSpinBox()
        self.xlabel_pad_spin.setMinimum(0)
        self.xlabel_pad_spin.setMaximum(60)
        self.xlabel_pad_spin.setValue(4)
        self.xlabel_pad_spin.setSuffix(" pt")
        self.xlabel_pad_spin.setToolTip(
            "Distance between the X-axis title and the axis."
        )
        xaxis_layout.addWidget(self.xlabel_pad_spin, 3, 1)

        # Some charts (e.g. Pipeline Latency Breakdown) draw a second
        # tier of X labels below the tick labels — the sampling ratio
        # group names ("4%", "8%" …). Charts without that second tier
        # ignore this value, so it's safe to keep here.
        xaxis_layout.addWidget(QLabel("Group label font:"), 4, 0)
        self.group_label_fontsize_spin = QSpinBox()
        self.group_label_fontsize_spin.setMinimum(6)
        self.group_label_fontsize_spin.setMaximum(18)
        self.group_label_fontsize_spin.setValue(9)
        self.group_label_fontsize_spin.setSuffix(" pt")
        self.group_label_fontsize_spin.setToolTip(
            "Font size for the secondary X tier (test/group names "
            "below the tick labels). Only used by charts that draw it."
        )
        xaxis_layout.addWidget(self.group_label_fontsize_spin, 4, 1)

        left_column.addWidget(xaxis_group)

        # Data labels group — controls the numeric labels drawn on top
        # of bars / data points. Lives in the left column below the
        # X-Axis group so the dialog height stays roughly even.
        bar_group = QGroupBox("Data labels")
        bar_layout = QGridLayout(bar_group)
        bar_layout.setSpacing(8)

        bar_layout.addWidget(QLabel("Value font:"), 0, 0)
        self.bar_label_fontsize_spin = QSpinBox()
        self.bar_label_fontsize_spin.setMinimum(6)
        self.bar_label_fontsize_spin.setMaximum(18)
        self.bar_label_fontsize_spin.setValue(8)
        self.bar_label_fontsize_spin.setSuffix(" pt")
        self.bar_label_fontsize_spin.setToolTip(
            "Font size of the numeric labels drawn above bars / points."
        )
        bar_layout.addWidget(self.bar_label_fontsize_spin, 0, 1)

        left_column.addWidget(bar_group)
        left_column.addStretch()

        # === RIGHT COLUMN ===
        right_column = QVBoxLayout()
        right_column.setSpacing(10)

        # Y-Axis group
        yaxis_group = QGroupBox("Y-Axis")
        yaxis_layout = QGridLayout(yaxis_group)
        yaxis_layout.setSpacing(8)

        yaxis_layout.addWidget(QLabel("Label:"), 0, 0)
        self.ylabel_edit = QLineEdit()
        self.ylabel_edit.setPlaceholderText("(auto)")
        yaxis_layout.addWidget(self.ylabel_edit, 0, 1)

        yaxis_layout.addWidget(QLabel("Label font:"), 1, 0)
        self.ylabel_fontsize_spin = QSpinBox()
        self.ylabel_fontsize_spin.setMinimum(6)
        self.ylabel_fontsize_spin.setMaximum(18)
        self.ylabel_fontsize_spin.setValue(11)
        self.ylabel_fontsize_spin.setSuffix(" pt")
        yaxis_layout.addWidget(self.ylabel_fontsize_spin, 1, 1)

        yaxis_layout.addWidget(QLabel("Tick font:"), 2, 0)
        self.ytick_fontsize_spin = QSpinBox()
        self.ytick_fontsize_spin.setMinimum(6)
        self.ytick_fontsize_spin.setMaximum(14)
        self.ytick_fontsize_spin.setValue(8)
        self.ytick_fontsize_spin.setSuffix(" pt")
        yaxis_layout.addWidget(self.ytick_fontsize_spin, 2, 1)

        yaxis_layout.addWidget(QLabel("Label pad:"), 3, 0)
        self.ylabel_pad_spin = QSpinBox()
        self.ylabel_pad_spin.setMinimum(0)
        self.ylabel_pad_spin.setMaximum(60)
        self.ylabel_pad_spin.setValue(4)
        self.ylabel_pad_spin.setSuffix(" pt")
        self.ylabel_pad_spin.setToolTip(
            "Distance between the Y-axis title and the axis."
        )
        yaxis_layout.addWidget(self.ylabel_pad_spin, 3, 1)

        right_column.addWidget(yaxis_group)

        # Scale group
        scale_group = QGroupBox("Y-Axis Scale")
        scale_layout = QGridLayout(scale_group)
        scale_layout.setSpacing(8)

        self.auto_scale_check = QCheckBox("Auto scale")
        self.auto_scale_check.setChecked(True)
        self.auto_scale_check.stateChanged.connect(self._on_auto_scale_changed)
        scale_layout.addWidget(self.auto_scale_check, 0, 0, 1, 2)

        scale_layout.addWidget(QLabel("Min:"), 1, 0)
        self.ymin_spin = QDoubleSpinBox()
        self.ymin_spin.setMinimum(-1000)
        self.ymin_spin.setMaximum(1000)
        self.ymin_spin.setValue(0)
        self.ymin_spin.setDecimals(2)
        self.ymin_spin.setEnabled(False)
        scale_layout.addWidget(self.ymin_spin, 1, 1)

        scale_layout.addWidget(QLabel("Max:"), 2, 0)
        self.ymax_spin = QDoubleSpinBox()
        self.ymax_spin.setMinimum(-1000)
        self.ymax_spin.setMaximum(1000)
        self.ymax_spin.setValue(1.0)
        self.ymax_spin.setDecimals(2)
        self.ymax_spin.setEnabled(False)
        scale_layout.addWidget(self.ymax_spin, 2, 1)

        right_column.addWidget(scale_group)
        right_column.addStretch()

        # Add columns to main layout
        columns_layout.addLayout(left_column)
        columns_layout.addLayout(right_column)
        main_layout.addLayout(columns_layout)

    def _on_auto_scale_changed(self, state):
        """Enable/disable manual scale inputs."""
        manual = not self.auto_scale_check.isChecked()
        self.ymin_spin.setEnabled(manual)
        self.ymax_spin.setEnabled(manual)

    def get_config(self) -> Dict[str, Any]:
        """Return the axes configuration."""
        return {
            'title': self.title_edit.text(),
            'title_fontsize': self.title_fontsize_spin.value(),
            'xlabel': self.xlabel_edit.text(),
            'xlabel_fontsize': self.xlabel_fontsize_spin.value(),
            'xlabel_pad': self.xlabel_pad_spin.value(),
            'xtick_fontsize': self.xtick_fontsize_spin.value(),
            'group_label_fontsize': self.group_label_fontsize_spin.value(),
            'ylabel': self.ylabel_edit.text(),
            'ylabel_fontsize': self.ylabel_fontsize_spin.value(),
            'ylabel_pad': self.ylabel_pad_spin.value(),
            'ytick_fontsize': self.ytick_fontsize_spin.value(),
            'bar_label_fontsize': self.bar_label_fontsize_spin.value(),
            'auto_scale': self.auto_scale_check.isChecked(),
            'ymin': self.ymin_spin.value(),
            'ymax': self.ymax_spin.value(),
        }

    def set_config(self, config: Dict[str, Any]):
        """Set the axes configuration."""
        if 'title' in config:
            self.title_edit.setText(config['title'])
        if 'title_fontsize' in config:
            self.title_fontsize_spin.setValue(config['title_fontsize'])
        if 'xlabel' in config:
            self.xlabel_edit.setText(config['xlabel'])
        if 'xlabel_fontsize' in config:
            self.xlabel_fontsize_spin.setValue(config['xlabel_fontsize'])
        if 'xlabel_pad' in config:
            self.xlabel_pad_spin.setValue(config['xlabel_pad'])
        if 'xtick_fontsize' in config:
            self.xtick_fontsize_spin.setValue(config['xtick_fontsize'])
        if 'group_label_fontsize' in config:
            self.group_label_fontsize_spin.setValue(config['group_label_fontsize'])
        if 'ylabel' in config:
            self.ylabel_edit.setText(config['ylabel'])
        if 'ylabel_fontsize' in config:
            self.ylabel_fontsize_spin.setValue(config['ylabel_fontsize'])
        if 'ylabel_pad' in config:
            self.ylabel_pad_spin.setValue(config['ylabel_pad'])
        if 'ytick_fontsize' in config:
            self.ytick_fontsize_spin.setValue(config['ytick_fontsize'])
        if 'bar_label_fontsize' in config:
            self.bar_label_fontsize_spin.setValue(config['bar_label_fontsize'])
        if 'auto_scale' in config:
            self.auto_scale_check.setChecked(config['auto_scale'])
            self._on_auto_scale_changed(None)
        if 'ymin' in config:
            self.ymin_spin.setValue(config['ymin'])
        if 'ymax' in config:
            self.ymax_spin.setValue(config['ymax'])


class LinesConfigTab(QWidget):
    """Tab for line and marker configuration."""

    # Marker options with display names and matplotlib codes
    MARKER_OPTIONS = [
        ("None", ""),
        ("Circle", "o"),
        ("Square", "s"),
        ("Triangle Up", "^"),
        ("Triangle Down", "v"),
        ("Diamond", "D"),
        ("Plus", "+"),
        ("Cross", "x"),
        ("Star", "*"),
        ("Pentagon", "p"),
        ("Hexagon", "h"),
    ]

    # Line style options with display names and matplotlib codes
    LINESTYLE_OPTIONS = [
        ("Solid", "-"),
        ("Dashed", "--"),
        ("Dotted", ":"),
        ("Dash-Dot", "-."),
        ("None", ""),
    ]

    def __init__(self, logger=None, parent=None):
        super().__init__(parent)
        self.logger = logger or logging.getLogger(__name__)
        self._setup_ui()

    def _setup_ui(self):
        """Setup the lines configuration UI with 2-column layout."""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)

        # Two-column layout
        columns_layout = QHBoxLayout()
        columns_layout.setSpacing(15)

        # === LEFT COLUMN ===
        left_column = QVBoxLayout()
        left_column.setSpacing(10)

        # Line Style group
        line_group = QGroupBox("Line Style")
        line_layout = QGridLayout(line_group)
        line_layout.setSpacing(8)

        line_layout.addWidget(QLabel("Style:"), 0, 0)
        self.linestyle_combo = QComboBox()
        for name, _ in self.LINESTYLE_OPTIONS:
            self.linestyle_combo.addItem(name)
        self.linestyle_combo.setCurrentIndex(0)  # Solid
        line_layout.addWidget(self.linestyle_combo, 0, 1)

        line_layout.addWidget(QLabel("Width:"), 1, 0)
        self.linewidth_spin = QDoubleSpinBox()
        self.linewidth_spin.setMinimum(0.5)
        self.linewidth_spin.setMaximum(5.0)
        self.linewidth_spin.setValue(1.5)
        self.linewidth_spin.setSingleStep(0.5)
        self.linewidth_spin.setSuffix(" pt")
        line_layout.addWidget(self.linewidth_spin, 1, 1)

        left_column.addWidget(line_group)
        left_column.addStretch()

        # === RIGHT COLUMN ===
        right_column = QVBoxLayout()
        right_column.setSpacing(10)

        # Marker group
        marker_group = QGroupBox("Markers")
        marker_layout = QGridLayout(marker_group)
        marker_layout.setSpacing(8)

        marker_layout.addWidget(QLabel("Type:"), 0, 0)
        self.marker_combo = QComboBox()
        for name, _ in self.MARKER_OPTIONS:
            self.marker_combo.addItem(name)
        self.marker_combo.setCurrentIndex(1)  # Circle
        marker_layout.addWidget(self.marker_combo, 0, 1)

        marker_layout.addWidget(QLabel("Size:"), 1, 0)
        self.markersize_spin = QDoubleSpinBox()
        self.markersize_spin.setMinimum(0)
        self.markersize_spin.setMaximum(15)
        self.markersize_spin.setValue(4)
        self.markersize_spin.setSingleStep(1)
        self.markersize_spin.setSuffix(" pt")
        marker_layout.addWidget(self.markersize_spin, 1, 1)

        right_column.addWidget(marker_group)
        right_column.addStretch()

        # Add columns to main layout
        columns_layout.addLayout(left_column)
        columns_layout.addLayout(right_column)
        main_layout.addLayout(columns_layout)

    def get_config(self) -> Dict[str, Any]:
        """Return the lines configuration."""
        linestyle_idx = self.linestyle_combo.currentIndex()
        marker_idx = self.marker_combo.currentIndex()

        return {
            'linestyle': self.LINESTYLE_OPTIONS[linestyle_idx][1],
            'linestyle_name': self.LINESTYLE_OPTIONS[linestyle_idx][0],
            'linewidth': self.linewidth_spin.value(),
            'marker': self.MARKER_OPTIONS[marker_idx][1],
            'marker_name': self.MARKER_OPTIONS[marker_idx][0],
            'markersize': self.markersize_spin.value(),
        }

    def set_config(self, config: Dict[str, Any]):
        """Set the lines configuration."""
        if 'linestyle' in config:
            # Find index by code
            for i, (_, code) in enumerate(self.LINESTYLE_OPTIONS):
                if code == config['linestyle']:
                    self.linestyle_combo.setCurrentIndex(i)
                    break
        if 'linewidth' in config:
            self.linewidth_spin.setValue(config['linewidth'])
        if 'marker' in config:
            # Find index by code
            for i, (_, code) in enumerate(self.MARKER_OPTIONS):
                if code == config['marker']:
                    self.marker_combo.setCurrentIndex(i)
                    break
        if 'markersize' in config:
            self.markersize_spin.setValue(config['markersize'])


class ChartConfigPopup(QDialog):
    """Configuration dialog with tabs for axes, legend, colors, and lines."""

    def __init__(self, parent=None, logger=None):
        super().__init__(parent)

        self.logger = logger or logging.getLogger(__name__)

        self.setWindowTitle("Chart Configuration")
        # Taller defaults so the new Axes-tab rows (Label pad, Group label
        # font, Data labels group) aren't cropped or forced into a scroll
        # bar on first open.
        self.setMinimumSize(580, 520)
        self.resize(620, 600)
        self.setModal(True)

        self._setup_ui()

    def _setup_ui(self):
        """Setup the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Tab widget
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #ccc;
                border-radius: 4px;
                background-color: white;
            }
            QTabBar::tab {
                padding: 8px 16px;
                margin-right: 2px;
                background-color: #f0f0f0;
                border: 1px solid #ccc;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom: 1px solid white;
            }
            QTabBar::tab:hover:!selected {
                background-color: #e0e0e0;
            }
        """)

        # Axes tab (first, as it's most commonly used)
        self.axes_tab = AxesConfigTab(logger=self.logger)
        self.tabs.addTab(self.axes_tab, "Axes")

        # Legend tab
        self.legend_tab = LegendConfigTab(logger=self.logger)
        self.tabs.addTab(self.legend_tab, "Legend")

        # Lines tab (line style and markers)
        self.lines_tab = LinesConfigTab(logger=self.logger)
        self.tabs.addTab(self.lines_tab, "Lines")

        # Colors tab
        self.colors_tab = ColorsConfigTab(logger=self.logger)
        self.tabs.addTab(self.colors_tab, "Colors")

        layout.addWidget(self.tabs)

        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        apply_btn = QPushButton("Apply")
        apply_btn.setMinimumWidth(80)
        apply_btn.setStyleSheet("""
            QPushButton {
                background-color: #0078d7;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #006cc1;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
        """)
        apply_btn.clicked.connect(self.accept)
        button_layout.addWidget(apply_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setMinimumWidth(80)
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        layout.addLayout(button_layout)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration from all tabs."""
        return {
            'axes': self.axes_tab.get_config(),
            'legend': self.legend_tab.get_config(),
            'lines': self.lines_tab.get_config(),
            'colors': self.colors_tab.get_config(),
        }

    def set_config(self, config: Dict[str, Any]):
        """Set configuration for all tabs."""
        if 'axes' in config:
            self.axes_tab.set_config(config['axes'])
        if 'legend' in config:
            self.legend_tab.set_config(config['legend'])
        if 'lines' in config:
            self.lines_tab.set_config(config['lines'])
        if 'colors' in config:
            self.colors_tab.set_config(config['colors'])


class CustomNavigationToolbar(NavigationToolbar2QT):
    """
    Custom navigation toolbar that extends the standard matplotlib toolbar
    with a chart configuration button that opens our custom config dialog.
    """

    # Remove the 'Subplots' button and keep 'Edit' button functionality
    toolitems = [t for t in NavigationToolbar2QT.toolitems if t[0] != 'Subplots']

    def __init__(self, canvas, parent, config_callback: Optional[Callable] = None):
        """
        Initialize the custom toolbar.

        Args:
            canvas: The matplotlib canvas
            parent: Parent widget
            config_callback: Callback function to open chart configuration dialog
        """
        super().__init__(canvas, parent)
        self._config_callback = config_callback

        # Replace the edit_parameters action with our custom one
        # Find and modify the existing edit action
        for action in self.actions():
            if action.text() == 'Customize':
                action.triggered.disconnect()
                action.triggered.connect(self._on_edit_parameters)
                action.setToolTip('Configure chart settings (legend, colors, axes)')
                break

    def _on_edit_parameters(self):
        """Handle the edit parameters button click."""
        if self._config_callback:
            self._config_callback()
        else:
            # Fallback to default matplotlib behavior
            super().edit_parameters()

    def set_config_callback(self, callback: Callable):
        """Set the callback function for chart configuration."""
        self._config_callback = callback
