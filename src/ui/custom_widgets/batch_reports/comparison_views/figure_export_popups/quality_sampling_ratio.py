"""Quality per Sampling Ratio popup (Fig 8 style)."""
from __future__ import annotations

import re
from typing import Any

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ui.custom_widgets.batch_reports.comparison_views.figure_export_popups._base import (
    BaseFigureExportPopup,
)


class QualityRowConfig:
    """Configuration for a row in the quality vs sampling ratio figure."""

    def __init__(self, test_idx: int = 0, label: str = ""):
        self.test_idx = test_idx  # Index in the tests list
        self.label = label  # Custom label (empty = use test name)


class QualityRowWidget(QFrame):
    """Widget for configuring a single row in the quality figure.

    Compact 2-line layout:
    - Line 1: Row N: Test: [combo box]
    - Line 2: Label: [text field] [delete button]
    """

    changed = pyqtSignal()
    remove_requested = pyqtSignal(object)

    def __init__(self, tests: list[dict[str, Any]], row_num: int, parent=None):
        super().__init__(parent)
        self._tests = tests
        self._row_num = row_num
        self._config = QualityRowConfig()
        self._setup_ui()

    def _setup_ui(self):
        self.setFrameStyle(QFrame.Box | QFrame.Raised)
        self.setStyleSheet("""
            QFrame {
                background-color: #fafafa;
                border: 1px solid #ddd;
                border-radius: 4px;
                margin: 2px;
            }
        """)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 6, 8, 6)
        main_layout.setSpacing(4)

        # Line 1: Row N: Test: [combo]
        line1 = QHBoxLayout()
        line1.setSpacing(6)

        self.row_label = QLabel(f"Row {self._row_num}:")
        self.row_label.setStyleSheet("font-weight: bold; min-width: 50px;")
        line1.addWidget(self.row_label)

        line1.addWidget(QLabel("Test:"))
        self.test_combo = QComboBox()
        self.test_combo.setMinimumWidth(200)
        for i, test in enumerate(self._tests):
            name = test.get("name", f"Test {i+1}")
            exp_name = test.get("_experiment_name", "")
            display = f"{name} ({exp_name})" if exp_name else name
            self.test_combo.addItem(display, i)
        self.test_combo.currentIndexChanged.connect(self._on_config_changed)
        line1.addWidget(self.test_combo, 1)

        main_layout.addLayout(line1)

        # Line 2: Label: [text field] [delete button]
        line2 = QHBoxLayout()
        line2.setSpacing(6)

        line2.addSpacing(56)  # Align with test combo
        line2.addWidget(QLabel("Label:"))
        self.label_edit = QLineEdit()
        self.label_edit.setPlaceholderText("Auto (from test name)")
        self.label_edit.textChanged.connect(self._on_config_changed)
        line2.addWidget(self.label_edit, 1)

        # Remove button
        remove_btn = QPushButton("✕")
        remove_btn.setFixedSize(24, 24)
        remove_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                border-radius: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
        """)
        remove_btn.clicked.connect(lambda: self.remove_requested.emit(self))
        line2.addWidget(remove_btn)

        main_layout.addLayout(line2)

    def _on_config_changed(self):
        """Emit changed signal when any config changes."""
        self._config.test_idx = self.test_combo.currentData()
        self._config.label = self.label_edit.text()
        self.changed.emit()

    def get_config(self) -> QualityRowConfig:
        """Get the current row configuration."""
        self._config.test_idx = self.test_combo.currentData()
        self._config.label = self.label_edit.text()
        return self._config

    def set_test_index(self, idx: int):
        """Set the test index."""
        if 0 <= idx < self.test_combo.count():
            self.test_combo.setCurrentIndex(idx)

    def set_row_number(self, num: int):
        """Update the row number display."""
        self._row_num = num
        self.row_label.setText(f"Row {num}:")


class QualitySamplingRatioPopup(BaseFigureExportPopup):
    """
    Popup for generating Quality per Sampling Ratio figure.
    Shows a table with rows for different sampling ratios and columns for
    SPI Reconstructed, Denoised, and Quality Metrics.

    Features:
    - Ground Truth centered at top with title
    - Column headers for the data table
    - Per-row test selection with editable labels
    - Single image index for entire table
    - Optional table lines (horizontal/vertical separators)
    - Configurable metrics to display
    - Same layout system as SamplesGrid (pixel-based, Fit Window)
    """

    def __init__(self, tests: list[dict[str, Any]], logger=None, parent=None):
        super().__init__(tests, logger, parent)
        self.setWindowTitle("Quality vs Sampling Ratio Figure")
        self.setMinimumSize(1100, 800)
        self.resize(1200, 850)

        self._figure = None
        self._canvas = None
        self._figure_dpi = 100
        self._natural_width_px = 800
        self._natural_height_px = 600
        self._row_widgets: list[QualityRowWidget] = []
        self._setup_ui()

    def _setup_ui(self):
        """Setup the popup UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)

        # Title
        title = QLabel("Quality Metrics Across Sampling Ratios")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(title)

        desc = QLabel(
            "Configure rows to compare different sampling ratios. "
            "Ground Truth is displayed centered at the top."
        )
        desc.setStyleSheet("color: #666;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Main splitter
        splitter = QSplitter(Qt.Horizontal)

        # Left panel: Configuration
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 10, 0)

        # Rows configuration (taller)
        rows_group = QGroupBox("Figure Rows")
        rows_layout = QVBoxLayout(rows_group)

        # Add row button
        add_row_layout = QHBoxLayout()
        add_row_btn = QPushButton("+ Add Row")
        add_row_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 6px 12px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        add_row_btn.clicked.connect(self._add_row)
        add_row_layout.addWidget(add_row_btn)
        add_row_layout.addStretch()
        rows_layout.addLayout(add_row_layout)

        # Scroll area for rows (taller to show more rows)
        self._rows_scroll = QScrollArea()
        self._rows_scroll.setWidgetResizable(True)
        self._rows_scroll.setMinimumHeight(350)
        self._rows_scroll.setStyleSheet("QScrollArea { border: none; }")

        self._rows_container = QWidget()
        self._rows_vlayout = QVBoxLayout(self._rows_container)
        self._rows_vlayout.setContentsMargins(0, 0, 0, 0)
        self._rows_vlayout.setSpacing(4)
        self._rows_vlayout.addStretch()

        self._rows_scroll.setWidget(self._rows_container)
        rows_layout.addWidget(self._rows_scroll, 1)

        left_layout.addWidget(rows_group, 1)

        # Display Options (compact grid like SamplesGrid)
        options_group = QGroupBox("Display Options")
        options_layout = QGridLayout(options_group)
        options_layout.setContentsMargins(8, 12, 8, 8)
        options_layout.setSpacing(6)

        # Image index (single for whole table)
        options_layout.addWidget(QLabel("Image index:"), 0, 0)
        self.image_spin = QSpinBox()
        self.image_spin.setMinimum(0)
        max_images = self._get_max_num_images()
        self.image_spin.setMaximum(max(0, max_images - 1))
        self.image_spin.setValue(0)
        self.image_spin.valueChanged.connect(self._update_preview)
        options_layout.addWidget(self.image_spin, 0, 1)

        # Image size (in pixels)
        options_layout.addWidget(QLabel("Image size (px):"), 0, 2)
        self.image_size_spin = QSpinBox()
        self.image_size_spin.setMinimum(32)
        self.image_size_spin.setMaximum(256)
        self.image_size_spin.setValue(80)
        self.image_size_spin.setSingleStep(8)
        self.image_size_spin.valueChanged.connect(self._update_preview)
        options_layout.addWidget(self.image_size_spin, 0, 3)

        # Row gap
        options_layout.addWidget(QLabel("Row gap (px):"), 1, 0)
        self.row_spacing_spin = QSpinBox()
        self.row_spacing_spin.setMinimum(0)
        self.row_spacing_spin.setMaximum(100)
        self.row_spacing_spin.setValue(5)
        self.row_spacing_spin.valueChanged.connect(self._update_preview)
        options_layout.addWidget(self.row_spacing_spin, 1, 1)

        # Column gap
        options_layout.addWidget(QLabel("Col gap (px):"), 1, 2)
        self.col_spacing_spin = QSpinBox()
        self.col_spacing_spin.setMinimum(0)
        self.col_spacing_spin.setMaximum(100)
        self.col_spacing_spin.setValue(5)
        self.col_spacing_spin.valueChanged.connect(self._update_preview)
        options_layout.addWidget(self.col_spacing_spin, 1, 3)

        # Colormap
        options_layout.addWidget(QLabel("Colormap:"), 2, 0)
        self.cmap_combo = self._create_colormap_combo()
        self.cmap_combo.currentTextChanged.connect(self._update_preview)
        options_layout.addWidget(self.cmap_combo, 2, 1)

        # Font size
        options_layout.addWidget(QLabel("Font size:"), 2, 2)
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setMinimum(6)
        self.font_size_spin.setMaximum(20)
        self.font_size_spin.setValue(10)
        self.font_size_spin.valueChanged.connect(self._update_preview)
        options_layout.addWidget(self.font_size_spin, 2, 3)

        # Table lines checkbox
        self.show_lines_cb = QCheckBox("Show table lines")
        self.show_lines_cb.setChecked(True)
        self.show_lines_cb.stateChanged.connect(self._update_preview)
        options_layout.addWidget(self.show_lines_cb, 3, 0, 1, 2)

        # Line width
        options_layout.addWidget(QLabel("Line width:"), 3, 2)
        self.line_width_spin = QSpinBox()
        self.line_width_spin.setMinimum(1)
        self.line_width_spin.setMaximum(5)
        self.line_width_spin.setValue(1)
        self.line_width_spin.valueChanged.connect(self._update_preview)
        options_layout.addWidget(self.line_width_spin, 3, 3)

        # Image padding (space between image and cell border)
        options_layout.addWidget(QLabel("Image padding:"), 4, 0)
        self.image_padding_spin = QSpinBox()
        self.image_padding_spin.setMinimum(0)
        self.image_padding_spin.setMaximum(20)
        self.image_padding_spin.setValue(4)
        self.image_padding_spin.valueChanged.connect(self._update_preview)
        options_layout.addWidget(self.image_padding_spin, 4, 1)

        left_layout.addWidget(options_group)

        # Column titles
        titles_group = QGroupBox("Column Titles")
        titles_layout = QGridLayout(titles_group)
        titles_layout.setContentsMargins(8, 12, 8, 8)
        titles_layout.setSpacing(4)

        titles_layout.addWidget(QLabel("Col 1:"), 0, 0)
        self.col1_title_edit = QLineEdit("Sampling\nRatio")
        self.col1_title_edit.textChanged.connect(self._update_preview)
        titles_layout.addWidget(self.col1_title_edit, 0, 1)

        titles_layout.addWidget(QLabel("Col 2:"), 1, 0)
        self.col2_title_edit = QLineEdit("SPI\nReconstructed\nImage")
        self.col2_title_edit.textChanged.connect(self._update_preview)
        titles_layout.addWidget(self.col2_title_edit, 1, 1)

        titles_layout.addWidget(QLabel("Col 3:"), 2, 0)
        self.col3_title_edit = QLineEdit("Denoised\nImage")
        self.col3_title_edit.textChanged.connect(self._update_preview)
        titles_layout.addWidget(self.col3_title_edit, 2, 1)

        titles_layout.addWidget(QLabel("Col 4:"), 3, 0)
        self.col4_title_edit = QLineEdit("Quality Metrics\n(Denoised vs GT)")
        self.col4_title_edit.textChanged.connect(self._update_preview)
        titles_layout.addWidget(self.col4_title_edit, 3, 1)

        left_layout.addWidget(titles_group)

        # Metrics selection
        metrics_group = QGroupBox("Quality Metrics to Show")
        metrics_layout = QHBoxLayout(metrics_group)

        self.show_psnr_cb = QCheckBox("PSNR")
        self.show_psnr_cb.setChecked(True)
        self.show_psnr_cb.stateChanged.connect(self._update_preview)
        metrics_layout.addWidget(self.show_psnr_cb)

        self.show_ssim_cb = QCheckBox("SSIM")
        self.show_ssim_cb.setChecked(True)
        self.show_ssim_cb.stateChanged.connect(self._update_preview)
        metrics_layout.addWidget(self.show_ssim_cb)

        self.show_lpips_cb = QCheckBox("LPIPS")
        self.show_lpips_cb.setChecked(True)
        self.show_lpips_cb.stateChanged.connect(self._update_preview)
        metrics_layout.addWidget(self.show_lpips_cb)

        metrics_layout.addStretch()
        left_layout.addWidget(metrics_group)

        # Buttons
        btn_layout = QHBoxLayout()
        self.save_btn = QPushButton("Save Figure...")
        self.save_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-weight: bold;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        self.save_btn.clicked.connect(self._on_save)
        btn_layout.addWidget(self.save_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)

        btn_layout.addStretch()
        left_layout.addLayout(btn_layout)

        splitter.addWidget(left_panel)

        # Right panel: Preview with scroll area (like SamplesGrid)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)

        # Preview header with Fit button
        preview_header = QHBoxLayout()
        preview_label = QLabel("Preview:")
        preview_label.setStyleSheet("font-weight: bold;")
        preview_header.addWidget(preview_label)

        self.fit_btn = QPushButton("Fit to Window")
        self.fit_btn.setToolTip("Scale preview to fit the visible area")
        self.fit_btn.clicked.connect(self._fit_to_window)
        preview_header.addWidget(self.fit_btn)

        self._scale_label = QLabel("Size: --")
        self._scale_label.setStyleSheet("color: #666; font-size: 11px;")
        preview_header.addWidget(self._scale_label)

        preview_header.addStretch()
        right_layout.addLayout(preview_header)

        # Scroll area for the figure
        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(False)
        self._scroll_area.setAlignment(Qt.AlignCenter)
        self._scroll_area.setStyleSheet("QScrollArea { background-color: #f0f0f0; }")

        # Figure with fixed DPI
        self._figure = Figure(figsize=(8, 10), dpi=self._figure_dpi, facecolor='white')
        self._canvas = FigureCanvas(self._figure)
        self._canvas.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self._canvas.setStyleSheet("background-color: white;")

        self._scroll_area.setWidget(self._canvas)
        right_layout.addWidget(self._scroll_area, 1)

        splitter.addWidget(right_panel)
        splitter.setSizes([380, 720])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        layout.addWidget(splitter, 1)

        # Add 4 initial rows by default
        for _ in range(min(4, len(self._tests))):
            self._add_row()

        # Initial preview
        self._update_preview()

    def _add_row(self):
        """Add a new row configuration widget."""
        row_num = len(self._row_widgets) + 1

        row_widget = QualityRowWidget(self._tests, row_num)
        row_widget.changed.connect(self._update_preview)
        row_widget.remove_requested.connect(self._remove_row)

        # Set default test to next available (if possible)
        default_idx = min(len(self._row_widgets), len(self._tests) - 1)
        row_widget.set_test_index(default_idx)

        self._row_widgets.append(row_widget)
        # Insert before stretch
        self._rows_vlayout.insertWidget(self._rows_vlayout.count() - 1, row_widget)

        self._update_preview()

    def _remove_row(self, row_widget: QualityRowWidget):
        """Remove a row configuration widget."""
        if row_widget in self._row_widgets:
            self._row_widgets.remove(row_widget)
            self._rows_vlayout.removeWidget(row_widget)
            row_widget.deleteLater()

            # Renumber remaining rows
            for i, widget in enumerate(self._row_widgets):
                widget.set_row_number(i + 1)

            self._update_preview()

    def _update_preview(self):
        """Update the preview figure using fixed pixel dimensions."""
        self._figure.clf()
        self._figure.set_facecolor('white')

        if not self._row_widgets:
            self._natural_width_px = 600
            self._natural_height_px = 400
            self._figure.set_size_inches(6, 4)
            self._apply_canvas_size()
            ax = self._figure.add_subplot(111)
            ax.text(0.5, 0.5, "Add rows to preview",
                   ha='center', va='center', fontsize=14, color='#999')
            ax.axis('off')
            self._canvas.draw()
            return

        # Get configuration values
        cmap = self.cmap_combo.currentText()
        font_size = self.font_size_spin.value()
        image_idx = self.image_spin.value()
        image_size_px = self.image_size_spin.value()
        row_gap_px = self.row_spacing_spin.value()
        col_gap_px = self.col_spacing_spin.value()
        show_psnr = self.show_psnr_cb.isChecked()
        show_ssim = self.show_ssim_cb.isChecked()
        show_lpips = self.show_lpips_cb.isChecked()
        show_lines = self.show_lines_cb.isChecked()
        line_width = self.line_width_spin.value()
        image_padding = self.image_padding_spin.value()

        n_data_rows = len(self._row_widgets)

        # Calculate widths for each column type (cell sizes)
        # Col 0: Sampling Ratio label - narrower
        # Col 1: SPI Reconstructed - image
        # Col 2: Denoised - image
        # Col 3: Quality Metrics - wider for text
        label_col_width = int(image_size_px * 0.8)
        metrics_col_width = int(image_size_px * 1.4)

        # Margins
        left_margin_px = 15
        right_margin_px = 15
        top_margin_px = int(font_size * 8)
        bottom_margin_px = 15

        # GT image height (1.2 times image size)
        gt_image_height = int(image_size_px * 1.2)
        gt_title_height = int(font_size * 2.5)
        header_height = int(font_size * 5.5)  # Taller for column titles

        # Calculate total figure size
        content_width_px = (label_col_width + 2 * image_size_px + metrics_col_width +
                           3 * col_gap_px)
        data_content_height_px = n_data_rows * image_size_px + (n_data_rows - 1) * row_gap_px

        # Total height: GT title + GT image + larger gap + headers + data rows
        # Use larger gap between GT and table for visual separation
        gt_table_gap = max(row_gap_px * 2, 20)  # At least 20px or 2x row_gap
        gt_section_height = gt_title_height + gt_image_height + gt_table_gap

        fig_width_px = left_margin_px + content_width_px + right_margin_px
        fig_height_px = (top_margin_px + gt_section_height + header_height +
                        data_content_height_px + bottom_margin_px)

        # Convert to inches
        fig_width_in = fig_width_px / self._figure_dpi
        fig_height_in = fig_height_px / self._figure_dpi

        self._natural_width_px = fig_width_px
        self._natural_height_px = fig_height_px

        self._figure.set_size_inches(fig_width_in, fig_height_in)
        self._apply_canvas_size()

        # Column x positions (left edge of each cell)
        col_x = [
            left_margin_px,  # Label
            left_margin_px + label_col_width + col_gap_px,  # Reconstructed
            left_margin_px + label_col_width + col_gap_px + image_size_px + col_gap_px,  # Denoised
            left_margin_px + label_col_width + col_gap_px + 2 * image_size_px + 2 * col_gap_px,  # Metrics
        ]
        col_widths = [label_col_width, image_size_px, image_size_px, metrics_col_width]

        # Y positions (from top, converted to bottom-up for matplotlib)
        # GT title
        gt_title_y_top = top_margin_px
        gt_title_y_bottom = fig_height_px - gt_title_y_top - gt_title_height

        # GT image (centered horizontally)
        gt_img_y_top = gt_title_y_top + gt_title_height
        gt_img_y_bottom = fig_height_px - gt_img_y_top - gt_image_height
        gt_img_x = left_margin_px + (content_width_px - gt_image_height) / 2  # Centered

        # Headers y position (after GT section with larger gap)
        headers_y_top = gt_img_y_top + gt_image_height + gt_table_gap
        headers_y_bottom = fig_height_px - headers_y_top - header_height

        # Data rows start immediately after headers (no gap)
        data_start_y_top = headers_y_top + header_height

        # Draw GT title
        ax_gt_title = self._figure.add_axes([
            left_margin_px / fig_width_px,
            gt_title_y_bottom / fig_height_px,
            content_width_px / fig_width_px,
            gt_title_height / fig_height_px
        ])
        ax_gt_title.axis('off')
        ax_gt_title.text(0.5, 0.5, "Ground Truth", ha='center', va='center',
                        fontsize=font_size + 2, fontweight='bold')

        # Draw GT image
        first_config = self._row_widgets[0].get_config()
        first_test = self._tests[first_config.test_idx]
        originals, _, _ = self._load_test_images(first_test)
        gt_img = None
        if originals is not None and image_idx < len(originals):
            gt_img = originals[image_idx]

        ax_gt_img = self._figure.add_axes([
            gt_img_x / fig_width_px,
            gt_img_y_bottom / fig_height_px,
            gt_image_height / fig_width_px,
            gt_image_height / fig_height_px
        ])
        if gt_img is not None:
            gt_img = np.array(gt_img)
            if gt_img.ndim == 3 and gt_img.shape[-1] == 1:
                gt_img = gt_img.squeeze(-1)
            ax_gt_img.imshow(gt_img, cmap=cmap, vmin=0, vmax=1)
        ax_gt_img.axis('off')

        # Draw column headers (use editable titles, replace \n for newlines)
        col_headers = [
            self.col1_title_edit.text().replace("\\n", "\n"),
            self.col2_title_edit.text().replace("\\n", "\n"),
            self.col3_title_edit.text().replace("\\n", "\n"),
            self.col4_title_edit.text().replace("\\n", "\n"),
        ]
        for col_idx, header in enumerate(col_headers):
            ax_header = self._figure.add_axes([
                col_x[col_idx] / fig_width_px,
                headers_y_bottom / fig_height_px,
                col_widths[col_idx] / fig_width_px,
                header_height / fig_height_px
            ])
            ax_header.axis('off')
            ax_header.text(0.5, 0.5, header, ha='center', va='center',
                          fontsize=font_size, fontweight='bold')

        # Draw data rows
        for row_idx, row_widget in enumerate(self._row_widgets):
            config = row_widget.get_config()
            test = self._tests[config.test_idx]

            # Calculate y position for this row
            row_y_top = data_start_y_top + row_idx * (image_size_px + row_gap_px)
            row_y_bottom = fig_height_px - row_y_top - image_size_px

            # Determine label
            if config.label.strip():
                label_text = config.label
            else:
                test_name = test.get("name", f"Test {row_idx+1}")
                match = re.search(r'(\d+)%', test_name)
                if match:
                    label_text = f"{match.group(1)}%"
                else:
                    density = test.get("scatter_point_density")
                    if density:
                        label_text = f"{density:.0f}%"
                    else:
                        label_text = test_name[:12]

            # Load images
            originals, reconstructions, denoised_arr = self._load_test_images(test)

            recon_img = None
            denoised_img = None
            if reconstructions is not None and image_idx < len(reconstructions):
                recon_img = reconstructions[image_idx]
            if denoised_arr is not None and image_idx < len(denoised_arr):
                denoised_img = denoised_arr[image_idx]

            # Get metrics
            quality_per_image = test.get("quality_per_image", {})
            psnr_list = quality_per_image.get("psnr_denoised", [])
            ssim_list = quality_per_image.get("ssim_denoised", [])
            lpips_list = quality_per_image.get("lpips_denoised", [])

            psnr = psnr_list[image_idx] if image_idx < len(psnr_list) else None
            ssim = ssim_list[image_idx] if image_idx < len(ssim_list) else None
            lpips = lpips_list[image_idx] if image_idx < len(lpips_list) else None

            # Column 0: Label (no padding needed for text)
            ax_label = self._figure.add_axes([
                col_x[0] / fig_width_px,
                row_y_bottom / fig_height_px,
                col_widths[0] / fig_width_px,
                image_size_px / fig_height_px
            ])
            ax_label.axis('off')
            ax_label.text(0.5, 0.5, label_text, ha='center', va='center',
                         fontsize=font_size + 2, fontweight='bold')

            # Column 1: Reconstructed (with padding)
            padded_size = image_size_px - 2 * image_padding
            if padded_size > 0:
                ax_recon = self._figure.add_axes([
                    (col_x[1] + image_padding) / fig_width_px,
                    (row_y_bottom + image_padding) / fig_height_px,
                    padded_size / fig_width_px,
                    padded_size / fig_height_px
                ])
            else:
                ax_recon = self._figure.add_axes([
                    col_x[1] / fig_width_px,
                    row_y_bottom / fig_height_px,
                    col_widths[1] / fig_width_px,
                    image_size_px / fig_height_px
                ])
            if recon_img is not None:
                recon_img = np.array(recon_img)
                if recon_img.ndim == 3 and recon_img.shape[-1] == 1:
                    recon_img = recon_img.squeeze(-1)
                ax_recon.imshow(recon_img, cmap=cmap, vmin=0, vmax=1)
            ax_recon.axis('off')

            # Column 2: Denoised (with padding)
            if padded_size > 0:
                ax_denoised = self._figure.add_axes([
                    (col_x[2] + image_padding) / fig_width_px,
                    (row_y_bottom + image_padding) / fig_height_px,
                    padded_size / fig_width_px,
                    padded_size / fig_height_px
                ])
            else:
                ax_denoised = self._figure.add_axes([
                    col_x[2] / fig_width_px,
                    row_y_bottom / fig_height_px,
                    col_widths[2] / fig_width_px,
                    image_size_px / fig_height_px
                ])
            if denoised_img is not None:
                denoised_img = np.array(denoised_img)
                if denoised_img.ndim == 3 and denoised_img.shape[-1] == 1:
                    denoised_img = denoised_img.squeeze(-1)
                ax_denoised.imshow(denoised_img, cmap=cmap, vmin=0, vmax=1)
            ax_denoised.axis('off')

            # Column 3: Metrics (no padding needed for text)
            ax_metrics = self._figure.add_axes([
                col_x[3] / fig_width_px,
                row_y_bottom / fig_height_px,
                col_widths[3] / fig_width_px,
                image_size_px / fig_height_px
            ])
            ax_metrics.axis('off')
            metrics_lines = []
            if show_psnr and psnr is not None:
                metrics_lines.append(f"PSNR = {psnr:.2f} dB")
            if show_ssim and ssim is not None:
                metrics_lines.append(f"SSIM = {ssim:.3f}")
            if show_lpips and lpips is not None:
                metrics_lines.append(f"LPIPS = {lpips:.3f}")
            metrics_text = "\n".join(metrics_lines)
            ax_metrics.text(0.1, 0.5, metrics_text, ha='left', va='center',
                           fontsize=font_size, family='monospace')

        # Draw table lines if enabled (no outer border, no top line)
        if show_lines and n_data_rows > 0:
            # Calculate y positions for lines
            # Top of headers (for vertical lines extent)
            header_top_y = (headers_y_bottom + header_height) / fig_height_px
            # Bottom of headers (= top of first data row)
            header_bottom_y = (fig_height_px - data_start_y_top) / fig_height_px

            # Horizontal line below headers (separates headers from data)
            self._figure.add_artist(Line2D(
                [col_x[0] / fig_width_px, (col_x[3] + col_widths[3]) / fig_width_px],
                [header_bottom_y, header_bottom_y],
                transform=self._figure.transFigure,
                color='black', linewidth=line_width, clip_on=False
            ))

            # Lines between data rows (not below last row)
            for row_idx in range(n_data_rows - 1):
                row_y_top = data_start_y_top + row_idx * (image_size_px + row_gap_px)
                row_y_bottom = fig_height_px - row_y_top - image_size_px
                line_y = row_y_bottom / fig_height_px
                self._figure.add_artist(Line2D(
                    [col_x[0] / fig_width_px, (col_x[3] + col_widths[3]) / fig_width_px],
                    [line_y, line_y],
                    transform=self._figure.transFigure,
                    color='black', linewidth=line_width, clip_on=False
                ))

            # Vertical lines: between columns, extending through headers (not at outer edges)
            last_row_y_top = data_start_y_top + (n_data_rows - 1) * (image_size_px + row_gap_px)
            last_row_y_bottom = fig_height_px - last_row_y_top - image_size_px
            v_line_bottom = last_row_y_bottom / fig_height_px
            v_line_top = header_top_y  # Extend to top of headers

            # Vertical lines between columns (not at left/right edges)
            for col_idx in range(1, 4):
                x = col_x[col_idx] / fig_width_px
                self._figure.add_artist(Line2D(
                    [x, x], [v_line_bottom, v_line_top],
                    transform=self._figure.transFigure,
                    color='black', linewidth=line_width, clip_on=False
                ))

        self._canvas.draw()

    def _apply_canvas_size(self):
        """Apply the canvas size based on natural figure dimensions."""
        if not hasattr(self, '_natural_width_px'):
            return

        width = int(self._natural_width_px)
        height = int(self._natural_height_px)

        self._canvas.setFixedSize(width, height)
        self._canvas.updateGeometry()
        self._scroll_area.viewport().update()
        self._scale_label.setText(f"Size: {width}×{height}px")

    def _fit_to_window(self):
        """Adjust image size to fit the figure in the visible scroll area."""
        if not self._row_widgets:
            return

        n_rows = len(self._row_widgets)

        # Get available space
        viewport = self._scroll_area.viewport()
        available_width = viewport.width() - 40
        available_height = viewport.height() - 40

        # Current gap values
        row_gap = self.row_spacing_spin.value()
        col_gap = self.col_spacing_spin.value()
        font_size = self.font_size_spin.value()

        # Estimate margins and fixed sections
        left_margin = 15
        right_margin = 15
        top_margin = int(font_size * 8)
        bottom_margin = 15

        # GT section and header heights scale with image size, estimate
        # We need to solve for image_size such that total fits

        # Approximate: total_height = top + 1.2*img + gap + 4*font + gap + n_rows*img + (n-1)*row_gap + bottom
        # Simplify: total_height ≈ fixed_overhead + (1.2 + n_rows) * img_size + (n_rows-1)*row_gap

        fixed_height_overhead = top_margin + bottom_margin + row_gap + int(font_size * 4) + row_gap
        gt_factor = 1.2  # GT image is 1.2 times image size

        # total_height = fixed_height_overhead + gt_factor * img + n_rows * img + (n_rows-1)*row_gap
        # available_height = fixed_height_overhead + (gt_factor + n_rows) * img + (n_rows-1)*row_gap
        # img = (available_height - fixed_height_overhead - (n_rows-1)*row_gap) / (gt_factor + n_rows)

        content_height_for_images = available_height - fixed_height_overhead - (n_rows - 1) * row_gap
        max_img_h = content_height_for_images / (gt_factor + n_rows)

        # For width: label_col = 0.8*img, metrics_col = 1.4*img, 2 image cols
        # total_width = left + 0.8*img + 3*col_gap + 2*img + 1.4*img + right
        # total_width = left + right + 3*col_gap + (0.8 + 2 + 1.4)*img = margins + 3*col_gap + 4.2*img
        fixed_width_overhead = left_margin + right_margin + 3 * col_gap
        width_factor = 4.2
        max_img_w = (available_width - fixed_width_overhead) / width_factor

        new_img_size = int(min(max_img_h, max_img_w))
        new_img_size = max(32, min(256, new_img_size))

        self.image_size_spin.setValue(new_img_size)

    def _on_save(self):
        """Save the current figure."""
        self._save_figure(self._figure, "quality_sampling_ratio.png")
