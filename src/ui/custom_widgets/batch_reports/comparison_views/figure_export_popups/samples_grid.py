"""Samples Grid popup (Fig 2 style)."""
from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QCheckBox,
    QColorDialog,
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
from ui.custom_widgets.batch_reports.comparison_views.figure_export_popups._grid_column_config import (
    GridColumnConfig,
    GridColumnListWidget,
)


class SamplesGridPopup(BaseFigureExportPopup):
    """
    Popup for generating Samples Grid figure.
    Shows multiple sample images (rows) at different sampling ratios (columns).
    """

    def __init__(self, tests: list[dict[str, Any]], logger=None, parent=None):
        super().__init__(tests, logger, parent)
        self.setWindowTitle("Samples Grid Figure")
        self.setMinimumSize(1100, 750)
        self.resize(1300, 850)

        self._figure = None
        self._canvas = None
        self._images_cache = {}
        self._row_spinboxes = []
        self._row_label_edits = []
        self._figure_dpi = 100
        self._natural_width_px = 600
        self._natural_height_px = 400
        self._setup_ui()

    def _setup_ui(self):
        """Setup the popup UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Title
        title = QLabel("Samples Grid (Multiple Images × Tests)")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(title)

        # Main splitter: controls on left, preview on right
        main_splitter = QSplitter(Qt.Horizontal)

        # Left panel: all controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 8, 0)
        left_layout.setSpacing(8)

        # Column configuration (compact)
        col_group = QGroupBox("Columns (drag to reorder, double-click to edit)")
        col_layout = QVBoxLayout(col_group)
        col_layout.setContentsMargins(8, 12, 8, 8)

        self.column_list = GridColumnListWidget(self._tests)
        self.column_list.columns_changed.connect(self._update_preview)
        col_layout.addWidget(self.column_list)

        left_layout.addWidget(col_group)

        # Row configuration
        rows_group = QGroupBox("Rows")
        rows_layout = QVBoxLayout(rows_group)
        rows_layout.setContentsMargins(8, 12, 8, 8)
        rows_layout.setSpacing(4)

        # Number of rows
        n_rows_layout = QHBoxLayout()
        n_rows_layout.addWidget(QLabel("Number of rows:"))
        self.n_rows_spin = QSpinBox()
        self.n_rows_spin.setMinimum(1)
        self.n_rows_spin.setMaximum(10)
        self.n_rows_spin.setValue(4)
        self.n_rows_spin.setFixedWidth(60)
        self.n_rows_spin.valueChanged.connect(self._update_row_spinboxes)
        n_rows_layout.addWidget(self.n_rows_spin)
        n_rows_layout.addStretch()
        rows_layout.addLayout(n_rows_layout)

        # Scroll area for row configuration
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(150)
        scroll.setFrameShape(QFrame.NoFrame)

        self.rows_container = QWidget()
        self.rows_container_layout = QGridLayout(self.rows_container)
        self.rows_container_layout.setContentsMargins(0, 4, 0, 0)
        self.rows_container_layout.setSpacing(4)
        self.rows_container_layout.setColumnStretch(1, 1)
        self.rows_container_layout.setColumnStretch(3, 1)
        scroll.setWidget(self.rows_container)

        rows_layout.addWidget(scroll)
        left_layout.addWidget(rows_group)

        # Display options (compact)
        options_group = QGroupBox("Display Options")
        options_layout = QGridLayout(options_group)
        options_layout.setContentsMargins(8, 12, 8, 8)
        options_layout.setSpacing(6)

        # Image size (in pixels)
        options_layout.addWidget(QLabel("Image size (px):"), 0, 0)
        self.image_size_spin = QSpinBox()
        self.image_size_spin.setMinimum(32)
        self.image_size_spin.setMaximum(256)
        self.image_size_spin.setValue(80)
        self.image_size_spin.setSingleStep(8)
        self.image_size_spin.valueChanged.connect(self._update_preview)
        options_layout.addWidget(self.image_size_spin, 0, 1)

        # Font size
        options_layout.addWidget(QLabel("Font size:"), 0, 2)
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setMinimum(6)
        self.font_size_spin.setMaximum(20)
        self.font_size_spin.setValue(11)
        self.font_size_spin.valueChanged.connect(self._update_preview)
        options_layout.addWidget(self.font_size_spin, 0, 3)

        # Row gap (in pixels)
        options_layout.addWidget(QLabel("Row gap (px):"), 1, 0)
        self.row_spacing_spin = QSpinBox()
        self.row_spacing_spin.setMinimum(0)
        self.row_spacing_spin.setMaximum(100)
        self.row_spacing_spin.setValue(10)
        self.row_spacing_spin.valueChanged.connect(self._update_preview)
        options_layout.addWidget(self.row_spacing_spin, 1, 1)

        # Column gap (in pixels)
        options_layout.addWidget(QLabel("Col gap (px):"), 1, 2)
        self.col_spacing_spin = QSpinBox()
        self.col_spacing_spin.setMinimum(0)
        self.col_spacing_spin.setMaximum(100)
        self.col_spacing_spin.setValue(10)
        self.col_spacing_spin.valueChanged.connect(self._update_preview)
        options_layout.addWidget(self.col_spacing_spin, 1, 3)

        # Colormap
        options_layout.addWidget(QLabel("Colormap:"), 2, 0)
        self.cmap_combo = self._create_colormap_combo()
        self.cmap_combo.currentTextChanged.connect(self._update_preview)
        options_layout.addWidget(self.cmap_combo, 2, 1)

        # Checkboxes row
        self.show_labels_cb = QCheckBox("Column titles")
        self.show_labels_cb.setChecked(True)
        self.show_labels_cb.stateChanged.connect(self._update_preview)
        options_layout.addWidget(self.show_labels_cb, 3, 0, 1, 2)

        self.show_row_labels_cb = QCheckBox("Row labels")
        self.show_row_labels_cb.setChecked(True)
        self.show_row_labels_cb.stateChanged.connect(self._update_preview)
        options_layout.addWidget(self.show_row_labels_cb, 3, 2, 1, 2)

        # Colored borders checkbox
        self.show_grid_cb = QCheckBox("Colored borders around groups")
        self.show_grid_cb.setChecked(False)
        self.show_grid_cb.stateChanged.connect(self._update_preview)
        options_layout.addWidget(self.show_grid_cb, 4, 0, 1, 4)

        # Border colors and width (in a sub-layout)
        border_layout = QHBoxLayout()
        border_layout.setContentsMargins(15, 0, 0, 0)
        border_layout.setSpacing(6)

        border_layout.addWidget(QLabel("GT:"))
        self._gt_color = QColor("#4CAF50")
        self.gt_color_btn = QPushButton()
        self.gt_color_btn.setFixedSize(28, 22)
        self.gt_color_btn.setStyleSheet(f"background-color: {self._gt_color.name()}; border: 1px solid #666;")
        self.gt_color_btn.clicked.connect(self._pick_gt_color)
        border_layout.addWidget(self.gt_color_btn)

        border_layout.addWidget(QLabel("Tests:"))
        self._test_color = QColor("#FF9800")
        self.test_color_btn = QPushButton()
        self.test_color_btn.setFixedSize(28, 22)
        self.test_color_btn.setStyleSheet(f"background-color: {self._test_color.name()}; border: 1px solid #666;")
        self.test_color_btn.clicked.connect(self._pick_test_color)
        border_layout.addWidget(self.test_color_btn)

        border_layout.addWidget(QLabel("Width:"))
        self.grid_width_spin = QSpinBox()
        self.grid_width_spin.setMinimum(1)
        self.grid_width_spin.setMaximum(8)
        self.grid_width_spin.setValue(3)
        self.grid_width_spin.setFixedWidth(45)
        self.grid_width_spin.valueChanged.connect(self._update_preview)
        border_layout.addWidget(self.grid_width_spin)

        border_layout.addWidget(QLabel("Padding:"))
        self.border_padding_spin = QSpinBox()
        self.border_padding_spin.setMinimum(0)
        self.border_padding_spin.setMaximum(20)
        self.border_padding_spin.setValue(3)
        self.border_padding_spin.setFixedWidth(45)
        self.border_padding_spin.valueChanged.connect(self._update_preview)
        border_layout.addWidget(self.border_padding_spin)

        border_layout.addStretch()
        options_layout.addLayout(border_layout, 5, 0, 1, 4)

        # Sampling ratio header
        self.show_ratio_header_cb = QCheckBox("Show 'Sampling ratio' header")
        self.show_ratio_header_cb.setChecked(True)
        self.show_ratio_header_cb.stateChanged.connect(self._update_preview)
        options_layout.addWidget(self.show_ratio_header_cb, 6, 0, 1, 4)

        # Per-column β value in the title
        self.show_beta_cb = QCheckBox("Show β (sampling ratio per column)")
        self.show_beta_cb.setToolTip(
            "Append β = n_patterns / n_pixels to each test column title "
            "(e.g. β=8.3%). Not shown for Ground Truth columns."
        )
        self.show_beta_cb.setChecked(False)
        self.show_beta_cb.stateChanged.connect(self._update_preview)
        options_layout.addWidget(self.show_beta_cb, 7, 0, 1, 4)

        left_layout.addWidget(options_group)

        # Save button
        btn_layout = QHBoxLayout()
        self.save_btn = QPushButton("Save Figure...")
        self.save_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                font-weight: bold;
                padding: 8px 20px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
        """)
        self.save_btn.clicked.connect(self._on_save)
        btn_layout.addWidget(self.save_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)

        btn_layout.addStretch()
        left_layout.addLayout(btn_layout)

        left_layout.addStretch()
        main_splitter.addWidget(left_panel)

        # Right panel: Preview (larger)
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

        self._scale_label = QLabel("Scale: 100%")
        self._scale_label.setStyleSheet("color: #666; font-size: 11px;")
        preview_header.addWidget(self._scale_label)

        preview_header.addStretch()
        right_layout.addLayout(preview_header)

        # Scroll area for the figure
        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(False)  # Fixed size content
        self._scroll_area.setAlignment(Qt.AlignCenter)
        self._scroll_area.setStyleSheet("QScrollArea { background-color: #f0f0f0; }")

        # Figure with fixed DPI (will be resized dynamically)
        self._figure_dpi = 100
        self._figure = Figure(figsize=(10, 8), dpi=self._figure_dpi, facecolor='white')
        self._canvas = FigureCanvas(self._figure)
        self._canvas.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self._canvas.setStyleSheet("background-color: white;")

        # Set canvas directly as scroll area widget (no container needed)
        self._scroll_area.setWidget(self._canvas)
        right_layout.addWidget(self._scroll_area, 1)

        main_splitter.addWidget(right_panel)

        # Set splitter sizes (controls: 320px, preview: rest)
        main_splitter.setSizes([320, 900])
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)

        layout.addWidget(main_splitter, 1)

        # Initialize row spinboxes
        self._update_row_spinboxes()

    def _update_row_spinboxes(self):
        """Update the row sample index spinboxes and label editors."""
        n_rows = self.n_rows_spin.value()

        # Clear existing widgets
        for spin in self._row_spinboxes:
            spin.deleteLater()
        for edit in self._row_label_edits:
            edit.deleteLater()
        self._row_spinboxes.clear()
        self._row_label_edits.clear()

        # Clear layout
        while self.rows_container_layout.count():
            item = self.rows_container_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Header row
        idx_header = QLabel("Sample Index")
        idx_header.setStyleSheet("font-weight: bold; font-size: 10px;")
        self.rows_container_layout.addWidget(idx_header, 0, 1)

        label_header = QLabel("Row Label")
        label_header.setStyleSheet("font-weight: bold; font-size: 10px;")
        self.rows_container_layout.addWidget(label_header, 0, 2)

        # Get max number of images for spinbox limit
        max_images = self._get_max_num_images()
        max_idx = max(0, max_images - 1)

        # Create row widgets
        for i in range(n_rows):
            row_label = QLabel(f"Row {i+1}:")
            row_label.setStyleSheet("font-size: 11px;")
            self.rows_container_layout.addWidget(row_label, i + 1, 0)

            spin = QSpinBox()
            spin.setMinimum(0)
            spin.setMaximum(max_idx)
            spin.setValue(min(i, max_idx))  # Default to sequential, but respect max
            spin.setFixedWidth(70)
            spin.valueChanged.connect(self._update_preview)
            self._row_spinboxes.append(spin)
            self.rows_container_layout.addWidget(spin, i + 1, 1)

            label_edit = QLineEdit(f"Sample #{i+1}")
            label_edit.setFixedWidth(100)
            label_edit.textChanged.connect(self._update_preview)
            self._row_label_edits.append(label_edit)
            self.rows_container_layout.addWidget(label_edit, i + 1, 2)

        self._update_preview()

    def _get_row_indices(self) -> list[int]:
        """Get the sample indices for each row."""
        return [spin.value() for spin in self._row_spinboxes]

    def _get_row_labels(self) -> list[str]:
        """Get the labels for each row."""
        return [edit.text() for edit in self._row_label_edits]

    def _pick_gt_color(self):
        """Open color picker for GT border."""
        color = QColorDialog.getColor(self._gt_color, self, "Select Ground Truth Border Color")
        if color.isValid():
            self._gt_color = color
            self.gt_color_btn.setStyleSheet(f"background-color: {color.name()}; border: 1px solid #666;")
            self._update_preview()

    def _pick_test_color(self):
        """Open color picker for test border."""
        color = QColorDialog.getColor(self._test_color, self, "Select Tests Border Color")
        if color.isValid():
            self._test_color = color
            self.test_color_btn.setStyleSheet(f"background-color: {color.name()}; border: 1px solid #666;")
            self._update_preview()

    def _compute_beta_percent(self, col_config: GridColumnConfig) -> float | None:
        """
        Compute the sampling ratio β = n_patterns / n_pixels for a test column,
        expressed as a percentage. Returns None when the information is missing.
        """
        if col_config.col_type != GridColumnConfig.TYPE_TEST:
            return None
        if not (0 <= col_config.test_idx < len(self._tests)):
            return None

        test = self._tests[col_config.test_idx]

        # Pixel count: prefer stored image shape over guessing from config.
        n_pixels = None
        originals, _, _ = self._load_test_images(test)
        if originals is not None and originals.ndim >= 3:
            n_pixels = int(originals.shape[-1]) * int(originals.shape[-2])
        if n_pixels is None:
            # Fall back to the experiment-level dataset_info when available.
            batch_meta = test.get("_experiment_metadata") or {}
            ds = batch_meta.get("dataset_info", {}) if isinstance(batch_meta, dict) else {}
            if "img_size" in ds:
                n_pixels = int(ds["img_size"]) ** 2

        # Pattern count: prefer the timing-phase count; otherwise derive it.
        n_patterns = test.get("timing_num_patterns")
        if n_patterns is None:
            mask_type = (test.get("mask_type") or "").lower()
            if mask_type == "scatter":
                n_patterns = test.get("scatter_num_patterns")
            elif mask_type.startswith("hadamard") or mask_type == "cal_sal":
                lo = test.get("hadamard_min_idx")
                hi = test.get("hadamard_max_idx")
                if lo is not None and hi is not None:
                    n_patterns = max(0, int(hi) - int(lo))

        if not n_pixels or not n_patterns:
            return None
        return 100.0 * float(n_patterns) / float(n_pixels)

    def _load_column_images(self, col_config: GridColumnConfig) -> tuple:
        """Load images for a column configuration."""
        if col_config.col_type == GridColumnConfig.TYPE_GROUND_TRUTH:
            # For ground truth, use first available test's originals
            for test in self._tests:
                originals, _, _ = self._load_test_images(test)
                if originals is not None:
                    return originals, None, None
            return None, None, None
        else:
            if 0 <= col_config.test_idx < len(self._tests):
                test = self._tests[col_config.test_idx]
                return self._load_test_images(test)
            return None, None, None

    def _update_preview(self):
        """Update the preview figure using fixed pixel dimensions."""
        # Clear figure completely and set white background to cover old content
        self._figure.clf()
        self._figure.set_facecolor('white')

        columns = self.column_list.get_columns()
        row_indices = self._get_row_indices()
        row_labels = self._get_row_labels()

        if not columns:
            self._natural_width_px = 600
            self._natural_height_px = 400
            self._figure.set_size_inches(6, 4)
            self._apply_canvas_size()
            ax = self._figure.add_subplot(111)
            ax.text(0.5, 0.5, "Add columns to preview",
                   ha='center', va='center', fontsize=14, color='#999')
            ax.axis('off')
            self._canvas.draw()
            return

        n_rows = len(row_indices)
        n_cols = len(columns)
        cmap = self.cmap_combo.currentText()
        show_labels = self.show_labels_cb.isChecked()
        show_row_labels = self.show_row_labels_cb.isChecked()
        show_borders = self.show_grid_cb.isChecked()
        border_width = self.grid_width_spin.value()
        border_padding_px = self.border_padding_spin.value()
        row_gap_px = self.row_spacing_spin.value()
        col_gap_px = self.col_spacing_spin.value()
        show_ratio_header = self.show_ratio_header_cb.isChecked()
        font_size = self.font_size_spin.value()
        image_size_px = self.image_size_spin.value()
        gt_color = self._gt_color.name()
        test_color = self._test_color.name()

        # Calculate margins in pixels (scale with font size for row labels)
        # "Sample #1" is about 10 characters, each char ~0.7 * font_size wide
        left_margin_px = int(font_size * 12) if show_row_labels else 10
        right_margin_px = 10
        top_margin_px = int(font_size * 5) if show_labels else 20
        if show_ratio_header and show_labels:
            top_margin_px += int(font_size * 2.5)  # Extra space for "Sampling ratio" header
        if self.show_beta_cb.isChecked() and show_labels:
            top_margin_px += int(font_size * 2)  # Extra line for β value
        bottom_margin_px = 10

        # Calculate total figure size in pixels
        content_width_px = n_cols * image_size_px + (n_cols - 1) * col_gap_px
        content_height_px = n_rows * image_size_px + (n_rows - 1) * row_gap_px
        fig_width_px = left_margin_px + content_width_px + right_margin_px
        fig_height_px = top_margin_px + content_height_px + bottom_margin_px

        # Convert to inches at our fixed DPI
        fig_width_in = fig_width_px / self._figure_dpi
        fig_height_in = fig_height_px / self._figure_dpi

        # Store natural size
        self._natural_width_px = fig_width_px
        self._natural_height_px = fig_height_px

        # Set figure size and apply canvas size
        self._figure.set_size_inches(fig_width_in, fig_height_in)
        self._apply_canvas_size()

        # Preload images for all columns
        column_images = {}
        for col_idx, col_config in enumerate(columns):
            originals, reconstructions, denoised = self._load_column_images(col_config)
            column_images[col_idx] = (originals, reconstructions, denoised)

        # Create axes manually with fixed pixel positions
        axes = []
        for row_idx in range(n_rows):
            row_axes = []
            for col_idx in range(n_cols):
                # Calculate position in pixels
                x_px = left_margin_px + col_idx * (image_size_px + col_gap_px)
                y_px = bottom_margin_px + (n_rows - 1 - row_idx) * (image_size_px + row_gap_px)

                # Convert to figure fractions
                x_frac = x_px / fig_width_px
                y_frac = y_px / fig_height_px
                w_frac = image_size_px / fig_width_px
                h_frac = image_size_px / fig_height_px

                ax = self._figure.add_axes([x_frac, y_frac, w_frac, h_frac])
                row_axes.append(ax)
            axes.append(row_axes)

        # Count ground truth columns and test columns
        gt_cols = [i for i, c in enumerate(columns) if c.col_type == GridColumnConfig.TYPE_GROUND_TRUTH]
        test_cols = [i for i, c in enumerate(columns) if c.col_type == GridColumnConfig.TYPE_TEST]

        # Fill the grid
        for row_idx, sample_idx in enumerate(row_indices):
            for col_idx, col_config in enumerate(columns):
                ax = axes[row_idx][col_idx]

                originals, reconstructions, denoised = column_images[col_idx]

                # Get the appropriate image
                img = None
                if col_config.col_type == GridColumnConfig.TYPE_GROUND_TRUTH:
                    if originals is not None and sample_idx < len(originals):
                        img = originals[sample_idx]
                else:
                    # Show reconstructed images for tests
                    if reconstructions is not None and sample_idx < len(reconstructions):
                        img = reconstructions[sample_idx]

                if img is not None:
                    img = np.array(img)
                    if img.ndim == 3 and img.shape[-1] == 1:
                        img = img.squeeze(-1)
                    ax.imshow(img, cmap=cmap, vmin=0, vmax=1, aspect='equal')
                else:
                    ax.set_facecolor('#f0f0f0')
                    ax.text(0.5, 0.5, "N/A", ha='center', va='center',
                           fontsize=font_size - 2, color='#999')

                ax.set_xticks([])
                ax.set_yticks([])

                # Hide individual spines (we'll draw group rectangles instead)
                for spine in ax.spines.values():
                    spine.set_visible(False)

        # Add column titles at the top
        if show_labels:
            show_beta = self.show_beta_cb.isChecked()
            for col_idx, col_config in enumerate(columns):
                ax = axes[0][col_idx]
                title = col_config.title
                if show_beta and col_config.col_type == GridColumnConfig.TYPE_TEST:
                    beta_pct = self._compute_beta_percent(col_config)
                    if beta_pct is not None:
                        title = f"{title}\nβ = {beta_pct:.1f}%"
                ax.set_title(title, fontsize=font_size, fontweight='bold', pad=15)

        # Add row labels on the left (outside the axes)
        if show_row_labels:
            for row_idx in range(n_rows):
                ax = axes[row_idx][0]
                label = row_labels[row_idx] if row_idx < len(row_labels) else f"Sample #{row_idx+1}"
                ax.annotate(label, xy=(-0.15, 0.5), xycoords='axes fraction',
                           fontsize=font_size, fontweight='bold', ha='right', va='center')

        # Convert border padding to figure fractions
        border_padding_x = border_padding_px / fig_width_px
        border_padding_y = border_padding_px / fig_height_px

        # Helper to calculate axis position in figure fractions (without needing draw)
        def get_ax_bounds(row_idx, col_idx):
            x_px = left_margin_px + col_idx * (image_size_px + col_gap_px)
            y_px = bottom_margin_px + (n_rows - 1 - row_idx) * (image_size_px + row_gap_px)
            x0 = x_px / fig_width_px
            y0 = y_px / fig_height_px
            x1 = (x_px + image_size_px) / fig_width_px
            y1 = (y_px + image_size_px) / fig_height_px
            return x0, y0, x1, y1

        # Draw group borders (rectangles around column groups)
        if show_borders and n_rows > 0:
            # Draw border around Ground Truth columns
            if gt_cols:
                first_gt = gt_cols[0]
                last_gt = gt_cols[-1]

                # Get bounds for first and last GT column
                tl_x0, _, _, tl_y1 = get_ax_bounds(0, first_gt)
                _, br_y0, br_x1, _ = get_ax_bounds(n_rows - 1, last_gt)

                rect_x = tl_x0 - border_padding_x
                rect_y = br_y0 - border_padding_y
                rect_width = (br_x1 - tl_x0) + 2 * border_padding_x
                rect_height = (tl_y1 - br_y0) + 2 * border_padding_y

                rect = Rectangle((rect_x, rect_y), rect_width, rect_height,
                                fill=False, edgecolor=gt_color,
                                linewidth=border_width, transform=self._figure.transFigure,
                                clip_on=False)
                self._figure.add_artist(rect)

            # Draw border around Test columns
            if test_cols:
                first_test = test_cols[0]
                last_test = test_cols[-1]

                # Get bounds for first and last test column
                tl_x0, _, _, tl_y1 = get_ax_bounds(0, first_test)
                _, br_y0, br_x1, _ = get_ax_bounds(n_rows - 1, last_test)

                rect_x = tl_x0 - border_padding_x
                rect_y = br_y0 - border_padding_y
                rect_width = (br_x1 - tl_x0) + 2 * border_padding_x
                rect_height = (tl_y1 - br_y0) + 2 * border_padding_y

                rect = Rectangle((rect_x, rect_y), rect_width, rect_height,
                                fill=False, edgecolor=test_color,
                                linewidth=border_width, transform=self._figure.transFigure,
                                clip_on=False)
                self._figure.add_artist(rect)

        # Add "Sampling ratio" header line above test columns
        if show_ratio_header and test_cols and show_labels:
            first_test_col = test_cols[0]
            last_test_col = test_cols[-1]

            # Get bounds for first and last test column
            first_x0, _, _, first_y1 = get_ax_bounds(0, first_test_col)
            _, _, last_x1, _ = get_ax_bounds(0, last_test_col)

            # Position line above the titles
            line_y = first_y1 + 40 / fig_height_px
            line_x_start = first_x0
            line_x_end = last_x1

            self._figure.add_artist(Line2D(
                [line_x_start, line_x_end], [line_y, line_y],
                transform=self._figure.transFigure,
                color='black', linewidth=1.5, clip_on=False
            ))

            text_x = (line_x_start + line_x_end) / 2
            self._figure.text(text_x, line_y + 0.02, "Sampling ratio",
                            fontsize=font_size, ha='center', va='bottom', fontweight='bold')

        self._canvas.draw()

    def _apply_canvas_size(self):
        """Apply the canvas size based on natural figure dimensions."""
        if not hasattr(self, '_natural_width_px'):
            return

        # Always use natural size (scale 1.0) to avoid layout issues
        width = int(self._natural_width_px)
        height = int(self._natural_height_px)

        # Set canvas fixed size to match figure
        self._canvas.setFixedSize(width, height)

        # Force visual update
        self._canvas.updateGeometry()
        self._scroll_area.viewport().update()

        # Update scale label (always 100% now)
        self._scale_label.setText(f"Size: {width}×{height}px")

    def _fit_to_window(self):
        """Adjust image size to fit the figure in the visible scroll area."""
        columns = self.column_list.get_columns()
        row_indices = self._get_row_indices()

        if not columns or not row_indices:
            return

        n_rows = len(row_indices)
        n_cols = len(columns)

        # Get available space
        viewport = self._scroll_area.viewport()
        available_width = viewport.width() - 40  # Leave margin
        available_height = viewport.height() - 40

        # Current gap values
        row_gap = self.row_spacing_spin.value()
        col_gap = self.col_spacing_spin.value()

        # Calculate margins (scale with font size)
        show_row_labels = self.show_row_labels_cb.isChecked()
        show_labels = self.show_labels_cb.isChecked()
        show_ratio_header = self.show_ratio_header_cb.isChecked()
        font_size = self.font_size_spin.value()

        left_margin = int(font_size * 12) if show_row_labels else 10
        right_margin = 10
        top_margin = int(font_size * 5) if show_labels else 20
        if show_ratio_header and show_labels:
            top_margin += int(font_size * 2.5)
        if self.show_beta_cb.isChecked() and show_labels:
            top_margin += int(font_size * 2)
        bottom_margin = 10

        # Calculate max image size that fits
        # Width: left_margin + n_cols * img_size + (n_cols-1) * col_gap + right_margin <= available_width
        # Height: top_margin + n_rows * img_size + (n_rows-1) * row_gap + bottom_margin <= available_height
        content_width = available_width - left_margin - right_margin - (n_cols - 1) * col_gap
        content_height = available_height - top_margin - bottom_margin - (n_rows - 1) * row_gap

        max_img_size_w = content_width / n_cols if n_cols > 0 else 256
        max_img_size_h = content_height / n_rows if n_rows > 0 else 256

        new_img_size = int(min(max_img_size_w, max_img_size_h))
        new_img_size = max(32, min(256, new_img_size))  # Clamp to valid range

        # Update the image size spinbox (this will trigger _update_preview)
        self.image_size_spin.setValue(new_img_size)

    def _on_save(self):
        """Save the current figure."""
        self._save_figure(self._figure, "samples_grid.png")
