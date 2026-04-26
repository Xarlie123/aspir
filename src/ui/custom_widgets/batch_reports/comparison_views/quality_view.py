"""
Quality view for Batch Reports - displays quality metrics charts.
"""
import logging
from typing import List, Dict, Any, Optional

import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QCheckBox, QFileDialog, QMessageBox, QComboBox, QMenu,
    QSplitter, QListWidget, QListWidgetItem, QGroupBox, QPushButton,
    QDialog, QStackedWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt, pyqtSignal

from ui.utils.file_formats import safe_test_dirname
from ui.custom_widgets.batch_reports.comparison_views.chart_config_popup import (
    ChartConfigPopup, CustomNavigationToolbar
)


class QualityView(QWidget):
    """
    Quality view displaying quality metrics charts.

    Features:
    - Left menu to select chart type
    - Quality Metrics Comparison: grouped bar chart comparing noisy vs denoised
    - Metrics per Image: line charts showing per-image values
    - Metrics Histogram: distribution of metric values
    - Selectable metrics via checkboxes (PSNR, SSIM, LPIPS)
    - Configurable legend position
    - Quality Metric Preview button for per-image visualization
    - Right-click to save chart
    """

    # Signal emitted when preview is requested
    previewRequested = pyqtSignal()

    # Chart type indices
    CHART_COMPARISON = 0
    CHART_PSNR_VS_RATIO = 1
    CHART_PER_IMAGE = 2
    CHART_HISTOGRAM = 3
    CHART_TABLE = 4

    def __init__(self, logger=None, parent=None):
        super().__init__(parent)

        if logger:
            self.logger = logger.getChild("QualityView")
        else:
            self.logger = logging.getLogger("QualityView")

        self._tests: List[Dict[str, Any]] = []

        # Chart configuration with defaults
        self._chart_config = {
            'axes': {
                'title': '',  # Empty = auto
                'title_fontsize': 13,
                'xlabel': '',
                'xlabel_fontsize': 11,
                'xtick_fontsize': 8,
                'ylabel': '',
                'ylabel_fontsize': 11,
                'ytick_fontsize': 8,
                'auto_scale': True,
                'ymin': 0.0,
                'ymax': 1.0,
            },
            'legend': {
                'position': 0,  # Inside (upper right)
                'fontsize': 9,
                'frameon': True,
                'shadow': False,
                'fancybox': True,
                'framealpha': 0.8,
                'ncol': 1,
            },
            'colors': {
                'psnr': '#1f77b4',
                'ssim': '#2ca02c',
                'lpips': '#d62728',
                'noisy': '#1f77b4',
                'denoised': '#2ca02c',
                'bar_alpha': 0.8,
                'hist_alpha': 0.7,
            }
        }

        self._setup_ui()

    def _setup_ui(self):
        """Setup the quality view UI with left menu."""
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)

        # Splitter for menu and chart area
        splitter = QSplitter(Qt.Horizontal)

        # Left panel: Chart type selection
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(5, 5, 5, 5)
        left_layout.setSpacing(10)

        # Chart type list - styled like Single Test -> Reports menu
        chart_label = QLabel("Chart Type:")
        chart_label.setStyleSheet("font-weight: bold; color: #333;")
        left_layout.addWidget(chart_label)

        self.chart_list = QListWidget()
        self.chart_list.setMaximumWidth(220)
        self.chart_list.setMinimumWidth(180)
        self.chart_list.addItem("Quality Metrics Comparison")
        self.chart_list.addItem("PSNR vs Sampling Ratio")
        self.chart_list.addItem("Metrics per Image")
        self.chart_list.addItem("Metrics Histogram")
        self.chart_list.addItem("Quality Metrics Table")
        self.chart_list.setCurrentRow(0)
        self.chart_list.currentRowChanged.connect(self._on_chart_type_changed)
        self.chart_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #ccc;
                border-radius: 4px;
                background-color: #f5f5f5;
            }
            QListWidget::item {
                padding: 10px;
                border-bottom: 1px solid #e0e0e0;
            }
            QListWidget::item:selected {
                background-color: #0078d7;
                color: white;
            }
            QListWidget::item:hover {
                background-color: #e5e5e5;
            }
        """)
        left_layout.addWidget(self.chart_list)

        # Metrics selection group
        metrics_group = QGroupBox("Metrics")
        metrics_group.setMaximumWidth(220)
        metrics_group.setMinimumWidth(180)
        metrics_layout = QVBoxLayout(metrics_group)
        metrics_layout.setSpacing(5)

        self.psnr_checkbox = QCheckBox("PSNR")
        self.psnr_checkbox.setChecked(True)
        self.psnr_checkbox.setToolTip("Peak Signal-to-Noise Ratio (higher is better)")
        self.psnr_checkbox.stateChanged.connect(self._on_checkbox_changed)
        metrics_layout.addWidget(self.psnr_checkbox)

        self.ssim_checkbox = QCheckBox("SSIM")
        self.ssim_checkbox.setChecked(True)
        self.ssim_checkbox.setToolTip("Structural Similarity Index (higher is better)")
        self.ssim_checkbox.stateChanged.connect(self._on_checkbox_changed)
        metrics_layout.addWidget(self.ssim_checkbox)

        self.lpips_checkbox = QCheckBox("LPIPS")
        self.lpips_checkbox.setChecked(True)
        self.lpips_checkbox.setToolTip("Learned Perceptual Image Patch Similarity (lower is better)")
        self.lpips_checkbox.stateChanged.connect(self._on_checkbox_changed)
        metrics_layout.addWidget(self.lpips_checkbox)

        left_layout.addWidget(metrics_group)

        # Quality Metrics Preview button
        self.preview_btn = QPushButton("Quality Metrics Preview")
        self.preview_btn.setMaximumWidth(220)
        self.preview_btn.setMinimumWidth(180)
        self.preview_btn.setMinimumHeight(36)
        self.preview_btn.setEnabled(False)  # Disabled until data is available
        self.preview_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
                font-size: 12px;
                padding: 8px;
            }
            QPushButton:hover:enabled {
                background-color: #45a049;
            }
            QPushButton:pressed:enabled {
                background-color: #3d8b40;
            }
            QPushButton:disabled {
                background-color: #ccc;
                color: #888;
            }
        """)
        self.preview_btn.setToolTip("Preview images and per-image metrics (requires per-image data)")
        self.preview_btn.clicked.connect(self._on_preview_clicked)
        left_layout.addWidget(self.preview_btn)

        left_layout.addStretch()

        splitter.addWidget(left_panel)

        # Right panel: Chart area
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(5, 5, 5, 5)
        right_layout.setSpacing(5)

        # Chart area
        self.figure = Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet("background-color: white;")

        # Enable right-click context menu on canvas
        self.canvas.setContextMenuPolicy(Qt.CustomContextMenu)
        self.canvas.customContextMenuRequested.connect(self._show_context_menu)

        # Navigation toolbar with custom chart config button
        self.toolbar = CustomNavigationToolbar(
            self.canvas, self,
            config_callback=self._on_open_chart_config
        )

        right_layout.addWidget(self.toolbar)

        # Table mode header: experiment filter + hint about right-click export.
        self.table_header = QWidget()
        table_header_layout = QHBoxLayout(self.table_header)
        table_header_layout.setContentsMargins(0, 0, 0, 4)
        table_header_layout.addWidget(QLabel("Experiment:"))
        self.experiment_combo = QComboBox()
        self.experiment_combo.setMinimumWidth(260)
        self.experiment_combo.currentIndexChanged.connect(self._on_experiment_filter_changed)
        table_header_layout.addWidget(self.experiment_combo)
        table_header_layout.addStretch()
        table_header_layout.addWidget(QLabel(
            "<span style='color:#888'>Right-click to export as CSV / LaTeX</span>"
        ))
        self.table_header.setVisible(False)
        right_layout.addWidget(self.table_header)

        # Stack: [0] chart canvas, [1] metrics table
        self.view_stack = QStackedWidget()
        self.view_stack.addWidget(self.canvas)
        self.metrics_table = QTableWidget()
        self.metrics_table.setAlternatingRowColors(True)
        self.metrics_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.metrics_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.metrics_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.metrics_table.customContextMenuRequested.connect(
            self._show_table_context_menu
        )
        self.metrics_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        self.view_stack.addWidget(self.metrics_table)
        right_layout.addWidget(self.view_stack, 1)

        # Info label
        self.info_label = QLabel("Load experiments to see quality comparison charts")
        self.info_label.setStyleSheet("color: #666; font-size: 11px;")
        self.info_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.info_label)

        splitter.addWidget(right_panel)

        # Set splitter sizes (menu:chart = 1:4)
        splitter.setSizes([180, 720])

        main_layout.addWidget(splitter)

    def _show_context_menu(self, pos):
        """Show context menu for saving the chart."""
        menu = QMenu(self)
        save_action = menu.addAction("Save chart as...")
        save_action.triggered.connect(self._on_save_chart)
        menu.exec_(self.canvas.mapToGlobal(pos))

    def _on_save_chart(self):
        """Save the chart to a file."""
        if not self._tests:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Chart",
            "quality_chart.png",
            "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg);;All Files (*.*)"
        )

        if not file_path:
            return

        try:
            self.figure.savefig(file_path, dpi=150, bbox_inches='tight',
                               facecolor='white', edgecolor='none')
            self.logger.info("Saved chart to %s", file_path)
            QMessageBox.information(
                self, "Save Complete",
                f"Chart saved to:\n{file_path}"
            )
        except Exception as e:
            self.logger.error("Failed to save chart: %s", e)
            QMessageBox.warning(self, "Save Error", f"Failed to save chart:\n{e}")

    def set_tests(self, tests: List[Dict[str, Any]]):
        """
        Set the tests to display in the charts.

        Args:
            tests: List of test dictionaries with metrics
        """
        self._tests = tests
        self._update_preview_button_state()
        self._refresh_chart()

    def _update_preview_button_state(self):
        """Enable/disable preview button based on data availability."""
        has_preview_data = self._has_preview_data()
        self.preview_btn.setEnabled(has_preview_data)
        if has_preview_data:
            self.preview_btn.setToolTip("Preview images and per-image metrics")
        else:
            self.preview_btn.setToolTip("Preview not available (requires per-image data from batch test)")

    def _has_preview_data(self) -> bool:
        """Check if any test has per-image quality data for preview."""
        for test in self._tests:
            per_image = test.get("quality_per_image", {})
            if per_image:
                # Check if we have any per-image metric arrays with data
                for key in ["psnr_noisy", "psnr_denoised", "ssim_noisy", "ssim_denoised"]:
                    if per_image.get(key) and len(per_image[key]) > 0:
                        return True
        return False

    def _on_preview_clicked(self):
        """Handle preview button click."""
        self.previewRequested.emit()

    def _on_chart_type_changed(self, index: int):
        """Handle chart type selection change."""
        self._refresh_chart()

    def _on_checkbox_changed(self, state: int = None):
        """Handle checkbox state change."""
        self._refresh_chart()

    def _on_open_chart_config(self):
        """Open chart configuration dialog."""
        popup = ChartConfigPopup(parent=self, logger=self.logger)
        popup.set_config(self._chart_config)

        if popup.exec_() == QDialog.Accepted:
            self._chart_config = popup.get_config()
            self.logger.debug("Chart config updated: %s", self._chart_config)
            self._refresh_chart()

    def _apply_axes_config(self, ax, default_title: str = "", default_xlabel: str = "",
                           default_ylabel: str = ""):
        """Apply axes configuration to a matplotlib axis.

        Args:
            ax: The matplotlib axis to configure
            default_title: Default title if not specified in config
            default_xlabel: Default x-label if not specified in config
            default_ylabel: Default y-label if not specified in config
        """
        axes_cfg = self._chart_config.get('axes', {})

        # Title
        title = axes_cfg.get('title', '') or default_title
        title_fontsize = axes_cfg.get('title_fontsize', 13)
        if title:
            ax.set_title(title, fontsize=title_fontsize, fontweight='bold')

        # X-axis label
        xlabel = axes_cfg.get('xlabel', '') or default_xlabel
        xlabel_fontsize = axes_cfg.get('xlabel_fontsize', 11)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=xlabel_fontsize)

        # Y-axis label
        ylabel = axes_cfg.get('ylabel', '') or default_ylabel
        ylabel_fontsize = axes_cfg.get('ylabel_fontsize', 11)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)

        # Tick label font sizes
        xtick_fontsize = axes_cfg.get('xtick_fontsize', 8)
        ytick_fontsize = axes_cfg.get('ytick_fontsize', 8)
        ax.tick_params(axis='x', labelsize=xtick_fontsize)
        ax.tick_params(axis='y', labelsize=ytick_fontsize)

        # Y-axis scale (only if not auto)
        if not axes_cfg.get('auto_scale', True):
            ymin = axes_cfg.get('ymin', 0)
            ymax = axes_cfg.get('ymax', 1)
            ax.set_ylim(ymin, ymax)

    def _refresh_chart(self):
        """Refresh the chart based on current settings."""
        chart_type = self.chart_list.currentRow()

        # Table mode uses its own widget, not the matplotlib canvas.
        if chart_type == self.CHART_TABLE:
            self.view_stack.setCurrentIndex(1)
            self.toolbar.setVisible(False)
            self.table_header.setVisible(True)
            self._refresh_experiment_combo()
            self._populate_metrics_table()
            experiment_count = len(set(t.get("_experiment_name", "") for t in self._tests))
            self.info_label.setText(
                f"Table: {len(self._tests)} tests from {experiment_count} experiment(s)"
            )
            return

        # Chart modes → show the canvas.
        self.view_stack.setCurrentIndex(0)
        self.toolbar.setVisible(True)
        self.table_header.setVisible(False)

        self.figure.clear()

        if not self._tests:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "No data to display\nLoad experiments first",
                   ha='center', va='center', fontsize=14, color='#999')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            self.canvas.draw()
            return

        # Check which metrics are selected
        show_psnr = self.psnr_checkbox.isChecked()
        show_ssim = self.ssim_checkbox.isChecked()
        show_lpips = self.lpips_checkbox.isChecked()

        # PSNR-vs-Sampling-Ratio is PSNR-only by construction, so the
        # metric checkboxes don't gate it.
        if (chart_type != self.CHART_PSNR_VS_RATIO
                and not (show_psnr or show_ssim or show_lpips)):
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "Select at least one metric",
                   ha='center', va='center', fontsize=14, color='#999')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            self.canvas.draw()
            return

        if chart_type == self.CHART_COMPARISON:
            self._draw_comparison_chart(show_psnr, show_ssim, show_lpips)
        elif chart_type == self.CHART_PSNR_VS_RATIO:
            # PSNR-only by design (the headline figure for the paper);
            # the metric checkboxes are ignored on purpose, SSIM/LPIPS
            # belong in the table.
            self._draw_psnr_vs_sampling_chart()
        elif chart_type == self.CHART_PER_IMAGE:
            self._draw_per_image_chart(show_psnr, show_ssim, show_lpips)
        else:  # CHART_HISTOGRAM
            self._draw_histogram_chart(show_psnr, show_ssim, show_lpips)

        # Adjust layout based on legend position (applies to all chart types)
        legend_pos = self._chart_config['legend']['position']
        if legend_pos == 4:  # Right side (outside)
            self.figure.tight_layout(rect=[0, 0.05, 0.85, 1])
        elif legend_pos == 5:  # Below (outside)
            self.figure.tight_layout(rect=[0, 0.15, 1, 1])
        else:  # Inside positions (0-3)
            self.figure.tight_layout(rect=[0, 0.05, 1, 1])

        self.canvas.draw()

        # Update info
        experiment_count = len(set(t.get("_experiment_name", "") for t in self._tests))
        self.info_label.setText(
            f"Showing {len(self._tests)} tests from {experiment_count} experiment(s)"
        )

    # ------------------------------------------------------------------
    # Quality Metrics Table
    # ------------------------------------------------------------------

    TABLE_HEADERS = [
        "M/N (%)",
        "M",
        "PSNR Recon (dB)",
        "SSIM Recon",
        "PSNR Denoised (dB)",
        "SSIM Denoised",
        "LPIPS Denoised",
        "ΔPSNR (dB)",
    ]

    def _refresh_experiment_combo(self):
        """Rebuild the 'Experiment' filter dropdown from loaded tests."""
        combo = self.experiment_combo
        prev = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("All experiments", None)
        seen = []
        for t in self._tests:
            name = t.get("_experiment_name", "")
            if name and name not in seen:
                seen.append(name)
        for name in seen:
            combo.addItem(name, name)
        # Restore previous selection if still present.
        if prev is not None:
            idx = combo.findData(prev)
            if idx >= 0:
                combo.setCurrentIndex(idx)
        combo.blockSignals(False)

    def _on_experiment_filter_changed(self, _index: int):
        if self.chart_list.currentRow() == self.CHART_TABLE:
            self._populate_metrics_table()

    def _filtered_tests(self) -> List[Dict[str, Any]]:
        """Apply the experiment filter to self._tests."""
        exp = self.experiment_combo.currentData()
        if not exp:
            return list(self._tests)
        return [t for t in self._tests if t.get("_experiment_name") == exp]

    def _n_pixels_for_test(self, test: Dict[str, Any]) -> Optional[int]:
        """Resolve n_pixels for a test (used for M/N and M)."""
        meta = test.get("_experiment_metadata") or {}
        img_size = None
        if isinstance(meta, dict):
            ds = meta.get("dataset_info", {}) or {}
            if "img_size" in ds:
                try:
                    img_size = int(ds["img_size"])
                except (TypeError, ValueError):
                    img_size = None
        if img_size is None:
            # Fall back to reading the stored test_images.npz shape. Cache on
            # the test dict so we don't hit disk repeatedly.
            cached = test.get("_img_size_cached")
            if cached is not None:
                img_size = cached
            else:
                import numpy as np
                from pathlib import Path
                try:
                    batch_dir = test.get("_batch_dir")
                    name = test.get("_original_name", test.get("name", ""))
                    if batch_dir and name:
                        safe = safe_test_dirname(name)
                        npz = Path(batch_dir) / "data" / safe / "test_images.npz"
                        if npz.exists():
                            with np.load(str(npz)) as data:
                                if "originals" in data.files:
                                    img_size = int(data["originals"].shape[-1])
                                    test["_img_size_cached"] = img_size
                except Exception as e:
                    self.logger.debug("img_size fallback failed: %s", e)
        if img_size:
            return img_size * img_size
        return None

    def _n_patterns_for_test(self, test: Dict[str, Any]) -> Optional[int]:
        n = test.get("timing_num_patterns")
        if n is not None:
            return int(n)
        mask_type = (test.get("mask_type") or "").lower()
        if mask_type == "scatter":
            return test.get("scatter_num_patterns")
        if mask_type.startswith("hadamard") or mask_type == "cal_sal":
            lo = test.get("hadamard_min_idx")
            hi = test.get("hadamard_max_idx")
            if lo is not None and hi is not None:
                return max(0, int(hi) - int(lo))
        return None

    def _table_rows(self) -> List[Dict[str, Any]]:
        """Build one dict per filtered test with all table column values.

        Values are the batch-averaged per-test metrics already stored in the
        report. Rows are sorted by β ascending so the table reads from lowest
        to highest sampling ratio.
        """
        rows = []
        for test in self._filtered_tests():
            n_patterns = self._n_patterns_for_test(test)
            n_pixels = self._n_pixels_for_test(test)
            beta = (100.0 * n_patterns / n_pixels
                    if n_patterns and n_pixels else None)
            psnr_r = test.get("psnr_recons")
            psnr_d = test.get("psnr_denoised")
            rows.append({
                "name": test.get("name", ""),
                "beta": beta,
                "M": n_patterns,
                "psnr_recon": psnr_r,
                "ssim_recon": test.get("ssim_recons"),
                "psnr_denoised": psnr_d,
                "ssim_denoised": test.get("ssim_denoised"),
                "lpips_denoised": test.get("lpips_denoised"),
                "delta_psnr": (psnr_d - psnr_r)
                    if (psnr_d is not None and psnr_r is not None) else None,
            })
        # Sort descending by β so the highest sampling ratio (most
        # measurements) appears first — matches the conventional layout in
        # compressive-imaging papers.
        rows.sort(key=lambda r: (r["beta"] if r["beta"] is not None else -1),
                  reverse=True)
        return rows

    def _populate_metrics_table(self):
        """Fill the QTableWidget with the filtered rows."""
        rows = self._table_rows()
        self.metrics_table.clear()
        self.metrics_table.setColumnCount(len(self.TABLE_HEADERS))
        self.metrics_table.setHorizontalHeaderLabels(self.TABLE_HEADERS)
        self.metrics_table.setRowCount(len(rows))

        def fmt(value, spec):
            if value is None:
                return "—"
            return spec.format(value)

        for r, row in enumerate(rows):
            cells = [
                fmt(row["beta"], "{:.1f}"),
                fmt(row["M"], "{}"),
                fmt(row["psnr_recon"], "{:.2f}"),
                fmt(row["ssim_recon"], "{:.4f}"),
                fmt(row["psnr_denoised"], "{:.2f}"),
                fmt(row["ssim_denoised"], "{:.4f}"),
                fmt(row["lpips_denoised"], "{:.4f}"),
                fmt(row["delta_psnr"],
                    "{:+.2f}" if row["delta_psnr"] is not None else "{}"),
            ]
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                self.metrics_table.setItem(r, c, item)

        # Tooltip with test name on each row's first cell — handy when tests
        # are renamed and β alone doesn't identify the row uniquely.
        for r, row in enumerate(rows):
            it = self.metrics_table.item(r, 0)
            if it is not None:
                it.setToolTip(row["name"])

        self.metrics_table.resizeColumnsToContents()

    def _show_table_context_menu(self, pos):
        menu = QMenu(self.metrics_table)
        menu.addAction("Save as CSV…", self._export_table_csv)
        menu.addAction("Save as LaTeX…", self._export_table_latex)
        menu.exec_(self.metrics_table.viewport().mapToGlobal(pos))

    def _export_table_csv(self):
        rows = self._table_rows()
        if not rows:
            QMessageBox.information(self, "Nothing to export", "The table is empty.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save table as CSV",
            "quality_metrics.csv",
            "CSV Files (*.csv);;All Files (*.*)",
        )
        if not path:
            return
        try:
            import csv
            with open(path, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["Test"] + self.TABLE_HEADERS)
                for row in rows:
                    w.writerow([
                        row["name"],
                        f"{row['beta']:.2f}" if row["beta"] is not None else "",
                        row["M"] if row["M"] is not None else "",
                        f"{row['psnr_recon']:.4f}" if row["psnr_recon"] is not None else "",
                        f"{row['ssim_recon']:.6f}" if row["ssim_recon"] is not None else "",
                        f"{row['psnr_denoised']:.4f}" if row["psnr_denoised"] is not None else "",
                        f"{row['ssim_denoised']:.6f}" if row["ssim_denoised"] is not None else "",
                        f"{row['lpips_denoised']:.6f}" if row["lpips_denoised"] is not None else "",
                        f"{row['delta_psnr']:.4f}" if row["delta_psnr"] is not None else "",
                    ])
            QMessageBox.information(self, "Saved", f"CSV saved to:\n{path}")
        except Exception as e:
            self.logger.error("CSV export failed: %s", e, exc_info=True)
            QMessageBox.warning(self, "Export error", f"CSV export failed:\n{e}")

    def _export_table_latex(self):
        rows = self._table_rows()
        if not rows:
            QMessageBox.information(self, "Nothing to export", "The table is empty.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save table as LaTeX",
            "quality_metrics.tex",
            "LaTeX Files (*.tex);;All Files (*.*)",
        )
        if not path:
            return

        def cell(value, spec, empty="--"):
            if value is None:
                return empty
            return spec.format(value)

        try:
            lines = []
            lines.append("% Generated by ASPIR — Quality Metrics Table")
            lines.append(r"\begin{tabular}{rr rr rrr r}")
            lines.append(r"\toprule")
            lines.append(
                r" & & \multicolumn{2}{c}{Reconstructed} "
                r"& \multicolumn{3}{c}{Denoised (DNN)} & \\"
            )
            lines.append(r"\cmidrule(lr){3-4} \cmidrule(lr){5-7}")
            lines.append(
                r"$M/N$ (\%) & $M$ & PSNR (dB) & SSIM "
                r"& PSNR (dB) & SSIM & LPIPS & $\Delta$PSNR (dB) \\"
            )
            lines.append(r"\midrule")
            for row in rows:
                lines.append(
                    " & ".join([
                        cell(row["beta"], "{:.1f}"),
                        cell(row["M"], "{}"),
                        cell(row["psnr_recon"], "{:.2f}"),
                        cell(row["ssim_recon"], "{:.4f}"),
                        cell(row["psnr_denoised"], "{:.2f}"),
                        cell(row["ssim_denoised"], "{:.4f}"),
                        cell(row["lpips_denoised"], "{:.4f}"),
                        cell(row["delta_psnr"], "{:+.2f}"),
                    ]) + r" \\"
                )
            lines.append(r"\bottomrule")
            lines.append(r"\end{tabular}")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")
            QMessageBox.information(self, "Saved", f"LaTeX saved to:\n{path}")
        except Exception as e:
            self.logger.error("LaTeX export failed: %s", e, exc_info=True)
            QMessageBox.warning(self, "Export error", f"LaTeX export failed:\n{e}")

    def _draw_comparison_chart(self, show_psnr: bool, show_ssim: bool, show_lpips: bool):
        """Draw Quality Metrics Comparison chart (grouped bar chart).

        X-axis shows "TestName Noisy" and "TestName Denoised" for each test.
        Bars are grouped by metric (PSNR, SSIM, LPIPS) with different colors.
        """
        ax = self.figure.add_subplot(111)

        # Collect unique test names preserving order from self._tests
        all_tests = []
        seen = set()
        for test in self._tests:
            test_name = test.get("name", "Unknown")
            if test_name not in seen:
                all_tests.append(test_name)
                seen.add(test_name)

        if not all_tests:
            ax.text(0.5, 0.5, "No valid metric data available",
                   ha='center', va='center', fontsize=12, color='#999')
            ax.axis('off')
            return

        # Build list of metrics to show with their properties
        # (key_base, label, higher_is_better, color, format_str)
        colors_cfg = self._chart_config['colors']
        metrics_to_show = []
        if show_psnr:
            metrics_to_show.append(("psnr", "PSNR \u2191", True, colors_cfg['psnr'], '{:.1f}'))
        if show_ssim:
            metrics_to_show.append(("ssim", "SSIM \u2191", True, colors_cfg['ssim'], '{:.3f}'))
        if show_lpips:
            metrics_to_show.append(("lpips", "LPIPS \u2193", False, colors_cfg['lpips'], '{:.3f}'))

        n_metrics = len(metrics_to_show)
        n_tests = len(all_tests)

        # X-axis: for each test we have 2 positions (Noisy, Denoised)
        n_positions = n_tests * 2
        x = np.arange(n_positions)

        # Build x-axis labels: "Test1 Noisy", "Test1 Denoised", ...
        x_labels = []
        for test_name in all_tests:
            short_name = test_name if len(test_name) <= 15 else test_name[:12] + "..."
            x_labels.append(f"{short_name}\nNoisy")
            x_labels.append(f"{short_name}\nDenoised")

        bar_width = 0.8 / n_metrics

        legend_handles = []
        legend_labels = []

        # Map each metric to its own absolute reference so bars remain
        # visible regardless of how good or bad the test set looks. The
        # previous global min-max approach pushed the worst value of every
        # colour to height 0, which made the legitimate "this is bad"
        # bars literally disappear — see issue raised on the Celeb batch.
        #
        # Picks per metric:
        #   PSNR  → divide by 40 dB (≈ "excellent reconstruction"; real
        #           IR results rarely exceed this) and clamp to [0, 1].
        #   SSIM  → already in [0, 1], keep as-is.
        #   LPIPS → invert as 1 - LPIPS so the "tall = better" semantic
        #           still holds, clamped to [0, 1].
        # Bar value labels show the raw number, so absolute precision is
        # not lost — only the bar height is mapped to a fixed scale.
        PSNR_REFERENCE_DB = 40.0

        def _normalize(metric: str, value: float) -> float:
            if metric == "psnr":
                return max(0.0, min(1.0, value / PSNR_REFERENCE_DB))
            if metric == "ssim":
                return max(0.0, min(1.0, value))
            if metric == "lpips":
                return max(0.0, min(1.0, 1.0 - value))
            return max(0.0, min(1.0, value))

        for metric_idx, (metric_base, metric_label, higher_is_better, color, fmt) in enumerate(metrics_to_show):
            noisy_key = f"{metric_base}_recons"
            denoised_key = f"{metric_base}_denoised"

            all_values = []
            all_norm_values = []

            for test_name in all_tests:
                test = next((t for t in self._tests if t.get("name") == test_name), None)
                if test:
                    noisy_val = self._get_nested_value(test, noisy_key)
                    denoised_val = self._get_nested_value(test, denoised_key)
                    noisy_val = noisy_val if noisy_val is not None else 0
                    denoised_val = denoised_val if denoised_val is not None else 0
                else:
                    noisy_val = 0
                    denoised_val = 0

                all_values.extend([noisy_val, denoised_val])
                all_norm_values.extend([
                    _normalize(metric_base, noisy_val),
                    _normalize(metric_base, denoised_val),
                ])

            offset = (metric_idx - (n_metrics - 1) / 2) * bar_width
            bars = ax.bar(x + offset, all_norm_values, bar_width * 0.9,
                         label=metric_label, color=color, alpha=colors_cfg['bar_alpha'])

            if metric_idx == 0 or bars not in legend_handles:
                legend_handles.append(bars)
                legend_labels.append(metric_label)

            for bar, val in zip(bars, all_values):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                           fmt.format(val), ha='center', va='bottom', fontsize=7)

        # Apply axes configuration
        self._apply_axes_config(
            ax,
            default_title="Quality Metrics Comparison",
            default_ylabel="Quality Score (normalized)"
        )

        ax.set_xticks(x)
        axes_cfg = self._chart_config.get('axes', {})
        ax.set_xticklabels(x_labels, fontsize=axes_cfg.get('xtick_fontsize', 8))

        # Set Y limits (only if auto_scale, otherwise _apply_axes_config handles it)
        if axes_cfg.get('auto_scale', True):
            ax.set_ylim(0, 1.15)

        ax.grid(axis='y', alpha=0.3)

        # Vertical separators between tests
        for i in range(1, n_tests):
            ax.axvline(x=i * 2 - 0.5, color='#ccc', linestyle='--', linewidth=0.8)

        # Legend with configuration
        legend_cfg = self._chart_config['legend']
        legend_pos = legend_cfg['position']
        legend_kwargs = {
            'fontsize': legend_cfg['fontsize'],
            'frameon': legend_cfg['frameon'],
            'shadow': legend_cfg['shadow'],
            'fancybox': legend_cfg['fancybox'],
            'framealpha': legend_cfg['framealpha'],
        }

        # Position mapping
        loc_map = {
            0: 'upper right',
            1: 'upper left',
            2: 'lower right',
            3: 'lower left',
        }

        if legend_pos in loc_map:  # Inside positions
            ax.legend(legend_handles, legend_labels, loc=loc_map[legend_pos],
                     ncol=legend_cfg['ncol'], **legend_kwargs)
        elif legend_pos == 4:  # Right side (outside)
            ax.legend(legend_handles, legend_labels, loc='center left',
                     bbox_to_anchor=(1.02, 0.5), ncol=legend_cfg['ncol'], **legend_kwargs)
        else:  # Below (outside)
            legend_kwargs['frameon'] = False  # No frame for below position
            ax.legend(legend_handles, legend_labels, loc='upper center',
                     bbox_to_anchor=(0.5, -0.15), ncol=n_metrics, **legend_kwargs)

    # ------------------------------------------------------------------
    # PSNR vs Sampling Ratio (paper-style figure)
    # ------------------------------------------------------------------

    # Markers/colours roughly mirror the user's hand-drawn spec:
    # red for the raw linear reconstruction, green for the NN-denoised
    # series. When several experiments are loaded the colours cycle
    # through tab10 and the experiment name is appended to the legend.
    _RECON_COLOR = '#d62728'
    _DENOISED_COLOR = '#2ca02c'
    # Used to colour-code experiments when several are loaded at once.
    # Same hex tones as matplotlib's tab10 minus the red/green that
    # already mean "recon" / "denoised" in the single-experiment case.
    _MULTI_EXPERIMENT_PALETTE = (
        '#1f77b4', '#9467bd', '#8c564b', '#e377c2',
        '#7f7f7f', '#17becf', '#bcbd22', '#ff7f0e',
    )

    def _psnr_vs_sampling_rows(self) -> List[Dict[str, Any]]:
        """Build per-test rows {experiment, beta, psnr_recons, psnr_denoised,
        std_recons, std_denoised} using ``_n_patterns_for_test`` and
        ``_n_pixels_for_test``. Tests missing either M or N (i.e. M/N can't
        be computed) are dropped — the chart cannot place them on the axis.
        """
        rows: List[Dict[str, Any]] = []
        for test in self._tests:
            m = self._n_patterns_for_test(test)
            n = self._n_pixels_for_test(test)
            if not m or not n:
                continue
            beta = 100.0 * float(m) / float(n)

            psnr_r = test.get("psnr_recons")
            psnr_d = test.get("psnr_denoised")
            if psnr_r is None and psnr_d is None:
                continue

            per_img = test.get("quality_per_image") or {}
            std_r = std_d = None
            try:
                arr = per_img.get("psnr_noisy") or per_img.get("psnr_recons")
                if arr:
                    std_r = float(np.std(arr))
            except Exception:
                std_r = None
            try:
                arr = per_img.get("psnr_denoised")
                if arr:
                    std_d = float(np.std(arr))
            except Exception:
                std_d = None

            rows.append({
                "experiment": test.get("_experiment_name", ""),
                "name": test.get("name", ""),
                "beta": beta,
                "psnr_recons": psnr_r,
                "psnr_denoised": psnr_d,
                "std_recons": std_r,
                "std_denoised": std_d,
            })
        return rows

    def _draw_psnr_vs_sampling_chart(self):
        """Plot PSNR (dB) vs sampling ratio M/N (%) with one (recon,
        denoised) pair per loaded experiment.

        The headline figure for the paper: line + markers, real dB on Y,
        true numeric M/N on X so the curvature is visible. Per-test
        ±1σ shading is drawn when the experiment was exported with
        ``quality_per_image`` arrays — otherwise we fall back to plain
        markers without a band.
        """
        ax = self.figure.add_subplot(111)
        rows = self._psnr_vs_sampling_rows()
        if not rows:
            ax.text(
                0.5, 0.5,
                "No tests with both M (num patterns) and N (img_size²)\n"
                "available — re-run the batch with timing enabled or\n"
                "make sure the report metadata carries dataset_info.",
                ha='center', va='center', fontsize=12, color='#999',
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            return

        # Group rows by experiment so each experiment becomes one
        # (recon, denoised) pair of lines.
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(row["experiment"], []).append(row)

        # Stable order: preserve the order experiments first appear.
        exp_order: List[str] = []
        for row in rows:
            if row["experiment"] not in exp_order:
                exp_order.append(row["experiment"])

        single_exp = len(exp_order) == 1

        for idx, exp_name in enumerate(exp_order):
            exp_rows = sorted(grouped[exp_name], key=lambda r: r["beta"])
            betas = [r["beta"] for r in exp_rows]
            recon = [r["psnr_recons"] for r in exp_rows]
            denoised = [r["psnr_denoised"] for r in exp_rows]
            std_r = [r["std_recons"] for r in exp_rows]
            std_d = [r["std_denoised"] for r in exp_rows]

            if single_exp:
                color_recon = self._RECON_COLOR
                color_den = self._DENOISED_COLOR
                lbl_recon = "Reconstructed (linear)"
                lbl_den = "Denoised (NN post-processing)"
            else:
                base = self._MULTI_EXPERIMENT_PALETTE[
                    idx % len(self._MULTI_EXPERIMENT_PALETTE)
                ]
                color_recon = base
                color_den = base
                short = exp_name[:18] + ("…" if len(exp_name) > 18 else "")
                lbl_recon = f"{short} — Recon"
                lbl_den = f"{short} — Denoised"

            # Reconstructed series — dashed line, circle markers
            valid_r = [(b, p, s) for b, p, s in zip(betas, recon, std_r)
                       if p is not None]
            if valid_r:
                bs = [v[0] for v in valid_r]
                ps = [v[1] for v in valid_r]
                ss = [v[2] for v in valid_r]
                ax.plot(bs, ps, marker='o', linestyle='--', linewidth=1.5,
                        color=color_recon, label=lbl_recon, markersize=7,
                        markerfacecolor=color_recon, markeredgecolor='white',
                        markeredgewidth=1.0)
                if all(s is not None for s in ss):
                    lo = [p - s for p, s in zip(ps, ss)]
                    hi = [p + s for p, s in zip(ps, ss)]
                    ax.fill_between(bs, lo, hi, color=color_recon, alpha=0.15)

            # Denoised series — solid line, square markers
            valid_d = [(b, p, s) for b, p, s in zip(betas, denoised, std_d)
                       if p is not None]
            if valid_d:
                bs = [v[0] for v in valid_d]
                ps = [v[1] for v in valid_d]
                ss = [v[2] for v in valid_d]
                ax.plot(bs, ps, marker='s', linestyle='-', linewidth=2.0,
                        color=color_den, label=lbl_den, markersize=7,
                        markerfacecolor=color_den, markeredgecolor='white',
                        markeredgewidth=1.0)
                if all(s is not None for s in ss):
                    lo = [p - s for p, s in zip(ps, ss)]
                    hi = [p + s for p, s in zip(ps, ss)]
                    ax.fill_between(bs, lo, hi, color=color_den, alpha=0.15)

        # Axis cosmetics. Titles/labels run through ``_apply_axes_config``
        # so the font-size + label-pad knobs from the chart config dialog
        # affect this plot too.
        self._apply_axes_config(
            ax,
            default_title="PSNR vs Sampling Ratio",
            default_xlabel="Sampling ratio M/N (%)",
            default_ylabel="PSNR (dB)",
        )
        ax.grid(True, alpha=0.3, linestyle=':')

        # Legend lives inside the plot by default (the trend usually
        # leaves room in the upper-left corner). Honour the global
        # legend-position setting from the chart config so users can
        # move it around if their data fills the axes.
        legend_cfg = self._chart_config['legend']
        legend_kwargs = {
            'fontsize': legend_cfg['fontsize'],
            'frameon': legend_cfg['frameon'],
            'shadow': legend_cfg['shadow'],
            'fancybox': legend_cfg['fancybox'],
            'framealpha': legend_cfg['framealpha'],
        }
        loc_map = {
            0: 'upper right', 1: 'upper left',
            2: 'lower right', 3: 'lower left',
        }
        legend_pos = legend_cfg['position']
        if legend_pos in loc_map:
            ax.legend(loc=loc_map[legend_pos],
                      ncol=legend_cfg['ncol'], **legend_kwargs)
        elif legend_pos == 4:
            ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
                      ncol=legend_cfg['ncol'], **legend_kwargs)
        else:
            legend_kwargs['frameon'] = False
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                      ncol=2, **legend_kwargs)

    def _draw_per_image_chart(self, show_psnr: bool, show_ssim: bool, show_lpips: bool):
        """Draw Metrics per Image chart (line charts showing per-image values)."""
        # Count selected metrics for subplots
        selected_metrics = []
        if show_psnr:
            selected_metrics.append(('psnr', 'PSNR (dB)', 'higher is better'))
        if show_ssim:
            selected_metrics.append(('ssim', 'SSIM', 'higher is better'))
        if show_lpips:
            selected_metrics.append(('lpips', 'LPIPS', 'lower is better'))

        n_metrics = len(selected_metrics)

        # Check if any test has per-image data
        has_per_image_data = any(
            test.get("quality_per_image") for test in self._tests
        )

        if not has_per_image_data:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "No per-image data available\n"
                   "Run batch test with Quality report enabled",
                   ha='center', va='center', fontsize=12, color='#999')
            ax.axis('off')
            return

        # Colors for tests
        colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336',
                  '#00BCD4', '#8BC34A', '#FFC107', '#673AB7', '#E91E63']

        for metric_idx, (metric_key, metric_label, direction) in enumerate(selected_metrics):
            ax = self.figure.add_subplot(n_metrics, 1, metric_idx + 1)

            noisy_key = f"{metric_key}_noisy"
            denoised_key = f"{metric_key}_denoised"

            for test_idx, test in enumerate(self._tests):
                test_name = test.get("name", f"Test {test_idx}")
                color = colors[test_idx % len(colors)]

                per_image = test.get("quality_per_image", {})
                noisy_values = per_image.get(noisy_key, [])
                denoised_values = per_image.get(denoised_key, [])

                if noisy_values:
                    x = np.arange(len(noisy_values))
                    ax.plot(x, noisy_values, '--', label=f'{test_name} (Noisy)',
                           color=color, alpha=0.6, linewidth=1)

                if denoised_values:
                    x = np.arange(len(denoised_values))
                    ax.plot(x, denoised_values, '-', label=f'{test_name} (Denoised)',
                           color=color, linewidth=1.5, marker='o', markersize=2)

            # Apply axes configuration (per subplot)
            axes_cfg = self._chart_config.get('axes', {})
            default_xlabel = 'Image Index' if metric_idx == n_metrics - 1 else ''
            self._apply_axes_config(
                ax,
                default_title=f'{metric_label} per Image ({direction})',
                default_xlabel=default_xlabel,
                default_ylabel=metric_label
            )

            # Set Y limits for specific metrics (if auto_scale)
            if axes_cfg.get('auto_scale', True):
                if metric_key == 'ssim':
                    ax.set_ylim(0, 1)
                elif metric_key == 'lpips':
                    ax.set_ylim(0, 1)

            ax.grid(True, alpha=0.3)

            # Apply legend with configuration
            legend_cfg = self._chart_config['legend']
            legend_pos = legend_cfg['position']
            # Use smaller font for per-image charts (many entries)
            per_image_fontsize = max(6, legend_cfg['fontsize'] - 2)
            legend_kwargs = {
                'fontsize': per_image_fontsize,
                'frameon': legend_cfg['frameon'],
                'shadow': legend_cfg['shadow'],
                'fancybox': legend_cfg['fancybox'],
                'framealpha': legend_cfg['framealpha'],
            }

            loc_map = {0: 'upper right', 1: 'upper left', 2: 'lower right', 3: 'lower left'}
            if legend_pos in loc_map:  # Inside
                ax.legend(loc=loc_map.get(legend_pos, 'best'), ncol=2, **legend_kwargs)
            elif legend_pos == 4:  # Right side
                ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), **legend_kwargs)
            else:  # Below
                legend_kwargs['frameon'] = False
                ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),
                         ncol=len(self._tests) * 2, **legend_kwargs)

    def _draw_histogram_chart(self, show_psnr: bool, show_ssim: bool, show_lpips: bool):
        """Draw Metrics Histogram chart (distribution of metric values)."""
        # Count selected metrics for subplots
        selected_metrics = []
        if show_psnr:
            selected_metrics.append(('psnr', 'PSNR (dB)', 'higher is better'))
        if show_ssim:
            selected_metrics.append(('ssim', 'SSIM', 'higher is better'))
        if show_lpips:
            selected_metrics.append(('lpips', 'LPIPS', 'lower is better'))

        n_metrics = len(selected_metrics)

        # Check if any test has per-image data
        has_per_image_data = any(
            test.get("quality_per_image") for test in self._tests
        )

        if not has_per_image_data:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "No per-image data available\n"
                   "Run batch test with Quality report enabled",
                   ha='center', va='center', fontsize=12, color='#999')
            ax.axis('off')
            return

        # Colors for noisy vs denoised from configuration
        colors_cfg = self._chart_config['colors']
        color_noisy = colors_cfg['noisy']
        color_denoised = colors_cfg['denoised']
        hist_alpha = colors_cfg['hist_alpha']

        for metric_idx, (metric_key, metric_label, direction) in enumerate(selected_metrics):
            ax = self.figure.add_subplot(n_metrics, 1, metric_idx + 1)

            noisy_key = f"{metric_key}_noisy"
            denoised_key = f"{metric_key}_denoised"

            # Collect all values across all tests
            all_noisy = []
            all_denoised = []

            for test in self._tests:
                per_image = test.get("quality_per_image", {})
                noisy_values = per_image.get(noisy_key, [])
                denoised_values = per_image.get(denoised_key, [])

                all_noisy.extend(noisy_values)
                all_denoised.extend(denoised_values)

            # Plot histograms
            data_to_plot = []
            labels = []
            colors = []

            if all_noisy:
                data_to_plot.append(all_noisy)
                labels.append('Noisy')
                colors.append(color_noisy)

            if all_denoised:
                data_to_plot.append(all_denoised)
                labels.append('Denoised')
                colors.append(color_denoised)

            if data_to_plot:
                ax.hist(data_to_plot, bins=20, label=labels, color=colors, alpha=hist_alpha)

            # Apply axes configuration (per subplot)
            self._apply_axes_config(
                ax,
                default_title=f'{metric_label} Distribution ({direction})',
                default_xlabel=metric_label,
                default_ylabel='Frequency'
            )

            # Set X limits for specific metrics
            if metric_key == 'ssim':
                ax.set_xlim(0, 1)
            elif metric_key == 'lpips':
                ax.set_xlim(0, 1)

            ax.grid(True, alpha=0.3)

            # Apply legend with configuration
            legend_cfg = self._chart_config['legend']
            legend_pos = legend_cfg['position']
            legend_kwargs = {
                'fontsize': legend_cfg['fontsize'],
                'frameon': legend_cfg['frameon'],
                'shadow': legend_cfg['shadow'],
                'fancybox': legend_cfg['fancybox'],
                'framealpha': legend_cfg['framealpha'],
            }

            loc_map = {0: 'upper right', 1: 'upper left', 2: 'lower right', 3: 'lower left'}
            if legend_pos in loc_map:  # Inside
                ax.legend(loc=loc_map.get(legend_pos, 'best'), ncol=legend_cfg['ncol'], **legend_kwargs)
            elif legend_pos == 4:  # Right side
                ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
                         ncol=legend_cfg['ncol'], **legend_kwargs)
            else:  # Below
                legend_kwargs['frameon'] = False
                ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),
                         ncol=2, **legend_kwargs)

    def _get_nested_value(self, data: dict, key: str):
        """Get a value from a nested dictionary using dot notation."""
        keys = key.split(".")
        value = data
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return None
        return value

    def clear(self):
        """Clear all data from the view."""
        self._tests = []
        self.figure.clear()
        self.canvas.draw()
        self.info_label.setText("Load experiments to see quality comparison charts")
        self.preview_btn.setEnabled(False)
