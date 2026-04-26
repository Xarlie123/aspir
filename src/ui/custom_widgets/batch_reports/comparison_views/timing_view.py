"""
Timing view for Batch Reports - displays timing metrics with Single Test style layout.
"""
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFileDialog, QMessageBox, QGroupBox, QGridLayout,
    QSplitter, QListWidget, QDialog, QDialogButtonBox, QMenu, QComboBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from ui.custom_widgets.batch_reports.comparison_views.chart_config_popup import (
    ChartConfigPopup, CustomNavigationToolbar
)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False


class TimingView(QWidget):
    """
    Timing view displaying timing results comparison.

    Features:
    - Left menu with chart type selection (QListWidget)
    - Shows all tests from Summary selection
    - Stacked bar chart comparing CPU vs GPU pipeline latency
    - Summary table with timing breakdown
    - Detailed Timing Report button
    - Launch profile with Nsight button
    - Navigation toolbar with chart configuration
    - Export charts to PNG/PDF
    """

    # Chart type indices — must stay in sync with the order of items
    # added to ``self.chart_list`` in ``_setup_ui``.
    CHART_PIPELINE = 0
    CHART_ENERGY_VS_RATIO = 1

    # Color palette for multiple tests
    COLORS = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336',
              '#00BCD4', '#8BC34A', '#FFC107', '#673AB7', '#E91E63']

    # Pipeline colors (matching Single Test)
    COLOR_ACQUISITION = '#abdda4'   # Green
    COLOR_RECONSTRUCTION = '#fdae61'  # Orange
    COLOR_INFERENCE_CPU = '#d7191c'   # Red
    COLOR_INFERENCE_GPU = '#2b83ba'   # Blue

    # Energy-vs-Sampling-Ratio defaults — same Tableau pair as the
    # PSNR-vs-Sampling-Ratio chart in Quality so the two figures of the
    # paper share a colour vocabulary (blue first, orange second).
    _ENERGY_CPU_COLOR = '#1f77b4'
    _ENERGY_GPU_COLOR = '#ff7f0e'

    def __init__(self, logger=None, parent=None):
        super().__init__(parent)

        if logger:
            self.logger = logger.getChild("TimingView")
        else:
            self.logger = logging.getLogger("TimingView")

        self._tests: List[Dict[str, Any]] = []

        # Chart configuration with defaults
        self._chart_config = {
            'axes': {
                'title': '',
                'title_fontsize': 13,
                'xlabel': '',
                'xlabel_fontsize': 11,
                # The Pipeline Latency Breakdown chart draws a secondary
                # row of "4 % / 8 % / …" group labels at axes-fraction
                # y=-0.12 (see ``_draw_pipeline_breakdown``). With the
                # generic 4 pt default the X axis title overlaps that
                # row; ~30 pt is enough to drop the title below it.
                # Users can still override from the chart-config dialog
                # (range 0–120 pt).
                'xlabel_pad': 30,
                'xtick_fontsize': 9,
                'group_label_fontsize': 9,
                'ylabel': '',
                'ylabel_fontsize': 11,
                'ylabel_pad': 8,
                'ytick_fontsize': 9,
                'bar_label_fontsize': 8,
                'auto_scale': True,
                'ymin': 0.0,
                'ymax': 100.0,
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
                'acquisition': self.COLOR_ACQUISITION,
                'reconstruction': self.COLOR_RECONSTRUCTION,
                'inference_cpu': self.COLOR_INFERENCE_CPU,
                'inference_gpu': self.COLOR_INFERENCE_GPU,
                'bar_alpha': 0.8,
            }
        }

        self._setup_ui()

    def _setup_ui(self):
        """Setup the timing view UI with left menu."""
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)

        # Splitter for menu and content area
        splitter = QSplitter(Qt.Horizontal)

        # Left panel: Menu and buttons
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(5, 5, 5, 5)
        left_layout.setSpacing(10)

        # Chart type list
        chart_label = QLabel("Chart Type:")
        chart_label.setStyleSheet("font-weight: bold; color: #333;")
        left_layout.addWidget(chart_label)

        self.chart_list = QListWidget()
        self.chart_list.setMaximumWidth(220)
        self.chart_list.setMinimumWidth(180)
        self.chart_list.addItem("Pipeline Latency Breakdown")
        self.chart_list.addItem("Energy per Image vs Sampling Ratio")
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

        # Detailed Timing Report button
        self.detailed_report_btn = QPushButton("Detailed Timing Report")
        self.detailed_report_btn.setMaximumWidth(220)
        self.detailed_report_btn.setMinimumWidth(180)
        self.detailed_report_btn.setMinimumHeight(36)
        self.detailed_report_btn.setEnabled(False)
        self.detailed_report_btn.setStyleSheet("""
            QPushButton {
                background-color: #0078d7;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
                font-size: 12px;
                padding: 8px;
            }
            QPushButton:hover:enabled {
                background-color: #005a9e;
            }
            QPushButton:pressed:enabled {
                background-color: #004275;
            }
            QPushButton:disabled {
                background-color: #ccc;
                color: #888;
            }
        """)
        self.detailed_report_btn.setToolTip("View detailed timing report with statistics and PyTorch profiler data")
        self.detailed_report_btn.clicked.connect(self._on_detailed_report_clicked)
        left_layout.addWidget(self.detailed_report_btn)

        # Launch profile with Nsight button
        self.nsight_btn = QPushButton("Launch profile with Nsight")
        self.nsight_btn.setMaximumWidth(220)
        self.nsight_btn.setMinimumWidth(180)
        self.nsight_btn.setMinimumHeight(36)
        self.nsight_btn.setEnabled(False)
        self.nsight_btn.setStyleSheet("""
            QPushButton {
                background-color: #76B900;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
                font-size: 12px;
                padding: 8px;
            }
            QPushButton:hover:enabled {
                background-color: #5A8F00;
            }
            QPushButton:pressed:enabled {
                background-color: #4A7500;
            }
            QPushButton:disabled {
                background-color: #ccc;
                color: #888;
            }
        """)
        self.nsight_btn.setToolTip(
            "Launch NVIDIA Nsight Systems to analyze CPU↔GPU performance"
        )
        self.nsight_btn.clicked.connect(self._on_nsight_clicked)
        left_layout.addWidget(self.nsight_btn)

        left_layout.addStretch()

        splitter.addWidget(left_panel)

        # Right panel: Results area
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(5, 5, 5, 5)
        right_layout.setSpacing(5)

        # Chart area
        self.figure = Figure(figsize=(10, 5), dpi=100)
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
        right_layout.addWidget(self.canvas, 1)

        # Timing Summary table
        self.summary_group = QGroupBox("Timing Summary")
        summary_layout = QGridLayout(self.summary_group)
        summary_layout.setSpacing(10)

        # Test selector for summary table
        selector_layout = QHBoxLayout()
        test_label = QLabel("Show details for:")
        test_label.setStyleSheet("font-weight: bold;")
        selector_layout.addWidget(test_label)

        self.test_combo = QComboBox()
        self.test_combo.setMinimumWidth(200)
        self.test_combo.currentIndexChanged.connect(self._on_test_changed)
        selector_layout.addWidget(self.test_combo)
        selector_layout.addStretch()
        summary_layout.addLayout(selector_layout, 0, 0, 1, 4)

        # Headers
        header_font = QFont()
        header_font.setBold(True)

        summary_layout.addWidget(QLabel(""), 1, 0)

        cpu_header = QLabel("CPU")
        cpu_header.setFont(header_font)
        cpu_header.setAlignment(Qt.AlignCenter)
        summary_layout.addWidget(cpu_header, 1, 1)

        gpu_header = QLabel("GPU")
        gpu_header.setFont(header_font)
        gpu_header.setAlignment(Qt.AlignCenter)
        summary_layout.addWidget(gpu_header, 1, 2)

        speedup_header = QLabel("Speedup")
        speedup_header.setFont(header_font)
        speedup_header.setAlignment(Qt.AlignCenter)
        summary_layout.addWidget(speedup_header, 1, 3)

        # Create summary labels
        self._summary_labels = {}
        rows = [
            ("T_mask_projection:", "t_acq"),
            ("T_reconstruction:", "t_recon"),
            ("T_inference:", "t_inf"),
            ("T_total:", "t_total"),
        ]

        for row_idx, (label_text, key) in enumerate(rows, start=2):
            row_label = QLabel(label_text)
            row_label.setFont(header_font)
            summary_layout.addWidget(row_label, row_idx, 0)

            cpu_label = QLabel("-")
            cpu_label.setAlignment(Qt.AlignCenter)
            summary_layout.addWidget(cpu_label, row_idx, 1)
            self._summary_labels[f"{key}_cpu"] = cpu_label

            gpu_label = QLabel("-")
            gpu_label.setAlignment(Qt.AlignCenter)
            summary_layout.addWidget(gpu_label, row_idx, 2)
            self._summary_labels[f"{key}_gpu"] = gpu_label

            speedup_label = QLabel("-")
            speedup_label.setAlignment(Qt.AlignCenter)
            summary_layout.addWidget(speedup_label, row_idx, 3)
            self._summary_labels[f"{key}_speedup"] = speedup_label

        # Style for total row
        self._summary_labels["t_total_cpu"].setStyleSheet("font-weight: bold;")
        self._summary_labels["t_total_gpu"].setStyleSheet("font-weight: bold;")
        self._summary_labels["t_total_speedup"].setStyleSheet(
            "font-weight: bold; color: #4CAF50; font-size: 14px;"
        )
        self._summary_labels["t_inf_speedup"].setStyleSheet(
            "font-weight: bold; color: #0078d7;"
        )

        right_layout.addWidget(self.summary_group)

        # Info label
        self.info_label = QLabel("Load experiments to see timing analysis")
        self.info_label.setStyleSheet("color: #666; font-size: 11px;")
        self.info_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.info_label)

        splitter.addWidget(right_panel)

        # Set splitter sizes (menu:content = 1:4)
        splitter.setSizes([180, 720])

        main_layout.addWidget(splitter)

    def set_tests(self, tests: List[Dict[str, Any]]):
        """
        Set the tests to display.

        Args:
            tests: List of test dictionaries with timing metrics
        """
        self._tests = tests

        # Update test combo for summary table
        self.test_combo.clear()
        for test in tests:
            test_name = test.get("name", "Unknown")
            exp_name = test.get("_experiment_name", "")
            if exp_name:
                self.test_combo.addItem(f"{test_name} ({exp_name})")
            else:
                self.test_combo.addItem(test_name)

        # Enable buttons
        has_tests = len(tests) > 0
        self.detailed_report_btn.setEnabled(has_tests)
        self.nsight_btn.setEnabled(has_tests and self._check_nsight_available())

        self._refresh_display()

    def _check_nsight_available(self) -> bool:
        """Check if nsys command is available."""
        import shutil
        return shutil.which("nsys") is not None

    def _on_test_changed(self, index: int):
        """Handle test selection change for summary table."""
        self._update_summary_table()

    def _on_chart_type_changed(self, index: int):
        """Handle chart type selection change."""
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
        """Apply axes configuration to a matplotlib axis."""
        axes_cfg = self._chart_config.get('axes', {})

        title = axes_cfg.get('title', '') or default_title
        title_fontsize = axes_cfg.get('title_fontsize', 13)
        if title:
            ax.set_title(title, fontsize=title_fontsize, fontweight='bold')

        xlabel = axes_cfg.get('xlabel', '') or default_xlabel
        xlabel_fontsize = axes_cfg.get('xlabel_fontsize', 11)
        xlabel_pad = axes_cfg.get('xlabel_pad', 4)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=xlabel_fontsize, labelpad=xlabel_pad)

        ylabel = axes_cfg.get('ylabel', '') or default_ylabel
        ylabel_fontsize = axes_cfg.get('ylabel_fontsize', 11)
        ylabel_pad = axes_cfg.get('ylabel_pad', 4)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=ylabel_fontsize, labelpad=ylabel_pad)

        xtick_fontsize = axes_cfg.get('xtick_fontsize', 9)
        ytick_fontsize = axes_cfg.get('ytick_fontsize', 9)
        ax.tick_params(axis='x', labelsize=xtick_fontsize)
        ax.tick_params(axis='y', labelsize=ytick_fontsize)

        if not axes_cfg.get('auto_scale', True):
            ymin = axes_cfg.get('ymin', 0)
            ymax = axes_cfg.get('ymax', 100)
            ax.set_ylim(ymin, ymax)

    def _refresh_display(self):
        """Refresh the entire display."""
        self._update_summary_table()
        self._refresh_chart()

        # Update info
        if self._tests:
            experiment_count = len(set(t.get("_experiment_name", "") for t in self._tests))
            self.info_label.setText(
                f"Showing {len(self._tests)} tests from {experiment_count} experiment(s)"
            )
        else:
            self.info_label.setText("Load experiments to see timing analysis")

    def _update_summary_table(self):
        """Update the timing summary table for the selected test."""
        current_idx = self.test_combo.currentIndex()

        if not self._tests or current_idx < 0 or current_idx >= len(self._tests):
            for label in self._summary_labels.values():
                label.setText("-")
            return

        test = self._tests[current_idx]

        # Get timing values from test data (new structure)
        t_acq = test.get("timing_acquisition_ms", 0) or 0
        t_recon = test.get("timing_reconstruction_ms", 0) or 0

        # CPU timing - from timing_cpu_mean_ms (new) or timing_mean_ms (old if not use_gpu)
        t_inf_cpu = test.get("timing_cpu_mean_ms", 0) or 0
        if t_inf_cpu == 0:
            # Fallback for old data format
            use_gpu = test.get("use_gpu", False)
            if not use_gpu:
                t_inf_cpu = test.get("timing_mean_ms", 0) or 0

        # GPU timing - from timing_gpu_mean_ms (new)
        t_inf_gpu = test.get("timing_gpu_mean_ms", 0) or 0

        # Calculate totals
        t_total_cpu = t_acq + t_recon + t_inf_cpu if t_inf_cpu > 0 else 0
        t_total_gpu = t_acq + t_recon + t_inf_gpu if t_inf_gpu > 0 else 0

        # Update T_acquisition (same for CPU and GPU, only show GPU if GPU timing exists)
        self._summary_labels["t_acq_cpu"].setText(f"{t_acq:.2f} ms" if t_acq > 0 else "-")
        self._summary_labels["t_acq_gpu"].setText(f"{t_acq:.2f} ms" if t_inf_gpu > 0 else "-")
        self._summary_labels["t_acq_speedup"].setText("1.00x" if t_acq > 0 and t_inf_gpu > 0 else "-")

        # Update T_reconstruction (same for both - runs on CPU)
        self._summary_labels["t_recon_cpu"].setText(f"{t_recon:.2f} ms" if t_recon > 0 else "-")
        self._summary_labels["t_recon_gpu"].setText(f"{t_recon:.2f} ms" if t_inf_gpu > 0 else "-")
        self._summary_labels["t_recon_speedup"].setText("1.00x" if t_recon > 0 and t_inf_gpu > 0 else "-")

        # Update T_inference
        self._summary_labels["t_inf_cpu"].setText(f"{t_inf_cpu:.2f} ms" if t_inf_cpu > 0 else "-")
        self._summary_labels["t_inf_gpu"].setText(f"{t_inf_gpu:.2f} ms" if t_inf_gpu > 0 else "-")
        if t_inf_cpu > 0 and t_inf_gpu > 0:
            inf_speedup = t_inf_cpu / t_inf_gpu
            self._summary_labels["t_inf_speedup"].setText(f"{inf_speedup:.2f}x")
        else:
            self._summary_labels["t_inf_speedup"].setText("-")

        # Update T_total
        self._summary_labels["t_total_cpu"].setText(f"{t_total_cpu:.2f} ms" if t_total_cpu > 0 else "-")
        self._summary_labels["t_total_gpu"].setText(f"{t_total_gpu:.2f} ms" if t_total_gpu > 0 else "-")
        if t_total_cpu > 0 and t_total_gpu > 0:
            total_speedup = t_total_cpu / t_total_gpu
            self._summary_labels["t_total_speedup"].setText(f"{total_speedup:.2f}x")
        else:
            self._summary_labels["t_total_speedup"].setText("-")

    def _refresh_chart(self):
        """Refresh the chart based on current settings."""
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

        if self.chart_list.currentRow() == self.CHART_ENERGY_VS_RATIO:
            self._draw_energy_vs_sampling_chart()
        else:
            self._draw_pipeline_breakdown()

        # Adjust layout based on legend position
        legend_pos = self._chart_config['legend']['position']
        if legend_pos == 4:  # Right side (outside)
            self.figure.tight_layout(rect=[0, 0.05, 0.85, 1])
        elif legend_pos == 5:  # Below (outside)
            self.figure.tight_layout(rect=[0, 0.15, 1, 1])
        else:  # Inside positions
            self.figure.tight_layout(rect=[0, 0.05, 1, 1])

        self.canvas.draw()

    # ------------------------------------------------------------------
    # Helpers shared by the Energy-vs-Sampling-Ratio chart
    # ------------------------------------------------------------------

    def _n_pixels_for_test(self, test: Dict[str, Any]) -> Optional[int]:
        """Resolve N = img_size² for a test from its experiment metadata."""
        meta = test.get("_experiment_metadata") or {}
        if isinstance(meta, dict):
            ds = meta.get("dataset_info", {}) or {}
            img_size = ds.get("img_size")
            if img_size:
                try:
                    s = int(img_size)
                    return s * s
                except (TypeError, ValueError):
                    return None
        return None

    def _n_patterns_for_test(self, test: Dict[str, Any]) -> Optional[int]:
        """Resolve M (number of mask patterns) for a test."""
        n = test.get("timing_num_patterns")
        if n is not None:
            try:
                return int(n)
            except (TypeError, ValueError):
                return None
        mask_type = (test.get("mask_type") or "").lower()
        if mask_type == "scatter":
            return test.get("scatter_num_patterns")
        if mask_type.startswith("hadamard") or mask_type == "cal_sal":
            lo = test.get("hadamard_min_idx")
            hi = test.get("hadamard_max_idx")
            if lo is not None and hi is not None:
                return max(0, int(hi) - int(lo))
        return None

    # ------------------------------------------------------------------
    # Energy per image vs Sampling Ratio (paper-style figure)
    # ------------------------------------------------------------------

    def _draw_energy_vs_sampling_chart(self):
        """Plot energy per image (mJ) vs M/N (%) — CPU run vs GPU run.

        Companion to the PSNR-vs-Sampling-Ratio chart in Quality. Each
        loaded test contributes one point at its sampling ratio, and we
        split the points into two series by the test's ``use_gpu``
        flag — so the user gets a CPU-pipeline curve and a GPU-pipeline
        curve from one combined load. ±1σ shading uses
        ``energy_std_mj`` from the report.

        Data sources per test:
            ``energy_mean_mj`` — total per-image energy (always present
            when energy was measured).
            ``energy_std_mj``  — standard deviation across the
            measurement runs; absent → no shading for that point.
            ``timing_num_patterns`` and ``_experiment_metadata`` for M
            and N, same path used by the Quality view's Sampling-Ratio
            chart.

        On Jetson the rail is shared so the CPU and GPU lines reflect a
        single physical reading (see ``JtopEnergyBackend`` notes); the
        meaningful comparison there is "CPU run vs GPU run", which is
        exactly what this chart plots — the difference is in
        *which compute path runs* during the measurement, not in what
        the sensor sees afterwards.
        """
        ax = self.figure.add_subplot(111)

        rows: List[Dict[str, Any]] = []
        for test in self._tests:
            m = self._n_patterns_for_test(test)
            n = self._n_pixels_for_test(test)
            if not m or not n:
                continue
            energy = test.get("energy_mean_mj")
            if energy is None:
                continue
            try:
                energy = float(energy)
            except (TypeError, ValueError):
                continue
            std = test.get("energy_std_mj")
            try:
                std = float(std) if std is not None else None
            except (TypeError, ValueError):
                std = None
            rows.append({
                "experiment": test.get("_experiment_name", ""),
                "name": test.get("name", ""),
                "beta": 100.0 * float(m) / float(n),
                "energy_mj": energy,
                "energy_std_mj": std,
                "use_gpu": bool(test.get("use_gpu", False)),
            })

        if not rows:
            ax.text(
                0.5, 0.5,
                "No tests with both energy and M/N metadata.\n"
                "Run a batch with the energy report enabled, or use\n"
                "Re-measure on a saved batch.",
                ha='center', va='center', fontsize=12, color='#999',
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            return

        cpu_rows = sorted([r for r in rows if not r["use_gpu"]],
                          key=lambda r: r["beta"])
        gpu_rows = sorted([r for r in rows if r["use_gpu"]],
                          key=lambda r: r["beta"])

        def _series(rows_for_series, label, color, marker):
            if not rows_for_series:
                return None, None
            xs = [r["beta"] for r in rows_for_series]
            ys = [r["energy_mj"] for r in rows_for_series]
            ss = [r["energy_std_mj"] for r in rows_for_series]
            ax.plot(
                xs, ys,
                marker=marker, linestyle='-', linewidth=2.0,
                color=color, label=label, markersize=8,
                markerfacecolor=color, markeredgecolor='white',
                markeredgewidth=1.0,
            )
            if all(s is not None for s in ss):
                lo = [y - s for y, s in zip(ys, ss)]
                hi = [y + s for y, s in zip(ys, ss)]
                ax.fill_between(xs, lo, hi, color=color, alpha=0.15)
            return ys, ss

        cpu_ys, _ = _series(cpu_rows, "CPU run",
                            self._ENERGY_CPU_COLOR, '^')
        gpu_ys, _ = _series(gpu_rows, "GPU run",
                            self._ENERGY_GPU_COLOR, 'D')

        # If one series is way off scale relative to the other, switch
        # to a log Y axis so both shapes stay legible. Threshold is the
        # 5× rule in the spec — applied conservatively (only when *both*
        # series have data, otherwise log on a single curve isn't useful).
        if cpu_ys and gpu_ys:
            cpu_max = max(cpu_ys) if cpu_ys else 0.0
            gpu_max = max(gpu_ys) if gpu_ys else 0.0
            cpu_min = min(cpu_ys) if cpu_ys else 0.0
            gpu_min = min(gpu_ys) if gpu_ys else 0.0
            big = max(cpu_max, gpu_max)
            small = min(cpu_min, gpu_min)
            if small > 0 and big / small > 5.0:
                ax.set_yscale('log')

        self._apply_axes_config(
            ax,
            default_title="Energy per Image vs Sampling Ratio",
            default_xlabel="Sampling ratio M/N (%)",
            default_ylabel="Energy per image (mJ)",
        )
        ax.grid(True, alpha=0.3, linestyle=':')

        # Place the legend opposite the dominant data cluster — usually
        # the CPU curve goes upward with M/N and dominates the upper
        # range, so the upper-left corner stays free; otherwise let the
        # global config decide.
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

    def _draw_pipeline_breakdown(self):
        """Draw stacked bar chart of pipeline latency breakdown for all tests.

        Shows CPU and GPU as separate positions on the X-axis (like Single Test),
        with clear labels for each bar.
        """
        ax = self.figure.add_subplot(111)

        if not self._tests:
            return

        # Read user-tunable font sizes from the shared chart config so the
        # ``Configure chart settings`` dialog actually reaches every label.
        axes_cfg = self._chart_config.get('axes', {})
        bar_label_fs = axes_cfg.get('bar_label_fontsize', 8)
        group_label_fs = axes_cfg.get('group_label_fontsize', 9)
        xtick_fs = axes_cfg.get('xtick_fontsize', 9)

        # Collect data for all tests
        test_names = []
        cpu_data = []  # List of (t_acq, t_recon, t_inf_cpu)
        gpu_data = []  # List of (t_acq, t_recon, t_inf_gpu) or None
        has_any_gpu = False

        for test in self._tests:
            test_name = test.get("name", "Unknown")
            if len(test_name) > 15:
                test_name = test_name[:12] + "..."
            test_names.append(test_name)

            t_acq = test.get("timing_acquisition_ms", 0) or 0
            t_recon = test.get("timing_reconstruction_ms", 0) or 0

            # CPU timing - from timing_cpu_mean_ms (new) or timing_mean_ms (old if not use_gpu)
            t_inf_cpu = test.get("timing_cpu_mean_ms", 0) or 0
            if t_inf_cpu == 0:
                # Fallback for old data format
                use_gpu = test.get("use_gpu", False)
                if not use_gpu:
                    t_inf_cpu = test.get("timing_mean_ms", 0) or 0

            # GPU timing - from timing_gpu_mean_ms (new)
            t_inf_gpu = test.get("timing_gpu_mean_ms", 0) or 0

            if t_inf_gpu > 0:
                gpu_data.append((t_acq, t_recon, t_inf_gpu))
                has_any_gpu = True
            else:
                gpu_data.append(None)

            cpu_data.append((t_acq, t_recon, t_inf_cpu))

        n_tests = len(test_names)

        # If any test has GPU, show CPU and GPU as separate X positions (like Single Test)
        if has_any_gpu:
            # Create separate X positions for CPU and GPU for each test
            # Layout: [Test1_CPU, Test1_GPU, gap, Test2_CPU, Test2_GPU, ...]
            x_positions = []
            x_labels = []
            bar_colors_inf = []  # Track inference color for each bar
            bar_data = []  # (t_acq, t_recon, t_inf)

            pos = 0.0  # X-axis position (can be float for gaps)
            bar_idx = 0  # Integer index into x_positions list
            group_positions = []  # For grouping labels: (start_idx, end_idx, test_name)

            for i, test_name in enumerate(test_names):
                group_start_idx = bar_idx

                # CPU bar
                x_positions.append(pos)
                x_labels.append("CPU")
                bar_data.append(cpu_data[i])
                bar_colors_inf.append(self.COLOR_INFERENCE_CPU)
                pos += 1
                bar_idx += 1

                # GPU bar (only if this test has GPU data)
                if gpu_data[i] is not None:
                    x_positions.append(pos)
                    x_labels.append("GPU")
                    bar_data.append(gpu_data[i])
                    bar_colors_inf.append(self.COLOR_INFERENCE_GPU)
                    pos += 1
                    bar_idx += 1

                group_positions.append((group_start_idx, bar_idx - 1, test_name))
                pos += 0.5  # Gap between tests

            x = np.array(x_positions)
            width = 0.7

            # Draw stacked bars
            acq = [d[0] for d in bar_data]
            recon = [d[1] for d in bar_data]
            inf = [d[2] for d in bar_data]

            # Acquisition and Reconstruction (same color for all)
            ax.bar(x, acq, width, label='Mask projection', color=self.COLOR_ACQUISITION, edgecolor='white')
            ax.bar(x, recon, width, bottom=acq, label='Reconstruction', color=self.COLOR_RECONSTRUCTION, edgecolor='white')

            # Inference - different colors for CPU vs GPU
            # Draw CPU inference bars
            cpu_mask = [c == self.COLOR_INFERENCE_CPU for c in bar_colors_inf]
            gpu_mask = [c == self.COLOR_INFERENCE_GPU for c in bar_colors_inf]

            for idx in range(len(x)):
                bottom = acq[idx] + recon[idx]
                if cpu_mask[idx]:
                    label = 'Inference (CPU)' if idx == 0 else None
                    ax.bar(x[idx], inf[idx], width, bottom=bottom,
                           label=label, color=self.COLOR_INFERENCE_CPU, edgecolor='white')
                else:
                    # Find first GPU bar for label
                    first_gpu = next((j for j, m in enumerate(gpu_mask) if m), -1)
                    label = 'Inference (GPU)' if idx == first_gpu else None
                    ax.bar(x[idx], inf[idx], width, bottom=bottom,
                           label=label, color=self.COLOR_INFERENCE_GPU, edgecolor='white')

            # Add total time labels on top of bars
            for idx in range(len(x)):
                total = acq[idx] + recon[idx] + inf[idx]
                if total > 0:
                    ax.text(x[idx], total + 0.3, f'{total:.1f}',
                           ha='center', va='bottom',
                           fontsize=bar_label_fs, fontweight='bold')

            # Set X-axis labels (CPU/GPU)
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels, fontsize=xtick_fs)

            # Add test name labels below the CPU/GPU labels
            for group_start, group_end, test_name in group_positions:
                group_center = (x_positions[group_start] + x_positions[group_end]) / 2
                # Use axes fraction for y (-0.12 places it below the tick labels)
                ax.text(group_center, -0.12, test_name,
                       ha='center', va='top',
                       fontsize=group_label_fs, fontweight='bold',
                       transform=ax.get_xaxis_transform())

            # Add extra bottom margin for test names
            self.figure.subplots_adjust(bottom=0.22)

        else:
            # Only CPU bars - one per test (with "CPU" label under each)
            x = np.arange(n_tests)
            width = 0.6

            acq = [d[0] for d in cpu_data]
            recon = [d[1] for d in cpu_data]
            inf = [d[2] for d in cpu_data]

            ax.bar(x, acq, width, label='Mask projection', color=self.COLOR_ACQUISITION, edgecolor='white')
            ax.bar(x, recon, width, bottom=acq, label='Reconstruction', color=self.COLOR_RECONSTRUCTION, edgecolor='white')
            ax.bar(x, inf, width, bottom=np.array(acq) + np.array(recon),
                   label='Inference (CPU)', color=self.COLOR_INFERENCE_CPU, edgecolor='white')

            # Add total time labels
            for i, cd in enumerate(cpu_data):
                total = cd[0] + cd[1] + cd[2]
                if total > 0:
                    ax.text(x[i], total + 0.5, f'{total:.1f}',
                           ha='center', va='bottom',
                           fontsize=bar_label_fs, fontweight='bold')

            # Two-line labels: test name + "CPU"
            labels = [f"{name}\nCPU" for name in test_names]
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=xtick_fs)

        # Apply axes configuration. The default_xlabel is set to the
        # paper convention because the X axis shows the sampling ratio
        # tier ("4 % / 8 % / …") — without a title the chart looks
        # untidy in screenshots.
        self._apply_axes_config(
            ax,
            default_title="Pipeline Latency Breakdown",
            default_xlabel="Sampling ratio M/N (%)",
            default_ylabel="Time (ms)",
        )

        ax.grid(axis='y', alpha=0.3)

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

        loc_map = {
            0: 'upper right',
            1: 'upper left',
            2: 'lower right',
            3: 'lower left',
        }

        if legend_pos in loc_map:
            ax.legend(loc=loc_map[legend_pos], ncol=legend_cfg['ncol'], **legend_kwargs)
        elif legend_pos == 4:  # Right side
            ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
                     ncol=legend_cfg['ncol'], **legend_kwargs)
        else:  # Below
            legend_kwargs['frameon'] = False
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                     ncol=4, **legend_kwargs)

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
            "timing_chart.png",
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

    def _on_detailed_report_clicked(self):
        """Handle detailed timing report button click."""
        if not self._tests:
            return

        from ui.custom_widgets.batch_reports.comparison_views.timing_report_popup import (
            BatchTimingReportPopup
        )

        current_idx = max(0, self.test_combo.currentIndex())
        popup = BatchTimingReportPopup(
            tests=self._tests,
            current_test_idx=current_idx,
            parent=self,
            logger=self.logger
        )
        popup.exec_()

    def _on_nsight_clicked(self):
        """Handle Nsight launch button click."""
        if not self._tests:
            return

        # Show test selector dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Launch Nsight Systems")
        dialog.setMinimumWidth(450)

        layout = QVBoxLayout(dialog)

        # Test selection
        group = QGroupBox("Select Test to Profile")
        group_layout = QVBoxLayout(group)

        label = QLabel("Choose which test's model to profile with Nsight:")
        group_layout.addWidget(label)

        test_combo = QComboBox()
        for test in self._tests:
            test_name = test.get("name", "Unknown")
            model_name = test.get("model_name", test.get("config", {}).get("model_name", "Unknown"))
            test_combo.addItem(f"{test_name} ({model_name})")
        group_layout.addWidget(test_combo)

        # Info about Nsight
        info_label = QLabel(
            "<i>This will launch NVIDIA Nsight Systems profiler to analyze "
            "CPU↔GPU performance including memory transfers and kernel timings.</i>"
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #666; margin-top: 10px;")
        group_layout.addWidget(info_label)

        layout.addWidget(group)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec_() != QDialog.Accepted:
            return

        selected_idx = test_combo.currentIndex()
        selected_test = self._tests[selected_idx]

        self._launch_nsight_for_test(selected_test)

    def _launch_nsight_for_test(self, test: Dict[str, Any]):
        """Launch Nsight Systems for the selected test."""
        try:
            from ui.custom_widgets.timing_analysis.nsight_profiler_popup import NsightProfilerPopup
        except ImportError as e:
            QMessageBox.warning(
                self,
                "Import Error",
                f"Could not import NsightProfilerPopup:\n{e}"
            )
            return

        # Check if we have a saved model for this test
        experiment_path = test.get("_experiment_path")
        test_name = test.get("name", "Unknown")

        model_path = None
        if experiment_path:
            report_path = Path(experiment_path)
            batch_dir = report_path.parent
            safe_name = test_name.replace(" ", "_").replace("/", "-")

            # Look for saved model
            possible_paths = [
                batch_dir / "models" / f"{safe_name}.pt",
                batch_dir / "data" / safe_name / "model.pt",
                batch_dir / "data" / safe_name / "model.pth",
            ]
            for p in possible_paths:
                if p.exists():
                    model_path = p
                    break

        if not model_path:
            QMessageBox.warning(
                self,
                "Model Not Found",
                f"Could not find saved model for test '{test_name}'.\n\n"
                "The model needs to be exported during batch test execution.\n"
                "Make sure to use 'Reports and Models' or 'All Data' export level."
            )
            return

        QMessageBox.information(
            self,
            "Launch Nsight",
            f"Model found at:\n{model_path}\n\n"
            "Nsight Systems profiling will be launched.\n"
            "This feature requires nsys to be installed."
        )

        self.logger.info("Nsight profiling requested for test: %s", test_name)

    def clear(self):
        """Clear all data from the view."""
        self._tests = []
        self.test_combo.clear()
        self.figure.clear()
        self.canvas.draw()

        for label in self._summary_labels.values():
            label.setText("-")

        self.info_label.setText("Load experiments to see timing analysis")
        self.detailed_report_btn.setEnabled(False)
        self.nsight_btn.setEnabled(False)
