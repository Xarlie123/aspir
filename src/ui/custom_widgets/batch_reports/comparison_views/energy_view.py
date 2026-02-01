"""
Energy view for Batch Reports - displays energy metrics charts.
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
    QPushButton, QFileDialog, QMessageBox, QGroupBox, QGridLayout,
    QSplitter, QListWidget, QMenu, QDialog, QComboBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from ui.custom_widgets.batch_reports.comparison_views.chart_config_popup import (
    ChartConfigPopup, CustomNavigationToolbar
)


def _shorten_backend_name(name: str, max_length: int = 20) -> str:
    """Shorten a backend name for display."""
    if " with " in name:
        name = name.split(" with ")[0]
    if len(name) > max_length:
        return name[:max_length - 3] + "..."
    return name


class EnergyView(QWidget):
    """
    Energy view displaying energy consumption comparison charts.

    Features:
    - Left menu with chart type selection (QListWidget)
    - Backend selector (CPU, GPU, CPU+GPU with separate bars)
    - Shows all tests from Summary selection
    - Bar chart comparing energy consumption
    - Power vs efficiency scatter
    - Energy Summary table with columns per backend (like Single Test)
    - Navigation toolbar with chart configuration
    - Generate Energy Report button
    """

    # Color palette for experiments
    COLORS = ['#FF5722', '#E91E63', '#9C27B0', '#673AB7', '#3F51B5',
              '#2196F3', '#00BCD4', '#009688', '#4CAF50', '#8BC34A']

    # Backend-specific colors
    COLOR_GPU = '#FF9800'  # Orange for GPU
    COLOR_CPU = '#2196F3'  # Blue for CPU

    # Backend filter options
    BACKEND_ALL = "CPU + GPU"
    BACKEND_CPU = "CPU Only"
    BACKEND_GPU = "GPU Only"

    def __init__(self, logger=None, parent=None):
        super().__init__(parent)

        if logger:
            self.logger = logger.getChild("EnergyView")
        else:
            self.logger = logging.getLogger("EnergyView")

        self._tests: List[Dict[str, Any]] = []
        self._backend_filter = self.BACKEND_ALL

        # Chart configuration with defaults
        self._chart_config = {
            'axes': {
                'title': '',
                'title_fontsize': 13,
                'xlabel': '',
                'xlabel_fontsize': 11,
                'xtick_fontsize': 9,
                'ylabel': '',
                'ylabel_fontsize': 11,
                'ytick_fontsize': 9,
                'auto_scale': True,
                'ymin': 0.0,
                'ymax': 100.0,
            },
            'legend': {
                'position': 0,
                'fontsize': 9,
                'frameon': True,
                'shadow': False,
                'fancybox': True,
                'framealpha': 0.8,
                'ncol': 1,
            },
            'colors': {
                'bar_alpha': 0.8,
            }
        }

        self._setup_ui()

    def _setup_ui(self):
        """Setup the energy view UI with left menu."""
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)

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
        self.chart_list.addItems([
            "Energy Bar Chart",
            "Power Comparison",
            "Efficiency (img/J)",
            "Distribution (Box Plot)"
        ])
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

        # Backend selector
        backend_label = QLabel("Backend:")
        backend_label.setStyleSheet("font-weight: bold; color: #333;")
        left_layout.addWidget(backend_label)

        self.backend_combo = QComboBox()
        self.backend_combo.setMaximumWidth(220)
        self.backend_combo.setMinimumWidth(180)
        self.backend_combo.addItems([self.BACKEND_ALL, self.BACKEND_CPU, self.BACKEND_GPU])
        self.backend_combo.setCurrentText(self.BACKEND_ALL)
        self.backend_combo.currentTextChanged.connect(self._on_backend_changed)
        self.backend_combo.setToolTip(
            "Select which energy backend data to display:\n"
            "- CPU + GPU: Shows separate bars for each backend\n"
            "- CPU Only: Energy from CPU (Intel RAPL)\n"
            "- GPU Only: Energy from GPU (NVIDIA NVML)"
        )
        left_layout.addWidget(self.backend_combo)

        # Generate Energy Report button
        self.report_btn = QPushButton("Generate Energy Report")
        self.report_btn.setMaximumWidth(220)
        self.report_btn.setMinimumWidth(180)
        self.report_btn.setMinimumHeight(40)
        self.report_btn.setEnabled(False)
        self.report_btn.setStyleSheet("""
            QPushButton {
                background-color: #0078d7;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
                font-size: 14px;
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
        self.report_btn.setToolTip("Generate detailed energy analysis report")
        self.report_btn.clicked.connect(self._on_generate_report)
        left_layout.addWidget(self.report_btn)

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

        self.canvas.setContextMenuPolicy(Qt.CustomContextMenu)
        self.canvas.customContextMenuRequested.connect(self._show_context_menu)

        self.toolbar = CustomNavigationToolbar(
            self.canvas, self,
            config_callback=self._on_open_chart_config
        )

        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas, 1)

        # Energy Summary table (dynamic columns per backend)
        self.summary_group = QGroupBox("Energy Summary")
        self.summary_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #ccc;
                border-radius: 4px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
            }
        """)
        self.summary_layout = QGridLayout(self.summary_group)
        self.summary_layout.setSpacing(8)

        # Test selector for summary table (like Timing view)
        selector_layout = QHBoxLayout()
        test_label = QLabel("Show details for:")
        test_label.setStyleSheet("font-weight: bold;")
        selector_layout.addWidget(test_label)

        self.test_combo = QComboBox()
        self.test_combo.setMinimumWidth(200)
        self.test_combo.currentIndexChanged.connect(self._on_test_changed)
        selector_layout.addWidget(self.test_combo)
        selector_layout.addStretch()
        self.summary_layout.addLayout(selector_layout, 0, 0, 1, 4)

        # Storage for dynamic labels
        self._summary_backend_labels = {}
        self._summary_header_labels = []
        self._summary_row_labels = []

        # Create initial table structure
        self._create_summary_table_structure()

        # Enable right-click for copy
        self.summary_group.setContextMenuPolicy(Qt.CustomContextMenu)
        self.summary_group.customContextMenuRequested.connect(self._show_summary_copy_menu)

        right_layout.addWidget(self.summary_group)

        # Info label
        self.info_label = QLabel("Load experiments to see energy analysis")
        self.info_label.setStyleSheet("color: #666; font-size: 11px;")
        self.info_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.info_label)

        splitter.addWidget(right_panel)
        splitter.setSizes([180, 720])

        main_layout.addWidget(splitter)

    def _create_summary_table_structure(self):
        """Create the initial summary table structure with row labels."""
        header_font = QFont()
        header_font.setBold(True)

        # Row definitions: (row_index, label_text, metric_key, unit, value_style)
        # Start from row 1 since row 0 has the test selector
        self._summary_rows = [
            (2, "Energy/image:", "energy", "mJ", "font-weight: bold; color: #FF9800;"),
            (3, "Avg Power:", "power", "W", "font-weight: bold; color: #4CAF50;"),
            (4, "Efficiency:", "efficiency", "img/J", "font-weight: bold; color: #2196F3;"),
        ]

        # Column 0: Metric names header (row 1)
        metric_header = QLabel("Metric")
        metric_header.setFont(header_font)
        metric_header.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.summary_layout.addWidget(metric_header, 1, 0)

        for row_idx, label_text, _, _, _ in self._summary_rows:
            row_label = QLabel(label_text)
            row_label.setFont(header_font)
            self.summary_layout.addWidget(row_label, row_idx, 0)
            self._summary_row_labels.append(row_label)

    def _clear_summary_backend_columns(self):
        """Remove all backend columns from the summary table."""
        for label in self._summary_header_labels:
            self.summary_layout.removeWidget(label)
            label.deleteLater()
        self._summary_header_labels.clear()

        for backend_labels in self._summary_backend_labels.values():
            for label in backend_labels.values():
                self.summary_layout.removeWidget(label)
                label.deleteLater()
        self._summary_backend_labels.clear()

    def _on_test_changed(self, index: int):
        """Handle test selection change for summary table."""
        self._update_summary_table()

    def _update_summary_table(self):
        """Update the summary table with energy values for the selected test."""
        self._clear_summary_backend_columns()

        current_idx = self.test_combo.currentIndex()

        if not self._tests or current_idx < 0 or current_idx >= len(self._tests):
            return

        test = self._tests[current_idx]

        header_font = QFont()
        header_font.setBold(True)

        # Get energy data for this test
        gpu_e = self._get_nested_value(test, "energy_gpu_mj")
        gpu_p = self._get_nested_value(test, "energy_gpu_watts")
        cpu_e = self._get_nested_value(test, "energy_cpu_mj")
        cpu_p = self._get_nested_value(test, "energy_cpu_watts")

        has_gpu = gpu_e is not None and gpu_e > 0
        has_cpu = cpu_e is not None and cpu_e > 0

        # If no per-backend data, fall back to combined data
        if not has_gpu and not has_cpu:
            combined_e = self._get_energy_value_combined(test)
            combined_p = self._get_power_value_combined(test)

            if combined_e is not None:
                col_idx = 1
                header_label = QLabel("Combined")
                header_label.setFont(header_font)
                header_label.setAlignment(Qt.AlignCenter)
                self.summary_layout.addWidget(header_label, 1, col_idx)
                self._summary_header_labels.append(header_label)

                backend_labels = {}
                for row_idx, _, metric_key, _, value_style in self._summary_rows:
                    value_label = QLabel("-")
                    value_label.setAlignment(Qt.AlignCenter)
                    if value_style:
                        value_label.setStyleSheet(value_style)
                    self.summary_layout.addWidget(value_label, row_idx, col_idx)
                    backend_labels[metric_key] = value_label

                backend_labels["energy"].setText(f"{combined_e:.2f}")
                if combined_p is not None:
                    backend_labels["power"].setText(f"{combined_p:.2f}")
                efficiency = 1000.0 / combined_e if combined_e > 0 else 0
                backend_labels["efficiency"].setText(f"{efficiency:.1f}")

                self._summary_backend_labels["Combined"] = backend_labels
                col_idx += 1
        else:
            col_idx = 1

            # GPU column
            if has_gpu:
                header_label = QLabel("GPU")
                header_label.setFont(header_font)
                header_label.setAlignment(Qt.AlignCenter)
                header_label.setStyleSheet("color: #FF9800;")
                self.summary_layout.addWidget(header_label, 1, col_idx)
                self._summary_header_labels.append(header_label)

                backend_labels = {}
                for row_idx, _, metric_key, _, value_style in self._summary_rows:
                    value_label = QLabel("-")
                    value_label.setAlignment(Qt.AlignCenter)
                    if value_style:
                        value_label.setStyleSheet(value_style)
                    self.summary_layout.addWidget(value_label, row_idx, col_idx)
                    backend_labels[metric_key] = value_label

                backend_labels["energy"].setText(f"{gpu_e:.2f}")
                if gpu_p is not None:
                    backend_labels["power"].setText(f"{gpu_p:.2f}")
                efficiency = 1000.0 / gpu_e if gpu_e > 0 else 0
                backend_labels["efficiency"].setText(f"{efficiency:.1f}")

                self._summary_backend_labels["GPU"] = backend_labels
                col_idx += 1

            # CPU column
            if has_cpu:
                header_label = QLabel("CPU")
                header_label.setFont(header_font)
                header_label.setAlignment(Qt.AlignCenter)
                header_label.setStyleSheet("color: #2196F3;")
                self.summary_layout.addWidget(header_label, 1, col_idx)
                self._summary_header_labels.append(header_label)

                backend_labels = {}
                for row_idx, _, metric_key, _, value_style in self._summary_rows:
                    value_label = QLabel("-")
                    value_label.setAlignment(Qt.AlignCenter)
                    if value_style:
                        value_label.setStyleSheet(value_style)
                    self.summary_layout.addWidget(value_label, row_idx, col_idx)
                    backend_labels[metric_key] = value_label

                backend_labels["energy"].setText(f"{cpu_e:.2f}")
                if cpu_p is not None:
                    backend_labels["power"].setText(f"{cpu_p:.2f}")
                efficiency = 1000.0 / cpu_e if cpu_e > 0 else 0
                backend_labels["efficiency"].setText(f"{efficiency:.1f}")

                self._summary_backend_labels["CPU"] = backend_labels
                col_idx += 1

        # Add Unit column
        if col_idx > 1:
            unit_header = QLabel("Unit")
            unit_header.setFont(header_font)
            unit_header.setAlignment(Qt.AlignCenter)
            self.summary_layout.addWidget(unit_header, 1, col_idx)
            self._summary_header_labels.append(unit_header)

            for row_idx, _, _, unit, _ in self._summary_rows:
                unit_label = QLabel(unit)
                unit_label.setAlignment(Qt.AlignCenter)
                unit_label.setStyleSheet("color: #666;")
                self.summary_layout.addWidget(unit_label, row_idx, col_idx)
                self._summary_header_labels.append(unit_label)

    def _show_summary_copy_menu(self, pos):
        """Show context menu for copying summary table."""
        menu = QMenu(self)
        copy_action = menu.addAction("Copy table")
        action = menu.exec_(self.summary_group.mapToGlobal(pos))

        if action == copy_action:
            self._copy_summary_table()

    def _copy_summary_table(self):
        """Copy the summary table to clipboard as tab-separated values."""
        from PyQt5.QtWidgets import QApplication

        lines = []

        # Header row
        headers = ["Metric"]
        for backend_name in self._summary_backend_labels.keys():
            headers.append(backend_name)
        headers.append("Unit")
        lines.append("\t".join(headers))

        # Data rows
        for row_idx, label_text, metric_key, unit, _ in self._summary_rows:
            row_data = [label_text]
            for backend_name, labels in self._summary_backend_labels.items():
                row_data.append(labels[metric_key].text())
            row_data.append(unit)
            lines.append("\t".join(row_data))

        text = "\n".join(lines)
        QApplication.clipboard().setText(text)
        self.logger.info("Summary table copied to clipboard")

    def _on_open_chart_config(self):
        """Open chart configuration dialog."""
        popup = ChartConfigPopup(parent=self, logger=self.logger)
        popup.set_config(self._chart_config)

        if popup.exec_() == QDialog.Accepted:
            self._chart_config = popup.get_config()
            self.logger.debug("Chart config updated: %s", self._chart_config)
            self._refresh_chart()

    def _on_backend_changed(self, backend: str):
        """Handle backend selection change."""
        self._backend_filter = backend
        self._refresh_chart()

    def _on_generate_report(self):
        """Generate and show the energy report popup."""
        if not self._tests:
            self.logger.warning("No data available for report generation")
            return

        from ui.custom_widgets.batch_reports.comparison_views.batch_energy_report_popup import (
            BatchEnergyReportPopup
        )

        popup = BatchEnergyReportPopup(parent=self, logger=self.logger)
        popup.set_data(self._tests)
        popup.exec_()

        self.logger.info("Batch energy report displayed")

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
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=xlabel_fontsize)

        ylabel = axes_cfg.get('ylabel', '') or default_ylabel
        ylabel_fontsize = axes_cfg.get('ylabel_fontsize', 11)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)

        xtick_fontsize = axes_cfg.get('xtick_fontsize', 9)
        ytick_fontsize = axes_cfg.get('ytick_fontsize', 9)
        ax.tick_params(axis='x', labelsize=xtick_fontsize)
        ax.tick_params(axis='y', labelsize=ytick_fontsize)

        if not axes_cfg.get('auto_scale', True):
            ymin = axes_cfg.get('ymin', 0)
            ymax = axes_cfg.get('ymax', 100)
            ax.set_ylim(ymin, ymax)

    def set_tests(self, tests: List[Dict[str, Any]]):
        """Set the tests to display in the charts."""
        self._tests = tests

        # Update test combo for summary table (like Timing view)
        self.test_combo.clear()
        for test in tests:
            test_name = test.get("name", "Unknown")
            exp_name = test.get("_experiment_name", "")
            if exp_name:
                self.test_combo.addItem(f"{test_name} ({exp_name})")
            else:
                self.test_combo.addItem(test_name)

        self._update_summary_table()
        self._refresh_chart()

    def _on_chart_type_changed(self, index: int):
        """Handle chart type selection change."""
        self._refresh_chart()

    def _has_energy_data(self, test: dict) -> bool:
        """Check if a test has energy data."""
        return (
            self._get_nested_value(test, "energy_mean_mj") is not None or
            self._get_nested_value(test, "mean_energy_mj") is not None or
            self._get_nested_value(test, "energy_gpu_mj") is not None or
            self._get_nested_value(test, "energy_cpu_mj") is not None
        )

    def _get_energy_value_combined(self, test: dict) -> Optional[float]:
        """Get combined energy value in mJ from a test."""
        for key in ["energy_mean_mj", "mean_energy_mj"]:
            val = self._get_nested_value(test, key)
            if val is not None:
                return val
        return None

    def _get_power_value_combined(self, test: dict) -> Optional[float]:
        """Get combined power value in Watts from a test."""
        for key in ["energy_mean_watts", "mean_power_watts", "power_mean_watts"]:
            val = self._get_nested_value(test, key)
            if val is not None:
                return val
        return None

    def _get_efficiency_value(self, test: dict) -> Optional[float]:
        """Get efficiency value (images/J) from a test."""
        for key in ["efficiency_images_per_joule", "energy_efficiency"]:
            val = self._get_nested_value(test, key)
            if val is not None:
                return val
        energy_mj = self._get_energy_value_combined(test)
        if energy_mj is not None and energy_mj > 0:
            return 1000.0 / energy_mj
        return None

    def _refresh_chart(self):
        """Refresh the chart based on current settings."""
        self.figure.clear()

        if not self._tests:
            self._draw_no_data_message("No data to display\nLoad experiments first")
            return

        tests_with_energy = [t for t in self._tests if self._has_energy_data(t)]
        if not tests_with_energy:
            self._draw_no_data_message(
                "No energy data available\n\n"
                "Energy measurement was not enabled for these tests.\n"
                "Enable 'energy' in report types when running batch tests."
            )
            return

        chart_type = self.chart_list.currentRow()

        if chart_type == 0:
            self._draw_energy_bar()
        elif chart_type == 1:
            self._draw_power_comparison()
        elif chart_type == 2:
            self._draw_efficiency_chart()
        else:
            self._draw_statistics_chart()

        legend_pos = self._chart_config['legend']['position']
        if legend_pos == 4:
            self.figure.tight_layout(rect=[0, 0.05, 0.85, 1])
        elif legend_pos == 5:
            self.figure.tight_layout(rect=[0, 0.15, 1, 1])
        else:
            self.figure.tight_layout(rect=[0, 0.05, 1, 1])

        self.canvas.draw()
        self.report_btn.setEnabled(True)

        experiment_count = len(set(t.get("_experiment_name", "") for t in tests_with_energy))
        backend_info = f" ({self._backend_filter})" if self._backend_filter != self.BACKEND_ALL else ""
        self.info_label.setText(
            f"Showing {len(tests_with_energy)} tests with energy data{backend_info} from {experiment_count} experiment(s)"
        )

    def _draw_no_data_message(self, message: str):
        """Draw a message when no data is available."""
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, message,
               ha='center', va='center', fontsize=12, color='#999',
               multialignment='center')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        self.canvas.draw()
        self.report_btn.setEnabled(False)

    def _collect_backend_data(self, gpu_key: str, cpu_key: str, combined_keys: list = None):
        """
        Collect GPU and CPU data for all tests.

        Returns:
            tuple: (test_names, gpu_values, cpu_values, has_gpu, has_cpu)
        """
        test_names = []
        gpu_values = []
        cpu_values = []

        for test in self._tests:
            test_name = test.get("name", "Unknown")
            if len(test_name) > 15:
                test_name = test_name[:12] + "..."
            test_names.append(test_name)

            gpu_val = self._get_nested_value(test, gpu_key)
            cpu_val = self._get_nested_value(test, cpu_key)

            # Fallback to combined if no per-backend data
            if gpu_val is None and cpu_val is None and combined_keys:
                for key in combined_keys:
                    combined = self._get_nested_value(test, key)
                    if combined is not None:
                        gpu_val = combined
                        break

            gpu_values.append(gpu_val if gpu_val else 0)
            cpu_values.append(cpu_val if cpu_val else 0)

        has_gpu = any(v > 0 for v in gpu_values)
        has_cpu = any(v > 0 for v in cpu_values)

        return test_names, gpu_values, cpu_values, has_gpu, has_cpu

    def _draw_grouped_bar_chart(self, ax, test_names, gpu_values, cpu_values,
                                 has_gpu, has_cpu, value_format=".1f",
                                 annotation_text=None):
        """
        Draw a grouped bar chart with CPU/GPU on X-axis and test names below.

        Layout: [GPU] [CPU]  gap  [GPU] [CPU]  gap  ...
                  Test 1              Test 2
        """
        if not has_gpu and not has_cpu:
            ax.text(0.5, 0.5, "No valid data available",
                   ha='center', va='center', fontsize=12, color='#999')
            ax.axis('off')
            return False

        # Build positions and data
        x_positions = []
        x_labels = []
        bar_values = []
        bar_colors = []
        group_positions = []  # (start_idx, end_idx, test_name)

        pos = 0.0
        bar_idx = 0
        gpu_first_idx = -1
        cpu_first_idx = -1

        for i, test_name in enumerate(test_names):
            group_start_idx = bar_idx

            # GPU bar (if data available for this backend)
            if has_gpu:
                x_positions.append(pos)
                x_labels.append("GPU")
                bar_values.append(gpu_values[i])
                bar_colors.append(self.COLOR_GPU)
                if gpu_first_idx < 0:
                    gpu_first_idx = bar_idx
                pos += 1
                bar_idx += 1

            # CPU bar (if data available for this backend)
            if has_cpu:
                x_positions.append(pos)
                x_labels.append("CPU")
                bar_values.append(cpu_values[i])
                bar_colors.append(self.COLOR_CPU)
                if cpu_first_idx < 0:
                    cpu_first_idx = bar_idx
                pos += 1
                bar_idx += 1

            group_end_idx = bar_idx - 1
            if group_end_idx >= group_start_idx:
                group_positions.append((group_start_idx, group_end_idx, test_name))

            pos += 0.5  # Gap between tests

        x = np.array(x_positions)
        width = 0.7

        # Draw bars
        for idx in range(len(x)):
            # Set label only for first occurrence of each color (for legend)
            label = None
            if bar_colors[idx] == self.COLOR_GPU and idx == gpu_first_idx:
                label = 'GPU'
            elif bar_colors[idx] == self.COLOR_CPU and idx == cpu_first_idx:
                label = 'CPU'

            ax.bar(x[idx], bar_values[idx], width, label=label,
                  color=bar_colors[idx], alpha=0.8)

            # Value label on top
            if bar_values[idx] > 0:
                ax.text(x[idx], bar_values[idx], f'{bar_values[idx]:{value_format}}',
                       ha='center', va='bottom', fontsize=8, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=9)

        # Add test name labels below X-axis (below CPU/GPU labels)
        for group_start_idx, group_end_idx, test_name in group_positions:
            group_center = (x_positions[group_start_idx] + x_positions[group_end_idx]) / 2
            # Use axes fraction for y (-0.12 places it below the tick labels)
            ax.text(group_center, -0.12, test_name,
                   ha='center', va='top', fontsize=9, fontweight='bold',
                   transform=ax.get_xaxis_transform())

        # Adjust bottom margin for test names
        self.figure.subplots_adjust(bottom=0.22)

        ax.grid(axis='y', alpha=0.3)

        if annotation_text:
            ax.annotate(annotation_text, xy=(1, 1), xycoords='axes fraction',
                       fontsize=9, color='#666', ha='right', va='top')

        return True

    def _draw_single_backend_chart(self, ax, test_names, values, is_gpu: bool,
                                    value_format=".1f", annotation_text=None):
        """Draw a chart for a single backend with test names on X-axis."""
        if not any(v > 0 for v in values):
            backend_name = "GPU" if is_gpu else "CPU"
            ax.text(0.5, 0.5, f"No {backend_name} data available",
                   ha='center', va='center', fontsize=12, color='#999')
            ax.axis('off')
            return False

        x = np.arange(len(test_names))
        width = 0.6
        color = self.COLOR_GPU if is_gpu else self.COLOR_CPU
        label = 'GPU' if is_gpu else 'CPU'

        bars = ax.bar(x, values, width, label=label, color=color, alpha=0.8)

        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                       f'{val:{value_format}}', ha='center', va='bottom', fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(test_names, rotation=45, ha='right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)

        if annotation_text:
            ax.annotate(annotation_text, xy=(1, 1), xycoords='axes fraction',
                       fontsize=9, color='#666', ha='right', va='top')

        return True

    def _draw_energy_bar(self):
        """Draw energy bar chart with CPU/GPU on X-axis and test names below."""
        ax = self.figure.add_subplot(111)

        test_names, gpu_values, cpu_values, has_gpu, has_cpu = self._collect_backend_data(
            "energy_gpu_mj", "energy_cpu_mj", ["energy_mean_mj", "mean_energy_mj"]
        )

        if self._backend_filter == self.BACKEND_ALL:
            success = self._draw_grouped_bar_chart(
                ax, test_names, gpu_values, cpu_values, has_gpu, has_cpu,
                value_format=".1f", annotation_text="(lower is better)"
            )
            if success:
                self._apply_axes_config(ax, "Energy Consumption Comparison", "", "Energy (mJ)")
                self._apply_legend(ax)
        elif self._backend_filter == self.BACKEND_GPU:
            success = self._draw_single_backend_chart(
                ax, test_names, gpu_values, is_gpu=True,
                value_format=".1f", annotation_text="(lower is better)"
            )
            if success:
                self._apply_axes_config(ax, "Energy Consumption (GPU)", "Test", "Energy (mJ)")
                self._apply_legend(ax)
        else:  # CPU Only
            success = self._draw_single_backend_chart(
                ax, test_names, cpu_values, is_gpu=False,
                value_format=".1f", annotation_text="(lower is better)"
            )
            if success:
                self._apply_axes_config(ax, "Energy Consumption (CPU)", "Test", "Energy (mJ)")
                self._apply_legend(ax)

    def _draw_power_comparison(self):
        """Draw power comparison chart with CPU/GPU on X-axis and test names below."""
        ax = self.figure.add_subplot(111)

        test_names, gpu_values, cpu_values, has_gpu, has_cpu = self._collect_backend_data(
            "energy_gpu_watts", "energy_cpu_watts", ["energy_mean_watts", "mean_power_watts"]
        )

        if self._backend_filter == self.BACKEND_ALL:
            success = self._draw_grouped_bar_chart(
                ax, test_names, gpu_values, cpu_values, has_gpu, has_cpu,
                value_format=".1f", annotation_text=None
            )
            if success:
                self._apply_axes_config(ax, "Power Consumption Comparison", "", "Power (W)")
                self._apply_legend(ax)
        elif self._backend_filter == self.BACKEND_GPU:
            success = self._draw_single_backend_chart(
                ax, test_names, gpu_values, is_gpu=True,
                value_format=".1f", annotation_text=None
            )
            if success:
                self._apply_axes_config(ax, "Power Consumption (GPU)", "Test", "Power (W)")
                self._apply_legend(ax)
        else:  # CPU Only
            success = self._draw_single_backend_chart(
                ax, test_names, cpu_values, is_gpu=False,
                value_format=".1f", annotation_text=None
            )
            if success:
                self._apply_axes_config(ax, "Power Consumption (CPU)", "Test", "Power (W)")
                self._apply_legend(ax)

    def _get_efficiency_per_backend(self, test: dict) -> tuple:
        """
        Calculate efficiency per backend (images per Joule).

        Returns:
            tuple: (gpu_efficiency, cpu_efficiency)
        """
        gpu_e = self._get_nested_value(test, "energy_gpu_mj")
        cpu_e = self._get_nested_value(test, "energy_cpu_mj")

        gpu_eff = 1000.0 / gpu_e if gpu_e and gpu_e > 0 else 0
        cpu_eff = 1000.0 / cpu_e if cpu_e and cpu_e > 0 else 0

        return gpu_eff, cpu_eff

    def _draw_efficiency_chart(self):
        """Draw efficiency chart with CPU/GPU on X-axis and test names below."""
        ax = self.figure.add_subplot(111)

        # Collect efficiency per backend
        test_names = []
        gpu_values = []
        cpu_values = []

        for test in self._tests:
            test_name = test.get("name", "Unknown")
            if len(test_name) > 15:
                test_name = test_name[:12] + "..."
            test_names.append(test_name)

            gpu_eff, cpu_eff = self._get_efficiency_per_backend(test)

            # Fallback to combined efficiency if no per-backend data
            if gpu_eff == 0 and cpu_eff == 0:
                combined_eff = self._get_efficiency_value(test)
                gpu_values.append(combined_eff if combined_eff else 0)
                cpu_values.append(0)
            else:
                gpu_values.append(gpu_eff)
                cpu_values.append(cpu_eff)

        has_gpu = any(v > 0 for v in gpu_values)
        has_cpu = any(v > 0 for v in cpu_values)

        if self._backend_filter == self.BACKEND_ALL:
            success = self._draw_grouped_bar_chart(
                ax, test_names, gpu_values, cpu_values, has_gpu, has_cpu,
                value_format=".0f", annotation_text="(higher is better)"
            )
            if success:
                self._apply_axes_config(ax, "Energy Efficiency Comparison", "", "Efficiency (images/J)")
                self._apply_legend(ax)
        elif self._backend_filter == self.BACKEND_GPU:
            success = self._draw_single_backend_chart(
                ax, test_names, gpu_values, is_gpu=True,
                value_format=".0f", annotation_text="(higher is better)"
            )
            if success:
                self._apply_axes_config(ax, "Energy Efficiency (GPU)", "Test", "Efficiency (images/J)")
                self._apply_legend(ax)
        else:  # CPU Only
            success = self._draw_single_backend_chart(
                ax, test_names, cpu_values, is_gpu=False,
                value_format=".0f", annotation_text="(higher is better)"
            )
            if success:
                self._apply_axes_config(ax, "Energy Efficiency (CPU)", "Test", "Efficiency (images/J)")
                self._apply_legend(ax)

    def _draw_statistics_chart(self):
        """Draw box plot of energy distribution with CPU/GPU on X-axis and test names below."""
        ax = self.figure.add_subplot(111)

        # Collect energy data per backend per test
        test_names, gpu_values, cpu_values, has_gpu, has_cpu = self._collect_backend_data(
            "energy_gpu_mj", "energy_cpu_mj", ["energy_mean_mj", "mean_energy_mj"]
        )

        if not has_gpu and not has_cpu:
            ax.text(0.5, 0.5, "No valid energy data available",
                   ha='center', va='center', fontsize=12, color='#999')
            ax.axis('off')
            return

        # Build data for box plot
        box_data = []
        box_labels = []
        box_colors = []
        group_positions = []  # (start_idx, end_idx, test_name)

        idx = 0
        for i, test_name in enumerate(test_names):
            group_start = idx

            if self._backend_filter == self.BACKEND_ALL:
                # Add GPU data point if available
                if has_gpu and gpu_values[i] > 0:
                    box_data.append([gpu_values[i]])
                    box_labels.append("GPU")
                    box_colors.append(self.COLOR_GPU)
                    idx += 1

                # Add CPU data point if available
                if has_cpu and cpu_values[i] > 0:
                    box_data.append([cpu_values[i]])
                    box_labels.append("CPU")
                    box_colors.append(self.COLOR_CPU)
                    idx += 1
            elif self._backend_filter == self.BACKEND_GPU:
                if gpu_values[i] > 0:
                    box_data.append([gpu_values[i]])
                    box_labels.append("GPU")
                    box_colors.append(self.COLOR_GPU)
                    idx += 1
            else:  # CPU Only
                if cpu_values[i] > 0:
                    box_data.append([cpu_values[i]])
                    box_labels.append("CPU")
                    box_colors.append(self.COLOR_CPU)
                    idx += 1

            group_end = idx - 1
            if group_end >= group_start:
                group_positions.append((group_start, group_end, test_name))

        if not box_data:
            ax.text(0.5, 0.5, "No valid energy data available",
                   ha='center', va='center', fontsize=12, color='#999')
            ax.axis('off')
            return

        # Create box plot
        positions = list(range(len(box_data)))

        # Add gaps between test groups
        adjusted_positions = []
        pos = 0
        current_group = 0
        for i in range(len(box_data)):
            # Check if we're starting a new group
            if current_group < len(group_positions):
                start, end, _ = group_positions[current_group]
                if i > end:
                    pos += 0.5  # Add gap
                    current_group += 1
            adjusted_positions.append(pos)
            pos += 1

        bp = ax.boxplot(box_data, positions=adjusted_positions, patch_artist=True, widths=0.6)

        # Color the boxes
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xticks(adjusted_positions)
        ax.set_xticklabels(box_labels, fontsize=9)

        # Add test name labels below (below CPU/GPU labels)
        for group_start, group_end, test_name in group_positions:
            if group_start < len(adjusted_positions) and group_end < len(adjusted_positions):
                group_center = (adjusted_positions[group_start] + adjusted_positions[group_end]) / 2
                # Use axes fraction for y (-0.12 places it below the tick labels)
                ax.text(group_center, -0.12, test_name,
                       ha='center', va='top', fontsize=9, fontweight='bold',
                       transform=ax.get_xaxis_transform())

        self.figure.subplots_adjust(bottom=0.22)
        ax.grid(axis='y', alpha=0.3)

        title_suffix = ""
        if self._backend_filter == self.BACKEND_GPU:
            title_suffix = " (GPU)"
        elif self._backend_filter == self.BACKEND_CPU:
            title_suffix = " (CPU)"

        self._apply_axes_config(ax, f"Energy Distribution{title_suffix}", "", "Energy (mJ)")

    def _apply_legend(self, ax):
        """Apply legend configuration to the axis."""
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
        elif legend_pos == 4:
            ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
                     ncol=legend_cfg['ncol'], **legend_kwargs)
        else:
            legend_kwargs['frameon'] = False
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                     ncol=4, **legend_kwargs)

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

    def _show_context_menu(self, pos):
        """Show context menu for saving the chart."""
        menu = QMenu(self)
        save_action = menu.addAction("Save chart as...")
        save_action.triggered.connect(self._on_save_chart)
        menu.exec_(self.canvas.mapToGlobal(pos))

    def _on_save_chart(self):
        """Save chart to file."""
        if not self._tests:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Chart",
            "energy_chart.png",
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

    def clear(self):
        """Clear all data from the view."""
        self._tests = []
        self.test_combo.clear()
        self.figure.clear()
        self.canvas.draw()
        self._clear_summary_backend_columns()
        self.info_label.setText("Load experiments to see energy analysis")
        self.report_btn.setEnabled(False)
