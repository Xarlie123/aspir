"""Popup dialog for displaying detailed batch energy analysis report."""
import logging
from typing import List, Dict, Any

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QGridLayout, QFileDialog, QSizePolicy,
    QWidget, QMenu, QApplication, QSplitter
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

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


class BatchEnergyReportPopup(QDialog):
    """
    Popup dialog showing detailed batch energy analysis report with:
    - Box plots for energy variability per backend (GPU/CPU)
    - Box plots for power variability per backend
    - Detailed statistics table (mean, std, min, max, percentiles)
    - Navigation toolbars with chart configuration
    - Resizable splitters for flexible layout
    - Right-click to save any chart
    """

    # Backend-specific colors
    COLOR_GPU = '#FF9800'  # Orange for GPU
    COLOR_CPU = '#2196F3'  # Blue for CPU

    def __init__(self, parent=None, logger=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Energy Analysis Report")
        self.setMinimumSize(1100, 800)
        self.resize(1200, 900)

        if logger:
            self.logger = logger.getChild("BatchEnergyReportPopup")
        else:
            self.logger = logging.getLogger("ASPIR.BatchEnergyReportPopup")

        self._tests: List[Dict[str, Any]] = []

        # Chart configurations for each plot
        self._chart_configs = {
            'energy_box': self._create_default_config(),
            'power_box': self._create_default_config(),
            'backend': self._create_default_config(),
        }

        self._setup_ui()

    def _create_default_config(self) -> dict:
        """Create a default chart configuration."""
        return {
            'axes': {
                'title': '',
                'title_fontsize': 12,
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

    def _setup_ui(self):
        """Setup the popup UI layout with splitters."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Title
        title = QLabel("<h2>Batch Energy Analysis Report</h2>")
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)

        # Main vertical splitter (charts vs statistics)
        main_splitter = QSplitter(Qt.Vertical)

        # Top section: Charts area
        charts_widget = QWidget()
        charts_layout = QVBoxLayout(charts_widget)
        charts_layout.setContentsMargins(0, 0, 0, 0)
        charts_layout.setSpacing(5)

        # Horizontal splitter for top row (Energy boxplot | Power boxplot)
        top_splitter = QSplitter(Qt.Horizontal)

        # Energy boxplot
        energy_box_widget = self._create_chart_widget(
            "Energy per Test (Box Plot)",
            "energy_box"
        )
        self.energy_box_figure = energy_box_widget.figure
        self.energy_box_canvas = energy_box_widget.canvas
        top_splitter.addWidget(energy_box_widget)

        # Power boxplot
        power_box_widget = self._create_chart_widget(
            "Power per Test (Box Plot)",
            "power_box"
        )
        self.power_box_figure = power_box_widget.figure
        self.power_box_canvas = power_box_widget.canvas
        top_splitter.addWidget(power_box_widget)

        top_splitter.setSizes([500, 500])
        charts_layout.addWidget(top_splitter, 1)

        # Backend comparison chart
        backend_widget = self._create_chart_widget(
            "Energy by Backend Comparison",
            "backend"
        )
        self.backend_figure = backend_widget.figure
        self.backend_canvas = backend_widget.canvas
        charts_layout.addWidget(backend_widget, 1)

        main_splitter.addWidget(charts_widget)

        # Bottom section: Statistics table
        stats_widget = QWidget()
        stats_layout = QVBoxLayout(stats_widget)
        stats_layout.setContentsMargins(5, 5, 5, 5)

        stats_group = QGroupBox("Detailed Statistics")
        stats_group.setContextMenuPolicy(Qt.CustomContextMenu)
        stats_group.customContextMenuRequested.connect(self._show_copy_menu)
        stats_grid = QGridLayout(stats_group)
        stats_grid.setSpacing(8)

        # Create statistics table
        self._stats_labels = {}
        self._setup_stats_table(stats_grid)

        stats_layout.addWidget(stats_group)
        main_splitter.addWidget(stats_widget)

        # Set initial splitter sizes (charts take more space than stats)
        main_splitter.setSizes([600, 200])

        main_layout.addWidget(main_splitter, 1)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.setMinimumHeight(36)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #607D8B;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
                font-size: 13px;
                padding: 8px 20px;
            }
            QPushButton:hover {
                background-color: #546E7A;
            }
        """)
        close_btn.clicked.connect(self.accept)
        main_layout.addWidget(close_btn, 0, Qt.AlignRight)

    def _create_chart_widget(self, title: str, config_key: str) -> QWidget:
        """Create a chart widget with figure, canvas, and toolbar."""
        widget = QGroupBox(title)
        widget.setContextMenuPolicy(Qt.CustomContextMenu)

        layout = QVBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(2)

        # Create figure and canvas
        figure = Figure(figsize=(5, 3), dpi=100)
        canvas = FigureCanvas(figure)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Create toolbar with config button
        toolbar = CustomNavigationToolbar(
            canvas, widget,
            config_callback=lambda ck=config_key: self._on_open_chart_config(ck)
        )

        layout.addWidget(toolbar)
        layout.addWidget(canvas, 1)

        # Store references on widget for easy access
        widget.figure = figure
        widget.canvas = canvas
        widget.toolbar = toolbar
        widget.config_key = config_key

        # Connect right-click menu
        widget.customContextMenuRequested.connect(
            lambda pos, w=widget, f=figure, n=config_key: self._show_save_menu(pos, w, f, n)
        )

        return widget

    def _on_open_chart_config(self, config_key: str):
        """Open chart configuration dialog for specified chart."""
        popup = ChartConfigPopup(parent=self, logger=self.logger)
        popup.set_config(self._chart_configs[config_key])

        if popup.exec() == QDialog.Accepted:
            self._chart_configs[config_key] = popup.get_config()
            self.logger.debug("Chart config updated for %s", config_key)
            self._update_charts()

    def _apply_axes_config(self, ax, config_key: str, default_title: str = "",
                           default_xlabel: str = "", default_ylabel: str = ""):
        """Apply axes configuration to a matplotlib axis."""
        axes_cfg = self._chart_configs[config_key].get('axes', {})

        title = axes_cfg.get('title', '') or default_title
        title_fontsize = axes_cfg.get('title_fontsize', 12)
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

    def _apply_legend(self, ax, config_key: str):
        """Apply legend configuration to the axis."""
        legend_cfg = self._chart_configs[config_key]['legend']
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

    def _setup_stats_table(self, layout: QGridLayout):
        """Setup the statistics table with headers."""
        header_font = QFont()
        header_font.setBold(True)

        # Column headers
        headers = ["", "GPU Energy (mJ)", "CPU Energy (mJ)", "GPU Power (W)", "CPU Power (W)"]
        for col, header in enumerate(headers):
            label = QLabel(header)
            label.setFont(header_font)
            label.setAlignment(Qt.AlignCenter)
            if col == 1:  # GPU
                label.setStyleSheet("color: #FF9800;")
            elif col == 2:  # CPU
                label.setStyleSheet("color: #2196F3;")
            layout.addWidget(label, 0, col)

        # Row labels
        row_labels = ["Mean", "Std", "CV%", "Min", "Max", "P25", "P50 (Median)", "P75"]
        for row, label_text in enumerate(row_labels, start=1):
            label = QLabel(label_text)
            label.setFont(header_font)
            layout.addWidget(label, row, 0)

            # Create value labels for each column
            for col in range(1, 5):
                value_label = QLabel("-")
                value_label.setAlignment(Qt.AlignCenter)
                layout.addWidget(value_label, row, col)

                # Store reference
                col_key = ["gpu_energy", "cpu_energy", "gpu_power", "cpu_power"][col - 1]
                row_key = row_labels[row - 1].lower().replace(" ", "_").replace("(", "").replace(")", "")
                key = f"{col_key}_{row_key}"
                self._stats_labels[key] = value_label

    def set_data(self, tests: List[Dict[str, Any]]):
        """Set the test data to display."""
        self._tests = tests
        self._update_charts()
        self._update_stats_table()

    def _update_charts(self):
        """Update all charts."""
        self._draw_energy_boxplot()
        self._draw_power_boxplot()
        self._draw_backend_comparison()

    def _draw_energy_boxplot(self):
        """Draw energy boxplot per backend."""
        self.energy_box_figure.clear()
        ax = self.energy_box_figure.add_subplot(111)

        # Collect data
        gpu_energies = []
        cpu_energies = []

        for test in self._tests:
            gpu_e = self._get_value(test, "energy_gpu_mj")
            cpu_e = self._get_value(test, "energy_cpu_mj")

            if gpu_e is not None:
                gpu_energies.append(gpu_e)
            if cpu_e is not None:
                cpu_energies.append(cpu_e)

        # Fallback to combined if no per-backend data
        if not gpu_energies and not cpu_energies:
            for test in self._tests:
                for key in ["energy_mean_mj", "mean_energy_mj"]:
                    val = self._get_value(test, key)
                    if val is not None:
                        gpu_energies.append(val)
                        break

        if not gpu_energies and not cpu_energies:
            ax.text(0.5, 0.5, "No energy data available",
                   ha='center', va='center', fontsize=12, color='#999')
            ax.axis('off')
            self.energy_box_canvas.draw()
            return

        # Prepare data and labels
        data = []
        labels = []
        colors = []

        if gpu_energies:
            data.append(gpu_energies)
            labels.append("GPU")
            colors.append(self.COLOR_GPU)
        if cpu_energies:
            data.append(cpu_energies)
            labels.append("CPU")
            colors.append(self.COLOR_CPU)

        bp = ax.boxplot(data, labels=labels, patch_artist=True)

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.grid(axis='y', alpha=0.3)

        self._apply_axes_config(ax, 'energy_box',
                                default_title="Energy Distribution by Backend",
                                default_ylabel="Energy (mJ)")

        self.energy_box_figure.tight_layout()
        self.energy_box_canvas.draw()

    def _draw_power_boxplot(self):
        """Draw power boxplot per backend."""
        self.power_box_figure.clear()
        ax = self.power_box_figure.add_subplot(111)

        # Collect data
        gpu_powers = []
        cpu_powers = []

        for test in self._tests:
            gpu_p = self._get_value(test, "energy_gpu_watts")
            cpu_p = self._get_value(test, "energy_cpu_watts")

            if gpu_p is not None:
                gpu_powers.append(gpu_p)
            if cpu_p is not None:
                cpu_powers.append(cpu_p)

        # Fallback to combined
        if not gpu_powers and not cpu_powers:
            for test in self._tests:
                for key in ["energy_mean_watts", "mean_power_watts"]:
                    val = self._get_value(test, key)
                    if val is not None:
                        gpu_powers.append(val)
                        break

        if not gpu_powers and not cpu_powers:
            ax.text(0.5, 0.5, "No power data available",
                   ha='center', va='center', fontsize=12, color='#999')
            ax.axis('off')
            self.power_box_canvas.draw()
            return

        # Prepare data and labels
        data = []
        labels = []
        colors = []

        if gpu_powers:
            data.append(gpu_powers)
            labels.append("GPU")
            colors.append(self.COLOR_GPU)
        if cpu_powers:
            data.append(cpu_powers)
            labels.append("CPU")
            colors.append(self.COLOR_CPU)

        bp = ax.boxplot(data, labels=labels, patch_artist=True)

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.grid(axis='y', alpha=0.3)

        self._apply_axes_config(ax, 'power_box',
                                default_title="Power Distribution by Backend",
                                default_ylabel="Power (W)")

        self.power_box_figure.tight_layout()
        self.power_box_canvas.draw()

    def _draw_backend_comparison(self):
        """Draw side-by-side backend comparison chart."""
        self.backend_figure.clear()
        ax = self.backend_figure.add_subplot(111)

        # Collect data per test
        test_names = []
        gpu_values = []
        cpu_values = []

        for test in self._tests:
            test_name = test.get("name", "Unknown")
            if len(test_name) > 12:
                test_name = test_name[:10] + "..."
            test_names.append(test_name)

            gpu_e = self._get_value(test, "energy_gpu_mj")
            cpu_e = self._get_value(test, "energy_cpu_mj")

            # Fallback to combined
            if gpu_e is None and cpu_e is None:
                for key in ["energy_mean_mj", "mean_energy_mj"]:
                    val = self._get_value(test, key)
                    if val is not None:
                        gpu_e = val
                        break

            gpu_values.append(gpu_e if gpu_e else 0)
            cpu_values.append(cpu_e if cpu_e else 0)

        has_gpu = any(v > 0 for v in gpu_values)
        has_cpu = any(v > 0 for v in cpu_values)

        if not has_gpu and not has_cpu:
            ax.text(0.5, 0.5, "No energy data available",
                   ha='center', va='center', fontsize=12, color='#999')
            ax.axis('off')
            self.backend_canvas.draw()
            return

        x = np.arange(len(test_names))

        if has_gpu and has_cpu:
            width = 0.35
            bars1 = ax.bar(x - width/2, gpu_values, width, label='GPU', color=self.COLOR_GPU, alpha=0.8)
            bars2 = ax.bar(x + width/2, cpu_values, width, label='CPU', color=self.COLOR_CPU, alpha=0.8)

            for bar, val in zip(bars1, gpu_values):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{val:.1f}', ha='center', va='bottom', fontsize=8)
            for bar, val in zip(bars2, cpu_values):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{val:.1f}', ha='center', va='bottom', fontsize=8)
        else:
            width = 0.6
            values = gpu_values if has_gpu else cpu_values
            color = self.COLOR_GPU if has_gpu else self.COLOR_CPU
            label = 'GPU' if has_gpu else 'CPU'

            bars = ax.bar(x, values, width, label=label, color=color, alpha=0.8)
            for bar, val in zip(bars, values):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{val:.1f}', ha='center', va='bottom', fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(test_names, rotation=0, ha='center', fontsize=9)
        ax.grid(axis='y', alpha=0.3)

        self._apply_axes_config(ax, 'backend',
                                default_title="Energy by Test and Backend",
                                default_ylabel="Energy (mJ)")
        self._apply_legend(ax, 'backend')

        self.backend_figure.tight_layout()
        self.backend_canvas.draw()

    def _update_stats_table(self):
        """Update the statistics table with calculated values."""
        # Collect data
        gpu_energies = []
        cpu_energies = []
        gpu_powers = []
        cpu_powers = []

        for test in self._tests:
            gpu_e = self._get_value(test, "energy_gpu_mj")
            cpu_e = self._get_value(test, "energy_cpu_mj")
            gpu_p = self._get_value(test, "energy_gpu_watts")
            cpu_p = self._get_value(test, "energy_cpu_watts")

            if gpu_e is not None:
                gpu_energies.append(gpu_e)
            if cpu_e is not None:
                cpu_energies.append(cpu_e)
            if gpu_p is not None:
                gpu_powers.append(gpu_p)
            if cpu_p is not None:
                cpu_powers.append(cpu_p)

        # Calculate and display statistics
        data_sets = {
            "gpu_energy": gpu_energies,
            "cpu_energy": cpu_energies,
            "gpu_power": gpu_powers,
            "cpu_power": cpu_powers,
        }

        for prefix, data in data_sets.items():
            if data:
                arr = np.array(data)
                mean = np.mean(arr)
                std = np.std(arr)
                cv = (std / mean * 100) if mean > 0 else 0

                self._set_stat_value(f"{prefix}_mean", f"{mean:.2f}")
                self._set_stat_value(f"{prefix}_std", f"{std:.2f}")
                self._set_stat_value(f"{prefix}_cv%", f"{cv:.1f}")
                self._set_stat_value(f"{prefix}_min", f"{np.min(arr):.2f}")
                self._set_stat_value(f"{prefix}_max", f"{np.max(arr):.2f}")
                self._set_stat_value(f"{prefix}_p25", f"{np.percentile(arr, 25):.2f}")
                self._set_stat_value(f"{prefix}_p50_median", f"{np.percentile(arr, 50):.2f}")
                self._set_stat_value(f"{prefix}_p75", f"{np.percentile(arr, 75):.2f}")
            else:
                for stat in ["mean", "std", "cv%", "min", "max", "p25", "p50_median", "p75"]:
                    self._set_stat_value(f"{prefix}_{stat}", "-")

    def _set_stat_value(self, key: str, value: str):
        """Set a statistics table value."""
        if key in self._stats_labels:
            self._stats_labels[key].setText(value)

    def _get_value(self, data: dict, key: str):
        """Get a value from a dictionary."""
        return data.get(key)

    def _show_save_menu(self, pos, widget, figure, chart_name):
        """Show context menu to save chart."""
        menu = QMenu(self)
        save_png = menu.addAction("Save as PNG...")
        save_pdf = menu.addAction("Save as PDF...")

        action = menu.exec(widget.mapToGlobal(pos))

        if action == save_png:
            self._save_figure(figure, chart_name, "png")
        elif action == save_pdf:
            self._save_figure(figure, chart_name, "pdf")

    def _save_figure(self, figure, chart_name: str, ext: str):
        """Save a figure to file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Chart",
            f"batch_{chart_name}.{ext}",
            f"{ext.upper()} Files (*.{ext});;All Files (*.*)"
        )

        if not file_path:
            return

        try:
            figure.savefig(file_path, dpi=150, bbox_inches='tight',
                          facecolor='white', edgecolor='none')
            self.logger.info("Saved chart to %s", file_path)
        except Exception as e:
            self.logger.error("Failed to save chart: %s", e)

    def _show_copy_menu(self, pos):
        """Show context menu for copying statistics table."""
        menu = QMenu(self)
        copy_action = menu.addAction("Copy statistics")
        action = menu.exec(self.sender().mapToGlobal(pos))

        if action == copy_action:
            self._copy_stats_table()

    def _copy_stats_table(self):
        """Copy the statistics table to clipboard."""
        lines = []

        # Header
        lines.append("\tGPU Energy (mJ)\tCPU Energy (mJ)\tGPU Power (W)\tCPU Power (W)")

        # Rows
        row_labels = ["Mean", "Std", "CV%", "Min", "Max", "P25", "P50 (Median)", "P75"]
        for row_label in row_labels:
            row_key = row_label.lower().replace(" ", "_").replace("(", "").replace(")", "")
            row_data = [row_label]
            for col_key in ["gpu_energy", "cpu_energy", "gpu_power", "cpu_power"]:
                key = f"{col_key}_{row_key}"
                val = self._stats_labels.get(key, None)
                row_data.append(val.text() if val else "-")
            lines.append("\t".join(row_data))

        text = "\n".join(lines)
        QApplication.clipboard().setText(text)
        self.logger.info("Statistics copied to clipboard")
