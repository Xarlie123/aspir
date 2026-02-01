"""Popup dialog for displaying detailed energy analysis report."""
import logging
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QGridLayout, QFileDialog, QSizePolicy, QScrollArea,
    QWidget, QFrame, QMenu, QApplication
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont


def _shorten_backend_name(name: str, max_length: int = 25) -> str:
    """
    Shorten a backend name for display in legends.

    Removes common suffixes like "with Radeon Graphics" and truncates if needed.
    """
    # Remove "with ..." suffix (common in AMD CPUs)
    if " with " in name:
        name = name.split(" with ")[0]

    # Truncate if still too long
    if len(name) > max_length:
        return name[:max_length - 3] + "..."
    return name


class EnergyReportPopup(QDialog):
    """
    Popup dialog showing detailed energy analysis report with:
    - Box plots (candlestick) for energy variability per backend
    - Box plots for power variability per backend
    - Detailed statistics table (mean, std, min, max, percentiles)
    - Right-click to save any chart
    """

    def __init__(self, parent=None, logger=None):
        super().__init__(parent)
        self.setWindowTitle("Energy Analysis Report")
        self.setMinimumSize(1100, 800)
        self.resize(1200, 850)

        if logger:
            self.logger = logger.getChild("EnergyReportPopup")
        else:
            self.logger = logging.getLogger("SPIm.EnergyReportPopup")

        # Data storage
        self._energy_data = {}

        self._setup_ui()

    def _setup_ui(self):
        """Setup the popup UI layout."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Title
        title = QLabel("<h2>Energy Analysis Report</h2>")
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)

        # Scroll area for content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setSpacing(15)

        # Top row: Energy boxplot + Power boxplot
        top_row = QHBoxLayout()
        top_row.setSpacing(15)

        # Energy boxplot
        energy_box_group = QGroupBox("Energy per Image (Box Plot)")
        energy_box_group.setContextMenuPolicy(Qt.CustomContextMenu)
        energy_box_group.customContextMenuRequested.connect(
            lambda pos: self._show_save_menu(pos, energy_box_group, self.energy_box_figure, "energy_boxplot")
        )
        energy_box_layout = QVBoxLayout(energy_box_group)
        self.energy_box_figure = Figure(figsize=(5, 4), dpi=100)
        self.energy_box_canvas = FigureCanvas(self.energy_box_figure)
        self.energy_box_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        energy_box_layout.addWidget(self.energy_box_canvas)
        top_row.addWidget(energy_box_group)

        # Power boxplot
        power_box_group = QGroupBox("Power per Image (Box Plot)")
        power_box_group.setContextMenuPolicy(Qt.CustomContextMenu)
        power_box_group.customContextMenuRequested.connect(
            lambda pos: self._show_save_menu(pos, power_box_group, self.power_box_figure, "power_boxplot")
        )
        power_box_layout = QVBoxLayout(power_box_group)
        self.power_box_figure = Figure(figsize=(5, 4), dpi=100)
        self.power_box_canvas = FigureCanvas(self.power_box_figure)
        self.power_box_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        power_box_layout.addWidget(self.power_box_canvas)
        top_row.addWidget(power_box_group)

        content_layout.addLayout(top_row)

        # Middle row: Energy distribution histogram
        middle_row = QHBoxLayout()
        middle_row.setSpacing(15)

        # Energy histogram
        hist_group = QGroupBox("Energy Distribution")
        hist_group.setContextMenuPolicy(Qt.CustomContextMenu)
        hist_group.customContextMenuRequested.connect(
            lambda pos: self._show_save_menu(pos, hist_group, self.hist_figure, "energy_histogram")
        )
        hist_layout = QVBoxLayout(hist_group)
        self.hist_figure = Figure(figsize=(5, 3), dpi=100)
        self.hist_canvas = FigureCanvas(self.hist_figure)
        self.hist_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        hist_layout.addWidget(self.hist_canvas)
        middle_row.addWidget(hist_group)

        content_layout.addLayout(middle_row)

        # Bottom row: Statistics table
        self._stats_group = QGroupBox("Detailed Statistics")
        stats_layout = QGridLayout(self._stats_group)
        stats_layout.setSpacing(8)

        header_font = QFont()
        header_font.setBold(True)

        # Headers
        self._stats_headers = ["Backend", "Type", "Mean", "Std", "CV%", "Min", "Max", "P25", "P50", "P75"]
        for col, h in enumerate(self._stats_headers):
            label = QLabel(h)
            label.setFont(header_font)
            label.setAlignment(Qt.AlignCenter)
            stats_layout.addWidget(label, 0, col)

        # Placeholder for dynamic rows
        self._stats_layout = stats_layout
        self._stats_rows = []

        # Enable right-click context menu for copying
        self._stats_group.setContextMenuPolicy(Qt.CustomContextMenu)
        self._stats_group.customContextMenuRequested.connect(self._show_stats_copy_menu)

        content_layout.addWidget(self._stats_group)

        scroll.setWidget(content_widget)
        main_layout.addWidget(scroll, 1)

        # Bottom buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        close_btn = QPushButton("Close")
        close_btn.setMinimumWidth(100)
        close_btn.clicked.connect(self.close)
        button_layout.addWidget(close_btn)

        main_layout.addLayout(button_layout)

    def set_data(self, energy_data: dict):
        """
        Set energy data and update all charts.

        Args:
            energy_data: Dictionary with energy measurement results including
                         'backends' for per-backend breakdown
        """
        self._energy_data = energy_data
        self._update_energy_boxplot()
        self._update_power_boxplot()
        self._update_histogram()
        self._update_statistics_table()

    def _update_energy_boxplot(self):
        """Update the energy boxplot."""
        self.energy_box_figure.clear()
        ax = self.energy_box_figure.add_subplot(111)

        backends = self._energy_data.get('backends', {})
        if not backends:
            ax.text(0.5, 0.5, "No backend data available", ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
            self.energy_box_canvas.draw()
            return

        # Prepare data for boxplot
        data = []
        labels = []
        colors = []

        color_map = {'gpu': '#FF9800', 'cpu': '#2196F3'}

        for backend_name, backend_data in backends.items():
            energy_values = backend_data.get('energy_per_image_mj', [])
            if energy_values:
                data.append(energy_values)
                backend_type = backend_data.get('type', 'gpu')
                short_name = 'GPU' if backend_type == 'gpu' else 'CPU'
                labels.append(f"{short_name}\n{_shorten_backend_name(backend_name, 20)}")
                colors.append(color_map.get(backend_type, '#9E9E9E'))

        if not data:
            ax.text(0.5, 0.5, "No energy data", ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
            self.energy_box_canvas.draw()
            return

        bp = ax.boxplot(data, patch_artist=True, labels=labels)

        # Color the boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Style whiskers and medians
        for whisker in bp['whiskers']:
            whisker.set(color='#666666', linewidth=1.5)
        for cap in bp['caps']:
            cap.set(color='#666666', linewidth=1.5)
        for median in bp['medians']:
            median.set(color='#D32F2F', linewidth=2)
        for flier in bp['fliers']:
            flier.set(marker='o', markerfacecolor='#666666', alpha=0.5, markersize=4)

        ax.set_ylabel('Energy (mJ)')
        ax.set_title('Energy Consumption per Inference')
        ax.grid(True, alpha=0.3, axis='y')

        self.energy_box_figure.tight_layout()
        self.energy_box_canvas.draw()

    def _update_power_boxplot(self):
        """Update the power boxplot."""
        self.power_box_figure.clear()
        ax = self.power_box_figure.add_subplot(111)

        backends = self._energy_data.get('backends', {})
        if not backends:
            ax.text(0.5, 0.5, "No backend data available", ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
            self.power_box_canvas.draw()
            return

        # Prepare data for boxplot
        data = []
        labels = []
        colors = []

        color_map = {'gpu': '#FF5722', 'cpu': '#009688'}

        for backend_name, backend_data in backends.items():
            power_values = backend_data.get('power_per_image_watts', [])
            if power_values:
                data.append(power_values)
                backend_type = backend_data.get('type', 'gpu')
                short_name = 'GPU' if backend_type == 'gpu' else 'CPU'
                labels.append(f"{short_name}\n{_shorten_backend_name(backend_name, 20)}")
                colors.append(color_map.get(backend_type, '#9E9E9E'))

        if not data:
            ax.text(0.5, 0.5, "No power data", ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
            self.power_box_canvas.draw()
            return

        bp = ax.boxplot(data, patch_artist=True, labels=labels)

        # Color the boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Style whiskers and medians
        for whisker in bp['whiskers']:
            whisker.set(color='#666666', linewidth=1.5)
        for cap in bp['caps']:
            cap.set(color='#666666', linewidth=1.5)
        for median in bp['medians']:
            median.set(color='#D32F2F', linewidth=2)
        for flier in bp['fliers']:
            flier.set(marker='o', markerfacecolor='#666666', alpha=0.5, markersize=4)

        ax.set_ylabel('Power (W)')
        ax.set_title('Average Power During Inference')
        ax.grid(True, alpha=0.3, axis='y')

        self.power_box_figure.tight_layout()
        self.power_box_canvas.draw()

    def _update_histogram(self):
        """Update the energy distribution histogram."""
        self.hist_figure.clear()
        ax = self.hist_figure.add_subplot(111)

        backends = self._energy_data.get('backends', {})
        if not backends:
            ax.text(0.5, 0.5, "No backend data available", ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
            self.hist_canvas.draw()
            return

        color_map = {'gpu': '#FF9800', 'cpu': '#2196F3'}

        for backend_name, backend_data in backends.items():
            energy_values = backend_data.get('energy_per_image_mj', [])
            if energy_values:
                backend_type = backend_data.get('type', 'gpu')
                color = color_map.get(backend_type, '#9E9E9E')
                short_name = 'GPU' if backend_type == 'gpu' else 'CPU'

                ax.hist(energy_values, bins=20, alpha=0.6, color=color,
                        label=f"{short_name}: {_shorten_backend_name(backend_name)}", edgecolor='white')

        ax.set_xlabel('Energy (mJ)')
        ax.set_ylabel('Frequency')
        ax.set_title('Energy Distribution')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

        self.hist_figure.tight_layout()
        self.hist_canvas.draw()

    def _update_statistics_table(self):
        """Update the statistics table."""
        # Clear existing rows
        for row_widgets in self._stats_rows:
            for widget in row_widgets:
                widget.deleteLater()
        self._stats_rows = []

        backends = self._energy_data.get('backends', {})
        if not backends:
            return

        row_idx = 1
        for backend_name, backend_data in backends.items():
            energy_values = np.array(backend_data.get('energy_per_image_mj', []))
            if len(energy_values) == 0:
                continue

            backend_type = backend_data.get('type', 'gpu')
            short_type = 'GPU' if backend_type == 'gpu' else 'CPU'

            # Calculate statistics
            mean_val = np.mean(energy_values)
            std_val = np.std(energy_values)
            cv_pct = (std_val / mean_val * 100) if mean_val > 0 else 0
            min_val = np.min(energy_values)
            max_val = np.max(energy_values)
            p25 = np.percentile(energy_values, 25)
            p50 = np.percentile(energy_values, 50)
            p75 = np.percentile(energy_values, 75)

            # Create row widgets
            row_widgets = []

            # Backend name (shortened)
            name_label = QLabel(_shorten_backend_name(backend_name))
            name_label.setToolTip(backend_name)
            self._stats_layout.addWidget(name_label, row_idx, 0)
            row_widgets.append(name_label)

            # Type
            type_label = QLabel(short_type)
            type_label.setAlignment(Qt.AlignCenter)
            type_label.setStyleSheet(
                f"color: {'#E65100' if backend_type == 'gpu' else '#1565C0'}; font-weight: bold;"
            )
            self._stats_layout.addWidget(type_label, row_idx, 1)
            row_widgets.append(type_label)

            # Statistics values
            values = [
                f"{mean_val:.3f}",
                f"{std_val:.3f}",
                f"{cv_pct:.1f}%",
                f"{min_val:.3f}",
                f"{max_val:.3f}",
                f"{p25:.3f}",
                f"{p50:.3f}",
                f"{p75:.3f}"
            ]

            for col, val in enumerate(values, start=2):
                val_label = QLabel(val)
                val_label.setAlignment(Qt.AlignCenter)
                self._stats_layout.addWidget(val_label, row_idx, col)
                row_widgets.append(val_label)

            self._stats_rows.append(row_widgets)
            row_idx += 1

        # Add total row if multiple backends
        if len(backends) > 1:
            total_energy = self._energy_data.get('energy_per_image_mj', [])
            if total_energy:
                energy_arr = np.array(total_energy)

                mean_val = np.mean(energy_arr)
                std_val = np.std(energy_arr)
                cv_pct = (std_val / mean_val * 100) if mean_val > 0 else 0
                min_val = np.min(energy_arr)
                max_val = np.max(energy_arr)
                p25 = np.percentile(energy_arr, 25)
                p50 = np.percentile(energy_arr, 50)
                p75 = np.percentile(energy_arr, 75)

                row_widgets = []

                # Total label
                total_label = QLabel("TOTAL")
                total_label.setStyleSheet("font-weight: bold;")
                self._stats_layout.addWidget(total_label, row_idx, 0)
                row_widgets.append(total_label)

                # Type
                type_label = QLabel("All")
                type_label.setAlignment(Qt.AlignCenter)
                type_label.setStyleSheet("font-weight: bold;")
                self._stats_layout.addWidget(type_label, row_idx, 1)
                row_widgets.append(type_label)

                # Statistics values
                values = [
                    f"{mean_val:.3f}",
                    f"{std_val:.3f}",
                    f"{cv_pct:.1f}%",
                    f"{min_val:.3f}",
                    f"{max_val:.3f}",
                    f"{p25:.3f}",
                    f"{p50:.3f}",
                    f"{p75:.3f}"
                ]

                for col, val in enumerate(values, start=2):
                    val_label = QLabel(val)
                    val_label.setAlignment(Qt.AlignCenter)
                    val_label.setStyleSheet("font-weight: bold;")
                    self._stats_layout.addWidget(val_label, row_idx, col)
                    row_widgets.append(val_label)

                self._stats_rows.append(row_widgets)

    def _show_save_menu(self, pos, widget, figure, chart_name):
        """Show context menu to save a chart."""
        menu = QMenu(self)
        save_png = menu.addAction("Save as PNG...")
        save_pdf = menu.addAction("Save as PDF...")
        save_svg = menu.addAction("Save as SVG...")

        action = menu.exec_(widget.mapToGlobal(pos))

        if action == save_png:
            self._save_figure(figure, chart_name, "png")
        elif action == save_pdf:
            self._save_figure(figure, chart_name, "pdf")
        elif action == save_svg:
            self._save_figure(figure, chart_name, "svg")

    def _save_figure(self, figure, chart_name, ext):
        """Save a figure to file."""
        default_name = f"energy_{chart_name}.{ext}"

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Chart", default_name,
            f"{ext.upper()} Files (*.{ext});;All Files (*.*)"
        )

        if file_path:
            try:
                figure.savefig(file_path, dpi=150, bbox_inches='tight')
                self.logger.info(f"Saved chart to {file_path}")
            except Exception as e:
                self.logger.error(f"Failed to save chart: {e}")

    def _show_stats_copy_menu(self, pos):
        """Show context menu to copy the statistics table."""
        menu = QMenu(self)
        copy_action = menu.addAction("Copy table")
        action = menu.exec_(self._stats_group.mapToGlobal(pos))

        if action == copy_action:
            self._copy_stats_table()

    def _copy_stats_table(self):
        """Copy the statistics table to clipboard as tab-separated values."""
        backends = self._energy_data.get('backends', {})
        if not backends:
            return

        lines = []
        # Header
        lines.append("\t".join(self._stats_headers))

        # Data rows
        for backend_name, backend_data in backends.items():
            energy_values = np.array(backend_data.get('energy_per_image_mj', []))
            if len(energy_values) == 0:
                continue

            backend_type = backend_data.get('type', 'gpu')
            short_type = 'GPU' if backend_type == 'gpu' else 'CPU'

            mean_val = np.mean(energy_values)
            std_val = np.std(energy_values)
            cv_pct = (std_val / mean_val * 100) if mean_val > 0 else 0
            min_val = np.min(energy_values)
            max_val = np.max(energy_values)
            p25 = np.percentile(energy_values, 25)
            p50 = np.percentile(energy_values, 50)
            p75 = np.percentile(energy_values, 75)

            row = [
                backend_name,
                short_type,
                f"{mean_val:.3f}",
                f"{std_val:.3f}",
                f"{cv_pct:.1f}%",
                f"{min_val:.3f}",
                f"{max_val:.3f}",
                f"{p25:.3f}",
                f"{p50:.3f}",
                f"{p75:.3f}"
            ]
            lines.append("\t".join(row))

        # Add total row if multiple backends
        if len(backends) > 1:
            total_energy = self._energy_data.get('energy_per_image_mj', [])
            if total_energy:
                energy_arr = np.array(total_energy)
                mean_val = np.mean(energy_arr)
                std_val = np.std(energy_arr)
                cv_pct = (std_val / mean_val * 100) if mean_val > 0 else 0
                min_val = np.min(energy_arr)
                max_val = np.max(energy_arr)
                p25 = np.percentile(energy_arr, 25)
                p50 = np.percentile(energy_arr, 50)
                p75 = np.percentile(energy_arr, 75)

                row = [
                    "TOTAL",
                    "All",
                    f"{mean_val:.3f}",
                    f"{std_val:.3f}",
                    f"{cv_pct:.1f}%",
                    f"{min_val:.3f}",
                    f"{max_val:.3f}",
                    f"{p25:.3f}",
                    f"{p50:.3f}",
                    f"{p75:.3f}"
                ]
                lines.append("\t".join(row))

        text = "\n".join(lines)
        QApplication.clipboard().setText(text)
        self.logger.info("Statistics table copied to clipboard")
