"""Popup dialog for displaying detailed timing analysis report."""
import logging
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QGridLayout, QFileDialog, QSizePolicy, QScrollArea,
    QWidget, QFrame, QMenu
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont


class TimingReportPopup(QDialog):
    """
    Popup dialog showing detailed timing analysis report with:
    - Time per image curves (acquisition, reconstruction, inference, total)
    - Time distribution histograms
    - Stacked bar chart (CPU vs GPU)
    - Detailed statistics table (mean, std, min, max, percentiles)
    """

    def __init__(self, parent=None, logger=None):
        super().__init__(parent)
        self.setWindowTitle("Timing Analysis Report")
        self.setMinimumSize(1100, 800)
        self.resize(1200, 850)

        if logger:
            self.logger = logger.getChild("TimingReportPopup")
        else:
            self.logger = logging.getLogger("ASPIR.TimingReportPopup")

        # Data storage
        self._timing_data = {}

        self._setup_ui()

    def _setup_ui(self):
        """Setup the popup UI layout."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Title
        title = QLabel("<h2>Timing Analysis Report</h2>")
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)

        # Scroll area for content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setSpacing(15)

        # Top row: Time per image + Distribution histograms
        top_row = QHBoxLayout()
        top_row.setSpacing(15)

        # Time per image curves
        curves_group = QGroupBox("Time per Image")
        curves_group.setContextMenuPolicy(Qt.CustomContextMenu)
        curves_group.customContextMenuRequested.connect(
            lambda pos: self._show_save_menu(pos, curves_group, self.curves_figure, "time_per_image")
        )
        curves_layout = QVBoxLayout(curves_group)
        self.curves_figure = Figure(figsize=(5, 4), dpi=100)
        self.curves_canvas = FigureCanvas(self.curves_figure)
        self.curves_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        curves_layout.addWidget(self.curves_canvas)
        top_row.addWidget(curves_group)

        # Distribution histograms
        hist_group = QGroupBox("Time Distribution")
        hist_group.setContextMenuPolicy(Qt.CustomContextMenu)
        hist_group.customContextMenuRequested.connect(
            lambda pos: self._show_save_menu(pos, hist_group, self.hist_figure, "time_distribution")
        )
        hist_layout = QVBoxLayout(hist_group)
        self.hist_figure = Figure(figsize=(5, 4), dpi=100)
        self.hist_canvas = FigureCanvas(self.hist_figure)
        self.hist_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        hist_layout.addWidget(self.hist_canvas)
        top_row.addWidget(hist_group)

        content_layout.addLayout(top_row)

        # Bottom row: Stacked bar + Statistics table
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(15)

        # Stacked bar chart
        bar_group = QGroupBox("Pipeline Latency Breakdown")
        bar_group.setContextMenuPolicy(Qt.CustomContextMenu)
        bar_group.customContextMenuRequested.connect(
            lambda pos: self._show_save_menu(pos, bar_group, self.bar_figure, "pipeline_breakdown")
        )
        bar_layout = QVBoxLayout(bar_group)
        self.bar_figure = Figure(figsize=(5, 4), dpi=100)
        self.bar_canvas = FigureCanvas(self.bar_figure)
        self.bar_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        bar_layout.addWidget(self.bar_canvas)
        bottom_row.addWidget(bar_group)

        # Statistics table
        stats_group = QGroupBox("Detailed Statistics")
        stats_layout = QGridLayout(stats_group)
        stats_layout.setSpacing(8)

        header_font = QFont()
        header_font.setBold(True)

        # Headers
        headers = ["", "Mean", "Std", "Min", "Max", "P25", "P50", "P75"]
        for col, h in enumerate(headers):
            label = QLabel(h)
            label.setFont(header_font)
            label.setAlignment(Qt.AlignCenter)
            stats_layout.addWidget(label, 0, col)

        # Rows for each timing component
        self.stats_labels = {}
        rows = [
            ("T_reconstruction", "t_recon"),
            ("T_inference (CPU)", "t_inf_cpu"),
            ("T_inference (GPU)", "t_inf_gpu"),
            ("T_total (CPU)", "t_total_cpu"),
            ("T_total (GPU)", "t_total_gpu")
        ]

        for row_idx, (label_text, key) in enumerate(rows, start=1):
            row_label = QLabel(label_text)
            row_label.setFont(header_font)
            stats_layout.addWidget(row_label, row_idx, 0)

            self.stats_labels[key] = []
            for col in range(1, 8):
                val_label = QLabel("-")
                val_label.setAlignment(Qt.AlignCenter)
                stats_layout.addWidget(val_label, row_idx, col)
                self.stats_labels[key].append(val_label)

        bottom_row.addWidget(stats_group)

        content_layout.addLayout(bottom_row)

        scroll.setWidget(content_widget)
        main_layout.addWidget(scroll, 1)

        # Buttons
        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch()

        self.export_pdf_button = QPushButton("Export PDF")
        self.export_pdf_button.clicked.connect(lambda: self._on_export("pdf"))
        buttons_layout.addWidget(self.export_pdf_button)

        self.export_png_button = QPushButton("Export PNG")
        self.export_png_button.clicked.connect(lambda: self._on_export("png"))
        buttons_layout.addWidget(self.export_png_button)

        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close)
        buttons_layout.addWidget(self.close_button)

        main_layout.addLayout(buttons_layout)

    def set_data(self, timing_data: dict):
        """
        Set the timing data and update all displays.

        Args:
            timing_data: Dictionary with timing values including:
                - t_acq_ms: acquisition time in ms
                - t_recon_ms: mean reconstruction time in ms
                - t_inf_cpu_ms: mean inference time on CPU in ms
                - t_inf_gpu_ms: mean inference time on GPU in ms (optional)
                - recon_times_ms: list of per-image reconstruction times
                - denoise_times_cpu_ms: list of per-image CPU inference times
                - denoise_times_gpu_ms: list of per-image GPU inference times (optional)
        """
        self._timing_data = timing_data

        self._update_curves_chart()
        self._update_histogram()
        self._update_stacked_bar()
        self._update_statistics()

    def _update_curves_chart(self):
        """Update the time per image curves chart."""
        self.curves_figure.clear()

        t_acq = self._timing_data.get('t_acq_ms', 0)
        recon_times = self._timing_data.get('recon_times_ms', [])
        denoise_cpu = self._timing_data.get('denoise_times_cpu_ms', [])
        denoise_gpu = self._timing_data.get('denoise_times_gpu_ms', [])

        if not recon_times and not denoise_cpu:
            ax = self.curves_figure.add_subplot(111)
            ax.text(0.5, 0.5, "No per-image data available", ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
            self.curves_canvas.draw()
            return

        ax = self.curves_figure.add_subplot(111)

        n_images = max(len(recon_times), len(denoise_cpu), len(denoise_gpu) if denoise_gpu else 0)
        x = np.arange(n_images)

        # Acquisition (constant)
        acq_arr = np.full(n_images, t_acq)
        ax.plot(x, acq_arr, '--', label='Acquisition', color='#abdda4', linewidth=2)

        # Reconstruction
        if recon_times:
            ax.plot(x, recon_times, label='Reconstruction', color='#fdae61', linewidth=1.5, marker='o', markersize=3)

        # Inference CPU
        if denoise_cpu:
            ax.plot(x, denoise_cpu, label='Inference (CPU)', color='#d7191c', linewidth=1.5, marker='s', markersize=3)

        # Inference GPU
        if denoise_gpu:
            ax.plot(x, denoise_gpu, label='Inference (GPU)', color='#2b83ba', linewidth=1.5, marker='^', markersize=3)

        # Total (CPU)
        if recon_times and denoise_cpu:
            total_cpu = acq_arr + np.array(recon_times[:n_images]) + np.array(denoise_cpu[:n_images])
            ax.plot(x, total_cpu, label='Total (CPU)', color='#d7191c', linewidth=2, linestyle=':', alpha=0.7)

        # Total (GPU)
        if recon_times and denoise_gpu:
            total_gpu = acq_arr + np.array(recon_times[:n_images]) + np.array(denoise_gpu[:n_images])
            ax.plot(x, total_gpu, label='Total (GPU)', color='#2b83ba', linewidth=2, linestyle=':', alpha=0.7)

        ax.set_xlabel('Image Index')
        ax.set_ylabel('Time (ms)')
        ax.set_title('Time per Image')
        # Place legend outside the plot area (below)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=7)
        ax.grid(True, alpha=0.3)

        self.curves_figure.subplots_adjust(bottom=0.25)
        self.curves_canvas.draw()

    def _update_histogram(self):
        """Update the time distribution histogram."""
        self.hist_figure.clear()

        recon_times = self._timing_data.get('recon_times_ms', [])
        denoise_cpu = self._timing_data.get('denoise_times_cpu_ms', [])
        denoise_gpu = self._timing_data.get('denoise_times_gpu_ms', [])

        if not recon_times and not denoise_cpu:
            ax = self.hist_figure.add_subplot(111)
            ax.text(0.5, 0.5, "No distribution data available", ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
            self.hist_canvas.draw()
            return

        # Use side-by-side layout (1 row, 2 columns) for better fit
        # Reconstruction histogram (left)
        if recon_times:
            ax1 = self.hist_figure.add_subplot(1, 2, 1)
            ax1.hist(recon_times, bins=20, color='#fdae61', alpha=0.7, edgecolor='white')
            ax1.set_xlabel('Time (ms)', fontsize=9)
            ax1.set_ylabel('Frequency', fontsize=9)
            ax1.set_title('Reconstruction', fontsize=10)
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(labelsize=8)

        # Inference histograms (right) - CPU and GPU overlapped
        if denoise_cpu or denoise_gpu:
            ax2 = self.hist_figure.add_subplot(1, 2, 2)

            if denoise_cpu:
                ax2.hist(denoise_cpu, bins=20, color='#d7191c', alpha=0.6, label='CPU', edgecolor='white')
            if denoise_gpu:
                ax2.hist(denoise_gpu, bins=20, color='#2b83ba', alpha=0.6, label='GPU', edgecolor='white')

            ax2.set_xlabel('Time (ms)', fontsize=9)
            ax2.set_ylabel('Frequency', fontsize=9)
            ax2.set_title('Inference', fontsize=10)
            ax2.legend(loc='upper right', fontsize=7)
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(labelsize=8)

        self.hist_figure.tight_layout()
        self.hist_canvas.draw()

    def _update_stacked_bar(self):
        """Update the stacked bar chart."""
        self.bar_figure.clear()

        t_acq = self._timing_data.get('t_acq_ms', 0)
        t_recon = self._timing_data.get('t_recon_ms', 0)
        t_inf_cpu = self._timing_data.get('t_inf_cpu_ms', 0)
        t_inf_gpu = self._timing_data.get('t_inf_gpu_ms', None)

        ax = self.bar_figure.add_subplot(111)

        # Colors
        c_acq = '#abdda4'
        c_recon = '#fdae61'
        c_cpu = '#d7191c'
        c_gpu = '#2b83ba'

        x = np.array([0, 1])
        width = 0.5

        # CPU bar
        ax.bar(x[0], t_acq, width, label='Acquisition', color=c_acq, edgecolor='white')
        ax.bar(x[0], t_recon, width, bottom=t_acq, label='Reconstruction', color=c_recon, edgecolor='white')
        ax.bar(x[0], t_inf_cpu, width, bottom=t_acq + t_recon, label='Inference (CPU)', color=c_cpu, edgecolor='white')

        # GPU bar
        if t_inf_gpu is not None:
            ax.bar(x[1], t_acq, width, color=c_acq, edgecolor='white')
            ax.bar(x[1], t_recon, width, bottom=t_acq, color=c_recon, edgecolor='white')
            ax.bar(x[1], t_inf_gpu, width, bottom=t_acq + t_recon, label='Inference (GPU)', color=c_gpu, edgecolor='white')

        ax.set_ylabel('Latency (ms)')
        ax.set_xticks(x)
        ax.set_xticklabels(['CPU', 'GPU'] if t_inf_gpu is not None else ['CPU', ''])
        # Place legend outside the plot area (below)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_title('Pipeline Latency Breakdown: CPU vs GPU')

        self.bar_figure.subplots_adjust(bottom=0.22)
        self.bar_canvas.draw()

    def _update_statistics(self):
        """Update the statistics table."""
        recon_times = self._timing_data.get('recon_times_ms', [])
        denoise_cpu = self._timing_data.get('denoise_times_cpu_ms', [])
        denoise_gpu = self._timing_data.get('denoise_times_gpu_ms', [])
        t_acq = self._timing_data.get('t_acq_ms', 0)

        def compute_stats(data):
            if not data:
                return ["-"] * 7
            arr = np.array(data)
            return [
                f"{np.mean(arr):.2f}",
                f"{np.std(arr):.2f}",
                f"{np.min(arr):.2f}",
                f"{np.max(arr):.2f}",
                f"{np.percentile(arr, 25):.2f}",
                f"{np.percentile(arr, 50):.2f}",
                f"{np.percentile(arr, 75):.2f}"
            ]

        # Reconstruction stats
        recon_stats = compute_stats(recon_times)
        for i, label in enumerate(self.stats_labels['t_recon']):
            label.setText(recon_stats[i])

        # Inference CPU stats
        cpu_stats = compute_stats(denoise_cpu)
        for i, label in enumerate(self.stats_labels['t_inf_cpu']):
            label.setText(cpu_stats[i])

        # Inference GPU stats
        gpu_stats = compute_stats(denoise_gpu)
        for i, label in enumerate(self.stats_labels['t_inf_gpu']):
            label.setText(gpu_stats[i])

        # Total CPU stats
        if recon_times and denoise_cpu:
            total_cpu = [t_acq + r + d for r, d in zip(recon_times, denoise_cpu)]
            total_cpu_stats = compute_stats(total_cpu)
        else:
            total_cpu_stats = ["-"] * 7
        for i, label in enumerate(self.stats_labels['t_total_cpu']):
            label.setText(total_cpu_stats[i])

        # Total GPU stats
        if recon_times and denoise_gpu:
            total_gpu = [t_acq + r + d for r, d in zip(recon_times, denoise_gpu)]
            total_gpu_stats = compute_stats(total_gpu)
        else:
            total_gpu_stats = ["-"] * 7
        for i, label in enumerate(self.stats_labels['t_total_gpu']):
            label.setText(total_gpu_stats[i])

    def _show_save_menu(self, pos, widget, figure, name):
        """Show context menu to save a specific figure."""
        menu = QMenu(self)
        save_png = menu.addAction("Save as PNG...")
        save_pdf = menu.addAction("Save as PDF...")

        action = menu.exec_(widget.mapToGlobal(pos))

        if action == save_png:
            self._save_figure(figure, name, "png")
        elif action == save_pdf:
            self._save_figure(figure, name, "pdf")

    def _save_figure(self, figure, name, ext):
        """Save a specific figure to file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, f"Save {name}", f"{name}.{ext}",
            f"{ext.upper()} Files (*.{ext});;All Files (*.*)"
        )

        if file_path:
            try:
                figure.savefig(file_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Figure saved to {file_path}")
            except Exception as e:
                self.logger.error(f"Failed to save figure: {e}")

    def _on_export(self, format_type):
        """Export all figures to files."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Report", f"timing_report.{format_type}",
            f"{format_type.upper()} Files (*.{format_type});;All Files (*.*)"
        )

        if not file_path:
            return

        base_path = file_path.rsplit('.', 1)[0] if '.' in file_path else file_path

        try:
            self.curves_figure.savefig(f"{base_path}_time_per_image.{format_type}",
                                       dpi=300, bbox_inches='tight')
            self.hist_figure.savefig(f"{base_path}_distribution.{format_type}",
                                     dpi=300, bbox_inches='tight')
            self.bar_figure.savefig(f"{base_path}_breakdown.{format_type}",
                                    dpi=300, bbox_inches='tight')
            self.logger.info(f"Report exported to {base_path}_*.{format_type}")
        except Exception as e:
            self.logger.error(f"Failed to export report: {e}")
