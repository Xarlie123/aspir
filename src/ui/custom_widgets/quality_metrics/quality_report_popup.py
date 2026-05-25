"""Popup dialog for displaying quality report with graphs and metrics."""
import logging
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QGridLayout, QFileDialog, QSizePolicy, QScrollArea,
    QWidget, QFrame
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont


class QualityReportPopup(QDialog):
    """
    Popup dialog showing quality metrics report with:
    - Line chart (image index vs selected metrics)
    - Histogram (metric values vs frequency)
    - Average metrics table for noisy and denoised images
    - Save button for plots
    """

    def __init__(self, parent=None, logger=None):
        super().__init__(parent)
        self.setWindowTitle("Quality Report")
        self.setMinimumSize(1000, 700)
        self.resize(1200, 800)

        if logger:
            self.logger = logger.getChild("QualityReportPopup")
        else:
            self.logger = logging.getLogger("ASPIR.QualityReportPopup")

        # Data storage
        self.metrics_data = {}
        self.selected_metrics = []

        self._setup_ui()

    def _setup_ui(self):
        """Setup the popup UI layout."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Title
        title = QLabel("<h2>Quality Metrics Report</h2>")
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)

        # Scroll area for content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setSpacing(15)

        # Metrics summary section
        self.metrics_group = QGroupBox("Average Metrics Summary")
        metrics_layout = QGridLayout(self.metrics_group)
        metrics_layout.setSpacing(10)

        # Headers
        header_font = QFont()
        header_font.setBold(True)

        metrics_layout.addWidget(QLabel(""), 0, 0)
        noisy_header = QLabel("Noisy (Input)")
        noisy_header.setFont(header_font)
        noisy_header.setAlignment(Qt.AlignCenter)
        metrics_layout.addWidget(noisy_header, 0, 1)
        recon_header = QLabel("Denoised (Output)")
        recon_header.setFont(header_font)
        recon_header.setAlignment(Qt.AlignCenter)
        metrics_layout.addWidget(recon_header, 0, 2)

        # Metric rows (will be populated dynamically)
        # PSNR - higher is better
        self.psnr_label = QLabel("PSNR (dB) \u2191:")  # ↑ arrow
        self.psnr_label.setFont(header_font)
        self.psnr_label.setToolTip("Higher is better")
        self.psnr_noisy_value = QLabel("-")
        self.psnr_noisy_value.setAlignment(Qt.AlignCenter)
        self.psnr_recon_value = QLabel("-")
        self.psnr_recon_value.setAlignment(Qt.AlignCenter)
        metrics_layout.addWidget(self.psnr_label, 1, 0)
        metrics_layout.addWidget(self.psnr_noisy_value, 1, 1)
        metrics_layout.addWidget(self.psnr_recon_value, 1, 2)

        # SSIM - higher is better
        self.ssim_label = QLabel("SSIM \u2191:")  # ↑ arrow
        self.ssim_label.setFont(header_font)
        self.ssim_label.setToolTip("Higher is better")
        self.ssim_noisy_value = QLabel("-")
        self.ssim_noisy_value.setAlignment(Qt.AlignCenter)
        self.ssim_recon_value = QLabel("-")
        self.ssim_recon_value.setAlignment(Qt.AlignCenter)
        metrics_layout.addWidget(self.ssim_label, 2, 0)
        metrics_layout.addWidget(self.ssim_noisy_value, 2, 1)
        metrics_layout.addWidget(self.ssim_recon_value, 2, 2)

        # LPIPS - lower is better
        self.lpips_label = QLabel("LPIPS \u2193:")  # ↓ arrow
        self.lpips_label.setFont(header_font)
        self.lpips_label.setToolTip("Lower is better")
        self.lpips_noisy_value = QLabel("-")
        self.lpips_noisy_value.setAlignment(Qt.AlignCenter)
        self.lpips_recon_value = QLabel("-")
        self.lpips_recon_value.setAlignment(Qt.AlignCenter)
        metrics_layout.addWidget(self.lpips_label, 3, 0)
        metrics_layout.addWidget(self.lpips_noisy_value, 3, 1)
        metrics_layout.addWidget(self.lpips_recon_value, 3, 2)

        content_layout.addWidget(self.metrics_group)

        # Charts section
        charts_layout = QHBoxLayout()
        charts_layout.setSpacing(15)

        # Line chart
        line_group = QGroupBox("Metrics per Image")
        line_layout = QVBoxLayout(line_group)
        self.line_figure = Figure(figsize=(5, 4), dpi=100)
        self.line_canvas = FigureCanvas(self.line_figure)
        self.line_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        line_layout.addWidget(self.line_canvas)
        charts_layout.addWidget(line_group)

        # Histogram
        hist_group = QGroupBox("Metrics Histogram")
        hist_layout = QVBoxLayout(hist_group)
        self.hist_figure = Figure(figsize=(5, 4), dpi=100)
        self.hist_canvas = FigureCanvas(self.hist_figure)
        self.hist_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        hist_layout.addWidget(self.hist_canvas)
        charts_layout.addWidget(hist_group)

        content_layout.addLayout(charts_layout)

        scroll.setWidget(content_widget)
        main_layout.addWidget(scroll, 1)

        # Buttons
        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch()

        self.save_button = QPushButton("Save Plots")
        self.save_button.clicked.connect(self._on_save_plots)
        buttons_layout.addWidget(self.save_button)

        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close)
        buttons_layout.addWidget(self.close_button)

        main_layout.addLayout(buttons_layout)

    def set_data(self, metrics_data: dict, selected_metrics: list):
        """
        Set the metrics data and update the display.

        Args:
            metrics_data: Dictionary with keys like 'psnr_noisy', 'psnr_recon', etc.
                         Each value is a list of per-image values.
            selected_metrics: List of selected metric names ('psnr', 'ssim', 'lpips')
        """
        self.metrics_data = metrics_data
        self.selected_metrics = selected_metrics

        self._update_metrics_table()
        self._update_line_chart()
        self._update_histogram()

    def _update_metrics_table(self):
        """Update the metrics summary table."""
        # Show/hide rows based on selection
        show_psnr = 'psnr' in self.selected_metrics
        show_ssim = 'ssim' in self.selected_metrics
        show_lpips = 'lpips' in self.selected_metrics

        self.psnr_label.setVisible(show_psnr)
        self.psnr_noisy_value.setVisible(show_psnr)
        self.psnr_recon_value.setVisible(show_psnr)

        self.ssim_label.setVisible(show_ssim)
        self.ssim_noisy_value.setVisible(show_ssim)
        self.ssim_recon_value.setVisible(show_ssim)

        self.lpips_label.setVisible(show_lpips)
        self.lpips_noisy_value.setVisible(show_lpips)
        self.lpips_recon_value.setVisible(show_lpips)

        # Update values
        if show_psnr and 'psnr_noisy' in self.metrics_data:
            psnr_n = np.mean(self.metrics_data['psnr_noisy'])
            psnr_r = np.mean(self.metrics_data['psnr_recon'])
            self.psnr_noisy_value.setText(f"{psnr_n:.2f}")
            self.psnr_recon_value.setText(f"{psnr_r:.2f}")

        if show_ssim and 'ssim_noisy' in self.metrics_data:
            ssim_n = np.mean(self.metrics_data['ssim_noisy'])
            ssim_r = np.mean(self.metrics_data['ssim_recon'])
            self.ssim_noisy_value.setText(f"{ssim_n:.4f}")
            self.ssim_recon_value.setText(f"{ssim_r:.4f}")

        if show_lpips and 'lpips_noisy' in self.metrics_data:
            lpips_n = np.mean(self.metrics_data['lpips_noisy'])
            lpips_r = np.mean(self.metrics_data['lpips_recon'])
            self.lpips_noisy_value.setText(f"{lpips_n:.4f}")
            self.lpips_recon_value.setText(f"{lpips_r:.4f}")

    def _update_line_chart(self):
        """Update the line chart with per-image metrics using separate Y-axes per metric type."""
        self.line_figure.clear()

        if not self.metrics_data or not self.selected_metrics:
            ax = self.line_figure.add_subplot(111)
            ax.text(0.5, 0.5, "No data available", ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
            self.line_canvas.draw()
            return

        # Get number of images
        n_images = 0
        for key in self.metrics_data:
            if self.metrics_data[key] is not None:
                n_images = len(self.metrics_data[key])
                break

        if n_images == 0:
            ax = self.line_figure.add_subplot(111)
            ax.text(0.5, 0.5, "No data available", ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
            self.line_canvas.draw()
            return

        x = np.arange(n_images)
        n_metrics = len(self.selected_metrics)

        # Colors for noisy and denoised
        color_noisy = '#1f77b4'
        color_recon = '#2ca02c'

        # Create subplots - one row per metric type
        for i, metric in enumerate(self.selected_metrics):
            ax = self.line_figure.add_subplot(n_metrics, 1, i + 1)

            noisy_key = f'{metric}_noisy'
            recon_key = f'{metric}_recon'

            if noisy_key in self.metrics_data and self.metrics_data[noisy_key] is not None:
                ax.plot(x, self.metrics_data[noisy_key],
                       label='Noisy', color=color_noisy,
                       linestyle='--', alpha=0.7, marker='o', markersize=3)

            if recon_key in self.metrics_data and self.metrics_data[recon_key] is not None:
                ax.plot(x, self.metrics_data[recon_key],
                       label='Denoised', color=color_recon,
                       marker='s', markersize=3)

            # Set Y-axis label and limits based on metric type
            if metric == 'psnr':
                ax.set_ylabel('PSNR (dB)', fontsize=9)
                ax.set_title('PSNR per Image (higher is better)', fontsize=10)
            elif metric == 'ssim':
                ax.set_ylabel('SSIM', fontsize=9)
                ax.set_ylim(0, 1)
                ax.set_title('SSIM per Image (higher is better)', fontsize=10)
            elif metric == 'lpips':
                ax.set_ylabel('LPIPS', fontsize=9)
                ax.set_ylim(0, 1)
                ax.set_title('LPIPS per Image (lower is better)', fontsize=10)
            else:
                ax.set_title(f'{metric.upper()} per Image', fontsize=10)
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)

            # Only show x-axis label on bottom subplot
            if i == n_metrics - 1:
                ax.set_xlabel('Image Index', fontsize=9)
            else:
                ax.set_xticklabels([])

        self.line_figure.tight_layout()
        self.line_canvas.draw()

    def _update_histogram(self):
        """Update the histogram with metric value distribution using separate subplots per metric type."""
        self.hist_figure.clear()

        if not self.metrics_data or not self.selected_metrics:
            ax = self.hist_figure.add_subplot(111)
            ax.text(0.5, 0.5, "No data available", ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
            self.hist_canvas.draw()
            return

        n_metrics = len(self.selected_metrics)
        color_noisy = '#1f77b4'
        color_recon = '#2ca02c'

        # Create subplots - one row per metric type
        for i, metric in enumerate(self.selected_metrics):
            ax = self.hist_figure.add_subplot(n_metrics, 1, i + 1)

            noisy_key = f'{metric}_noisy'
            recon_key = f'{metric}_recon'

            data_to_plot = []
            labels = []
            colors = []

            if noisy_key in self.metrics_data and self.metrics_data[noisy_key] is not None:
                data_to_plot.append(self.metrics_data[noisy_key])
                labels.append('Noisy')
                colors.append(color_noisy)

            if recon_key in self.metrics_data and self.metrics_data[recon_key] is not None:
                data_to_plot.append(self.metrics_data[recon_key])
                labels.append('Denoised')
                colors.append(color_recon)

            if data_to_plot:
                ax.hist(data_to_plot, bins=15, label=labels, color=colors, alpha=0.7)

            # Set X-axis label based on metric type
            if metric == 'psnr':
                ax.set_xlabel('PSNR (dB)', fontsize=9)
                ax.set_title('PSNR Distribution (higher is better)', fontsize=10)
            elif metric == 'ssim':
                ax.set_xlabel('SSIM', fontsize=9)
                ax.set_xlim(0, 1)
                ax.set_title('SSIM Distribution (higher is better)', fontsize=10)
            elif metric == 'lpips':
                ax.set_xlabel('LPIPS', fontsize=9)
                ax.set_xlim(0, 1)
                ax.set_title('LPIPS Distribution (lower is better)', fontsize=10)
            else:
                ax.set_title(f'{metric.upper()} Distribution', fontsize=10)

            ax.set_ylabel('Frequency', fontsize=9)
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)

        self.hist_figure.tight_layout()
        self.hist_canvas.draw()

    def _on_save_plots(self):
        """Save the plots to files."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Plots", "",
            "PNG Files (*.png);;PDF Files (*.pdf);;All Files (*.*)"
        )

        if not file_path:
            return

        # Save line chart
        base_path = file_path.rsplit('.', 1)[0] if '.' in file_path else file_path
        ext = file_path.rsplit('.', 1)[1] if '.' in file_path else 'png'

        try:
            self.line_figure.savefig(f"{base_path}_line_chart.{ext}",
                                     dpi=300, bbox_inches='tight')
            self.hist_figure.savefig(f"{base_path}_histogram.{ext}",
                                     dpi=300, bbox_inches='tight')
            self.logger.info(f"Plots saved to {base_path}_*.{ext}")
        except Exception as e:
            self.logger.error(f"Failed to save plots: {e}")
