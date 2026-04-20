"""Widget to display noise analysis metrics including PSNR, SSIM, and LPIPS."""
import logging
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QGraphicsScene
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import pyqtSignal, Qt

# Import the generated UI
from ui.custom_widgets.noise_analysis_metric.noise_analysis_metrics import Ui_Analisis_Ruido_metricas


class NoiseAnalysisMetricsWidget(QtWidgets.QWidget, Ui_Analisis_Ruido_metricas):
    """
    Widget to display PSNR/SSIM comparison for noise vs reconstruction.
    Updated to display LPIPS averages.
    """

    # Configuration constants
    TITLE_FONT    = 20
    LABEL_FONT    = 18
    TICK_FONT     = 18
    LEGEND_FONT   = 18
    WIDTH_FACTOR  = 1.2
    DPI_WIDGET    = 100
    HIGHRES_DPI   = 300
    HIGHRES_SIZE  = (8, 6)

    runAnalysisRequested = pyqtSignal()

    def __init__(self, simulation, parent=None, logger=None):
        super().__init__(parent)
        self.setupUi(self)
        self.simulation = simulation

        # Logger
        if logger is not None:
            self.logger = logger.getChild("NoiseAnalysisMetricsWidget")
        else:
            self.logger = logging.getLogger("ASPIR.NoiseAnalysisMetricsWidget")
        self.logger.setLevel(logging.DEBUG)

        # Scenes
        self.scene_psnr      = QGraphicsScene(self)
        self.scene_psnr_hist = QGraphicsScene(self)
        self.scene_ssim      = QGraphicsScene(self)
        self.scene_ssim_hist = QGraphicsScene(self)
        self.plot_psnr.setScene(self.scene_psnr)
        self.plot_psnr_histograma.setScene(self.scene_psnr_hist)
        self.plot_ssim.setScene(self.scene_ssim)
        self.plot_ssim_histograma.setScene(self.scene_ssim_hist)

        # Button
        self.analyze_noise_button.clicked.connect(self.runAnalysisRequested.emit)

    def setOverallMetrics(self, pn_mean, sn_mean, ln_mean, pr_mean, sr_mean, lr_mean):
        """
        Display average PSNR/SSIM/LPIPS in individual labels.
        Arguments:
            pn_mean, sn_mean, ln_mean: PSNR, SSIM, LPIPS for noisy images
            pr_mean, sr_mean, lr_mean: PSNR, SSIM, LPIPS for reconstructed images
        """
        # Noise (Input) Metrics
        self.psnr_avg_ruido_value.setText(f"{pn_mean:.2f}")
        self.ssim_avg_ruido_value.setText(f"{sn_mean:.4f}")
        self.lpips_avg_ruido_value.setText(f"{ln_mean:.4f}")

        # Reconstruction (Output) Metrics
        self.psnr_avg_reconstruido_value.setText(f"{pr_mean:.2f}")
        self.ssim_avg_reconstruido_value.setText(f"{sr_mean:.4f}")
        self.lpips_avg_reconstruido_value.setText(f"{lr_mean:.4f}")

    def setPerImageMetrics(self, pn_vals, sn_vals, ln_vals, pr_vals, sr_vals, lr_vals):
        """
        Render comparison plots and histograms.
        Note: LPIPS lists (ln_vals, lr_vals) are accepted but not plotted as there are no UI widgets for them.
        """
        n = len(pn_vals)
        self.logger.info(f"Rendering per-image metrics for {n} images")

        # Helper for curves
        def render_plot(x1, x2, title, xlabel, ylabel, scene, view):
            buf = BytesIO()
            w, h = view.viewport().width(), view.viewport().height()

            # Embedded plot
            fig = plt.figure(
                figsize=((w / self.DPI_WIDGET) * self.WIDTH_FACTOR, h / self.DPI_WIDGET),
                dpi=self.DPI_WIDGET
            )
            ax = fig.add_subplot(111)
            ax.plot(x1, label='Raw reconstruction')
            ax.plot(x2, label='Denoised')
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.tick_params(axis='both')
            if ylabel.lower() == 'ssim':
                ax.set_ylim(0, 1)
            ax.legend(loc='best')
            fig.savefig(buf, format='png', dpi=self.DPI_WIDGET,
                        bbox_inches='tight', pad_inches=0.2)
            plt.close(fig)

            # Mostrar en widget
            buf.seek(0)
            img = QImage.fromData(buf.getvalue())
            pix = QPixmap.fromImage(img).scaled(w, h,
                Qt.KeepAspectRatio, Qt.SmoothTransformation)
            scene.clear()
            scene.addPixmap(pix)

            # –– Plot de alta calidad ––
            fig2 = plt.figure(figsize=self.HIGHRES_SIZE, dpi=self.HIGHRES_DPI)
            ax2 = fig2.add_subplot(111)
            ax2.plot(x1, label='Raw reconstruction')
            ax2.plot(x2, label='Denoised')
            ax2.set_title(title, fontsize=self.TITLE_FONT + 4)
            ax2.set_xlabel(xlabel, fontsize=self.LABEL_FONT + 2)
            ax2.set_ylabel(ylabel, fontsize=self.LABEL_FONT + 2)
            ax2.tick_params(axis='both', labelsize=self.TICK_FONT + 2)
            if ylabel.lower() == 'ssim':
                ax2.set_ylim(0, 1)
            ax2.legend(loc='best', fontsize=self.LEGEND_FONT)
            fig2.tight_layout()
            filename = f"{title.replace(' ', '_')}.png"
            fig2.savefig(filename, dpi=self.HIGHRES_DPI,
                         bbox_inches='tight', pad_inches=0.2)

        # Helper para histogramas
        def render_histogram_bars(data1, data2, title, xlabel, ylabel, scene, view):
            bins = np.linspace(min(min(data1), min(data2)),
                               max(max(data1), max(data2)), 20)
            h1, _ = np.histogram(data1, bins=bins)
            h2, _ = np.histogram(data2, bins=bins)
            centers = (bins[:-1] + bins[1:]) / 2
            width = centers[1] - centers[0]

            buf = BytesIO()
            w, h = view.viewport().width(), view.viewport().height()

            # –– Histograma embebido ––
            fig = plt.figure(
                figsize=((w / self.DPI_WIDGET) * self.WIDTH_FACTOR, h / self.DPI_WIDGET),
                dpi=self.DPI_WIDGET
            )
            ax = fig.add_subplot(111)
            ax.bar(centers - width/2, h1, width, alpha=0.7, label='Raw reconstruction')
            ax.bar(centers + width/2, h2, width, alpha=0.7, label='Denoised')
            ax2_title = title
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.tick_params(axis='both')
            ax.legend(loc='best')
            fig.savefig(buf, format='png', dpi=self.DPI_WIDGET,
                        bbox_inches='tight', pad_inches=0.2)
            plt.close(fig)

            # Mostrar en widget
            buf.seek(0)
            img = QImage.fromData(buf.getvalue())
            pix = QPixmap.fromImage(img).scaled(w, h,
                Qt.KeepAspectRatio, Qt.SmoothTransformation)
            scene.clear()
            scene.addPixmap(pix)

            # –– Histograma de alta calidad ––
            fig2 = plt.figure(figsize=self.HIGHRES_SIZE, dpi=self.HIGHRES_DPI)
            ax2 = fig2.add_subplot(111)
            ax2.bar(centers - width/2, h1, width, alpha=0.7, label='Raw reconstruction')
            ax2.bar(centers + width/2, h2, width, alpha=0.7, label='Denoised')
            ax2.set_title(ax2_title, fontsize=self.TITLE_FONT + 4)
            ax2.set_xlabel(xlabel, fontsize=self.LABEL_FONT + 2)
            ax2.set_ylabel(ylabel, fontsize=self.LABEL_FONT + 2)
            ax2.tick_params(axis='both', labelsize=self.TICK_FONT + 2)
            ax2.legend(loc='best', fontsize=self.LEGEND_FONT)
            fig2.tight_layout()
            filename = f"{ax2_title.replace(' ', '_')}.png"
            fig2.savefig(filename, dpi=self.HIGHRES_DPI,
                         bbox_inches='tight', pad_inches=0.2)

        # Llamadas a los helpers (Solo PSNR y SSIM)
        render_plot(
            pn_vals, pr_vals,
            title='PSNR per image', xlabel='Image index', ylabel='PSNR',
            scene=self.scene_psnr, view=self.plot_psnr
        )
        render_histogram_bars(
            pn_vals, pr_vals,
            title='Histogram PSNR', xlabel='PSNR Value', ylabel='Frequency',
            scene=self.scene_psnr_hist, view=self.plot_psnr_histograma
        )
        render_plot(
            sn_vals, sr_vals,
            title='SSIM per image', xlabel='Image index', ylabel='SSIM',
            scene=self.scene_ssim, view=self.plot_ssim
        )
        render_histogram_bars(
            sn_vals, sr_vals,
            title='Histogram SSIM', xlabel='SSIM Value', ylabel='Frequency',
            scene=self.scene_ssim_hist, view=self.plot_ssim_histograma
        )

        self.logger.info("All per-image plots (PSNR/SSIM) and histograms rendered")