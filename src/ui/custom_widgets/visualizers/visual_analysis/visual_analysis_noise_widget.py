"""Widget to visualize noise analysis metrics and quality comparisons."""
import logging
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QGraphicsScene, QLabel, QSizePolicy
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib import cm

# Import generated UI class
from ui.custom_widgets.visualizers.visual_analysis.visual_analysis_noise import Ui_Visual_Noise_Analysis

class VisualNoiseAnalysisWidget(QtWidgets.QWidget, Ui_Visual_Noise_Analysis):
    """Widget for displaying noise analysis results with quality metrics."""
    def __init__(self, parent=None, logger=None):
        super().__init__(parent)
        self.setupUi(self)

        # Logger
        if logger is not None:
            self.logger = logger.getChild("VisualNoiseAnalysisWidget")
        else:
            self.logger = logging.getLogger("ASPIR.VisualNoiseAnalysisWidget")
        self.logger.setLevel(logging.DEBUG)
        self.logger.debug("Initializing VisualNoiseAnalysisWidget")

        # Thermal colormap
        self.cmap = cm.get_cmap('hot')

        # Create scenes
        self._scene_orig  = QGraphicsScene(self)
        self._scene_ruido = QGraphicsScene(self)
        self._scene_recon = QGraphicsScene(self)
        self._scene_plot  = QGraphicsScene(self)

        self.preview_image_original.setScene(self._scene_orig)
        self.preview_image_noise.setScene(self._scene_ruido)
        self.preview_image_reconstructed.setScene(self._scene_recon)
        self.plot_bars_psnr_and_ssim.setScene(self._scene_plot)
        self.logger.debug("Graphics scenes created and assigned")

        # Placeholder if no images
        self._placeholder = QLabel("No images available", self)
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._placeholder.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.verticalLayout_2.addWidget(self._placeholder)
        self._placeholder.hide()

        # Connect slider
        self.slider_value.valueChanged.connect(self._on_slider_changed)
        self.slider_index_label.setText("Index: 0")

        # Internal storage
        self._orig_images  = []
        self._noise_images = []
        self._recon_images = []
        self.psnr_noise    = []
        self.ssim_noise    = []
        self.psnr_recon    = []
        self.ssim_recon    = []

    def update_info(self,
                    num_images: int,
                    img_size: int,
                    dataset_type: str,
                    mask_type: str,
                    postprocessor_type: str):
        """
        Update metadata labels.
        """
        self.logger.debug(
            "Updating info: num_images=%d, img_size=%d, dataset=%s, mask=%s, postproc=%s",
            num_images, img_size, dataset_type, mask_type, postprocessor_type
        )
        self.number_images.setText(str(num_images))
        self.image_size.setText(f"{img_size}×{img_size}")
        self.dataset_type.setText(dataset_type)
        self.mask_type.setText(mask_type)
        self.postprocessor_type.setText(postprocessor_type)

        max_idx = max(0, num_images - 1)
        self.slider_value.setRange(0, max_idx)
        self.slider_index_label.setText(f"Index: {self.slider_value.value()}")

    def set_images_and_metrics(self,
                               orig_images, noise_images, recon_images,
                               psnr_noise, ssim_noise,
                               psnr_recon, ssim_recon):
        """
        Store lists and metrics, show/hide widgets, init slider and display first.
        """
        n = len(orig_images)
        self.logger.info("Setting images and metrics: n=%d", n)
        self._orig_images  = list(orig_images)
        self._noise_images = list(noise_images)
        self._recon_images = list(recon_images or [])
        self.psnr_noise    = psnr_noise
        self.ssim_noise    = ssim_noise
        self.psnr_recon    = psnr_recon
        self.ssim_recon    = ssim_recon

        if n == 0:
            self.logger.warning("No images provided, showing placeholder")
            self._placeholder.show()
            for w in (self.preview_image_original,
                      self.preview_image_noise,
                      self.preview_image_reconstructed,
                      self.plot_bars_psnr_and_ssim,
                      self.slider_value,
                      self.slider_index_label):
                w.hide()
            return

        self._placeholder.hide()
        for w in (self.preview_image_original,
                  self.preview_image_noise,
                  self.preview_image_reconstructed,
                  self.plot_bars_psnr_and_ssim,
                  self.slider_value,
                  self.slider_index_label):
            w.show()

        self.slider_value.setRange(0, n - 1)
        self.slider_value.setValue(0)
        self._on_slider_changed(0)
        self.logger.debug("Images and metrics set, slider initialized")

    def _add_to_scene(self, arr: np.ndarray, scene: QGraphicsScene, view: QtWidgets.QGraphicsView):
        """
        Normalize, apply thermal colormap, convert to QPixmap,
        and add to the graphics scene.
        """
        scene.clear()
        arr = np.array(arr, copy=False)
        # Scale to [0, 1]
        if arr.ndim == 0:
            norm = np.zeros_like(arr, dtype=float)
        else:
            amin, amax = arr.min(), arr.max()
            norm = (arr - amin) / (amax - amin) if amax > amin else np.zeros_like(arr, dtype=float)

        # Apply thermal colormap
        rgba = self.cmap(norm)               # (h, w) -> (h, w, 4)
        rgb  = (rgba[..., :3] * 255).astype(np.uint8)

        h, w = rgb.shape[:2]
        data = rgb.tobytes()
        bytes_per_line = 3 * w
        qimg = QImage(data, w, h, bytes_per_line, QImage.Format_RGB888)

        pix = QPixmap.fromImage(qimg)
        vw = view.viewport().width()
        vh = view.viewport().height()
        pix = pix.scaled(vw, vh, Qt.KeepAspectRatio, Qt.FastTransformation)

        scene.addPixmap(pix)
        scene.setSceneRect(0, 0, pix.width(), pix.height())
        self.logger.debug("Added image to scene (%dx%d)", pix.width(), pix.height())

    def _plot_bar_chart(self, idx: int):
        """
        Draw a dual-axis bar chart:
         - PSNR (dB) on left Y at x=0, y limits [0, max_dataset_psnr]
         - SSIM on right Y at x=1, y limits [0, 1]
        """
        self.logger.debug("Plotting bar chart for index %d", idx)
        self._scene_plot.clear()

        all_psnr = np.array(self.psnr_noise + self.psnr_recon)
        max_psnr = float(np.max(all_psnr)) if all_psnr.size > 0 else 1.0

        fig = Figure(figsize=(3,3), dpi=100)
        canvas = FigureCanvasAgg(fig)
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()

        x = np.array([0, 1])
        w = 0.35

        b1 = ax1.bar(x[0] - w/2, self.psnr_noise[idx], width=w)
        b2 = ax1.bar(x[0] + w/2, self.psnr_recon[idx], width=w)
        ax1.set_ylabel('PSNR (dB)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(['PSNR', 'SSIM'])
        ax1.set_ylim(0, max_psnr * 1.05)

        ax2.bar(x[1] - w/2, self.ssim_noise[idx], width=w)
        ax2.bar(x[1] + w/2, self.ssim_recon[idx], width=w)
        ax2.set_ylabel('SSIM')
        ax2.set_ylim(0, 1)

        noise_patch = b1[0]
        recon_patch = b2[0]
        ax1.legend([noise_patch, recon_patch], ['Noisy', 'Reconstructed'], loc='upper center')

        fig.tight_layout()
        canvas.draw()
        buf = canvas.buffer_rgba()
        width, height = canvas.get_width_height()
        qimg = QImage(buf, width, height, QImage.Format_RGBA8888)
        pix = QPixmap.fromImage(qimg)
        pix = pix.scaled(self.plot_bars_psnr_and_ssim.viewport().size(),
                         Qt.KeepAspectRatio, Qt.SmoothTransformation)

        self._scene_plot.addPixmap(pix)
        self._scene_plot.setSceneRect(0, 0, pix.width(), pix.height())
        self.logger.info("Bar chart rendered for index %d", idx)

    def _on_slider_changed(self, idx: int):
        """
        Update index label, display images and bar chart.
        """
        self.logger.debug("Slider changed: idx=%d", idx)
        self.slider_index_label.setText(f"Index: {idx}")
        if not (0 <= idx < len(self._orig_images)):
            self.logger.warning("Index %d out of range", idx)
            return

        self._add_to_scene(self._orig_images[idx],  self._scene_orig,  self.preview_image_original)
        self._add_to_scene(self._noise_images[idx], self._scene_ruido, self.preview_image_noise)
        self._add_to_scene(self._recon_images[idx], self._scene_recon, self.preview_image_reconstructed)
        self._plot_bar_chart(idx)
