import logging
import numpy as np
from typing import Sequence
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QTimer, QEvent
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QGraphicsScene, QLabel, QSizePolicy, QDialog,
                              QVBoxLayout, QHBoxLayout, QMenu, QFileDialog)
from matplotlib import cm

from ui.custom_widgets.visualizers.visual_postprocessor.visual_postprocessor import Ui_Visual_Postprocessor
from ui.custom_widgets.common.multi_phase_progress import MultiPhaseProgressWidget
from ui._4_postprocessor.postprocessor_worker import PostprocesadoWorker


class ZoomableImageLabel(QLabel):
    """QLabel that supports mouse wheel zoom and panning."""

    def __init__(self, pixmap: QPixmap, parent=None):
        super().__init__(parent)
        self._original_pixmap = pixmap
        self._zoom_factor = 1.0
        self._min_zoom = 0.5
        self._max_zoom = 5.0
        self._pan_start = None
        self._scroll_area = None

        self.setPixmap(pixmap)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("border: 1px solid #ccc; background-color: #f5f5f5;")
        self.setCursor(Qt.OpenHandCursor)

    def set_scroll_area(self, scroll_area):
        """Set the parent scroll area for panning."""
        self._scroll_area = scroll_area

    def wheelEvent(self, event):
        """Handle mouse wheel for zooming."""
        delta = event.angleDelta().y()
        if delta > 0:
            self._zoom_factor = min(self._max_zoom, self._zoom_factor * 1.15)
        else:
            self._zoom_factor = max(self._min_zoom, self._zoom_factor / 1.15)

        new_size = self._original_pixmap.size() * self._zoom_factor
        scaled = self._original_pixmap.scaled(new_size, Qt.KeepAspectRatio, Qt.FastTransformation)
        self.setPixmap(scaled)
        event.accept()

    def mousePressEvent(self, event):
        """Start panning on mouse press."""
        if event.button() == Qt.LeftButton:
            self._pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()

    def mouseMoveEvent(self, event):
        """Pan the image while dragging."""
        if self._pan_start and self._scroll_area:
            delta = event.pos() - self._pan_start
            h_bar = self._scroll_area.horizontalScrollBar()
            v_bar = self._scroll_area.verticalScrollBar()
            h_bar.setValue(h_bar.value() - delta.x())
            v_bar.setValue(v_bar.value() - delta.y())
            event.accept()

    def mouseReleaseEvent(self, event):
        """End panning on mouse release."""
        if event.button() == Qt.LeftButton:
            self._pan_start = None
            self.setCursor(Qt.OpenHandCursor)
            event.accept()


class CombinedImagePopupDialog(QDialog):
    """Dialog to display all three images (Ground-Truth, Noisy, Denoised) side by side with zoom."""

    def __init__(self, orig_pixmap: QPixmap, noisy_pixmap: QPixmap, denoised_pixmap: QPixmap, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Image Comparison (Scroll to zoom, drag to pan)")
        self.setModal(True)

        # Calculate max size per image (fit 3 images + margins in 85% of screen)
        screen = QtWidgets.QApplication.primaryScreen().geometry()
        max_total_width = screen.width() * 0.85
        max_height = screen.height() * 0.7
        max_img_width = (max_total_width - 60) / 3  # 60px for margins/spacing
        max_img_size = min(max_img_width, max_height - 100)  # 100px for labels + hint

        # Scale pixmaps for initial display (use FastTransformation for pixel-perfect display)
        def scale_pixmap(pix):
            if pix.width() > max_img_size or pix.height() > max_img_size:
                return pix.scaled(int(max_img_size), int(max_img_size), Qt.KeepAspectRatio, Qt.FastTransformation)
            return pix

        # Store original pixmaps for zoom
        self._orig_pixmaps = [orig_pixmap, noisy_pixmap, denoised_pixmap]

        orig_scaled = scale_pixmap(orig_pixmap)
        noisy_scaled = scale_pixmap(noisy_pixmap)
        denoised_scaled = scale_pixmap(denoised_pixmap)

        # Create layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Images row
        images_layout = QHBoxLayout()
        images_layout.setSpacing(15)

        # Helper to create image column with label and zoomable scroll area
        def create_image_column(title: str, pixmap: QPixmap):
            col_layout = QVBoxLayout()
            col_layout.setSpacing(5)

            title_label = QLabel(f"<b>{title}</b>")
            title_label.setAlignment(Qt.AlignCenter)
            col_layout.addWidget(title_label)

            # Create scroll area for panning
            scroll_area = QtWidgets.QScrollArea()
            scroll_area.setWidgetResizable(False)
            scroll_area.setAlignment(Qt.AlignCenter)
            scroll_area.setMinimumSize(int(max_img_size) + 20, int(max_img_size) + 20)
            scroll_area.setStyleSheet("QScrollArea { border: 1px solid #ccc; background-color: #f5f5f5; }")

            # Create zoomable label
            img_label = ZoomableImageLabel(pixmap)
            img_label.set_scroll_area(scroll_area)
            scroll_area.setWidget(img_label)

            col_layout.addWidget(scroll_area)
            return col_layout

        images_layout.addLayout(create_image_column("Ground-Truth", orig_scaled))
        images_layout.addLayout(create_image_column("Noisy", noisy_scaled))
        images_layout.addLayout(create_image_column("Denoised", denoised_scaled))

        main_layout.addLayout(images_layout)

        # Hint to close
        hint_label = QLabel("Scroll to zoom • Drag to pan • Press Escape to close")
        hint_label.setAlignment(Qt.AlignCenter)
        hint_label.setStyleSheet("color: gray; font-size: 10px;")
        main_layout.addWidget(hint_label)

        # Calculate dialog size
        total_width = int(max_img_size * 3 + 100)
        dialog_height = int(max_img_size + 120)
        self.resize(total_width, dialog_height)

    def keyPressEvent(self, event):
        """Close on Escape key."""
        if event.key() == Qt.Key_Escape:
            self.accept()
        else:
            super().keyPressEvent(event)


class VisualPostprocessorWidget(QtWidgets.QWidget, Ui_Visual_Postprocessor):
    def __init__(self, parent=None, logger=None):
        super().__init__(parent)
        self.setupUi(self)

        # Logger
        if logger is not None:
            self.logger = logger.getChild("VisualPostprocessorWidget")
        else:
            self.logger = logging.getLogger("SPIm.VisualPostprocessorWidget")
        self.logger.setLevel(logging.DEBUG)
        self.logger.debug("Initializing VisualPostprocessorWidget")

        # Replace single progress bar with multi-phase progress widget
        self._setup_multi_phase_progress()

        # Thermal colormap
        self.cmap = cm.get_cmap('hot')

        # Placeholder for no images
        self._placeholder = QLabel("No images available")
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._placeholder.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._placeholder.hide()
        if hasattr(self, 'previsualiza_postprocesado_layout'):
            self.previsualiza_postprocesado_layout.addWidget(self._placeholder)
        else:
            self.layout().addWidget(self._placeholder)

        # Graphics scenes
        self._scene_orig  = QGraphicsScene(self)
        self._scene_noise = QGraphicsScene(self)
        self._scene_recon = QGraphicsScene(self)
        self.preview_image_original.setScene(self._scene_orig)
        self.preview_image_noise.setScene(self._scene_noise)
        self.preview_image_reconstructed.setScene(self._scene_recon)

        # Internal lists
        self._orig_images    = []
        self._recons_images  = []
        self._denoised_images = []

        # Lists for storing training metrics
        self.val_losses = []
        self.test_losses = []
        self.val_psnr = []
        self.val_ssim = []
        self.val_lpips = []

        # Slider signal and index label setup
        self.image_slider_value.valueChanged.connect(self._on_slider_changed)
        self.slider_index_label.setText("Index: 0")

        # Set size policies for the graphics views to maintain proper aspect ratio
        for view in [self.preview_image_original, self.preview_image_noise, self.preview_image_reconstructed]:
            view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            view.setCursor(Qt.PointingHandCursor)
            # Enable context menu
            view.setContextMenuPolicy(Qt.CustomContextMenu)

        # Install event filters to capture mouse clicks on graphics views
        self.preview_image_original.viewport().installEventFilter(self)
        self.preview_image_noise.viewport().installEventFilter(self)
        self.preview_image_reconstructed.viewport().installEventFilter(self)

        # Connect context menu signals
        self.preview_image_original.customContextMenuRequested.connect(
            lambda pos: self._show_context_menu(pos, "original"))
        self.preview_image_noise.customContextMenuRequested.connect(
            lambda pos: self._show_context_menu(pos, "noisy"))
        self.preview_image_reconstructed.customContextMenuRequested.connect(
            lambda pos: self._show_context_menu(pos, "denoised"))

        # Add context menu for training curve plot
        self.plot_training_curve.setContextMenuPolicy(Qt.CustomContextMenu)
        self.plot_training_curve.customContextMenuRequested.connect(self._show_plot_context_menu)
        self._current_plot_figure = None  # Store current figure for saving

    def _setup_multi_phase_progress(self):
        """Replace single progress bar with multi-phase progress widget."""
        # Hide the original progress bar
        self.progress_bar.hide()

        # Create multi-phase progress widget
        self.phase_progress = MultiPhaseProgressWidget(
            phases=[
                PostprocesadoWorker.PHASE_RECONSTRUCTION,
                PostprocesadoWorker.PHASE_TRAINING,
            ],
            title="Training Progress"
        )

        # Insert the multi-phase progress where the original progress bar was
        if hasattr(self, 'previsualiza_postprocesado_layout'):
            # Find where the progress bar was and insert the new widget
            layout = self.previsualiza_postprocesado_layout
            # Remove the old progress bar from layout if it's there
            idx = layout.indexOf(self.progress_bar)
            if idx >= 0:
                layout.insertWidget(idx, self.phase_progress)
            else:
                layout.addWidget(self.phase_progress)

    def on_phase_started(self, phase_name: str):
        """Handle phase started signal."""
        self.phase_progress.start_phase(phase_name)

    def on_phase_progress(self, phase_name: str, progress: int):
        """Handle phase progress signal."""
        self.phase_progress.update_phase_progress(phase_name, progress)

    def on_phase_completed(self, phase_name: str):
        """Handle phase completed signal."""
        self.phase_progress.complete_phase(phase_name)

    def reset_progress(self):
        """Reset all progress bars to initial state."""
        self.phase_progress.reset_all()

    def eventFilter(self, obj, event):
        """Handle click events on image previews to show combined popup."""
        if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
            # Check if click is on any of the preview viewports
            if obj in [self.preview_image_original.viewport(),
                       self.preview_image_noise.viewport(),
                       self.preview_image_reconstructed.viewport()]:
                if self._orig_images and self._recons_images and self._denoised_images:
                    self._show_combined_popup()
                    return True
        return super().eventFilter(obj, event)

    def _show_context_menu(self, pos, image_type: str):
        """Show context menu with Save As option."""
        idx = self.image_slider_value.value()

        # Get the appropriate image array
        if image_type == "original" and idx < len(self._orig_images):
            arr = self._orig_images[idx]
            default_name = f"original_image_{idx}.png"
        elif image_type == "noisy" and idx < len(self._recons_images):
            arr = self._recons_images[idx]
            default_name = f"noisy_image_{idx}.png"
        elif image_type == "denoised" and idx < len(self._denoised_images):
            arr = self._denoised_images[idx]
            default_name = f"denoised_image_{idx}.png"
        else:
            return

        menu = QMenu(self)
        save_action = menu.addAction("Save As...")

        # Get the view that triggered the menu
        if image_type == "original":
            view = self.preview_image_original
        elif image_type == "noisy":
            view = self.preview_image_noise
        else:
            view = self.preview_image_reconstructed

        action = menu.exec_(view.mapToGlobal(pos))
        if action == save_action:
            self._save_image(arr, default_name)

    def _save_image(self, arr: np.ndarray, default_name: str):
        """Save image to file at full resolution without compression."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Image",
            default_name,
            "PNG Image (*.png);;TIFF Image (*.tiff *.tif);;All Files (*)"
        )

        if not file_path:
            return

        try:
            # Convert array to full-resolution image with colormap
            arr = np.array(arr, copy=False)
            amin, amax = arr.min(), arr.max()
            norm = (arr - amin) / (amax - amin) if amax > amin else np.zeros_like(arr, dtype=float)

            rgba = self.cmap(norm)
            rgb = (rgba[..., :3] * 255).astype(np.uint8)

            h, w = rgb.shape[:2]
            data = rgb.tobytes()
            bytes_per_line = 3 * w
            qimg = QImage(data, w, h, bytes_per_line, QImage.Format_RGB888)

            # Save without compression (PNG is lossless)
            qimg.save(file_path)
            self.logger.info(f"Image saved to {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to save image: {e}")
            QtWidgets.QMessageBox.warning(self, "Save Error", f"Failed to save image: {e}")

    def _show_plot_context_menu(self, pos):
        """Show context menu for the training plot with Save As option."""
        if self._current_plot_figure is None:
            return

        menu = QMenu(self)
        save_action = menu.addAction("Save As...")

        action = menu.exec_(self.plot_training_curve.mapToGlobal(pos))
        if action == save_action:
            self._save_plot()

    def _save_plot(self):
        """Save the training plot to file at high resolution."""
        if self._current_plot_figure is None:
            return

        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save Training Plot",
            "training_metrics.png",
            "PNG Image (*.png);;PDF Document (*.pdf);;SVG Vector (*.svg);;All Files (*)"
        )

        if not file_path:
            return

        try:
            # Save at high resolution
            dpi = 300
            self._current_plot_figure.savefig(file_path, dpi=dpi, bbox_inches='tight',
                                               facecolor='white', edgecolor='none')
            self.logger.info(f"Training plot saved to {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to save plot: {e}")
            QtWidgets.QMessageBox.warning(self, "Save Error", f"Failed to save plot: {e}")

    def _array_to_pixmap(self, arr: np.ndarray, scale_factor: int = 4) -> QPixmap:
        """Convert numpy array to QPixmap with colormap, scaled up for visibility."""
        arr = np.array(arr, copy=False)
        if arr.ndim == 0:
            return QPixmap()

        amin, amax = arr.min(), arr.max()
        norm = (arr - amin) / (amax - amin) if amax > amin else np.zeros_like(arr, dtype=float)

        rgba = self.cmap(norm)
        rgb = (rgba[..., :3] * 255).astype(np.uint8)

        h, w = rgb.shape[:2]
        data = rgb.tobytes()
        bytes_per_line = 3 * w
        qimg = QImage(data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Scale up for better visibility (use FastTransformation for pixel-perfect display)
        target_size = max(300, w * scale_factor, h * scale_factor)
        return QPixmap.fromImage(qimg).scaled(target_size, target_size, Qt.KeepAspectRatio, Qt.FastTransformation)

    def _show_combined_popup(self):
        """Show combined popup with all three images."""
        idx = self.image_slider_value.value()

        if not (0 <= idx < len(self._orig_images)):
            return

        orig_pixmap = self._array_to_pixmap(self._orig_images[idx])
        noisy_pixmap = self._array_to_pixmap(self._recons_images[idx])
        denoised_pixmap = self._array_to_pixmap(self._denoised_images[idx])

        dialog = CombinedImagePopupDialog(orig_pixmap, noisy_pixmap, denoised_pixmap, self)
        dialog.exec_()

    def resizeEvent(self, event):
        """
        Maintain proper aspect ratio for the preview images when the widget is resized.
        Make the images frame height match the width of one image cell (square layout).
        """
        super().resizeEvent(event)

        # Make the images frame square: height = width of one image cell
        if hasattr(self, 'images_frame'):
            frame_width = self.images_frame.width()
            # Each image cell should be square, so height = width / 3 (for 3 images)
            # Add some padding for labels (~25px)
            cell_width = (frame_width - 20) // 3  # Account for spacing
            target_height = cell_width + 25  # Add space for labels
            target_height = max(100, min(target_height, 400))  # Clamp between 100-400px
            self.images_frame.setFixedHeight(target_height)

        # Re-render images after resize if we have data
        if self._denoised_images:
            QTimer.singleShot(0, lambda: self._on_slider_changed(self.image_slider_value.value()))

        # Re-render loss plot if we have data
        if self.val_losses and self.test_losses:
            QTimer.singleShot(0, self.plot_losses)

    def update_info(self,
                    num_images: int,
                    img_size: int,
                    dataset_type: str,
                    mask_type: str,
                    postprocessor_type: str,
                    n_params: int = None):
        """
        Update UI labels with postprocessing metadata.
        """
        self.logger.debug(
            "Updating info: num_images=%d, img_size=%d, dataset=%s, mask=%s, postproc=%s, params=%s",
            num_images, img_size, dataset_type, mask_type, postprocessor_type, n_params
        )
        self.num_images.setText(str(num_images))
        self.img_size.setText(f"{img_size}×{img_size}")
        self.dataset_type.setText(dataset_type)
        self.mask_type.setText(mask_type)
        self.postprocessor_type.setText(postprocessor_type)

        if n_params is not None:
            self.num_parameters_value.setText(f"{n_params:,}")

        max_idx = max(0, num_images - 1)
        self.image_slider_value.setRange(0, max_idx)
        self.slider_index_label.setText(f"Index: {self.image_slider_value.value()}")

    def set_images(self,
        orig_images: Sequence[np.ndarray],
        recons_images: Sequence[np.ndarray],
        denoised_images: Sequence[np.ndarray]):
        """
        Store image lists, configure slider, and display first image.
        """
        self.logger.debug(
           "Setting images: orig=%d, recons=%d, denoised=%d",
            len(orig_images), len(recons_images), len(denoised_images or [])
        )
        self._orig_images     = list(orig_images)
        self._recons_images   = list(recons_images)
        self._denoised_images = list(denoised_images or [])

        if not self._denoised_images:
            self.logger.info("No denoised images: hiding views")
            self.preview_image_original.hide()
            self.preview_image_noise.hide()
            self.preview_image_reconstructed.hide()
            self.image_slider_value.hide()
            self.slider_index_label.hide()
            self._placeholder.show()
            return

        self._placeholder.hide()
        self.preview_image_original.show()
        self.preview_image_noise.show()
        self.preview_image_reconstructed.show()
        self.image_slider_value.show()
        self.slider_index_label.show()

        n = len(self._denoised_images)
        self.image_slider_value.setRange(0, n - 1)
        self.image_slider_value.setValue(0)
        self.slider_index_label.setText("Index: 0")
        self._on_slider_changed(0)

    def _add_to_scene(self, arr: np.ndarray, scene: QGraphicsScene, view: QtWidgets.QGraphicsView):
        """
        Normalize a numpy array, apply thermal colormap, convert to QPixmap,
        and add to the graphics scene filling the available space while maintaining aspect ratio.
        """
        scene.clear()
        arr = np.array(arr, copy=False)
        # scale to [0,1]
        if arr.ndim == 0:
            norm = np.zeros_like(arr, dtype=float)
        else:
            amin, amax = arr.min(), arr.max()
            norm = (arr - amin) / (amax - amin) if amax > amin else np.zeros_like(arr, dtype=float)

        # apply thermal colormap
        rgba = self.cmap(norm)             # shape (h, w)→(h, w, 4)
        rgb  = (rgba[..., :3] * 255).astype(np.uint8)

        h, w = rgb.shape[:2]
        data = rgb.tobytes()
        bytes_per_line = 3 * w
        qimg = QImage(data, w, h, bytes_per_line, QImage.Format_RGB888)

        pix = QPixmap.fromImage(qimg)
        vw, vh = view.viewport().width(), view.viewport().height()

        # Scale to fill the viewport while maintaining aspect ratio
        if vw > 0 and vh > 0:
            pix = pix.scaled(vw, vh, Qt.KeepAspectRatio, Qt.FastTransformation)

        # Add pixmap centered in the scene
        pixmap_item = scene.addPixmap(pix)

        # Center the pixmap in the viewport
        px, py = pix.width(), pix.height()
        offset_x = (vw - px) / 2 if vw > px else 0
        offset_y = (vh - py) / 2 if vh > py else 0
        pixmap_item.setPos(offset_x, offset_y)

        # Set scene rect to match viewport for proper centering
        scene.setSceneRect(0, 0, vw, vh)
        self.logger.debug("Added image to scene (%dx%d) at offset (%.1f, %.1f)", px, py, offset_x, offset_y)

    def _on_slider_changed(self, idx: int):
        """
        Handle slider move: update label and display the corresponding images.
        """
        self.logger.debug("Slider changed: idx=%d", idx)
        self.slider_index_label.setText(f"Index: {idx}")
        if not (0 <= idx < len(self._orig_images)):
            return
        self._add_to_scene(self._orig_images[idx],  self._scene_orig,  self.preview_image_original)
        self._add_to_scene(self._recons_images[idx], self._scene_noise, self.preview_image_noise)
        self._add_to_scene(self._denoised_images[idx], self._scene_recon, self.preview_image_reconstructed)

    def plot_losses(self):
        """
        Plot training metrics: Loss evolution on top, PSNR/SSIM/LPIPS evolution below (vertical stacking).
        """
        if not self.val_losses or not self.test_losses:
            self.logger.debug("No loss data to plot")
            return

        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from PyQt5.QtGui import QImage, QPixmap

        self.logger.debug(
            "Plotting metrics: %d val_loss, %d test_loss, %d PSNR, %d SSIM, %d LPIPS points",
            len(self.val_losses), len(self.test_losses),
            len(self.val_psnr), len(self.val_ssim), len(self.val_lpips)
        )
        scene = QGraphicsScene(self)
        self.plot_training_curve.setScene(scene)

        # Check if we have PSNR/SSIM data
        has_quality_metrics = self.val_psnr and self.val_ssim
        has_lpips = self.val_lpips and len(self.val_lpips) > 0

        if has_quality_metrics:
            # Two subplots stacked vertically: Loss on top, quality metrics below
            fig = Figure(figsize=(6, 5), dpi=100)
            canvas = FigureCanvasAgg(fig)

            # Loss subplot (top)
            ax1 = fig.add_subplot(211)
            ax1.plot(self.val_losses, label='Validation Loss', color='tab:blue')
            ax1.plot(self.test_losses, label='Test Loss', color='tab:orange')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Loss Evolution')
            ax1.legend(loc='upper right', fontsize='small')
            ax1.grid(True, alpha=0.3)

            # Quality metrics subplot (bottom) with multiple y-axes
            ax2 = fig.add_subplot(212)
            line_psnr, = ax2.plot(self.val_psnr, label='PSNR (dB)', color='tab:green')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('PSNR (dB)', color='tab:green')
            ax2.tick_params(axis='y', labelcolor='tab:green')
            ax2.set_title('Quality Metrics Evolution')
            ax2.grid(True, alpha=0.3)

            # Second y-axis for SSIM
            ax2_ssim = ax2.twinx()
            line_ssim, = ax2_ssim.plot(self.val_ssim, label='SSIM', color='tab:red', linestyle='--')
            ax2_ssim.set_ylabel('SSIM', color='tab:red')
            ax2_ssim.tick_params(axis='y', labelcolor='tab:red')
            ax2_ssim.set_ylim(0, 1)

            # Third y-axis for LPIPS (if available)
            if has_lpips:
                ax2_lpips = ax2.twinx()
                # Offset the third axis to the right
                ax2_lpips.spines['right'].set_position(('axes', 1.15))
                line_lpips, = ax2_lpips.plot(self.val_lpips, label='LPIPS', color='tab:purple', linestyle=':')
                ax2_lpips.set_ylabel('LPIPS', color='tab:purple')
                ax2_lpips.tick_params(axis='y', labelcolor='tab:purple')
                ax2_lpips.set_ylim(0, 1)
                # Combined legend with all three
                ax2.legend([line_psnr, line_ssim, line_lpips],
                          ['PSNR (dB)', 'SSIM', 'LPIPS'],
                          loc='lower right', fontsize='small')
            else:
                # Combined legend with just PSNR and SSIM
                ax2.legend([line_psnr, line_ssim],
                          ['PSNR (dB)', 'SSIM'],
                          loc='lower right', fontsize='small')
        else:
            # Single plot for loss only (backward compatibility)
            fig = Figure(figsize=(6, 3), dpi=100)
            canvas = FigureCanvasAgg(fig)
            ax = fig.add_subplot(111)

            ax.plot(self.val_losses, label='Validation Loss')
            ax.plot(self.test_losses, label='Test Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Loss Evolution')
            ax.legend()
            ax.grid(True, alpha=0.3)

        fig.tight_layout()

        # Store figure for Save As functionality
        self._current_plot_figure = fig

        canvas.draw()
        buf = canvas.buffer_rgba()
        w, h = canvas.get_width_height()
        qimg = QImage(buf, w, h, QImage.Format_RGBA8888)
        pixmap = QPixmap.fromImage(qimg)

        vw, vh = self.plot_training_curve.viewport().width(), self.plot_training_curve.viewport().height()
        pixmap = pixmap.scaled(vw, vh, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        scene.addPixmap(pixmap)
        scene.setSceneRect(0, 0, pixmap.width(), pixmap.height())
        self.logger.info("Training metrics plot rendered to view")
