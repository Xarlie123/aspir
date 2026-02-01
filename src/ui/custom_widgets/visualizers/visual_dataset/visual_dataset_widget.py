import logging
import numpy as np
from math import ceil, sqrt
from typing import Sequence
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel, QGridLayout, QSizePolicy, QMenu, QFileDialog
from matplotlib import cm

from ui.custom_widgets.visualizers.visual_dataset.visual_dataset import Ui_Visual_Dataset
from ui.custom_widgets.common.hoverable_image_label import HoverableImageLabel
from ui.custom_widgets.common.multi_phase_progress import MultiPhaseProgressWidget

# Phase name constant
PHASE_LOADING = "Loading"


class VisualDatasetWidget(QtWidgets.QWidget, Ui_Visual_Dataset):
    """
    Widget to display thumbnails of an image dataset,
    using a thermal colormap 'hot'.
    """
    def __init__(self, parent=None, logger=None):
        super().__init__(parent)
        self.setupUi(self)

        # Logger
        if logger is None:
            self.logger = logging.getLogger(self.__class__.__name__)
        else:
            self.logger = logger.getChild(self.__class__.__name__)
        self.logger.debug("Initializing VisualDatasetWidget")

        # ─── Layout for thumbnails with margins to prevent clipping ───
        self.image_layout = QGridLayout()
        self.image_layout.setContentsMargins(5, 5, 5, 5)
        self.image_layout.setSpacing(5)
        self.preview_image_graphics.setLayout(self.image_layout)

        self._data: Sequence[np.ndarray] = []
        self._data_format: str = None  # Data format for hover display

        # Thermal colormap
        self.cmap = cm.get_cmap('hot')

        # Connect slider and combobox
        self.select_image_slider_value.valueChanged.connect(self._on_slider_moved)
        self.number_images_preview_value.currentIndexChanged.connect(
            lambda _: self._on_slider_moved(self.select_image_slider_value.value())
        )

        # Make the preview widget maintain square aspect ratio
        self.preview_image_graphics.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # Enable right-click context menu
        self.preview_image_graphics.setContextMenuPolicy(Qt.CustomContextMenu)
        self.preview_image_graphics.customContextMenuRequested.connect(self._show_context_menu)

        # Setup multi-phase progress bar (unified style)
        self._setup_multi_phase_progress()

    def _setup_multi_phase_progress(self):
        """Replace default progress bar with multi-phase progress widget."""
        # Hide the original progress bar
        self.dataset_progress_bar.hide()

        # Create multi-phase progress widget with single "Loading" phase
        self.phase_progress = MultiPhaseProgressWidget(
            phases=[PHASE_LOADING],
            show_title=False
        )

        # Insert the multi-phase progress where the original progress bar was
        layout = self.verticalLayout
        idx = layout.indexOf(self.dataset_progress_bar)
        if idx >= 0:
            layout.insertWidget(idx, self.phase_progress)
        else:
            layout.addWidget(self.phase_progress)

    def set_progress(self, value: int):
        """Update progress bar value (0-100)."""
        self.phase_progress.start_phase(PHASE_LOADING)
        self.phase_progress.update_phase_progress(PHASE_LOADING, value)
        if value >= 100:
            self.phase_progress.complete_phase(PHASE_LOADING)

    def reset_progress(self):
        """Reset progress bar to initial state."""
        self.phase_progress.reset_all()

    def _show_context_menu(self, pos):
        """Show context menu with Save As option."""
        idx = self.select_image_slider_value.value()
        if not self._data or idx >= len(self._data):
            return

        menu = QMenu(self)
        save_action = menu.addAction("Save As...")

        action = menu.exec_(self.preview_image_graphics.mapToGlobal(pos))
        if action == save_action:
            self._save_image(idx)

    def _save_image(self, idx: int):
        """Save image to file at full resolution without compression."""
        if idx >= len(self._data):
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Image",
            f"dataset_image_{idx}.png",
            "PNG Image (*.png);;TIFF Image (*.tiff *.tif);;All Files (*)"
        )

        if not file_path:
            return

        try:
            img = self._data[idx]

            # Normalize to 0–255 if float (handles float16, float32, float64)
            if isinstance(img, np.ndarray) and np.issubdtype(img.dtype, np.floating):
                mn, mx = float(img.min()), float(img.max())
                denom = (mx - mn) if (mx - mn) > 0 else 1.0
                arr = ((img.astype(np.float32) - mn) / denom * 255).astype(np.uint8)
            elif isinstance(img, np.ndarray) and img.dtype == np.uint8:
                arr = img
            else:
                # Convert other integer types to uint8
                arr = img.astype(np.uint8) if isinstance(img, np.ndarray) else img

            h, w = arr.shape[:2]

            # Apply colormap for grayscale
            if arr.ndim == 2:
                norm = arr.astype(np.float32) / 255.0
                rgba = self.cmap(norm)
                rgb = (rgba[..., :3] * 255).astype(np.uint8)
                data = rgb.tobytes()
                bytes_per_line = 3 * w
                qimg = QImage(data, w, h, bytes_per_line, QImage.Format_RGB888)
            else:
                data = arr.tobytes()
                bytes_per_line = 3 * w
                qimg = QImage(data, w, h, bytes_per_line, QImage.Format_RGB888)

            qimg.save(file_path)
            self.logger.info(f"Image saved to {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to save image: {e}")
            QtWidgets.QMessageBox.warning(self, "Save Error", f"Failed to save image: {e}")

    def set_controls_height(self, height: int):
        """Set custom controls height (use when hiding progress bar)."""
        self._controls_height = height

    def resizeEvent(self, event):
        """Maintain square aspect ratio for the preview widget based on available space."""
        super().resizeEvent(event)

        # Calculate available height for the preview (total height minus controls)
        # Default: slider row (~35px) + info form (~70px) + progress bar (~50px) + margins (~15px) = 170
        # Without progress bar: slider row (~35px) + info form (~70px) + margins (~15px) = 120
        controls_height = getattr(self, '_controls_height', 170)
        available_height = self.height() - controls_height

        # Use the minimum of width and available height to maintain square without overflow
        width = self.preview_image_graphics.width()
        if width > 0 and available_height > 0:
            square_size = min(width, available_height)
            # Ensure minimum size of 100px
            square_size = max(square_size, 100)
            self.preview_image_graphics.setFixedHeight(square_size)

        # Re-render thumbnails after resize
        QTimer.singleShot(0, lambda: self._on_slider_moved(self.select_image_slider_value.value()))

    def set_data(self, data: Sequence[np.ndarray], data_format: str = None):
        """
        Stores images and forces first rendering.

        Args:
            data: Sequence of numpy arrays (images)
            data_format: Optional data format string ("FP32", "INT8", "INT4")
        """
        self._data = list(data) if data is not None else []
        self._data_format = data_format
        max_idx = max(0, len(self._data) - 1)
        self.select_image_slider_value.setRange(0, max_idx)
        self.select_image_slider_value.setValue(0)
        QTimer.singleShot(0, lambda: self._on_slider_moved(0))

    def set_data_format(self, data_format: str):
        """Set the data format for hover display."""
        self._data_format = data_format

    def update_info(self, num_images: int, img_dim: int, dataset_type: str):
        """Update the dataset info QLabel widgets."""
        self.logger.debug(
            "update_info -> count: %d, dim: %dx%d, type: %s",
            num_images, img_dim, img_dim, dataset_type
        )
        self.dataset_size_info_value.setText(str(num_images))
        self.image_dimension_info_value.setText(f"{img_dim}×{img_dim}")
        self.dataset_type_info_value.setText(dataset_type)

    def _on_slider_moved(self, idx: int):
        """
        Responds to slider change: updates thumbnails based on index and count,
        applying the 'hot' colormap.'.
        """
        self.logger.debug("Slider moved to index %d", idx)
        self.select_image_slider_label.setText(f"Index: {idx}")
        try:
            count = int(self.number_images_preview_value.currentText())
        except ValueError:
            count = 1

        # Clear previous thumbnails
        while self.image_layout.count():
            item = self.image_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not self._data or idx < 0:
            self.logger.debug("No data to display or invalid index")
            return

        grid = int(ceil(sqrt(count)))
        # Account for margins and spacing
        margins = self.image_layout.contentsMargins()
        spacing = self.image_layout.spacing()
        view_w = self.preview_image_graphics.width() - margins.left() - margins.right()
        view_h = self.preview_image_graphics.height() - margins.top() - margins.bottom()
        # Use square area based on minimum dimension for consistent aspect ratio
        square_size = min(view_w, view_h)
        # Subtract spacing between cells and calculate square cell size
        cell_size = (square_size - spacing * (grid - 1)) // grid
        cell_w = cell_size
        cell_h = cell_size
        self.logger.debug("Grid %dx%d, cell %dx%d (square area: %d)", grid, grid, cell_w, cell_h, square_size)

        for i in range(count):
            img_idx = idx + i
            if img_idx >= len(self._data):
                break

            img = self._data[img_idx]

            # Normalize to 0–255 if float (handles float16, float32, float64)
            if isinstance(img, np.ndarray) and np.issubdtype(img.dtype, np.floating):
                mn, mx = float(img.min()), float(img.max())
                denom = (mx - mn) if (mx - mn) > 0 else 1.0
                arr = ((img.astype(np.float32) - mn) / denom * 255).astype(np.uint8)
            elif isinstance(img, np.ndarray) and img.dtype == np.uint8:
                arr = img
            else:
                # Convert other integer types to uint8
                arr = img.astype(np.uint8) if isinstance(img, np.ndarray) else img

            h, w = arr.shape[:2]

            # Apply 'hot' colormap to grayscale image
            if arr.ndim == 2:
                norm = arr.astype(np.float32) / 255.0           # scale [0, 1]
                rgba = self.cmap(norm)                          # (h, w, 4)
                rgb  = (rgba[..., :3] * 255).astype(np.uint8)   # drop alpha
                data = rgb.tobytes()
                bytes_per_line = 3 * w
                qimg = QImage(data, w, h, bytes_per_line, QImage.Format_RGB888)
            else:
                # Already in RGB
                data = arr.tobytes()
                bytes_per_line = 3 * w
                qimg = QImage(data, w, h, bytes_per_line, QImage.Format_RGB888)

            pix = QPixmap.fromImage(qimg).scaled(
                cell_w, cell_h,
                Qt.KeepAspectRatio,
                Qt.FastTransformation
            )

            lbl = HoverableImageLabel()
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setPixmapWithData(pix, img, self._data_format)  # Pass original image and format
            row, col = divmod(i, grid)
            self.image_layout.addWidget(lbl, row, col)

    def _show_single_image(self, idx: int):
        """
        Display single image, scaled to widget size.
        """
        self.logger.debug("Showing single image at index %d", idx)
        while self.image_layout.count():
            item = self.image_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if self._data is None or idx < 0 or idx >= len(self._data):
            self.logger.debug("Index out of range or empty data: %d", idx)
            return

        img = self._data[idx]
        # Normalize to 0–255 if float (handles float16, float32, float64)
        if isinstance(img, np.ndarray) and np.issubdtype(img.dtype, np.floating):
            mn, mx = float(img.min()), float(img.max())
            denom = (mx - mn) if (mx - mn) > 0 else 1.0
            arr = ((img.astype(np.float32) - mn) / denom * 255).astype(np.uint8)
        elif isinstance(img, np.ndarray) and img.dtype == np.uint8:
            arr = img
        else:
            # Convert other integer types to uint8
            arr = img.astype(np.uint8) if isinstance(img, np.ndarray) else img

        h, w = arr.shape[:2]
        if arr.ndim == 2:
            norm = arr.astype(np.float32) / 255.0
            rgba = self.cmap(norm)
            rgb  = (rgba[..., :3] * 255).astype(np.uint8)
            data = rgb.tobytes()
            bytes_per_line = 3 * w
            qimg = QImage(data, w, h, bytes_per_line, QImage.Format_RGB888)
        else:
            data = arr.tobytes()
            bytes_per_line = 3 * w
            qimg = QImage(data, w, h, bytes_per_line, QImage.Format_RGB888)

        view_w = self.preview_image_graphics.width()
        view_h = self.preview_image_graphics.height()
        # Use square size based on minimum dimension
        square_size = min(view_w, view_h)
        pix = QPixmap.fromImage(qimg).scaled(
            square_size, square_size,
            Qt.KeepAspectRatio,
            Qt.FastTransformation
        )

        lbl = HoverableImageLabel()
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setPixmapWithData(pix, img, self._data_format)  # Pass original image and format
        self.image_layout.addWidget(lbl, 0, 0, alignment=Qt.AlignCenter)
