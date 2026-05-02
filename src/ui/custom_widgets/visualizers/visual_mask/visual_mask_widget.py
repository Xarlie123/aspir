"""Widget to display mask pattern previews with thermal colormap."""
import logging
import numpy as np
from math import ceil, sqrt
from typing import Sequence
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel, QGridLayout, QSizePolicy, QMenu, QFileDialog

from ui.custom_widgets.visualizers.visual_mask.visual_mask import Ui_Visual_Mask
from ui.custom_widgets.common.hoverable_image_label import HoverableImageLabel
from ui.custom_widgets.common.multi_phase_progress import MultiPhaseProgressWidget

# Phase name constant
PHASE_GENERATING = "Generating"


class VisualMaskWidget(QtWidgets.QWidget, Ui_Visual_Mask):
    """Preview widget for masks, modeled on VisualDatasetWidget, with logging."""
    def __init__(self, parent=None, logger=None):
        super().__init__(parent)
        self.setupUi(self)

        # Initialize logger
        if logger is None:
            self.logger = logging.getLogger(self.__class__.__name__)
        else:
            self.logger = logger.getChild(self.__class__.__name__)
        self.logger.debug("Initializing VisualMaskWidget")

        # Set zero margins on outer layout to match dataset widget
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)

        # Layout for mask thumbnails with margins to prevent clipping
        self.mask_layout = QGridLayout()
        self.mask_layout.setContentsMargins(5, 5, 5, 5)
        self.mask_layout.setSpacing(5)
        self.preview_masks_graphics.setLayout(self.mask_layout)

        self._masks: Sequence[np.ndarray] = []
        self._data_format: str = None  # Data format for hover display

        # Connect slider and combo to update
        self.select_mask_slider_value.valueChanged.connect(self._on_slider_moved)
        self.number_masks_preview_value.currentIndexChanged.connect(
            lambda _: self._on_slider_moved(self.select_mask_slider_value.value())
        )

        # Initialize slider to 0
        self.select_mask_slider_value.setValue(0)

        # Match dataset widget's minimum size
        self.setMinimumSize(300, 400)

        # Make the preview widget maintain square aspect ratio
        self.preview_masks_graphics.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # Enable right-click context menu
        self.preview_masks_graphics.setContextMenuPolicy(Qt.CustomContextMenu)
        self.preview_masks_graphics.customContextMenuRequested.connect(self._show_context_menu)

        # Setup multi-phase progress bar (unified style)
        self._setup_multi_phase_progress()

    def _setup_multi_phase_progress(self):
        """Replace default progress bar with multi-phase progress widget."""
        # Hide the original progress bar
        self.masks_progress_bar.hide()

        # Create multi-phase progress widget with single "Generating" phase
        self.phase_progress = MultiPhaseProgressWidget(
            phases=[PHASE_GENERATING],
            show_title=False
        )

        # Insert the multi-phase progress where the original progress bar was
        layout = self.verticalLayout
        idx = layout.indexOf(self.masks_progress_bar)
        if idx >= 0:
            layout.insertWidget(idx, self.phase_progress)
        else:
            layout.addWidget(self.phase_progress)

    def set_progress(self, value: int):
        """Update progress bar value (0-100)."""
        self.phase_progress.start_phase(PHASE_GENERATING)
        self.phase_progress.update_phase_progress(PHASE_GENERATING, value)
        if value >= 100:
            self.phase_progress.complete_phase(PHASE_GENERATING)

    def reset_progress(self):
        """Reset progress bar to initial state."""
        self.phase_progress.reset_all()

    def _show_context_menu(self, pos):
        """Show context menu with Save As option."""
        idx = self.select_mask_slider_value.value()
        if not self._masks or idx >= len(self._masks):
            return

        menu = QMenu(self)
        save_action = menu.addAction("Save As...")

        action = menu.exec_(self.preview_masks_graphics.mapToGlobal(pos))
        if action == save_action:
            self._save_mask(idx)

    def _save_mask(self, idx: int):
        """Save mask to file at full resolution without compression."""
        if idx >= len(self._masks):
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Mask",
            f"mask_{idx}.png",
            "PNG Image (*.png);;TIFF Image (*.tiff *.tif);;All Files (*)"
        )

        if not file_path:
            return

        try:
            mask = self._masks[idx]

            # Min-max stretch to uint8 [0, 255] regardless of input
            # dtype — covers float {-1,+1} (Hadamard), float {0,1}
            # (Scatter), uint8 {0,1} (Sweep) and any future mask
            # family with a different native range. A passthrough
            # branch for "uint8 → uint8" used to live here, but it
            # rendered Sweep masks as nearly black after their
            # storage switched to {0, 1}.
            if isinstance(mask, np.ndarray):
                mn, mx = float(mask.min()), float(mask.max())
                denom = (mx - mn) if (mx - mn) > 0 else 1.0
                arr = ((mask.astype(np.float32) - mn) / denom * 255).astype(np.uint8)
            else:
                arr = mask

            h, w = arr.shape[:2]
            qimg = QImage(arr.data, w, h, w, QImage.Format_Grayscale8)
            qimg.save(file_path)
            self.logger.info(f"Mask saved to {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to save mask: {e}")
            QtWidgets.QMessageBox.warning(self, "Save Error", f"Failed to save mask: {e}")

    def resizeEvent(self, event):
        """Maintain square aspect ratio for the preview widget based on available space."""
        super().resizeEvent(event)

        # Calculate available height for the preview (total height minus controls)
        # Controls: slider row (~35px) + info form (~70px) + progress bar (~50px) + margins (~15px)
        controls_height = 170
        available_height = self.height() - controls_height

        # Use the minimum of width and available height to maintain square without overflow
        width = self.preview_masks_graphics.width()
        if width > 0 and available_height > 0:
            square_size = min(width, available_height)
            # Ensure minimum size of 100px
            square_size = max(square_size, 100)
            self.preview_masks_graphics.setFixedHeight(square_size)

        # Re-render thumbnails after resize
        QTimer.singleShot(0, lambda: self._on_slider_moved(self.select_mask_slider_value.value()))

    def set_masks(self, masks: Sequence[np.ndarray], data_format: str = None):
        """Update the mask list and render the first set.

        Args:
            masks: Sequence of numpy arrays (masks)
            data_format: Optional data format string ("FP32", "INT8", "INT4")
        """
        self._masks = list(masks) if masks is not None else []
        self._data_format = data_format
        count = len(self._masks)
        self.logger.info("Setting %d masks", count)
        max_idx = max(0, count - 1)
        self.select_mask_slider_value.setRange(0, max_idx)
        self.select_mask_slider_value.setValue(0)
        self._on_slider_moved(0)

    def set_data_format(self, data_format: str):
        """Set the data format for hover display."""
        self._data_format = data_format

    def update_info(self, num_masks: int, img_dim: int, mask_type: str):
        """Update info labels inside the widget."""
        self.number_masks_info_value.setText(str(num_masks))
        self.mask_dimension_info_value.setText(f"{img_dim}×{img_dim}")
        self.mask_type_info_value.setText(mask_type)
        self.logger.info(
            "Updated info: num_masks=%d, img_dim=%d, mask_type=%s",
            num_masks, img_dim, mask_type
        )

    def _on_slider_moved(self, idx: int):
        """Slot that responds to slider moves: show `count` consecutive masks from idx."""
        self.select_mask_slider_label.setText(f"Index: {idx}")
        try:
            count = int(self.number_masks_preview_value.currentText())
        except ValueError:
            count = 1
        self.logger.debug("Rendering masks starting at idx %d, count %d", idx, count)

        # Clear previous thumbnails
        while self.mask_layout.count():
            item = self.mask_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not self._masks or idx < 0:
            self.logger.debug("No masks to render or invalid idx %d", idx)
            return

        grid_size = int(ceil(sqrt(count)))
        # Account for margins and spacing
        margins = self.mask_layout.contentsMargins()
        spacing = self.mask_layout.spacing()
        view_w = self.preview_masks_graphics.width() - margins.left() - margins.right()
        view_h = self.preview_masks_graphics.height() - margins.top() - margins.bottom()
        # Use square area based on minimum dimension for consistent aspect ratio
        square_size = min(view_w, view_h)
        # Subtract spacing between cells and calculate square cell size
        cell_size = (square_size - spacing * (grid_size - 1)) // grid_size
        cell_w = cell_size
        cell_h = cell_size

        for i in range(count):
            mask_idx = idx + i
            if mask_idx >= len(self._masks):
                break

            mask = self._masks[mask_idx]
            # Min-max stretch to uint8 [0, 255] regardless of input
            # dtype — covers float {-1,+1} (Hadamard), float {0,1}
            # (Scatter), uint8 {0,1} (Sweep) and any future mask
            # family with a different native range. A passthrough
            # branch for "uint8 → uint8" used to live here, but it
            # rendered Sweep masks as nearly black after their
            # storage switched to {0, 1}.
            if isinstance(mask, np.ndarray):
                mn, mx = float(mask.min()), float(mask.max())
                denom = (mx - mn) if (mx - mn) > 0 else 1.0
                arr = ((mask.astype(np.float32) - mn) / denom * 255).astype(np.uint8)
            else:
                arr = mask

            h, w = arr.shape[:2]
            qimg = QImage(arr.data, w, h, w, QImage.Format_Grayscale8)
            pix = QPixmap.fromImage(qimg).scaled(
                cell_w, cell_h,
                Qt.KeepAspectRatio,
                Qt.FastTransformation
            )

            lbl = HoverableImageLabel()
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setPixmapWithData(pix, mask, self._data_format)  # Pass original mask and format
            row, col = divmod(i, grid_size)
            self.mask_layout.addWidget(lbl, row, col)
        self.logger.debug("Rendered %d masks", min(count, len(self._masks) - idx))
