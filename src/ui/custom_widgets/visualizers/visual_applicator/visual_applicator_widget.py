"""Widget to display applicator preview with thermal colormap (matching dataset widget style)."""
import logging
import numpy as np
from math import ceil, sqrt
from typing import Sequence
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QLabel, QGridLayout, QSizePolicy, QVBoxLayout, QComboBox,
                              QSlider, QFormLayout, QWidget, QMenu, QFileDialog)

from matplotlib import cm

from ui.custom_widgets.common.hoverable_image_label import HoverableImageLabel


class VisualApplicatorWidget(QtWidgets.QWidget):
    """
    Widget to visualize mask application to the dataset.
    Rewritten to match VisualDatasetWidget's approach (QWidget + QGridLayout instead of QGraphicsView).
    """
    # Mapping of applicator classes to the paper's algorithm names.
    APPLICATOR_NAMES = {
        'ApplicatorScatter': 'Ghost Imaging',
        'ApplicatorPseudoinverse': 'Pseudoinverse',
        'ApplicatorFISTA': 'FISTA',
        'ApplicatorTV': 'TV-norm',
        'ApplicatorSweep': 'Sweep Linear',
        'ApplicatorHadamard': 'Hadamard Linear',
        # Legacy class names kept so old saved experiments still render.
        'ApplicatorScatterPseudoinverse': 'Pseudoinverse',
        'ApplicatorScatterFISTA': 'FISTA',
        'ApplicatorScatterTV': 'TV-norm',
    }

    def __init__(self, simulation, parent=None, logger=None, status_manager=None):
        super().__init__(parent)
        self.simulation = simulation
        self.status_manager = status_manager

        # Thermal colormap
        self.cmap = cm.get_cmap('hot')

        # Configure logger
        if logger is None:
            self.logger = logging.getLogger(self.__class__.__name__)
        else:
            self.logger = logger.getChild(self.__class__.__name__)
        self.logger.debug("Initializing VisualApplicatorWidget")

        # Build UI programmatically (matching dataset widget structure)
        self._setup_ui()

        # Internal data
        self._dataset = []     # list of np.ndarray
        self._masks = []       # list of np.ndarray
        self._applicator = None
        self._img_idx = 0      # image index for which we will apply the mask
        self._data_format: str = None  # Data format for hover display

        # Connect signals
        self.select_image_slider_value.valueChanged.connect(self._on_slider_moved)
        self.number_images_preview_value.currentIndexChanged.connect(
            lambda _: self._on_slider_moved(self.select_image_slider_value.value())
        )
        self.logger.debug("Signal connections established")

    def _setup_ui(self):
        """Build UI to match VisualDatasetWidget structure."""
        # Main layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(5)

        # Inner vertical layout
        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setSpacing(5)

        # Preview area (QWidget with QGridLayout for thumbnails)
        self.preview_applicator_graphics = QWidget()
        self.preview_applicator_graphics.setMinimumSize(300, 300)
        self.preview_applicator_graphics.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc;")
        self.preview_applicator_graphics.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # Enable right-click context menu
        self.preview_applicator_graphics.setContextMenuPolicy(Qt.CustomContextMenu)
        self.preview_applicator_graphics.customContextMenuRequested.connect(self._show_context_menu)

        # Layout for thumbnails with margins to prevent clipping
        self.image_layout = QGridLayout()
        self.image_layout.setContentsMargins(5, 5, 5, 5)
        self.image_layout.setSpacing(5)
        self.preview_applicator_graphics.setLayout(self.image_layout)

        self.verticalLayout.addWidget(self.preview_applicator_graphics)

        # Controls row (combobox, label, slider)
        self.gridLayout_2 = QGridLayout()
        self.gridLayout_2.setSpacing(5)

        self.number_images_preview_value = QComboBox()
        for val in ["1", "2", "4", "9", "16", "25", "36", "49", "64"]:
            self.number_images_preview_value.addItem(val)
        self.gridLayout_2.addWidget(self.number_images_preview_value, 0, 0)

        self.select_image_slider_label = QLabel("Mask index: 0")
        self.gridLayout_2.addWidget(self.select_image_slider_label, 0, 1)

        self.select_image_slider_value = QSlider(Qt.Horizontal)
        self.select_image_slider_value.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
        self.gridLayout_2.addWidget(self.select_image_slider_value, 0, 2)

        self.verticalLayout.addLayout(self.gridLayout_2)

        # Info form layout
        self.Info_prev_imagen_layout = QFormLayout()

        self.applicator_name_info_text = QLabel("Applicator Type:")
        self.applicator_name_info_value = QLabel("")
        self.Info_prev_imagen_layout.addRow(self.applicator_name_info_text, self.applicator_name_info_value)

        self.dataset_type_info_text = QLabel("Dataset type:")
        self.dataset_type_info_value = QLabel("")
        self.Info_prev_imagen_layout.addRow(self.dataset_type_info_text, self.dataset_type_info_value)

        self.mask_size_info_text = QLabel("Number of masks:")
        self.mask_size_info_value = QLabel("")
        self.Info_prev_imagen_layout.addRow(self.mask_size_info_text, self.mask_size_info_value)

        self.image_dimension_info_text = QLabel("Image dimension (pix):")
        self.image_dimension_info_value = QLabel("")
        self.Info_prev_imagen_layout.addRow(self.image_dimension_info_text, self.image_dimension_info_value)

        self.verticalLayout.addLayout(self.Info_prev_imagen_layout)

        # Add stretch to push content to top
        self.verticalLayout.addStretch()

        self.main_layout.addLayout(self.verticalLayout)

        # Set minimum size
        self.setMinimumSize(300, 400)

    def resizeEvent(self, event):
        """Maintain square aspect ratio for the preview widget based on available space."""
        super().resizeEvent(event)

        # Calculate available height for the preview (total height minus controls)
        # Controls take approximately 120px (slider row + info form + margins)
        controls_height = 120
        available_height = self.height() - controls_height

        # Use the minimum of width and available height to maintain square without overflow
        width = self.preview_applicator_graphics.width()
        if width > 0 and available_height > 0:
            square_size = min(width, available_height)
            # Ensure minimum size of 100px
            square_size = max(square_size, 100)
            self.preview_applicator_graphics.setFixedHeight(square_size)

        # Re-render thumbnails after resize
        QTimer.singleShot(0, lambda: self._on_slider_moved(self.select_image_slider_value.value()))

    def set_data(self, dataset, mask, applicator):
        """Load data and reset view."""
        self._dataset = list(dataset.data) if getattr(dataset, 'data', None) is not None else []
        self._masks = list(mask.masks) if hasattr(mask, 'masks') and mask.masks is not None else []
        self._applicator = applicator

        # Get data format from dataset if available
        self._data_format = getattr(dataset, 'data_format', None)

        # Adjust ranges
        max_mask = max(0, len(self._masks) - 1)
        self.select_image_slider_value.setRange(0, max_mask)
        self.select_image_slider_value.setValue(0)
        self.number_images_preview_value.setCurrentIndex(0)

        # Quick info
        if applicator:
            class_name = applicator.__class__.__name__
            applicator_name = self.APPLICATOR_NAMES.get(class_name, class_name)
        else:
            applicator_name = 'N/A'
        dataset_type = self.simulation.dataset.__class__.__name__ if self.simulation.dataset else 'N/A'
        num_masks = len(self._masks)
        img_dim = self._dataset[self._img_idx].shape[0] if self._dataset else 0
        self.update_info(applicator_name, dataset_type, num_masks, img_dim)

        self.logger.info(f"Data loaded: dataset={dataset_type}, applicator={applicator_name}, masks={num_masks}")

        # Initial render
        self._on_slider_moved(0)

    def set_image_index(self, idx: int):
        """Public slot: sets image index for applying masks."""
        self._img_idx = idx
        self.logger.debug(f"Image index updated: {idx}")
        mask_idx = self.select_image_slider_value.value()
        self._on_slider_moved(mask_idx)

    def _on_slider_moved(self, mask_idx: int):
        """Redraws grid views with applied masks."""
        self.select_image_slider_label.setText(f"Mask index: {mask_idx}")

        try:
            count = int(self.number_images_preview_value.currentText())
        except ValueError:
            count = 1

        # Clear previous thumbnails
        while self.image_layout.count():
            item = self.image_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not self._dataset or not self._masks or self._applicator is None:
            self.logger.debug("No data, masks, or applicator to render.")
            return

        # Notify status manager that reconstruction is starting
        if self.status_manager:
            applicator_name = type(self._applicator).__name__
            friendly_name = self.APPLICATOR_NAMES.get(applicator_name, applicator_name)
            self.status_manager.start_task(f"Reconstruction ({friendly_name})")

        try:
            grid = int(ceil(sqrt(count)))
            # Account for margins and spacing
            margins = self.image_layout.contentsMargins()
            spacing = self.image_layout.spacing()
            view_w = self.preview_applicator_graphics.width() - margins.left() - margins.right()
            view_h = self.preview_applicator_graphics.height() - margins.top() - margins.bottom()
            # Use square area based on minimum dimension for consistent aspect ratio
            square_size = min(view_w, view_h)
            # Subtract spacing between cells and calculate square cell size
            cell_size = (square_size - spacing * (grid - 1)) // grid
            cell_w = cell_size
            cell_h = cell_size

            base_img_idx = self._img_idx
            for i in range(count):
                img_idx = base_img_idx + i
                if img_idx >= len(self._dataset):
                    break

                applied = self._applicator.apply_mask_range(0, mask_idx, img_idx)
                arr = self._normalize(applied)  # uint8 0–255, shape (h,w)

                h, w = arr.shape

                # Apply 'hot' colormap to grayscale image
                norm = arr.astype(np.float32) / 255.0
                rgba = self.cmap(norm)
                rgb = (rgba[..., :3] * 255).astype(np.uint8)

                data = rgb.tobytes()
                bytes_per_line = 3 * w
                qimg = QImage(data, w, h, bytes_per_line, QImage.Format_RGB888)

                pix = QPixmap.fromImage(qimg).scaled(
                    cell_w, cell_h,
                    Qt.KeepAspectRatio,
                    Qt.FastTransformation
                )

                lbl = HoverableImageLabel()
                lbl.setAlignment(Qt.AlignCenter)
                lbl.setPixmapWithData(pix, applied, self._data_format)  # Pass reconstruction and format
                row, col = divmod(i, grid)
                self.image_layout.addWidget(lbl, row, col)

            self.logger.debug(f"Rendered {count} views with mask {mask_idx}")
        finally:
            # Notify status manager that reconstruction is finished
            if self.status_manager:
                self.status_manager.finish_task()

    def update_info(self, applicator_name: str, dataset_type: str, num_masks: int, img_dim: int):
        """Updates info labels within the widget."""
        self.applicator_name_info_value.setText(applicator_name)
        self.dataset_type_info_value.setText(dataset_type)
        self.mask_size_info_value.setText(str(num_masks))
        self.image_dimension_info_value.setText(f"{img_dim}×{img_dim}")

    def set_data_format(self, data_format: str):
        """Set the data format for hover display."""
        self._data_format = data_format

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        """Normalize image to uint8 [0-255] for visualization."""
        # Handle all float types (float16, float32, float64)
        if np.issubdtype(img.dtype, np.floating):
            mn, mx = float(img.min()), float(img.max())
            denom = (mx - mn) if (mx - mn) > 0 else 1.0
            return ((img.astype(np.float32) - mn) / denom * 255).astype(np.uint8)
        elif img.dtype == np.uint8:
            return img
        else:
            # Convert other integer types to uint8
            return img.astype(np.uint8)

    def _show_context_menu(self, pos):
        """Show context menu with Save As option."""
        mask_idx = self.select_image_slider_value.value()
        if not self._dataset or not self._masks or self._applicator is None:
            return

        menu = QMenu(self)
        save_action = menu.addAction("Save As...")

        action = menu.exec_(self.preview_applicator_graphics.mapToGlobal(pos))
        if action == save_action:
            self._save_reconstruction(mask_idx)

    def _save_reconstruction(self, mask_idx: int):
        """Save reconstructed image to file at full resolution without compression."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Reconstruction",
            f"reconstruction_mask_{mask_idx}.png",
            "PNG Image (*.png);;TIFF Image (*.tiff *.tif);;All Files (*)"
        )

        if not file_path:
            return

        try:
            # Apply mask and get reconstruction at full resolution
            applied = self._applicator.apply_mask_range(0, mask_idx, self._img_idx)
            arr = self._normalize(applied)

            h, w = arr.shape

            # Apply colormap
            norm = arr.astype(np.float32) / 255.0
            rgba = self.cmap(norm)
            rgb = (rgba[..., :3] * 255).astype(np.uint8)

            from PyQt5.QtGui import QImage
            data = rgb.tobytes()
            bytes_per_line = 3 * w
            qimg = QImage(data, w, h, bytes_per_line, QImage.Format_RGB888)
            qimg.save(file_path)
            self.logger.info(f"Reconstruction saved to {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to save reconstruction: {e}")
            QtWidgets.QMessageBox.warning(self, "Save Error", f"Failed to save reconstruction: {e}")
