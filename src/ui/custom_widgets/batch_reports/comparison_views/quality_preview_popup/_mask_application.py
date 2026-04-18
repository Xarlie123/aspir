"""Popup showing mask-by-mask reconstruction for the quality preview."""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from matplotlib import cm
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
)

from ui.custom_widgets.batch_reports.comparison_views.quality_preview_popup._reconstruction import (
    ITERATIVE_METHODS_AVAILABLE,
    reconstruct_fista,
    reconstruct_ghost_imaging,
    reconstruct_pseudoinverse,
    reconstruct_tv_norm,
)


class MaskApplicationPopup(QDialog):
    """
    Popup showing mask application evolution.

    Shows how masks are progressively applied to create the noisy image
    using the actual reconstruction algorithm.

    Supported methods:
    - Ghost Imaging (Conventional): x̂ = (1/N) * Σ(B_i - B_avg) * M_i
    - Pseudoinverse: x = S^+ @ y
    - FISTA: minimize ||y - S@x||² + λ||x||₁ (slow, iterative)
    - TV Norm: minimize ||y - S@x||² + λ TV(x) (slow, iterative)
    """

    # Fixed display size for images
    DISPLAY_SIZE = 250

    # Slow methods that show a warning
    SLOW_METHODS = {"FISTA", "TV Norm"}

    def __init__(self, original: np.ndarray, reconstructed: np.ndarray,
                 masks: Optional[np.ndarray] = None, image_idx: int = 0,
                 test_name: str = "", reconstruction_method: str = "Ghost Imaging",
                 logger=None, parent=None):
        super().__init__(parent)

        self.logger = logger.getChild("MaskApplicationPopup") if logger else logging.getLogger("MaskApplicationPopup")
        self._original = original.astype(np.float64) if original is not None else None
        self._reconstructed = reconstructed
        self._masks = masks
        self._image_idx = image_idx
        self._test_name = test_name
        self._reconstruction_method = reconstruction_method
        self._current_mask_idx = 0

        # Determine effective method for preview
        self._effective_method = self._get_effective_method()

        # Precompute data needed for reconstruction
        self._measurements = None
        self._masks_matrix = None
        if self._original is not None and self._masks is not None and len(self._masks) > 0:
            self._precompute_data()

        self.cmap = cm.get_cmap('hot')
        self._setup_ui()
        self._update_display()

    def _get_effective_method(self) -> str:
        """Determine which reconstruction method to use."""
        method_lower = self._reconstruction_method.lower()

        if "pseudoinverse" in method_lower:
            return "Pseudoinverse"
        elif "fista" in method_lower:
            if ITERATIVE_METHODS_AVAILABLE:
                return "FISTA"
            else:
                self.logger.warning("pylops/pyproximal not available, falling back to Ghost Imaging")
                return "Ghost Imaging"
        elif "tv" in method_lower:
            if ITERATIVE_METHODS_AVAILABLE:
                return "TV Norm"
            else:
                self.logger.warning("pylops/pyproximal not available, falling back to Ghost Imaging")
                return "Ghost Imaging"
        elif "ghost" in method_lower or "conventional" in method_lower:
            return "Ghost Imaging"
        else:
            # Default to Ghost Imaging
            return "Ghost Imaging"

    def _precompute_data(self):
        """Precompute data needed for reconstruction methods."""
        # Precompute measurements for all methods
        self._measurements = []
        for mask in self._masks:
            mask_float = mask.astype(np.float64)
            measurement = (self._original * mask_float).sum()
            self._measurements.append(measurement)

        # For methods that need the masks matrix, precompute it
        if self._effective_method in {"Pseudoinverse", "FISTA", "TV Norm"}:
            self._masks_matrix = np.array([
                mask.flatten().astype(np.float64) for mask in self._masks
            ], dtype=np.float64)

    def _setup_ui(self):
        self.setWindowTitle(f"Mask Application - {self._test_name} - Image {self._image_idx}")
        self.setMinimumSize(850, 500)
        self.resize(950, 550)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Title
        title = QLabel("<h3>Mask Application Visualization</h3>")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Images side by side
        images_layout = QHBoxLayout()
        images_layout.setSpacing(15)

        # Original image
        orig_container = QVBoxLayout()
        orig_title = QLabel("Ground-Truth")
        orig_title.setAlignment(Qt.AlignCenter)
        orig_title.setStyleSheet("font-weight: bold;")
        orig_container.addWidget(orig_title)
        self.orig_label = QLabel()
        self.orig_label.setAlignment(Qt.AlignCenter)
        self.orig_label.setFixedSize(self.DISPLAY_SIZE, self.DISPLAY_SIZE)
        self.orig_label.setStyleSheet("background-color: #2a2a2a; border: 1px solid #555;")
        orig_container.addWidget(self.orig_label)
        orig_container.addStretch()
        images_layout.addLayout(orig_container)

        # Current mask preview (if available)
        mask_container = QVBoxLayout()
        self.mask_title = QLabel("Mask Pattern (0)")
        self.mask_title.setAlignment(Qt.AlignCenter)
        self.mask_title.setStyleSheet("font-weight: bold;")
        mask_container.addWidget(self.mask_title)
        self.mask_label = QLabel()
        self.mask_label.setAlignment(Qt.AlignCenter)
        self.mask_label.setFixedSize(self.DISPLAY_SIZE, self.DISPLAY_SIZE)
        self.mask_label.setStyleSheet("background-color: #2a2a2a; border: 1px solid #555;")
        mask_container.addWidget(self.mask_label)
        mask_container.addStretch()
        images_layout.addLayout(mask_container)

        # Progressive reconstruction using actual reconstruction method
        recon_container = QVBoxLayout()
        self.recon_title = QLabel("Reconstruction (0 masks)")
        self.recon_title.setAlignment(Qt.AlignCenter)
        self.recon_title.setStyleSheet("font-weight: bold;")
        recon_container.addWidget(self.recon_title)
        self.recon_label = QLabel()
        self.recon_label.setAlignment(Qt.AlignCenter)
        self.recon_label.setFixedSize(self.DISPLAY_SIZE, self.DISPLAY_SIZE)
        self.recon_label.setStyleSheet("background-color: #2a2a2a; border: 1px solid #555;")
        recon_container.addWidget(self.recon_label)
        recon_container.addStretch()
        images_layout.addLayout(recon_container)

        layout.addLayout(images_layout, 1)

        # Mask slider
        slider_group = QGroupBox("Mask Navigation")
        slider_layout = QHBoxLayout(slider_group)
        slider_layout.setSpacing(10)

        slider_label = QLabel("Masks applied:")
        slider_layout.addWidget(slider_label)

        self.mask_slider = QSlider(Qt.Horizontal)
        self.mask_slider.setMinimum(0)
        self.mask_slider.setMaximum(0)
        self.mask_slider.setValue(0)
        self.mask_slider.valueChanged.connect(self._on_mask_slider_changed)
        slider_layout.addWidget(self.mask_slider, 1)

        self.mask_idx_label = QLabel("0 / 0")
        self.mask_idx_label.setMinimumWidth(80)
        slider_layout.addWidget(self.mask_idx_label)

        layout.addWidget(slider_group)

        # Info label with warning for slow methods
        info_text = f"Method: {self._effective_method}"
        if self._masks is not None:
            info_text += f" | {len(self._masks)} masks"

        self.info_label = QLabel(info_text)
        self.info_label.setAlignment(Qt.AlignCenter)

        # Show warning for slow methods
        if self._effective_method in self.SLOW_METHODS:
            self.info_label.setStyleSheet("color: #FF6600; font-size: 11px; font-weight: bold;")
            warning_label = QLabel("⚠️ Slow method - reconstruction may take time when moving the slider")
            warning_label.setStyleSheet("color: #FF6600; font-size: 10px;")
            warning_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(warning_label)
        else:
            self.info_label.setStyleSheet("color: #666; font-size: 11px;")

        layout.addWidget(self.info_label)

        # Close button
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

        # Setup slider range
        if self._masks is not None and len(self._masks) > 0:
            self.mask_slider.setMaximum(len(self._masks) - 1)
            self.mask_idx_label.setText(f"0 / {len(self._masks) - 1}")
        else:
            self.mask_label.setText("No masks\navailable")
            self.recon_label.setText("No masks\navailable")

    def _on_mask_slider_changed(self, idx: int):
        """Handle mask slider change."""
        self._current_mask_idx = idx
        n_masks = len(self._masks) if self._masks is not None else 0
        self.mask_idx_label.setText(f"{idx} / {n_masks - 1 if n_masks > 0 else 0}")
        self.mask_title.setText(f"Mask Pattern ({idx})")
        self.recon_title.setText(f"Reconstruction ({idx + 1} masks)")
        self._update_mask_display()
        self._update_progressive_display()

    def _update_display(self):
        """Update all image displays."""
        if self._original is not None:
            self._display_image(self._original, self.orig_label)
        else:
            self.orig_label.setText("Not available")

        self._update_mask_display()
        self._update_progressive_display()

    def _update_mask_display(self):
        """Update the current mask pattern display."""
        if self._masks is not None and self._current_mask_idx < len(self._masks):
            mask = self._masks[self._current_mask_idx]
            self._display_image(mask, self.mask_label, use_grayscale=True)
        else:
            self.mask_label.setText("No mask")

    def _update_progressive_display(self):
        """Update the progressive reconstruction display using the appropriate method."""
        if self._original is None:
            self.recon_label.setText("No original")
            return

        if self._measurements is None or self._masks is None:
            self.recon_label.setText("No masks")
            return

        idx_max = self._current_mask_idx + 1  # inclusive (masks 0 to current)
        if idx_max <= 0:
            self.recon_label.setText("Move slider")
            return

        # Show busy cursor for slow methods
        if self._effective_method in self.SLOW_METHODS:
            QApplication.setOverrideCursor(Qt.WaitCursor)

        try:
            shape = self._original.shape
            if self._effective_method == "Pseudoinverse":
                reconstructed = reconstruct_pseudoinverse(
                    idx_max, self._measurements, self._masks, self._masks_matrix,
                    shape, self.logger
                )
            elif self._effective_method == "FISTA":
                reconstructed = reconstruct_fista(
                    idx_max, self._measurements, self._masks, self._masks_matrix,
                    shape, self.logger
                )
            elif self._effective_method == "TV Norm":
                reconstructed = reconstruct_tv_norm(
                    idx_max, self._measurements, self._masks, self._masks_matrix,
                    shape, self.logger
                )
            else:
                # Ghost Imaging (default)
                reconstructed = reconstruct_ghost_imaging(
                    idx_max, self._measurements, self._masks, shape
                )

            if reconstructed is not None:
                self._display_image(reconstructed, self.recon_label)
            else:
                self.recon_label.setText("Reconstruction failed")
        finally:
            # Restore cursor for slow methods
            if self._effective_method in self.SLOW_METHODS:
                QApplication.restoreOverrideCursor()

    def _display_image(self, arr: np.ndarray, label: QLabel, use_grayscale: bool = False):
        """Display an image without interpolation using fixed size."""
        arr = np.array(arr, copy=False)

        if arr.ndim == 0 or arr.size == 0:
            label.setText("No image")
            return

        # Normalize to [0, 1]
        amin, amax = arr.min(), arr.max()
        norm = (arr - amin) / (amax - amin) if amax > amin else np.zeros_like(arr, dtype=float)

        if use_grayscale:
            # Grayscale for masks
            gray = (norm * 255).astype(np.uint8)
            h, w = gray.shape
            qimg = QImage(gray.data, w, h, w, QImage.Format_Grayscale8)
        else:
            # Thermal colormap for images
            rgba = self.cmap(norm)
            rgb = (rgba[..., :3] * 255).astype(np.uint8)
            h, w = rgb.shape[:2]
            data = rgb.tobytes()
            bytes_per_line = 3 * w
            qimg = QImage(data, w, h, bytes_per_line, QImage.Format_RGB888)

        pix = QPixmap.fromImage(qimg)

        # Scale to fixed size without interpolation (FastTransformation = nearest neighbor)
        display_size = self.DISPLAY_SIZE - 4
        pix = pix.scaled(display_size, display_size, Qt.KeepAspectRatio, Qt.FastTransformation)

        label.setPixmap(pix)
