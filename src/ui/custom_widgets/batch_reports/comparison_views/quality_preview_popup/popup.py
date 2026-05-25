"""Quality Metrics Preview popup — main dialog for Batch Reports."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
from matplotlib import cm
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QDialog, QLabel

from ui.custom_widgets.batch_reports.comparison_views.quality_preview_popup._mask_application import (
    MaskApplicationPopup,
)
from ui.custom_widgets.batch_reports.comparison_views.quality_preview_popup._ui_builder import (
    build_ui,
)


class QualityPreviewPopup(QDialog):
    """
    Popup dialog for previewing per-image quality metrics in Batch Reports.

    Features:
    - Test selection dropdown
    - Image slider for navigating through images
    - Side-by-side display of Ground-Truth, Noisy, Denoised images
    - Per-image metrics table with change percentages
    - Bar chart showing normalized quality scores
    - Button to view mask application visualization
    """

    # Fixed display size for images
    IMAGE_DISPLAY_SIZE = 220

    def __init__(self, tests: list[dict[str, Any]], logger=None, parent=None):
        super().__init__(parent)

        if logger:
            self.logger = logger.getChild("QualityPreviewPopup")
        else:
            self.logger = logging.getLogger("QualityPreviewPopup")

        self._tests = tests
        self._current_test = None
        self._current_per_image = {}
        self._current_idx = 0

        # Cached images for current test
        self._originals = None
        self._reconstructions = None
        self._denoised = None
        self._masks = None

        # Thermal colormap for images
        self.cmap = cm.get_cmap('hot')

        build_ui(self)
        self._populate_tests()

    def _populate_tests(self):
        """Populate the test selection dropdown."""
        self.test_combo.clear()

        for test in self._tests:
            per_image = test.get("quality_per_image", {})
            if per_image:
                test_name = test.get("name", "Unknown Test")
                n_images = len(per_image.get("psnr_noisy", []))
                self.test_combo.addItem(f"{test_name} ({n_images} images)", test)

        if self.test_combo.count() > 0:
            self.test_combo.setCurrentIndex(0)
        else:
            self.info_label.setText("No tests with per-image data available")

    def _load_test_images(self, test: dict[str, Any]) -> bool:
        """Load test images from the NPZ file if available."""
        self._originals = None
        self._reconstructions = None
        self._denoised = None
        self._masks = None

        experiment_path = test.get("_experiment_path")
        if not experiment_path:
            self.logger.debug("No experiment path in test data")
            return False

        report_path = Path(experiment_path)
        batch_dir = report_path.parent

        test_name = test.get("name", "Unknown")
        safe_name = test_name.replace(" ", "_").replace("/", "-")

        test_images_path = batch_dir / "data" / safe_name / "test_images.npz"
        masks_path = batch_dir / "data" / safe_name / "masks.npz"

        self.logger.debug("Looking for images at: %s", test_images_path)

        if test_images_path.exists():
            try:
                data = np.load(str(test_images_path))
                self._originals = data.get("originals")
                self._reconstructions = data.get("reconstructions")
                self._denoised = data.get("denoised")

                self.logger.info(
                    "Loaded test images: originals=%s, reconstructions=%s, denoised=%s",
                    self._originals.shape if self._originals is not None else None,
                    self._reconstructions.shape if self._reconstructions is not None else None,
                    self._denoised.shape if self._denoised is not None else None
                )

                if masks_path.exists():
                    try:
                        mask_data = np.load(str(masks_path))
                        self._masks = mask_data.get("masks")
                        self.logger.debug("Loaded masks: %s", self._masks.shape if self._masks is not None else None)
                    except Exception as e:
                        self.logger.warning("Failed to load masks: %s", e)

                return True

            except Exception as e:
                self.logger.error("Failed to load test images: %s", e)
                return False
        else:
            self.logger.debug("Test images file not found: %s", test_images_path)
            return False

    def _on_test_changed(self, index: int):
        """Handle test selection change."""
        if index < 0:
            return

        test = self.test_combo.itemData(index)
        if not test:
            return

        self._current_test = test
        self._current_per_image = test.get("quality_per_image", {})

        images_loaded = self._load_test_images(test)

        if images_loaded:
            self.data_status_label.setText("Images loaded")
            self.data_status_label.setStyleSheet("color: #228B22; font-style: italic;")
            # Enable mask button if masks are available
            self.mask_btn.setEnabled(self._masks is not None and len(self._masks) > 0)
        else:
            self.data_status_label.setText("Images not available (metrics only)")
            self.data_status_label.setStyleSheet("color: #666; font-style: italic;")
            self.mask_btn.setEnabled(False)

        n_images = len(self._current_per_image.get("psnr_noisy", []))
        self.image_slider.setMaximum(max(0, n_images - 1))
        self.image_slider.setValue(0)
        self.index_label.setText(f"0  ({n_images} images)")

        if n_images > 0:
            self._on_slider_changed(0)
            self.info_label.setText(f"Showing {n_images} images from {test.get('name', 'test')}")
        else:
            self._clear_displays()
            self.info_label.setText("No images available for this test")

    def _on_view_mask_application(self):
        """Open the mask application visualization popup."""
        if self._originals is None or self._reconstructions is None:
            return

        idx = self._current_idx
        if idx >= len(self._originals) or idx >= len(self._reconstructions):
            return

        test_name = self._current_test.get("name", "Unknown") if self._current_test else "Unknown"
        reconstruction_method = self._current_test.get("reconstruction_method", "Ghost Imaging") if self._current_test else "Ghost Imaging"

        popup = MaskApplicationPopup(
            original=self._originals[idx],
            reconstructed=self._reconstructions[idx],
            masks=self._masks,
            image_idx=idx,
            test_name=test_name,
            reconstruction_method=reconstruction_method,
            logger=self.logger,
            parent=self
        )
        popup.exec()

    def _on_slider_changed(self, idx: int):
        """Handle image slider change."""
        self._current_idx = idx
        per_image = self._current_per_image
        n_images = len(per_image.get("psnr_noisy", []))

        self.index_label.setText(f"{idx}  ({n_images} images)")

        if not (0 <= idx < n_images):
            return

        self._update_metrics(idx)
        self._update_bar_chart(idx)
        self._update_images(idx)

    def _update_metrics(self, idx: int):
        """Update the metrics display for the current image."""
        per_image = self._current_per_image

        psnr_noisy = per_image.get("psnr_noisy", [])
        psnr_denoised = per_image.get("psnr_denoised", [])
        psnr_n = psnr_noisy[idx] if idx < len(psnr_noisy) else None
        psnr_d = psnr_denoised[idx] if idx < len(psnr_denoised) else None
        self.psnr_noisy_display.setText(f"{psnr_n:.2f}" if psnr_n is not None else "-")
        self.psnr_recon_display.setText(f"{psnr_d:.2f}" if psnr_d is not None else "-")
        self._set_change_label(self.psnr_change_display, psnr_n, psnr_d, higher_is_better=True)

        ssim_noisy = per_image.get("ssim_noisy", [])
        ssim_denoised = per_image.get("ssim_denoised", [])
        ssim_n = ssim_noisy[idx] if idx < len(ssim_noisy) else None
        ssim_d = ssim_denoised[idx] if idx < len(ssim_denoised) else None
        self.ssim_noisy_display.setText(f"{ssim_n:.4f}" if ssim_n is not None else "-")
        self.ssim_recon_display.setText(f"{ssim_d:.4f}" if ssim_d is not None else "-")
        self._set_change_label(self.ssim_change_display, ssim_n, ssim_d, higher_is_better=True)

        lpips_noisy = per_image.get("lpips_noisy", [])
        lpips_denoised = per_image.get("lpips_denoised", [])
        lpips_n = lpips_noisy[idx] if idx < len(lpips_noisy) else None
        lpips_d = lpips_denoised[idx] if idx < len(lpips_denoised) else None
        self.lpips_noisy_display.setText(f"{lpips_n:.4f}" if lpips_n is not None else "-")
        self.lpips_recon_display.setText(f"{lpips_d:.4f}" if lpips_d is not None else "-")
        self._set_change_label(self.lpips_change_display, lpips_n, lpips_d, higher_is_better=False)

    def _set_change_label(self, label: QLabel, noisy_val: Optional[float],
                          recon_val: Optional[float], higher_is_better: bool):
        """Set the change label with color based on improvement."""
        if noisy_val is None or recon_val is None or noisy_val == 0:
            label.setText("-")
            label.setStyleSheet("font-size: 13px; font-weight: bold;")
            return

        pct_change = ((recon_val - noisy_val) / abs(noisy_val)) * 100
        is_improvement = pct_change > 0 if higher_is_better else pct_change < 0
        text = f"+{pct_change:.1f}%" if pct_change >= 0 else f"{pct_change:.1f}%"

        if is_improvement:
            style = "font-size: 13px; font-weight: bold; color: #228B22;"
        else:
            style = "font-size: 13px; font-weight: bold; color: #DC143C;"

        label.setText(text)
        label.setStyleSheet(style)

    def _update_bar_chart(self, idx: int):
        """Update the bar chart for the current image."""
        self.bar_figure.clear()
        per_image = self._current_per_image

        psnr_n = per_image.get("psnr_noisy", [])[idx] if idx < len(per_image.get("psnr_noisy", [])) else 0
        psnr_d = per_image.get("psnr_denoised", [])[idx] if idx < len(per_image.get("psnr_denoised", [])) else 0
        ssim_n = per_image.get("ssim_noisy", [])[idx] if idx < len(per_image.get("ssim_noisy", [])) else 0
        ssim_d = per_image.get("ssim_denoised", [])[idx] if idx < len(per_image.get("ssim_denoised", [])) else 0
        lpips_n = per_image.get("lpips_noisy", [])[idx] if idx < len(per_image.get("lpips_noisy", [])) else 0
        lpips_d = per_image.get("lpips_denoised", [])[idx] if idx < len(per_image.get("lpips_denoised", [])) else 0

        metrics = [
            ('PSNR \u2191', psnr_n, psnr_d, psnr_n / 50.0, psnr_d / 50.0, '#1f77b4', '{:.1f}'),
            ('SSIM \u2191', ssim_n, ssim_d, ssim_n, ssim_d, '#2ca02c', '{:.3f}'),
            ('LPIPS \u2193', lpips_n, lpips_d, 1.0 - lpips_n, 1.0 - lpips_d, '#d62728', '{:.3f}'),
        ]

        ax = self.bar_figure.add_subplot(111)
        x = np.arange(2)
        n_metrics = 3
        width = 0.8 / n_metrics

        for i, (name, noisy_val, recon_val, noisy_norm, recon_norm, color, fmt) in enumerate(metrics):
            offset = (i - (n_metrics - 1) / 2) * width
            bars = ax.bar(x + offset, [noisy_norm, recon_norm], width * 0.9,
                          label=name, color=color, alpha=0.8)
            ax.text(bars[0].get_x() + bars[0].get_width()/2, bars[0].get_height() + 0.02,
                    fmt.format(noisy_val), ha='center', va='bottom', fontsize=7)
            ax.text(bars[1].get_x() + bars[1].get_width()/2, bars[1].get_height() + 0.02,
                    fmt.format(recon_val), ha='center', va='bottom', fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(['Noisy', 'Denoised'], fontsize=9)
        ax.set_ylabel('Quality Score', fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.set_title(f'Image {idx}', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
                  ncol=3, fontsize=7, frameon=False)

        self.bar_figure.subplots_adjust(left=0.12, right=0.95, top=0.88, bottom=0.22)
        self.bar_canvas.draw()

    def _update_images(self, idx: int):
        """Update the image displays."""
        if self._originals is not None and idx < len(self._originals):
            self._display_image(self._originals[idx], self.orig_image)
        else:
            self.orig_image.setText("Not available")
            self.orig_image.setStyleSheet("background-color: #2a2a2a; border: 1px solid #555; color: #888;")

        if self._reconstructions is not None and idx < len(self._reconstructions):
            self._display_image(self._reconstructions[idx], self.noisy_image)
        else:
            self.noisy_image.setText("Not available")
            self.noisy_image.setStyleSheet("background-color: #2a2a2a; border: 1px solid #555; color: #888;")

        if self._denoised is not None and idx < len(self._denoised):
            self._display_image(self._denoised[idx], self.recon_image)
        else:
            self.recon_image.setText("Not available")
            self.recon_image.setStyleSheet("background-color: #2a2a2a; border: 1px solid #555; color: #888;")

    def _display_image(self, arr, label: QLabel):
        """Display an image with thermal colormap without interpolation using fixed size."""
        arr = np.array(arr, copy=False)

        if arr.ndim == 0 or arr.size == 0:
            label.setText("Empty")
            return

        amin, amax = arr.min(), arr.max()
        norm = (arr - amin) / (amax - amin) if amax > amin else np.zeros_like(arr, dtype=float)

        rgba = self.cmap(norm)
        rgb = (rgba[..., :3] * 255).astype(np.uint8)

        h, w = rgb.shape[:2]
        data = rgb.tobytes()
        bytes_per_line = 3 * w
        qimg = QImage(data, w, h, bytes_per_line, QImage.Format_RGB888)

        pix = QPixmap.fromImage(qimg)
        # Use fixed display size (FastTransformation = nearest neighbor, no interpolation)
        display_size = self.IMAGE_DISPLAY_SIZE - 4
        pix = pix.scaled(display_size, display_size, Qt.KeepAspectRatio, Qt.FastTransformation)
        label.setPixmap(pix)

    def _clear_displays(self):
        """Clear all displays."""
        self.orig_image.clear()
        self.noisy_image.clear()
        self.recon_image.clear()
        self.bar_figure.clear()
        self.bar_canvas.draw()

        self.psnr_noisy_display.setText("-")
        self.psnr_recon_display.setText("-")
        self.psnr_change_display.setText("-")
        self.ssim_noisy_display.setText("-")
        self.ssim_recon_display.setText("-")
        self.ssim_change_display.setText("-")
        self.lpips_noisy_display.setText("-")
        self.lpips_recon_display.setText("-")
        self.lpips_change_display.setText("-")
