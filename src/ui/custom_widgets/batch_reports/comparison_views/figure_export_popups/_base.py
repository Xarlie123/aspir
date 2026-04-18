"""Base dialog and shared UI primitives for figure export popups."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
)

from ui.utils.file_formats import BATCH_TESTS_DIR


class BaseFigureExportPopup(QDialog):
    """Base class for figure export popups."""

    # Available colormaps
    COLORMAPS = ['hot', 'viridis', 'gray', 'inferno', 'magma', 'plasma', 'jet', 'turbo']

    def __init__(self, tests: list[dict[str, Any]], logger=None, parent=None):
        super().__init__(parent)
        self._tests = tests
        if logger:
            self.logger = logger.getChild(self.__class__.__name__)
        else:
            self.logger = logging.getLogger(self.__class__.__name__)

        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint)

    def _create_colormap_combo(self) -> QComboBox:
        """Create a colormap selection combobox."""
        combo = QComboBox()
        combo.addItems(self.COLORMAPS)
        combo.setCurrentText('hot')
        return combo

    def _create_test_selector(self, multi_select: bool = False) -> QListWidget:
        """Create a test selector list widget."""
        list_widget = QListWidget()
        if multi_select:
            list_widget.setSelectionMode(QAbstractItemView.ExtendedSelection)
        else:
            list_widget.setSelectionMode(QAbstractItemView.SingleSelection)

        for i, test in enumerate(self._tests):
            name = test.get("name", f"Test {i+1}")
            exp_name = test.get("_experiment_name", "")
            display = f"{name} ({exp_name})" if exp_name else name
            item = QListWidgetItem(display)
            item.setData(Qt.UserRole, i)
            list_widget.addItem(item)

        if list_widget.count() > 0:
            list_widget.setCurrentRow(0)

        return list_widget

    def _save_figure(self, figure: Figure, default_name: str):
        """Save a figure to file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Figure",
            str(BATCH_TESTS_DIR / default_name),
            "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg);;All Files (*.*)"
        )

        if file_path:
            try:
                figure.savefig(file_path, dpi=300, bbox_inches='tight',
                              facecolor='white', edgecolor='none')
                self.logger.info("Saved figure to %s", file_path)
                QMessageBox.information(self, "Saved", f"Figure saved to:\n{file_path}")
            except Exception as e:
                self.logger.error("Failed to save figure: %s", e)
                QMessageBox.warning(self, "Error", f"Failed to save figure:\n{e}")

    def _load_test_images(self, test: dict) -> tuple:
        """
        Load images from NPZ file for a test.
        Returns: (originals, reconstructions, denoised) numpy arrays or (None, None, None)
        """
        # Use _original_name for file path (survives renames in UI)
        original_name = test.get("_original_name", test.get("name", ""))
        batch_dir = test.get("_batch_dir")

        if not batch_dir:
            self.logger.warning("No batch_dir in test data")
            return None, None, None

        batch_dir = Path(batch_dir)

        # Sanitize test name for file path (use original name, not renamed)
        safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in original_name)

        # Try to load from NPZ file
        test_images_path = batch_dir / "data" / safe_name / "test_images.npz"

        if not test_images_path.exists():
            self.logger.warning("Test images not found: %s", test_images_path)
            return None, None, None

        try:
            data = np.load(str(test_images_path))
            originals = data.get("originals")
            reconstructions = data.get("reconstructions")
            denoised = data.get("denoised")
            self.logger.debug("Loaded images from %s", test_images_path)
            return originals, reconstructions, denoised
        except Exception as e:
            self.logger.error("Failed to load images: %s", e)
            return None, None, None

    def _get_num_images(self, test: dict) -> int:
        """Get the number of images in a test."""
        quality_per_image = test.get("quality_per_image", {})
        # quality_per_image is a dict like {"psnr_noisy": [...], "psnr_denoised": [...], ...}
        # The length of any list gives us the number of images
        for _, values in quality_per_image.items():
            if isinstance(values, list):
                return len(values)
        return 0

    def _get_max_num_images(self) -> int:
        """Get the maximum number of images across all tests."""
        max_images = 0
        for test in self._tests:
            num = self._get_num_images(test)
            if num > max_images:
                max_images = num
        return max_images if max_images > 0 else 10  # Default to 10 if no images found


class DropIndicatorWidget(QFrame):
    """Visual indicator for drop position (shared by column-list widgets)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(4)
        self.setMinimumHeight(80)
        self.setStyleSheet("""
            DropIndicatorWidget {
                background-color: #2196F3;
                border-radius: 2px;
            }
        """)
        self.hide()
