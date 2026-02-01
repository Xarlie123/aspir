"""
Popup dialogs for exporting publication-quality figures from batch reports.
"""
import logging
import io
import base64
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox,
    QPushButton, QFileDialog, QMessageBox, QScrollArea,
    QGridLayout, QFrame, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QLineEdit, QSizePolicy, QSplitter, QListWidget,
    QListWidgetItem, QAbstractItemView, QTabWidget, QTextEdit,
    QWidget, QApplication, QColorDialog
)
from PyQt5.QtCore import Qt, pyqtSignal, QMimeData, QPoint, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QFont, QCursor, QDrag, QPixmap, QPainter, QColor

from ui.utils.file_formats import BATCH_TESTS_DIR

# Try to import pylops for iterative reconstruction methods
try:
    import pylops
    PYLOPS_AVAILABLE = True
except ImportError:
    PYLOPS_AVAILABLE = False


# ============================================================================
# Base Popup Class
# ============================================================================

class BaseFigureExportPopup(QDialog):
    """Base class for figure export popups."""

    # Available colormaps
    COLORMAPS = ['hot', 'viridis', 'gray', 'inferno', 'magma', 'plasma', 'jet', 'turbo']

    def __init__(self, tests: List[Dict[str, Any]], logger=None, parent=None):
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
        experiment_name = test.get("_experiment_name", "")
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
        for key, values in quality_per_image.items():
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


# ============================================================================
# Column Configuration for Visual Comparison
# ============================================================================

class ColumnConfig:
    """Configuration for a single column in the visual comparison figure."""

    # Image types
    TYPE_GROUND_TRUTH = "ground_truth"
    TYPE_LINEAR_RECON = "linear_recon"
    TYPE_ITERATIVE_CS = "iterative_cs"
    TYPE_LINEAR_RECON_DNN = "linear_recon_dnn"

    TYPE_LABELS = {
        TYPE_GROUND_TRUTH: "Ground Truth",
        TYPE_LINEAR_RECON: "Linear Recon.",
        TYPE_ITERATIVE_CS: "Iterative CS\n(TV-Norm)",
        TYPE_LINEAR_RECON_DNN: "Linear Reconstruction\n+ U-Net",
    }

    def __init__(self, col_type: str = TYPE_GROUND_TRUTH):
        self.col_type = col_type
        self.title = self.TYPE_LABELS.get(col_type, "Column")
        # Custom text for ground truth
        self.custom_text = "Reference Image" if col_type == self.TYPE_GROUND_TRUTH else ""
        # Metric checkboxes - default based on type
        if col_type == self.TYPE_GROUND_TRUTH:
            self.show_time = False
            self.show_psnr = False
            self.show_ssim = False
            self.show_lpips = False
        else:
            self.show_time = True
            self.show_psnr = True
            self.show_ssim = False
            self.show_lpips = False

    def get_default_title(self) -> str:
        """Get default title for current type."""
        return self.TYPE_LABELS.get(self.col_type, "Column")


class ColumnCardWidget(QFrame):
    """Draggable card widget representing a column configuration."""

    double_clicked = pyqtSignal(object)  # Emits self when double-clicked
    config_changed = pyqtSignal()
    drag_started = pyqtSignal(object)  # Emits self when drag starts

    def __init__(self, config: ColumnConfig, index: int, parent=None):
        super().__init__(parent)
        self.config = config
        self.index = index
        self._drag_start_pos = None
        self._mouse_pressed = False
        self._setup_ui()

    def _setup_ui(self):
        self.setFrameStyle(QFrame.Box | QFrame.Raised)
        self.setLineWidth(2)
        self.setMinimumSize(120, 80)
        self.setMaximumSize(150, 100)
        self.setCursor(Qt.OpenHandCursor)

        self._update_style()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(2)

        self.title_label = QLabel(self.config.title.replace('\n', ' '))
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setWordWrap(True)
        self.title_label.setStyleSheet("font-weight: bold; font-size: 10px;")
        layout.addWidget(self.title_label)

        self.type_label = QLabel(f"[{self.config.col_type.replace('_', ' ').title()}]")
        self.type_label.setAlignment(Qt.AlignCenter)
        self.type_label.setStyleSheet("font-size: 9px; color: #666;")
        layout.addWidget(self.type_label)

    def _update_style(self, dragging: bool = False):
        colors = {
            ColumnConfig.TYPE_GROUND_TRUTH: "#e8f5e9",
            ColumnConfig.TYPE_LINEAR_RECON: "#fff3e0",
            ColumnConfig.TYPE_ITERATIVE_CS: "#e3f2fd",
            ColumnConfig.TYPE_LINEAR_RECON_DNN: "#f3e5f5",
        }
        color = colors.get(self.config.col_type, "#f5f5f5")
        opacity = "0.4" if dragging else "1.0"
        self.setStyleSheet(f"""
            ColumnCardWidget {{
                background-color: {color};
                border: 2px solid #999;
                border-radius: 6px;
            }}
            ColumnCardWidget:hover {{
                border: 2px solid #333;
            }}
        """)
        # Set opacity effect for dragging state
        from PyQt5.QtWidgets import QGraphicsOpacityEffect
        if dragging:
            effect = QGraphicsOpacityEffect(self)
            effect.setOpacity(0.4)
            self.setGraphicsEffect(effect)
        else:
            self.setGraphicsEffect(None)

    def update_display(self):
        """Update display after config change."""
        self.title_label.setText(self.config.title.replace('\n', ' '))
        self.type_label.setText(f"[{self.config.col_type.replace('_', ' ').title()}]")
        self._update_style()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_start_pos = event.pos()
            self._mouse_pressed = True
            self.setCursor(Qt.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        # Only start drag if left button is actually pressed AND we have a start position
        if (self._drag_start_pos is not None and
            self._mouse_pressed and
            event.buttons() & Qt.LeftButton):
            if (event.pos() - self._drag_start_pos).manhattanLength() > 10:
                self._start_drag()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._drag_start_pos = None
        self._mouse_pressed = False
        self.setCursor(Qt.OpenHandCursor)
        super().mouseReleaseEvent(event)

    def _start_drag(self):
        """Start a drag operation with visual feedback."""
        if not self._mouse_pressed:
            return

        self._drag_start_pos = None
        self._mouse_pressed = False

        # Create pixmap of this card with transparency
        pixmap = QPixmap(self.size())
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setOpacity(0.7)
        self.render(painter)
        painter.end()

        # Create drag object
        drag = QDrag(self)
        mime_data = QMimeData()
        mime_data.setText(str(self.index))
        drag.setMimeData(mime_data)
        drag.setPixmap(pixmap)
        drag.setHotSpot(QPoint(pixmap.width() // 2, pixmap.height() // 2))

        # Make the original card semi-transparent while dragging
        self._update_style(dragging=True)
        self.drag_started.emit(self)

        # Execute drag
        result = drag.exec_(Qt.MoveAction)

        # Restore card appearance
        self._update_style(dragging=False)
        self.setCursor(Qt.OpenHandCursor)

    def mouseDoubleClickEvent(self, event):
        # Reset drag state before opening dialog
        self._drag_start_pos = None
        self._mouse_pressed = False
        self.setCursor(Qt.OpenHandCursor)
        self.double_clicked.emit(self)
        super().mouseDoubleClickEvent(event)


class ColumnConfigDialog(QDialog):
    """Dialog for configuring a column."""

    def __init__(self, config: ColumnConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self.setWindowTitle("Configure Column")
        self.setMinimumWidth(420)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        # Image Type
        type_group = QGroupBox("Image Type")
        type_layout = QVBoxLayout(type_group)

        self.type_combo = QComboBox()
        self.type_combo.addItem("Ground Truth", ColumnConfig.TYPE_GROUND_TRUTH)
        self.type_combo.addItem("Linear Reconstruction (Pseudoinverse)", ColumnConfig.TYPE_LINEAR_RECON)
        self.type_combo.addItem("Iterative CS (TV-Norm / Split Bregman)", ColumnConfig.TYPE_ITERATIVE_CS)
        self.type_combo.addItem("Linear Recon. + DNN", ColumnConfig.TYPE_LINEAR_RECON_DNN)

        # Set current
        for i in range(self.type_combo.count()):
            if self.type_combo.itemData(i) == self.config.col_type:
                self.type_combo.setCurrentIndex(i)
                break

        self.type_combo.currentIndexChanged.connect(self._on_type_changed)
        type_layout.addWidget(self.type_combo)

        # Warning for iterative methods
        self.iterative_warning = QLabel("⚠ Iterative methods may take several seconds to compute")
        self.iterative_warning.setStyleSheet("color: #ff9800; font-size: 11px;")
        self.iterative_warning.setVisible(self.config.col_type == ColumnConfig.TYPE_ITERATIVE_CS)
        type_layout.addWidget(self.iterative_warning)

        layout.addWidget(type_group)

        # Title
        title_group = QGroupBox("Column Title")
        title_layout = QVBoxLayout(title_group)

        self.title_edit = QLineEdit(self.config.title)
        title_layout.addWidget(self.title_edit)

        reset_btn = QPushButton("Reset to Default")
        reset_btn.clicked.connect(self._reset_title)
        title_layout.addWidget(reset_btn)

        layout.addWidget(title_group)

        # Bottom Text - Metrics selection with checkboxes
        bottom_group = QGroupBox("Bottom Text (below image)")
        bottom_layout = QVBoxLayout(bottom_group)

        # Time display (automatic based on type)
        time_row = QHBoxLayout()
        self.time_cb = QCheckBox("Show Time")
        self.time_cb.setChecked(self.config.show_time)
        time_row.addWidget(self.time_cb)

        self.time_info_label = QLabel()
        self.time_info_label.setStyleSheet("color: #666; font-size: 11px;")
        self._update_time_info_label()
        time_row.addWidget(self.time_info_label)
        time_row.addStretch()
        bottom_layout.addLayout(time_row)

        # Quality metrics checkboxes
        metrics_label = QLabel("Quality Metrics:")
        metrics_label.setStyleSheet("font-weight: bold; margin-top: 8px;")
        bottom_layout.addWidget(metrics_label)

        metrics_row = QHBoxLayout()
        self.psnr_cb = QCheckBox("PSNR")
        self.psnr_cb.setChecked(self.config.show_psnr)
        metrics_row.addWidget(self.psnr_cb)

        self.ssim_cb = QCheckBox("SSIM")
        self.ssim_cb.setChecked(self.config.show_ssim)
        metrics_row.addWidget(self.ssim_cb)

        self.lpips_cb = QCheckBox("LPIPS")
        self.lpips_cb.setChecked(self.config.show_lpips)
        metrics_row.addWidget(self.lpips_cb)

        metrics_row.addStretch()
        bottom_layout.addLayout(metrics_row)

        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        bottom_layout.addWidget(separator)

        # Custom text (for Ground Truth)
        custom_row = QHBoxLayout()
        custom_row.addWidget(QLabel("Custom text:"))
        self.custom_edit = QLineEdit(self.config.custom_text)
        self.custom_edit.setPlaceholderText("Optional additional text...")
        custom_row.addWidget(self.custom_edit)
        bottom_layout.addLayout(custom_row)

        layout.addWidget(bottom_group)

        # Update enabled state based on type
        self._update_metrics_enabled()

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        ok_btn = QPushButton("OK")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self._on_accept)
        btn_layout.addWidget(ok_btn)

        layout.addLayout(btn_layout)

    def _update_time_info_label(self):
        """Update the time info label based on current type."""
        col_type = self.type_combo.currentData()
        if col_type == ColumnConfig.TYPE_GROUND_TRUTH:
            self.time_info_label.setText("(N/A for Ground Truth)")
        elif col_type == ColumnConfig.TYPE_LINEAR_RECON:
            self.time_info_label.setText("(CPU time)")
        elif col_type == ColumnConfig.TYPE_ITERATIVE_CS:
            self.time_info_label.setText("(CPU time)")
        elif col_type == ColumnConfig.TYPE_LINEAR_RECON_DNN:
            self.time_info_label.setText("(CPU + GPU time)")

    def _update_metrics_enabled(self):
        """Enable/disable metrics based on image type."""
        col_type = self.type_combo.currentData()
        is_ground_truth = col_type == ColumnConfig.TYPE_GROUND_TRUTH

        # Disable all metrics for Ground Truth
        self.time_cb.setEnabled(not is_ground_truth)
        self.psnr_cb.setEnabled(not is_ground_truth)
        self.ssim_cb.setEnabled(not is_ground_truth)
        self.lpips_cb.setEnabled(not is_ground_truth)

        if is_ground_truth:
            self.time_cb.setChecked(False)
            self.psnr_cb.setChecked(False)
            self.ssim_cb.setChecked(False)
            self.lpips_cb.setChecked(False)

    def _on_type_changed(self):
        new_type = self.type_combo.currentData()
        # Update title if it was the default
        if self.title_edit.text() == self.config.get_default_title():
            self.config.col_type = new_type
            self.title_edit.setText(self.config.get_default_title())
        # Show/hide iterative warning
        self.iterative_warning.setVisible(new_type == ColumnConfig.TYPE_ITERATIVE_CS)
        # Update time info and metrics enabled state
        self._update_time_info_label()
        self._update_metrics_enabled()

    def _reset_title(self):
        self.config.col_type = self.type_combo.currentData()
        self.title_edit.setText(self.config.get_default_title())

    def _on_accept(self):
        self.config.col_type = self.type_combo.currentData()
        self.config.title = self.title_edit.text()
        self.config.custom_text = self.custom_edit.text()
        # Save checkbox states
        self.config.show_time = self.time_cb.isChecked()
        self.config.show_psnr = self.psnr_cb.isChecked()
        self.config.show_ssim = self.ssim_cb.isChecked()
        self.config.show_lpips = self.lpips_cb.isChecked()
        self.accept()


class DropIndicatorWidget(QFrame):
    """Visual indicator for drop position."""

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


class ColumnListWidget(QWidget):
    """Widget containing draggable column cards with visual drag & drop."""

    columns_changed = pyqtSignal()

    def __init__(self, tests: List[Dict], parent=None):
        super().__init__(parent)
        self.tests = tests
        self.columns: List[ColumnConfig] = []
        self.cards: List[ColumnCardWidget] = []
        self._dragged_card = None
        self._drop_index = -1

        self._setup_ui()

    def _setup_ui(self):
        self.setAcceptDrops(True)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Cards container
        self.cards_container = QWidget()
        self.cards_layout = QHBoxLayout(self.cards_container)
        self.cards_layout.setContentsMargins(5, 5, 5, 5)
        self.cards_layout.setSpacing(10)
        self.cards_layout.addStretch()

        # Drop indicator
        self.drop_indicator = DropIndicatorWidget(self.cards_container)

        main_layout.addWidget(self.cards_container)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(5, 0, 5, 5)

        add_btn = QPushButton("+ Add Column")
        add_btn.clicked.connect(self._add_column)
        btn_layout.addWidget(add_btn)

        remove_btn = QPushButton("- Remove Last")
        remove_btn.clicked.connect(self._remove_last_column)
        btn_layout.addWidget(remove_btn)

        btn_layout.addStretch()
        main_layout.addLayout(btn_layout)

        # Add default columns
        self._add_default_columns()

    def _add_default_columns(self):
        """Add default column configuration."""
        defaults = [
            ColumnConfig.TYPE_GROUND_TRUTH,
            ColumnConfig.TYPE_LINEAR_RECON,
            ColumnConfig.TYPE_ITERATIVE_CS,
            ColumnConfig.TYPE_LINEAR_RECON_DNN,
        ]

        for col_type in defaults:
            config = ColumnConfig(col_type)
            self._add_column_with_config(config)

    def _add_column(self):
        """Add a new column."""
        if len(self.columns) >= 6:
            return
        config = ColumnConfig(ColumnConfig.TYPE_LINEAR_RECON_DNN)
        self._add_column_with_config(config)
        self.columns_changed.emit()

    def _add_column_with_config(self, config: ColumnConfig):
        """Add a column with specific config."""
        self.columns.append(config)

        card = ColumnCardWidget(config, len(self.cards))
        card.double_clicked.connect(self._on_card_double_clicked)
        card.drag_started.connect(self._on_drag_started)
        self.cards.append(card)

        # Insert before stretch
        self.cards_layout.insertWidget(self.cards_layout.count() - 1, card)

    def _remove_last_column(self):
        """Remove the last column."""
        if len(self.columns) <= 1:
            return

        self.columns.pop()
        card = self.cards.pop()
        self.cards_layout.removeWidget(card)
        card.deleteLater()
        self.columns_changed.emit()

    def _on_card_double_clicked(self, card: ColumnCardWidget):
        """Open config dialog for card."""
        dialog = ColumnConfigDialog(card.config, self)
        if dialog.exec_() == QDialog.Accepted:
            card.update_display()
            self.columns_changed.emit()

    def _on_drag_started(self, card: ColumnCardWidget):
        """Track which card is being dragged."""
        self._dragged_card = card

    def _get_drop_index(self, pos: QPoint) -> int:
        """Get the drop index based on cursor position."""
        # Map position to cards_container coordinates
        container_pos = self.cards_container.mapFrom(self, pos)
        x = container_pos.x()

        # Find insertion point
        for i, card in enumerate(self.cards):
            card_rect = card.geometry()
            card_center = card_rect.center().x()
            if x < card_center:
                return i

        return len(self.cards)

    def _show_drop_indicator(self, drop_index: int):
        """Show the drop indicator at the specified index."""
        if drop_index < 0 or not self.cards:
            self.drop_indicator.hide()
            return

        # Calculate indicator position
        if drop_index < len(self.cards):
            target_card = self.cards[drop_index]
            x = target_card.geometry().left() - 7
        else:
            target_card = self.cards[-1]
            x = target_card.geometry().right() + 3

        y = self.cards[0].geometry().top()
        height = self.cards[0].geometry().height()

        self.drop_indicator.setFixedHeight(height)
        self.drop_indicator.move(x, y)
        self.drop_indicator.show()
        self.drop_indicator.raise_()

    def dragEnterEvent(self, event):
        """Handle drag enter - accept if it's from our cards."""
        if event.mimeData().hasText():
            event.acceptProposedAction()
            self._drop_index = self._get_drop_index(event.pos())
            self._show_drop_indicator(self._drop_index)

    def dragMoveEvent(self, event):
        """Handle drag move - update drop indicator position."""
        if event.mimeData().hasText():
            event.acceptProposedAction()
            new_drop_index = self._get_drop_index(event.pos())
            if new_drop_index != self._drop_index:
                self._drop_index = new_drop_index
                self._show_drop_indicator(self._drop_index)

    def dragLeaveEvent(self, event):
        """Handle drag leave - hide indicator."""
        self.drop_indicator.hide()
        self._drop_index = -1

    def dropEvent(self, event):
        """Handle drop - reorder cards."""
        self.drop_indicator.hide()

        if not event.mimeData().hasText() or self._dragged_card is None:
            return

        event.acceptProposedAction()

        old_idx = self.cards.index(self._dragged_card)
        new_idx = self._drop_index

        # Adjust index if dropping after the original position
        if new_idx > old_idx:
            new_idx -= 1

        if old_idx != new_idx and 0 <= new_idx < len(self.cards):
            # Remove from old position
            self.cards.remove(self._dragged_card)
            self.columns.remove(self._dragged_card.config)

            # Insert at new position
            self.cards.insert(new_idx, self._dragged_card)
            self.columns.insert(new_idx, self._dragged_card.config)

            # Update layout
            self.cards_layout.removeWidget(self._dragged_card)
            self.cards_layout.insertWidget(new_idx, self._dragged_card)

            # Update card indices
            for i, card in enumerate(self.cards):
                card.index = i

            self.columns_changed.emit()

        self._dragged_card = None
        self._drop_index = -1

    def get_columns(self) -> List[ColumnConfig]:
        """Get current column configurations in order."""
        return self.columns.copy()


# ============================================================================
# Visual Comparison Popup (Fig 9 style)
# ============================================================================

class VisualComparisonPopup(BaseFigureExportPopup):
    """
    Popup for generating Visual Comparison figure.
    Shows configurable columns with: Ground Truth | Linear Recon | Iterative CS | DNN Output
    With configurable PSNR and timing metrics below each image.
    """

    def __init__(self, tests: List[Dict[str, Any]], logger=None, parent=None):
        super().__init__(tests, logger, parent)
        self.setWindowTitle("Visual Comparison Figure")
        self.setMinimumSize(1100, 800)
        self.resize(1200, 850)

        self._figure = None
        self._canvas = None
        self._images_cache = {}  # Cache loaded images
        self._setup_ui()

    def _setup_ui(self):
        """Setup the popup UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)

        # Title
        title = QLabel("Visual Comparison of Reconstruction Methods")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(title)

        desc = QLabel("Double-click on a column card to configure it. Drag cards to reorder.")
        desc.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(desc)

        # Column configuration area
        config_group = QGroupBox("Column Configuration (drag to reorder, double-click to edit)")
        config_layout = QVBoxLayout(config_group)
        config_layout.setContentsMargins(10, 15, 10, 10)

        self.column_list = ColumnListWidget(self._tests)
        self.column_list.columns_changed.connect(self._update_preview)
        config_layout.addWidget(self.column_list)

        layout.addWidget(config_group)

        # Options row
        options_layout = QHBoxLayout()

        # Test selection (common for all columns)
        options_layout.addWidget(QLabel("Source Test:"))
        self.test_combo = QComboBox()
        for i, test in enumerate(self._tests):
            name = test.get("name", f"Test {i+1}")
            exp_name = test.get("_experiment_name", "")
            display = f"{name} ({exp_name})" if exp_name else name
            self.test_combo.addItem(display, i)
        self.test_combo.currentIndexChanged.connect(self._update_preview)
        options_layout.addWidget(self.test_combo)

        options_layout.addSpacing(20)

        # Image selection
        options_layout.addWidget(QLabel("Image index:"))
        self.image_spin = QSpinBox()
        self.image_spin.setMinimum(0)
        max_images = self._get_max_num_images()
        self.image_spin.setMaximum(max(0, max_images - 1))
        self.image_spin.setValue(0)
        self.image_spin.valueChanged.connect(self._update_preview)
        options_layout.addWidget(self.image_spin)

        options_layout.addSpacing(20)

        # Colormap
        options_layout.addWidget(QLabel("Colormap:"))
        self.cmap_combo = self._create_colormap_combo()
        self.cmap_combo.currentTextChanged.connect(self._update_preview)
        options_layout.addWidget(self.cmap_combo)

        options_layout.addStretch()

        # Save button
        self.save_btn = QPushButton("Save Figure...")
        self.save_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 8px 20px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.save_btn.clicked.connect(self._on_save)
        options_layout.addWidget(self.save_btn)

        layout.addLayout(options_layout)

        # Preview
        preview_label = QLabel("Preview:")
        preview_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(preview_label)

        self._figure = Figure(figsize=(12, 4), dpi=100)
        self._canvas = FigureCanvas(self._figure)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self._canvas, 1)

        # Close button
        close_layout = QHBoxLayout()
        close_layout.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        close_layout.addWidget(close_btn)
        layout.addLayout(close_layout)

        # Initial preview
        self._update_preview()

    def _load_masks(self, test: dict) -> Optional[np.ndarray]:
        """Load masks from NPZ file for a test."""
        batch_dir = test.get("_batch_dir")
        # Use _original_name for file path (survives renames in UI)
        original_name = test.get("_original_name", test.get("name", ""))

        if not batch_dir:
            return None

        batch_dir = Path(batch_dir)
        safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in original_name)
        masks_path = batch_dir / "data" / safe_name / "masks.npz"

        if not masks_path.exists():
            self.logger.warning("Masks not found: %s", masks_path)
            return None

        try:
            data = np.load(str(masks_path))
            masks = data.get("masks")
            self.logger.debug("Loaded masks from %s", masks_path)
            return masks
        except Exception as e:
            self.logger.error("Failed to load masks: %s", e)
            return None

    def _compute_pseudoinverse_reconstruction(self, original: np.ndarray, masks: np.ndarray) -> np.ndarray:
        """Compute pseudoinverse (linear) reconstruction from ground truth and masks."""
        H, W = original.shape[:2]
        n_masks = masks.shape[0]

        # Compute measurements from ground truth
        measurements = np.array([np.sum(original * masks[i]) for i in range(n_masks)])

        # Flatten masks to sensing matrix S: (n_masks, H*W)
        S = masks.reshape(n_masks, -1)

        # Compute pseudoinverse reconstruction
        try:
            S_pinv = np.linalg.pinv(S)
            reconstructed = S_pinv @ measurements
            reconstructed = reconstructed.reshape(H, W)

            # Normalize to [0, 1]
            reconstructed = np.clip(reconstructed, 0, 1)
            return reconstructed
        except Exception as e:
            self.logger.error("Pseudoinverse failed: %s", e)
            return original

    def _compute_fista_reconstruction(self, original: np.ndarray, masks: np.ndarray) -> np.ndarray:
        """
        Compute TV-regularized reconstruction from ground truth and masks.
        Uses Total Variation norm which is more appropriate for images than L1.

        Solves: minimize ||y - S @ x||_2^2 + λ TV(x)
        """
        self.logger.info("_compute_fista_reconstruction called: original shape=%s, masks shape=%s",
                        original.shape, masks.shape)

        if not PYLOPS_AVAILABLE:
            self.logger.warning("pylops not available, falling back to pseudoinverse")
            return self._compute_pseudoinverse_reconstruction(original, masks)

        try:
            import pyproximal
        except ImportError:
            self.logger.warning("pyproximal not available, falling back to pseudoinverse")
            return self._compute_pseudoinverse_reconstruction(original, masks)

        H, W = original.shape[:2]
        n_masks = masks.shape[0]

        # Compute measurements from ground truth
        y = np.array([np.sum(original.astype(np.float64) * masks[i].astype(np.float64))
                      for i in range(n_masks)], dtype=np.float64)

        # Flatten masks to sensing matrix S: (n_masks, H*W)
        S = np.array([m.flatten().astype(np.float64) for m in masks], dtype=np.float64)
        M, N = S.shape

        try:
            # Create pylops LinearOperator
            Sop = pylops.MatrixMult(S)

            # Use Split Bregman for TV-regularized reconstruction
            # This is more appropriate for images than L1 on pixels
            from pylops.optimization.sparsity import splitbregman

            # Create 2D gradient operator for TV norm
            Gop = pylops.Gradient(dims=(H, W), edge=True, kind='forward')

            # TV regularization parameter - higher values = smoother result
            mu = 0.1  # Data fidelity weight
            lamda = 0.05  # TV regularization weight

            self.logger.info("Running Split Bregman TV reconstruction...")

            # Run Split Bregman algorithm for TV minimization
            # Returns (x, niter, cost)
            x_tv, niter, cost = splitbregman(
                Sop,
                y,
                [Gop],
                niter_outer=20,
                niter_inner=5,
                mu=mu,
                epsRL1s=[lamda],
                tol=1e-4,
                show=False
            )

            self.logger.info("Split Bregman completed in %d iterations", niter)

            # Reshape back to image
            reconstructed = x_tv.reshape(H, W)

            # Normalize to [0, 1]
            reconstructed = np.clip(reconstructed, 0, 1)
            return reconstructed.astype(np.float32)

        except Exception as e:
            self.logger.error("TV reconstruction failed: %s, trying L1-FISTA", e)
            # Fallback to L1-FISTA with higher regularization
            return self._compute_fista_l1_fallback(original, masks, S, y, H, W)

    def _compute_fista_l1_fallback(self, original: np.ndarray, masks: np.ndarray,
                                    S: np.ndarray, y: np.ndarray, H: int, W: int) -> np.ndarray:
        """Fallback L1-FISTA with higher regularization for visible difference."""
        try:
            import pyproximal

            Sop = pylops.MatrixMult(S)

            # Higher regularization for visible difference
            l2 = pyproximal.proximal.L2(Op=Sop, b=y)
            lam = 0.1  # Higher lambda for visible sparsity effect
            l1 = lam * pyproximal.proximal.L1()

            L_val = np.abs((Sop.H * Sop).eigs(1)[0])
            tau = 0.95 / L_val

            x0 = np.zeros(H * W, dtype=np.float64)

            opt = pyproximal.optimization.primal.ProximalGradient(
                l2, l1,
                tau=tau,
                x0=x0,
                niter=200,
                acceleration='fista',
                show=False
            )

            x_fista = opt if isinstance(opt, np.ndarray) else (opt.run() if hasattr(opt, 'run') else opt.x)
            reconstructed = x_fista.reshape(H, W)
            reconstructed = np.clip(reconstructed, 0, 1)
            return reconstructed.astype(np.float32)

        except Exception as e:
            self.logger.error("L1-FISTA fallback failed: %s, using pseudoinverse", e)
            return self._compute_pseudoinverse_reconstruction(original, masks)

    def _compute_fista_alternative(self, original: np.ndarray, masks: np.ndarray,
                                    S: np.ndarray, measurements: np.ndarray,
                                    H: int, W: int) -> np.ndarray:
        """Alternative FISTA implementation using scipy (backup method)."""
        from scipy.optimize import minimize

        lambd = 1e-3

        def objective(x):
            residual = S @ x - measurements
            return 0.5 * np.sum(residual ** 2) + lambd * np.sum(np.abs(x))

        def gradient(x):
            residual = S @ x - measurements
            return S.T @ residual + lambd * np.sign(x)

        # Initial guess from pseudoinverse
        x0 = np.linalg.lstsq(S, measurements, rcond=None)[0]

        try:
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                jac=gradient,
                options={'maxiter': 100, 'disp': False}
            )
            reconstructed = result.x.reshape(H, W)
            reconstructed = np.clip(reconstructed, 0, 1)
            return reconstructed.astype(np.float32)
        except Exception as e:
            self.logger.error("Alternative FISTA failed: %s", e)
            return self._compute_pseudoinverse_reconstruction(original, masks)

    def _compute_psnr(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """Compute PSNR between original and reconstructed images."""
        original = np.asarray(original, dtype=np.float64)
        reconstructed = np.asarray(reconstructed, dtype=np.float64)

        # Ensure same shape
        if original.shape != reconstructed.shape:
            self.logger.warning("Shape mismatch: original=%s, reconstructed=%s",
                              original.shape, reconstructed.shape)
            return 0.0

        # Compute MSE
        mse = np.mean((original - reconstructed) ** 2)
        if mse == 0:
            return float('inf')

        # Assume data range is [0, 1]
        max_val = 1.0
        psnr = 10.0 * np.log10((max_val ** 2) / mse)
        return psnr

    def _get_image_for_column(self, config: ColumnConfig, test_idx: int, image_idx: int):
        """Get the appropriate image for a column configuration."""
        if test_idx >= len(self._tests):
            return None

        test = self._tests[test_idx]
        test_name = test.get("name", "")

        # Check cache for loaded images
        cache_key = (test_name, test_idx)
        if cache_key not in self._images_cache:
            originals, reconstructions, denoised = self._load_test_images(test)
            self._images_cache[cache_key] = {
                'originals': originals,
                'reconstructions': reconstructions,
                'denoised': denoised,
                'masks': None,  # Loaded on demand
                'fista_cache': {},  # Cache computed FISTA results (images)
                'pinv_cache': {},   # Cache computed pseudoinverse results (images)
                'fista_metrics': {},  # Cache metrics: {image_idx: {'psnr': ..., 'time_ms': ...}}
                'pinv_metrics': {},   # Cache metrics for pseudoinverse
            }

        cached = self._images_cache[cache_key]
        originals = cached['originals']
        reconstructions = cached['reconstructions']
        denoised = cached['denoised']

        # Select appropriate image based on type
        if config.col_type == ColumnConfig.TYPE_GROUND_TRUTH:
            if originals is not None and image_idx < len(originals):
                return originals[image_idx]

        elif config.col_type == ColumnConfig.TYPE_LINEAR_RECON:
            # Compute pseudoinverse reconstruction on-the-fly
            self.logger.info("LINEAR_RECON: originals=%s, image_idx=%d",
                           originals is not None, image_idx)
            if originals is not None and image_idx < len(originals):
                if image_idx in cached['pinv_cache']:
                    self.logger.info("LINEAR_RECON: Returning cached pinv result")
                    return cached['pinv_cache'][image_idx]

                # Load masks if not loaded
                if cached['masks'] is None:
                    self.logger.info("LINEAR_RECON: Loading masks...")
                    cached['masks'] = self._load_masks(test)

                self.logger.info("LINEAR_RECON: masks loaded = %s",
                               cached['masks'] is not None)
                if cached['masks'] is not None:
                    self.logger.info("LINEAR_RECON: Computing pseudoinverse...")
                    QApplication.setOverrideCursor(Qt.WaitCursor)
                    try:
                        original = originals[image_idx]
                        t_start = time.perf_counter()
                        recon = self._compute_pseudoinverse_reconstruction(
                            original, cached['masks']
                        )
                        t_end = time.perf_counter()
                        time_ms = (t_end - t_start) * 1000

                        # Compute PSNR
                        psnr = self._compute_psnr(original, recon)

                        # Cache results and metrics
                        cached['pinv_cache'][image_idx] = recon
                        cached['pinv_metrics'][image_idx] = {
                            'psnr': psnr,
                            'time_ms': time_ms
                        }
                        self.logger.info("LINEAR_RECON: Pseudoinverse complete, PSNR=%.2f dB, time=%.1f ms",
                                       psnr, time_ms)
                        return recon
                    finally:
                        QApplication.restoreOverrideCursor()
                elif reconstructions is not None and image_idx < len(reconstructions):
                    # Fallback to stored reconstructions
                    self.logger.info("LINEAR_RECON: Using stored reconstructions (fallback)")
                    return reconstructions[image_idx]

        elif config.col_type == ColumnConfig.TYPE_ITERATIVE_CS:
            # Compute TV-norm reconstruction on-the-fly
            self.logger.info("ITERATIVE_CS: originals=%s, image_idx=%d",
                           originals is not None, image_idx)
            if originals is not None and image_idx < len(originals):
                if image_idx in cached['fista_cache']:
                    self.logger.info("ITERATIVE_CS: Returning cached TV-norm result")
                    return cached['fista_cache'][image_idx]

                # Load masks if not loaded
                if cached['masks'] is None:
                    self.logger.info("ITERATIVE_CS: Loading masks...")
                    cached['masks'] = self._load_masks(test)

                self.logger.info("ITERATIVE_CS: masks loaded = %s",
                               cached['masks'] is not None)
                if cached['masks'] is not None:
                    self.logger.info("ITERATIVE_CS: Computing TV-norm with %d masks...",
                                   len(cached['masks']))
                    QApplication.setOverrideCursor(Qt.WaitCursor)
                    try:
                        original = originals[image_idx]
                        t_start = time.perf_counter()
                        recon = self._compute_fista_reconstruction(
                            original, cached['masks']
                        )
                        t_end = time.perf_counter()
                        time_ms = (t_end - t_start) * 1000

                        # Compute PSNR
                        psnr = self._compute_psnr(original, recon)

                        # Cache results and metrics
                        cached['fista_cache'][image_idx] = recon
                        cached['fista_metrics'][image_idx] = {
                            'psnr': psnr,
                            'time_ms': time_ms
                        }
                        self.logger.info("ITERATIVE_CS: TV-norm complete, PSNR=%.2f dB, time=%.1f ms",
                                       psnr, time_ms)
                        return recon
                    finally:
                        QApplication.restoreOverrideCursor()
                else:
                    self.logger.warning("ITERATIVE_CS: No masks available, cannot compute TV-norm")

        elif config.col_type == ColumnConfig.TYPE_LINEAR_RECON_DNN:
            if denoised is not None and image_idx < len(denoised):
                return denoised[image_idx]

        return None

    def _get_bottom_text(self, config: ColumnConfig, test_idx: int, image_idx: int = 0) -> str:
        """Get the bottom text for a column based on checkbox selections."""
        parts = []

        # Custom text (always shown if provided)
        if config.custom_text:
            parts.append(config.custom_text)

        # Ground Truth only shows custom text
        if config.col_type == ColumnConfig.TYPE_GROUND_TRUTH:
            return "\n".join(parts) if parts else ""

        if test_idx >= len(self._tests):
            return "\n".join(parts) if parts else ""

        test = self._tests[test_idx]
        test_name = test.get("name", "")

        # Check if we have computed metrics in the cache
        cache_key = (test_name, test_idx)
        cached = self._images_cache.get(cache_key, {})

        # Get computed metrics for Linear Recon and Iterative CS
        computed_metrics = None
        if config.col_type == ColumnConfig.TYPE_LINEAR_RECON:
            computed_metrics = cached.get('pinv_metrics', {}).get(image_idx)
        elif config.col_type == ColumnConfig.TYPE_ITERATIVE_CS:
            computed_metrics = cached.get('fista_metrics', {}).get(image_idx)

        # Add time if checkbox is checked
        if config.show_time:
            time_text = self._get_time_text(config, test, computed_metrics)
            if time_text:
                parts.append(time_text)

        # Add PSNR if checkbox is checked
        if config.show_psnr:
            if config.col_type == ColumnConfig.TYPE_LINEAR_RECON_DNN:
                psnr = test.get("psnr_denoised")
            elif computed_metrics:
                psnr = computed_metrics.get('psnr')
            else:
                psnr = test.get("psnr_recons")

            if psnr is not None:
                parts.append(f"PSNR: {psnr:.2f} dB")

        # Add SSIM if checkbox is checked
        if config.show_ssim:
            if config.col_type == ColumnConfig.TYPE_LINEAR_RECON_DNN:
                ssim = test.get("ssim_denoised")
            elif computed_metrics:
                # Compute SSIM if we have the images
                ssim = self._get_computed_ssim(cached, image_idx, config.col_type)
            else:
                ssim = test.get("ssim_recons")

            if ssim is not None:
                parts.append(f"SSIM: {ssim:.4f}")

        # Add LPIPS if checkbox is checked
        if config.show_lpips:
            if config.col_type == ColumnConfig.TYPE_LINEAR_RECON_DNN:
                lpips = test.get("lpips_denoised")
            else:
                lpips = test.get("lpips_recons")

            if lpips is not None:
                parts.append(f"LPIPS: {lpips:.4f}")

        return "\n".join(parts)

    def _get_time_text(self, config: ColumnConfig, test: dict, computed_metrics: dict = None) -> str:
        """Get appropriate time text based on column type (automatic)."""
        if config.col_type == ColumnConfig.TYPE_GROUND_TRUTH:
            return ""
        elif config.col_type == ColumnConfig.TYPE_LINEAR_RECON:
            # Use computed metrics if available
            if computed_metrics and 'time_ms' in computed_metrics:
                return f"Time: {computed_metrics['time_ms']:.1f} ms (CPU)"
            recon_time = test.get("timing_reconstruction_ms")
            if recon_time is not None:
                return f"Time: {recon_time:.1f} ms (CPU)"
        elif config.col_type == ColumnConfig.TYPE_ITERATIVE_CS:
            # Use computed metrics if available (this is the actual TV-norm time)
            if computed_metrics and 'time_ms' in computed_metrics:
                return f"Time: {computed_metrics['time_ms']:.1f} ms (CPU)"
            # Fallback to stored time (but this is usually for pseudoinverse)
            recon_time = test.get("timing_reconstruction_ms")
            if recon_time is not None:
                return f"Time: {recon_time:.1f} ms (CPU)"
        elif config.col_type == ColumnConfig.TYPE_LINEAR_RECON_DNN:
            # For DNN: reconstruction time (CPU) + inference time (GPU)
            recon_time = test.get("timing_reconstruction_ms", 0) or 0
            gpu_time = test.get("timing_gpu_mean_ms", 0) or 0
            total = recon_time + gpu_time
            return f"Time: {total:.1f} ms (CPU+GPU)"
        return ""

    def _get_computed_ssim(self, cached: dict, image_idx: int, col_type: str) -> Optional[float]:
        """Get or compute SSIM for a reconstructed image."""
        # Check if we already have it cached
        if col_type == ColumnConfig.TYPE_LINEAR_RECON:
            metrics = cached.get('pinv_metrics', {}).get(image_idx, {})
            if 'ssim' in metrics:
                return metrics['ssim']
            # Compute if we have the images
            recon = cached.get('pinv_cache', {}).get(image_idx)
        elif col_type == ColumnConfig.TYPE_ITERATIVE_CS:
            metrics = cached.get('fista_metrics', {}).get(image_idx, {})
            if 'ssim' in metrics:
                return metrics['ssim']
            recon = cached.get('fista_cache', {}).get(image_idx)
        else:
            return None

        originals = cached.get('originals')
        if recon is None or originals is None or image_idx >= len(originals):
            return None

        try:
            from skimage.metrics import structural_similarity as ssim_func
            original = np.asarray(originals[image_idx], dtype=np.float64)
            reconstructed = np.asarray(recon, dtype=np.float64)

            # Ensure 2D
            if original.ndim == 3:
                original = original.squeeze()
            if reconstructed.ndim == 3:
                reconstructed = reconstructed.squeeze()

            ssim_val = ssim_func(original, reconstructed, data_range=1.0)

            # Cache the result
            if col_type == ColumnConfig.TYPE_LINEAR_RECON:
                if image_idx not in cached.get('pinv_metrics', {}):
                    cached.setdefault('pinv_metrics', {})[image_idx] = {}
                cached['pinv_metrics'][image_idx]['ssim'] = ssim_val
            elif col_type == ColumnConfig.TYPE_ITERATIVE_CS:
                if image_idx not in cached.get('fista_metrics', {}):
                    cached.setdefault('fista_metrics', {})[image_idx] = {}
                cached['fista_metrics'][image_idx]['ssim'] = ssim_val

            return ssim_val
        except ImportError:
            self.logger.warning("skimage not available for SSIM computation")
            return None
        except Exception as e:
            self.logger.error("Failed to compute SSIM: %s", e)
            return None

    def _update_preview(self):
        """Update the preview figure."""
        self._figure.clear()

        columns = self.column_list.get_columns()
        if not columns:
            ax = self._figure.add_subplot(111)
            ax.text(0.5, 0.5, "Add columns to preview",
                   ha='center', va='center', fontsize=14, color='#999')
            ax.axis('off')
            self._canvas.draw()
            return

        n_cols = len(columns)
        cmap = self.cmap_combo.currentText()
        image_idx = self.image_spin.value()
        test_idx = self.test_combo.currentData() if self.test_combo.count() > 0 else 0

        # Create subplots with more bottom margin for text
        self._figure.subplots_adjust(bottom=0.15, top=0.88, left=0.02, right=0.98, wspace=0.1)
        axes = self._figure.subplots(1, n_cols)
        if n_cols == 1:
            axes = [axes]

        for col, config in enumerate(columns):
            ax = axes[col]

            # Get image using the common test index
            img = self._get_image_for_column(config, test_idx, image_idx)

            if img is not None:
                img = np.array(img)
                if img.ndim == 3 and img.shape[-1] == 1:
                    img = img.squeeze(-1)
                ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
            else:
                ax.text(0.5, 0.5, "No image", ha='center', va='center',
                       fontsize=10, color='#999')
                ax.set_facecolor('#f0f0f0')

            ax.axis('off')

            # Title
            ax.set_title(config.title, fontsize=11, fontweight='bold', pad=8)

            # Bottom text (pass image_idx to get metrics for the correct image)
            bottom_text = self._get_bottom_text(config, test_idx, image_idx)
            if bottom_text:
                ax.text(0.5, -0.08, bottom_text,
                       ha='center', va='top', transform=ax.transAxes,
                       fontsize=9)

        self._canvas.draw()

    def _on_save(self):
        """Save the current figure."""
        self._save_figure(self._figure, "visual_comparison.png")


# ============================================================================
# Quality per Sampling Ratio Popup (Fig 8 style)
# ============================================================================

class QualityRowConfig:
    """Configuration for a row in the quality vs sampling ratio figure."""

    def __init__(self, test_idx: int = 0, label: str = ""):
        self.test_idx = test_idx  # Index in the tests list
        self.label = label  # Custom label (empty = use test name)


class QualityRowWidget(QFrame):
    """Widget for configuring a single row in the quality figure.

    Compact 2-line layout:
    - Line 1: Row N: Test: [combo box]
    - Line 2: Label: [text field] [delete button]
    """

    changed = pyqtSignal()
    remove_requested = pyqtSignal(object)

    def __init__(self, tests: List[Dict[str, Any]], row_num: int, parent=None):
        super().__init__(parent)
        self._tests = tests
        self._row_num = row_num
        self._config = QualityRowConfig()
        self._setup_ui()

    def _setup_ui(self):
        self.setFrameStyle(QFrame.Box | QFrame.Raised)
        self.setStyleSheet("""
            QFrame {
                background-color: #fafafa;
                border: 1px solid #ddd;
                border-radius: 4px;
                margin: 2px;
            }
        """)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 6, 8, 6)
        main_layout.setSpacing(4)

        # Line 1: Row N: Test: [combo]
        line1 = QHBoxLayout()
        line1.setSpacing(6)

        self.row_label = QLabel(f"Row {self._row_num}:")
        self.row_label.setStyleSheet("font-weight: bold; min-width: 50px;")
        line1.addWidget(self.row_label)

        line1.addWidget(QLabel("Test:"))
        self.test_combo = QComboBox()
        self.test_combo.setMinimumWidth(200)
        for i, test in enumerate(self._tests):
            name = test.get("name", f"Test {i+1}")
            exp_name = test.get("_experiment_name", "")
            display = f"{name} ({exp_name})" if exp_name else name
            self.test_combo.addItem(display, i)
        self.test_combo.currentIndexChanged.connect(self._on_config_changed)
        line1.addWidget(self.test_combo, 1)

        main_layout.addLayout(line1)

        # Line 2: Label: [text field] [delete button]
        line2 = QHBoxLayout()
        line2.setSpacing(6)

        line2.addSpacing(56)  # Align with test combo
        line2.addWidget(QLabel("Label:"))
        self.label_edit = QLineEdit()
        self.label_edit.setPlaceholderText("Auto (from test name)")
        self.label_edit.textChanged.connect(self._on_config_changed)
        line2.addWidget(self.label_edit, 1)

        # Remove button
        remove_btn = QPushButton("✕")
        remove_btn.setFixedSize(24, 24)
        remove_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                border-radius: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
        """)
        remove_btn.clicked.connect(lambda: self.remove_requested.emit(self))
        line2.addWidget(remove_btn)

        main_layout.addLayout(line2)

    def _on_config_changed(self):
        """Emit changed signal when any config changes."""
        self._config.test_idx = self.test_combo.currentData()
        self._config.label = self.label_edit.text()
        self.changed.emit()

    def get_config(self) -> QualityRowConfig:
        """Get the current row configuration."""
        self._config.test_idx = self.test_combo.currentData()
        self._config.label = self.label_edit.text()
        return self._config

    def set_test_index(self, idx: int):
        """Set the test index."""
        if 0 <= idx < self.test_combo.count():
            self.test_combo.setCurrentIndex(idx)

    def set_row_number(self, num: int):
        """Update the row number display."""
        self._row_num = num
        self.row_label.setText(f"Row {num}:")


class QualitySamplingRatioPopup(BaseFigureExportPopup):
    """
    Popup for generating Quality per Sampling Ratio figure.
    Shows a table with rows for different sampling ratios and columns for
    SPI Reconstructed, Denoised, and Quality Metrics.

    Features:
    - Ground Truth centered at top with title
    - Column headers for the data table
    - Per-row test selection with editable labels
    - Single image index for entire table
    - Optional table lines (horizontal/vertical separators)
    - Configurable metrics to display
    - Same layout system as SamplesGrid (pixel-based, Fit Window)
    """

    def __init__(self, tests: List[Dict[str, Any]], logger=None, parent=None):
        super().__init__(tests, logger, parent)
        self.setWindowTitle("Quality vs Sampling Ratio Figure")
        self.setMinimumSize(1100, 800)
        self.resize(1200, 850)

        self._figure = None
        self._canvas = None
        self._figure_dpi = 100
        self._natural_width_px = 800
        self._natural_height_px = 600
        self._row_widgets: List[QualityRowWidget] = []
        self._setup_ui()

    def _setup_ui(self):
        """Setup the popup UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)

        # Title
        title = QLabel("Quality Metrics Across Sampling Ratios")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(title)

        desc = QLabel(
            "Configure rows to compare different sampling ratios. "
            "Ground Truth is displayed centered at the top."
        )
        desc.setStyleSheet("color: #666;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Main splitter
        splitter = QSplitter(Qt.Horizontal)

        # Left panel: Configuration
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 10, 0)

        # Rows configuration (taller)
        rows_group = QGroupBox("Figure Rows")
        rows_layout = QVBoxLayout(rows_group)

        # Add row button
        add_row_layout = QHBoxLayout()
        add_row_btn = QPushButton("+ Add Row")
        add_row_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 6px 12px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        add_row_btn.clicked.connect(self._add_row)
        add_row_layout.addWidget(add_row_btn)
        add_row_layout.addStretch()
        rows_layout.addLayout(add_row_layout)

        # Scroll area for rows (taller to show more rows)
        self._rows_scroll = QScrollArea()
        self._rows_scroll.setWidgetResizable(True)
        self._rows_scroll.setMinimumHeight(350)
        self._rows_scroll.setStyleSheet("QScrollArea { border: none; }")

        self._rows_container = QWidget()
        self._rows_vlayout = QVBoxLayout(self._rows_container)
        self._rows_vlayout.setContentsMargins(0, 0, 0, 0)
        self._rows_vlayout.setSpacing(4)
        self._rows_vlayout.addStretch()

        self._rows_scroll.setWidget(self._rows_container)
        rows_layout.addWidget(self._rows_scroll, 1)

        left_layout.addWidget(rows_group, 1)

        # Display Options (compact grid like SamplesGrid)
        options_group = QGroupBox("Display Options")
        options_layout = QGridLayout(options_group)
        options_layout.setContentsMargins(8, 12, 8, 8)
        options_layout.setSpacing(6)

        # Image index (single for whole table)
        options_layout.addWidget(QLabel("Image index:"), 0, 0)
        self.image_spin = QSpinBox()
        self.image_spin.setMinimum(0)
        max_images = self._get_max_num_images()
        self.image_spin.setMaximum(max(0, max_images - 1))
        self.image_spin.setValue(0)
        self.image_spin.valueChanged.connect(self._update_preview)
        options_layout.addWidget(self.image_spin, 0, 1)

        # Image size (in pixels)
        options_layout.addWidget(QLabel("Image size (px):"), 0, 2)
        self.image_size_spin = QSpinBox()
        self.image_size_spin.setMinimum(32)
        self.image_size_spin.setMaximum(256)
        self.image_size_spin.setValue(80)
        self.image_size_spin.setSingleStep(8)
        self.image_size_spin.valueChanged.connect(self._update_preview)
        options_layout.addWidget(self.image_size_spin, 0, 3)

        # Row gap
        options_layout.addWidget(QLabel("Row gap (px):"), 1, 0)
        self.row_spacing_spin = QSpinBox()
        self.row_spacing_spin.setMinimum(0)
        self.row_spacing_spin.setMaximum(100)
        self.row_spacing_spin.setValue(5)
        self.row_spacing_spin.valueChanged.connect(self._update_preview)
        options_layout.addWidget(self.row_spacing_spin, 1, 1)

        # Column gap
        options_layout.addWidget(QLabel("Col gap (px):"), 1, 2)
        self.col_spacing_spin = QSpinBox()
        self.col_spacing_spin.setMinimum(0)
        self.col_spacing_spin.setMaximum(100)
        self.col_spacing_spin.setValue(5)
        self.col_spacing_spin.valueChanged.connect(self._update_preview)
        options_layout.addWidget(self.col_spacing_spin, 1, 3)

        # Colormap
        options_layout.addWidget(QLabel("Colormap:"), 2, 0)
        self.cmap_combo = self._create_colormap_combo()
        self.cmap_combo.currentTextChanged.connect(self._update_preview)
        options_layout.addWidget(self.cmap_combo, 2, 1)

        # Font size
        options_layout.addWidget(QLabel("Font size:"), 2, 2)
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setMinimum(6)
        self.font_size_spin.setMaximum(20)
        self.font_size_spin.setValue(10)
        self.font_size_spin.valueChanged.connect(self._update_preview)
        options_layout.addWidget(self.font_size_spin, 2, 3)

        # Table lines checkbox
        self.show_lines_cb = QCheckBox("Show table lines")
        self.show_lines_cb.setChecked(True)
        self.show_lines_cb.stateChanged.connect(self._update_preview)
        options_layout.addWidget(self.show_lines_cb, 3, 0, 1, 2)

        # Line width
        options_layout.addWidget(QLabel("Line width:"), 3, 2)
        self.line_width_spin = QSpinBox()
        self.line_width_spin.setMinimum(1)
        self.line_width_spin.setMaximum(5)
        self.line_width_spin.setValue(1)
        self.line_width_spin.valueChanged.connect(self._update_preview)
        options_layout.addWidget(self.line_width_spin, 3, 3)

        # Image padding (space between image and cell border)
        options_layout.addWidget(QLabel("Image padding:"), 4, 0)
        self.image_padding_spin = QSpinBox()
        self.image_padding_spin.setMinimum(0)
        self.image_padding_spin.setMaximum(20)
        self.image_padding_spin.setValue(4)
        self.image_padding_spin.valueChanged.connect(self._update_preview)
        options_layout.addWidget(self.image_padding_spin, 4, 1)

        left_layout.addWidget(options_group)

        # Column titles
        titles_group = QGroupBox("Column Titles")
        titles_layout = QGridLayout(titles_group)
        titles_layout.setContentsMargins(8, 12, 8, 8)
        titles_layout.setSpacing(4)

        titles_layout.addWidget(QLabel("Col 1:"), 0, 0)
        self.col1_title_edit = QLineEdit("Sampling\nRatio")
        self.col1_title_edit.textChanged.connect(self._update_preview)
        titles_layout.addWidget(self.col1_title_edit, 0, 1)

        titles_layout.addWidget(QLabel("Col 2:"), 1, 0)
        self.col2_title_edit = QLineEdit("SPI\nReconstructed\nImage")
        self.col2_title_edit.textChanged.connect(self._update_preview)
        titles_layout.addWidget(self.col2_title_edit, 1, 1)

        titles_layout.addWidget(QLabel("Col 3:"), 2, 0)
        self.col3_title_edit = QLineEdit("Denoised\nImage")
        self.col3_title_edit.textChanged.connect(self._update_preview)
        titles_layout.addWidget(self.col3_title_edit, 2, 1)

        titles_layout.addWidget(QLabel("Col 4:"), 3, 0)
        self.col4_title_edit = QLineEdit("Quality Metrics\n(Denoised vs GT)")
        self.col4_title_edit.textChanged.connect(self._update_preview)
        titles_layout.addWidget(self.col4_title_edit, 3, 1)

        left_layout.addWidget(titles_group)

        # Metrics selection
        metrics_group = QGroupBox("Quality Metrics to Show")
        metrics_layout = QHBoxLayout(metrics_group)

        self.show_psnr_cb = QCheckBox("PSNR")
        self.show_psnr_cb.setChecked(True)
        self.show_psnr_cb.stateChanged.connect(self._update_preview)
        metrics_layout.addWidget(self.show_psnr_cb)

        self.show_ssim_cb = QCheckBox("SSIM")
        self.show_ssim_cb.setChecked(True)
        self.show_ssim_cb.stateChanged.connect(self._update_preview)
        metrics_layout.addWidget(self.show_ssim_cb)

        self.show_lpips_cb = QCheckBox("LPIPS")
        self.show_lpips_cb.setChecked(True)
        self.show_lpips_cb.stateChanged.connect(self._update_preview)
        metrics_layout.addWidget(self.show_lpips_cb)

        metrics_layout.addStretch()
        left_layout.addWidget(metrics_group)

        # Buttons
        btn_layout = QHBoxLayout()
        self.save_btn = QPushButton("Save Figure...")
        self.save_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-weight: bold;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        self.save_btn.clicked.connect(self._on_save)
        btn_layout.addWidget(self.save_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)

        btn_layout.addStretch()
        left_layout.addLayout(btn_layout)

        splitter.addWidget(left_panel)

        # Right panel: Preview with scroll area (like SamplesGrid)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)

        # Preview header with Fit button
        preview_header = QHBoxLayout()
        preview_label = QLabel("Preview:")
        preview_label.setStyleSheet("font-weight: bold;")
        preview_header.addWidget(preview_label)

        self.fit_btn = QPushButton("Fit to Window")
        self.fit_btn.setToolTip("Scale preview to fit the visible area")
        self.fit_btn.clicked.connect(self._fit_to_window)
        preview_header.addWidget(self.fit_btn)

        self._scale_label = QLabel("Size: --")
        self._scale_label.setStyleSheet("color: #666; font-size: 11px;")
        preview_header.addWidget(self._scale_label)

        preview_header.addStretch()
        right_layout.addLayout(preview_header)

        # Scroll area for the figure
        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(False)
        self._scroll_area.setAlignment(Qt.AlignCenter)
        self._scroll_area.setStyleSheet("QScrollArea { background-color: #f0f0f0; }")

        # Figure with fixed DPI
        self._figure = Figure(figsize=(8, 10), dpi=self._figure_dpi, facecolor='white')
        self._canvas = FigureCanvas(self._figure)
        self._canvas.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self._canvas.setStyleSheet("background-color: white;")

        self._scroll_area.setWidget(self._canvas)
        right_layout.addWidget(self._scroll_area, 1)

        splitter.addWidget(right_panel)
        splitter.setSizes([380, 720])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        layout.addWidget(splitter, 1)

        # Add 4 initial rows by default
        for _ in range(min(4, len(self._tests))):
            self._add_row()

        # Initial preview
        self._update_preview()

    def _add_row(self):
        """Add a new row configuration widget."""
        row_num = len(self._row_widgets) + 1

        row_widget = QualityRowWidget(self._tests, row_num)
        row_widget.changed.connect(self._update_preview)
        row_widget.remove_requested.connect(self._remove_row)

        # Set default test to next available (if possible)
        default_idx = min(len(self._row_widgets), len(self._tests) - 1)
        row_widget.set_test_index(default_idx)

        self._row_widgets.append(row_widget)
        # Insert before stretch
        self._rows_vlayout.insertWidget(self._rows_vlayout.count() - 1, row_widget)

        self._update_preview()

    def _remove_row(self, row_widget: QualityRowWidget):
        """Remove a row configuration widget."""
        if row_widget in self._row_widgets:
            self._row_widgets.remove(row_widget)
            self._rows_vlayout.removeWidget(row_widget)
            row_widget.deleteLater()

            # Renumber remaining rows
            for i, widget in enumerate(self._row_widgets):
                widget.set_row_number(i + 1)

            self._update_preview()

    def _update_preview(self):
        """Update the preview figure using fixed pixel dimensions."""
        self._figure.clf()
        self._figure.set_facecolor('white')

        if not self._row_widgets:
            self._natural_width_px = 600
            self._natural_height_px = 400
            self._figure.set_size_inches(6, 4)
            self._apply_canvas_size()
            ax = self._figure.add_subplot(111)
            ax.text(0.5, 0.5, "Add rows to preview",
                   ha='center', va='center', fontsize=14, color='#999')
            ax.axis('off')
            self._canvas.draw()
            return

        # Get configuration values
        cmap = self.cmap_combo.currentText()
        font_size = self.font_size_spin.value()
        image_idx = self.image_spin.value()
        image_size_px = self.image_size_spin.value()
        row_gap_px = self.row_spacing_spin.value()
        col_gap_px = self.col_spacing_spin.value()
        show_psnr = self.show_psnr_cb.isChecked()
        show_ssim = self.show_ssim_cb.isChecked()
        show_lpips = self.show_lpips_cb.isChecked()
        show_lines = self.show_lines_cb.isChecked()
        line_width = self.line_width_spin.value()
        image_padding = self.image_padding_spin.value()

        n_data_rows = len(self._row_widgets)

        # Calculate widths for each column type (cell sizes)
        # Col 0: Sampling Ratio label - narrower
        # Col 1: SPI Reconstructed - image
        # Col 2: Denoised - image
        # Col 3: Quality Metrics - wider for text
        label_col_width = int(image_size_px * 0.8)
        metrics_col_width = int(image_size_px * 1.4)

        # Margins
        left_margin_px = 15
        right_margin_px = 15
        top_margin_px = int(font_size * 8)
        bottom_margin_px = 15

        # GT image height (1.2 times image size)
        gt_image_height = int(image_size_px * 1.2)
        gt_title_height = int(font_size * 2.5)
        header_height = int(font_size * 5.5)  # Taller for column titles

        # Calculate total figure size
        content_width_px = (label_col_width + 2 * image_size_px + metrics_col_width +
                           3 * col_gap_px)
        data_content_height_px = n_data_rows * image_size_px + (n_data_rows - 1) * row_gap_px

        # Total height: GT title + GT image + larger gap + headers + data rows
        # Use larger gap between GT and table for visual separation
        gt_table_gap = max(row_gap_px * 2, 20)  # At least 20px or 2x row_gap
        gt_section_height = gt_title_height + gt_image_height + gt_table_gap

        fig_width_px = left_margin_px + content_width_px + right_margin_px
        fig_height_px = (top_margin_px + gt_section_height + header_height +
                        data_content_height_px + bottom_margin_px)

        # Convert to inches
        fig_width_in = fig_width_px / self._figure_dpi
        fig_height_in = fig_height_px / self._figure_dpi

        self._natural_width_px = fig_width_px
        self._natural_height_px = fig_height_px

        self._figure.set_size_inches(fig_width_in, fig_height_in)
        self._apply_canvas_size()

        # Column x positions (left edge of each cell)
        col_x = [
            left_margin_px,  # Label
            left_margin_px + label_col_width + col_gap_px,  # Reconstructed
            left_margin_px + label_col_width + col_gap_px + image_size_px + col_gap_px,  # Denoised
            left_margin_px + label_col_width + col_gap_px + 2 * image_size_px + 2 * col_gap_px,  # Metrics
        ]
        col_widths = [label_col_width, image_size_px, image_size_px, metrics_col_width]

        # Y positions (from top, converted to bottom-up for matplotlib)
        # GT title
        gt_title_y_top = top_margin_px
        gt_title_y_bottom = fig_height_px - gt_title_y_top - gt_title_height

        # GT image (centered horizontally)
        gt_img_y_top = gt_title_y_top + gt_title_height
        gt_img_y_bottom = fig_height_px - gt_img_y_top - gt_image_height
        gt_img_x = left_margin_px + (content_width_px - gt_image_height) / 2  # Centered

        # Headers y position (after GT section with larger gap)
        headers_y_top = gt_img_y_top + gt_image_height + gt_table_gap
        headers_y_bottom = fig_height_px - headers_y_top - header_height

        # Data rows start immediately after headers (no gap)
        data_start_y_top = headers_y_top + header_height

        # Draw GT title
        ax_gt_title = self._figure.add_axes([
            left_margin_px / fig_width_px,
            gt_title_y_bottom / fig_height_px,
            content_width_px / fig_width_px,
            gt_title_height / fig_height_px
        ])
        ax_gt_title.axis('off')
        ax_gt_title.text(0.5, 0.5, "Ground Truth", ha='center', va='center',
                        fontsize=font_size + 2, fontweight='bold')

        # Draw GT image
        first_config = self._row_widgets[0].get_config()
        first_test = self._tests[first_config.test_idx]
        originals, _, _ = self._load_test_images(first_test)
        gt_img = None
        if originals is not None and image_idx < len(originals):
            gt_img = originals[image_idx]

        ax_gt_img = self._figure.add_axes([
            gt_img_x / fig_width_px,
            gt_img_y_bottom / fig_height_px,
            gt_image_height / fig_width_px,
            gt_image_height / fig_height_px
        ])
        if gt_img is not None:
            gt_img = np.array(gt_img)
            if gt_img.ndim == 3 and gt_img.shape[-1] == 1:
                gt_img = gt_img.squeeze(-1)
            ax_gt_img.imshow(gt_img, cmap=cmap, vmin=0, vmax=1)
        ax_gt_img.axis('off')

        # Draw column headers (use editable titles, replace \n for newlines)
        col_headers = [
            self.col1_title_edit.text().replace("\\n", "\n"),
            self.col2_title_edit.text().replace("\\n", "\n"),
            self.col3_title_edit.text().replace("\\n", "\n"),
            self.col4_title_edit.text().replace("\\n", "\n"),
        ]
        for col_idx, header in enumerate(col_headers):
            ax_header = self._figure.add_axes([
                col_x[col_idx] / fig_width_px,
                headers_y_bottom / fig_height_px,
                col_widths[col_idx] / fig_width_px,
                header_height / fig_height_px
            ])
            ax_header.axis('off')
            ax_header.text(0.5, 0.5, header, ha='center', va='center',
                          fontsize=font_size, fontweight='bold')

        # Draw data rows
        for row_idx, row_widget in enumerate(self._row_widgets):
            config = row_widget.get_config()
            test = self._tests[config.test_idx]

            # Calculate y position for this row
            row_y_top = data_start_y_top + row_idx * (image_size_px + row_gap_px)
            row_y_bottom = fig_height_px - row_y_top - image_size_px

            # Determine label
            if config.label.strip():
                label_text = config.label
            else:
                test_name = test.get("name", f"Test {row_idx+1}")
                import re
                match = re.search(r'(\d+)%', test_name)
                if match:
                    label_text = f"{match.group(1)}%"
                else:
                    density = test.get("scatter_point_density")
                    if density:
                        label_text = f"{density:.0f}%"
                    else:
                        label_text = test_name[:12]

            # Load images
            originals, reconstructions, denoised_arr = self._load_test_images(test)

            recon_img = None
            denoised_img = None
            if reconstructions is not None and image_idx < len(reconstructions):
                recon_img = reconstructions[image_idx]
            if denoised_arr is not None and image_idx < len(denoised_arr):
                denoised_img = denoised_arr[image_idx]

            # Get metrics
            quality_per_image = test.get("quality_per_image", {})
            psnr_list = quality_per_image.get("psnr_denoised", [])
            ssim_list = quality_per_image.get("ssim_denoised", [])
            lpips_list = quality_per_image.get("lpips_denoised", [])

            psnr = psnr_list[image_idx] if image_idx < len(psnr_list) else None
            ssim = ssim_list[image_idx] if image_idx < len(ssim_list) else None
            lpips = lpips_list[image_idx] if image_idx < len(lpips_list) else None

            # Column 0: Label (no padding needed for text)
            ax_label = self._figure.add_axes([
                col_x[0] / fig_width_px,
                row_y_bottom / fig_height_px,
                col_widths[0] / fig_width_px,
                image_size_px / fig_height_px
            ])
            ax_label.axis('off')
            ax_label.text(0.5, 0.5, label_text, ha='center', va='center',
                         fontsize=font_size + 2, fontweight='bold')

            # Column 1: Reconstructed (with padding)
            padded_size = image_size_px - 2 * image_padding
            if padded_size > 0:
                ax_recon = self._figure.add_axes([
                    (col_x[1] + image_padding) / fig_width_px,
                    (row_y_bottom + image_padding) / fig_height_px,
                    padded_size / fig_width_px,
                    padded_size / fig_height_px
                ])
            else:
                ax_recon = self._figure.add_axes([
                    col_x[1] / fig_width_px,
                    row_y_bottom / fig_height_px,
                    col_widths[1] / fig_width_px,
                    image_size_px / fig_height_px
                ])
            if recon_img is not None:
                recon_img = np.array(recon_img)
                if recon_img.ndim == 3 and recon_img.shape[-1] == 1:
                    recon_img = recon_img.squeeze(-1)
                ax_recon.imshow(recon_img, cmap=cmap, vmin=0, vmax=1)
            ax_recon.axis('off')

            # Column 2: Denoised (with padding)
            if padded_size > 0:
                ax_denoised = self._figure.add_axes([
                    (col_x[2] + image_padding) / fig_width_px,
                    (row_y_bottom + image_padding) / fig_height_px,
                    padded_size / fig_width_px,
                    padded_size / fig_height_px
                ])
            else:
                ax_denoised = self._figure.add_axes([
                    col_x[2] / fig_width_px,
                    row_y_bottom / fig_height_px,
                    col_widths[2] / fig_width_px,
                    image_size_px / fig_height_px
                ])
            if denoised_img is not None:
                denoised_img = np.array(denoised_img)
                if denoised_img.ndim == 3 and denoised_img.shape[-1] == 1:
                    denoised_img = denoised_img.squeeze(-1)
                ax_denoised.imshow(denoised_img, cmap=cmap, vmin=0, vmax=1)
            ax_denoised.axis('off')

            # Column 3: Metrics (no padding needed for text)
            ax_metrics = self._figure.add_axes([
                col_x[3] / fig_width_px,
                row_y_bottom / fig_height_px,
                col_widths[3] / fig_width_px,
                image_size_px / fig_height_px
            ])
            ax_metrics.axis('off')
            metrics_lines = []
            if show_psnr and psnr is not None:
                metrics_lines.append(f"PSNR = {psnr:.2f} dB")
            if show_ssim and ssim is not None:
                metrics_lines.append(f"SSIM = {ssim:.3f}")
            if show_lpips and lpips is not None:
                metrics_lines.append(f"LPIPS = {lpips:.3f}")
            metrics_text = "\n".join(metrics_lines)
            ax_metrics.text(0.1, 0.5, metrics_text, ha='left', va='center',
                           fontsize=font_size, family='monospace')

        # Draw table lines if enabled (no outer border, no top line)
        if show_lines and n_data_rows > 0:
            # Calculate y positions for lines
            # Top of headers (for vertical lines extent)
            header_top_y = (headers_y_bottom + header_height) / fig_height_px
            # Bottom of headers (= top of first data row)
            header_bottom_y = (fig_height_px - data_start_y_top) / fig_height_px

            # Horizontal line below headers (separates headers from data)
            self._figure.add_artist(Line2D(
                [col_x[0] / fig_width_px, (col_x[3] + col_widths[3]) / fig_width_px],
                [header_bottom_y, header_bottom_y],
                transform=self._figure.transFigure,
                color='black', linewidth=line_width, clip_on=False
            ))

            # Lines between data rows (not below last row)
            for row_idx in range(n_data_rows - 1):
                row_y_top = data_start_y_top + row_idx * (image_size_px + row_gap_px)
                row_y_bottom = fig_height_px - row_y_top - image_size_px
                line_y = row_y_bottom / fig_height_px
                self._figure.add_artist(Line2D(
                    [col_x[0] / fig_width_px, (col_x[3] + col_widths[3]) / fig_width_px],
                    [line_y, line_y],
                    transform=self._figure.transFigure,
                    color='black', linewidth=line_width, clip_on=False
                ))

            # Vertical lines: between columns, extending through headers (not at outer edges)
            last_row_y_top = data_start_y_top + (n_data_rows - 1) * (image_size_px + row_gap_px)
            last_row_y_bottom = fig_height_px - last_row_y_top - image_size_px
            v_line_bottom = last_row_y_bottom / fig_height_px
            v_line_top = header_top_y  # Extend to top of headers

            # Vertical lines between columns (not at left/right edges)
            for col_idx in range(1, 4):
                x = col_x[col_idx] / fig_width_px
                self._figure.add_artist(Line2D(
                    [x, x], [v_line_bottom, v_line_top],
                    transform=self._figure.transFigure,
                    color='black', linewidth=line_width, clip_on=False
                ))

        self._canvas.draw()

    def _apply_canvas_size(self):
        """Apply the canvas size based on natural figure dimensions."""
        if not hasattr(self, '_natural_width_px'):
            return

        width = int(self._natural_width_px)
        height = int(self._natural_height_px)

        self._canvas.setFixedSize(width, height)
        self._canvas.updateGeometry()
        self._scroll_area.viewport().update()
        self._scale_label.setText(f"Size: {width}×{height}px")

    def _fit_to_window(self):
        """Adjust image size to fit the figure in the visible scroll area."""
        if not self._row_widgets:
            return

        n_rows = len(self._row_widgets)

        # Get available space
        viewport = self._scroll_area.viewport()
        available_width = viewport.width() - 40
        available_height = viewport.height() - 40

        # Current gap values
        row_gap = self.row_spacing_spin.value()
        col_gap = self.col_spacing_spin.value()
        font_size = self.font_size_spin.value()

        # Estimate margins and fixed sections
        left_margin = 15
        right_margin = 15
        top_margin = int(font_size * 8)
        bottom_margin = 15

        # GT section and header heights scale with image size, estimate
        # We need to solve for image_size such that total fits

        # Approximate: total_height = top + 1.2*img + gap + 4*font + gap + n_rows*img + (n-1)*row_gap + bottom
        # Simplify: total_height ≈ fixed_overhead + (1.2 + n_rows) * img_size + (n_rows-1)*row_gap

        fixed_height_overhead = top_margin + bottom_margin + row_gap + int(font_size * 4) + row_gap
        gt_factor = 1.2  # GT image is 1.2 times image size

        # total_height = fixed_height_overhead + gt_factor * img + n_rows * img + (n_rows-1)*row_gap
        # available_height = fixed_height_overhead + (gt_factor + n_rows) * img + (n_rows-1)*row_gap
        # img = (available_height - fixed_height_overhead - (n_rows-1)*row_gap) / (gt_factor + n_rows)

        content_height_for_images = available_height - fixed_height_overhead - (n_rows - 1) * row_gap
        max_img_h = content_height_for_images / (gt_factor + n_rows)

        # For width: label_col = 0.8*img, metrics_col = 1.4*img, 2 image cols
        # total_width = left + 0.8*img + 3*col_gap + 2*img + 1.4*img + right
        # total_width = left + right + 3*col_gap + (0.8 + 2 + 1.4)*img = margins + 3*col_gap + 4.2*img
        fixed_width_overhead = left_margin + right_margin + 3 * col_gap
        width_factor = 4.2
        max_img_w = (available_width - fixed_width_overhead) / width_factor

        new_img_size = int(min(max_img_h, max_img_w))
        new_img_size = max(32, min(256, new_img_size))

        self.image_size_spin.setValue(new_img_size)

    def _on_save(self):
        """Save the current figure."""
        self._save_figure(self._figure, "quality_sampling_ratio.png")


# ============================================================================
# Samples Grid Column Configuration
# ============================================================================

class GridColumnConfig:
    """Configuration for a column in the samples grid."""

    TYPE_GROUND_TRUTH = "ground_truth"
    TYPE_TEST = "test"

    def __init__(self, col_type: str = TYPE_GROUND_TRUTH, test_idx: int = -1, test_name: str = ""):
        self.col_type = col_type
        self.test_idx = test_idx  # Index in the tests list (-1 for ground truth)
        self.test_name = test_name  # Display name
        self.title = "Ground Truth" if col_type == self.TYPE_GROUND_TRUTH else test_name


class GridColumnCardWidget(QFrame):
    """Draggable card for grid column configuration."""

    double_clicked = pyqtSignal(object)
    drag_started = pyqtSignal(object)
    remove_requested = pyqtSignal(object)

    def __init__(self, config: GridColumnConfig, index: int, parent=None):
        super().__init__(parent)
        self.config = config
        self.index = index
        self._drag_start_pos = None
        self._mouse_pressed = False
        self._setup_ui()

    def _setup_ui(self):
        self.setFrameStyle(QFrame.Box | QFrame.Raised)
        self.setLineWidth(2)
        self.setMinimumSize(100, 70)
        self.setMaximumSize(130, 85)
        self.setCursor(Qt.OpenHandCursor)
        self._update_style()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(2)

        self.title_label = QLabel(self._get_display_text())
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setWordWrap(True)
        self.title_label.setStyleSheet("font-weight: bold; font-size: 9px;")
        layout.addWidget(self.title_label)

        self.type_label = QLabel(self._get_type_text())
        self.type_label.setAlignment(Qt.AlignCenter)
        self.type_label.setStyleSheet("font-size: 8px; color: #666;")
        layout.addWidget(self.type_label)

    def _get_display_text(self) -> str:
        if self.config.col_type == GridColumnConfig.TYPE_GROUND_TRUTH:
            return "Ground Truth"
        return self.config.title[:15] if len(self.config.title) > 15 else self.config.title

    def _get_type_text(self) -> str:
        if self.config.col_type == GridColumnConfig.TYPE_GROUND_TRUTH:
            return "[Reference]"
        return "[Test]"

    def _update_style(self, dragging: bool = False):
        if self.config.col_type == GridColumnConfig.TYPE_GROUND_TRUTH:
            color = "#e8f5e9"  # Green tint
        else:
            color = "#fff3e0"  # Orange tint

        self.setStyleSheet(f"""
            GridColumnCardWidget {{
                background-color: {color};
                border: 2px solid #999;
                border-radius: 6px;
            }}
            GridColumnCardWidget:hover {{
                border: 2px solid #333;
            }}
        """)

        from PyQt5.QtWidgets import QGraphicsOpacityEffect
        if dragging:
            effect = QGraphicsOpacityEffect(self)
            effect.setOpacity(0.4)
            self.setGraphicsEffect(effect)
        else:
            self.setGraphicsEffect(None)

    def update_display(self):
        self.title_label.setText(self._get_display_text())
        self.type_label.setText(self._get_type_text())
        self._update_style()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_start_pos = event.pos()
            self._mouse_pressed = True
            self.setCursor(Qt.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if (self._drag_start_pos is not None and
            self._mouse_pressed and
            event.buttons() & Qt.LeftButton):
            if (event.pos() - self._drag_start_pos).manhattanLength() > 10:
                self._start_drag()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._drag_start_pos = None
        self._mouse_pressed = False
        self.setCursor(Qt.OpenHandCursor)
        super().mouseReleaseEvent(event)

    def _start_drag(self):
        if not self._mouse_pressed:
            return
        self._drag_start_pos = None
        self._mouse_pressed = False

        pixmap = QPixmap(self.size())
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setOpacity(0.7)
        self.render(painter)
        painter.end()

        drag = QDrag(self)
        mime_data = QMimeData()
        mime_data.setText(f"grid_col:{self.index}")
        drag.setMimeData(mime_data)
        drag.setPixmap(pixmap)
        drag.setHotSpot(QPoint(pixmap.width() // 2, pixmap.height() // 2))

        self._update_style(dragging=True)
        self.drag_started.emit(self)
        drag.exec_(Qt.MoveAction)
        self._update_style(dragging=False)
        self.setCursor(Qt.OpenHandCursor)

    def mouseDoubleClickEvent(self, event):
        self._drag_start_pos = None
        self._mouse_pressed = False
        self.setCursor(Qt.OpenHandCursor)
        self.double_clicked.emit(self)
        super().mouseDoubleClickEvent(event)


class GridColumnConfigDialog(QDialog):
    """Dialog to configure a grid column."""

    def __init__(self, config: GridColumnConfig, tests: List[Dict], parent=None):
        super().__init__(parent)
        self.config = config
        self.tests = tests
        self.setWindowTitle("Configure Column")
        self.setMinimumWidth(350)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        # Column type
        type_group = QGroupBox("Column Type")
        type_layout = QVBoxLayout(type_group)

        self.type_combo = QComboBox()
        self.type_combo.addItem("Ground Truth (Reference)", GridColumnConfig.TYPE_GROUND_TRUTH)
        for i, test in enumerate(self.tests):
            name = test.get("name", f"Test {i+1}")
            exp_name = test.get("_experiment_name", "")
            display = f"{name} ({exp_name})" if exp_name else name
            self.type_combo.addItem(display, i)

        # Set current
        if self.config.col_type == GridColumnConfig.TYPE_GROUND_TRUTH:
            self.type_combo.setCurrentIndex(0)
        else:
            # Find the test index
            for i in range(1, self.type_combo.count()):
                if self.type_combo.itemData(i) == self.config.test_idx:
                    self.type_combo.setCurrentIndex(i)
                    break

        type_layout.addWidget(self.type_combo)
        layout.addWidget(type_group)

        # Title
        title_group = QGroupBox("Column Title (optional)")
        title_layout = QVBoxLayout(title_group)

        self.title_edit = QLineEdit(self.config.title)
        self.title_edit.setPlaceholderText("Auto-generated if empty")
        title_layout.addWidget(self.title_edit)

        self.type_combo.currentIndexChanged.connect(self._on_type_changed)
        layout.addWidget(title_group)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        ok_btn = QPushButton("OK")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self._on_accept)
        btn_layout.addWidget(ok_btn)

        layout.addLayout(btn_layout)

    def _on_type_changed(self):
        idx = self.type_combo.currentIndex()
        if idx == 0:
            # Ground Truth
            if not self.title_edit.text() or self.title_edit.text() == self.config.title:
                self.title_edit.setText("Ground Truth")
        else:
            # Test
            test_idx = self.type_combo.itemData(idx)
            test = self.tests[test_idx]
            name = test.get("name", f"Test {test_idx+1}")
            if not self.title_edit.text() or self.title_edit.text() == self.config.title:
                self.title_edit.setText(name)

    def _on_accept(self):
        idx = self.type_combo.currentIndex()
        if idx == 0:
            self.config.col_type = GridColumnConfig.TYPE_GROUND_TRUTH
            self.config.test_idx = -1
            self.config.test_name = ""
        else:
            self.config.col_type = GridColumnConfig.TYPE_TEST
            self.config.test_idx = self.type_combo.itemData(idx)
            test = self.tests[self.config.test_idx]
            self.config.test_name = test.get("name", f"Test {self.config.test_idx+1}")

        self.config.title = self.title_edit.text() or (
            "Ground Truth" if self.config.col_type == GridColumnConfig.TYPE_GROUND_TRUTH
            else self.config.test_name
        )
        self.accept()


class GridColumnListWidget(QWidget):
    """Widget containing draggable column cards for grid."""

    columns_changed = pyqtSignal()

    def __init__(self, tests: List[Dict], parent=None):
        super().__init__(parent)
        self.tests = tests
        self.columns: List[GridColumnConfig] = []
        self.cards: List[GridColumnCardWidget] = []
        self._dragged_card = None
        self._drop_index = -1
        self._setup_ui()

    def _setup_ui(self):
        self.setAcceptDrops(True)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Cards container
        self.cards_container = QWidget()
        self.cards_layout = QHBoxLayout(self.cards_container)
        self.cards_layout.setContentsMargins(5, 5, 5, 5)
        self.cards_layout.setSpacing(8)
        self.cards_layout.addStretch()

        # Drop indicator
        self.drop_indicator = DropIndicatorWidget(self.cards_container)

        main_layout.addWidget(self.cards_container)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(5, 0, 5, 5)

        add_gt_btn = QPushButton("+ Ground Truth")
        add_gt_btn.clicked.connect(self._add_ground_truth)
        btn_layout.addWidget(add_gt_btn)

        add_test_btn = QPushButton("+ Test")
        add_test_btn.clicked.connect(self._add_test)
        btn_layout.addWidget(add_test_btn)

        remove_btn = QPushButton("- Remove Last")
        remove_btn.clicked.connect(self._remove_last)
        btn_layout.addWidget(remove_btn)

        btn_layout.addStretch()
        main_layout.addLayout(btn_layout)

        # Add default columns
        self._add_default_columns()

    def _add_default_columns(self):
        # Ground truth first
        self._add_column_with_config(GridColumnConfig(GridColumnConfig.TYPE_GROUND_TRUTH))

        # Add all tests in order
        for i in range(len(self.tests)):
            test = self.tests[i]
            name = test.get("name", f"Test {i+1}")
            config = GridColumnConfig(GridColumnConfig.TYPE_TEST, i, name)
            self._add_column_with_config(config)

    def _add_ground_truth(self):
        if len(self.columns) >= 8:
            return
        config = GridColumnConfig(GridColumnConfig.TYPE_GROUND_TRUTH)
        self._add_column_with_config(config)
        self.columns_changed.emit()

    def _add_test(self):
        if len(self.columns) >= 8 or not self.tests:
            return
        # Add first unused test, or first test if all used
        used_indices = {c.test_idx for c in self.columns if c.col_type == GridColumnConfig.TYPE_TEST}
        test_idx = 0
        for i in range(len(self.tests)):
            if i not in used_indices:
                test_idx = i
                break
        test = self.tests[test_idx]
        name = test.get("name", f"Test {test_idx+1}")
        config = GridColumnConfig(GridColumnConfig.TYPE_TEST, test_idx, name)
        self._add_column_with_config(config)
        self.columns_changed.emit()

    def _add_column_with_config(self, config: GridColumnConfig):
        self.columns.append(config)
        card = GridColumnCardWidget(config, len(self.cards))
        card.double_clicked.connect(self._on_card_double_clicked)
        card.drag_started.connect(self._on_drag_started)
        self.cards.append(card)
        self.cards_layout.insertWidget(self.cards_layout.count() - 1, card)

    def _remove_last(self):
        if len(self.columns) <= 1:
            return
        self.columns.pop()
        card = self.cards.pop()
        self.cards_layout.removeWidget(card)
        card.deleteLater()
        self.columns_changed.emit()

    def _on_card_double_clicked(self, card: GridColumnCardWidget):
        dialog = GridColumnConfigDialog(card.config, self.tests, self)
        if dialog.exec_() == QDialog.Accepted:
            card.update_display()
            self.columns_changed.emit()

    def _on_drag_started(self, card):
        self._dragged_card = card

    def _get_drop_index(self, pos: QPoint) -> int:
        container_pos = self.cards_container.mapFrom(self, pos)
        x = container_pos.x()
        for i, card in enumerate(self.cards):
            card_rect = card.geometry()
            card_center = card_rect.center().x()
            if x < card_center:
                return i
        return len(self.cards)

    def _show_drop_indicator(self, drop_index: int):
        if drop_index < 0 or not self.cards:
            self.drop_indicator.hide()
            return
        if drop_index < len(self.cards):
            target_card = self.cards[drop_index]
            x = target_card.geometry().left() - 6
        else:
            target_card = self.cards[-1]
            x = target_card.geometry().right() + 2
        y = self.cards[0].geometry().top()
        height = self.cards[0].geometry().height()
        self.drop_indicator.setFixedHeight(height)
        self.drop_indicator.move(x, y)
        self.drop_indicator.show()
        self.drop_indicator.raise_()

    def dragEnterEvent(self, event):
        if event.mimeData().hasText() and event.mimeData().text().startswith("grid_col:"):
            event.acceptProposedAction()
            self._drop_index = self._get_drop_index(event.pos())
            self._show_drop_indicator(self._drop_index)

    def dragMoveEvent(self, event):
        if event.mimeData().hasText() and event.mimeData().text().startswith("grid_col:"):
            event.acceptProposedAction()
            new_drop_index = self._get_drop_index(event.pos())
            if new_drop_index != self._drop_index:
                self._drop_index = new_drop_index
                self._show_drop_indicator(self._drop_index)

    def dragLeaveEvent(self, event):
        self.drop_indicator.hide()
        self._drop_index = -1

    def dropEvent(self, event):
        self.drop_indicator.hide()
        if not event.mimeData().hasText() or self._dragged_card is None:
            return
        event.acceptProposedAction()

        old_idx = self.cards.index(self._dragged_card)
        new_idx = self._drop_index
        if new_idx > old_idx:
            new_idx -= 1

        if old_idx != new_idx and 0 <= new_idx < len(self.cards):
            self.cards.remove(self._dragged_card)
            self.columns.remove(self._dragged_card.config)
            self.cards.insert(new_idx, self._dragged_card)
            self.columns.insert(new_idx, self._dragged_card.config)
            self.cards_layout.removeWidget(self._dragged_card)
            self.cards_layout.insertWidget(new_idx, self._dragged_card)
            for i, card in enumerate(self.cards):
                card.index = i
            self.columns_changed.emit()

        self._dragged_card = None
        self._drop_index = -1

    def get_columns(self) -> List[GridColumnConfig]:
        return self.columns.copy()


# ============================================================================
# Samples Grid Popup (Fig 2 style)
# ============================================================================

class SamplesGridPopup(BaseFigureExportPopup):
    """
    Popup for generating Samples Grid figure.
    Shows multiple sample images (rows) at different sampling ratios (columns).
    """

    def __init__(self, tests: List[Dict[str, Any]], logger=None, parent=None):
        super().__init__(tests, logger, parent)
        self.setWindowTitle("Samples Grid Figure")
        self.setMinimumSize(1100, 750)
        self.resize(1300, 850)

        self._figure = None
        self._canvas = None
        self._images_cache = {}
        self._row_spinboxes = []
        self._row_label_edits = []
        self._figure_dpi = 100
        self._natural_width_px = 600
        self._natural_height_px = 400
        self._setup_ui()

    def _setup_ui(self):
        """Setup the popup UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Title
        title = QLabel("Samples Grid (Multiple Images × Tests)")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(title)

        # Main splitter: controls on left, preview on right
        main_splitter = QSplitter(Qt.Horizontal)

        # Left panel: all controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 8, 0)
        left_layout.setSpacing(8)

        # Column configuration (compact)
        col_group = QGroupBox("Columns (drag to reorder, double-click to edit)")
        col_layout = QVBoxLayout(col_group)
        col_layout.setContentsMargins(8, 12, 8, 8)

        self.column_list = GridColumnListWidget(self._tests)
        self.column_list.columns_changed.connect(self._update_preview)
        col_layout.addWidget(self.column_list)

        left_layout.addWidget(col_group)

        # Row configuration
        rows_group = QGroupBox("Rows")
        rows_layout = QVBoxLayout(rows_group)
        rows_layout.setContentsMargins(8, 12, 8, 8)
        rows_layout.setSpacing(4)

        # Number of rows
        n_rows_layout = QHBoxLayout()
        n_rows_layout.addWidget(QLabel("Number of rows:"))
        self.n_rows_spin = QSpinBox()
        self.n_rows_spin.setMinimum(1)
        self.n_rows_spin.setMaximum(10)
        self.n_rows_spin.setValue(4)
        self.n_rows_spin.setFixedWidth(60)
        self.n_rows_spin.valueChanged.connect(self._update_row_spinboxes)
        n_rows_layout.addWidget(self.n_rows_spin)
        n_rows_layout.addStretch()
        rows_layout.addLayout(n_rows_layout)

        # Scroll area for row configuration
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(150)
        scroll.setFrameShape(QFrame.NoFrame)

        self.rows_container = QWidget()
        self.rows_container_layout = QGridLayout(self.rows_container)
        self.rows_container_layout.setContentsMargins(0, 4, 0, 0)
        self.rows_container_layout.setSpacing(4)
        self.rows_container_layout.setColumnStretch(1, 1)
        self.rows_container_layout.setColumnStretch(3, 1)
        scroll.setWidget(self.rows_container)

        rows_layout.addWidget(scroll)
        left_layout.addWidget(rows_group)

        # Display options (compact)
        options_group = QGroupBox("Display Options")
        options_layout = QGridLayout(options_group)
        options_layout.setContentsMargins(8, 12, 8, 8)
        options_layout.setSpacing(6)

        # Image size (in pixels)
        options_layout.addWidget(QLabel("Image size (px):"), 0, 0)
        self.image_size_spin = QSpinBox()
        self.image_size_spin.setMinimum(32)
        self.image_size_spin.setMaximum(256)
        self.image_size_spin.setValue(80)
        self.image_size_spin.setSingleStep(8)
        self.image_size_spin.valueChanged.connect(self._update_preview)
        options_layout.addWidget(self.image_size_spin, 0, 1)

        # Font size
        options_layout.addWidget(QLabel("Font size:"), 0, 2)
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setMinimum(6)
        self.font_size_spin.setMaximum(20)
        self.font_size_spin.setValue(11)
        self.font_size_spin.valueChanged.connect(self._update_preview)
        options_layout.addWidget(self.font_size_spin, 0, 3)

        # Row gap (in pixels)
        options_layout.addWidget(QLabel("Row gap (px):"), 1, 0)
        self.row_spacing_spin = QSpinBox()
        self.row_spacing_spin.setMinimum(0)
        self.row_spacing_spin.setMaximum(100)
        self.row_spacing_spin.setValue(10)
        self.row_spacing_spin.valueChanged.connect(self._update_preview)
        options_layout.addWidget(self.row_spacing_spin, 1, 1)

        # Column gap (in pixels)
        options_layout.addWidget(QLabel("Col gap (px):"), 1, 2)
        self.col_spacing_spin = QSpinBox()
        self.col_spacing_spin.setMinimum(0)
        self.col_spacing_spin.setMaximum(100)
        self.col_spacing_spin.setValue(10)
        self.col_spacing_spin.valueChanged.connect(self._update_preview)
        options_layout.addWidget(self.col_spacing_spin, 1, 3)

        # Colormap
        options_layout.addWidget(QLabel("Colormap:"), 2, 0)
        self.cmap_combo = self._create_colormap_combo()
        self.cmap_combo.currentTextChanged.connect(self._update_preview)
        options_layout.addWidget(self.cmap_combo, 2, 1)

        # Checkboxes row
        self.show_labels_cb = QCheckBox("Column titles")
        self.show_labels_cb.setChecked(True)
        self.show_labels_cb.stateChanged.connect(self._update_preview)
        options_layout.addWidget(self.show_labels_cb, 3, 0, 1, 2)

        self.show_row_labels_cb = QCheckBox("Row labels")
        self.show_row_labels_cb.setChecked(True)
        self.show_row_labels_cb.stateChanged.connect(self._update_preview)
        options_layout.addWidget(self.show_row_labels_cb, 3, 2, 1, 2)

        # Colored borders checkbox
        self.show_grid_cb = QCheckBox("Colored borders around groups")
        self.show_grid_cb.setChecked(False)
        self.show_grid_cb.stateChanged.connect(self._update_preview)
        options_layout.addWidget(self.show_grid_cb, 4, 0, 1, 4)

        # Border colors and width (in a sub-layout)
        border_layout = QHBoxLayout()
        border_layout.setContentsMargins(15, 0, 0, 0)
        border_layout.setSpacing(6)

        border_layout.addWidget(QLabel("GT:"))
        self._gt_color = QColor("#4CAF50")
        self.gt_color_btn = QPushButton()
        self.gt_color_btn.setFixedSize(28, 22)
        self.gt_color_btn.setStyleSheet(f"background-color: {self._gt_color.name()}; border: 1px solid #666;")
        self.gt_color_btn.clicked.connect(self._pick_gt_color)
        border_layout.addWidget(self.gt_color_btn)

        border_layout.addWidget(QLabel("Tests:"))
        self._test_color = QColor("#FF9800")
        self.test_color_btn = QPushButton()
        self.test_color_btn.setFixedSize(28, 22)
        self.test_color_btn.setStyleSheet(f"background-color: {self._test_color.name()}; border: 1px solid #666;")
        self.test_color_btn.clicked.connect(self._pick_test_color)
        border_layout.addWidget(self.test_color_btn)

        border_layout.addWidget(QLabel("Width:"))
        self.grid_width_spin = QSpinBox()
        self.grid_width_spin.setMinimum(1)
        self.grid_width_spin.setMaximum(8)
        self.grid_width_spin.setValue(3)
        self.grid_width_spin.setFixedWidth(45)
        self.grid_width_spin.valueChanged.connect(self._update_preview)
        border_layout.addWidget(self.grid_width_spin)

        border_layout.addWidget(QLabel("Padding:"))
        self.border_padding_spin = QSpinBox()
        self.border_padding_spin.setMinimum(0)
        self.border_padding_spin.setMaximum(20)
        self.border_padding_spin.setValue(3)
        self.border_padding_spin.setFixedWidth(45)
        self.border_padding_spin.valueChanged.connect(self._update_preview)
        border_layout.addWidget(self.border_padding_spin)

        border_layout.addStretch()
        options_layout.addLayout(border_layout, 5, 0, 1, 4)

        # Sampling ratio header
        self.show_ratio_header_cb = QCheckBox("Show 'Sampling ratio' header")
        self.show_ratio_header_cb.setChecked(True)
        self.show_ratio_header_cb.stateChanged.connect(self._update_preview)
        options_layout.addWidget(self.show_ratio_header_cb, 6, 0, 1, 4)

        left_layout.addWidget(options_group)

        # Save button
        btn_layout = QHBoxLayout()
        self.save_btn = QPushButton("Save Figure...")
        self.save_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                font-weight: bold;
                padding: 8px 20px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
        """)
        self.save_btn.clicked.connect(self._on_save)
        btn_layout.addWidget(self.save_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)

        btn_layout.addStretch()
        left_layout.addLayout(btn_layout)

        left_layout.addStretch()
        main_splitter.addWidget(left_panel)

        # Right panel: Preview (larger)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)

        # Preview header with Fit button
        preview_header = QHBoxLayout()
        preview_label = QLabel("Preview:")
        preview_label.setStyleSheet("font-weight: bold;")
        preview_header.addWidget(preview_label)

        self.fit_btn = QPushButton("Fit to Window")
        self.fit_btn.setToolTip("Scale preview to fit the visible area")
        self.fit_btn.clicked.connect(self._fit_to_window)
        preview_header.addWidget(self.fit_btn)

        self._scale_label = QLabel("Scale: 100%")
        self._scale_label.setStyleSheet("color: #666; font-size: 11px;")
        preview_header.addWidget(self._scale_label)

        preview_header.addStretch()
        right_layout.addLayout(preview_header)

        # Scroll area for the figure
        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(False)  # Fixed size content
        self._scroll_area.setAlignment(Qt.AlignCenter)
        self._scroll_area.setStyleSheet("QScrollArea { background-color: #f0f0f0; }")

        # Figure with fixed DPI (will be resized dynamically)
        self._figure_dpi = 100
        self._figure = Figure(figsize=(10, 8), dpi=self._figure_dpi, facecolor='white')
        self._canvas = FigureCanvas(self._figure)
        self._canvas.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self._canvas.setStyleSheet("background-color: white;")

        # Set canvas directly as scroll area widget (no container needed)
        self._scroll_area.setWidget(self._canvas)
        right_layout.addWidget(self._scroll_area, 1)

        main_splitter.addWidget(right_panel)

        # Set splitter sizes (controls: 320px, preview: rest)
        main_splitter.setSizes([320, 900])
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)

        layout.addWidget(main_splitter, 1)

        # Initialize row spinboxes
        self._update_row_spinboxes()

    def _update_row_spinboxes(self):
        """Update the row sample index spinboxes and label editors."""
        n_rows = self.n_rows_spin.value()

        # Clear existing widgets
        for spin in self._row_spinboxes:
            spin.deleteLater()
        for edit in self._row_label_edits:
            edit.deleteLater()
        self._row_spinboxes.clear()
        self._row_label_edits.clear()

        # Clear layout
        while self.rows_container_layout.count():
            item = self.rows_container_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Header row
        idx_header = QLabel("Sample Index")
        idx_header.setStyleSheet("font-weight: bold; font-size: 10px;")
        self.rows_container_layout.addWidget(idx_header, 0, 1)

        label_header = QLabel("Row Label")
        label_header.setStyleSheet("font-weight: bold; font-size: 10px;")
        self.rows_container_layout.addWidget(label_header, 0, 2)

        # Get max number of images for spinbox limit
        max_images = self._get_max_num_images()
        max_idx = max(0, max_images - 1)

        # Create row widgets
        for i in range(n_rows):
            row_label = QLabel(f"Row {i+1}:")
            row_label.setStyleSheet("font-size: 11px;")
            self.rows_container_layout.addWidget(row_label, i + 1, 0)

            spin = QSpinBox()
            spin.setMinimum(0)
            spin.setMaximum(max_idx)
            spin.setValue(min(i, max_idx))  # Default to sequential, but respect max
            spin.setFixedWidth(70)
            spin.valueChanged.connect(self._update_preview)
            self._row_spinboxes.append(spin)
            self.rows_container_layout.addWidget(spin, i + 1, 1)

            label_edit = QLineEdit(f"Sample #{i+1}")
            label_edit.setFixedWidth(100)
            label_edit.textChanged.connect(self._update_preview)
            self._row_label_edits.append(label_edit)
            self.rows_container_layout.addWidget(label_edit, i + 1, 2)

        self._update_preview()

    def _get_row_indices(self) -> List[int]:
        """Get the sample indices for each row."""
        return [spin.value() for spin in self._row_spinboxes]

    def _get_row_labels(self) -> List[str]:
        """Get the labels for each row."""
        return [edit.text() for edit in self._row_label_edits]

    def _pick_gt_color(self):
        """Open color picker for GT border."""
        color = QColorDialog.getColor(self._gt_color, self, "Select Ground Truth Border Color")
        if color.isValid():
            self._gt_color = color
            self.gt_color_btn.setStyleSheet(f"background-color: {color.name()}; border: 1px solid #666;")
            self._update_preview()

    def _pick_test_color(self):
        """Open color picker for test border."""
        color = QColorDialog.getColor(self._test_color, self, "Select Tests Border Color")
        if color.isValid():
            self._test_color = color
            self.test_color_btn.setStyleSheet(f"background-color: {color.name()}; border: 1px solid #666;")
            self._update_preview()

    def _load_column_images(self, col_config: GridColumnConfig) -> tuple:
        """Load images for a column configuration."""
        if col_config.col_type == GridColumnConfig.TYPE_GROUND_TRUTH:
            # For ground truth, use first available test's originals
            for test in self._tests:
                originals, _, _ = self._load_test_images(test)
                if originals is not None:
                    return originals, None, None
            return None, None, None
        else:
            if 0 <= col_config.test_idx < len(self._tests):
                test = self._tests[col_config.test_idx]
                return self._load_test_images(test)
            return None, None, None

    def _update_preview(self):
        """Update the preview figure using fixed pixel dimensions."""
        # Clear figure completely and set white background to cover old content
        self._figure.clf()
        self._figure.set_facecolor('white')

        columns = self.column_list.get_columns()
        row_indices = self._get_row_indices()
        row_labels = self._get_row_labels()

        if not columns:
            self._natural_width_px = 600
            self._natural_height_px = 400
            self._figure.set_size_inches(6, 4)
            self._apply_canvas_size()
            ax = self._figure.add_subplot(111)
            ax.text(0.5, 0.5, "Add columns to preview",
                   ha='center', va='center', fontsize=14, color='#999')
            ax.axis('off')
            self._canvas.draw()
            return

        n_rows = len(row_indices)
        n_cols = len(columns)
        cmap = self.cmap_combo.currentText()
        show_labels = self.show_labels_cb.isChecked()
        show_row_labels = self.show_row_labels_cb.isChecked()
        show_borders = self.show_grid_cb.isChecked()
        border_width = self.grid_width_spin.value()
        border_padding_px = self.border_padding_spin.value()
        row_gap_px = self.row_spacing_spin.value()
        col_gap_px = self.col_spacing_spin.value()
        show_ratio_header = self.show_ratio_header_cb.isChecked()
        font_size = self.font_size_spin.value()
        image_size_px = self.image_size_spin.value()
        gt_color = self._gt_color.name()
        test_color = self._test_color.name()

        # Calculate margins in pixels (scale with font size for row labels)
        # "Sample #1" is about 10 characters, each char ~0.7 * font_size wide
        left_margin_px = int(font_size * 12) if show_row_labels else 10
        right_margin_px = 10
        top_margin_px = int(font_size * 5) if show_labels else 20
        if show_ratio_header and show_labels:
            top_margin_px += int(font_size * 2.5)  # Extra space for "Sampling ratio" header
        bottom_margin_px = 10

        # Calculate total figure size in pixels
        content_width_px = n_cols * image_size_px + (n_cols - 1) * col_gap_px
        content_height_px = n_rows * image_size_px + (n_rows - 1) * row_gap_px
        fig_width_px = left_margin_px + content_width_px + right_margin_px
        fig_height_px = top_margin_px + content_height_px + bottom_margin_px

        # Convert to inches at our fixed DPI
        fig_width_in = fig_width_px / self._figure_dpi
        fig_height_in = fig_height_px / self._figure_dpi

        # Store natural size
        self._natural_width_px = fig_width_px
        self._natural_height_px = fig_height_px

        # Set figure size and apply canvas size
        self._figure.set_size_inches(fig_width_in, fig_height_in)
        self._apply_canvas_size()

        # Preload images for all columns
        column_images = {}
        for col_idx, col_config in enumerate(columns):
            originals, reconstructions, denoised = self._load_column_images(col_config)
            column_images[col_idx] = (originals, reconstructions, denoised)

        # Create axes manually with fixed pixel positions
        axes = []
        for row_idx in range(n_rows):
            row_axes = []
            for col_idx in range(n_cols):
                # Calculate position in pixels
                x_px = left_margin_px + col_idx * (image_size_px + col_gap_px)
                y_px = bottom_margin_px + (n_rows - 1 - row_idx) * (image_size_px + row_gap_px)

                # Convert to figure fractions
                x_frac = x_px / fig_width_px
                y_frac = y_px / fig_height_px
                w_frac = image_size_px / fig_width_px
                h_frac = image_size_px / fig_height_px

                ax = self._figure.add_axes([x_frac, y_frac, w_frac, h_frac])
                row_axes.append(ax)
            axes.append(row_axes)

        # Count ground truth columns and test columns
        gt_cols = [i for i, c in enumerate(columns) if c.col_type == GridColumnConfig.TYPE_GROUND_TRUTH]
        test_cols = [i for i, c in enumerate(columns) if c.col_type == GridColumnConfig.TYPE_TEST]

        # Fill the grid
        for row_idx, sample_idx in enumerate(row_indices):
            for col_idx, col_config in enumerate(columns):
                ax = axes[row_idx][col_idx]

                originals, reconstructions, denoised = column_images[col_idx]

                # Get the appropriate image
                img = None
                if col_config.col_type == GridColumnConfig.TYPE_GROUND_TRUTH:
                    if originals is not None and sample_idx < len(originals):
                        img = originals[sample_idx]
                else:
                    # Show reconstructed images for tests
                    if reconstructions is not None and sample_idx < len(reconstructions):
                        img = reconstructions[sample_idx]

                if img is not None:
                    img = np.array(img)
                    if img.ndim == 3 and img.shape[-1] == 1:
                        img = img.squeeze(-1)
                    ax.imshow(img, cmap=cmap, vmin=0, vmax=1, aspect='equal')
                else:
                    ax.set_facecolor('#f0f0f0')
                    ax.text(0.5, 0.5, "N/A", ha='center', va='center',
                           fontsize=font_size - 2, color='#999')

                ax.set_xticks([])
                ax.set_yticks([])

                # Hide individual spines (we'll draw group rectangles instead)
                for spine in ax.spines.values():
                    spine.set_visible(False)

        # Add column titles at the top
        if show_labels:
            for col_idx, col_config in enumerate(columns):
                ax = axes[0][col_idx]
                ax.set_title(col_config.title, fontsize=font_size, fontweight='bold', pad=15)

        # Add row labels on the left (outside the axes)
        if show_row_labels:
            for row_idx in range(n_rows):
                ax = axes[row_idx][0]
                label = row_labels[row_idx] if row_idx < len(row_labels) else f"Sample #{row_idx+1}"
                ax.annotate(label, xy=(-0.15, 0.5), xycoords='axes fraction',
                           fontsize=font_size, fontweight='bold', ha='right', va='center')

        # Convert border padding to figure fractions
        border_padding_x = border_padding_px / fig_width_px
        border_padding_y = border_padding_px / fig_height_px

        # Helper to calculate axis position in figure fractions (without needing draw)
        def get_ax_bounds(row_idx, col_idx):
            x_px = left_margin_px + col_idx * (image_size_px + col_gap_px)
            y_px = bottom_margin_px + (n_rows - 1 - row_idx) * (image_size_px + row_gap_px)
            x0 = x_px / fig_width_px
            y0 = y_px / fig_height_px
            x1 = (x_px + image_size_px) / fig_width_px
            y1 = (y_px + image_size_px) / fig_height_px
            return x0, y0, x1, y1

        # Draw group borders (rectangles around column groups)
        if show_borders and n_rows > 0:
            # Draw border around Ground Truth columns
            if gt_cols:
                first_gt = gt_cols[0]
                last_gt = gt_cols[-1]

                # Get bounds for first and last GT column
                tl_x0, _, _, tl_y1 = get_ax_bounds(0, first_gt)
                _, br_y0, br_x1, _ = get_ax_bounds(n_rows - 1, last_gt)

                rect_x = tl_x0 - border_padding_x
                rect_y = br_y0 - border_padding_y
                rect_width = (br_x1 - tl_x0) + 2 * border_padding_x
                rect_height = (tl_y1 - br_y0) + 2 * border_padding_y

                rect = Rectangle((rect_x, rect_y), rect_width, rect_height,
                                fill=False, edgecolor=gt_color,
                                linewidth=border_width, transform=self._figure.transFigure,
                                clip_on=False)
                self._figure.add_artist(rect)

            # Draw border around Test columns
            if test_cols:
                first_test = test_cols[0]
                last_test = test_cols[-1]

                # Get bounds for first and last test column
                tl_x0, _, _, tl_y1 = get_ax_bounds(0, first_test)
                _, br_y0, br_x1, _ = get_ax_bounds(n_rows - 1, last_test)

                rect_x = tl_x0 - border_padding_x
                rect_y = br_y0 - border_padding_y
                rect_width = (br_x1 - tl_x0) + 2 * border_padding_x
                rect_height = (tl_y1 - br_y0) + 2 * border_padding_y

                rect = Rectangle((rect_x, rect_y), rect_width, rect_height,
                                fill=False, edgecolor=test_color,
                                linewidth=border_width, transform=self._figure.transFigure,
                                clip_on=False)
                self._figure.add_artist(rect)

        # Add "Sampling ratio" header line above test columns
        if show_ratio_header and test_cols and show_labels:
            first_test_col = test_cols[0]
            last_test_col = test_cols[-1]

            # Get bounds for first and last test column
            first_x0, _, _, first_y1 = get_ax_bounds(0, first_test_col)
            _, _, last_x1, _ = get_ax_bounds(0, last_test_col)

            # Position line above the titles
            line_y = first_y1 + 40 / fig_height_px
            line_x_start = first_x0
            line_x_end = last_x1

            self._figure.add_artist(Line2D(
                [line_x_start, line_x_end], [line_y, line_y],
                transform=self._figure.transFigure,
                color='black', linewidth=1.5, clip_on=False
            ))

            text_x = (line_x_start + line_x_end) / 2
            self._figure.text(text_x, line_y + 0.02, "Sampling ratio",
                            fontsize=font_size, ha='center', va='bottom', fontweight='bold')

        self._canvas.draw()

    def _apply_canvas_size(self):
        """Apply the canvas size based on natural figure dimensions."""
        if not hasattr(self, '_natural_width_px'):
            return

        # Always use natural size (scale 1.0) to avoid layout issues
        width = int(self._natural_width_px)
        height = int(self._natural_height_px)

        # Set canvas fixed size to match figure
        self._canvas.setFixedSize(width, height)

        # Force visual update
        self._canvas.updateGeometry()
        self._scroll_area.viewport().update()

        # Update scale label (always 100% now)
        self._scale_label.setText(f"Size: {width}×{height}px")

    def _fit_to_window(self):
        """Adjust image size to fit the figure in the visible scroll area."""
        columns = self.column_list.get_columns()
        row_indices = self._get_row_indices()

        if not columns or not row_indices:
            return

        n_rows = len(row_indices)
        n_cols = len(columns)

        # Get available space
        viewport = self._scroll_area.viewport()
        available_width = viewport.width() - 40  # Leave margin
        available_height = viewport.height() - 40

        # Current gap values
        row_gap = self.row_spacing_spin.value()
        col_gap = self.col_spacing_spin.value()

        # Calculate margins (scale with font size)
        show_row_labels = self.show_row_labels_cb.isChecked()
        show_labels = self.show_labels_cb.isChecked()
        show_ratio_header = self.show_ratio_header_cb.isChecked()
        font_size = self.font_size_spin.value()

        left_margin = int(font_size * 12) if show_row_labels else 10
        right_margin = 10
        top_margin = int(font_size * 5) if show_labels else 20
        if show_ratio_header and show_labels:
            top_margin += int(font_size * 2.5)
        bottom_margin = 10

        # Calculate max image size that fits
        # Width: left_margin + n_cols * img_size + (n_cols-1) * col_gap + right_margin <= available_width
        # Height: top_margin + n_rows * img_size + (n_rows-1) * row_gap + bottom_margin <= available_height
        content_width = available_width - left_margin - right_margin - (n_cols - 1) * col_gap
        content_height = available_height - top_margin - bottom_margin - (n_rows - 1) * row_gap

        max_img_size_w = content_width / n_cols if n_cols > 0 else 256
        max_img_size_h = content_height / n_rows if n_rows > 0 else 256

        new_img_size = int(min(max_img_size_w, max_img_size_h))
        new_img_size = max(32, min(256, new_img_size))  # Clamp to valid range

        # Update the image size spinbox (this will trigger _update_preview)
        self.image_size_spin.setValue(new_img_size)

    def _on_save(self):
        """Save the current figure."""
        self._save_figure(self._figure, "samples_grid.png")


# ============================================================================
# Interactive HTML Report Popup
# ============================================================================

class InteractiveHTMLPopup(BaseFigureExportPopup):
    """
    Popup for exporting an interactive HTML report with all charts.
    Uses Plotly for interactive visualizations.
    """

    def __init__(self, tests: List[Dict[str, Any]], logger=None, parent=None):
        super().__init__(tests, logger, parent)
        self.setWindowTitle("Interactive HTML Report")
        self.setMinimumSize(800, 600)
        self.resize(900, 650)

        self._setup_ui()

    def _setup_ui(self):
        """Setup the popup UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Title
        title = QLabel("Export Interactive HTML Report")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(title)

        desc = QLabel(
            "Generate an interactive HTML report with Plotly charts. "
            "The report will be self-contained and can be opened in any modern web browser."
        )
        desc.setStyleSheet("color: #666;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Content options
        content_group = QGroupBox("Report Content")
        content_layout = QVBoxLayout(content_group)

        self.include_summary = QCheckBox("Include summary statistics")
        self.include_summary.setChecked(True)
        content_layout.addWidget(self.include_summary)

        self.include_quality = QCheckBox("Include quality metrics charts (PSNR, SSIM, LPIPS)")
        self.include_quality.setChecked(True)
        content_layout.addWidget(self.include_quality)

        self.include_timing = QCheckBox("Include timing comparison chart")
        self.include_timing.setChecked(True)
        content_layout.addWidget(self.include_timing)

        self.include_energy = QCheckBox("Include energy consumption chart")
        self.include_energy.setChecked(True)
        content_layout.addWidget(self.include_energy)

        self.include_table = QCheckBox("Include detailed results table")
        self.include_table.setChecked(True)
        content_layout.addWidget(self.include_table)

        layout.addWidget(content_group)

        # Style options
        style_group = QGroupBox("Style Options")
        style_layout = QGridLayout(style_group)

        style_layout.addWidget(QLabel("Color theme:"), 0, 0)
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Light", "Dark", "Seaborn"])
        style_layout.addWidget(self.theme_combo, 0, 1)

        style_layout.addWidget(QLabel("Chart height (px):"), 1, 0)
        self.chart_height_spin = QSpinBox()
        self.chart_height_spin.setMinimum(300)
        self.chart_height_spin.setMaximum(800)
        self.chart_height_spin.setValue(450)
        style_layout.addWidget(self.chart_height_spin, 1, 1)

        layout.addWidget(style_group)

        # Output path
        path_group = QGroupBox("Output File")
        path_layout = QHBoxLayout(path_group)

        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Select output file...")
        self.path_edit.setReadOnly(True)
        path_layout.addWidget(self.path_edit)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._on_browse)
        path_layout.addWidget(browse_btn)

        layout.addWidget(path_group)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.export_btn = QPushButton("Export HTML Report")
        self.export_btn.setEnabled(False)
        self.export_btn.setMinimumWidth(180)
        self.export_btn.setStyleSheet("""
            QPushButton {
                background-color: #9C27B0;
                color: white;
                font-weight: bold;
                padding: 10px 20px;
                border-radius: 4px;
                font-size: 13px;
            }
            QPushButton:hover:enabled {
                background-color: #7B1FA2;
            }
            QPushButton:disabled {
                background-color: #ccc;
                color: #888;
            }
        """)
        self.export_btn.clicked.connect(self._on_export)
        btn_layout.addWidget(self.export_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)

        layout.addLayout(btn_layout)

        layout.addStretch()

    def _on_browse(self):
        """Handle browse button click."""
        default_name = f"batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save HTML Report",
            str(BATCH_TESTS_DIR / default_name),
            "HTML Files (*.html);;All Files (*.*)"
        )

        if file_path:
            self.path_edit.setText(file_path)
            self.export_btn.setEnabled(True)

    def _on_export(self):
        """Export the HTML report."""
        output_path = Path(self.path_edit.text())

        try:
            html_content = self._generate_html()

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            self.logger.info("Exported HTML report to %s", output_path)
            QMessageBox.information(
                self, "Export Complete",
                f"HTML report exported to:\n{output_path}"
            )

        except Exception as e:
            self.logger.error("Failed to export HTML: %s", e)
            QMessageBox.warning(self, "Error", f"Failed to export HTML:\n{e}")

    def _generate_html(self) -> str:
        """Generate the HTML content with Plotly charts."""
        theme = self.theme_combo.currentText().lower()
        chart_height = self.chart_height_spin.value()

        # Collect data
        experiments = sorted(set(t.get("_experiment_name", "Unknown") for t in self._tests))
        test_names = [t.get("name", f"Test {i}") for i, t in enumerate(self._tests)]

        # Build chart data
        psnr_data = [t.get("psnr_denoised", 0) for t in self._tests]
        ssim_data = [t.get("ssim_denoised", 0) for t in self._tests]
        lpips_data = [t.get("lpips_denoised", 0) for t in self._tests]
        timing_data = [t.get("timing_mean_ms", 0) for t in self._tests]
        energy_data = [t.get("energy_mean_mj", 0) for t in self._tests]

        # Determine template based on theme
        if theme == "dark":
            template = "plotly_dark"
            bg_color = "#1e1e1e"
            text_color = "#fff"
            card_bg = "#2d2d2d"
        elif theme == "seaborn":
            template = "seaborn"
            bg_color = "#f5f5f5"
            text_color = "#333"
            card_bg = "#fff"
        else:
            template = "plotly_white"
            bg_color = "#f5f5f5"
            text_color = "#333"
            card_bg = "#fff"

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Batch Test Comparison Report</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: {bg_color};
            color: {text_color};
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        h1 {{
            color: #1976D2;
            border-bottom: 3px solid #1976D2;
            padding-bottom: 15px;
            margin-bottom: 10px;
        }}
        .meta {{
            color: #666;
            font-size: 14px;
            margin-bottom: 30px;
        }}
        .summary-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            margin-bottom: 30px;
        }}
        .summary-card {{
            flex: 1;
            min-width: 180px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        .summary-card.green {{ background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }}
        .summary-card.orange {{ background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }}
        .summary-card.blue {{ background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); }}
        .summary-value {{
            font-size: 32px;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .summary-label {{
            font-size: 13px;
            opacity: 0.9;
        }}
        .chart-card {{
            background: {card_bg};
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 25px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        }}
        .chart-title {{
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 15px;
            color: {text_color};
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #1976D2;
            color: white;
            font-weight: bold;
        }}
        tr:hover {{
            background-color: rgba(25, 118, 210, 0.1);
        }}
        .footer {{
            margin-top: 40px;
            padding: 20px;
            text-align: center;
            color: #999;
            font-size: 12px;
            border-top: 1px solid #ddd;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Batch Test Comparison Report</h1>
        <p class="meta">
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
            Experiments: {', '.join(experiments)}<br>
            Total tests: {len(self._tests)}
        </p>
"""

        # Summary cards
        if self.include_summary.isChecked():
            avg_psnr = np.mean([v for v in psnr_data if v]) if any(psnr_data) else 0
            avg_ssim = np.mean([v for v in ssim_data if v]) if any(ssim_data) else 0
            avg_time = np.mean([v for v in timing_data if v]) if any(timing_data) else 0
            avg_energy = np.mean([v for v in energy_data if v]) if any(energy_data) else 0

            html += f"""
        <div class="summary-row">
            <div class="summary-card green">
                <div class="summary-value">{avg_psnr:.2f} dB</div>
                <div class="summary-label">Average PSNR</div>
            </div>
            <div class="summary-card blue">
                <div class="summary-value">{avg_ssim:.4f}</div>
                <div class="summary-label">Average SSIM</div>
            </div>
            <div class="summary-card orange">
                <div class="summary-value">{avg_time:.2f} ms</div>
                <div class="summary-label">Average Inference Time</div>
            </div>
            <div class="summary-card">
                <div class="summary-value">{avg_energy:.2f} mJ</div>
                <div class="summary-label">Average Energy</div>
            </div>
        </div>
"""

        # Quality chart
        if self.include_quality.isChecked():
            html += f"""
        <div class="chart-card">
            <div class="chart-title">Quality Metrics Comparison</div>
            <div id="quality-chart"></div>
        </div>
        <script>
            var qualityData = [
                {{
                    x: {test_names},
                    y: {psnr_data},
                    name: 'PSNR (dB)',
                    type: 'bar',
                    marker: {{ color: '#4CAF50' }}
                }}
            ];
            var qualityLayout = {{
                template: '{template}',
                height: {chart_height},
                barmode: 'group',
                yaxis: {{ title: 'PSNR (dB)' }},
                legend: {{ orientation: 'h', y: -0.2 }}
            }};
            Plotly.newPlot('quality-chart', qualityData, qualityLayout, {{responsive: true}});
        </script>
"""

        # Timing chart
        if self.include_timing.isChecked():
            html += f"""
        <div class="chart-card">
            <div class="chart-title">Inference Time Comparison</div>
            <div id="timing-chart"></div>
        </div>
        <script>
            var timingData = [
                {{
                    x: {test_names},
                    y: {timing_data},
                    type: 'bar',
                    marker: {{
                        color: {timing_data},
                        colorscale: 'Blues',
                        reversescale: true
                    }}
                }}
            ];
            var timingLayout = {{
                template: '{template}',
                height: {chart_height},
                yaxis: {{ title: 'Time (ms)' }},
                showlegend: false
            }};
            Plotly.newPlot('timing-chart', timingData, timingLayout, {{responsive: true}});
        </script>
"""

        # Energy chart
        if self.include_energy.isChecked() and any(energy_data):
            html += f"""
        <div class="chart-card">
            <div class="chart-title">Energy Consumption Comparison</div>
            <div id="energy-chart"></div>
        </div>
        <script>
            var energyData = [
                {{
                    x: {test_names},
                    y: {energy_data},
                    type: 'bar',
                    marker: {{
                        color: {energy_data},
                        colorscale: 'Oranges'
                    }}
                }}
            ];
            var energyLayout = {{
                template: '{template}',
                height: {chart_height},
                yaxis: {{ title: 'Energy (mJ)' }},
                showlegend: false
            }};
            Plotly.newPlot('energy-chart', energyData, energyLayout, {{responsive: true}});
        </script>
"""

        # Results table
        if self.include_table.isChecked():
            html += """
        <div class="chart-card">
            <div class="chart-title">Detailed Results</div>
            <table>
                <thead>
                    <tr>
                        <th>Experiment</th>
                        <th>Test</th>
                        <th>Mask</th>
                        <th>Model</th>
                        <th>PSNR (dB)</th>
                        <th>SSIM</th>
                        <th>Time (ms)</th>
                    </tr>
                </thead>
                <tbody>
"""
            for test in self._tests:
                html += f"""
                    <tr>
                        <td>{test.get('_experiment_name', '-')}</td>
                        <td>{test.get('name', '-')}</td>
                        <td>{test.get('mask_type', '-')}</td>
                        <td>{test.get('model_name', '-')}</td>
                        <td>{test.get('psnr_denoised', 0):.2f}</td>
                        <td>{test.get('ssim_denoised', 0):.4f}</td>
                        <td>{test.get('timing_mean_ms', 0):.2f}</td>
                    </tr>
"""
            html += """
                </tbody>
            </table>
        </div>
"""

        # Footer
        html += """
        <div class="footer">
            Generated by SPIm - Batch Reports Module
        </div>
    </div>
</body>
</html>
"""

        return html
