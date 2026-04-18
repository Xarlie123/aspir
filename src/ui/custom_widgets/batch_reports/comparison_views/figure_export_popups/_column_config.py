"""Column configuration widgets for the Visual Comparison popup (Fig 9)."""
from __future__ import annotations

from PyQt5.QtCore import QMimeData, QPoint, Qt, pyqtSignal
from PyQt5.QtGui import QDrag, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFrame,
    QGraphicsOpacityEffect,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ui.custom_widgets.batch_reports.comparison_views.figure_export_popups._base import (
    DropIndicatorWidget,
)


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
        drag.exec_(Qt.MoveAction)

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


class ColumnListWidget(QWidget):
    """Widget containing draggable column cards with visual drag & drop."""

    columns_changed = pyqtSignal()

    def __init__(self, tests: list[dict], parent=None):
        super().__init__(parent)
        self.tests = tests
        self.columns: list[ColumnConfig] = []
        self.cards: list[ColumnCardWidget] = []
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

    def get_columns(self) -> list[ColumnConfig]:
        """Get current column configurations in order."""
        return self.columns.copy()
