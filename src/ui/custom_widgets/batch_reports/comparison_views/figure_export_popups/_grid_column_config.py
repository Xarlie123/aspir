"""Column configuration widgets for the Samples Grid popup (Fig 2)."""
from __future__ import annotations

from PyQt5.QtCore import QMimeData, QPoint, Qt, pyqtSignal
from PyQt5.QtGui import QDrag, QPainter, QPixmap
from PyQt5.QtWidgets import (
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

    def __init__(self, config: GridColumnConfig, tests: list[dict], parent=None):
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

    def __init__(self, tests: list[dict], parent=None):
        super().__init__(parent)
        self.tests = tests
        self.columns: list[GridColumnConfig] = []
        self.cards: list[GridColumnCardWidget] = []
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

    def get_columns(self) -> list[GridColumnConfig]:
        return self.columns.copy()
