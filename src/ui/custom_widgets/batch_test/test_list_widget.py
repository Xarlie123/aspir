"""
Widget for displaying and managing the list of tests in a batch.
Supports drag-and-drop reordering and context menu operations.
"""
import logging
from typing import Optional, List

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QPushButton, QLabel, QMenu, QSizePolicy, QAbstractItemView, QProgressBar,
    QApplication, QShortcut
)
from PySide6.QtCore import Signal, Qt, QSize
from PySide6.QtGui import QColor, QFont, QKeySequence, QDrag

from ui.custom_widgets.batch_test.test_config_model import TestConfiguration, TestStatus
from ui.custom_widgets.common.button_styles import (
    BUTTON_STYLE_GREEN, BUTTON_STYLE_BLUE, BUTTON_STYLE_RED, apply_button_style
)


class TestListItemWidget(QWidget):
    """Custom widget for displaying a test item with status, phase progress, and cancel button."""

    cancel_clicked = Signal(int)  # Emits index when cancel is clicked

    STATUS_ICONS = {
        TestStatus.PENDING: "○",
        TestStatus.RUNNING: "🔄",
        TestStatus.COMPLETED: "✅",
        TestStatus.FAILED: "❌",
        TestStatus.CANCELLED: "⏹",
    }

    STATUS_COLORS = {
        TestStatus.PENDING: "#999999",
        TestStatus.RUNNING: "#2196F3",
        TestStatus.COMPLETED: "#4CAF50",
        TestStatus.FAILED: "#F44336",
        TestStatus.CANCELLED: "#FF9800",
    }

    # Phase colors for progress bar
    PHASE_COLORS = {
        "Masks": "#9C27B0",          # Purple
        "Reconstruction": "#2196F3",  # Blue
        "Model Setup": "#00BCD4",     # Cyan
        "Training": "#FF9800",        # Orange
        "Analysis": "#4CAF50",        # Green
        "Export": "#795548",          # Brown
    }

    def __init__(self, config: TestConfiguration, index: int, parent=None):
        super().__init__(parent)
        self.config = config
        self.index = index
        self._current_phase = ""
        self._phase_progress = 0

        self._setup_ui()
        self.update_display()

    def _setup_ui(self):
        """Setup the item widget UI."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 4, 8, 4)
        main_layout.setSpacing(2)

        # Top row: status, name, progress %, cancel
        top_row = QHBoxLayout()
        top_row.setSpacing(8)

        # Status icon
        self.status_icon = QLabel()
        self.status_icon.setFixedWidth(20)
        self.status_icon.setAlignment(Qt.AlignCenter)
        top_row.addWidget(self.status_icon)

        # Test name
        self.name_label = QLabel()
        self.name_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        top_row.addWidget(self.name_label, 1)

        # Progress label (shown when running)
        self.progress_label = QLabel()
        self.progress_label.setStyleSheet("color: #2196F3; font-size: 11px;")
        self.progress_label.setFixedWidth(50)
        self.progress_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.progress_label.hide()
        top_row.addWidget(self.progress_label)

        # Cancel button (shown only when this test is running)
        self.cancel_btn = QPushButton("⏹")
        self.cancel_btn.setFixedSize(24, 24)
        self.cancel_btn.setToolTip("Stop this test")
        self.cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #fff3e0;
                border: 1px solid #ffcc80;
                border-radius: 12px;
                color: #e65100;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #ffe0b2;
                border-color: #ffb74d;
            }
            QPushButton:pressed {
                background-color: #ffcc80;
            }
        """)
        self.cancel_btn.clicked.connect(lambda: self.cancel_clicked.emit(self.index))
        self.cancel_btn.hide()
        top_row.addWidget(self.cancel_btn)

        main_layout.addLayout(top_row)

        # Bottom row: phase progress (hidden by default)
        self.phase_container = QWidget()
        phase_layout = QHBoxLayout(self.phase_container)
        phase_layout.setContentsMargins(20, 0, 0, 0)  # Indent to align with name
        phase_layout.setSpacing(5)

        # Phase name label
        self.phase_label = QLabel()
        self.phase_label.setStyleSheet("color: #666; font-size: 10px;")
        self.phase_label.setFixedWidth(90)
        phase_layout.addWidget(self.phase_label)

        # Phase progress bar
        self.phase_progress_bar = QProgressBar()
        self.phase_progress_bar.setFixedHeight(8)
        self.phase_progress_bar.setTextVisible(False)
        self.phase_progress_bar.setRange(0, 100)
        self.phase_progress_bar.setValue(0)
        self.phase_progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #e0e0e0;
                border: none;
                border-radius: 4px;
            }
            QProgressBar::chunk {
                background-color: #2196F3;
                border-radius: 4px;
            }
        """)
        phase_layout.addWidget(self.phase_progress_bar, 1)

        self.phase_container.hide()
        main_layout.addWidget(self.phase_container)

    def update_display(self):
        """Update the display based on config state."""
        status = self.config.status

        # Status icon
        icon = self.STATUS_ICONS.get(status, "○")
        color = self.STATUS_COLORS.get(status, "#999999")
        self.status_icon.setText(icon)
        self.status_icon.setStyleSheet(f"font-size: 14px;")

        # Name with color
        self.name_label.setText(self.config.name)
        self.name_label.setStyleSheet(f"color: {color}; font-weight: {'bold' if status == TestStatus.RUNNING else 'normal'};")

        # Progress
        if status == TestStatus.RUNNING and self.config.progress > 0:
            self.progress_label.setText(f"{self.config.progress}%")
            self.progress_label.show()
        else:
            self.progress_label.hide()

        # Cancel button visibility - only show when this test is actually running
        self.cancel_btn.setVisible(status == TestStatus.RUNNING)

        # Phase progress visibility
        if status == TestStatus.RUNNING and self._current_phase:
            self.phase_container.show()
        else:
            self.phase_container.hide()
            self._current_phase = ""
            self._phase_progress = 0

    def set_cancel_visible(self, visible: bool):
        """Control cancel button visibility (used during batch execution)."""
        # Only show cancel button for currently running test
        if visible and self.config.status == TestStatus.RUNNING:
            self.cancel_btn.show()
        else:
            self.cancel_btn.hide()

    def start_phase(self, phase_name: str):
        """Start a new phase - show progress bar with phase name."""
        self._current_phase = phase_name
        self._phase_progress = 0

        # Update phase label
        self.phase_label.setText(phase_name)

        # Update progress bar color based on phase
        color = self.PHASE_COLORS.get(phase_name, "#2196F3")
        self.phase_progress_bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: #e0e0e0;
                border: none;
                border-radius: 4px;
            }}
            QProgressBar::chunk {{
                background-color: {color};
                border-radius: 4px;
            }}
        """)
        self.phase_progress_bar.setValue(0)

        # Show phase container if running
        if self.config.status == TestStatus.RUNNING:
            self.phase_container.show()

    def update_phase_progress(self, phase_name: str, progress: int):
        """Update progress for a phase."""
        if phase_name == self._current_phase:
            self._phase_progress = progress
            self.phase_progress_bar.setValue(progress)

    def complete_phase(self, phase_name: str):
        """Mark a phase as complete."""
        if phase_name == self._current_phase:
            self.phase_progress_bar.setValue(100)

    def reset_phase_progress(self):
        """Reset phase progress (called when test completes/fails/cancels)."""
        self._current_phase = ""
        self._phase_progress = 0
        self.phase_container.hide()
        self.phase_progress_bar.setValue(0)


class DraggableListWidget(QListWidget):
    """
    QListWidget subclass that supports drag-and-drop reordering.
    Emits item_moved signal when an item is moved to a new position.
    """

    item_moved = Signal(int, int)  # from_index, to_index

    def __init__(self, parent=None):
        super().__init__(parent)
        self._drag_enabled = True

        # Enable drag and drop
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.InternalMove)
        self.setDefaultDropAction(Qt.MoveAction)
        self.setDropIndicatorShown(True)

        # Connect to model's rowsMoved signal for accurate index tracking
        self.model().rowsMoved.connect(self._on_rows_moved)

    def set_drag_enabled(self, enabled: bool):
        """Enable or disable drag-and-drop (disable during batch execution)."""
        self._drag_enabled = enabled
        self.setDragEnabled(enabled)
        self.setAcceptDrops(enabled)

    def _on_rows_moved(self, parent, start, end, destination, row):
        """Handle the model's rowsMoved signal for accurate index tracking."""
        if not self._drag_enabled:
            return

        # Calculate actual indices
        from_index = start
        to_index = row if row < start else row - 1

        if from_index != to_index:
            self.item_moved.emit(from_index, to_index)


class TestListWidget(QWidget):
    """
    Widget for displaying and managing the list of tests.

    Supports drag-and-drop reordering and context menu operations.

    Signals:
        test_selected(int): Emitted when a test is selected (index)
        test_added(): Emitted when a test is added
        test_removed(int): Emitted when a test is removed
        test_duplicated(int): Emitted when a test is duplicated
        tests_reordered(int, int): Emitted when tests are reordered (from_index, to_index)
        cancel_test_requested(int): Emitted when cancel is requested for a test
    """

    test_selected = Signal(int)
    test_added = Signal()
    test_removed = Signal(int)
    test_duplicated = Signal(int)
    tests_reordered = Signal(int, int)  # from_index, to_index
    cancel_test_requested = Signal(int)

    def __init__(self, parent=None, logger=None):
        super().__init__(parent)

        if logger:
            self.logger = logger.getChild("TestListWidget")
        else:
            self.logger = logging.getLogger("TestListWidget")

        self._tests: List[TestConfiguration] = []
        self._item_widgets: List[TestListItemWidget] = []
        self._read_only = False
        self._batch_running = False

        self._setup_ui()

    def _setup_ui(self):
        """Setup the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        # Header
        header = QLabel("Test List (drag to reorder)")
        header.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(header)

        # List widget with drag-and-drop support
        self.list_widget = DraggableListWidget()
        self.list_widget.setAlternatingRowColors(True)
        self.list_widget.currentRowChanged.connect(self._on_selection_changed)
        self.list_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list_widget.customContextMenuRequested.connect(self._show_context_menu)
        self.list_widget.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list_widget.item_moved.connect(self._on_item_moved)
        self.list_widget.setStyleSheet("""
            QListWidget {
                border: 1px solid #ccc;
                border-radius: 4px;
                background-color: #fafafa;
            }
            QListWidget::item {
                border-bottom: 1px solid #eee;
                padding: 0px;
            }
            QListWidget::item:selected {
                background-color: #e3f2fd;
            }
            QListWidget::item:hover {
                background-color: #f5f5f5;
            }
        """)
        layout.addWidget(self.list_widget, 1)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(5)

        self.add_btn = QPushButton("+ Add")
        self.add_btn.clicked.connect(self._on_add_clicked)
        self.add_btn.setToolTip("Add a new test configuration")
        apply_button_style(self.add_btn, BUTTON_STYLE_GREEN)
        btn_layout.addWidget(self.add_btn)

        self.remove_btn = QPushButton("− Remove")
        self.remove_btn.clicked.connect(self._on_remove_clicked)
        self.remove_btn.setEnabled(False)
        self.remove_btn.setToolTip("Remove selected test")
        apply_button_style(self.remove_btn, BUTTON_STYLE_RED)
        btn_layout.addWidget(self.remove_btn)

        self.duplicate_btn = QPushButton("📋 Duplicate")
        self.duplicate_btn.clicked.connect(self._on_duplicate_clicked)
        self.duplicate_btn.setEnabled(False)
        self.duplicate_btn.setToolTip("Duplicate selected test")
        apply_button_style(self.duplicate_btn, BUTTON_STYLE_BLUE)
        btn_layout.addWidget(self.duplicate_btn)

        layout.addLayout(btn_layout)

    def set_tests(self, tests: List[TestConfiguration]):
        """Set the list of tests."""
        self._tests = tests
        self._refresh_list()

    def _refresh_list(self):
        """Refresh the list widget from tests."""
        self.list_widget.clear()
        self._item_widgets.clear()

        for i, config in enumerate(self._tests):
            # Create custom widget for item
            item_widget = TestListItemWidget(config, i)
            item_widget.cancel_clicked.connect(self._on_cancel_clicked)

            # Create list item (taller to accommodate phase progress row)
            item = QListWidgetItem(self.list_widget)
            item.setSizeHint(QSize(0, 56))  # Increased from 40 to 56 for phase row

            self.list_widget.addItem(item)
            self.list_widget.setItemWidget(item, item_widget)
            self._item_widgets.append(item_widget)

        # Update button states
        self._update_button_states()

    def update_test_status(self, index: int):
        """Update the display for a specific test."""
        if 0 <= index < len(self._item_widgets):
            self._item_widgets[index].update_display()

    def set_read_only(self, read_only: bool):
        """Set read-only mode (during test execution)."""
        self._read_only = read_only
        self._batch_running = read_only
        self._update_button_states()

        # Disable drag-and-drop during batch execution
        self.list_widget.set_drag_enabled(not read_only)

        # Show/hide cancel buttons based on batch running state
        for widget in self._item_widgets:
            widget.set_cancel_visible(read_only)

    def _update_button_states(self):
        """Update button enabled states."""
        has_selection = self.list_widget.currentRow() >= 0
        self.add_btn.setEnabled(not self._read_only)
        self.remove_btn.setEnabled(not self._read_only and has_selection)
        self.duplicate_btn.setEnabled(not self._read_only and has_selection)

    def get_selected_index(self) -> int:
        """Get currently selected test index."""
        return self.list_widget.currentRow()

    def select_test(self, index: int):
        """Select a test by index."""
        if 0 <= index < self.list_widget.count():
            self.list_widget.setCurrentRow(index)

    def _on_selection_changed(self, row: int):
        """Handle selection change."""
        self._update_button_states()
        if row >= 0:
            self.test_selected.emit(row)

    def _on_add_clicked(self):
        """Handle add button click."""
        self.test_added.emit()

    def _on_remove_clicked(self):
        """Handle remove button click."""
        row = self.list_widget.currentRow()
        if row >= 0:
            self.test_removed.emit(row)

    def _on_duplicate_clicked(self):
        """Handle duplicate button click."""
        row = self.list_widget.currentRow()
        if row >= 0:
            self.test_duplicated.emit(row)

    def _on_cancel_clicked(self, index: int):
        """Handle cancel button click on item."""
        self.cancel_test_requested.emit(index)

    def _on_item_moved(self, from_index: int, to_index: int):
        """Handle item moved via drag-and-drop."""
        self.logger.debug("Item moved from %d to %d", from_index, to_index)
        self.tests_reordered.emit(from_index, to_index)

    def _show_context_menu(self, pos):
        """Show context menu for test item."""
        item = self.list_widget.itemAt(pos)
        menu = QMenu(self)

        if not self._read_only:
            # Add New - always available
            add_action = menu.addAction("Add New")
            add_action.triggered.connect(self._on_add_clicked)

            if item:
                row = self.list_widget.row(item)
                if row >= 0:
                    menu.addSeparator()

                    # Move actions
                    move_up_action = menu.addAction("Move Up")
                    move_up_action.setEnabled(row > 0)
                    move_up_action.triggered.connect(lambda: self._move_test(row, row - 1))

                    move_down_action = menu.addAction("Move Down")
                    move_down_action.setEnabled(row < len(self._tests) - 1)
                    move_down_action.triggered.connect(lambda: self._move_test(row, row + 1))

                    menu.addSeparator()

                    # Duplicate
                    duplicate_action = menu.addAction("Duplicate")
                    duplicate_action.triggered.connect(lambda: self.test_duplicated.emit(row))

                    menu.addSeparator()

                    # Remove
                    remove_action = menu.addAction("Remove")
                    remove_action.triggered.connect(lambda: self.test_removed.emit(row))
        else:
            # In read-only mode, only allow cancel for running/pending tests
            if item:
                row = self.list_widget.row(item)
                if row >= 0 and row < len(self._tests):
                    config = self._tests[row]
                    if config.status in (TestStatus.RUNNING, TestStatus.PENDING):
                        cancel_action = menu.addAction("Cancel Test")
                        cancel_action.triggered.connect(lambda: self.cancel_test_requested.emit(row))

        menu.exec(self.list_widget.mapToGlobal(pos))

    def _move_test(self, from_index: int, to_index: int):
        """Move a test from one position to another."""
        if from_index == to_index:
            return
        if not (0 <= from_index < len(self._tests) and 0 <= to_index < len(self._tests)):
            return

        self.logger.debug("Moving test from %d to %d", from_index, to_index)
        self.tests_reordered.emit(from_index, to_index)

    # Phase progress methods for routing updates to specific test items

    def start_phase(self, index: int, phase_name: str):
        """Start a phase for a specific test."""
        if 0 <= index < len(self._item_widgets):
            self._item_widgets[index].start_phase(phase_name)

    def update_phase_progress(self, index: int, phase_name: str, progress: int):
        """Update phase progress for a specific test."""
        if 0 <= index < len(self._item_widgets):
            self._item_widgets[index].update_phase_progress(phase_name, progress)

    def complete_phase(self, index: int, phase_name: str):
        """Complete a phase for a specific test."""
        if 0 <= index < len(self._item_widgets):
            self._item_widgets[index].complete_phase(phase_name)

    def reset_phase_progress(self, index: int):
        """Reset phase progress for a specific test."""
        if 0 <= index < len(self._item_widgets):
            self._item_widgets[index].reset_phase_progress()
