"""
Summary view for Batch Reports - displays all tests in a sortable table.
Supports row reordering via drag-and-drop, context menu, and test selection for filtering.
"""
import logging
from typing import List, Dict, Any, Optional

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QPushButton, QLabel, QFileDialog, QMessageBox,
    QAbstractItemView, QMenu, QCheckBox, QAction, QStyle, QInputDialog
)
from PyQt5.QtCore import Qt, pyqtSignal, QMimeData, QPoint
from PyQt5.QtGui import QColor, QBrush, QDrag, QPixmap, QPainter


class DraggableRowTableWidget(QTableWidget):
    """
    QTableWidget subclass that supports dragging entire rows.
    Uses custom drag-drop handling to avoid Qt's InternalMove item manipulation.
    """

    rows_reordered = pyqtSignal(int, int)  # from_row, to_row

    def __init__(self, parent=None):
        super().__init__(parent)

        # Enable row selection mode
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.SingleSelection)

        # Use DragDrop mode but ignore the default drop action
        # This gives us the visual feedback without Qt moving items
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DragDrop)
        self.setDefaultDropAction(Qt.IgnoreAction)
        self.setDropIndicatorShown(True)

        # Track the row being dragged
        self._drag_start_row = -1

    def startDrag(self, supportedActions):
        """Remember which row we're dragging before the drag starts."""
        self._drag_start_row = self.currentRow()
        super().startDrag(supportedActions)

    def dropEvent(self, event):
        """Handle drop event to reorder rows."""
        # Only accept drops from this widget
        if event.source() != self:
            event.ignore()
            return

        # Get the row being dropped on
        drop_row = self.rowAt(event.pos().y())
        if drop_row == -1:
            # Dropped below last row
            drop_row = self.rowCount() - 1

        # Use the saved drag start row
        drag_row = self._drag_start_row
        if drag_row == -1 or drag_row == drop_row:
            event.ignore()
            return

        # Emit signal for external handling (data reordering)
        # The handler will refresh the table completely
        self.rows_reordered.emit(drag_row, drop_row)

        # Ignore the event to prevent Qt from manipulating items
        event.ignore()

        # Reset drag state
        self._drag_start_row = -1

    def dragEnterEvent(self, event):
        """Accept drag events from this widget."""
        if event.source() == self:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        """Show drop indicator during drag."""
        if event.source() == self:
            event.accept()
        else:
            event.ignore()


class SummaryView(QWidget):
    """
    Summary view displaying a sortable table of all tests across experiments.

    Features:
    - Checkbox column for selecting tests to include in analyses
    - Drag rows to reorder (click and drag any cell)
    - Context menu for move up/down/top/bottom
    - Sortable columns (click header)
    - Color-coded metrics (best green, worst red)
    - Export to CSV

    Signals:
        tests_reordered: Emitted when tests are reordered (list of test dicts in new order)
        selection_changed: Emitted when checkbox selection changes (list of selected test indices)
    """

    # Signals
    tests_reordered = pyqtSignal(list)  # List of test dicts in new order
    selection_changed = pyqtSignal(list)  # List of selected (checked) test indices

    # Columns configuration: (header, data_key, format_func, higher_is_better, default_visible)
    # Keys match the flat structure exported by BatchTestRunner
    COLUMNS = [
        ("", "_checkbox", None, None, True),  # Checkbox column - always visible
        ("Experiment", "_experiment_name", str, None, True),
        ("Test", "name", str, None, True),
        ("Mask Type", "mask_type", str, None, True),
        ("Reconstruction", "reconstruction_method", str, None, True),
        ("Model", "model_name", str, None, True),
        # Quality metrics - denoised (after DNN)
        ("PSNR (dB)", "psnr_denoised", lambda x: f"{x:.2f}", True, True),
        ("SSIM", "ssim_denoised", lambda x: f"{x:.4f}", True, True),
        ("LPIPS", "lpips_denoised", lambda x: f"{x:.4f}", False, True),  # Lower is better
        # Quality metrics - reconstructed (before DNN, after mask application)
        ("PSNR Recons", "psnr_recons", lambda x: f"{x:.2f}", True, False),
        ("SSIM Recons", "ssim_recons", lambda x: f"{x:.4f}", True, False),
        ("LPIPS Recons", "lpips_recons", lambda x: f"{x:.4f}", False, False),
        # Timing metrics
        ("Time (ms)", "timing_mean_ms", lambda x: f"{x:.2f}", False, True),  # Lower is better
        ("Time Std", "timing_std_ms", lambda x: f"{x:.3f}", False, False),
        ("Time Min", "timing_min_ms", lambda x: f"{x:.2f}", False, False),
        ("Time Max", "timing_max_ms", lambda x: f"{x:.2f}", False, False),
        ("Acq Time (ms)", "timing_acquisition_ms", lambda x: f"{x:.3f}", False, False),
        ("Num Patterns", "timing_num_patterns", lambda x: f"{int(x)}", None, False),
        # Energy metrics
        ("Energy (mJ)", "energy_mean_mj", lambda x: f"{x:.3f}", False, True),  # Lower is better
        ("Energy Std", "energy_std_mj", lambda x: f"{x:.3f}", False, False),
        ("Power (W)", "energy_mean_watts", lambda x: f"{x:.2f}", False, False),
        ("Power Std", "energy_std_watts", lambda x: f"{x:.3f}", False, False),
        ("Efficiency (img/J)", "efficiency_images_per_joule", lambda x: f"{x:.1f}", True, False),
        ("GPU/Device", "energy_device_name", str, None, False),
        # Training parameters
        ("Epochs", "epochs", lambda x: f"{int(x)}", None, False),
        ("Batch Size", "batch_size", lambda x: f"{int(x)}", None, False),
        ("Learning Rate", "learning_rate", lambda x: f"{x:.5f}", None, False),
        # Dataset split
        ("Train %", "train_split", lambda x: f"{int(x)}%", None, False),
        ("Val %", "val_split", lambda x: f"{int(x)}%", None, False),
        ("Test %", "test_split", lambda x: f"{int(x)}%", None, False),
        ("Status", "status", str, None, False),
    ]

    def __init__(self, logger=None, parent=None):
        super().__init__(parent)

        if logger:
            self.logger = logger.getChild("SummaryView")
        else:
            self.logger = logging.getLogger("SummaryView")

        self._tests: List[Dict[str, Any]] = []
        self._selected_indices: set = set()  # Indices of selected tests
        self._checkbox_widgets: List[QCheckBox] = []  # Track checkbox widgets

        # Track visible columns (by index) - initialize from default_visible
        self._visible_columns: set = {
            i for i, col in enumerate(self.COLUMNS) if col[4]  # default_visible
        }

        self._setup_ui()

    def _setup_ui(self):
        """Setup the summary view UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Header row with info, select all, and export button
        header_layout = QHBoxLayout()

        self.info_label = QLabel("No tests loaded")
        self.info_label.setStyleSheet("color: #666; font-size: 12px;")
        header_layout.addWidget(self.info_label)

        header_layout.addStretch()

        # Select all / none buttons
        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.setEnabled(False)
        self.select_all_btn.setStyleSheet("""
            QPushButton {
                background-color: #f5f5f5;
                color: #333;
                border: 1px solid #ccc;
                padding: 4px 10px;
                border-radius: 3px;
                font-size: 11px;
            }
            QPushButton:hover:enabled {
                background-color: #e0e0e0;
            }
            QPushButton:disabled {
                color: #999;
            }
        """)
        self.select_all_btn.clicked.connect(self._on_select_all)
        header_layout.addWidget(self.select_all_btn)

        self.select_none_btn = QPushButton("Select None")
        self.select_none_btn.setEnabled(False)
        self.select_none_btn.setStyleSheet("""
            QPushButton {
                background-color: #f5f5f5;
                color: #333;
                border: 1px solid #ccc;
                padding: 4px 10px;
                border-radius: 3px;
                font-size: 11px;
            }
            QPushButton:hover:enabled {
                background-color: #e0e0e0;
            }
            QPushButton:disabled {
                color: #999;
            }
        """)
        self.select_none_btn.clicked.connect(self._on_select_none)
        header_layout.addWidget(self.select_none_btn)

        header_layout.addSpacing(10)

        # Columns visibility button
        self.columns_btn = QPushButton("Columns ▼")
        self.columns_btn.setStyleSheet("""
            QPushButton {
                background-color: #f5f5f5;
                color: #333;
                border: 1px solid #ccc;
                padding: 4px 10px;
                border-radius: 3px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
            QPushButton::menu-indicator {
                width: 0px;
            }
        """)
        self.columns_btn.clicked.connect(self._show_columns_menu)
        header_layout.addWidget(self.columns_btn)

        header_layout.addSpacing(10)

        self.copy_btn = QPushButton("Copy to Clipboard")
        self.copy_btn.setEnabled(False)
        self.copy_btn.setStyleSheet("""
            QPushButton {
                background-color: #607D8B;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
            }
            QPushButton:hover:enabled {
                background-color: #455A64;
            }
            QPushButton:disabled {
                background-color: #ccc;
                color: #888;
            }
        """)
        self.copy_btn.clicked.connect(self._on_copy_to_clipboard)
        header_layout.addWidget(self.copy_btn)

        header_layout.addSpacing(5)

        self.export_csv_btn = QPushButton("Export to CSV")
        self.export_csv_btn.setEnabled(False)
        self.export_csv_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
            }
            QPushButton:hover:enabled {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #ccc;
                color: #888;
            }
        """)
        self.export_csv_btn.clicked.connect(self._on_export_csv)
        header_layout.addWidget(self.export_csv_btn)

        layout.addLayout(header_layout)

        # Table with draggable rows
        self.table = DraggableRowTableWidget()
        self.table.setColumnCount(len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels([col[0] for col in self.COLUMNS])
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)

        # Connect row reorder signal
        self.table.rows_reordered.connect(self._on_row_dragged)

        # Context menu
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._show_context_menu)

        self.table.setStyleSheet("""
            QTableWidget {
                border: 1px solid #ccc;
                border-radius: 4px;
                background-color: white;
                gridline-color: #e0e0e0;
            }
            QTableWidget::item {
                padding: 4px;
            }
            QTableWidget::item:selected {
                background-color: #e3f2fd;
                color: #1976d2;
            }
            QHeaderView::section {
                background-color: #f5f5f5;
                border: none;
                border-bottom: 1px solid #ccc;
                border-right: 1px solid #e0e0e0;
                padding: 6px;
                font-weight: bold;
            }
            QHeaderView::section:vertical {
                background-color: #e8e8e8;
                border: none;
                border-bottom: 1px solid #ccc;
                border-right: 1px solid #ccc;
                padding: 4px 8px;
                min-width: 30px;
            }
            QHeaderView::section:vertical:hover {
                background-color: #d0d0d0;
            }
        """)

        # Set header resize modes
        header = self.table.horizontalHeader()
        # Interactive mode allows users to drag column borders to resize
        header.setSectionResizeMode(QHeaderView.Interactive)
        # Stretch last section to fill remaining space
        header.setStretchLastSection(True)
        # Allow horizontal scrollbar when columns exceed window width
        self.table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)

        # Set minimum widths for better readability
        # Checkbox column - fixed narrow width
        header.setSectionResizeMode(0, QHeaderView.Fixed)
        self.table.setColumnWidth(0, 30)

        # Set reasonable default widths for other columns
        # These will be applied to all columns up to the count in the table
        default_width = 80  # Default width for any column
        for col in range(1, self.table.columnCount()):
            self.table.setColumnWidth(col, default_width)

        # Override specific widths for readability
        specific_widths = {
            1: 100,   # Experiment
            2: 120,   # Test
            3: 100,   # Mask Type
            4: 100,   # Reconstruction
            5: 70,    # Model
        }
        for col, width in specific_widths.items():
            if col < self.table.columnCount():
                self.table.setColumnWidth(col, width)

        # Sortable header with click to sort
        header.setSectionsClickable(True)
        header.sectionClicked.connect(self._on_header_clicked)
        self._sort_column = -1
        self._sort_order = Qt.AscendingOrder

        # Apply initial column visibility
        self._apply_column_visibility()

        layout.addWidget(self.table, 1)

        # Bottom row: hint and legend
        bottom_layout = QHBoxLayout()

        hint_label = QLabel("💡 Drag rows to reorder • Right-click for more options")
        hint_label.setStyleSheet("color: #888; font-size: 10px;")
        bottom_layout.addWidget(hint_label)

        bottom_layout.addStretch()

        legend_label = QLabel("Color legend:")
        legend_label.setStyleSheet("color: #666; font-size: 11px;")
        bottom_layout.addWidget(legend_label)

        best_label = QLabel("  Best")
        best_label.setStyleSheet(
            "background-color: #c8e6c9; color: #2e7d32; padding: 2px 8px; "
            "border-radius: 3px; font-size: 11px;"
        )
        bottom_layout.addWidget(best_label)

        worst_label = QLabel("  Worst")
        worst_label.setStyleSheet(
            "background-color: #ffcdd2; color: #c62828; padding: 2px 8px; "
            "border-radius: 3px; font-size: 11px;"
        )
        bottom_layout.addWidget(worst_label)

        layout.addLayout(bottom_layout)

    def set_tests(self, tests: List[Dict[str, Any]]):
        """
        Set the tests to display in the table.

        Args:
            tests: List of test dictionaries with metrics
        """
        self._tests = list(tests)  # Make a copy
        # Select all by default
        self._selected_indices = set(range(len(tests)))
        self._refresh_table()

    def _refresh_table(self):
        """Refresh the table with current test data."""
        # Block signals during refresh to avoid spurious emissions
        self.table.blockSignals(True)

        self.table.setRowCount(len(self._tests))
        self._checkbox_widgets.clear()

        self.table.blockSignals(False)

        if not self._tests:
            self.info_label.setText("No tests loaded")
            self.export_csv_btn.setEnabled(False)
            self.copy_btn.setEnabled(False)
            self.select_all_btn.setEnabled(False)
            self.select_none_btn.setEnabled(False)
            return

        # Calculate min/max for color coding
        metrics_ranges = self._calculate_metric_ranges()

        # Populate table
        for row, test in enumerate(self._tests):
            for col, (header_text, key, format_func, higher_is_better, _) in enumerate(self.COLUMNS):
                if key == "_checkbox":
                    # Checkbox column
                    checkbox = QCheckBox()
                    checkbox.setChecked(row in self._selected_indices)
                    checkbox.setStyleSheet("margin-left: 6px;")
                    checkbox.stateChanged.connect(
                        lambda state, r=row: self._on_checkbox_changed(r, state)
                    )
                    self._checkbox_widgets.append(checkbox)

                    # Create a widget to center the checkbox
                    widget = QWidget()
                    cb_layout = QHBoxLayout(widget)
                    cb_layout.addWidget(checkbox)
                    cb_layout.setAlignment(Qt.AlignCenter)
                    cb_layout.setContentsMargins(0, 0, 0, 0)
                    self.table.setCellWidget(row, col, widget)
                    continue

                value = self._get_nested_value(test, key)
                item = QTableWidgetItem()

                if value is not None:
                    if callable(format_func):
                        try:
                            display_text = format_func(value)
                        except (TypeError, ValueError):
                            display_text = str(value)
                    else:
                        display_text = str(value)
                    item.setText(display_text)

                    # Store numeric value for sorting
                    if isinstance(value, (int, float)):
                        item.setData(Qt.UserRole, value)

                    # Color code numeric metrics
                    if higher_is_better is not None and key in metrics_ranges:
                        min_val, max_val = metrics_ranges[key]
                        if min_val != max_val:
                            color = self._get_metric_color(value, min_val, max_val, higher_is_better)
                            item.setBackground(color)
                else:
                    item.setText("-")

                item.setTextAlignment(Qt.AlignCenter)
                # Make items not editable but still selectable
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(row, col, item)

        # Update info label
        self._update_info_label()
        self.export_csv_btn.setEnabled(True)
        self.copy_btn.setEnabled(True)
        self.select_all_btn.setEnabled(True)
        self.select_none_btn.setEnabled(True)

    def _update_info_label(self):
        """Update the info label with current selection status."""
        if not self._tests:
            self.info_label.setText("No tests loaded")
            return

        experiment_count = len(set(t.get("_experiment_name", "") for t in self._tests))
        selected = len(self._selected_indices)
        total = len(self._tests)

        if selected == total:
            self.info_label.setText(
                f"{total} tests from {experiment_count} experiment(s)"
            )
        else:
            self.info_label.setText(
                f"{selected}/{total} tests selected from {experiment_count} experiment(s)"
            )

    def _on_checkbox_changed(self, row: int, state: int):
        """Handle checkbox state change."""
        if state == Qt.Checked:
            self._selected_indices.add(row)
        else:
            self._selected_indices.discard(row)

        self._update_info_label()
        self.selection_changed.emit(list(self._selected_indices))
        self.logger.debug("Selection changed: %d tests selected", len(self._selected_indices))

    def _on_select_all(self):
        """Select all tests."""
        self._selected_indices = set(range(len(self._tests)))
        for cb in self._checkbox_widgets:
            cb.blockSignals(True)
            cb.setChecked(True)
            cb.blockSignals(False)
        self._update_info_label()
        self.selection_changed.emit(list(self._selected_indices))

    def _on_select_none(self):
        """Deselect all tests."""
        self._selected_indices.clear()
        for cb in self._checkbox_widgets:
            cb.blockSignals(True)
            cb.setChecked(False)
            cb.blockSignals(False)
        self._update_info_label()
        self.selection_changed.emit(list(self._selected_indices))

    def _show_columns_menu(self):
        """Show dropdown menu to toggle column visibility."""
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: white;
                border: 1px solid #ccc;
                padding: 5px;
            }
            QMenu::item {
                padding: 5px 20px;
            }
            QMenu::item:selected {
                background-color: #e3f2fd;
            }
        """)

        for col_idx, (header, key, _, _, _) in enumerate(self.COLUMNS):
            if key == "_checkbox":
                continue  # Don't allow hiding checkbox column

            action = QAction(header if header else key, menu)
            action.setCheckable(True)
            action.setChecked(col_idx in self._visible_columns)
            action.triggered.connect(
                lambda checked, idx=col_idx: self._toggle_column_visibility(idx, checked)
            )
            menu.addAction(action)

        # Show menu below the button
        menu.exec_(self.columns_btn.mapToGlobal(
            QPoint(0, self.columns_btn.height())
        ))

    def _toggle_column_visibility(self, col_idx: int, visible: bool):
        """Toggle visibility of a column."""
        if visible:
            self._visible_columns.add(col_idx)
        else:
            self._visible_columns.discard(col_idx)

        self._apply_column_visibility()
        self.logger.debug("Column %d visibility: %s", col_idx, visible)

    def _apply_column_visibility(self):
        """Apply current column visibility settings to the table."""
        for col_idx in range(len(self.COLUMNS)):
            if col_idx == 0:  # Checkbox column always visible
                self.table.setColumnHidden(col_idx, False)
            else:
                self.table.setColumnHidden(col_idx, col_idx not in self._visible_columns)

    def _on_row_dragged(self, from_row: int, to_row: int):
        """Handle row dragged via vertical header."""
        if from_row == to_row:
            return

        self.logger.debug("Row dragged from %d to %d", from_row, to_row)

        # Move test in internal list
        test = self._tests.pop(from_row)
        self._tests.insert(to_row, test)

        # Move checkbox widget reference
        if from_row < len(self._checkbox_widgets):
            cb = self._checkbox_widgets.pop(from_row)
            self._checkbox_widgets.insert(to_row, cb)

        # Update selected indices
        new_selected = set()
        for idx in self._selected_indices:
            if idx == from_row:
                new_selected.add(to_row)
            elif from_row < to_row:
                # Moving down: indices between shift up
                if from_row < idx <= to_row:
                    new_selected.add(idx - 1)
                else:
                    new_selected.add(idx)
            else:
                # Moving up: indices between shift down
                if to_row <= idx < from_row:
                    new_selected.add(idx + 1)
                else:
                    new_selected.add(idx)
        self._selected_indices = new_selected

        # Refresh to sync visual state with data
        self._refresh_table()

        # Emit reorder signal with current ordered tests
        self.tests_reordered.emit(self._tests.copy())

    def _on_header_clicked(self, column: int):
        """Handle header click for sorting."""
        if column == 0:  # Don't sort by checkbox column
            return

        # Enable sorting temporarily
        self.table.setSortingEnabled(True)

        if self._sort_column == column:
            # Toggle sort order
            self._sort_order = Qt.DescendingOrder if self._sort_order == Qt.AscendingOrder else Qt.AscendingOrder
        else:
            self._sort_column = column
            self._sort_order = Qt.AscendingOrder

        self.table.sortItems(column, self._sort_order)

        # Update internal test order to match visual order
        self._sync_data_with_visual_order()

        # Disable sorting to allow drag-drop
        self.table.setSortingEnabled(False)

    def _sync_data_with_visual_order(self):
        """Sync internal data order with visual table order after sorting."""
        v_header = self.table.verticalHeader()
        new_tests = []
        new_selected = set()

        for visual_row in range(self.table.rowCount()):
            logical_row = v_header.logicalIndex(visual_row)
            if logical_row < len(self._tests):
                new_tests.append(self._tests[logical_row])
                if logical_row in self._selected_indices:
                    new_selected.add(visual_row)

        self._tests = new_tests
        self._selected_indices = new_selected

        # Emit reorder signal
        self.tests_reordered.emit(self._tests.copy())

    def _show_context_menu(self, pos):
        """Show context menu for row operations."""
        item = self.table.itemAt(pos)
        if item is None:
            return

        row = item.row()
        menu = QMenu(self)

        # Move actions
        move_up = QAction("Move Up", self)
        move_up.setEnabled(row > 0)
        move_up.triggered.connect(lambda: self._move_row(row, row - 1))
        menu.addAction(move_up)

        move_down = QAction("Move Down", self)
        move_down.setEnabled(row < len(self._tests) - 1)
        move_down.triggered.connect(lambda: self._move_row(row, row + 1))
        menu.addAction(move_down)

        menu.addSeparator()

        move_top = QAction("Move to Top", self)
        move_top.setEnabled(row > 0)
        move_top.triggered.connect(lambda: self._move_row(row, 0))
        menu.addAction(move_top)

        move_bottom = QAction("Move to Bottom", self)
        move_bottom.setEnabled(row < len(self._tests) - 1)
        move_bottom.triggered.connect(lambda: self._move_row(row, len(self._tests) - 1))
        menu.addAction(move_bottom)

        menu.addSeparator()

        # Selection actions
        if row in self._selected_indices:
            deselect = QAction("Deselect this test", self)
            deselect.triggered.connect(lambda: self._toggle_selection(row, False))
            menu.addAction(deselect)
        else:
            select = QAction("Select this test", self)
            select.triggered.connect(lambda: self._toggle_selection(row, True))
            menu.addAction(select)

        menu.addSeparator()

        # Rename action
        rename_action = QAction("Rename...", self)
        rename_action.triggered.connect(lambda: self._rename_test(row))
        menu.addAction(rename_action)

        menu.exec_(self.table.viewport().mapToGlobal(pos))

    def _move_row(self, from_index: int, to_index: int):
        """Move a row from one position to another."""
        if from_index == to_index:
            return

        # Move in data
        test = self._tests.pop(from_index)
        self._tests.insert(to_index, test)

        # Update selected indices
        new_selected = set()
        for idx in self._selected_indices:
            if idx == from_index:
                new_selected.add(to_index)
            elif from_index < to_index:
                if from_index < idx <= to_index:
                    new_selected.add(idx - 1)
                else:
                    new_selected.add(idx)
            else:
                if to_index <= idx < from_index:
                    new_selected.add(idx + 1)
                else:
                    new_selected.add(idx)
        self._selected_indices = new_selected

        # Refresh table
        self._refresh_table()
        self.table.selectRow(to_index)

        # Emit signal
        self.tests_reordered.emit(self._tests.copy())
        self.logger.debug("Moved row from %d to %d via context menu", from_index, to_index)

    def _toggle_selection(self, row: int, selected: bool):
        """Toggle selection for a single row."""
        if row < len(self._checkbox_widgets):
            self._checkbox_widgets[row].setChecked(selected)

    def _rename_test(self, row: int):
        """Rename a test via input dialog."""
        if row < 0 or row >= len(self._tests):
            return

        test = self._tests[row]
        current_name = test.get("name", "")

        new_name, ok = QInputDialog.getText(
            self,
            "Rename Test",
            "Enter new name:",
            text=current_name
        )

        if ok and new_name and new_name != current_name:
            # Update the test name
            self._tests[row]["name"] = new_name

            # Update the table cell directly (column 2 is "Test" name)
            name_col = 2  # "Test" column index
            item = self.table.item(row, name_col)
            if item:
                item.setText(new_name)

            self.logger.info("Renamed test from '%s' to '%s'", current_name, new_name)

            # Emit signal so other views can update
            self.tests_reordered.emit(self._tests.copy())

    def get_selected_tests(self) -> List[Dict[str, Any]]:
        """Get list of currently selected tests in current order."""
        return [self._tests[i] for i in sorted(self._selected_indices) if i < len(self._tests)]

    def get_ordered_tests(self) -> List[Dict[str, Any]]:
        """Get all tests in current display order."""
        return self._tests.copy()

    def _get_nested_value(self, data: dict, key: str):
        """Get a value from a nested dictionary using dot notation."""
        keys = key.split(".")
        value = data
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return None
        return value

    def _calculate_metric_ranges(self) -> Dict[str, tuple]:
        """Calculate min/max ranges for numeric metrics."""
        ranges = {}
        for header_text, key, format_func, higher_is_better, _ in self.COLUMNS:
            if higher_is_better is not None:
                values = []
                for test in self._tests:
                    val = self._get_nested_value(test, key)
                    if isinstance(val, (int, float)):
                        values.append(val)
                if values:
                    ranges[key] = (min(values), max(values))
        return ranges

    def _get_metric_color(self, value: float, min_val: float, max_val: float,
                          higher_is_better: bool) -> QBrush:
        """
        Get a color for a metric value based on its position in the range.

        Args:
            value: The metric value
            min_val: Minimum value in the range
            max_val: Maximum value in the range
            higher_is_better: If True, higher values get green; if False, lower is better

        Returns:
            QBrush with appropriate color
        """
        if max_val == min_val:
            return QBrush(QColor(255, 255, 255))

        # Normalize to 0-1 range
        normalized = (value - min_val) / (max_val - min_val)

        # Invert if lower is better
        if not higher_is_better:
            normalized = 1 - normalized

        # Interpolate between red (worst) and green (best)
        # Red: #ffcdd2 (255, 205, 210), Green: #c8e6c9 (200, 230, 201)
        if normalized >= 0.5:
            # Transition from white to green
            t = (normalized - 0.5) * 2
            r = int(255 - (255 - 200) * t)
            g = int(255 - (255 - 230) * t)
            b = int(255 - (255 - 201) * t)
        else:
            # Transition from red to white
            t = normalized * 2
            r = int(255 - (255 - 255) * t)
            g = int(205 + (255 - 205) * t)
            b = int(210 + (255 - 210) * t)

        return QBrush(QColor(r, g, b))

    def _on_export_csv(self):
        """Export the table data to CSV."""
        if not self._tests:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export to CSV",
            "batch_comparison.csv",
            "CSV Files (*.csv);;All Files (*.*)"
        )

        if not file_path:
            return

        try:
            import csv
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)

                # Write header (skip checkbox column)
                writer.writerow([col[0] for col in self.COLUMNS if col[1] != "_checkbox"])

                # Write data rows (only selected tests, in current order)
                tests_to_export = self.get_selected_tests() if self._selected_indices else self._tests
                for test in tests_to_export:
                    row = []
                    for header_text, key, format_func, _ in self.COLUMNS:
                        if key == "_checkbox":
                            continue
                        value = self._get_nested_value(test, key)
                        if value is not None:
                            row.append(value)
                        else:
                            row.append("")
                    writer.writerow(row)

            self.logger.info("Exported %d tests to %s", len(tests_to_export), file_path)
            QMessageBox.information(
                self, "Export Complete",
                f"Successfully exported {len(tests_to_export)} tests to:\n{file_path}"
            )

        except Exception as e:
            self.logger.error("Failed to export CSV: %s", e)
            QMessageBox.warning(self, "Export Error", f"Failed to export CSV:\n{e}")

    def _on_copy_to_clipboard(self):
        """Copy the table data to clipboard as tab-separated values."""
        if not self._tests:
            return

        try:
            from PyQt5.QtWidgets import QApplication

            lines = []

            # Build header (only visible columns, skip checkbox)
            headers = []
            for col_idx, (header, key, _, _, _) in enumerate(self.COLUMNS):
                if key == "_checkbox":
                    continue
                if col_idx in self._visible_columns:
                    headers.append(header if header else key)
            lines.append("\t".join(headers))

            # Build data rows (only selected tests, in current order)
            tests_to_copy = self.get_selected_tests() if self._selected_indices else self._tests
            for test in tests_to_copy:
                row = []
                for col_idx, (header, key, format_func, _, _) in enumerate(self.COLUMNS):
                    if key == "_checkbox":
                        continue
                    if col_idx not in self._visible_columns:
                        continue
                    value = self._get_nested_value(test, key)
                    if value is not None:
                        if callable(format_func):
                            try:
                                row.append(format_func(value))
                            except (TypeError, ValueError):
                                row.append(str(value))
                        else:
                            row.append(str(value))
                    else:
                        row.append("")
                lines.append("\t".join(row))

            clipboard_text = "\n".join(lines)
            QApplication.clipboard().setText(clipboard_text)

            self.logger.info("Copied %d tests to clipboard", len(tests_to_copy))
            QMessageBox.information(
                self, "Copied",
                f"Copied {len(tests_to_copy)} tests to clipboard"
            )

        except Exception as e:
            self.logger.error("Failed to copy to clipboard: %s", e)
            QMessageBox.warning(self, "Copy Error", f"Failed to copy to clipboard:\n{e}")

    def clear(self):
        """Clear all data from the view."""
        self._tests = []
        self._selected_indices.clear()
        self._checkbox_widgets.clear()
        self.table.setRowCount(0)
        self.info_label.setText("No tests loaded")
        self.export_csv_btn.setEnabled(False)
        self.copy_btn.setEnabled(False)
        self.select_all_btn.setEnabled(False)
        self.select_none_btn.setEnabled(False)
