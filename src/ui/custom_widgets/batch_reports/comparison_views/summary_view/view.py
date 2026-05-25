"""
Summary view for Batch Reports - displays all tests in a sortable table.
Supports row reordering via drag-and-drop, context menu, and test selection for filtering.
"""
from __future__ import annotations

import logging
from typing import Any

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QCheckBox, QInputDialog, QMenu, QWidget

from ui.custom_widgets.batch_reports.comparison_views.summary_view._export import (
    copy_to_clipboard,
    export_csv,
)
from ui.custom_widgets.batch_reports.comparison_views.summary_view._table_logic import (
    refresh_table,
    sync_data_with_visual_order,
)
from ui.custom_widgets.batch_reports.comparison_views.summary_view._ui_builder import (
    build_ui,
)


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
    tests_reordered = Signal(list)  # List of test dicts in new order
    selection_changed = Signal(list)  # List of selected (checked) test indices

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

        self._tests: list[dict[str, Any]] = []
        self._selected_indices: set = set()  # Indices of selected tests
        self._checkbox_widgets: list[QCheckBox] = []  # Track checkbox widgets

        # Track visible columns (by index) - initialize from default_visible
        self._visible_columns: set = {
            i for i, col in enumerate(self.COLUMNS) if col[4]  # default_visible
        }

        build_ui(self)

    # ----- Public API -------------------------------------------------------

    def set_tests(self, tests: list[dict[str, Any]]):
        """
        Set the tests to display in the table.

        Args:
            tests: List of test dictionaries with metrics
        """
        self._tests = list(tests)  # Make a copy
        # Select all by default
        self._selected_indices = set(range(len(tests)))
        refresh_table(self)

    def get_selected_tests(self) -> list[dict[str, Any]]:
        """Get list of currently selected tests in current order."""
        return [self._tests[i] for i in sorted(self._selected_indices) if i < len(self._tests)]

    def get_ordered_tests(self) -> list[dict[str, Any]]:
        """Get all tests in current display order."""
        return self._tests.copy()

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

    # ----- Info / selection -------------------------------------------------

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

    def _toggle_selection(self, row: int, selected: bool):
        """Toggle selection for a single row."""
        if row < len(self._checkbox_widgets):
            self._checkbox_widgets[row].setChecked(selected)

    # ----- Column visibility ------------------------------------------------

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
        menu.exec(self.columns_btn.mapToGlobal(
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

    # ----- Row reordering ---------------------------------------------------

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
        refresh_table(self)

        # Emit reorder signal with current ordered tests
        self.tests_reordered.emit(self._tests.copy())

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
        refresh_table(self)
        self.table.selectRow(to_index)

        # Emit signal
        self.tests_reordered.emit(self._tests.copy())
        self.logger.debug("Moved row from %d to %d via context menu", from_index, to_index)

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
        sync_data_with_visual_order(self)

        # Disable sorting to allow drag-drop
        self.table.setSortingEnabled(False)

    # ----- Context menu -----------------------------------------------------

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

        menu.exec(self.table.viewport().mapToGlobal(pos))

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

    # ----- Export -----------------------------------------------------------

    def _on_export_csv(self):
        """Export the table data to CSV."""
        export_csv(self)

    def _on_copy_to_clipboard(self):
        """Copy the table data to clipboard as tab-separated values."""
        copy_to_clipboard(self)
