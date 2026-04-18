"""Table refresh and metric-coloring helpers for the summary view.

The ``view`` argument is the :class:`SummaryView` instance. These helpers
read ``view._tests``, ``view.COLUMNS``, the table widget and related state,
and rebuild the table contents.
"""
from __future__ import annotations

from typing import Any

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtWidgets import QCheckBox, QHBoxLayout, QTableWidgetItem, QWidget


def get_nested_value(data: dict, key: str):
    """Get a value from a nested dictionary using dot notation."""
    keys = key.split(".")
    value = data
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return None
    return value


def calculate_metric_ranges(view) -> dict[str, tuple]:
    """Calculate min/max ranges for numeric metrics across ``view._tests``."""
    ranges = {}
    for _header_text, key, _format_func, higher_is_better, _ in view.COLUMNS:
        if higher_is_better is not None:
            values = []
            for test in view._tests:
                val = get_nested_value(test, key)
                if isinstance(val, (int, float)):
                    values.append(val)
            if values:
                ranges[key] = (min(values), max(values))
    return ranges


def get_metric_color(value: float, min_val: float, max_val: float,
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


def refresh_table(view):
    """Refresh the table with current test data."""
    # Block signals during refresh to avoid spurious emissions
    view.table.blockSignals(True)

    view.table.setRowCount(len(view._tests))
    view._checkbox_widgets.clear()

    view.table.blockSignals(False)

    if not view._tests:
        view.info_label.setText("No tests loaded")
        view.export_csv_btn.setEnabled(False)
        view.copy_btn.setEnabled(False)
        view.select_all_btn.setEnabled(False)
        view.select_none_btn.setEnabled(False)
        return

    # Calculate min/max for color coding
    metrics_ranges = calculate_metric_ranges(view)

    # Populate table
    for row, test in enumerate(view._tests):
        for col, (_header_text, key, format_func, higher_is_better, _) in enumerate(view.COLUMNS):
            if key == "_checkbox":
                # Checkbox column
                checkbox = QCheckBox()
                checkbox.setChecked(row in view._selected_indices)
                checkbox.setStyleSheet("margin-left: 6px;")
                checkbox.stateChanged.connect(
                    lambda state, r=row: view._on_checkbox_changed(r, state)
                )
                view._checkbox_widgets.append(checkbox)

                # Create a widget to center the checkbox
                widget = QWidget()
                cb_layout = QHBoxLayout(widget)
                cb_layout.addWidget(checkbox)
                cb_layout.setAlignment(Qt.AlignCenter)
                cb_layout.setContentsMargins(0, 0, 0, 0)
                view.table.setCellWidget(row, col, widget)
                continue

            value = get_nested_value(test, key)
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
                        color = get_metric_color(value, min_val, max_val, higher_is_better)
                        item.setBackground(color)
            else:
                item.setText("-")

            item.setTextAlignment(Qt.AlignCenter)
            # Make items not editable but still selectable
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            view.table.setItem(row, col, item)

    # Update info label
    view._update_info_label()
    view.export_csv_btn.setEnabled(True)
    view.copy_btn.setEnabled(True)
    view.select_all_btn.setEnabled(True)
    view.select_none_btn.setEnabled(True)


def sync_data_with_visual_order(view):
    """Sync internal data order with visual table order after sorting."""
    v_header = view.table.verticalHeader()
    new_tests: list[dict[str, Any]] = []
    new_selected: set = set()

    for visual_row in range(view.table.rowCount()):
        logical_row = v_header.logicalIndex(visual_row)
        if logical_row < len(view._tests):
            new_tests.append(view._tests[logical_row])
            if logical_row in view._selected_indices:
                new_selected.add(visual_row)

    view._tests = new_tests
    view._selected_indices = new_selected

    # Emit reorder signal
    view.tests_reordered.emit(view._tests.copy())
