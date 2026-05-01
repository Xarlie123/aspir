"""Summary-table helpers — build, clear, refresh and copy the per-backend table.

The table is structured as one **column per compute path** (CPU run vs
GPU run) for the selected test name. When the user loads two
re-measurement reports that differ only in ``use_gpu`` (a typical
"Run both compute paths" output on Jetson), the same test name
appears twice in the loaded data — once with ``use_gpu=False`` and
once with ``use_gpu=True`` — and this builder pairs them so the user
sees the head-to-head comparison in one view.

Falling back to a single column when only one pass is loaded keeps
the original single-pass workflow working unchanged.

When at least one of the displayed passes carries the dynamic-energy
fields (i.e. its parent batch was run with the idle-baseline phase
enabled), three extra rows are appended below the totals — Dynamic
Energy / Power / Efficiency — so the user can read both the total
and the baseline-subtracted numbers without juggling toggles.
"""
from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication, QLabel

from ui.custom_widgets.batch_reports.comparison_views.energy_view._helpers import (
    get_nested_value,
)


# Row layout (rows): 0 = baseline banner, 1 = test selector,
# 2 = column headers, 3..5 = total rows, 6..8 = dynamic rows
# (created on demand). Keeping the dynamic rows starting at row 6
# means the totals stay anchored at known indices for callers that
# read ``_summary_rows`` directly (e.g. the clipboard exporter).
_TOTAL_ROWS = [
    (3, "Energy/image:",  "energy",     "mJ",    "font-weight: bold; color: #FF9800;"),
    (4, "Avg Power:",     "power",      "W",     "font-weight: bold; color: #4CAF50;"),
    (5, "Efficiency:",    "efficiency", "img/J", "font-weight: bold; color: #2196F3;"),
]
_DYNAMIC_ROWS = [
    (6, "Dynamic E/image:", "dyn_energy",     "mJ",    "color: #FF9800; font-style: italic;"),
    (7, "Dynamic Power:",   "dyn_power",      "W",     "color: #4CAF50; font-style: italic;"),
    (8, "Dynamic Eff.:",    "dyn_efficiency", "img/J", "color: #2196F3; font-style: italic;"),
]


def create_summary_table_structure(view):
    """Create the initial summary table structure with row labels."""
    header_font = QFont()
    header_font.setBold(True)

    # Track which rows we set up so other helpers can iterate without
    # having to know the exact row indices.
    view._summary_rows = list(_TOTAL_ROWS)
    view._dynamic_summary_rows = list(_DYNAMIC_ROWS)
    view._dynamic_row_label_widgets: list[QLabel] = []

    # Column 0: Metric names header (row 2 — see layout note above)
    metric_header = QLabel("Metric")
    metric_header.setFont(header_font)
    metric_header.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    view.summary_layout.addWidget(metric_header, 2, 0)

    for row_idx, label_text, _, _, _ in view._summary_rows:
        row_label = QLabel(label_text)
        row_label.setFont(header_font)
        view.summary_layout.addWidget(row_label, row_idx, 0)
        view._summary_row_labels.append(row_label)

    # Pre-create the dynamic row labels but keep them hidden; we'll
    # show them in update_summary_table when the matched tests carry
    # dynamic_* fields. Pre-creating means no widget churn between
    # refreshes — only the value cells are rebuilt.
    for row_idx, label_text, _, _, _ in view._dynamic_summary_rows:
        row_label = QLabel(label_text)
        row_label.setFont(header_font)
        row_label.setStyleSheet("color: #555; font-style: italic;")
        row_label.hide()
        view.summary_layout.addWidget(row_label, row_idx, 0)
        view._dynamic_row_label_widgets.append(row_label)


def clear_summary_backend_columns(view):
    """Remove all backend columns from the summary table."""
    for label in view._summary_header_labels:
        view.summary_layout.removeWidget(label)
        label.deleteLater()
    view._summary_header_labels.clear()

    for backend_labels in view._summary_backend_labels.values():
        for label in backend_labels.values():
            view.summary_layout.removeWidget(label)
            label.deleteLater()
    view._summary_backend_labels.clear()


def _selected_combo(view) -> tuple[Optional[str], Optional[str]]:
    """Return ``(name, exp_name)`` from the combo's userData, or
    ``(None, None)`` if nothing is selected. The combo is populated
    from ``set_tests`` with ``userData=(name, exp)`` per entry."""
    idx = view.test_combo.currentIndex()
    if idx < 0:
        return None, None
    data = view.test_combo.itemData(idx)
    if isinstance(data, tuple) and len(data) == 2:
        return data
    # Older combo entries (before the dedup work) had no userData;
    # fall back to using the visible text as the name.
    return view.test_combo.itemText(idx), None


def _find_test(tests, name, exp, *, use_gpu) -> Optional[dict]:
    """Look up a test entry that matches ``name``, optionally also
    matching ``exp`` (when the combo entry was experiment-qualified
    because of a name collision), and the requested ``use_gpu``
    flag."""
    for t in tests:
        if t.get("name") != name:
            continue
        if exp and (t.get("_experiment_name", "") or "") != exp:
            continue
        if bool(t.get("use_gpu")) != bool(use_gpu):
            continue
        return t
    return None


def _add_value_column(view, col_idx, header_text, header_color,
                      total_test, dynamic_test):
    """Render one full column (header + 3 total rows + 3 dynamic rows
    when applicable) for a given test.

    ``total_test`` carries the headline numbers; ``dynamic_test`` is
    usually the same dict but kept separate so we can fall back to a
    paired test in the future without touching the totals.
    """
    header_font = QFont()
    header_font.setBold(True)

    header_label = QLabel(header_text)
    header_label.setFont(header_font)
    header_label.setAlignment(Qt.AlignCenter)
    if header_color:
        header_label.setStyleSheet(f"color: {header_color};")
    view.summary_layout.addWidget(header_label, 2, col_idx)
    view._summary_header_labels.append(header_label)

    backend_labels: dict[str, QLabel] = {}

    # Totals — always populated (the test must have at least the
    # combined energy / power for it to even reach this function).
    e_mj = get_nested_value(total_test, "energy_mean_mj")
    if e_mj is None:
        e_mj = view._get_energy_value_combined(total_test)
    p_w = get_nested_value(total_test, "energy_mean_watts")
    if p_w is None:
        p_w = view._get_power_value_combined(total_test)

    for row_idx, _, metric_key, _, value_style in view._summary_rows:
        value_label = QLabel("-")
        value_label.setAlignment(Qt.AlignCenter)
        if value_style:
            value_label.setStyleSheet(value_style)
        view.summary_layout.addWidget(value_label, row_idx, col_idx)
        backend_labels[metric_key] = value_label

    if e_mj is not None:
        backend_labels["energy"].setText(f"{e_mj:.2f}")
    if p_w is not None:
        backend_labels["power"].setText(f"{p_w:.2f}")
    if e_mj is not None and e_mj > 0:
        backend_labels["efficiency"].setText(f"{1000.0 / e_mj:.1f}")

    # Dynamic rows — only populated when the test carries the
    # post-baseline triple. ``None`` values in the underlying data
    # render as ``-`` in the cell so a partially-populated batch
    # still reads cleanly.
    for row_idx, _, metric_key, _, value_style in view._dynamic_summary_rows:
        value_label = QLabel("-")
        value_label.setAlignment(Qt.AlignCenter)
        if value_style:
            value_label.setStyleSheet(value_style)
        view.summary_layout.addWidget(value_label, row_idx, col_idx)
        backend_labels[metric_key] = value_label

    dyn_e = get_nested_value(dynamic_test, "dynamic_energy_mj")
    dyn_p = get_nested_value(dynamic_test, "dynamic_power_W")
    dyn_eff = get_nested_value(dynamic_test, "dynamic_efficiency_imgs_per_J")
    if dyn_e is not None:
        backend_labels["dyn_energy"].setText(f"{dyn_e:.2f}")
    if dyn_p is not None:
        backend_labels["dyn_power"].setText(f"{dyn_p:.2f}")
    if dyn_eff is not None:
        backend_labels["dyn_efficiency"].setText(f"{dyn_eff:.1f}")

    return backend_labels


def update_summary_table(view):
    """Update the summary table with energy values for the selected test.

    Bucketing rule: the combo is keyed on test name (with experiment
    disambiguation when the same name appears in multiple loaded
    experiments). For the selected name we look up the matching CPU
    pass (``use_gpu=False``) and GPU pass (``use_gpu=True``) entries
    and render one column per pass. Single-pass loads collapse to one
    column.
    """
    clear_summary_backend_columns(view)

    name, exp = _selected_combo(view)
    if not view._tests or name is None:
        # Hide dynamic row labels until something is selected
        for lbl in view._dynamic_row_label_widgets:
            lbl.hide()
        return

    cpu_test = _find_test(view._tests, name, exp, use_gpu=False)
    gpu_test = _find_test(view._tests, name, exp, use_gpu=True)

    # Compose the column list. Tradition: CPU first then GPU so the
    # eye reads "left = baseline workload, right = accelerated".
    columns: list[tuple[str, str, dict]] = []
    if cpu_test is not None:
        columns.append(("CPU run", "#2196F3", cpu_test))
    if gpu_test is not None:
        columns.append(("GPU run", "#FF9800", gpu_test))

    # Single-pass fallback: combo selected a name that matches a test
    # but neither a CPU nor a GPU pass — possible when the loaded
    # report doesn't carry use_gpu (legacy data) or uses some other
    # value. Render that one entry as a single column labelled by its
    # actual flag.
    if not columns:
        for t in view._tests:
            if t.get("name") != name:
                continue
            if exp and (t.get("_experiment_name", "") or "") != exp:
                continue
            label = "GPU run" if t.get("use_gpu") else "CPU run"
            color = "#FF9800" if t.get("use_gpu") else "#2196F3"
            columns.append((label, color, t))
            break

    if not columns:
        for lbl in view._dynamic_row_label_widgets:
            lbl.hide()
        return

    # Show / hide the dynamic row labels based on whether *any*
    # matched test carries dynamic data — saves us pre-emptily
    # showing "Dynamic Power: -" rows on legacy reports.
    show_dynamic = any(
        t.get("dynamic_power_W") is not None for _, _, t in columns
    )
    for lbl in view._dynamic_row_label_widgets:
        lbl.setVisible(show_dynamic)

    col_idx = 1
    for header_text, header_color, t in columns:
        backend_labels = _add_value_column(
            view, col_idx, header_text, header_color, t, t,
        )
        view._summary_backend_labels[header_text] = backend_labels
        col_idx += 1

    # Unit column — uses "Metric" font weight at the header to match
    # the rest of the headers. Values themselves are static labels.
    header_font = QFont()
    header_font.setBold(True)
    unit_header = QLabel("Unit")
    unit_header.setFont(header_font)
    unit_header.setAlignment(Qt.AlignCenter)
    view.summary_layout.addWidget(unit_header, 2, col_idx)
    view._summary_header_labels.append(unit_header)

    rows_for_units = list(view._summary_rows)
    if show_dynamic:
        rows_for_units += list(view._dynamic_summary_rows)
    for row_idx, _, _, unit, _ in rows_for_units:
        unit_label = QLabel(unit)
        unit_label.setAlignment(Qt.AlignCenter)
        unit_label.setStyleSheet("color: #666;")
        view.summary_layout.addWidget(unit_label, row_idx, col_idx)
        view._summary_header_labels.append(unit_label)


def copy_summary_table(view):
    """Copy the summary table to clipboard as tab-separated values."""
    lines = []

    headers = ["Metric"]
    for backend_name in view._summary_backend_labels:
        headers.append(backend_name)
    headers.append("Unit")
    lines.append("\t".join(headers))

    rows = list(view._summary_rows)
    # Include dynamic rows in the clipboard only when the relevant
    # labels are visible — i.e. when they had data to populate.
    if view._dynamic_row_label_widgets and \
            view._dynamic_row_label_widgets[0].isVisible():
        rows += list(view._dynamic_summary_rows)

    for _row_idx, label_text, metric_key, unit, _ in rows:
        row_data = [label_text]
        for _backend_name, labels in view._summary_backend_labels.items():
            cell = labels.get(metric_key)
            row_data.append(cell.text() if cell is not None else "-")
        row_data.append(unit)
        lines.append("\t".join(row_data))

    text = "\n".join(lines)
    QApplication.clipboard().setText(text)
    view.logger.info("Summary table copied to clipboard")
