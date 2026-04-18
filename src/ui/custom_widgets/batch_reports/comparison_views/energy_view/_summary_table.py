"""Summary-table helpers — build, clear, refresh and copy the per-backend table."""
from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication, QLabel

from ui.custom_widgets.batch_reports.comparison_views.energy_view._helpers import (
    get_nested_value,
)


def create_summary_table_structure(view):
    """Create the initial summary table structure with row labels."""
    header_font = QFont()
    header_font.setBold(True)

    # Row definitions: (row_index, label_text, metric_key, unit, value_style)
    # Start from row 1 since row 0 has the test selector
    view._summary_rows = [
        (2, "Energy/image:", "energy", "mJ", "font-weight: bold; color: #FF9800;"),
        (3, "Avg Power:", "power", "W", "font-weight: bold; color: #4CAF50;"),
        (4, "Efficiency:", "efficiency", "img/J", "font-weight: bold; color: #2196F3;"),
    ]

    # Column 0: Metric names header (row 1)
    metric_header = QLabel("Metric")
    metric_header.setFont(header_font)
    metric_header.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    view.summary_layout.addWidget(metric_header, 1, 0)

    for row_idx, label_text, _, _, _ in view._summary_rows:
        row_label = QLabel(label_text)
        row_label.setFont(header_font)
        view.summary_layout.addWidget(row_label, row_idx, 0)
        view._summary_row_labels.append(row_label)


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


def update_summary_table(view):
    """Update the summary table with energy values for the selected test."""
    clear_summary_backend_columns(view)

    current_idx = view.test_combo.currentIndex()

    if not view._tests or current_idx < 0 or current_idx >= len(view._tests):
        return

    test = view._tests[current_idx]

    header_font = QFont()
    header_font.setBold(True)

    # Get energy data for this test
    gpu_e = get_nested_value(test, "energy_gpu_mj")
    gpu_p = get_nested_value(test, "energy_gpu_watts")
    cpu_e = get_nested_value(test, "energy_cpu_mj")
    cpu_p = get_nested_value(test, "energy_cpu_watts")

    has_gpu = gpu_e is not None and gpu_e > 0
    has_cpu = cpu_e is not None and cpu_e > 0

    # If no per-backend data, fall back to combined data
    if not has_gpu and not has_cpu:
        combined_e = view._get_energy_value_combined(test)
        combined_p = view._get_power_value_combined(test)

        if combined_e is not None:
            col_idx = 1
            header_label = QLabel("Combined")
            header_label.setFont(header_font)
            header_label.setAlignment(Qt.AlignCenter)
            view.summary_layout.addWidget(header_label, 1, col_idx)
            view._summary_header_labels.append(header_label)

            backend_labels = {}
            for row_idx, _, metric_key, _, value_style in view._summary_rows:
                value_label = QLabel("-")
                value_label.setAlignment(Qt.AlignCenter)
                if value_style:
                    value_label.setStyleSheet(value_style)
                view.summary_layout.addWidget(value_label, row_idx, col_idx)
                backend_labels[metric_key] = value_label

            backend_labels["energy"].setText(f"{combined_e:.2f}")
            if combined_p is not None:
                backend_labels["power"].setText(f"{combined_p:.2f}")
            efficiency = 1000.0 / combined_e if combined_e > 0 else 0
            backend_labels["efficiency"].setText(f"{efficiency:.1f}")

            view._summary_backend_labels["Combined"] = backend_labels
            col_idx += 1
    else:
        col_idx = 1

        # GPU column
        if has_gpu:
            header_label = QLabel("GPU")
            header_label.setFont(header_font)
            header_label.setAlignment(Qt.AlignCenter)
            header_label.setStyleSheet("color: #FF9800;")
            view.summary_layout.addWidget(header_label, 1, col_idx)
            view._summary_header_labels.append(header_label)

            backend_labels = {}
            for row_idx, _, metric_key, _, value_style in view._summary_rows:
                value_label = QLabel("-")
                value_label.setAlignment(Qt.AlignCenter)
                if value_style:
                    value_label.setStyleSheet(value_style)
                view.summary_layout.addWidget(value_label, row_idx, col_idx)
                backend_labels[metric_key] = value_label

            backend_labels["energy"].setText(f"{gpu_e:.2f}")
            if gpu_p is not None:
                backend_labels["power"].setText(f"{gpu_p:.2f}")
            efficiency = 1000.0 / gpu_e if gpu_e > 0 else 0
            backend_labels["efficiency"].setText(f"{efficiency:.1f}")

            view._summary_backend_labels["GPU"] = backend_labels
            col_idx += 1

        # CPU column
        if has_cpu:
            header_label = QLabel("CPU")
            header_label.setFont(header_font)
            header_label.setAlignment(Qt.AlignCenter)
            header_label.setStyleSheet("color: #2196F3;")
            view.summary_layout.addWidget(header_label, 1, col_idx)
            view._summary_header_labels.append(header_label)

            backend_labels = {}
            for row_idx, _, metric_key, _, value_style in view._summary_rows:
                value_label = QLabel("-")
                value_label.setAlignment(Qt.AlignCenter)
                if value_style:
                    value_label.setStyleSheet(value_style)
                view.summary_layout.addWidget(value_label, row_idx, col_idx)
                backend_labels[metric_key] = value_label

            backend_labels["energy"].setText(f"{cpu_e:.2f}")
            if cpu_p is not None:
                backend_labels["power"].setText(f"{cpu_p:.2f}")
            efficiency = 1000.0 / cpu_e if cpu_e > 0 else 0
            backend_labels["efficiency"].setText(f"{efficiency:.1f}")

            view._summary_backend_labels["CPU"] = backend_labels
            col_idx += 1

    # Add Unit column
    if col_idx > 1:
        unit_header = QLabel("Unit")
        unit_header.setFont(header_font)
        unit_header.setAlignment(Qt.AlignCenter)
        view.summary_layout.addWidget(unit_header, 1, col_idx)
        view._summary_header_labels.append(unit_header)

        for row_idx, _, _, unit, _ in view._summary_rows:
            unit_label = QLabel(unit)
            unit_label.setAlignment(Qt.AlignCenter)
            unit_label.setStyleSheet("color: #666;")
            view.summary_layout.addWidget(unit_label, row_idx, col_idx)
            view._summary_header_labels.append(unit_label)


def copy_summary_table(view):
    """Copy the summary table to clipboard as tab-separated values."""
    lines = []

    # Header row
    headers = ["Metric"]
    for backend_name in view._summary_backend_labels:
        headers.append(backend_name)
    headers.append("Unit")
    lines.append("\t".join(headers))

    # Data rows
    for _row_idx, label_text, metric_key, unit, _ in view._summary_rows:
        row_data = [label_text]
        for _backend_name, labels in view._summary_backend_labels.items():
            row_data.append(labels[metric_key].text())
        row_data.append(unit)
        lines.append("\t".join(row_data))

    text = "\n".join(lines)
    QApplication.clipboard().setText(text)
    view.logger.info("Summary table copied to clipboard")
