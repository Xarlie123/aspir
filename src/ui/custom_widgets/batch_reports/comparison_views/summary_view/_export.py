"""CSV export and clipboard copy for :class:`SummaryView`."""
from __future__ import annotations

import csv

from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox

from ui.custom_widgets.batch_reports.comparison_views.summary_view._table_logic import (
    get_nested_value,
)


def export_csv(view):
    """Export the table data to CSV."""
    if not view._tests:
        return

    file_path, _ = QFileDialog.getSaveFileName(
        view,
        "Export to CSV",
        "batch_comparison.csv",
        "CSV Files (*.csv);;All Files (*.*)"
    )

    if not file_path:
        return

    try:
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Write header (skip checkbox column)
            writer.writerow([col[0] for col in view.COLUMNS if col[1] != "_checkbox"])

            # Write data rows (only selected tests, in current order)
            tests_to_export = view.get_selected_tests() if view._selected_indices else view._tests
            for test in tests_to_export:
                row = []
                for _header_text, key, _format_func, _higher_is_better, _visible in view.COLUMNS:
                    if key == "_checkbox":
                        continue
                    value = get_nested_value(test, key)
                    if value is not None:
                        row.append(value)
                    else:
                        row.append("")
                writer.writerow(row)

        view.logger.info("Exported %d tests to %s", len(tests_to_export), file_path)
        QMessageBox.information(
            view, "Export Complete",
            f"Successfully exported {len(tests_to_export)} tests to:\n{file_path}"
        )

    except Exception as e:
        view.logger.error("Failed to export CSV: %s", e)
        QMessageBox.warning(view, "Export Error", f"Failed to export CSV:\n{e}")


def copy_to_clipboard(view):
    """Copy the table data to clipboard as tab-separated values."""
    if not view._tests:
        return

    try:
        lines = []

        # Build header (only visible columns, skip checkbox)
        headers = []
        for col_idx, (header, key, _, _, _) in enumerate(view.COLUMNS):
            if key == "_checkbox":
                continue
            if col_idx in view._visible_columns:
                headers.append(header if header else key)
        lines.append("\t".join(headers))

        # Build data rows (only selected tests, in current order)
        tests_to_copy = view.get_selected_tests() if view._selected_indices else view._tests
        for test in tests_to_copy:
            row = []
            for col_idx, (_header, key, format_func, _, _) in enumerate(view.COLUMNS):
                if key == "_checkbox":
                    continue
                if col_idx not in view._visible_columns:
                    continue
                value = get_nested_value(test, key)
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

        view.logger.info("Copied %d tests to clipboard", len(tests_to_copy))
        QMessageBox.information(
            view, "Copied",
            f"Copied {len(tests_to_copy)} tests to clipboard"
        )

    except Exception as e:
        view.logger.error("Failed to copy to clipboard: %s", e)
        QMessageBox.warning(view, "Copy Error", f"Failed to copy to clipboard:\n{e}")
