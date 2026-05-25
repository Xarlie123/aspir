"""Profiler-tab chart drawing functions for :class:`BatchTimingReportPopup`.

All functions take the popup as first argument — they read the current test
data, the device combo, and draw on the matplotlib figures attached to the
popup.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from PySide6.QtWidgets import QTableWidgetItem


def update_profiler_display(popup):
    """Update the profiler tab display."""
    test = popup._tests[popup._current_test_idx]

    # Check if profiler data is available in test results
    profiler_data = test.get("profiler_results", None)

    if not profiler_data:
        # Show info message and clear charts
        popup.profiler_device_combo.clear()
        popup.profiler_device_combo.setEnabled(False)
        show_profiler_unavailable_message(popup)
        return

    # Determine available devices and update combo box
    popup.profiler_device_combo.blockSignals(True)
    popup.profiler_device_combo.clear()

    # New format: {"cpu": {...}, "gpu": {...}}
    # Old format: {"device": "cuda", ...} (single dict)
    if "cpu" in profiler_data or "gpu" in profiler_data:
        # New format with separate CPU/GPU data
        popup._profiler_data_format = "new"
        available_devices = []
        if "cpu" in profiler_data:
            available_devices.append(("CPU", "cpu"))
        if "gpu" in profiler_data:
            available_devices.append(("GPU (CUDA)", "gpu"))

        for label, key in available_devices:
            popup.profiler_device_combo.addItem(label, key)

        # Default to GPU if available, otherwise CPU
        if "gpu" in profiler_data:
            gpu_index = next((i for i, (_, k) in enumerate(available_devices) if k == "gpu"), 0)
            popup.profiler_device_combo.setCurrentIndex(gpu_index)
        else:
            popup.profiler_device_combo.setCurrentIndex(0)
    else:
        # Old format: single profiler results dict
        popup._profiler_data_format = "old"
        device = profiler_data.get("device", "cpu")
        label = "GPU (CUDA)" if device == "cuda" else "CPU"
        popup.profiler_device_combo.addItem(label, "legacy")

    popup.profiler_device_combo.setEnabled(popup.profiler_device_combo.count() > 1)
    popup.profiler_device_combo.blockSignals(False)

    # Update charts for selected device
    popup.profiler_info_label.hide()
    update_profiler_charts_for_selected_device(popup)


def update_profiler_charts_for_selected_device(popup):
    """Update profiler charts based on selected device."""
    test = popup._tests[popup._current_test_idx]
    profiler_data = test.get("profiler_results", None)

    if not profiler_data:
        return

    # Get the selected device's data
    selected_key = popup.profiler_device_combo.currentData()

    if selected_key == "legacy":
        # Old format: use the entire profiler_data dict
        device_data = profiler_data
    else:
        # New format: get the specific device's data
        device_data = profiler_data.get(selected_key, {})

    if device_data:
        update_profiler_charts(popup, device_data)
    else:
        show_profiler_unavailable_message(popup)


def show_profiler_unavailable_message(popup):
    """Show message when profiler data is not available."""
    test = popup._tests[popup._current_test_idx]
    test_name = test.get("name", "Unknown")
    model_name = test.get("model_name", test.get("config", {}).get("model_name", "Unknown"))

    popup.profiler_info_label.setText(
        f"<b>PyTorch Profiler data not available for this test.</b><br><br>"
        f"<b>Test:</b> {test_name}<br>"
        f"<b>Model:</b> {model_name}<br><br>"
        f"To generate profiler data, use the <b>Profile DNN Inference</b> option "
        f"in Single Test → Reports → Timing Analysis before running batch tests.<br><br>"
        f"Alternatively, if the model is saved, you can run profiling from the main timing page "
        f"using the <b>Launch profile with Nsight</b> button."
    )
    popup.profiler_info_label.show()

    # Clear profiler charts
    popup.profiler_bar_figure.clear()
    ax = popup.profiler_bar_figure.add_subplot(111)
    ax.text(0.5, 0.5, "No profiler data available", ha='center', va='center',
            transform=ax.transAxes, fontsize=11, color='#999')
    ax.axis('off')
    popup.profiler_bar_canvas.draw()

    popup.profiler_pie_figure.clear()
    ax = popup.profiler_pie_figure.add_subplot(111)
    ax.text(0.5, 0.5, "No profiler data available", ha='center', va='center',
            transform=ax.transAxes, fontsize=11, color='#999')
    ax.axis('off')
    popup.profiler_pie_canvas.draw()

    popup.profiler_summary_text.setPlainText("Profiler data not available for this test.")
    popup.profiler_ops_table.setRowCount(0)


def update_profiler_charts(popup, profiler_data: dict[str, Any]):
    """Update profiler charts with available data."""
    popup.profiler_info_label.hide()

    # Update summary text
    summary = profiler_data.get('summary', 'No summary available')
    popup.profiler_summary_text.setPlainText(summary)

    # Update bottlenecks bar chart
    update_profiler_bar_chart(popup, profiler_data)

    # Update pie chart
    update_profiler_pie_chart(popup, profiler_data)

    # Update operations table
    update_profiler_table(popup, profiler_data)


def update_profiler_bar_chart(popup, profiler_data: dict[str, Any]):
    """Update profiler bottlenecks bar chart."""
    popup.profiler_bar_figure.clear()
    ax = popup.profiler_bar_figure.add_subplot(111)

    bottlenecks = profiler_data.get('bottlenecks', [])[:10]
    if not bottlenecks:
        ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
        popup.profiler_bar_canvas.draw()
        return

    device = profiler_data.get('device', 'cpu')

    names = []
    times = []
    for op in bottlenecks:
        name = op.get('name', 'Unknown')
        if len(name) > 25:
            name = name[:22] + "..."
        names.append(name)

        if device == 'cuda' and op.get('cuda_time_ms', 0) > 0:
            times.append(op['cuda_time_ms'])
        else:
            times.append(op.get('cpu_time_ms', 0))

    y_pos = np.arange(len(names))
    colors = ['#d7191c' if i == 0 else '#fdae61' if i < 3 else '#2b83ba' for i in range(len(names))]

    ax.barh(y_pos, times, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Time (ms)')
    ax.grid(True, alpha=0.3, axis='x')

    popup.profiler_bar_figure.tight_layout()
    popup.profiler_bar_canvas.draw()


def update_profiler_pie_chart(popup, profiler_data: dict[str, Any]):
    """Update profiler pie chart."""
    popup.profiler_pie_figure.clear()

    layer_breakdown = profiler_data.get('layer_breakdown', [])
    if not layer_breakdown:
        ax = popup.profiler_pie_figure.add_subplot(111)
        ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
        popup.profiler_pie_canvas.draw()
        return

    labels = []
    sizes = []
    for layer in layer_breakdown[:10]:
        if layer.get('total_time_ms', 0) > 0:
            labels.append(layer.get('category', 'Unknown'))
            sizes.append(layer['total_time_ms'])

    if not sizes:
        ax = popup.profiler_pie_figure.add_subplot(111)
        ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
        popup.profiler_pie_canvas.draw()
        return

    colors = [
        '#d7191c', '#fdae61', '#abdda4', '#2b83ba', '#9C27B0',
        '#607D8B', '#FF5722', '#00BCD4', '#8BC34A', '#795548'
    ]

    ax = popup.profiler_pie_figure.add_subplot(111)

    def autopct_func(pct):
        return f'{pct:.1f}%' if pct > 5 else ''

    wedges, _texts, autotexts = ax.pie(
        sizes,
        autopct=autopct_func,
        colors=colors[:len(sizes)],
        textprops={'fontsize': 8, 'weight': 'bold'},
        pctdistance=0.7
    )

    for autotext in autotexts:
        autotext.set_color('white')

    device = profiler_data.get('device', 'cpu')
    device_label = "GPU Kernel" if device == 'cuda' else "CPU"
    total_time = sum(sizes)
    ax.set_title(f'{device_label} Time by Operation\n({total_time:.1f} ms)',
                 fontsize=10, fontweight='bold', pad=5)

    legend_labels = [f"{label} ({size:.1f} ms)" for label, size in zip(labels, sizes)]
    ax.legend(
        wedges,
        legend_labels,
        title="Operation Type",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        fontsize=8,
        title_fontsize=9,
        frameon=False
    )

    popup.profiler_pie_figure.tight_layout()
    popup.profiler_pie_canvas.draw()


def update_profiler_table(popup, profiler_data: dict[str, Any]):
    """Update profiler operations table."""
    bottlenecks = profiler_data.get('bottlenecks', [])
    device = profiler_data.get('device', 'cpu')

    popup.profiler_ops_table.setRowCount(len(bottlenecks))

    for row, op in enumerate(bottlenecks):
        popup.profiler_ops_table.setItem(row, 0, QTableWidgetItem(op.get('name', '')))
        popup.profiler_ops_table.setItem(row, 1, QTableWidgetItem(f"{op.get('cpu_time_ms', 0):.3f}"))
        popup.profiler_ops_table.setItem(row, 2, QTableWidgetItem(
            f"{op.get('cuda_time_ms', 0):.3f}" if device == 'cuda' else "-"
        ))
        popup.profiler_ops_table.setItem(row, 3, QTableWidgetItem(str(op.get('calls', 0))))

        time_per_call = op.get('cuda_time_per_call_ms', 0) if device == 'cuda' else op.get('cpu_time_per_call_ms', 0)
        popup.profiler_ops_table.setItem(row, 4, QTableWidgetItem(f"{time_per_call:.3f}"))
