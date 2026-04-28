"""Chart-drawing functions for the energy view.

All functions take ``view`` (the :class:`EnergyView` instance) as first argument
so they can read ``view._tests``, ``view._backend_filter``, ``view.figure`` and
backend colour class attributes. They draw on ``view.figure`` and never
modify view state beyond the figure itself.
"""
from __future__ import annotations

import numpy as np

from ui.custom_widgets.batch_reports.comparison_views.energy_view._helpers import (
    apply_axes_config,
    apply_legend,
    get_nested_value,
)


def collect_compute_path_data(view, value_keys: list[str]):
    """Bucket test values by compute path (``use_gpu`` flag).

    For every distinct ``test["name"]`` in ``view._tests``, look for one
    entry that ran on the CPU (``use_gpu=False``) and one that ran on
    the GPU (``use_gpu=True``); pull each entry's value from the first
    key in ``value_keys`` that resolves to a number. The result is a
    pair of aligned lists ``(gpu_values, cpu_values)`` indexed by the
    deduplicated test names — exactly the shape the existing
    ``draw_grouped_bar_chart`` consumes, so no downstream change is
    needed.

    The ``BACKEND_*`` filter is applied here: in CPU-only mode we drop
    tests with no CPU-pass data, and conversely for GPU-only.

    Tests with the same name *and* the same ``use_gpu`` (e.g. the user
    re-measured twice) collapse into the first occurrence; the value
    from the second pass is silently ignored. Loading two complete
    passes — one with ``use_gpu=False`` and one with ``use_gpu=True``,
    which is exactly what "Run both compute paths" produces — fills
    every group.
    """
    def _value_for(test):
        for key in value_keys:
            val = get_nested_value(test, key)
            if val is not None:
                return val
        return None

    grouped: dict[str, dict[str, float]] = {}
    name_order: list[str] = []
    for test in view._tests:
        name = test.get("name", "Unknown")
        if name not in grouped:
            grouped[name] = {}
            name_order.append(name)
        bucket = "gpu" if bool(test.get("use_gpu")) else "cpu"
        if bucket in grouped[name]:
            continue  # keep first occurrence per (name, path)
        val = _value_for(test)
        if val is not None:
            grouped[name][bucket] = float(val)

    test_names: list[str] = []
    gpu_values: list[float] = []
    cpu_values: list[float] = []
    show_gpu = view._backend_filter in (view.BACKEND_ALL, view.BACKEND_GPU)
    show_cpu = view._backend_filter in (view.BACKEND_ALL, view.BACKEND_CPU)

    for name in name_order:
        entry = grouped[name]
        gpu_v = entry.get("gpu") if show_gpu else None
        cpu_v = entry.get("cpu") if show_cpu else None
        # Skip tests that contribute nothing under the active filter
        # (e.g. CPU-only filter applied to a GPU-only experiment).
        if gpu_v is None and cpu_v is None:
            continue
        display = name if len(name) <= 15 else name[:12] + "..."
        test_names.append(display)
        gpu_values.append(gpu_v if gpu_v is not None else 0)
        cpu_values.append(cpu_v if cpu_v is not None else 0)

    has_gpu = any(v > 0 for v in gpu_values)
    has_cpu = any(v > 0 for v in cpu_values)
    return test_names, gpu_values, cpu_values, has_gpu, has_cpu


def collect_backend_data(view, gpu_key: str, cpu_key: str,
                         combined_keys: list | None = None):
    """Backwards-compatible wrapper.

    The old API took separate GPU / CPU field keys (e.g.
    ``energy_gpu_mj`` / ``energy_cpu_mj``) and read both from a single
    test entry. Under the new compute-path semantics both bars come
    from the same total field on different test entries; collapse the
    arguments to ``collect_compute_path_data`` so the call sites in
    this file don't need to change.

    When the view's "Subtract idle baseline" toggle is on, the
    energy / power lookups are swapped for their dynamic equivalents
    (``dynamic_energy_mj`` / ``dynamic_power_W``). Tests without a
    dynamic value drop out — the toggle is gated upstream so this
    only happens when at least one experiment carries a baseline.
    """
    # The "gpu_key" is conventionally the per-rail GPU field; we
    # promote the matching combined fields ("energy_mean_mj" /
    # "mean_energy_mj" / "energy_mean_watts" / "mean_power_watts") to
    # the primary lookup, and keep the per-rail keys + any legacy
    # ``combined_keys`` as fallbacks for old reports.
    primary_keys = {
        "energy_gpu_mj":   ["energy_mean_mj", "mean_energy_mj"],
        "energy_gpu_watts":["energy_mean_watts", "mean_power_watts"],
    }
    # Map total → dynamic for the baseline-subtracted view.
    dynamic_keys = {
        "energy_gpu_mj":    ["dynamic_energy_mj"],
        "energy_gpu_watts": ["dynamic_power_W"],
    }
    use_dynamic = bool(getattr(view, "_subtract_baseline", False))
    value_keys: list[str] = []
    if use_dynamic:
        for k in dynamic_keys.get(gpu_key, []):
            if k not in value_keys:
                value_keys.append(k)
    else:
        for k in primary_keys.get(gpu_key, []):
            if k not in value_keys:
                value_keys.append(k)
        for k in (combined_keys or []):
            if k not in value_keys:
                value_keys.append(k)
        # Per-rail keys remain as last-ditch fallbacks for ancient reports
        # that only stored ``energy_gpu_mj`` / ``energy_cpu_mj``.
        for k in (gpu_key, cpu_key):
            if k and k not in value_keys:
                value_keys.append(k)
    return collect_compute_path_data(view, value_keys)


def draw_grouped_bar_chart(view, ax, test_names, gpu_values, cpu_values,
                           has_gpu, has_cpu, value_format=".1f",
                           annotation_text=None):
    """
    Draw a grouped bar chart with CPU/GPU on X-axis and test names below.

    Layout: [GPU] [CPU]  gap  [GPU] [CPU]  gap  ...
              Test 1              Test 2
    """
    if not has_gpu and not has_cpu:
        ax.text(0.5, 0.5, "No valid data available",
                ha='center', va='center', fontsize=12, color='#999')
        ax.axis('off')
        return False

    # Build positions and data
    x_positions = []
    x_labels = []
    bar_values = []
    bar_colors = []
    group_positions = []  # (start_idx, end_idx, test_name)

    pos = 0.0
    bar_idx = 0
    gpu_first_idx = -1
    cpu_first_idx = -1

    for i, test_name in enumerate(test_names):
        group_start_idx = bar_idx

        # GPU run bar (test had use_gpu=True). Labels read "GPU run"
        # rather than just "GPU" so the user reads the chart as
        # "compute path = GPU" instead of "energy backend = GPU".
        if has_gpu:
            x_positions.append(pos)
            x_labels.append("GPU run")
            bar_values.append(gpu_values[i])
            bar_colors.append(view.COLOR_GPU)
            if gpu_first_idx < 0:
                gpu_first_idx = bar_idx
            pos += 1
            bar_idx += 1

        # CPU run bar (test had use_gpu=False)
        if has_cpu:
            x_positions.append(pos)
            x_labels.append("CPU run")
            bar_values.append(cpu_values[i])
            bar_colors.append(view.COLOR_CPU)
            if cpu_first_idx < 0:
                cpu_first_idx = bar_idx
            pos += 1
            bar_idx += 1

        group_end_idx = bar_idx - 1
        if group_end_idx >= group_start_idx:
            group_positions.append((group_start_idx, group_end_idx, test_name))

        pos += 0.5  # Gap between tests

    x = np.array(x_positions)
    width = 0.7

    # Draw bars
    for idx in range(len(x)):
        # Set label only for first occurrence of each color (for legend)
        label = None
        if bar_colors[idx] == view.COLOR_GPU and idx == gpu_first_idx:
            label = 'GPU run'
        elif bar_colors[idx] == view.COLOR_CPU and idx == cpu_first_idx:
            label = 'CPU run'

        ax.bar(x[idx], bar_values[idx], width, label=label,
               color=bar_colors[idx], alpha=0.8)

        # Value label on top
        if bar_values[idx] > 0:
            ax.text(x[idx], bar_values[idx], f'{bar_values[idx]:{value_format}}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=9)

    # Add test name labels below X-axis (below CPU/GPU labels)
    for group_start_idx, group_end_idx, test_name in group_positions:
        group_center = (x_positions[group_start_idx] + x_positions[group_end_idx]) / 2
        # Use axes fraction for y (-0.12 places it below the tick labels)
        ax.text(group_center, -0.12, test_name,
                ha='center', va='top', fontsize=9, fontweight='bold',
                transform=ax.get_xaxis_transform())

    # Adjust bottom margin for test names
    view.figure.subplots_adjust(bottom=0.22)

    ax.grid(axis='y', alpha=0.3)

    if annotation_text:
        ax.annotate(annotation_text, xy=(1, 1), xycoords='axes fraction',
                    fontsize=9, color='#666', ha='right', va='top')

    return True


def draw_single_backend_chart(view, ax, test_names, values, is_gpu: bool,
                              value_format=".1f", annotation_text=None):
    """Draw a chart for a single compute path with test names on X-axis."""
    if not any(v > 0 for v in values):
        path_name = "GPU run" if is_gpu else "CPU run"
        ax.text(0.5, 0.5, f"No {path_name} data available",
                ha='center', va='center', fontsize=12, color='#999')
        ax.axis('off')
        return False

    x = np.arange(len(test_names))
    width = 0.6
    color = view.COLOR_GPU if is_gpu else view.COLOR_CPU
    label = 'GPU run' if is_gpu else 'CPU run'

    bars = ax.bar(x, values, width, label=label, color=color, alpha=0.8)

    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:{value_format}}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(test_names, rotation=45, ha='right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    if annotation_text:
        ax.annotate(annotation_text, xy=(1, 1), xycoords='axes fraction',
                    fontsize=9, color='#666', ha='right', va='top')

    return True


def draw_energy_bar(view):
    """Draw energy bar chart with CPU/GPU on X-axis and test names below."""
    ax = view.figure.add_subplot(111)

    test_names, gpu_values, cpu_values, has_gpu, has_cpu = collect_backend_data(
        view, "energy_gpu_mj", "energy_cpu_mj", ["energy_mean_mj", "mean_energy_mj"]
    )

    if view._backend_filter == view.BACKEND_ALL:
        success = draw_grouped_bar_chart(
            view, ax, test_names, gpu_values, cpu_values, has_gpu, has_cpu,
            value_format=".1f", annotation_text="(lower is better)"
        )
        if success:
            apply_axes_config(view, ax, "Energy Consumption Comparison", "", "Energy (mJ)")
            apply_legend(view, ax)
    elif view._backend_filter == view.BACKEND_GPU:
        success = draw_single_backend_chart(
            view, ax, test_names, gpu_values, is_gpu=True,
            value_format=".1f", annotation_text="(lower is better)"
        )
        if success:
            apply_axes_config(view, ax, "Energy Consumption (GPU)", "Test", "Energy (mJ)")
            apply_legend(view, ax)
    else:  # CPU Only
        success = draw_single_backend_chart(
            view, ax, test_names, cpu_values, is_gpu=False,
            value_format=".1f", annotation_text="(lower is better)"
        )
        if success:
            apply_axes_config(view, ax, "Energy Consumption (CPU)", "Test", "Energy (mJ)")
            apply_legend(view, ax)


def draw_power_comparison(view):
    """Draw power comparison chart with CPU/GPU on X-axis and test names below."""
    ax = view.figure.add_subplot(111)

    test_names, gpu_values, cpu_values, has_gpu, has_cpu = collect_backend_data(
        view, "energy_gpu_watts", "energy_cpu_watts", ["energy_mean_watts", "mean_power_watts"]
    )

    if view._backend_filter == view.BACKEND_ALL:
        success = draw_grouped_bar_chart(
            view, ax, test_names, gpu_values, cpu_values, has_gpu, has_cpu,
            value_format=".1f", annotation_text=None
        )
        if success:
            apply_axes_config(view, ax, "Power Consumption Comparison", "", "Power (W)")
            apply_legend(view, ax)
    elif view._backend_filter == view.BACKEND_GPU:
        success = draw_single_backend_chart(
            view, ax, test_names, gpu_values, is_gpu=True,
            value_format=".1f", annotation_text=None
        )
        if success:
            apply_axes_config(view, ax, "Power Consumption (GPU)", "Test", "Power (W)")
            apply_legend(view, ax)
    else:  # CPU Only
        success = draw_single_backend_chart(
            view, ax, test_names, cpu_values, is_gpu=False,
            value_format=".1f", annotation_text=None
        )
        if success:
            apply_axes_config(view, ax, "Power Consumption (CPU)", "Test", "Power (W)")
            apply_legend(view, ax)


def _get_efficiency_per_backend(test: dict) -> tuple:
    """
    Calculate efficiency per backend (images per Joule).

    Returns:
        tuple: (gpu_efficiency, cpu_efficiency)
    """
    gpu_e = get_nested_value(test, "energy_gpu_mj")
    cpu_e = get_nested_value(test, "energy_cpu_mj")

    gpu_eff = 1000.0 / gpu_e if gpu_e and gpu_e > 0 else 0
    cpu_eff = 1000.0 / cpu_e if cpu_e and cpu_e > 0 else 0

    return gpu_eff, cpu_eff


def draw_efficiency_chart(view):
    """Draw efficiency chart with CPU/GPU on X-axis and test names below."""
    ax = view.figure.add_subplot(111)

    # Collect efficiency per backend
    test_names = []
    gpu_values = []
    cpu_values = []

    for test in view._tests:
        test_name = test.get("name", "Unknown")
        if len(test_name) > 15:
            test_name = test_name[:12] + "..."
        test_names.append(test_name)

        gpu_eff, cpu_eff = _get_efficiency_per_backend(test)

        # Fallback to combined efficiency if no per-backend data
        if gpu_eff == 0 and cpu_eff == 0:
            combined_eff = view._get_efficiency_value(test)
            gpu_values.append(combined_eff if combined_eff else 0)
            cpu_values.append(0)
        else:
            gpu_values.append(gpu_eff)
            cpu_values.append(cpu_eff)

    has_gpu = any(v > 0 for v in gpu_values)
    has_cpu = any(v > 0 for v in cpu_values)

    if view._backend_filter == view.BACKEND_ALL:
        success = draw_grouped_bar_chart(
            view, ax, test_names, gpu_values, cpu_values, has_gpu, has_cpu,
            value_format=".0f", annotation_text="(higher is better)"
        )
        if success:
            apply_axes_config(view, ax, "Energy Efficiency Comparison", "", "Efficiency (images/J)")
            apply_legend(view, ax)
    elif view._backend_filter == view.BACKEND_GPU:
        success = draw_single_backend_chart(
            view, ax, test_names, gpu_values, is_gpu=True,
            value_format=".0f", annotation_text="(higher is better)"
        )
        if success:
            apply_axes_config(view, ax, "Energy Efficiency (GPU)", "Test", "Efficiency (images/J)")
            apply_legend(view, ax)
    else:  # CPU Only
        success = draw_single_backend_chart(
            view, ax, test_names, cpu_values, is_gpu=False,
            value_format=".0f", annotation_text="(higher is better)"
        )
        if success:
            apply_axes_config(view, ax, "Energy Efficiency (CPU)", "Test", "Efficiency (images/J)")
            apply_legend(view, ax)


def draw_statistics_chart(view):
    """Draw box plot of energy distribution with CPU/GPU on X-axis and test names below."""
    ax = view.figure.add_subplot(111)

    # Collect energy data per backend per test
    test_names, gpu_values, cpu_values, has_gpu, has_cpu = collect_backend_data(
        view, "energy_gpu_mj", "energy_cpu_mj", ["energy_mean_mj", "mean_energy_mj"]
    )

    if not has_gpu and not has_cpu:
        ax.text(0.5, 0.5, "No valid energy data available",
                ha='center', va='center', fontsize=12, color='#999')
        ax.axis('off')
        return

    # Build data for box plot
    box_data = []
    box_labels = []
    box_colors = []
    group_positions = []  # (start_idx, end_idx, test_name)

    idx = 0
    for i, test_name in enumerate(test_names):
        group_start = idx

        if view._backend_filter == view.BACKEND_ALL:
            # Add GPU data point if available
            if has_gpu and gpu_values[i] > 0:
                box_data.append([gpu_values[i]])
                box_labels.append("GPU")
                box_colors.append(view.COLOR_GPU)
                idx += 1

            # Add CPU data point if available
            if has_cpu and cpu_values[i] > 0:
                box_data.append([cpu_values[i]])
                box_labels.append("CPU")
                box_colors.append(view.COLOR_CPU)
                idx += 1
        elif view._backend_filter == view.BACKEND_GPU:
            if gpu_values[i] > 0:
                box_data.append([gpu_values[i]])
                box_labels.append("GPU")
                box_colors.append(view.COLOR_GPU)
                idx += 1
        else:  # CPU Only
            if cpu_values[i] > 0:
                box_data.append([cpu_values[i]])
                box_labels.append("CPU")
                box_colors.append(view.COLOR_CPU)
                idx += 1

        group_end = idx - 1
        if group_end >= group_start:
            group_positions.append((group_start, group_end, test_name))

    if not box_data:
        ax.text(0.5, 0.5, "No valid energy data available",
                ha='center', va='center', fontsize=12, color='#999')
        ax.axis('off')
        return

    # Create box plot
    # Add gaps between test groups
    adjusted_positions = []
    pos = 0
    current_group = 0
    for i in range(len(box_data)):
        # Check if we're starting a new group
        if current_group < len(group_positions):
            start, end, _ = group_positions[current_group]
            if i > end:
                pos += 0.5  # Add gap
                current_group += 1
        adjusted_positions.append(pos)
        pos += 1

    bp = ax.boxplot(box_data, positions=adjusted_positions, patch_artist=True, widths=0.6)

    # Color the boxes
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticks(adjusted_positions)
    ax.set_xticklabels(box_labels, fontsize=9)

    # Add test name labels below (below CPU/GPU labels)
    for group_start, group_end, test_name in group_positions:
        if group_start < len(adjusted_positions) and group_end < len(adjusted_positions):
            group_center = (adjusted_positions[group_start] + adjusted_positions[group_end]) / 2
            # Use axes fraction for y (-0.12 places it below the tick labels)
            ax.text(group_center, -0.12, test_name,
                    ha='center', va='top', fontsize=9, fontweight='bold',
                    transform=ax.get_xaxis_transform())

    view.figure.subplots_adjust(bottom=0.22)
    ax.grid(axis='y', alpha=0.3)

    title_suffix = ""
    if view._backend_filter == view.BACKEND_GPU:
        title_suffix = " (GPU)"
    elif view._backend_filter == view.BACKEND_CPU:
        title_suffix = " (CPU)"

    apply_axes_config(view, ax, f"Energy Distribution{title_suffix}", "", "Energy (mJ)")
