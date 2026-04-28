"""Energy view for Batch Reports — displays energy metrics charts."""
from __future__ import annotations

import logging
from typing import Any, Optional

import matplotlib
matplotlib.use('Qt5Agg')
from PyQt5.QtWidgets import (
    QDialog,
    QFileDialog,
    QMenu,
    QMessageBox,
    QWidget,
)

from ui.custom_widgets.batch_reports.comparison_views.chart_config_popup import (
    ChartConfigPopup,
)
from ui.custom_widgets.batch_reports.comparison_views.energy_view import _charts as charts
from ui.custom_widgets.batch_reports.comparison_views.energy_view._helpers import (
    draw_no_data_message,
    get_nested_value,
)
from ui.custom_widgets.batch_reports.comparison_views.energy_view._summary_table import (
    clear_summary_backend_columns,
    copy_summary_table,
    update_summary_table,
)
from ui.custom_widgets.batch_reports.comparison_views.energy_view._ui_builder import build_ui


class EnergyView(QWidget):
    """
    Energy view displaying energy consumption comparison charts.

    Features:
    - Left menu with chart type selection (QListWidget)
    - Backend selector (CPU, GPU, CPU+GPU with separate bars)
    - Shows all tests from Summary selection
    - Bar chart comparing energy consumption
    - Power vs efficiency scatter
    - Energy Summary table with columns per backend (like Single Test)
    - Navigation toolbar with chart configuration
    - Generate Energy Report button
    """

    # Color palette for experiments
    COLORS = ['#FF5722', '#E91E63', '#9C27B0', '#673AB7', '#3F51B5',
              '#2196F3', '#00BCD4', '#009688', '#4CAF50', '#8BC34A']

    # Backend-specific colors
    COLOR_GPU = '#FF9800'  # Orange for GPU
    COLOR_CPU = '#2196F3'  # Blue for CPU

    # Compute-path filter options. The constant names keep the old
    # ``BACKEND_*`` spelling so existing call sites don't churn, but the
    # display labels — and the semantics — are about which compute path
    # the test ran on (use_gpu=False vs use_gpu=True), not about which
    # energy backend produced the reading. On Jetson the rail is shared
    # and the per-rail breakdown is meaningless; "compute path" is the
    # comparison that actually matters.
    BACKEND_ALL = "CPU run + GPU run"
    BACKEND_CPU = "CPU run only"
    BACKEND_GPU = "GPU run only"

    def __init__(self, logger=None, parent=None):
        super().__init__(parent)

        if logger:
            self.logger = logger.getChild("EnergyView")
        else:
            self.logger = logging.getLogger("EnergyView")

        self._tests: list[dict[str, Any]] = []
        self._backend_filter = self.BACKEND_ALL
        # When True, the energy/power charts substitute the dynamic
        # equivalents (total − idle baseline) for the totals. Driven by
        # the "Subtract idle baseline" checkbox, which is only enabled
        # when at least one loaded experiment carries a baseline.
        self._subtract_baseline = False

        # Chart configuration with defaults
        self._chart_config = {
            'axes': {
                'title': '',
                'title_fontsize': 13,
                'xlabel': '',
                'xlabel_fontsize': 11,
                'xtick_fontsize': 9,
                'ylabel': '',
                'ylabel_fontsize': 11,
                'ytick_fontsize': 9,
                'auto_scale': True,
                'ymin': 0.0,
                'ymax': 100.0,
            },
            'legend': {
                'position': 0,
                'fontsize': 9,
                'frameon': True,
                'shadow': False,
                'fancybox': True,
                'framealpha': 0.8,
                'ncol': 1,
            },
            'colors': {
                'bar_alpha': 0.8,
            }
        }

        build_ui(self)

    # ----- Event handlers ---------------------------------------------------

    def _on_test_changed(self, index: int):
        """Handle test selection change for summary table."""
        update_summary_table(self)

    def _show_summary_copy_menu(self, pos):
        """Show context menu for copying summary table."""
        menu = QMenu(self)
        copy_action = menu.addAction("Copy table")
        action = menu.exec_(self.summary_group.mapToGlobal(pos))

        if action == copy_action:
            copy_summary_table(self)

    def _on_open_chart_config(self):
        """Open chart configuration dialog."""
        popup = ChartConfigPopup(parent=self, logger=self.logger)
        popup.set_config(self._chart_config)

        if popup.exec_() == QDialog.Accepted:
            self._chart_config = popup.get_config()
            self.logger.debug("Chart config updated: %s", self._chart_config)
            self._refresh_chart()

    def _on_backend_changed(self, backend: str):
        """Handle backend selection change."""
        self._backend_filter = backend
        self._refresh_chart()

    def _on_baseline_toggle(self, checked: bool):
        """Toggle handler for "Subtract idle baseline"."""
        self._subtract_baseline = bool(checked)
        self._refresh_chart()

    def _any_test_has_baseline_dynamics(self) -> bool:
        """True iff any loaded test carries a non-None ``dynamic_power_W``
        — the cue we use to enable / disable the toggle."""
        for t in self._tests:
            if t.get("dynamic_power_W") is not None:
                return True
        return False

    def _refresh_baseline_banner(self) -> None:
        """Update the "Idle baseline: …" line at the top of the
        summary group from the loaded experiments' metadata.

        One experiment with a baseline → single line with mean ± std,
        duration and backend label. Multiple experiments with
        different baselines → one line each, prefixed by the
        experiment name so the user notices when reports came from
        runs against different idle pedestals. No baseline anywhere →
        muted "(no idle baseline captured)".
        """
        # Walk loaded tests, dedup by experiment name so we don't
        # repeat the same metadata 5 times for a 5-test batch.
        seen_experiments: dict[str, dict] = {}
        for t in self._tests:
            exp_name = t.get("_experiment_name", "")
            if exp_name in seen_experiments:
                continue
            meta = t.get("_experiment_metadata") or {}
            baseline = meta.get("idle_baseline") if isinstance(meta, dict) else None
            seen_experiments[exp_name] = baseline

        baselines = [(name, b) for name, b in seen_experiments.items()
                     if b is not None]
        if not baselines:
            if not self._tests:
                self.baseline_label.setText(
                    "Idle baseline: (no experiments loaded)")
            else:
                self.baseline_label.setText(
                    "<i>Idle baseline: not captured for any loaded experiment.</i>"
                )
            return

        def _fmt_one(b: dict) -> str:
            mean = b.get("total_power_W", 0.0)
            std  = b.get("total_power_std_W", 0.0)
            dur  = b.get("duration_s", 0.0)
            # Backend label — single backend on Jetson, list when more.
            per = b.get("per_backend") or []
            backend_lbl = (per[0].get("backend") if len(per) == 1
                           else f"{len(per)} backends")
            return (f"<b>{mean:.2f} W</b> ± {std:.2f} W "
                    f"({dur:.0f}s · {backend_lbl})")

        if len(baselines) == 1:
            self.baseline_label.setText(
                f"Idle baseline: {_fmt_one(baselines[0][1])}"
            )
        else:
            parts = [f"<b>{name or '(unnamed)'}</b>: {_fmt_one(b)}"
                     for name, b in baselines]
            self.baseline_label.setText(
                "Idle baseline — " + " &nbsp;|&nbsp; ".join(parts)
            )

    def _on_generate_report(self):
        """Generate and show the energy report popup."""
        if not self._tests:
            self.logger.warning("No data available for report generation")
            return

        from ui.custom_widgets.batch_reports.comparison_views.batch_energy_report_popup import (
            BatchEnergyReportPopup,
        )

        popup = BatchEnergyReportPopup(parent=self, logger=self.logger)
        popup.set_data(self._tests)
        popup.exec_()

        self.logger.info("Batch energy report displayed")

    def _on_chart_type_changed(self, index: int):
        """Handle chart type selection change."""
        self._refresh_chart()

    def _show_context_menu(self, pos):
        """Show context menu for saving the chart."""
        menu = QMenu(self)
        save_action = menu.addAction("Save chart as...")
        save_action.triggered.connect(self._on_save_chart)
        menu.exec_(self.canvas.mapToGlobal(pos))

    def _on_save_chart(self):
        """Save chart to file."""
        if not self._tests:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Chart",
            "energy_chart.png",
            "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg);;All Files (*.*)"
        )

        if not file_path:
            return

        try:
            self.figure.savefig(file_path, dpi=150, bbox_inches='tight',
                                facecolor='white', edgecolor='none')
            self.logger.info("Saved chart to %s", file_path)
            QMessageBox.information(
                self, "Save Complete",
                f"Chart saved to:\n{file_path}"
            )
        except Exception as e:
            self.logger.error("Failed to save chart: %s", e)
            QMessageBox.warning(self, "Save Error", f"Failed to save chart:\n{e}")

    # ----- Public API -------------------------------------------------------

    def set_tests(self, tests: list[dict[str, Any]]):
        """Set the tests to display in the charts."""
        self._tests = tests

        # Update test combo for summary table (like Timing view)
        self.test_combo.clear()
        for test in tests:
            test_name = test.get("name", "Unknown")
            exp_name = test.get("_experiment_name", "")
            if exp_name:
                self.test_combo.addItem(f"{test_name} ({exp_name})")
            else:
                self.test_combo.addItem(test_name)

        # Enable the baseline toggle only when at least one loaded
        # test carries a dynamic value. If the toggle was on but no
        # baseline survives the new dataset, flip it back off so the
        # chart doesn't quietly draw zeros.
        has_dynamics = self._any_test_has_baseline_dynamics()
        self.baseline_check.setEnabled(has_dynamics)
        if not has_dynamics and self._subtract_baseline:
            self.baseline_check.setChecked(False)
            self._subtract_baseline = False

        self._refresh_baseline_banner()
        update_summary_table(self)
        self._refresh_chart()

    def clear(self):
        """Clear all data from the view."""
        self._tests = []
        self.test_combo.clear()
        self.figure.clear()
        self.canvas.draw()
        clear_summary_backend_columns(self)
        self.info_label.setText("Load experiments to see energy analysis")

    # ----- Data getters (used by chart and summary modules) -----------------

    def _has_energy_data(self, test: dict) -> bool:
        """Check if a test has energy data."""
        return (
            get_nested_value(test, "energy_mean_mj") is not None or
            get_nested_value(test, "mean_energy_mj") is not None or
            get_nested_value(test, "energy_gpu_mj") is not None or
            get_nested_value(test, "energy_cpu_mj") is not None
        )

    def _get_energy_value_combined(self, test: dict) -> Optional[float]:
        """Get combined energy value in mJ from a test."""
        for key in ["energy_mean_mj", "mean_energy_mj"]:
            val = get_nested_value(test, key)
            if val is not None:
                return val
        return None

    def _get_power_value_combined(self, test: dict) -> Optional[float]:
        """Get combined power value in Watts from a test."""
        for key in ["energy_mean_watts", "mean_power_watts", "power_mean_watts"]:
            val = get_nested_value(test, key)
            if val is not None:
                return val
        return None

    def _get_efficiency_value(self, test: dict) -> Optional[float]:
        """Get efficiency value (images/J) from a test."""
        for key in ["efficiency_images_per_joule", "energy_efficiency"]:
            val = get_nested_value(test, key)
            if val is not None:
                return val
        energy_mj = self._get_energy_value_combined(test)
        if energy_mj is not None and energy_mj > 0:
            return 1000.0 / energy_mj
        return None

    # ----- Chart dispatch ---------------------------------------------------

    def _refresh_chart(self):
        """Refresh the chart based on current settings."""
        self.figure.clear()

        if not self._tests:
            draw_no_data_message(self, "No data to display\nLoad experiments first")
            return

        tests_with_energy = [t for t in self._tests if self._has_energy_data(t)]
        if not tests_with_energy:
            draw_no_data_message(
                self,
                "No energy data available\n\n"
                "Energy measurement was not enabled for these tests.\n"
                "Enable 'energy' in report types when running batch tests."
            )
            return

        chart_type = self.chart_list.currentRow()

        if chart_type == 0:
            charts.draw_energy_bar(self)
        elif chart_type == 1:
            charts.draw_power_comparison(self)
        elif chart_type == 2:
            charts.draw_efficiency_chart(self)
        else:
            charts.draw_statistics_chart(self)

        legend_pos = self._chart_config['legend']['position']
        if legend_pos == 4:
            self.figure.tight_layout(rect=[0, 0.05, 0.85, 1])
        elif legend_pos == 5:
            self.figure.tight_layout(rect=[0, 0.15, 1, 1])
        else:
            self.figure.tight_layout(rect=[0, 0.05, 1, 1])

        self.canvas.draw()
        self.report_btn.setEnabled(True)

        experiment_count = len(set(t.get("_experiment_name", "") for t in tests_with_energy))
        backend_info = f" ({self._backend_filter})" if self._backend_filter != self.BACKEND_ALL else ""
        self.info_label.setText(
            f"Showing {len(tests_with_energy)} tests with energy data{backend_info} from {experiment_count} experiment(s)"
        )
