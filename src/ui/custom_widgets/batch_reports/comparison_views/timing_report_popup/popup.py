"""Popup dialog for displaying detailed timing report in Batch Reports mode."""
from __future__ import annotations

import logging
from typing import Any

from PySide6.QtWidgets import QDialog, QFileDialog, QMenu, QMessageBox

from ui.custom_widgets.batch_reports.comparison_views.chart_config_popup import (
    ChartConfigPopup,
)
from ui.custom_widgets.batch_reports.comparison_views.timing_report_popup import (
    _profiler_charts as profiler_charts,
)
from ui.custom_widgets.batch_reports.comparison_views.timing_report_popup import (
    _timing_charts as timing_charts,
)
from ui.custom_widgets.batch_reports.comparison_views.timing_report_popup._ui_builder import (
    build_ui,
)


class BatchTimingReportPopup(QDialog):
    """
    Popup dialog showing detailed timing analysis report for batch tests.

    Features:
    - Tab 1: Timing Report with:
        - Time per image curves (acquisition, reconstruction, inference, total)
        - Time distribution histograms
        - Stacked bar chart (CPU vs GPU)
        - Detailed statistics table
    - Tab 2: PyTorch Profiler data (if available)
    - Test selector to switch between tests
    """

    # Pipeline colors (matching Single Test)
    COLOR_ACQUISITION = '#abdda4'   # Green
    COLOR_RECONSTRUCTION = '#fdae61'  # Orange
    COLOR_INFERENCE_CPU = '#d7191c'   # Red
    COLOR_INFERENCE_GPU = '#2b83ba'   # Blue

    def __init__(
        self,
        tests: list[dict[str, Any]],
        current_test_idx: int = 0,
        parent=None,
        logger=None
    ):
        super().__init__(parent)
        self.setWindowTitle("Detailed Timing Report")
        self.setMinimumSize(1100, 800)
        self.resize(1200, 850)

        if logger:
            self.logger = logger.getChild("BatchTimingReportPopup")
        else:
            self.logger = logging.getLogger("ASPIR.BatchTimingReportPopup")

        self._tests = tests
        self._current_test_idx = current_test_idx

        # Chart configuration with defaults
        self._chart_config = {
            'axes': {
                'title': '',
                'title_fontsize': 12,
                'xlabel': '',
                'xlabel_fontsize': 10,
                'xtick_fontsize': 8,
                'ylabel': '',
                'ylabel_fontsize': 10,
                'ytick_fontsize': 8,
                'auto_scale': True,
                'ymin': 0.0,
                'ymax': 100.0,
            },
            'legend': {
                'position': 5,  # Below (outside)
                'fontsize': 8,
                'frameon': False,
                'shadow': False,
                'fancybox': True,
                'framealpha': 0.8,
                'ncol': 4,
            },
            'colors': {
                'acquisition': self.COLOR_ACQUISITION,
                'reconstruction': self.COLOR_RECONSTRUCTION,
                'inference_cpu': self.COLOR_INFERENCE_CPU,
                'inference_gpu': self.COLOR_INFERENCE_GPU,
            }
        }

        build_ui(self)
        self._update_display()

    # ----- Event handlers ---------------------------------------------------

    def _on_test_changed(self, index: int):
        """Handle test selection change."""
        self._current_test_idx = index
        self._update_display()

    def _on_open_chart_config(self):
        """Open chart configuration dialog."""
        popup = ChartConfigPopup(parent=self, logger=self.logger)
        popup.set_config(self._chart_config)

        if popup.exec() == QDialog.Accepted:
            self._chart_config = popup.get_config()
            self.logger.debug("Chart config updated: %s", self._chart_config)
            timing_charts.update_timing_charts(self)

    def _on_profiler_device_changed(self, index: int):
        """Handle profiler device selector change."""
        profiler_charts.update_profiler_charts_for_selected_device(self)

    def _update_display(self):
        """Update all displays for the current test."""
        if not self._tests or self._current_test_idx >= len(self._tests):
            return

        test = self._tests[self._current_test_idx]
        test_name = test.get("name", "Unknown")

        # Update window title
        self.setWindowTitle(f"Detailed Timing Report - {test_name}")

        # Update timing tab
        timing_charts.update_timing_charts(self)
        timing_charts.update_statistics(self)

        # Update profiler tab
        profiler_charts.update_profiler_display(self)

    # ----- Save / export ----------------------------------------------------

    def _show_save_menu(self, pos, widget, figure, name):
        """Show context menu to save a specific figure."""
        menu = QMenu(self)
        save_png = menu.addAction("Save as PNG...")
        save_pdf = menu.addAction("Save as PDF...")

        action = menu.exec(widget.mapToGlobal(pos))

        if action == save_png:
            self._save_figure(figure, name, "png")
        elif action == save_pdf:
            self._save_figure(figure, name, "pdf")

    def _save_figure(self, figure, name, ext):
        """Save a specific figure to file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, f"Save {name}", f"{name}.{ext}",
            f"{ext.upper()} Files (*.{ext});;All Files (*.*)"
        )

        if file_path:
            try:
                figure.savefig(file_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Figure saved to {file_path}")
            except Exception as e:
                self.logger.error(f"Failed to save figure: {e}")

    def _on_chart_click(self, event, figure, chart_name):
        """Handle right-click on chart to save as image."""
        if event.button != 3:  # Only right click
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            f"Save {chart_name.replace('_', ' ').title()} Chart",
            f"profiler_{chart_name}.png",
            "PNG Image (*.png);;PDF Document (*.pdf);;SVG Vector (*.svg);;All Files (*.*)"
        )

        if not file_path:
            return

        try:
            dpi = 150
            if file_path.lower().endswith('.pdf') or file_path.lower().endswith('.svg'):
                dpi = 300

            figure.savefig(file_path, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
            self.logger.info(f"Chart saved to {file_path}")
            QMessageBox.information(self, "Chart Saved", f"Chart saved to:\n{file_path}")
        except Exception as e:
            self.logger.error(f"Failed to save chart: {e}")
            QMessageBox.critical(self, "Save Error", f"Failed to save chart:\n{e}")

    def _on_export(self, format_type):
        """Export all figures to files."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Report", f"timing_report.{format_type}",
            f"{format_type.upper()} Files (*.{format_type});;All Files (*.*)"
        )

        if not file_path:
            return

        base_path = file_path.rsplit('.', 1)[0] if '.' in file_path else file_path

        try:
            self.curves_figure.savefig(f"{base_path}_time_per_image.{format_type}",
                                       dpi=300, bbox_inches='tight')
            self.hist_figure.savefig(f"{base_path}_distribution.{format_type}",
                                     dpi=300, bbox_inches='tight')
            self.bar_figure.savefig(f"{base_path}_breakdown.{format_type}",
                                    dpi=300, bbox_inches='tight')
            self.logger.info(f"Report exported to {base_path}_*.{format_type}")
            QMessageBox.information(
                self, "Export Complete",
                f"Report exported to:\n{base_path}_*.{format_type}"
            )
        except Exception as e:
            self.logger.error(f"Failed to export report: {e}")
            QMessageBox.critical(self, "Export Error", f"Failed to export report:\n{e}")
