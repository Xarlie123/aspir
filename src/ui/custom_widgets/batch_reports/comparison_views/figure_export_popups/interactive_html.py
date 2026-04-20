"""Interactive HTML Report popup — exports self-contained Plotly report."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from ui.custom_widgets.batch_reports.comparison_views.figure_export_popups._base import (
    BaseFigureExportPopup,
)
from ui.utils.file_formats import BATCH_TESTS_DIR


class InteractiveHTMLPopup(BaseFigureExportPopup):
    """
    Popup for exporting an interactive HTML report with all charts.
    Uses Plotly for interactive visualizations.
    """

    def __init__(self, tests: list[dict[str, Any]], logger=None, parent=None):
        super().__init__(tests, logger, parent)
        self.setWindowTitle("Interactive HTML Report")
        self.setMinimumSize(800, 600)
        self.resize(900, 650)

        self._setup_ui()

    def _setup_ui(self):
        """Setup the popup UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Title
        title = QLabel("Export Interactive HTML Report")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(title)

        desc = QLabel(
            "Generate an interactive HTML report with Plotly charts. "
            "The report will be self-contained and can be opened in any modern web browser."
        )
        desc.setStyleSheet("color: #666;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Content options
        content_group = QGroupBox("Report Content")
        content_layout = QVBoxLayout(content_group)

        self.include_summary = QCheckBox("Include summary statistics")
        self.include_summary.setChecked(True)
        content_layout.addWidget(self.include_summary)

        self.include_quality = QCheckBox("Include quality metrics charts (PSNR, SSIM, LPIPS)")
        self.include_quality.setChecked(True)
        content_layout.addWidget(self.include_quality)

        self.include_timing = QCheckBox("Include timing comparison chart")
        self.include_timing.setChecked(True)
        content_layout.addWidget(self.include_timing)

        self.include_energy = QCheckBox("Include energy consumption chart")
        self.include_energy.setChecked(True)
        content_layout.addWidget(self.include_energy)

        self.include_table = QCheckBox("Include detailed results table")
        self.include_table.setChecked(True)
        content_layout.addWidget(self.include_table)

        layout.addWidget(content_group)

        # Style options
        style_group = QGroupBox("Style Options")
        style_layout = QGridLayout(style_group)

        style_layout.addWidget(QLabel("Color theme:"), 0, 0)
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Light", "Dark", "Seaborn"])
        style_layout.addWidget(self.theme_combo, 0, 1)

        style_layout.addWidget(QLabel("Chart height (px):"), 1, 0)
        self.chart_height_spin = QSpinBox()
        self.chart_height_spin.setMinimum(300)
        self.chart_height_spin.setMaximum(800)
        self.chart_height_spin.setValue(450)
        style_layout.addWidget(self.chart_height_spin, 1, 1)

        layout.addWidget(style_group)

        # Output path
        path_group = QGroupBox("Output File")
        path_layout = QHBoxLayout(path_group)

        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Select output file...")
        self.path_edit.setReadOnly(True)
        path_layout.addWidget(self.path_edit)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._on_browse)
        path_layout.addWidget(browse_btn)

        layout.addWidget(path_group)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.export_btn = QPushButton("Export HTML Report")
        self.export_btn.setEnabled(False)
        self.export_btn.setMinimumWidth(180)
        self.export_btn.setStyleSheet("""
            QPushButton {
                background-color: #9C27B0;
                color: white;
                font-weight: bold;
                padding: 10px 20px;
                border-radius: 4px;
                font-size: 13px;
            }
            QPushButton:hover:enabled {
                background-color: #7B1FA2;
            }
            QPushButton:disabled {
                background-color: #ccc;
                color: #888;
            }
        """)
        self.export_btn.clicked.connect(self._on_export)
        btn_layout.addWidget(self.export_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)

        layout.addLayout(btn_layout)

        layout.addStretch()

    def _on_browse(self):
        """Handle browse button click."""
        default_name = f"batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save HTML Report",
            str(BATCH_TESTS_DIR / default_name),
            "HTML Files (*.html);;All Files (*.*)"
        )

        if file_path:
            self.path_edit.setText(file_path)
            self.export_btn.setEnabled(True)

    def _on_export(self):
        """Export the HTML report."""
        output_path = Path(self.path_edit.text())

        try:
            html_content = self._generate_html()

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            self.logger.info("Exported HTML report to %s", output_path)
            QMessageBox.information(
                self, "Export Complete",
                f"HTML report exported to:\n{output_path}"
            )

        except Exception as e:
            self.logger.error("Failed to export HTML: %s", e)
            QMessageBox.warning(self, "Error", f"Failed to export HTML:\n{e}")

    def _generate_html(self) -> str:
        """Generate the HTML content with Plotly charts."""
        theme = self.theme_combo.currentText().lower()
        chart_height = self.chart_height_spin.value()

        # Collect data
        experiments = sorted(set(t.get("_experiment_name", "Unknown") for t in self._tests))
        test_names = [t.get("name", f"Test {i}") for i, t in enumerate(self._tests)]

        # Build chart data
        psnr_data = [t.get("psnr_denoised", 0) for t in self._tests]
        ssim_data = [t.get("ssim_denoised", 0) for t in self._tests]
        lpips_data = [t.get("lpips_denoised", 0) for t in self._tests]  # noqa: F841
        timing_data = [t.get("timing_mean_ms", 0) for t in self._tests]
        energy_data = [t.get("energy_mean_mj", 0) for t in self._tests]

        # Determine template based on theme
        if theme == "dark":
            template = "plotly_dark"
            bg_color = "#1e1e1e"
            text_color = "#fff"
            card_bg = "#2d2d2d"
        elif theme == "seaborn":
            template = "seaborn"
            bg_color = "#f5f5f5"
            text_color = "#333"
            card_bg = "#fff"
        else:
            template = "plotly_white"
            bg_color = "#f5f5f5"
            text_color = "#333"
            card_bg = "#fff"

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Batch Test Comparison Report</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: {bg_color};
            color: {text_color};
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        h1 {{
            color: #1976D2;
            border-bottom: 3px solid #1976D2;
            padding-bottom: 15px;
            margin-bottom: 10px;
        }}
        .meta {{
            color: #666;
            font-size: 14px;
            margin-bottom: 30px;
        }}
        .summary-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            margin-bottom: 30px;
        }}
        .summary-card {{
            flex: 1;
            min-width: 180px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        .summary-card.green {{ background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }}
        .summary-card.orange {{ background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }}
        .summary-card.blue {{ background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); }}
        .summary-value {{
            font-size: 32px;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .summary-label {{
            font-size: 13px;
            opacity: 0.9;
        }}
        .chart-card {{
            background: {card_bg};
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 25px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        }}
        .chart-title {{
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 15px;
            color: {text_color};
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #1976D2;
            color: white;
            font-weight: bold;
        }}
        tr:hover {{
            background-color: rgba(25, 118, 210, 0.1);
        }}
        .footer {{
            margin-top: 40px;
            padding: 20px;
            text-align: center;
            color: #999;
            font-size: 12px;
            border-top: 1px solid #ddd;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Batch Test Comparison Report</h1>
        <p class="meta">
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
            Experiments: {', '.join(experiments)}<br>
            Total tests: {len(self._tests)}
        </p>
"""

        # Summary cards
        if self.include_summary.isChecked():
            avg_psnr = np.mean([v for v in psnr_data if v]) if any(psnr_data) else 0
            avg_ssim = np.mean([v for v in ssim_data if v]) if any(ssim_data) else 0
            avg_time = np.mean([v for v in timing_data if v]) if any(timing_data) else 0
            avg_energy = np.mean([v for v in energy_data if v]) if any(energy_data) else 0

            html += f"""
        <div class="summary-row">
            <div class="summary-card green">
                <div class="summary-value">{avg_psnr:.2f} dB</div>
                <div class="summary-label">Average PSNR</div>
            </div>
            <div class="summary-card blue">
                <div class="summary-value">{avg_ssim:.4f}</div>
                <div class="summary-label">Average SSIM</div>
            </div>
            <div class="summary-card orange">
                <div class="summary-value">{avg_time:.2f} ms</div>
                <div class="summary-label">Average Inference Time</div>
            </div>
            <div class="summary-card">
                <div class="summary-value">{avg_energy:.2f} mJ</div>
                <div class="summary-label">Average Energy</div>
            </div>
        </div>
"""

        # Quality chart
        if self.include_quality.isChecked():
            html += f"""
        <div class="chart-card">
            <div class="chart-title">Quality Metrics Comparison</div>
            <div id="quality-chart"></div>
        </div>
        <script>
            var qualityData = [
                {{
                    x: {test_names},
                    y: {psnr_data},
                    name: 'PSNR (dB)',
                    type: 'bar',
                    marker: {{ color: '#4CAF50' }}
                }}
            ];
            var qualityLayout = {{
                template: '{template}',
                height: {chart_height},
                barmode: 'group',
                yaxis: {{ title: 'PSNR (dB)' }},
                legend: {{ orientation: 'h', y: -0.2 }}
            }};
            Plotly.newPlot('quality-chart', qualityData, qualityLayout, {{responsive: true}});
        </script>
"""

        # Timing chart
        if self.include_timing.isChecked():
            html += f"""
        <div class="chart-card">
            <div class="chart-title">Inference Time Comparison</div>
            <div id="timing-chart"></div>
        </div>
        <script>
            var timingData = [
                {{
                    x: {test_names},
                    y: {timing_data},
                    type: 'bar',
                    marker: {{
                        color: {timing_data},
                        colorscale: 'Blues',
                        reversescale: true
                    }}
                }}
            ];
            var timingLayout = {{
                template: '{template}',
                height: {chart_height},
                yaxis: {{ title: 'Time (ms)' }},
                showlegend: false
            }};
            Plotly.newPlot('timing-chart', timingData, timingLayout, {{responsive: true}});
        </script>
"""

        # Energy chart
        if self.include_energy.isChecked() and any(energy_data):
            html += f"""
        <div class="chart-card">
            <div class="chart-title">Energy Consumption Comparison</div>
            <div id="energy-chart"></div>
        </div>
        <script>
            var energyData = [
                {{
                    x: {test_names},
                    y: {energy_data},
                    type: 'bar',
                    marker: {{
                        color: {energy_data},
                        colorscale: 'Oranges'
                    }}
                }}
            ];
            var energyLayout = {{
                template: '{template}',
                height: {chart_height},
                yaxis: {{ title: 'Energy (mJ)' }},
                showlegend: false
            }};
            Plotly.newPlot('energy-chart', energyData, energyLayout, {{responsive: true}});
        </script>
"""

        # Results table
        if self.include_table.isChecked():
            html += """
        <div class="chart-card">
            <div class="chart-title">Detailed Results</div>
            <table>
                <thead>
                    <tr>
                        <th>Experiment</th>
                        <th>Test</th>
                        <th>Mask</th>
                        <th>Model</th>
                        <th>PSNR (dB)</th>
                        <th>SSIM</th>
                        <th>Time (ms)</th>
                    </tr>
                </thead>
                <tbody>
"""
            for test in self._tests:
                html += f"""
                    <tr>
                        <td>{test.get('_experiment_name', '-')}</td>
                        <td>{test.get('name', '-')}</td>
                        <td>{test.get('mask_type', '-')}</td>
                        <td>{test.get('model_name', '-')}</td>
                        <td>{test.get('psnr_denoised', 0):.2f}</td>
                        <td>{test.get('ssim_denoised', 0):.4f}</td>
                        <td>{test.get('timing_mean_ms', 0):.2f}</td>
                    </tr>
"""
            html += """
                </tbody>
            </table>
        </div>
"""

        # Footer
        html += """
        <div class="footer">
            Generated by ASPIR - Batch Reports Module
        </div>
    </div>
</body>
</html>
"""

        return html
