"""Popup dialog for displaying profiler results and bottleneck analysis."""
import logging
from typing import Dict, Any, List

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QGridLayout, QFileDialog, QSizePolicy, QScrollArea,
    QWidget, QFrame, QTextEdit, QTableWidget, QTableWidgetItem,
    QHeaderView, QSplitter, QProgressBar, QMessageBox, QApplication
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont




class ProfilerResultsPopup(QDialog):
    """
    Popup dialog showing profiler results with:
    - Summary text
    - Bottlenecks bar chart
    - Layer breakdown pie chart
    - Detailed operations table
    """

    def __init__(self, parent=None, simulation=None, use_gpu=False, logger=None):
        super().__init__(parent)
        self.setWindowTitle("Performance Profiler - Bottleneck Analysis")
        self.setMinimumSize(1000, 700)
        self.resize(1100, 750)

        if logger:
            self.logger = logger.getChild("ProfilerResultsPopup")
        else:
            self.logger = logging.getLogger("SPIm.ProfilerResultsPopup")

        self.simulation = simulation
        self._results = None

        # Determine device based on use_gpu setting
        import torch
        self._gpu_available = torch.cuda.is_available()
        self._use_gpu = use_gpu and self._gpu_available
        self._device = "cuda" if self._use_gpu else "cpu"
        self._device_name = torch.cuda.get_device_name(0) if self._use_gpu else "CPU"

        self._setup_ui()

    def _setup_ui(self):
        """Setup the popup UI layout."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Title
        title = QLabel("<h2>Performance Profiler</h2>")
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)

        # Device status indicator
        device_widget = QWidget()
        device_layout = QHBoxLayout(device_widget)
        device_layout.setContentsMargins(20, 5, 20, 5)

        device_indicator_color = "#4CAF50" if self._use_gpu else "#2196F3"
        device_text = f"GPU: {self._device_name}" if self._use_gpu else "CPU"
        device_tooltip = (
            f"Profiling will run on: {self._device_name}\n\n"
            "To change, close this dialog and toggle the\n"
            "'Use GPU for inference' checkbox in Timing Analysis."
        )

        device_label = QLabel(
            f"<span style='color: {device_indicator_color};'>●</span> "
            f"<b>Profiling device:</b> {device_text}"
        )
        device_label.setToolTip(device_tooltip)
        device_layout.addWidget(device_label)
        device_layout.addStretch()

        if not self._gpu_available:
            no_gpu_label = QLabel(
                "<span style='color: #FF5722;'>⚠</span> "
                "<i>CUDA not available</i>"
            )
            no_gpu_label.setToolTip("No compatible GPU detected. Profiling will use CPU only.")
            device_layout.addWidget(no_gpu_label)

        main_layout.addWidget(device_widget)

        # Description section
        desc_widget = QWidget()
        desc_layout = QVBoxLayout(desc_widget)
        desc_layout.setContentsMargins(20, 10, 20, 10)

        desc_label = QLabel(
            "<b>Profiling Options:</b><br><br>"
            "<span style='color: #FF9800;'>&#9632;</span> <b>Profile DNN Inference</b> - "
            "Analyzes only the neural network operations (convolutions, batch norm, activations, etc.) "
            "to identify which layers consume the most time.<br><br>"
            "<span style='color: #9C27B0;'>&#9632;</span> <b>Profile Full Pipeline</b> - "
            "Analyzes the complete processing pipeline including image reconstruction and DNN inference. "
            "Shows the time breakdown between reconstruction and denoising stages."
        )
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet(
            "QLabel { background-color: #F5F5F5; padding: 12px; "
            "border-radius: 4px; border: 1px solid #E0E0E0; }"
        )
        desc_layout.addWidget(desc_label)
        main_layout.addWidget(desc_widget)

        # Progress section (shown during profiling)
        self.progress_widget = QWidget()
        progress_layout = QVBoxLayout(self.progress_widget)
        self.progress_label = QLabel("Select a profiling option above")
        self.progress_label.setAlignment(Qt.AlignCenter)
        self.progress_label.setStyleSheet("color: #666; font-style: italic;")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_label)
        progress_layout.addWidget(self.progress_bar)
        main_layout.addWidget(self.progress_widget)

        # Results section (hidden until profiling complete)
        self.results_widget = QWidget()
        self.results_widget.setVisible(False)
        results_layout = QVBoxLayout(self.results_widget)
        results_layout.setSpacing(10)

        # Use splitter for resizable sections
        splitter = QSplitter(Qt.Horizontal)

        # Left: Charts
        charts_widget = QWidget()
        charts_layout = QVBoxLayout(charts_widget)
        charts_layout.setContentsMargins(5, 5, 5, 5)

        # Bottlenecks bar chart
        bar_group = QGroupBox("Top Bottlenecks (Time in ms)")
        bar_group.setToolTip("Right-click on the chart to save as image")
        bar_layout = QVBoxLayout(bar_group)
        bar_layout.setContentsMargins(2, 2, 2, 2)
        self.bar_figure = Figure(dpi=100)
        self.bar_figure.set_tight_layout(True)
        self.bar_canvas = FigureCanvas(self.bar_figure)
        self.bar_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.bar_canvas.setMinimumHeight(150)
        self.bar_canvas.setToolTip("Right-click to save chart as PNG/PDF/SVG")
        bar_layout.addWidget(self.bar_canvas)
        charts_layout.addWidget(bar_group, 1)  # stretch factor 1

        # Layer breakdown pie chart
        pie_group = QGroupBox("Time by Operation Type")
        pie_group.setToolTip("Right-click on the chart to save as image")
        pie_layout = QVBoxLayout(pie_group)
        pie_layout.setContentsMargins(2, 2, 2, 2)
        self.pie_figure = Figure(dpi=100)
        self.pie_figure.set_tight_layout(True)
        self.pie_canvas = FigureCanvas(self.pie_figure)
        self.pie_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.pie_canvas.setMinimumHeight(150)
        self.pie_canvas.setToolTip("Right-click to save chart as PNG/PDF/SVG")
        pie_layout.addWidget(self.pie_canvas)
        charts_layout.addWidget(pie_group, 1)  # stretch factor 1

        # Add click-to-save functionality for charts
        self.bar_canvas.mpl_connect('button_press_event', lambda e: self._on_chart_click(e, self.bar_figure, "bottlenecks"))
        self.pie_canvas.mpl_connect('button_press_event', lambda e: self._on_chart_click(e, self.pie_figure, "time_distribution"))

        # Connect resize events to redraw charts
        self.bar_canvas.mpl_connect('resize_event', self._on_bar_resize)
        self.pie_canvas.mpl_connect('resize_event', self._on_pie_resize)

        splitter.addWidget(charts_widget)

        # Right: Summary and table
        details_widget = QWidget()
        details_layout = QVBoxLayout(details_widget)
        details_layout.setContentsMargins(5, 5, 5, 5)

        # Summary text
        summary_group = QGroupBox("Summary")
        summary_layout = QVBoxLayout(summary_group)
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setFont(QFont("Monospace", 9))
        self.summary_text.setMaximumHeight(200)
        summary_layout.addWidget(self.summary_text)
        details_layout.addWidget(summary_group)

        # Operations table
        table_group = QGroupBox("Detailed Operations")
        table_layout = QVBoxLayout(table_group)
        self.ops_table = QTableWidget()
        self.ops_table.setColumnCount(5)
        self.ops_table.setHorizontalHeaderLabels([
            "Operation", "CPU Time (ms)", "CUDA Time (ms)", "Calls", "Time/Call (ms)"
        ])
        self.ops_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.ops_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.ops_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.ops_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.ops_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.ops_table.setAlternatingRowColors(True)
        table_layout.addWidget(self.ops_table)
        details_layout.addWidget(table_group)

        splitter.addWidget(details_widget)
        splitter.setSizes([500, 500])

        results_layout.addWidget(splitter)
        main_layout.addWidget(self.results_widget, 1)

        # Buttons
        buttons_layout = QHBoxLayout()

        self.run_button = QPushButton("Profile DNN Inference")
        self.run_button.setMinimumHeight(35)
        self.run_button.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #F57C00; }
            QPushButton:disabled { background-color: #ccc; color: #666; }
        """)
        self.run_button.setToolTip(
            f"<b>Profile DNN Inference Only</b><br><br>"
            f"Analyzes the neural network forward pass to identify<br>"
            f"which layers and operations are the slowest.<br><br>"
            f"<b>Device:</b> {self._device_name}<br><br>"
            f"<i>Use this to optimize your model architecture.</i>"
        )
        self.run_button.clicked.connect(self._on_run_profiler)
        buttons_layout.addWidget(self.run_button)

        self.run_full_button = QPushButton("Profile Full Pipeline")
        self.run_full_button.setMinimumHeight(35)
        self.run_full_button.setStyleSheet("""
            QPushButton {
                background-color: #9C27B0;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #7B1FA2; }
            QPushButton:disabled { background-color: #ccc; color: #666; }
        """)
        self.run_full_button.setToolTip(
            f"<b>Profile Complete Pipeline</b><br><br>"
            f"Analyzes both:<br>"
            f"1. <b>Reconstruction</b> - Image reconstruction (always CPU)<br>"
            f"2. <b>DNN Inference</b> - Neural network denoising ({self._device_name})<br><br>"
            f"<i>Use this to see the full picture of processing time.</i>"
        )
        self.run_full_button.clicked.connect(lambda: self._on_run_profiler(full_pipeline=True))
        buttons_layout.addWidget(self.run_full_button)

        buttons_layout.addStretch()

        self.export_button = QPushButton("Export Results")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._on_export)
        buttons_layout.addWidget(self.export_button)

        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close)
        buttons_layout.addWidget(self.close_button)

        main_layout.addLayout(buttons_layout)

    def _on_run_profiler(self, full_pipeline=False):
        """Run profiler directly in main thread (required for CUDA profiling)."""
        if self.simulation is None:
            QMessageBox.warning(self, "Error", "No simulation available")
            return

        post = getattr(self.simulation, 'postprocessor', None)
        if post is None or not getattr(post, 'trained', False):
            QMessageBox.warning(self, "Error", "No trained model available. Train or load a model first.")
            return

        # Disable buttons during profiling
        self.run_button.setEnabled(False)
        self.run_full_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_label.setText("Profiling... please wait")
        QApplication.processEvents()  # Update UI

        try:
            from simulation_engine._5_analyzer.profiler import PipelineProfiler

            # Use the device setting from constructor (based on GPU checkbox)
            device = self._device

            self.progress_label.setText(f"Initializing profiler on {self._device_name}...")
            QApplication.processEvents()

            profiler = PipelineProfiler(self.simulation, logger=self.logger)

            # Get the size of the validation dataset
            post = self.simulation.postprocessor
            val_loader = post.loaders.get("val")
            if val_loader is not None:
                num_images = len(val_loader.dataset)
            else:
                num_images = 10  # Fallback

            if full_pipeline:
                self.progress_label.setText(f"Profiling full pipeline on {self._device_name} ({num_images} images)...")
                QApplication.processEvents()
                results = profiler.profile_full_pipeline(
                    num_images=num_images,
                    device=device
                )
            else:
                self.progress_label.setText(f"Profiling DNN inference on {self._device_name} ({num_images} images)...")
                QApplication.processEvents()
                results = profiler.profile_inference(
                    num_images=num_images,
                    warmup_runs=3,
                    device=device
                )

            # Add summary and layer breakdown
            results['summary'] = profiler.get_summary()
            results['layer_breakdown'] = profiler.get_layer_breakdown()
            results['key_averages_table'] = results.get('key_averages_table', '')

            # Update display
            self._results = results
            self.progress_bar.setVisible(False)
            self.progress_label.setText("Profiling complete!")
            self.export_button.setEnabled(True)
            self.results_widget.setVisible(True)
            self._update_display()

        except Exception as e:
            self.logger.error(f"Profiling failed: {e}", exc_info=True)
            self.progress_bar.setVisible(False)
            self.progress_label.setText(f"Error: {str(e)[:50]}...")
            QMessageBox.critical(self, "Profiling Error", f"Failed to profile:\n{e}")

        finally:
            self.run_button.setEnabled(True)
            self.run_full_button.setEnabled(True)

    def _update_display(self):
        """Update all displays with profiling results."""
        if not self._results:
            return

        self._update_summary()
        self._update_bar_chart()
        self._update_pie_chart()
        self._update_table()

    def _on_bar_resize(self, event):
        """Handle bar chart canvas resize."""
        if self._results:
            self._update_bar_chart()

    def _on_pie_resize(self, event):
        """Handle pie chart canvas resize."""
        if self._results:
            self._update_pie_chart()

    def _update_summary(self):
        """Update summary text."""
        summary = self._results.get('summary', 'No summary available')
        self.summary_text.setPlainText(summary)

    def _update_bar_chart(self):
        """Update bottlenecks bar chart."""
        self.bar_figure.clear()
        ax = self.bar_figure.add_subplot(111)

        bottlenecks = self._results.get('bottlenecks', [])[:10]
        if not bottlenecks:
            ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
            self.bar_canvas.draw()
            return

        device = self._results.get('device', 'cpu')

        names = []
        times = []
        for op in bottlenecks:
            # Shorten name for display
            name = op['name']
            if len(name) > 25:
                name = name[:22] + "..."
            names.append(name)

            if device == 'cuda' and op['cuda_time_ms'] > 0:
                times.append(op['cuda_time_ms'])
            else:
                times.append(op['cpu_time_ms'])

        y_pos = np.arange(len(names))
        colors = ['#d7191c' if i == 0 else '#fdae61' if i < 3 else '#2b83ba' for i in range(len(names))]

        ax.barh(y_pos, times, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel('Time (ms)')
        ax.grid(True, alpha=0.3, axis='x')

        self.bar_figure.tight_layout()
        self.bar_canvas.draw()

    def _update_pie_chart(self):
        """Update layer breakdown pie chart."""
        self.pie_figure.clear()

        layer_breakdown = self._results.get('layer_breakdown', [])
        if not layer_breakdown:
            ax = self.pie_figure.add_subplot(111)
            ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
            self.pie_canvas.draw()
            return

        # Get number of images and device for title
        num_images = self._results.get('num_images', 1)
        device = self._results.get('device', 'cpu')
        device_label = "GPU Kernel" if device == 'cuda' else "CPU"

        labels = []
        sizes = []
        for layer in layer_breakdown[:10]:  # Top 10 categories
            if layer['total_time_ms'] > 0:
                labels.append(layer['category'])
                sizes.append(layer['total_time_ms'])

        if not sizes:
            ax = self.pie_figure.add_subplot(111)
            ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
            self.pie_canvas.draw()
            return

        colors = [
            '#d7191c',  # Red - Convolution
            '#fdae61',  # Orange - BatchNorm
            '#abdda4',  # Light green - Activation (ReLU)
            '#2b83ba',  # Blue - Pooling
            '#9C27B0',  # Purple - Linear
            '#607D8B',  # Blue Gray - Add
            '#FF5722',  # Deep Orange - Upsample
            '#00BCD4',  # Cyan - Memory Transfer
            '#8BC34A',  # Light Green - Reshape
            '#795548',  # Brown - Dropout
            '#3F51B5',  # Indigo - Tensor Allocation
            '#FFEB3B',  # Yellow - Element-wise Ops
            '#E91E63',  # Pink - Activation (Other)
            '#009688',  # Teal - Concatenation
            '#FF9800',  # Orange - CUDNN Ops
            '#9E9E9E',  # Gray - Other
        ]

        # Get figure size in pixels for better layout decisions
        fig_width, fig_height = self.pie_figure.get_size_inches()
        dpi = self.pie_figure.get_dpi()
        width_px = fig_width * dpi
        height_px = fig_height * dpi

        # Calculate total time from breakdown for display
        total_time = sum(sizes)

        # Only show percentages inside the pie for large slices
        def autopct_func(pct):
            return f'{pct:.1f}%' if pct > 5 else ''

        # Decide layout based on available space
        # Use side-by-side layout only if we have enough width (>500px)
        use_side_legend = width_px > 500

        if use_side_legend:
            # Two-column layout: pie on left, legend on right
            # Use GridSpec for more control
            from matplotlib.gridspec import GridSpec
            gs = GridSpec(1, 2, figure=self.pie_figure, width_ratios=[1, 0.8], wspace=0.05)
            ax = self.pie_figure.add_subplot(gs[0])
            ax_legend = self.pie_figure.add_subplot(gs[1])
            ax_legend.axis('off')

            wedges, texts, autotexts = ax.pie(
                sizes,
                autopct=autopct_func,
                colors=colors[:len(sizes)],
                textprops={'fontsize': 9, 'weight': 'bold'},
                pctdistance=0.7
            )

            # Make percentage text white for better contrast
            for autotext in autotexts:
                autotext.set_color('white')

            ax.set_title(f'{device_label} Time by Operation\n({total_time:.1f} ms, {num_images} images)',
                        fontsize=10, fontweight='bold', pad=5)

            # Add legend in the right subplot
            legend_labels = [f"{label} ({size:.1f} ms)" for label, size in zip(labels, sizes)]
            ax_legend.legend(
                wedges,
                legend_labels,
                title="Operation Type",
                loc="center",
                fontsize=9,
                title_fontsize=10,
                frameon=False
            )
            self.pie_figure.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.05)
        else:
            # Vertical layout: pie on top, legend below
            ax = self.pie_figure.add_subplot(111)

            wedges, texts, autotexts = ax.pie(
                sizes,
                autopct=autopct_func,
                colors=colors[:len(sizes)],
                textprops={'fontsize': 8, 'weight': 'bold'},
                pctdistance=0.7
            )

            # Make percentage text white for better contrast
            for autotext in autotexts:
                autotext.set_color('white')

            ax.set_title(f'{device_label} Time by Operation\n({total_time:.1f} ms, {num_images} images)',
                        fontsize=9, fontweight='bold', pad=3)

            # Legend below with multiple columns for compact display
            legend_labels = [f"{label} ({size:.1f} ms)" for label, size in zip(labels, sizes)]
            ncols = min(3, len(labels)) if len(labels) > 4 else 2
            ax.legend(
                wedges,
                legend_labels,
                title="Operation Type",
                loc="upper center",
                bbox_to_anchor=(0.5, -0.02),
                fontsize=7,
                title_fontsize=8,
                ncol=ncols,
                frameon=False
            )
            self.pie_figure.subplots_adjust(left=0.02, right=0.98, top=0.85, bottom=0.30)

        self.pie_canvas.draw()

    def _update_table(self):
        """Update operations table."""
        bottlenecks = self._results.get('bottlenecks', [])
        device = self._results.get('device', 'cpu')

        self.ops_table.setRowCount(len(bottlenecks))

        for row, op in enumerate(bottlenecks):
            self.ops_table.setItem(row, 0, QTableWidgetItem(op['name']))
            self.ops_table.setItem(row, 1, QTableWidgetItem(f"{op['cpu_time_ms']:.3f}"))
            self.ops_table.setItem(row, 2, QTableWidgetItem(
                f"{op['cuda_time_ms']:.3f}" if device == 'cuda' else "-"
            ))
            self.ops_table.setItem(row, 3, QTableWidgetItem(str(op['calls'])))

            time_per_call = op['cuda_time_per_call_ms'] if device == 'cuda' else op['cpu_time_per_call_ms']
            self.ops_table.setItem(row, 4, QTableWidgetItem(f"{time_per_call:.3f}"))

    def _on_export(self):
        """Export results to file."""
        if not self._results:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Profiler Results", "profiler_results.txt",
            "Text Files (*.txt);;All Files (*.*)"
        )

        if not file_path:
            return

        try:
            with open(file_path, 'w') as f:
                f.write(self._results.get('summary', ''))
                f.write("\n\n")
                f.write("=" * 60 + "\n")
                f.write("DETAILED OPERATIONS TABLE\n")
                f.write("=" * 60 + "\n")
                f.write(self._results.get('key_averages_table', ''))

            self.logger.info(f"Results exported to {file_path}")
            QMessageBox.information(self, "Export Complete", f"Results saved to:\n{file_path}")
        except Exception as e:
            self.logger.error(f"Failed to export: {e}")
            QMessageBox.critical(self, "Export Error", f"Failed to export:\n{e}")

    def _on_chart_click(self, event, figure, chart_name):
        """Handle right-click on chart to save as image."""
        if event.button != 3:  # Only right click
            return

        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            f"Save {chart_name.replace('_', ' ').title()} Chart",
            f"profiler_{chart_name}.png",
            "PNG Image (*.png);;PDF Document (*.pdf);;SVG Vector (*.svg);;All Files (*.*)"
        )

        if not file_path:
            return

        try:
            # Determine DPI based on format
            dpi = 150
            if file_path.lower().endswith('.pdf') or file_path.lower().endswith('.svg'):
                dpi = 300  # Higher quality for vector formats

            figure.savefig(file_path, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
            self.logger.info(f"Chart saved to {file_path}")
            QMessageBox.information(self, "Chart Saved", f"Chart saved to:\n{file_path}")
        except Exception as e:
            self.logger.error(f"Failed to save chart: {e}")
            QMessageBox.critical(self, "Save Error", f"Failed to save chart:\n{e}")
