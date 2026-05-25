"""Timing Analysis page widget with report generation and results preview."""
import logging
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QGridLayout, QSizePolicy, QFrame, QSplitter,
    QScrollArea, QFormLayout, QDoubleSpinBox, QSpinBox, QCheckBox,
    QMenu, QFileDialog
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False


class TimingAnalysisPage(QWidget):
    """
    Timing Analysis page with two main sections:
    - Left: Configuration controls (GPU, parameters, run button)
    - Right: Results preview (stacked bar chart, summary table)
    """

    # Signal emitted when analysis is requested
    analysisRequested = Signal()
    # Signal emitted when PyTorch profiling is requested
    profilingRequested = Signal()
    # Signal emitted when Nsight profiling is requested
    nsightProfilingRequested = Signal()

    def __init__(self, parent=None, logger=None):
        super().__init__(parent)

        if logger:
            self.logger = logger.getChild("TimingAnalysisPage")
        else:
            self.logger = logging.getLogger("ASPIR.TimingAnalysisPage")

        # Data storage
        self._timing_data = {}
        self._has_data = False

        # GPU availability
        self._gpu_available = self._check_gpu_availability()

        self._setup_ui()
        self._update_gpu_status_display()

    def _check_gpu_availability(self) -> bool:
        """Check if CUDA GPU is available."""
        if not TORCH_AVAILABLE:
            self.logger.warning("PyTorch not available, GPU disabled")
            return False
        available = torch.cuda.is_available()
        if available:
            device_name = torch.cuda.get_device_name(0)
            self.logger.info(f"GPU available: {device_name}")
        else:
            self.logger.info("No CUDA GPU available")
        return available

    def _setup_ui(self):
        """Setup the main UI layout."""
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(10)

        # Use splitter for resizable sections
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        # Left section: Configuration
        left_widget = self._create_config_section()
        splitter.addWidget(left_widget)

        # Right section: Results Preview
        right_widget = self._create_results_section()
        splitter.addWidget(right_widget)

        # Set initial sizes (roughly 1:2 ratio)
        splitter.setSizes([350, 700])

        main_layout.addWidget(splitter)

    def _create_config_section(self):
        """Create the left section for configuration controls."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)

        # Title
        title = QLabel("<h3>Timing Configuration</h3>")
        layout.addWidget(title)

        # GPU Status group
        gpu_group = QGroupBox()
        gpu_layout = QVBoxLayout(gpu_group)
        gpu_layout.setSpacing(8)

        # GPU Status label
        self.gpu_status_label = QLabel("GPU Status: Checking...")
        font_status = QFont()
        font_status.setBold(True)
        self.gpu_status_label.setFont(font_status)
        self.gpu_status_label.setMinimumHeight(28)
        gpu_layout.addWidget(self.gpu_status_label)

        # GPU checkbox
        self.gpu_checkbox = QCheckBox("Use GPU for inference")
        self.gpu_checkbox.setChecked(True)
        self.gpu_checkbox.toggled.connect(self._on_gpu_checkbox_toggled)
        gpu_layout.addWidget(self.gpu_checkbox)

        layout.addWidget(gpu_group)

        # Parameters group
        params_group = QGroupBox("Parameters")
        params_layout = QFormLayout(params_group)
        params_layout.setSpacing(10)

        # Sampling rate
        self.sampling_rate_spinbox = QDoubleSpinBox()
        self.sampling_rate_spinbox.setDecimals(3)
        self.sampling_rate_spinbox.setMaximum(1000.0)
        self.sampling_rate_spinbox.setValue(10.752)
        self.sampling_rate_spinbox.setSuffix(" kHz")
        params_layout.addRow("Sampling rate:", self.sampling_rate_spinbox)

        # Warmup runs — 20 to match the unified app default
        # (Batch Test, Re-measure dialog and the energy analysis page
        # all default to the same number for cross-comparable runs).
        self.warmup_spinbox = QSpinBox()
        self.warmup_spinbox.setMinimum(0)
        self.warmup_spinbox.setMaximum(100)
        self.warmup_spinbox.setValue(20)
        params_layout.addRow("Warmup runs:", self.warmup_spinbox)

        # Measurement runs — 800 by default. The 500 ceiling that
        # used to live here predated the unified energy-measurement
        # convention; raised to 2000 so the user can also push past
        # the default for paper-quality numbers.
        self.measurement_spinbox = QSpinBox()
        self.measurement_spinbox.setMinimum(1)
        self.measurement_spinbox.setMaximum(2000)
        self.measurement_spinbox.setValue(800)
        params_layout.addRow("Measurement runs:", self.measurement_spinbox)

        layout.addWidget(params_group)

        # Formula explanation
        formula_label = QLabel(
            "<b>T<sub>total</sub></b> = T<sub>acquisition</sub> + "
            "T<sub>reconstruction</sub> + T<sub>DNN inference</sub>"
        )
        formula_label.setWordWrap(True)
        formula_label.setStyleSheet(
            "QLabel { background-color: #E3F2FD; padding: 8px; "
            "border-radius: 4px; border: 1px solid #90CAF9; color: #1565C0; }"
        )
        layout.addWidget(formula_label)

        # Run analysis button
        self.analyze_button = QPushButton("Run Timing Analysis")
        self.analyze_button.setMinimumHeight(40)
        self.analyze_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton:disabled {
                background-color: #ccc;
                color: #666;
            }
        """)
        self.analyze_button.clicked.connect(self.analysisRequested.emit)
        layout.addWidget(self.analyze_button)

        # Generate report button
        self.generate_button = QPushButton("Generate Timing Report")
        self.generate_button.setMinimumHeight(40)
        self.generate_button.setStyleSheet("""
            QPushButton {
                background-color: #0078d7;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #005a9e;
            }
            QPushButton:pressed {
                background-color: #004275;
            }
            QPushButton:disabled {
                background-color: #ccc;
                color: #666;
            }
        """)
        self.generate_button.clicked.connect(self._on_generate_report)
        self.generate_button.setEnabled(False)
        layout.addWidget(self.generate_button)

        # Profile with PyTorch button (orange)
        self.profile_button = QPushButton("Profile with PyTorch")
        self.profile_button.setMinimumHeight(40)
        self.profile_button.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
            QPushButton:pressed {
                background-color: #E65100;
            }
            QPushButton:disabled {
                background-color: #ccc;
                color: #666;
            }
        """)
        self.profile_button.clicked.connect(self.profilingRequested.emit)
        self.profile_button.setEnabled(False)
        self.profile_button.setToolTip(
            "<b>Profile with PyTorch Profiler</b><br><br>"
            "Analyze operation-level performance:<br>"
            "• GPU kernel execution times<br>"
            "• Layer-by-layer breakdown<br>"
            "• Bottleneck identification"
        )
        layout.addWidget(self.profile_button)

        # Profile with Nsight button (NVIDIA green)
        self.nsight_button = QPushButton("Profile with Nsight")
        self.nsight_button.setMinimumHeight(40)
        self.nsight_button.setStyleSheet("""
            QPushButton {
                background-color: #76B900;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #5A8F00;
            }
            QPushButton:pressed {
                background-color: #4A7500;
            }
            QPushButton:disabled {
                background-color: #ccc;
                color: #666;
            }
        """)
        self.nsight_button.clicked.connect(self.nsightProfilingRequested.emit)
        self.nsight_button.setEnabled(False)
        self.nsight_button.setToolTip(
            "<b>Profile with NVIDIA Nsight Systems</b><br><br>"
            "Detailed CPU↔GPU analysis:<br>"
            "• Memory transfers (cudaMemcpy)<br>"
            "• Kernel launch timeline<br>"
            "• PCIe bandwidth usage<br>"
            "• Synchronization points"
        )
        layout.addWidget(self.nsight_button)

        # Status label
        self.status_label = QLabel("Run analysis to see results")
        self.status_label.setStyleSheet("color: #888;")
        layout.addWidget(self.status_label)

        layout.addStretch()

        return container

    def _create_results_section(self):
        """Create the right section for results preview."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Title
        title = QLabel("<h3>Timing Results</h3>")
        layout.addWidget(title)

        # Stacked bar chart (CPU vs GPU)
        chart_group = QGroupBox("Pipeline Latency Breakdown: CPU vs GPU")
        chart_group.setContextMenuPolicy(Qt.CustomContextMenu)
        chart_group.customContextMenuRequested.connect(
            lambda pos: self._show_save_menu(pos, chart_group)
        )
        chart_layout = QVBoxLayout(chart_group)

        self.stacked_bar_figure = Figure(figsize=(6, 4), dpi=100)
        self.stacked_bar_canvas = FigureCanvas(self.stacked_bar_figure)
        self.stacked_bar_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.stacked_bar_canvas.setMinimumHeight(250)
        chart_layout.addWidget(self.stacked_bar_canvas)

        layout.addWidget(chart_group)

        # Summary table
        self.summary_group = QGroupBox("Timing Summary")
        summary_layout = QGridLayout(self.summary_group)
        summary_layout.setSpacing(10)

        # Headers
        header_font = QFont()
        header_font.setBold(True)

        summary_layout.addWidget(QLabel(""), 0, 0)

        cpu_header = QLabel("CPU")
        cpu_header.setFont(header_font)
        cpu_header.setAlignment(Qt.AlignCenter)
        summary_layout.addWidget(cpu_header, 0, 1)

        gpu_header = QLabel("GPU")
        gpu_header.setFont(header_font)
        gpu_header.setAlignment(Qt.AlignCenter)
        summary_layout.addWidget(gpu_header, 0, 2)

        speedup_header = QLabel("Speedup")
        speedup_header.setFont(header_font)
        speedup_header.setAlignment(Qt.AlignCenter)
        summary_layout.addWidget(speedup_header, 0, 3)

        # T_acquisition row
        t_acq_label = QLabel("T_acquisition:")
        t_acq_label.setFont(header_font)
        summary_layout.addWidget(t_acq_label, 1, 0)
        self.t_acq_cpu_label = QLabel("-")
        self.t_acq_cpu_label.setAlignment(Qt.AlignCenter)
        summary_layout.addWidget(self.t_acq_cpu_label, 1, 1)
        self.t_acq_gpu_label = QLabel("-")
        self.t_acq_gpu_label.setAlignment(Qt.AlignCenter)
        summary_layout.addWidget(self.t_acq_gpu_label, 1, 2)
        self.t_acq_speedup_label = QLabel("-")
        self.t_acq_speedup_label.setAlignment(Qt.AlignCenter)
        summary_layout.addWidget(self.t_acq_speedup_label, 1, 3)

        # T_reconstruction row
        t_recon_label = QLabel("T_reconstruction:")
        t_recon_label.setFont(header_font)
        summary_layout.addWidget(t_recon_label, 2, 0)
        self.t_recon_cpu_label = QLabel("-")
        self.t_recon_cpu_label.setAlignment(Qt.AlignCenter)
        summary_layout.addWidget(self.t_recon_cpu_label, 2, 1)
        self.t_recon_gpu_label = QLabel("-")
        self.t_recon_gpu_label.setAlignment(Qt.AlignCenter)
        summary_layout.addWidget(self.t_recon_gpu_label, 2, 2)
        self.t_recon_speedup_label = QLabel("-")
        self.t_recon_speedup_label.setAlignment(Qt.AlignCenter)
        summary_layout.addWidget(self.t_recon_speedup_label, 2, 3)

        # T_inference row
        t_inf_label = QLabel("T_inference:")
        t_inf_label.setFont(header_font)
        summary_layout.addWidget(t_inf_label, 3, 0)
        self.t_inf_cpu_label = QLabel("-")
        self.t_inf_cpu_label.setAlignment(Qt.AlignCenter)
        summary_layout.addWidget(self.t_inf_cpu_label, 3, 1)
        self.t_inf_gpu_label = QLabel("-")
        self.t_inf_gpu_label.setAlignment(Qt.AlignCenter)
        summary_layout.addWidget(self.t_inf_gpu_label, 3, 2)
        self.t_inf_speedup_label = QLabel("-")
        self.t_inf_speedup_label.setAlignment(Qt.AlignCenter)
        self.t_inf_speedup_label.setStyleSheet("font-weight: bold; color: #0078d7;")
        summary_layout.addWidget(self.t_inf_speedup_label, 3, 3)

        # T_total row
        t_total_label = QLabel("T_total:")
        t_total_label.setFont(header_font)
        summary_layout.addWidget(t_total_label, 4, 0)
        self.t_total_cpu_label = QLabel("-")
        self.t_total_cpu_label.setAlignment(Qt.AlignCenter)
        self.t_total_cpu_label.setStyleSheet("font-weight: bold;")
        summary_layout.addWidget(self.t_total_cpu_label, 4, 1)
        self.t_total_gpu_label = QLabel("-")
        self.t_total_gpu_label.setAlignment(Qt.AlignCenter)
        self.t_total_gpu_label.setStyleSheet("font-weight: bold;")
        summary_layout.addWidget(self.t_total_gpu_label, 4, 2)
        self.t_total_speedup_label = QLabel("-")
        self.t_total_speedup_label.setAlignment(Qt.AlignCenter)
        self.t_total_speedup_label.setStyleSheet("font-weight: bold; color: #4CAF50; font-size: 14px;")
        summary_layout.addWidget(self.t_total_speedup_label, 4, 3)

        layout.addWidget(self.summary_group)

        layout.addStretch()

        return container

    def _update_gpu_status_display(self):
        """Update the GPU status label with color indicator."""
        if self._gpu_available:
            device_name = torch.cuda.get_device_name(0) if TORCH_AVAILABLE else "Unknown"
            self.gpu_status_label.setText(f"GPU Available: {device_name}")
            self.gpu_status_label.setStyleSheet(
                "QLabel { color: #228B22; background-color: #E8F5E9; "
                "padding: 4px 8px; border-radius: 4px; border: 1px solid #A5D6A7; }"
            )
            self.gpu_checkbox.setEnabled(True)
        else:
            self.gpu_status_label.setText("GPU Not Available (CPU only)")
            self.gpu_status_label.setStyleSheet(
                "QLabel { color: #B22222; background-color: #FFEBEE; "
                "padding: 4px 8px; border-radius: 4px; border: 1px solid #EF9A9A; }"
            )
            self.gpu_checkbox.setEnabled(False)
            self.gpu_checkbox.setChecked(False)

    def _on_gpu_checkbox_toggled(self, checked: bool):
        """Handle GPU checkbox state change."""
        if checked and not self._gpu_available:
            self.gpu_checkbox.setChecked(False)
            self.logger.warning("Cannot enable GPU - not available")
            return
        self.logger.debug(f"GPU usage {'enabled' if checked else 'disabled'}")

    # --- Properties for accessing configuration ---

    @property
    def use_gpu(self) -> bool:
        """Return True if GPU should be used for inference."""
        return self.gpu_checkbox.isChecked() and self._gpu_available

    @property
    def sampling_rate_khz(self) -> float:
        """Return the sampling rate in kHz."""
        return self.sampling_rate_spinbox.value()

    @property
    def warmup_runs(self) -> int:
        """Return the number of warmup runs."""
        return self.warmup_spinbox.value()

    @property
    def measurement_runs(self) -> int:
        """Return the number of measurement runs."""
        return self.measurement_spinbox.value()

    # --- Data methods ---

    def set_data(self, timing_data: dict):
        """
        Set the timing data and update the display.

        Args:
            timing_data: Dictionary with timing values:
                - t_acq_ms: acquisition time in ms
                - t_recon_ms: reconstruction time in ms
                - t_inf_cpu_ms: inference time on CPU in ms
                - t_inf_gpu_ms: inference time on GPU in ms (optional)
                - recon_times_ms: list of per-image reconstruction times
                - denoise_times_ms: list of per-image denoising times
        """
        self._timing_data = timing_data
        self._has_data = True

        self._update_summary_table()
        self._update_stacked_bar_chart()

        self.generate_button.setEnabled(True)
        self.profile_button.setEnabled(True)
        self.nsight_button.setEnabled(True)
        self.status_label.setText("Analysis complete")
        self.status_label.setStyleSheet("color: #080;")

        self.logger.debug("Timing data set and display updated")

    def _update_summary_table(self):
        """Update the summary table with timing values."""
        if not self._timing_data:
            return

        t_acq = self._timing_data.get('t_acq_ms', 0)
        t_recon = self._timing_data.get('t_recon_ms', 0)
        t_inf_cpu = self._timing_data.get('t_inf_cpu_ms', 0)
        t_inf_gpu = self._timing_data.get('t_inf_gpu_ms', None)

        # Acquisition (same for CPU and GPU)
        self.t_acq_cpu_label.setText(f"{t_acq:.2f} ms")
        self.t_acq_gpu_label.setText(f"{t_acq:.2f} ms")
        self.t_acq_speedup_label.setText("-")

        # Reconstruction (same for CPU and GPU - runs on CPU)
        self.t_recon_cpu_label.setText(f"{t_recon:.2f} ms")
        self.t_recon_gpu_label.setText(f"{t_recon:.2f} ms")
        self.t_recon_speedup_label.setText("-")

        # Inference
        self.t_inf_cpu_label.setText(f"{t_inf_cpu:.2f} ms")
        if t_inf_gpu is not None:
            self.t_inf_gpu_label.setText(f"{t_inf_gpu:.2f} ms")
            if t_inf_gpu > 0:
                speedup = t_inf_cpu / t_inf_gpu
                self.t_inf_speedup_label.setText(f"{speedup:.1f}x")
        else:
            self.t_inf_gpu_label.setText("-")
            self.t_inf_speedup_label.setText("-")

        # Total
        t_total_cpu = t_acq + t_recon + t_inf_cpu
        self.t_total_cpu_label.setText(f"{t_total_cpu:.2f} ms")

        if t_inf_gpu is not None:
            t_total_gpu = t_acq + t_recon + t_inf_gpu
            self.t_total_gpu_label.setText(f"{t_total_gpu:.2f} ms")
            if t_total_gpu > 0:
                total_speedup = t_total_cpu / t_total_gpu
                self.t_total_speedup_label.setText(f"{total_speedup:.1f}x")
        else:
            self.t_total_gpu_label.setText("-")
            self.t_total_speedup_label.setText("-")

    def _update_stacked_bar_chart(self):
        """Update the stacked bar chart comparing CPU vs GPU pipeline latency."""
        self.stacked_bar_figure.clear()

        if not self._timing_data:
            ax = self.stacked_bar_figure.add_subplot(111)
            ax.text(0.5, 0.5, "No data available", ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
            self.stacked_bar_canvas.draw()
            return

        t_acq = self._timing_data.get('t_acq_ms', 0)
        t_recon = self._timing_data.get('t_recon_ms', 0)
        t_inf_cpu = self._timing_data.get('t_inf_cpu_ms', 0)
        t_inf_gpu = self._timing_data.get('t_inf_gpu_ms', None)

        ax = self.stacked_bar_figure.add_subplot(111)

        # Colors (consistent with stacked_bar_plot.py)
        c_acq = '#abdda4'    # Green (Acquisition)
        c_recon = '#fdae61'  # Orange (Reconstruction)
        c_cpu = '#d7191c'    # Red (Inference CPU)
        c_gpu = '#2b83ba'    # Blue (Inference GPU)

        x = np.array([0, 1])
        width = 0.5

        # CPU bar (left)
        ax.bar(x[0], t_acq, width, label='Acquisition', color=c_acq, edgecolor='white')
        ax.bar(x[0], t_recon, width, bottom=t_acq, label='Reconstruction', color=c_recon, edgecolor='white')
        ax.bar(x[0], t_inf_cpu, width, bottom=t_acq + t_recon, label='Inference (CPU)', color=c_cpu, edgecolor='white')

        # GPU bar (right) - only if GPU data available
        if t_inf_gpu is not None:
            ax.bar(x[1], t_acq, width, color=c_acq, edgecolor='white')
            ax.bar(x[1], t_recon, width, bottom=t_acq, color=c_recon, edgecolor='white')
            ax.bar(x[1], t_inf_gpu, width, bottom=t_acq + t_recon, label='Inference (GPU)', color=c_gpu, edgecolor='white')

        # Configure axes
        ax.set_ylabel('Latency (ms)')
        ax.set_xticks(x)
        ax.set_xticklabels(['CPU', 'GPU'] if t_inf_gpu is not None else ['CPU', ''])
        # Place legend outside the plot area (below)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

        # Use subplots_adjust to leave room for legend below
        self.stacked_bar_figure.subplots_adjust(bottom=0.22)
        self.stacked_bar_canvas.draw()

    def _on_generate_report(self):
        """Generate and show the timing report popup."""
        if not self._has_data:
            self.logger.warning("No data available for report generation")
            return

        from ui.custom_widgets.timing_analysis.timing_report_popup import TimingReportPopup

        popup = TimingReportPopup(self, logger=self.logger)
        popup.set_data(self._timing_data)
        popup.exec()

        self.logger.info("Timing report displayed")

    def _show_save_menu(self, pos, widget):
        """Show context menu to save the stacked bar chart."""
        menu = QMenu(self)
        save_png = menu.addAction("Save as PNG...")
        save_pdf = menu.addAction("Save as PDF...")

        action = menu.exec(widget.mapToGlobal(pos))

        if action == save_png:
            self._save_figure("png")
        elif action == save_pdf:
            self._save_figure("pdf")

    def _save_figure(self, ext):
        """Save the stacked bar chart to file at original resolution."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Chart", f"pipeline_breakdown.{ext}",
            f"{ext.upper()} Files (*.{ext});;All Files (*.*)"
        )

        if file_path:
            try:
                # Save at high resolution (300 DPI)
                self.stacked_bar_figure.savefig(file_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Chart saved to {file_path}")
            except Exception as e:
                self.logger.error(f"Failed to save chart: {e}")
