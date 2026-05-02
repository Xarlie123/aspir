"""Energy Analysis page widget with configuration and results visualization."""
import logging
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QGridLayout, QSizePolicy, QFrame, QSplitter,
    QScrollArea, QFormLayout, QSpinBox, QCheckBox,
    QMenu, QFileDialog, QTextEdit, QApplication
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False


def _shorten_backend_name(name: str, max_length: int = 25) -> str:
    """
    Shorten a backend name for display in legends.

    Removes common suffixes like "with Radeon Graphics" and truncates if needed.
    """
    # Remove "with ..." suffix (common in AMD CPUs)
    if " with " in name:
        name = name.split(" with ")[0]

    # Truncate if still too long
    if len(name) > max_length:
        return name[:max_length - 3] + "..."
    return name


class EnergyAnalysisPage(QWidget):
    """
    Energy Analysis page with two main sections:
    - Left: Configuration controls (backend detection, parameters)
    - Right: Results preview (energy chart, summary table)
    """

    # Signal emitted when analysis is requested
    analysisRequested = pyqtSignal()

    def __init__(self, parent=None, logger=None):
        super().__init__(parent)

        if logger:
            self.logger = logger.getChild("EnergyAnalysisPage")
        else:
            self.logger = logging.getLogger("ASPIR.EnergyAnalysisPage")

        # Data storage
        self._energy_data = {}
        self._has_data = False

        # Platform info (will be set by handler)
        self._platform_info = {}
        self._available_backends = []

        self._setup_ui()

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
        splitter.setSizes([400, 650])

        main_layout.addWidget(splitter)

    def _create_config_section(self):
        """Create the left section for configuration controls."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)

        # Title
        title = QLabel("<h3>Energy Configuration</h3>")
        layout.addWidget(title)

        # Platform Detection group
        platform_group = QGroupBox("Platform Detection")
        platform_layout = QVBoxLayout(platform_group)
        platform_layout.setSpacing(8)

        # Detect button
        self.detect_button = QPushButton("Detect Energy Backends")
        self.detect_button.setMinimumHeight(32)
        self.detect_button.setStyleSheet("""
            QPushButton {
                background-color: #607D8B;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #546E7A;
            }
        """)
        self.detect_button.clicked.connect(self._on_detect_clicked)
        platform_layout.addWidget(self.detect_button)

        # Platform info display
        self.platform_info_text = QTextEdit()
        self.platform_info_text.setReadOnly(True)
        self.platform_info_text.setMaximumHeight(120)
        self.platform_info_text.setPlaceholderText("Click 'Detect Energy Backends' to scan hardware...")
        self.platform_info_text.setStyleSheet("""
            QTextEdit {
                background-color: #F5F5F5;
                border: 1px solid #DDD;
                border-radius: 4px;
                font-family: monospace;
                font-size: 11px;
            }
        """)
        platform_layout.addWidget(self.platform_info_text)

        layout.addWidget(platform_group)

        # Backend selection group
        backends_group = QGroupBox("Energy Backends")
        backends_layout = QVBoxLayout(backends_group)
        backends_layout.setSpacing(8)

        # GPU energy checkbox
        self.gpu_energy_checkbox = QCheckBox("GPU Energy (NVML)")
        self.gpu_energy_checkbox.setChecked(True)
        self.gpu_energy_checkbox.setToolTip(
            "Measure GPU energy consumption.\n"
            "Uses NVML for desktop GPUs or sysfs for Jetson devices."
        )
        backends_layout.addWidget(self.gpu_energy_checkbox)

        # CPU energy checkbox
        self.cpu_energy_checkbox = QCheckBox("CPU Energy (RAPL)")
        self.cpu_energy_checkbox.setChecked(False)
        self.cpu_energy_checkbox.setToolTip(
            "Measure CPU energy consumption via Intel RAPL.\n"
            "Only available on Linux with Intel CPUs."
        )
        backends_layout.addWidget(self.cpu_energy_checkbox)

        layout.addWidget(backends_group)

        # Measurement Parameters group
        params_group = QGroupBox("Measurement Parameters")
        params_layout = QFormLayout(params_group)
        params_layout.setSpacing(10)

        # Warmup runs — 20 to match the unified default across
        # Single Test → Timing, Batch Test, and Re-measure.
        self.warmup_spinbox = QSpinBox()
        self.warmup_spinbox.setMinimum(0)
        self.warmup_spinbox.setMaximum(200)
        self.warmup_spinbox.setValue(20)
        self.warmup_spinbox.setToolTip(
            "Number of warmup iterations before measurement.\n"
            "Ensures CUDA kernels are compiled and GPU is at operating temperature."
        )
        params_layout.addRow("Warmup runs:", self.warmup_spinbox)

        # Measurement runs
        self.measurement_spinbox = QSpinBox()
        self.measurement_spinbox.setMinimum(1)
        self.measurement_spinbox.setMaximum(2000)
        self.measurement_spinbox.setValue(800)
        self.measurement_spinbox.setToolTip(
            "Number of inference runs per measurement.\n"
            "Higher values improve accuracy (GPU energy counters\n"
            "update every ~50ms, so total time should exceed this).\n"
            "Recommended: 500-1000 for fast models."
        )
        params_layout.addRow("Measurement runs:", self.measurement_spinbox)

        layout.addWidget(params_group)

        # Image Selection group
        image_group = QGroupBox("Image Selection")
        image_layout = QVBoxLayout(image_group)
        image_layout.setSpacing(8)

        # Range selection row
        range_layout = QHBoxLayout()
        range_layout.setSpacing(10)

        range_layout.addWidget(QLabel("From:"))
        self.image_start_spinbox = QSpinBox()
        self.image_start_spinbox.setMinimum(0)
        self.image_start_spinbox.setMaximum(9)
        self.image_start_spinbox.setValue(0)
        self.image_start_spinbox.setToolTip("First image index (0-based)")
        self.image_start_spinbox.valueChanged.connect(self._on_image_range_changed)
        range_layout.addWidget(self.image_start_spinbox)

        range_layout.addWidget(QLabel("To:"))
        self.image_end_spinbox = QSpinBox()
        self.image_end_spinbox.setMinimum(0)
        self.image_end_spinbox.setMaximum(9)
        self.image_end_spinbox.setValue(9)
        self.image_end_spinbox.setToolTip("Last image index (inclusive)")
        self.image_end_spinbox.valueChanged.connect(self._on_image_range_changed)
        range_layout.addWidget(self.image_end_spinbox)

        range_layout.addStretch()
        image_layout.addLayout(range_layout)

        # Selection info label
        self.image_selection_label = QLabel("Selected 10 of 10 inference images")
        self.image_selection_label.setStyleSheet("color: #666; font-size: 11px;")
        image_layout.addWidget(self.image_selection_label)

        layout.addWidget(image_group)

        # Store max images for later update
        self._max_test_images = 10

        # Formula explanation
        formula_label = QLabel(
            "<b>E<sub>total</sub></b> = E<sub>recon</sub> + E<sub>inference</sub><br>"
            "<b>P<sub>avg</sub></b> = E<sub>total</sub> / T<sub>total</sub>"
        )
        formula_label.setWordWrap(True)
        formula_label.setStyleSheet(
            "QLabel { background-color: #FFF3E0; padding: 8px; "
            "border-radius: 4px; border: 1px solid #FFE0B2; color: #E65100; }"
        )
        layout.addWidget(formula_label)

        # Run analysis button (green style matching Timing Analysis)
        self.analyze_button = QPushButton("Run Energy Analysis")
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
        self.generate_button = QPushButton("Generate Energy Report")
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

        # Status label
        self.status_label = QLabel("Detect backends and run analysis to see results")
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
        title = QLabel("<h3>Energy Results</h3>")
        layout.addWidget(title)

        # Energy bar chart
        chart_group = QGroupBox("Energy Consumption per Image")
        chart_group.setContextMenuPolicy(Qt.CustomContextMenu)
        chart_group.customContextMenuRequested.connect(
            lambda pos: self._show_save_menu(pos, chart_group, "energy")
        )
        chart_layout = QVBoxLayout(chart_group)

        self.energy_figure = Figure(figsize=(6, 3), dpi=100)
        self.energy_canvas = FigureCanvas(self.energy_figure)
        self.energy_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.energy_canvas.setMinimumHeight(200)
        chart_layout.addWidget(self.energy_canvas)

        layout.addWidget(chart_group)

        # Power chart
        power_group = QGroupBox("Average Power During Inference")
        power_group.setContextMenuPolicy(Qt.CustomContextMenu)
        power_group.customContextMenuRequested.connect(
            lambda pos: self._show_save_menu(pos, power_group, "power")
        )
        power_layout = QVBoxLayout(power_group)

        self.power_figure = Figure(figsize=(6, 2.5), dpi=100)
        self.power_canvas = FigureCanvas(self.power_figure)
        self.power_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.power_canvas.setMinimumHeight(150)
        power_layout.addWidget(self.power_canvas)

        layout.addWidget(power_group)

        # Summary table (dynamic columns per backend)
        self.summary_group = QGroupBox("Energy Summary")
        self.summary_layout = QGridLayout(self.summary_group)
        self.summary_layout.setSpacing(8)

        # Store references to dynamically created labels
        self._summary_backend_labels = {}  # {backend_name: {metric: label}}
        self._summary_header_labels = []
        self._summary_row_labels = []

        # Create initial empty table structure
        self._create_summary_table_structure()

        # Enable right-click context menu for copying
        self.summary_group.setContextMenuPolicy(Qt.CustomContextMenu)
        self.summary_group.customContextMenuRequested.connect(self._show_summary_copy_menu)

        layout.addWidget(self.summary_group)

        layout.addStretch()

        return container

    def _on_detect_clicked(self):
        """Handle detect button click - emits signal to handler."""
        self.status_label.setText("Detecting energy backends...")
        self.status_label.setStyleSheet("color: #FF9800;")
        # The actual detection will be done by the handler

    # --- Properties for accessing configuration ---

    @property
    def enable_gpu_energy(self) -> bool:
        """Return True if GPU energy measurement is enabled."""
        return self.gpu_energy_checkbox.isChecked()

    @property
    def enable_cpu_energy(self) -> bool:
        """Return True if CPU energy measurement is enabled."""
        return self.cpu_energy_checkbox.isChecked()

    @property
    def pmlib_server_ip(self) -> str:
        """Return PMLib server IP (deprecated, returns default)."""
        return "127.0.0.1"

    @property
    def pmlib_server_port(self) -> int:
        """Return PMLib server port (deprecated, returns default)."""
        return 6526

    @property
    def warmup_runs(self) -> int:
        """Return the number of warmup runs."""
        return self.warmup_spinbox.value()

    @property
    def measurement_runs(self) -> int:
        """Return the number of measurement runs."""
        return self.measurement_spinbox.value()

    @property
    def image_start_index(self) -> int:
        """Return the start index for image selection."""
        return self.image_start_spinbox.value()

    @property
    def image_end_index(self) -> int:
        """Return the end index for image selection (exclusive, for Python slicing)."""
        # UI uses inclusive indexing, but return exclusive for slice: images[start:end]
        return self.image_end_spinbox.value() + 1

    def set_max_test_images(self, count: int):
        """
        Update the maximum number of available test images.

        Args:
            count: Total number of test images available
        """
        self._max_test_images = count
        max_idx = max(0, count - 1)
        self.image_start_spinbox.setMaximum(max_idx)
        self.image_end_spinbox.setMaximum(max_idx)
        # Auto-adjust end if exceeds new max
        if self.image_end_spinbox.value() > max_idx:
            self.image_end_spinbox.setValue(max_idx)
        self._on_image_range_changed()

    def _on_image_range_changed(self):
        """Handle changes to image range selection."""
        start = self.image_start_spinbox.value()
        end = self.image_end_spinbox.value()

        # Ensure start <= end (allow single image selection)
        if start > end:
            if self.sender() == self.image_start_spinbox:
                self.image_end_spinbox.setValue(start)
                end = start
            else:
                self.image_start_spinbox.setValue(end)
                start = end

        count = end - start + 1  # Inclusive range
        self.image_selection_label.setText(
            f"Selected {count} of {self._max_test_images} inference images (index {start} to {end})"
        )

    # --- Data methods ---

    def set_platform_info(self, info: dict, backends: list):
        """
        Set platform detection information.

        Args:
            info: Dictionary with platform information
            backends: List of available backend names
        """
        self._platform_info = info
        self._available_backends = backends

        # Format info text
        lines = []
        lines.append(f"Platform: {info.get('platform', 'Unknown')}")

        if info.get('is_jetson'):
            lines.append(f"Jetson: {info.get('jetson_type', 'Yes')}")
        elif info.get('has_nvidia_gpu'):
            lines.append(f"GPU: {info.get('gpu_name', 'NVIDIA GPU')}")
        else:
            lines.append("GPU: Not detected")

        if info.get('has_rapl'):
            domains = info.get('rapl_domains', [])
            # Check if RAPL backend is actually active (has permissions)
            # RAPL works on both Intel and AMD CPUs via Linux kernel abstraction
            rapl_active = any(
                'cpu' in b.lower() or 'intel' in b.lower() or
                'amd' in b.lower() or 'ryzen' in b.lower() or 'core' in b.lower()
                for b in backends
            )
            if rapl_active:
                lines.append(f"RAPL: Active ({', '.join(domains[:3])})")
            else:
                lines.append(f"RAPL: No permission (run: sudo chmod +r /sys/class/powercap/intel-rapl/*/energy_uj)")
        else:
            lines.append("RAPL: Not available")

        lines.append("")
        lines.append(f"Active backends: {', '.join(backends) if backends else 'None'}")

        self.platform_info_text.setText("\n".join(lines))

        # Update checkbox states based on availability
        has_gpu = info.get('has_nvidia_gpu') or info.get('is_jetson')
        self.gpu_energy_checkbox.setEnabled(has_gpu)
        if not has_gpu:
            self.gpu_energy_checkbox.setChecked(False)

        # Only enable CPU checkbox if RAPL is actually accessible (in backends)
        # RAPL works on both Intel and AMD CPUs
        rapl_accessible = any(
            'cpu' in b.lower() or 'intel' in b.lower() or
            'amd' in b.lower() or 'ryzen' in b.lower() or 'core' in b.lower()
            for b in backends
        )
        self.cpu_energy_checkbox.setEnabled(rapl_accessible)
        self.cpu_energy_checkbox.setChecked(rapl_accessible)  # Auto-enable if available
        if not rapl_accessible:
            if info.get('has_rapl'):
                self.cpu_energy_checkbox.setToolTip(
                    "RAPL detected but no read permission.\n"
                    "Run: sudo chmod +r /sys/class/powercap/intel-rapl/*/energy_uj"
                )

        if backends:
            self.status_label.setText(f"Ready: {len(backends)} backend(s) available")
            self.status_label.setStyleSheet("color: #4CAF50;")
        else:
            self.status_label.setText("Warning: No energy backends available")
            self.status_label.setStyleSheet("color: #F44336;")

    def set_data(self, energy_data: dict):
        """
        Set the energy data and update the display.

        Args:
            energy_data: Dictionary with energy values:
                - energy_per_image_mj: list of energy per image in mJ
                - power_per_image_watts: list of power per image in W
                - time_per_image_ms: list of time per image in ms
                - mean_energy_mj: mean energy in mJ
                - std_energy_mj: std of energy in mJ
                - mean_power_watts: mean power in W
                - mean_time_ms: mean time in ms
                - device_name: name of the device
                - efficiency_images_per_joule: processing efficiency
                - mean_temperature: mean temperature (optional)
        """
        self._energy_data = energy_data
        self._has_data = True

        self._update_summary_table()
        self._update_energy_chart()
        self._update_power_chart()

        self.generate_button.setEnabled(True)
        self.status_label.setText("Analysis complete")
        self.status_label.setStyleSheet("color: #4CAF50;")

        self.logger.debug("Energy data set and display updated")

    def _create_summary_table_structure(self):
        """Create the initial summary table structure with row labels."""
        header_font = QFont()
        header_font.setBold(True)

        # Row definitions: (row_index, label_text, metric_key, unit, value_style)
        self._summary_rows = [
            (1, "Device:", "device", "", None),
            (2, "Energy/image:", "energy", "mJ", "font-weight: bold; color: #FF9800;"),
            (3, "Avg Power:", "power", "W", "font-weight: bold; color: #4CAF50;"),
            (4, "Avg Time:", "time", "ms", None),
            (5, "Efficiency:", "efficiency", "img/J", "font-weight: bold; color: #2196F3;"),
            (6, "Temperature:", "temperature", "°C", None),
        ]

        # Column 0: Row labels (Metric names)
        metric_header = QLabel("Metric")
        metric_header.setFont(header_font)
        metric_header.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.summary_layout.addWidget(metric_header, 0, 0)

        for row_idx, label_text, _, _, _ in self._summary_rows:
            row_label = QLabel(label_text)
            row_label.setFont(header_font)
            self.summary_layout.addWidget(row_label, row_idx, 0)
            self._summary_row_labels.append(row_label)

    def _clear_summary_backend_columns(self):
        """Remove all backend columns from the summary table."""
        # Remove header labels
        for label in self._summary_header_labels:
            self.summary_layout.removeWidget(label)
            label.deleteLater()
        self._summary_header_labels.clear()

        # Remove value labels
        for backend_labels in self._summary_backend_labels.values():
            for label in backend_labels.values():
                self.summary_layout.removeWidget(label)
                label.deleteLater()
        self._summary_backend_labels.clear()

    def _update_summary_table(self):
        """Update the summary table with energy values, one column per backend."""
        if not self._energy_data:
            return

        # Clear existing backend columns
        self._clear_summary_backend_columns()

        backends = self._energy_data.get('backends', {})
        header_font = QFont()
        header_font.setBold(True)

        if not backends:
            # Fallback: single column with legacy data
            backends = {
                self._energy_data.get('device_name', 'Device'): {
                    'type': 'gpu',
                    'device_name': self._energy_data.get('device_name', 'Unknown'),
                    'mean_energy_mj': self._energy_data.get('mean_energy_mj', 0),
                    'std_energy_mj': self._energy_data.get('std_energy_mj', 0),
                    'mean_power_watts': self._energy_data.get('mean_power_watts', 0),
                    'mean_time_ms': self._energy_data.get('mean_time_ms', 0),
                    'efficiency_images_per_joule': self._energy_data.get('efficiency_images_per_joule', 0),
                    'mean_temperature': self._energy_data.get('mean_temperature'),
                }
            }

        # Create columns for each backend
        col_idx = 1
        for backend_name, data in backends.items():
            backend_type = data.get('type', 'gpu')

            # Header with short name
            short_name = _shorten_backend_name(backend_name, 20)
            type_label = "GPU" if backend_type == 'gpu' else "CPU"
            header_text = f"{type_label}\n({short_name})"

            header_label = QLabel(header_text)
            header_label.setFont(header_font)
            header_label.setAlignment(Qt.AlignCenter)
            header_label.setToolTip(backend_name)  # Full name on hover
            if backend_type == 'gpu':
                header_label.setStyleSheet("color: #FF9800;")
            else:
                header_label.setStyleSheet("color: #2196F3;")
            self.summary_layout.addWidget(header_label, 0, col_idx)
            self._summary_header_labels.append(header_label)

            # Create value labels for this backend
            backend_labels = {}

            for row_idx, _, metric_key, _, value_style in self._summary_rows:
                value_label = QLabel("-")
                value_label.setAlignment(Qt.AlignCenter)
                if value_style:
                    value_label.setStyleSheet(value_style)
                self.summary_layout.addWidget(value_label, row_idx, col_idx)
                backend_labels[metric_key] = value_label

            # Fill in values
            backend_labels["device"].setText(short_name)

            mean_energy = data.get('mean_energy_mj', 0)
            std_energy = data.get('std_energy_mj', 0)
            if std_energy > 0:
                backend_labels["energy"].setText(f"{mean_energy:.2f} ± {std_energy:.2f}")
            else:
                backend_labels["energy"].setText(f"{mean_energy:.2f}")

            mean_power = data.get('mean_power_watts', 0)
            backend_labels["power"].setText(f"{mean_power:.2f}")

            mean_time = data.get('mean_time_ms', 0)
            backend_labels["time"].setText(f"{mean_time:.2f}")

            efficiency = data.get('efficiency_images_per_joule', 0)
            backend_labels["efficiency"].setText(f"{efficiency:.1f}")

            mean_temp = data.get('mean_temperature')
            if mean_temp is not None:
                backend_labels["temperature"].setText(f"{mean_temp:.1f}")
            else:
                backend_labels["temperature"].setText("-")

            self._summary_backend_labels[backend_name] = backend_labels
            col_idx += 1

        # Add Unit column at the end
        unit_header = QLabel("Unit")
        unit_header.setFont(header_font)
        unit_header.setAlignment(Qt.AlignCenter)
        self.summary_layout.addWidget(unit_header, 0, col_idx)
        self._summary_header_labels.append(unit_header)

        for row_idx, _, _, unit, _ in self._summary_rows:
            unit_label = QLabel(unit)
            unit_label.setAlignment(Qt.AlignCenter)
            unit_label.setStyleSheet("color: #666;")
            self.summary_layout.addWidget(unit_label, row_idx, col_idx)
            self._summary_header_labels.append(unit_label)

    def _update_energy_chart(self):
        """Update the energy per image bar chart with per-backend breakdown."""
        self.energy_figure.clear()

        if not self._energy_data:
            ax = self.energy_figure.add_subplot(111)
            ax.text(0.5, 0.5, "No data available", ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
            self.energy_canvas.draw()
            return

        backends = self._energy_data.get('backends', {})
        n_images = self._energy_data.get('n_images', 0)

        if not backends or n_images == 0:
            # Fallback to old behavior
            energy_values = self._energy_data.get('energy_per_image_mj', [])
            if not energy_values:
                return
            ax = self.energy_figure.add_subplot(111)
            x = np.arange(len(energy_values))
            ax.bar(x, energy_values, color='#FF9800', edgecolor='white', alpha=0.8)
            mean_energy = self._energy_data.get('mean_energy_mj', 0)
            ax.axhline(y=mean_energy, color='#E65100', linestyle='--', linewidth=2,
                       label=f'Mean: {mean_energy:.2f} mJ')
            ax.set_xlabel('Image Index')
            ax.set_ylabel('Energy (mJ)')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')
            self.energy_figure.tight_layout()
            self.energy_canvas.draw()
            return

        ax = self.energy_figure.add_subplot(111)
        x = np.arange(n_images)

        # Colors for GPU (orange) and CPU (blue)
        colors = {'gpu': '#FF9800', 'cpu': '#2196F3'}
        mean_colors = {'gpu': '#E65100', 'cpu': '#1565C0'}

        n_backends = len(backends)
        width = 0.8 / n_backends if n_backends > 1 else 0.8

        for i, (backend_name, data) in enumerate(backends.items()):
            backend_type = data.get('type', 'gpu')
            color = colors.get(backend_type, '#9E9E9E')
            mean_color = mean_colors.get(backend_type, '#616161')

            energy_values = data.get('energy_per_image_mj', [])
            if not energy_values:
                continue

            offset = (i - n_backends / 2 + 0.5) * width if n_backends > 1 else 0
            short_name = 'GPU' if backend_type == 'gpu' else 'CPU'

            ax.bar(x + offset, energy_values, width, color=color, edgecolor='white',
                   alpha=0.8, label=f'{short_name}: {_shorten_backend_name(backend_name)}')

            # Add mean line for this backend
            mean_energy = data.get('mean_energy_mj', 0)
            ax.axhline(y=mean_energy, color=mean_color, linestyle='--', linewidth=1.5,
                       label=f'{short_name} Mean: {mean_energy:.2f} mJ')

        ax.set_xlabel('Image Index')
        ax.set_ylabel('Energy (mJ)')
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.3, axis='y')

        self.energy_figure.tight_layout()
        self.energy_canvas.draw()

    def _update_power_chart(self):
        """Update the power per image bar chart with per-backend breakdown."""
        self.power_figure.clear()

        if not self._energy_data:
            ax = self.power_figure.add_subplot(111)
            ax.text(0.5, 0.5, "No data available", ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
            self.power_canvas.draw()
            return

        backends = self._energy_data.get('backends', {})
        n_images = self._energy_data.get('n_images', 0)

        if not backends or n_images == 0:
            # Fallback to old behavior
            power_values = self._energy_data.get('power_per_image_watts', [])
            if not power_values:
                return
            ax = self.power_figure.add_subplot(111)
            x = np.arange(len(power_values))
            ax.bar(x, power_values, color='#4CAF50', edgecolor='white', alpha=0.8)
            mean_power = self._energy_data.get('mean_power_watts', 0)
            ax.axhline(y=mean_power, color='#2E7D32', linestyle='--', linewidth=2,
                       label=f'Mean: {mean_power:.2f} W')
            ax.set_xlabel('Image Index')
            ax.set_ylabel('Power (W)')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')
            self.power_figure.tight_layout()
            self.power_canvas.draw()
            return

        ax = self.power_figure.add_subplot(111)
        x = np.arange(n_images)

        # Colors for GPU (orange/red tones) and CPU (green/teal tones)
        colors = {'gpu': '#FF5722', 'cpu': '#009688'}
        mean_colors = {'gpu': '#BF360C', 'cpu': '#00695C'}

        n_backends = len(backends)
        width = 0.8 / n_backends if n_backends > 1 else 0.8

        for i, (backend_name, data) in enumerate(backends.items()):
            backend_type = data.get('type', 'gpu')
            color = colors.get(backend_type, '#9E9E9E')
            mean_color = mean_colors.get(backend_type, '#616161')

            power_values = data.get('power_per_image_watts', [])
            if not power_values:
                continue

            offset = (i - n_backends / 2 + 0.5) * width if n_backends > 1 else 0
            short_name = 'GPU' if backend_type == 'gpu' else 'CPU'

            ax.bar(x + offset, power_values, width, color=color, edgecolor='white',
                   alpha=0.8, label=f'{short_name}: {_shorten_backend_name(backend_name)}')

            # Add mean line for this backend
            mean_power = data.get('mean_power_watts', 0)
            ax.axhline(y=mean_power, color=mean_color, linestyle='--', linewidth=1.5,
                       label=f'{short_name} Mean: {mean_power:.2f} W')

        ax.set_xlabel('Image Index')
        ax.set_ylabel('Power (W)')
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.3, axis='y')

        self.power_figure.tight_layout()
        self.power_canvas.draw()

    def _on_generate_report(self):
        """Generate and show the energy report popup."""
        if not self._has_data:
            self.logger.warning("No data available for report generation")
            return

        from ui.custom_widgets.energy_analysis.energy_report_popup import EnergyReportPopup

        popup = EnergyReportPopup(parent=self, logger=self.logger)
        popup.set_data(self._energy_data)
        popup.exec_()

        self.logger.info("Energy report displayed")

    def _show_save_menu(self, pos, widget, chart_type):
        """Show context menu to save the chart."""
        menu = QMenu(self)
        save_png = menu.addAction("Save as PNG...")
        save_pdf = menu.addAction("Save as PDF...")

        action = menu.exec_(widget.mapToGlobal(pos))

        if action == save_png:
            self._save_figure(chart_type, "png")
        elif action == save_pdf:
            self._save_figure(chart_type, "pdf")

    def _save_figure(self, chart_type, ext):
        """Save the specified chart to file."""
        figure = self.energy_figure if chart_type == "energy" else self.power_figure
        default_name = f"energy_{chart_type}.{ext}"

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Chart", default_name,
            f"{ext.upper()} Files (*.{ext});;All Files (*.*)"
        )

        if file_path:
            try:
                figure.savefig(file_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Chart saved to {file_path}")
            except Exception as e:
                self.logger.error(f"Failed to save chart: {e}")

    def _show_summary_copy_menu(self, pos):
        """Show context menu to copy the summary table."""
        menu = QMenu(self)
        copy_action = menu.addAction("Copy table")
        action = menu.exec_(self.summary_group.mapToGlobal(pos))

        if action == copy_action:
            self._copy_summary_table()

    def _copy_summary_table(self):
        """Copy the energy summary table to clipboard as tab-separated values."""
        if not self._energy_data or not self._summary_backend_labels:
            return

        lines = []

        # Header row: Metric + backend names + Unit
        backend_names = list(self._summary_backend_labels.keys())
        header = "Metric\t" + "\t".join(_shorten_backend_name(n, 20) for n in backend_names) + "\tUnit"
        lines.append(header)

        # Data rows
        for row_idx, label_text, metric_key, unit, _ in self._summary_rows:
            row_values = [label_text.rstrip(":")]
            for backend_name in backend_names:
                backend_labels = self._summary_backend_labels.get(backend_name, {})
                label = backend_labels.get(metric_key)
                row_values.append(label.text() if label else "-")
            row_values.append(unit)
            lines.append("\t".join(row_values))

        text = "\n".join(lines)
        QApplication.clipboard().setText(text)
        self.logger.info("Energy summary table copied to clipboard")
