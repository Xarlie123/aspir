"""UI builder for :class:`EnergyView` — attaches widgets to the view instance."""
from __future__ import annotations

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ui.custom_widgets.batch_reports.comparison_views.chart_config_popup import (
    CustomNavigationToolbar,
)
from ui.custom_widgets.batch_reports.comparison_views.energy_view._summary_table import (
    create_summary_table_structure,
)


def build_ui(view):
    """Setup the energy view UI with left menu and attach widgets to ``view``."""
    main_layout = QHBoxLayout(view)
    main_layout.setContentsMargins(5, 5, 5, 5)
    main_layout.setSpacing(5)

    splitter = QSplitter(Qt.Horizontal)

    # Left panel: Menu and buttons
    left_panel = QWidget()
    left_layout = QVBoxLayout(left_panel)
    left_layout.setContentsMargins(5, 5, 5, 5)
    left_layout.setSpacing(10)

    # Chart type list
    chart_label = QLabel("Chart Type:")
    chart_label.setStyleSheet("font-weight: bold; color: #333;")
    left_layout.addWidget(chart_label)

    view.chart_list = QListWidget()
    view.chart_list.setMaximumWidth(220)
    view.chart_list.setMinimumWidth(180)
    view.chart_list.addItems([
        "Energy Bar Chart",
        "Power Comparison",
        "Efficiency (img/J)",
        "Distribution (Box Plot)"
    ])
    view.chart_list.setCurrentRow(0)
    view.chart_list.currentRowChanged.connect(view._on_chart_type_changed)
    view.chart_list.setStyleSheet("""
        QListWidget {
            border: 1px solid #ccc;
            border-radius: 4px;
            background-color: #f5f5f5;
        }
        QListWidget::item {
            padding: 10px;
            border-bottom: 1px solid #e0e0e0;
        }
        QListWidget::item:selected {
            background-color: #0078d7;
            color: white;
        }
        QListWidget::item:hover {
            background-color: #e5e5e5;
        }
    """)
    left_layout.addWidget(view.chart_list)

    # Compute-path selector: "which device ran the inference?", not
    # "which energy sensor produced the reading". The per-rail
    # breakdown (RAPL / NVML / jtop) is determined by the host and the
    # user has no useful choice there; what they want to compare is
    # the cost of running on CPU vs GPU.
    backend_label = QLabel("Compute path:")
    backend_label.setStyleSheet("font-weight: bold; color: #333;")
    left_layout.addWidget(backend_label)

    view.backend_combo = QComboBox()
    view.backend_combo.setMaximumWidth(220)
    view.backend_combo.setMinimumWidth(180)
    view.backend_combo.addItems([view.BACKEND_ALL, view.BACKEND_CPU, view.BACKEND_GPU])
    view.backend_combo.setCurrentText(view.BACKEND_ALL)
    view.backend_combo.currentTextChanged.connect(view._on_backend_changed)
    view.backend_combo.setToolTip(
        "Group bars by which compute path ran the inference:\n"
        "- CPU run + GPU run: side-by-side bars per test\n"
        "- CPU run only: hide tests measured on GPU\n"
        "- GPU run only: hide tests measured on CPU\n"
        "\n"
        "Tests are paired by name; load the CPU and GPU re-measurement\n"
        "reports together (or use 'Run both compute paths' on Re-measure)\n"
        "to populate both bars."
    )
    left_layout.addWidget(view.backend_combo)

    # Subtract-idle-baseline toggle. Enabled only when at least one
    # loaded experiment has a captured baseline in metadata; otherwise
    # the toggle stays disabled so the user doesn't get silent zeroes.
    from PyQt5.QtWidgets import QCheckBox
    view.baseline_check = QCheckBox("Subtract idle baseline")
    view.baseline_check.setMaximumWidth(220)
    view.baseline_check.setMinimumWidth(180)
    view.baseline_check.setChecked(False)
    view.baseline_check.setEnabled(False)
    view.baseline_check.setToolTip(
        "Plot dynamic power / energy = total − idle baseline.\n"
        "Enabled only when at least one loaded experiment has a\n"
        "baseline captured (run a Batch Test or Re-measure with\n"
        "'Capture idle baseline' enabled to populate it)."
    )
    view.baseline_check.toggled.connect(view._on_baseline_toggle)
    left_layout.addWidget(view.baseline_check)

    # Generate Energy Report button
    view.report_btn = QPushButton("Generate Energy Report")
    view.report_btn.setMaximumWidth(220)
    view.report_btn.setMinimumWidth(180)
    view.report_btn.setMinimumHeight(40)
    view.report_btn.setEnabled(False)
    view.report_btn.setStyleSheet("""
        QPushButton {
            background-color: #0078d7;
            color: white;
            border: none;
            border-radius: 4px;
            font-weight: bold;
            font-size: 14px;
            padding: 8px;
        }
        QPushButton:hover:enabled {
            background-color: #005a9e;
        }
        QPushButton:pressed:enabled {
            background-color: #004275;
        }
        QPushButton:disabled {
            background-color: #ccc;
            color: #888;
        }
    """)
    view.report_btn.setToolTip("Generate detailed energy analysis report")
    view.report_btn.clicked.connect(view._on_generate_report)
    left_layout.addWidget(view.report_btn)

    left_layout.addStretch()

    splitter.addWidget(left_panel)

    # Right panel: Results area
    right_panel = QWidget()
    right_layout = QVBoxLayout(right_panel)
    right_layout.setContentsMargins(5, 5, 5, 5)
    right_layout.setSpacing(5)

    # Chart area
    view.figure = Figure(figsize=(10, 5), dpi=100)
    view.canvas = FigureCanvas(view.figure)
    view.canvas.setStyleSheet("background-color: white;")

    view.canvas.setContextMenuPolicy(Qt.CustomContextMenu)
    view.canvas.customContextMenuRequested.connect(view._show_context_menu)

    view.toolbar = CustomNavigationToolbar(
        view.canvas, view,
        config_callback=view._on_open_chart_config
    )

    right_layout.addWidget(view.toolbar)
    right_layout.addWidget(view.canvas, 1)

    # Energy Summary table (dynamic columns per backend)
    view.summary_group = QGroupBox("Energy Summary")
    view.summary_group.setStyleSheet("""
        QGroupBox {
            font-weight: bold;
            border: 1px solid #ccc;
            border-radius: 4px;
            margin-top: 10px;
            padding-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 5px;
        }
    """)
    view.summary_layout = QGridLayout(view.summary_group)
    view.summary_layout.setSpacing(8)

    # Test selector for summary table (like Timing view)
    selector_layout = QHBoxLayout()
    test_label = QLabel("Show details for:")
    test_label.setStyleSheet("font-weight: bold;")
    selector_layout.addWidget(test_label)

    view.test_combo = QComboBox()
    view.test_combo.setMinimumWidth(200)
    view.test_combo.currentIndexChanged.connect(view._on_test_changed)
    selector_layout.addWidget(view.test_combo)
    selector_layout.addStretch()
    view.summary_layout.addLayout(selector_layout, 0, 0, 1, 4)

    # Storage for dynamic labels
    view._summary_backend_labels = {}
    view._summary_header_labels = []
    view._summary_row_labels = []

    # Create initial table structure
    create_summary_table_structure(view)

    # Enable right-click for copy
    view.summary_group.setContextMenuPolicy(Qt.CustomContextMenu)
    view.summary_group.customContextMenuRequested.connect(view._show_summary_copy_menu)

    right_layout.addWidget(view.summary_group)

    # Info label
    view.info_label = QLabel("Load experiments to see energy analysis")
    view.info_label.setStyleSheet("color: #666; font-size: 11px;")
    view.info_label.setAlignment(Qt.AlignCenter)
    right_layout.addWidget(view.info_label)

    splitter.addWidget(right_panel)
    splitter.setSizes([180, 720])

    main_layout.addWidget(splitter)
