"""UI builders for :class:`BatchTimingReportPopup`.

``build_ui`` sets up the top-level layout with tabs and export buttons; the
two tab builders create all widgets and attach them to the popup so its
methods can reference them.
"""
from __future__ import annotations

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ui.custom_widgets.batch_reports.comparison_views.chart_config_popup import (
    CustomNavigationToolbar,
)


def build_ui(popup):
    """Build the top-level dialog layout and attach widgets to ``popup``."""
    main_layout = QVBoxLayout(popup)
    main_layout.setContentsMargins(10, 10, 10, 10)
    main_layout.setSpacing(10)

    # Title and test selector row
    header_layout = QHBoxLayout()

    title = QLabel("<h2>Detailed Timing Report</h2>")
    header_layout.addWidget(title)

    header_layout.addStretch()

    # Test selector (only show if multiple tests)
    if len(popup._tests) > 1:
        test_label = QLabel("Test:")
        test_label.setStyleSheet("font-weight: bold;")
        header_layout.addWidget(test_label)

        popup.test_combo = QComboBox()
        popup.test_combo.setMinimumWidth(250)
        for test in popup._tests:
            test_name = test.get("name", "Unknown")
            exp_name = test.get("_experiment_name", "")
            if exp_name:
                popup.test_combo.addItem(f"{test_name} ({exp_name})")
            else:
                popup.test_combo.addItem(test_name)
        popup.test_combo.setCurrentIndex(popup._current_test_idx)
        popup.test_combo.currentIndexChanged.connect(popup._on_test_changed)
        header_layout.addWidget(popup.test_combo)
    else:
        popup.test_combo = None

    main_layout.addLayout(header_layout)

    # Tabs
    popup.tabs = QTabWidget()
    popup.tabs.setStyleSheet("""
        QTabWidget::pane {
            border: 1px solid #ccc;
            border-radius: 4px;
            background-color: white;
        }
        QTabBar::tab {
            padding: 8px 16px;
            margin-right: 2px;
        }
        QTabBar::tab:selected {
            background-color: #0078d7;
            color: white;
        }
    """)

    # Tab 1: Timing Report
    popup.timing_tab = build_timing_tab(popup)
    popup.tabs.addTab(popup.timing_tab, "Timing Report")

    # Tab 2: PyTorch Profiler
    popup.profiler_tab = build_profiler_tab(popup)
    popup.tabs.addTab(popup.profiler_tab, "PyTorch Profiler")

    main_layout.addWidget(popup.tabs, 1)

    # Buttons
    buttons_layout = QHBoxLayout()
    buttons_layout.addStretch()

    popup.export_pdf_btn = QPushButton("Export PDF")
    popup.export_pdf_btn.clicked.connect(lambda: popup._on_export("pdf"))
    buttons_layout.addWidget(popup.export_pdf_btn)

    popup.export_png_btn = QPushButton("Export PNG")
    popup.export_png_btn.clicked.connect(lambda: popup._on_export("png"))
    buttons_layout.addWidget(popup.export_png_btn)

    popup.close_btn = QPushButton("Close")
    popup.close_btn.clicked.connect(popup.close)
    buttons_layout.addWidget(popup.close_btn)

    main_layout.addLayout(buttons_layout)


def build_timing_tab(popup) -> QWidget:
    """Create the timing report tab."""
    widget = QWidget()
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(5, 5, 5, 5)
    layout.setSpacing(10)

    # Scroll area for content
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.NoFrame)

    content_widget = QWidget()
    content_layout = QVBoxLayout(content_widget)
    content_layout.setSpacing(15)

    # Top row: Time per image + Distribution histograms
    top_row = QHBoxLayout()
    top_row.setSpacing(15)

    # Time per image curves
    curves_group = QGroupBox("Time per Image")
    curves_group.setContextMenuPolicy(Qt.CustomContextMenu)
    curves_group.customContextMenuRequested.connect(
        lambda pos: popup._show_save_menu(pos, curves_group, popup.curves_figure, "time_per_image")
    )
    curves_layout = QVBoxLayout(curves_group)
    popup.curves_figure = Figure(figsize=(5, 4), dpi=100)
    popup.curves_canvas = FigureCanvas(popup.curves_figure)
    popup.curves_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    # Navigation toolbar with chart config button
    popup.curves_toolbar = CustomNavigationToolbar(
        popup.curves_canvas, popup,
        config_callback=popup._on_open_chart_config
    )
    curves_layout.addWidget(popup.curves_toolbar)
    curves_layout.addWidget(popup.curves_canvas)
    top_row.addWidget(curves_group)

    # Distribution histograms
    hist_group = QGroupBox("Time Distribution")
    hist_group.setContextMenuPolicy(Qt.CustomContextMenu)
    hist_group.customContextMenuRequested.connect(
        lambda pos: popup._show_save_menu(pos, hist_group, popup.hist_figure, "time_distribution")
    )
    hist_layout = QVBoxLayout(hist_group)
    popup.hist_figure = Figure(figsize=(5, 4), dpi=100)
    popup.hist_canvas = FigureCanvas(popup.hist_figure)
    popup.hist_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    hist_layout.addWidget(popup.hist_canvas)
    top_row.addWidget(hist_group)

    content_layout.addLayout(top_row)

    # Bottom row: Stacked bar + Statistics table
    bottom_row = QHBoxLayout()
    bottom_row.setSpacing(15)

    # Stacked bar chart
    bar_group = QGroupBox("Pipeline Latency Breakdown")
    bar_group.setContextMenuPolicy(Qt.CustomContextMenu)
    bar_group.customContextMenuRequested.connect(
        lambda pos: popup._show_save_menu(pos, bar_group, popup.bar_figure, "pipeline_breakdown")
    )
    bar_layout = QVBoxLayout(bar_group)
    popup.bar_figure = Figure(figsize=(5, 4), dpi=100)
    popup.bar_canvas = FigureCanvas(popup.bar_figure)
    popup.bar_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    bar_layout.addWidget(popup.bar_canvas)
    bottom_row.addWidget(bar_group)

    # Statistics table
    stats_group = QGroupBox("Detailed Statistics")
    stats_layout = QGridLayout(stats_group)
    stats_layout.setSpacing(8)

    header_font = QFont()
    header_font.setBold(True)

    # Headers
    headers = ["", "Mean", "Std", "Min", "Max", "P25", "P50", "P75"]
    for col, h in enumerate(headers):
        label = QLabel(h)
        label.setFont(header_font)
        label.setAlignment(Qt.AlignCenter)
        stats_layout.addWidget(label, 0, col)

    # Rows for each timing component
    popup.stats_labels = {}
    rows = [
        ("T_reconstruction", "t_recon"),
        ("T_inference (CPU)", "t_inf_cpu"),
        ("T_inference (GPU)", "t_inf_gpu"),
        ("T_total (CPU)", "t_total_cpu"),
        ("T_total (GPU)", "t_total_gpu")
    ]

    for row_idx, (label_text, key) in enumerate(rows, start=1):
        row_label = QLabel(label_text)
        row_label.setFont(header_font)
        stats_layout.addWidget(row_label, row_idx, 0)

        popup.stats_labels[key] = []
        for col in range(1, 8):
            val_label = QLabel("-")
            val_label.setAlignment(Qt.AlignCenter)
            stats_layout.addWidget(val_label, row_idx, col)
            popup.stats_labels[key].append(val_label)

    bottom_row.addWidget(stats_group)

    content_layout.addLayout(bottom_row)

    scroll.setWidget(content_widget)
    layout.addWidget(scroll, 1)

    return widget


def build_profiler_tab(popup) -> QWidget:
    """Create the PyTorch profiler tab."""
    widget = QWidget()
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(10, 10, 10, 10)
    layout.setSpacing(10)

    # Device selector (CPU/GPU) at the top
    device_selector_layout = QHBoxLayout()
    device_label = QLabel("Show profiler data for:")
    device_label.setStyleSheet("font-weight: bold;")
    device_selector_layout.addWidget(device_label)

    popup.profiler_device_combo = QComboBox()
    popup.profiler_device_combo.setMinimumWidth(150)
    popup.profiler_device_combo.currentIndexChanged.connect(popup._on_profiler_device_changed)
    device_selector_layout.addWidget(popup.profiler_device_combo)

    device_selector_layout.addStretch()
    layout.addLayout(device_selector_layout)

    # Splitter for charts and table
    splitter = QSplitter(Qt.Horizontal)

    # Left: Charts
    charts_widget = QWidget()
    charts_layout = QVBoxLayout(charts_widget)
    charts_layout.setContentsMargins(5, 5, 5, 5)

    # Bottlenecks bar chart
    bar_group = QGroupBox("Top Bottlenecks (Time in ms)")
    bar_group.setToolTip("Right-click to save chart")
    bar_layout = QVBoxLayout(bar_group)
    bar_layout.setContentsMargins(2, 2, 2, 2)
    popup.profiler_bar_figure = Figure(dpi=100)
    popup.profiler_bar_figure.set_tight_layout(True)
    popup.profiler_bar_canvas = FigureCanvas(popup.profiler_bar_figure)
    popup.profiler_bar_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    popup.profiler_bar_canvas.setMinimumHeight(150)
    popup.profiler_bar_canvas.mpl_connect(
        'button_press_event',
        lambda e: popup._on_chart_click(e, popup.profiler_bar_figure, "bottlenecks")
    )
    bar_layout.addWidget(popup.profiler_bar_canvas)
    charts_layout.addWidget(bar_group, 1)

    # Pie chart
    pie_group = QGroupBox("Time by Operation Type")
    pie_group.setToolTip("Right-click to save chart")
    pie_layout = QVBoxLayout(pie_group)
    pie_layout.setContentsMargins(2, 2, 2, 2)
    popup.profiler_pie_figure = Figure(dpi=100)
    popup.profiler_pie_figure.set_tight_layout(True)
    popup.profiler_pie_canvas = FigureCanvas(popup.profiler_pie_figure)
    popup.profiler_pie_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    popup.profiler_pie_canvas.setMinimumHeight(150)
    popup.profiler_pie_canvas.mpl_connect(
        'button_press_event',
        lambda e: popup._on_chart_click(e, popup.profiler_pie_figure, "time_distribution")
    )
    pie_layout.addWidget(popup.profiler_pie_canvas)
    charts_layout.addWidget(pie_group, 1)

    splitter.addWidget(charts_widget)

    # Right: Summary and table
    details_widget = QWidget()
    details_layout = QVBoxLayout(details_widget)
    details_layout.setContentsMargins(5, 5, 5, 5)

    # Summary text
    summary_group = QGroupBox("Summary")
    summary_layout = QVBoxLayout(summary_group)
    popup.profiler_summary_text = QTextEdit()
    popup.profiler_summary_text.setReadOnly(True)
    popup.profiler_summary_text.setFont(QFont("Monospace", 9))
    popup.profiler_summary_text.setMaximumHeight(200)
    summary_layout.addWidget(popup.profiler_summary_text)
    details_layout.addWidget(summary_group)

    # Operations table
    table_group = QGroupBox("Detailed Operations")
    table_layout = QVBoxLayout(table_group)
    popup.profiler_ops_table = QTableWidget()
    popup.profiler_ops_table.setColumnCount(5)
    popup.profiler_ops_table.setHorizontalHeaderLabels([
        "Operation", "CPU Time (ms)", "CUDA Time (ms)", "Calls", "Time/Call (ms)"
    ])
    popup.profiler_ops_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
    popup.profiler_ops_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
    popup.profiler_ops_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
    popup.profiler_ops_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
    popup.profiler_ops_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
    popup.profiler_ops_table.setAlternatingRowColors(True)
    table_layout.addWidget(popup.profiler_ops_table)
    details_layout.addWidget(table_group)

    splitter.addWidget(details_widget)
    splitter.setSizes([500, 500])

    layout.addWidget(splitter, 1)

    # Info label for when no profiler data is available
    popup.profiler_info_label = QLabel()
    popup.profiler_info_label.setAlignment(Qt.AlignCenter)
    popup.profiler_info_label.setWordWrap(True)
    popup.profiler_info_label.setStyleSheet("""
        QLabel {
            background-color: #fff3cd;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #ffc107;
            color: #856404;
        }
    """)
    layout.addWidget(popup.profiler_info_label)
    popup.profiler_info_label.hide()

    return widget
