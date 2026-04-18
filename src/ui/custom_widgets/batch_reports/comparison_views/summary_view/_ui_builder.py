"""UI builder for :class:`SummaryView` — attaches widgets to the view."""
from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QVBoxLayout,
)

from ui.custom_widgets.batch_reports.comparison_views.summary_view._draggable_table import (
    DraggableRowTableWidget,
)


def build_ui(view):
    """Setup the summary view UI and attach widgets to ``view``."""
    layout = QVBoxLayout(view)
    layout.setContentsMargins(10, 10, 10, 10)
    layout.setSpacing(10)

    # Header row with info, select all, and export button
    header_layout = QHBoxLayout()

    view.info_label = QLabel("No tests loaded")
    view.info_label.setStyleSheet("color: #666; font-size: 12px;")
    header_layout.addWidget(view.info_label)

    header_layout.addStretch()

    # Select all / none buttons
    view.select_all_btn = QPushButton("Select All")
    view.select_all_btn.setEnabled(False)
    view.select_all_btn.setStyleSheet("""
        QPushButton {
            background-color: #f5f5f5;
            color: #333;
            border: 1px solid #ccc;
            padding: 4px 10px;
            border-radius: 3px;
            font-size: 11px;
        }
        QPushButton:hover:enabled {
            background-color: #e0e0e0;
        }
        QPushButton:disabled {
            color: #999;
        }
    """)
    view.select_all_btn.clicked.connect(view._on_select_all)
    header_layout.addWidget(view.select_all_btn)

    view.select_none_btn = QPushButton("Select None")
    view.select_none_btn.setEnabled(False)
    view.select_none_btn.setStyleSheet("""
        QPushButton {
            background-color: #f5f5f5;
            color: #333;
            border: 1px solid #ccc;
            padding: 4px 10px;
            border-radius: 3px;
            font-size: 11px;
        }
        QPushButton:hover:enabled {
            background-color: #e0e0e0;
        }
        QPushButton:disabled {
            color: #999;
        }
    """)
    view.select_none_btn.clicked.connect(view._on_select_none)
    header_layout.addWidget(view.select_none_btn)

    header_layout.addSpacing(10)

    # Columns visibility button
    view.columns_btn = QPushButton("Columns ▼")
    view.columns_btn.setStyleSheet("""
        QPushButton {
            background-color: #f5f5f5;
            color: #333;
            border: 1px solid #ccc;
            padding: 4px 10px;
            border-radius: 3px;
            font-size: 11px;
        }
        QPushButton:hover {
            background-color: #e0e0e0;
        }
        QPushButton::menu-indicator {
            width: 0px;
        }
    """)
    view.columns_btn.clicked.connect(view._show_columns_menu)
    header_layout.addWidget(view.columns_btn)

    header_layout.addSpacing(10)

    view.copy_btn = QPushButton("Copy to Clipboard")
    view.copy_btn.setEnabled(False)
    view.copy_btn.setStyleSheet("""
        QPushButton {
            background-color: #607D8B;
            color: white;
            border: none;
            padding: 6px 12px;
            border-radius: 4px;
        }
        QPushButton:hover:enabled {
            background-color: #455A64;
        }
        QPushButton:disabled {
            background-color: #ccc;
            color: #888;
        }
    """)
    view.copy_btn.clicked.connect(view._on_copy_to_clipboard)
    header_layout.addWidget(view.copy_btn)

    header_layout.addSpacing(5)

    view.export_csv_btn = QPushButton("Export to CSV")
    view.export_csv_btn.setEnabled(False)
    view.export_csv_btn.setStyleSheet("""
        QPushButton {
            background-color: #2196F3;
            color: white;
            border: none;
            padding: 6px 12px;
            border-radius: 4px;
        }
        QPushButton:hover:enabled {
            background-color: #1976D2;
        }
        QPushButton:disabled {
            background-color: #ccc;
            color: #888;
        }
    """)
    view.export_csv_btn.clicked.connect(view._on_export_csv)
    header_layout.addWidget(view.export_csv_btn)

    layout.addLayout(header_layout)

    # Table with draggable rows
    view.table = DraggableRowTableWidget()
    view.table.setColumnCount(len(view.COLUMNS))
    view.table.setHorizontalHeaderLabels([col[0] for col in view.COLUMNS])
    view.table.setAlternatingRowColors(True)
    view.table.setSelectionBehavior(QTableWidget.SelectRows)
    view.table.setSelectionMode(QTableWidget.SingleSelection)

    # Connect row reorder signal
    view.table.rows_reordered.connect(view._on_row_dragged)

    # Context menu
    view.table.setContextMenuPolicy(Qt.CustomContextMenu)
    view.table.customContextMenuRequested.connect(view._show_context_menu)

    view.table.setStyleSheet("""
        QTableWidget {
            border: 1px solid #ccc;
            border-radius: 4px;
            background-color: white;
            gridline-color: #e0e0e0;
        }
        QTableWidget::item {
            padding: 4px;
        }
        QTableWidget::item:selected {
            background-color: #e3f2fd;
            color: #1976d2;
        }
        QHeaderView::section {
            background-color: #f5f5f5;
            border: none;
            border-bottom: 1px solid #ccc;
            border-right: 1px solid #e0e0e0;
            padding: 6px;
            font-weight: bold;
        }
        QHeaderView::section:vertical {
            background-color: #e8e8e8;
            border: none;
            border-bottom: 1px solid #ccc;
            border-right: 1px solid #ccc;
            padding: 4px 8px;
            min-width: 30px;
        }
        QHeaderView::section:vertical:hover {
            background-color: #d0d0d0;
        }
    """)

    # Set header resize modes
    header = view.table.horizontalHeader()
    # Interactive mode allows users to drag column borders to resize
    header.setSectionResizeMode(QHeaderView.Interactive)
    # Stretch last section to fill remaining space
    header.setStretchLastSection(True)
    # Allow horizontal scrollbar when columns exceed window width
    view.table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)

    # Set minimum widths for better readability
    # Checkbox column - fixed narrow width
    header.setSectionResizeMode(0, QHeaderView.Fixed)
    view.table.setColumnWidth(0, 30)

    # Set reasonable default widths for other columns
    default_width = 80  # Default width for any column
    for col in range(1, view.table.columnCount()):
        view.table.setColumnWidth(col, default_width)

    # Override specific widths for readability
    specific_widths = {
        1: 100,   # Experiment
        2: 120,   # Test
        3: 100,   # Mask Type
        4: 100,   # Reconstruction
        5: 70,    # Model
    }
    for col, width in specific_widths.items():
        if col < view.table.columnCount():
            view.table.setColumnWidth(col, width)

    # Sortable header with click to sort
    header.setSectionsClickable(True)
    header.sectionClicked.connect(view._on_header_clicked)
    view._sort_column = -1
    view._sort_order = Qt.AscendingOrder

    # Apply initial column visibility
    view._apply_column_visibility()

    layout.addWidget(view.table, 1)

    # Bottom row: hint and legend
    bottom_layout = QHBoxLayout()

    hint_label = QLabel("💡 Drag rows to reorder • Right-click for more options")
    hint_label.setStyleSheet("color: #888; font-size: 10px;")
    bottom_layout.addWidget(hint_label)

    bottom_layout.addStretch()

    legend_label = QLabel("Color legend:")
    legend_label.setStyleSheet("color: #666; font-size: 11px;")
    bottom_layout.addWidget(legend_label)

    best_label = QLabel("  Best")
    best_label.setStyleSheet(
        "background-color: #c8e6c9; color: #2e7d32; padding: 2px 8px; "
        "border-radius: 3px; font-size: 11px;"
    )
    bottom_layout.addWidget(best_label)

    worst_label = QLabel("  Worst")
    worst_label.setStyleSheet(
        "background-color: #ffcdd2; color: #c62828; padding: 2px 8px; "
        "border-radius: 3px; font-size: 11px;"
    )
    bottom_layout.addWidget(worst_label)

    layout.addLayout(bottom_layout)
