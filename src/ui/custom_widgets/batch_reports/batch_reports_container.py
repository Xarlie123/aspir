"""
Main container for Batch Reports mode.
Allows loading and comparing results from multiple batch test exports.
"""
import logging
from pathlib import Path
from typing import Optional, List

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QPushButton,
    QLabel, QListWidget, QListWidgetItem, QTabWidget, QGroupBox,
    QFileDialog, QMessageBox, QSizePolicy, QFrame, QMenu, QInputDialog
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QCursor

from ui.utils.file_formats import BATCH_TESTS_DIR, FileExtensions
from ui.custom_widgets.batch_reports.batch_report_model import BatchReportModel, LoadedExperiment
from ui.custom_widgets.batch_reports.comparison_views.summary_view import SummaryView
from ui.custom_widgets.batch_reports.comparison_views.quality_view import QualityView
from ui.custom_widgets.batch_reports.comparison_views.quality_preview_popup import QualityPreviewPopup
from ui.custom_widgets.batch_reports.comparison_views.timing_view import TimingView
from ui.custom_widgets.batch_reports.comparison_views.energy_view import EnergyView
from ui.custom_widgets.batch_reports.comparison_views.training_view import TrainingView
from ui.custom_widgets.batch_reports.comparison_views.export_view import ExportView


class BatchReportsContainer(QWidget):
    """
    Main container for Batch Reports mode.

    Features:
    - Load multiple batch test results (.batch_analysis_report)
    - Compare tests across experiments
    - Generate comparison charts and tables
    - Export reports in HTML, LaTeX, PDF, PNG formats
    """

    def __init__(self, logger=None, parent=None):
        super().__init__(parent)

        if logger:
            self.logger = logger.getChild("BatchReportsContainer")
        else:
            self.logger = logging.getLogger("BatchReportsContainer")

        # Use the BatchReportModel for data management
        self.model = BatchReportModel(logger=self.logger)
        self.model.experiments_changed.connect(self._on_model_changed)

        # Path to last session's batch report (set from BatchTestContainer signal)
        self._last_session_path: Optional[str] = None

        self._setup_ui()

        self.logger.debug("BatchReportsContainer initialized")

    def _setup_ui(self):
        """Setup the main UI layout."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Title
        title = QLabel("Batch Reports - Explore Executed Batch Tests")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #333;")
        main_layout.addWidget(title)

        # Main splitter: experiment list | comparison tabs
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        # Left panel: Experiment list
        left_panel = self._create_experiment_panel()
        splitter.addWidget(left_panel)

        # Right panel: Comparison tabs
        right_panel = self._create_comparison_tabs()
        splitter.addWidget(right_panel)

        # Set initial sizes (250px left, rest right)
        splitter.setSizes([250, 800])

        main_layout.addWidget(splitter, 1)

    def _create_experiment_panel(self) -> QWidget:
        """Create the left panel with experiment list and controls."""
        panel = QWidget()
        panel.setMinimumWidth(200)
        panel.setMaximumWidth(350)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 10, 0)
        layout.setSpacing(10)

        # Header
        header = QLabel("Loaded Experiments")
        header.setStyleSheet("font-weight: bold; font-size: 12px; color: #333;")
        layout.addWidget(header)

        # Experiment list with drag-and-drop reordering
        self.experiment_list = QListWidget()
        self.experiment_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.experiment_list.setDragDropMode(QListWidget.InternalMove)
        self.experiment_list.setDefaultDropAction(Qt.MoveAction)
        self.experiment_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #ccc;
                border-radius: 4px;
                background-color: white;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #eee;
            }
            QListWidget::item:selected {
                background-color: #e3f2fd;
                color: #1976d2;
            }
            QListWidget::item:hover {
                background-color: #f5f5f5;
            }
        """)
        self.experiment_list.itemSelectionChanged.connect(self._on_experiment_selection_changed)
        self.experiment_list.model().rowsMoved.connect(self._on_rows_moved)
        self.experiment_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.experiment_list.customContextMenuRequested.connect(self._show_experiment_context_menu)
        layout.addWidget(self.experiment_list, 1)

        # Buttons
        button_layout = QVBoxLayout()
        button_layout.setSpacing(5)

        self.load_btn = QPushButton("Load Batch Test from File")
        self.load_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.load_btn.clicked.connect(self._on_load_batch)
        button_layout.addWidget(self.load_btn)

        self.load_last_btn = QPushButton("Load Executed Batch Test")
        self.load_last_btn.setToolTip("Load the last batch test executed in this session")
        self.load_last_btn.setEnabled(False)  # Disabled until a batch is run
        self.load_last_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover:enabled {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #ccc;
                color: #888;
            }
        """)
        self.load_last_btn.clicked.connect(self._on_load_last_session)
        button_layout.addWidget(self.load_last_btn)

        self.remove_btn = QPushButton("Remove Selected")
        self.remove_btn.setEnabled(False)
        self.remove_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover:enabled {
                background-color: #da190b;
            }
            QPushButton:disabled {
                background-color: #ccc;
                color: #888;
            }
        """)
        self.remove_btn.clicked.connect(self._on_remove_selected)
        button_layout.addWidget(self.remove_btn)

        self.clear_btn = QPushButton("Clear All")
        self.clear_btn.setEnabled(False)
        self.clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff9800;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover:enabled {
                background-color: #f57c00;
            }
            QPushButton:disabled {
                background-color: #ccc;
                color: #888;
            }
        """)
        self.clear_btn.clicked.connect(self._on_clear_all)
        button_layout.addWidget(self.clear_btn)

        layout.addLayout(button_layout)

        # Info label
        self.info_label = QLabel("No experiments loaded")
        self.info_label.setStyleSheet("color: #666; font-size: 11px;")
        self.info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.info_label)

        return panel

    def _create_comparison_tabs(self) -> QWidget:
        """Create the right panel with comparison view tabs."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #ccc;
                border-radius: 4px;
                background-color: white;
            }
            QTabBar::tab {
                padding: 8px 16px;
                margin-right: 2px;
                background-color: #f5f5f5;
                border: 1px solid #ccc;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom: 1px solid white;
            }
            QTabBar::tab:hover:!selected {
                background-color: #e0e0e0;
            }
        """)

        # Create actual views and placeholder tabs
        self.summary_view = SummaryView(logger=self.logger)
        self.summary_view.selection_changed.connect(self._on_test_selection_changed)
        self.summary_view.tests_reordered.connect(self._on_tests_reordered)
        self.tabs.addTab(self.summary_view, "Summary")

        self.training_view = TrainingView(logger=self.logger)
        self.tabs.addTab(self.training_view, "Training")

        self.quality_view = QualityView(logger=self.logger)
        self.quality_view.previewRequested.connect(self._on_quality_preview_requested)
        self.tabs.addTab(self.quality_view, "Quality")

        self.timing_view = TimingView(logger=self.logger)
        self.tabs.addTab(self.timing_view, "Timing")

        self.energy_view = EnergyView(logger=self.logger)
        self.tabs.addTab(self.energy_view, "Energy")

        self.export_view = ExportView(logger=self.logger)
        self.tabs.addTab(self.export_view, "Export")

        layout.addWidget(self.tabs)
        return panel

    def _create_placeholder_tab(self, title: str, description: str) -> QWidget:
        """Create a placeholder tab widget."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(40, 40, 40, 40)

        # Placeholder frame
        frame = QFrame()
        frame.setStyleSheet("""
            QFrame {
                background-color: #f5f5f5;
                border: 2px dashed #ccc;
                border-radius: 8px;
            }
        """)
        frame_layout = QVBoxLayout(frame)
        frame_layout.setContentsMargins(20, 40, 20, 40)
        frame_layout.setSpacing(15)

        # Title
        title_label = QLabel(title)
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #333;")
        title_label.setAlignment(Qt.AlignCenter)
        frame_layout.addWidget(title_label)

        # Description
        desc_label = QLabel(description)
        desc_label.setStyleSheet("font-size: 12px; color: #666;")
        desc_label.setAlignment(Qt.AlignCenter)
        frame_layout.addWidget(desc_label)

        # Coming soon
        soon_label = QLabel("Load experiments to see comparison")
        soon_label.setStyleSheet("""
            background-color: #e3f2fd;
            color: #1976d2;
            padding: 8px 16px;
            border-radius: 4px;
        """)
        soon_label.setAlignment(Qt.AlignCenter)
        frame_layout.addWidget(soon_label, alignment=Qt.AlignCenter)

        layout.addWidget(frame)
        layout.addStretch()

        return widget

    def _on_load_batch(self):
        """Handle load batch button click."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Batch Analysis Report",
            str(BATCH_TESTS_DIR),
            f"Batch Analysis Report (*{FileExtensions.BATCH_ANALYSIS_REPORT});;All Files (*.*)"
        )

        if file_path:
            experiment = self.model.load_experiment(Path(file_path))
            if experiment is None:
                QMessageBox.warning(self, "Load Error", "Failed to load batch report.")

    def _on_load_last_session(self):
        """Handle load last session button click."""
        if not self._last_session_path:
            QMessageBox.information(
                self,
                "No Session Available",
                "No batch test has been executed in this session yet."
            )
            return

        path = Path(self._last_session_path)
        if not path.exists():
            QMessageBox.warning(
                self,
                "File Not Found",
                f"The last session report no longer exists:\n{self._last_session_path}"
            )
            return

        experiment = self.model.load_experiment(path)
        if experiment is None:
            QMessageBox.warning(self, "Load Error", "Failed to load the last session report.")
        else:
            self.logger.info(f"Loaded last session report: {self._last_session_path}")

    def set_last_session_path(self, path: str):
        """
        Set the path to the last session's batch report.

        Called from main_window when BatchTestContainer emits batch_report_available.
        Enables the "Load Last Session" button.
        """
        self._last_session_path = path
        self.load_last_btn.setEnabled(True)
        self.load_last_btn.setToolTip(f"Load: {Path(path).name}")
        self.logger.debug(f"Last session path set: {path}")

    def _on_remove_selected(self):
        """Remove selected experiments from the list."""
        selected_items = self.experiment_list.selectedItems()
        if not selected_items:
            return

        # Get indices to remove (in reverse order to avoid index shifting)
        indices = sorted([item.data(Qt.UserRole) for item in selected_items], reverse=True)

        for idx in indices:
            self.model.remove_experiment(idx)

    def _on_clear_all(self):
        """Clear all loaded experiments."""
        if self.model.is_empty():
            return

        reply = QMessageBox.question(
            self,
            "Clear All",
            "Remove all loaded experiments?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.model.clear_all()

    def _on_experiment_selection_changed(self):
        """Handle experiment selection change in the list."""
        has_selection = len(self.experiment_list.selectedItems()) > 0
        self.remove_btn.setEnabled(has_selection)

    def _show_experiment_context_menu(self, position):
        """Show context menu for the experiment list."""
        menu = QMenu(self)

        # Load new batch test (always available)
        load_action = menu.addAction("Load new Batch Test...")
        load_action.triggered.connect(self._on_load_batch)

        menu.addSeparator()

        # Get selected item
        item = self.experiment_list.itemAt(position)
        has_item = item is not None
        current_row = self.experiment_list.row(item) if has_item else -1
        total_items = self.experiment_list.count()

        # Rename
        rename_action = menu.addAction("Rename...")
        rename_action.setEnabled(has_item)
        rename_action.triggered.connect(lambda: self._rename_experiment(current_row))

        menu.addSeparator()

        # Move actions
        move_up_action = menu.addAction("Move Up")
        move_up_action.setEnabled(has_item and current_row > 0)
        move_up_action.triggered.connect(lambda: self._move_experiment(current_row, current_row - 1))

        move_down_action = menu.addAction("Move Down")
        move_down_action.setEnabled(has_item and current_row < total_items - 1)
        move_down_action.triggered.connect(lambda: self._move_experiment(current_row, current_row + 1))

        move_top_action = menu.addAction("Move to Top")
        move_top_action.setEnabled(has_item and current_row > 0)
        move_top_action.triggered.connect(lambda: self._move_experiment(current_row, 0))

        move_bottom_action = menu.addAction("Move to Bottom")
        move_bottom_action.setEnabled(has_item and current_row < total_items - 1)
        move_bottom_action.triggered.connect(lambda: self._move_experiment(current_row, total_items - 1))

        menu.addSeparator()

        # Remove
        remove_action = menu.addAction("Remove")
        remove_action.setEnabled(has_item)
        remove_action.triggered.connect(lambda: self._remove_experiment_at(current_row))

        # Show menu at cursor position
        menu.exec_(self.experiment_list.mapToGlobal(position))

    def _rename_experiment(self, index: int):
        """Rename an experiment at the given index."""
        if index < 0 or index >= len(self.model.experiments):
            return

        exp = self.model.experiments[index]
        new_name, ok = QInputDialog.getText(
            self,
            "Rename Experiment",
            "Enter new name:",
            text=exp.name
        )

        if ok and new_name.strip():
            self.model.rename_experiment(index, new_name.strip())
            self.logger.info("Renamed experiment %d to '%s'", index, new_name.strip())

    def _move_experiment(self, from_index: int, to_index: int):
        """Move an experiment from one position to another."""
        if from_index == to_index:
            return

        self.model.move_experiment(from_index, to_index)
        self.logger.info("Moved experiment from %d to %d", from_index, to_index)

    def _remove_experiment_at(self, index: int):
        """Remove an experiment at the given index."""
        if index < 0 or index >= len(self.model.experiments):
            return

        self.model.remove_experiment(index)
        self.logger.info("Removed experiment at index %d", index)

    def _on_rows_moved(self, parent, start, end, destination, row):
        """Handle drag-and-drop reordering in the experiment list."""
        # Calculate the actual move in the model
        # When dragging down, the destination row is the position after removal
        # When dragging up, it's the direct target position
        from_index = start
        to_index = row if row < start else row - 1

        if from_index != to_index:
            # Temporarily disconnect to avoid recursive updates
            self.model.experiments_changed.disconnect(self._on_model_changed)

            # Update the model to match the new visual order
            self.model.move_experiment(from_index, to_index)

            # Reconnect and update views (order is now synced)
            self.model.experiments_changed.connect(self._on_model_changed)

            # Update UserRole data to reflect new indices
            for i in range(self.experiment_list.count()):
                self.experiment_list.item(i).setData(Qt.UserRole, i)

            # Refresh comparison views with new order
            self._update_comparison_views()

            self.logger.info("Reordered experiments: moved %d to %d", from_index, to_index)

    def _on_model_changed(self):
        """Handle model data changes - rebuild the experiment list."""
        self.experiment_list.clear()

        for i, exp in enumerate(self.model.experiments):
            item = QListWidgetItem()
            # Map export_level to user-friendly label
            export_level = exp.export_level.lower() if exp.export_level else "reports_only"
            if "all_data" in export_level:
                data_type = "Reports + datasets"
            elif "models" in export_level:
                data_type = "Reports + models"
            else:
                data_type = "Reports only"
            item.setText(f"{exp.name}\n  {exp.test_count} tests · {data_type}")
            item.setData(Qt.UserRole, i)
            self.experiment_list.addItem(item)

        self._update_ui_state()
        self._update_comparison_views()

    def _update_ui_state(self):
        """Update UI elements based on current state."""
        has_experiments = not self.model.is_empty()
        self.clear_btn.setEnabled(has_experiments)

        if has_experiments:
            total_tests = sum(exp.test_count for exp in self.model.experiments)
            self.info_label.setText(
                f"{len(self.model)} experiments, {total_tests} tests total"
            )
        else:
            self.info_label.setText("No experiments loaded")

    def _update_comparison_views(self):
        """Update all comparison views with current data."""
        self.logger.debug("Updating views with %d experiments", len(self.model))

        # Update summary view with all tests
        all_tests = self.model.get_all_tests()
        self.summary_view.set_tests(all_tests)

        # Other views get the selected tests from summary view
        selected_tests = self.summary_view.get_selected_tests()
        self.quality_view.set_tests(selected_tests)
        self.timing_view.set_tests(selected_tests)
        self.energy_view.set_tests(selected_tests)
        self.training_view.set_tests(selected_tests)
        self.export_view.set_tests(selected_tests)

    def _on_test_selection_changed(self, selected_indices: list):
        """Handle test selection change from SummaryView."""
        # Update other views with only selected tests
        selected_tests = self.summary_view.get_selected_tests()
        self.quality_view.set_tests(selected_tests)
        self.timing_view.set_tests(selected_tests)
        self.energy_view.set_tests(selected_tests)
        self.training_view.set_tests(selected_tests)
        self.export_view.set_tests(selected_tests)
        self.logger.debug("Test selection changed: %d tests selected", len(selected_indices))

    def _on_tests_reordered(self, new_order: list):
        """Handle test reordering from SummaryView."""
        # Update other views with reordered tests
        ordered_tests = self.summary_view.get_ordered_tests()
        # Filter to only selected tests
        selected_tests = self.summary_view.get_selected_tests()
        self.quality_view.set_tests(selected_tests)
        self.timing_view.set_tests(selected_tests)
        self.energy_view.set_tests(selected_tests)
        self.training_view.set_tests(selected_tests)
        self.export_view.set_tests(selected_tests)
        self.logger.debug("Tests reordered")

    def get_all_tests(self) -> List[dict]:
        """Get flattened list of all tests from all experiments."""
        return self.model.get_all_tests()

    def get_comparison_data(self):
        """Get structured data for comparison views."""
        return self.model.get_comparison_data()

    def _on_quality_preview_requested(self):
        """Handle quality preview button click from QualityView."""
        # Get the currently selected tests from quality view
        tests = self.summary_view.get_selected_tests()

        # Filter to only tests with per-image data
        tests_with_data = [t for t in tests if t.get("quality_per_image")]

        if not tests_with_data:
            QMessageBox.information(
                self,
                "No Preview Data",
                "No tests with per-image quality data available.\n"
                "Run batch tests with Quality report enabled to generate per-image data."
            )
            return

        # Open the preview popup
        popup = QualityPreviewPopup(tests_with_data, logger=self.logger, parent=self)
        popup.exec_()
