"""
Training view for Batch Reports - displays training curves charts.
"""
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFileDialog, QMessageBox, QMenu,
    QSplitter, QListWidget, QDialog
)
from PySide6.QtCore import Qt

from ui.custom_widgets.batch_reports.comparison_views.training_summary_popup import (
    TrainingSummaryPopup
)
from ui.custom_widgets.batch_reports.comparison_views.chart_config_popup import (
    ChartConfigPopup, CustomNavigationToolbar
)

# Import for architecture preview
from ui.custom_widgets.postprocessor_control.architecture_config.architecture_preview_popup import (
    ArchitecturePreviewPopup, PDFLATEX_AVAILABLE
)
from simulation_engine._4_postprocessor.postprocessor_nn import MODEL_REGISTRY


class TrainingView(QWidget):
    """
    Training view displaying training curves comparison charts.

    Features:
    - Left menu to select chart type (QListWidget)
    - Loss evolution (validation/test loss over epochs)
    - Quality metrics evolution (PSNR, SSIM, LPIPS over epochs)
    - Comparison across multiple experiments
    - Training Summary button opens detailed popup
    - Export to PNG via right-click context menu
    """

    # Color palette for experiments
    COLORS = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336',
              '#00BCD4', '#8BC34A', '#FFC107', '#673AB7', '#E91E63']

    def __init__(self, logger=None, parent=None):
        super().__init__(parent)

        if logger:
            self.logger = logger.getChild("TrainingView")
        else:
            self.logger = logging.getLogger("TrainingView")

        self._tests: List[Dict[str, Any]] = []

        # Chart configuration with defaults
        self._chart_config = {
            'axes': {
                'title': '',  # Empty = auto
                'title_fontsize': 13,
                'xlabel': '',
                'xlabel_fontsize': 11,
                'xtick_fontsize': 8,
                'ylabel': '',
                'ylabel_fontsize': 11,
                'ytick_fontsize': 8,
                'auto_scale': True,
                'ymin': 0.0,
                'ymax': 1.0,
            },
            'legend': {
                'position': 0,  # Inside (upper right)
                'fontsize': 9,
                'frameon': True,
                'shadow': False,
                'fancybox': True,
                'framealpha': 0.8,
                'ncol': 2,
            },
            'lines': {
                'linestyle': '-',
                'linewidth': 1.5,
                'marker': 'o',
                'markersize': 4,
            },
            'colors': {
                'psnr': '#2ca02c',
                'ssim': '#d62728',
                'lpips': '#9467bd',
                'noisy': '#1f77b4',
                'denoised': '#2ca02c',
                'bar_alpha': 0.8,
                'hist_alpha': 0.7,
            }
        }

        self._setup_ui()

    def _setup_ui(self):
        """Setup the training view UI with left menu."""
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)

        # Splitter for menu and chart area
        splitter = QSplitter(Qt.Horizontal)

        # Left panel: Chart type selection
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(5, 5, 5, 5)
        left_layout.setSpacing(10)

        # Chart type list - styled like Quality view
        chart_label = QLabel("Chart Type:")
        chart_label.setStyleSheet("font-weight: bold; color: #333;")
        left_layout.addWidget(chart_label)

        self.chart_list = QListWidget()
        self.chart_list.setMaximumWidth(220)
        self.chart_list.setMinimumWidth(180)
        self.chart_list.addItem("Loss Evolution")
        self.chart_list.addItem("PSNR Evolution")
        self.chart_list.addItem("SSIM Evolution")
        self.chart_list.addItem("LPIPS Evolution")
        self.chart_list.addItem("Combined Quality Metrics")
        self.chart_list.setCurrentRow(0)
        self.chart_list.currentRowChanged.connect(self._on_chart_type_changed)
        self.chart_list.setStyleSheet("""
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
        left_layout.addWidget(self.chart_list)

        # Training Summary button
        self.summary_btn = QPushButton("Training Summary")
        self.summary_btn.setMaximumWidth(220)
        self.summary_btn.setMinimumWidth(180)
        self.summary_btn.setMinimumHeight(36)
        self.summary_btn.setEnabled(False)  # Disabled until data is available
        self.summary_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
                font-size: 12px;
                padding: 8px;
            }
            QPushButton:hover:enabled {
                background-color: #45a049;
            }
            QPushButton:pressed:enabled {
                background-color: #3d8b40;
            }
            QPushButton:disabled {
                background-color: #ccc;
                color: #888;
            }
        """)
        self.summary_btn.setToolTip("View detailed training summary with hyperparameters and charts")
        self.summary_btn.clicked.connect(self._on_summary_clicked)
        left_layout.addWidget(self.summary_btn)

        # Preview Architecture button
        self.preview_arch_btn = QPushButton("Preview Architecture")
        self.preview_arch_btn.setMaximumWidth(220)
        self.preview_arch_btn.setMinimumWidth(180)
        self.preview_arch_btn.setMinimumHeight(36)
        self.preview_arch_btn.setEnabled(False)  # Disabled until data is available
        self.preview_arch_btn.setStyleSheet("""
            QPushButton {
                background-color: #673AB7;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
                font-size: 12px;
                padding: 8px;
            }
            QPushButton:hover:enabled {
                background-color: #5E35B1;
            }
            QPushButton:pressed:enabled {
                background-color: #512DA8;
            }
            QPushButton:disabled {
                background-color: #ccc;
                color: #888;
            }
        """)
        self.preview_arch_btn.setToolTip("Preview neural network architecture diagram")
        self.preview_arch_btn.clicked.connect(self._on_preview_architecture_clicked)
        left_layout.addWidget(self.preview_arch_btn)

        left_layout.addStretch()

        splitter.addWidget(left_panel)

        # Right panel: Chart area
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(5, 5, 5, 5)
        right_layout.setSpacing(5)

        # Chart area
        self.figure = Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet("background-color: white;")

        # Enable right-click context menu on canvas
        self.canvas.setContextMenuPolicy(Qt.CustomContextMenu)
        self.canvas.customContextMenuRequested.connect(self._show_context_menu)

        # Navigation toolbar with custom chart config button
        self.toolbar = CustomNavigationToolbar(
            self.canvas, self,
            config_callback=self._on_open_chart_config
        )

        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas, 1)

        # Info label
        self.info_label = QLabel("Load experiments to see training curves")
        self.info_label.setStyleSheet("color: #666; font-size: 11px;")
        self.info_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.info_label)

        splitter.addWidget(right_panel)

        # Set splitter sizes (menu:chart = 1:4)
        splitter.setSizes([180, 720])

        main_layout.addWidget(splitter)

    def set_tests(self, tests: List[Dict[str, Any]]):
        """
        Set the tests to display in the charts.

        Args:
            tests: List of test dictionaries with training curves
        """
        self._tests = tests
        self._update_summary_button_state()
        self._refresh_chart()

    def _update_summary_button_state(self):
        """Enable/disable summary button based on data availability."""
        has_training_data = self._has_training_data()
        self.summary_btn.setEnabled(has_training_data)
        if has_training_data:
            self.summary_btn.setToolTip("View detailed training summary with hyperparameters and charts")
        else:
            self.summary_btn.setToolTip("No training curves available")

        # Enable preview architecture if we have any tests with training data
        has_tests = len(self._tests) > 0
        self.preview_arch_btn.setEnabled(has_tests)
        if has_tests:
            self.preview_arch_btn.setToolTip("Preview neural network architecture diagram")
        else:
            self.preview_arch_btn.setToolTip("No tests available")

    def _has_training_data(self) -> bool:
        """Check if any test has training curves data."""
        for test in self._tests:
            if test.get("training_curves"):
                return True
        return False

    def _show_context_menu(self, pos):
        """Show context menu for saving the chart."""
        menu = QMenu(self)
        save_action = menu.addAction("Save chart as...")
        save_action.triggered.connect(self._on_export_chart)
        menu.exec(self.canvas.mapToGlobal(pos))

    def _on_summary_clicked(self):
        """Handle summary button click - open TrainingSummaryPopup."""
        if not self._tests:
            return

        popup = TrainingSummaryPopup(
            tests=self._tests,
            parent=self,
            logger=self.logger
        )
        popup.exec()

    def _on_preview_architecture_clicked(self):
        """Handle preview architecture button click."""
        if not self._tests:
            return

        # Check if pdflatex is available
        if not PDFLATEX_AVAILABLE:
            QMessageBox.warning(
                self,
                "LaTeX Not Available",
                "pdflatex is required for architecture visualization.\n\n"
                "Install with:\n"
                "  sudo apt install texlive-latex-base texlive-latex-extra\n\n"
                "Or on other systems:\n"
                "  macOS: brew install --cask mactex\n"
                "  Windows: Install MiKTeX from https://miktex.org/"
            )
            return

        # Open architecture preview directly with first test
        # The popup will have a combobox to switch between tests
        first_test = self._tests[0]
        self._show_architecture_preview(first_test)

    def _show_architecture_preview(self, test: Dict[str, Any]):
        """Show the architecture preview popup for the selected test."""
        # Get model name from test
        model_name = test.get("model_name")
        if not model_name:
            config = test.get("config", {})
            model_name = config.get("model_name", "u-net")

        # Get test name
        test_name = test.get("name", "Unknown")

        # Get model entry from registry
        model_entry = MODEL_REGISTRY.get(model_name.lower())
        if not model_entry:
            # Try with original case
            model_entry = MODEL_REGISTRY.get(model_name)
        if not model_entry:
            QMessageBox.warning(
                self,
                "Model Not Found",
                f"Model '{model_name}' not found in MODEL_REGISTRY.\n\n"
                f"Available models: {', '.join(MODEL_REGISTRY.keys())}"
            )
            return

        # Get input size from test or use default
        input_size = test.get("img_size", 64)
        if not input_size:
            config = test.get("config", {})
            input_size = config.get("img_size", 64)

        # Create model instance
        try:
            import inspect
            model_cls = model_entry["cls"]
            defaults = model_entry.get("defaults", {}).copy()

            # Apply per-test architecture overrides so the preview matches
            # what was actually trained (e.g. features=[8,16] instead of default).
            arch_overrides = test.get("architecture_config") or {}
            defaults.update(arch_overrides)

            # Set input size if model accepts it
            if "img_size" in defaults:
                defaults["img_size"] = input_size

            # Filter to kwargs the model accepts (robust to legacy keys).
            sig = inspect.signature(model_cls.__init__)
            valid = set(sig.parameters) - {"self"}
            kwargs = {k: v for k, v in defaults.items() if k in valid}

            model = model_cls(**kwargs)
            self.logger.info("Created model instance: %s with kwargs: %s", model_name, kwargs)
        except Exception as e:
            self.logger.error("Failed to create model: %s", e, exc_info=True)
            QMessageBox.warning(
                self,
                "Model Creation Failed",
                f"Failed to create model '{model_name}':\n{e}"
            )
            return

        # Try to load images from exported dataset
        ground_truth_images = None
        noisy_images = None
        denoised_images = None

        experiment_path = test.get("_experiment_path")
        if experiment_path:
            report_path = Path(experiment_path)
            batch_dir = report_path.parent

            safe_name = test_name.replace(" ", "_").replace("/", "-")
            test_images_path = batch_dir / "data" / safe_name / "test_images.npz"

            if test_images_path.exists():
                try:
                    data = np.load(test_images_path)
                    if "originals" in data:
                        ground_truth_images = data["originals"]
                    if "reconstructions" in data:
                        noisy_images = data["reconstructions"]
                    if "denoised" in data:
                        denoised_images = data["denoised"]
                    self.logger.info("Loaded images from %s", test_images_path)
                except Exception as e:
                    self.logger.warning("Failed to load images: %s", e)

        # Open architecture preview popup with test selector
        popup = ArchitecturePreviewPopup(
            model=model,
            model_name=model_name,
            input_size=input_size,
            config=kwargs,
            ground_truth_images=ground_truth_images,
            noisy_images=noisy_images,
            denoised_images=denoised_images,
            parent=self,
            logger=self.logger,
            # Batch Reports mode: pass tests list for test selector
            tests=self._tests,
            test_name=test_name
        )
        popup.exec()

    def _on_open_chart_config(self):
        """Open chart configuration dialog."""
        popup = ChartConfigPopup(parent=self, logger=self.logger)
        popup.set_config(self._chart_config)

        if popup.exec() == QDialog.Accepted:
            self._chart_config = popup.get_config()
            self.logger.debug("Chart config updated: %s", self._chart_config)
            self._refresh_chart()

    def _apply_axes_config(self, ax, default_title: str = "", default_xlabel: str = "",
                           default_ylabel: str = ""):
        """Apply axes configuration to a matplotlib axis.

        Args:
            ax: The matplotlib axis to configure
            default_title: Default title if not specified in config
            default_xlabel: Default x-label if not specified in config
            default_ylabel: Default y-label if not specified in config
        """
        axes_cfg = self._chart_config.get('axes', {})

        # Title
        title = axes_cfg.get('title', '') or default_title
        title_fontsize = axes_cfg.get('title_fontsize', 13)
        if title:
            ax.set_title(title, fontsize=title_fontsize, fontweight='bold')

        # X-axis label
        xlabel = axes_cfg.get('xlabel', '') or default_xlabel
        xlabel_fontsize = axes_cfg.get('xlabel_fontsize', 11)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=xlabel_fontsize)

        # Y-axis label
        ylabel = axes_cfg.get('ylabel', '') or default_ylabel
        ylabel_fontsize = axes_cfg.get('ylabel_fontsize', 11)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)

        # Tick label font sizes
        xtick_fontsize = axes_cfg.get('xtick_fontsize', 8)
        ytick_fontsize = axes_cfg.get('ytick_fontsize', 8)
        ax.tick_params(axis='x', labelsize=xtick_fontsize)
        ax.tick_params(axis='y', labelsize=ytick_fontsize)

        # Y-axis scale (only if not auto)
        if not axes_cfg.get('auto_scale', True):
            ymin = axes_cfg.get('ymin', 0)
            ymax = axes_cfg.get('ymax', 1)
            ax.set_ylim(ymin, ymax)

    def _apply_legend(self, ax, handles=None, labels=None):
        """Apply legend configuration to a matplotlib axis.

        Args:
            ax: The matplotlib axis
            handles: Optional list of handles for legend
            labels: Optional list of labels for legend
        """
        legend_cfg = self._chart_config.get('legend', {})
        legend_pos = legend_cfg.get('position', 0)
        legend_kwargs = {
            'fontsize': legend_cfg.get('fontsize', 9),
            'frameon': legend_cfg.get('frameon', True),
            'shadow': legend_cfg.get('shadow', False),
            'fancybox': legend_cfg.get('fancybox', True),
            'framealpha': legend_cfg.get('framealpha', 0.8),
        }

        # Position mapping
        loc_map = {
            0: 'upper right',
            1: 'upper left',
            2: 'lower right',
            3: 'lower left',
        }

        ncol = legend_cfg.get('ncol', 2)

        if handles and labels:
            if legend_pos in loc_map:
                ax.legend(handles, labels, loc=loc_map[legend_pos],
                         ncol=ncol, **legend_kwargs)
            elif legend_pos == 4:  # Right side (outside)
                ax.legend(handles, labels, loc='center left',
                         bbox_to_anchor=(1.02, 0.5), ncol=ncol, **legend_kwargs)
            else:  # Below (outside)
                legend_kwargs['frameon'] = False
                ax.legend(handles, labels, loc='upper center',
                         bbox_to_anchor=(0.5, -0.15), ncol=ncol, **legend_kwargs)
        else:
            if legend_pos in loc_map:
                ax.legend(loc=loc_map[legend_pos], ncol=ncol, **legend_kwargs)
            elif legend_pos == 4:  # Right side (outside)
                ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
                         ncol=ncol, **legend_kwargs)
            else:  # Below (outside)
                legend_kwargs['frameon'] = False
                ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                         ncol=ncol, **legend_kwargs)

    def _on_chart_type_changed(self, index: int):
        """Handle chart type selection change."""
        self._refresh_chart()

    def _refresh_chart(self):
        """Refresh the chart based on current settings."""
        self.figure.clear()

        # Filter tests that have training curves
        tests_with_curves = [t for t in self._tests if t.get("training_curves")]

        if not tests_with_curves:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "No training curves available\nEnable 'Training Curves' in reports",
                   ha='center', va='center', fontsize=14, color='#999')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            self.canvas.draw()
            self.info_label.setText("No training curves available in loaded experiments")
            return

        chart_type = self.chart_list.currentRow()

        if chart_type == 0:
            self._draw_loss_evolution(tests_with_curves)
        elif chart_type == 1:
            self._draw_metric_evolution(tests_with_curves, "val_psnr", "PSNR (dB)")
        elif chart_type == 2:
            self._draw_metric_evolution(tests_with_curves, "val_ssim", "SSIM")
        elif chart_type == 3:
            self._draw_metric_evolution(tests_with_curves, "val_lpips", "LPIPS")
        else:
            self._draw_combined_quality(tests_with_curves)

        self.figure.tight_layout()
        self.canvas.draw()

        # Update info
        self.info_label.setText(
            f"Showing training curves for {len(tests_with_curves)} test(s)"
        )

    def _draw_loss_evolution(self, tests: List[Dict]):
        """Draw loss evolution chart."""
        ax = self.figure.add_subplot(111)

        lines_cfg = self._chart_config.get('lines', {})
        linestyle = lines_cfg.get('linestyle', '-')
        linewidth = lines_cfg.get('linewidth', 1.5)
        marker = lines_cfg.get('marker', 'o')
        markersize = lines_cfg.get('markersize', 4)

        for i, test in enumerate(tests):
            curves = test.get("training_curves", {})
            val_losses = curves.get("val_losses", [])
            test_losses = curves.get("test_losses", [])
            test_name = test.get("name", f"Test {i}")
            color = self.COLORS[i % len(self.COLORS)]

            if val_losses:
                epochs = range(1, len(val_losses) + 1)
                ax.plot(epochs, val_losses, linestyle=linestyle, marker=marker,
                       label=f'{test_name} (val)', color=color,
                       linewidth=linewidth, markersize=markersize)
            if test_losses:
                epochs = range(1, len(test_losses) + 1)
                ax.plot(epochs, test_losses, linestyle='--', marker=marker,
                       label=f'{test_name} (test)', color=color,
                       linewidth=linewidth, markersize=markersize, alpha=0.7)

        self._apply_axes_config(ax, default_title="Loss Evolution During Training",
                               default_xlabel="Epoch", default_ylabel="Loss")
        self._apply_legend(ax)
        ax.grid(True, alpha=0.3)

    def _draw_metric_evolution(self, tests: List[Dict], metric_key: str, metric_label: str):
        """Draw a single metric evolution chart."""
        ax = self.figure.add_subplot(111)

        lines_cfg = self._chart_config.get('lines', {})
        linestyle = lines_cfg.get('linestyle', '-')
        linewidth = lines_cfg.get('linewidth', 1.5)
        marker = lines_cfg.get('marker', 'o')
        markersize = lines_cfg.get('markersize', 4)

        for i, test in enumerate(tests):
            curves = test.get("training_curves", {})
            values = curves.get(metric_key, [])
            test_name = test.get("name", f"Test {i}")
            color = self.COLORS[i % len(self.COLORS)]

            if values:
                epochs = range(1, len(values) + 1)
                ax.plot(epochs, values, linestyle=linestyle, marker=marker,
                       label=test_name, color=color,
                       linewidth=linewidth, markersize=markersize)

        self._apply_axes_config(ax,
                               default_title=f"{metric_label} Evolution During Training",
                               default_xlabel="Epoch", default_ylabel=metric_label)
        self._apply_legend(ax)
        ax.grid(True, alpha=0.3)

        # Add direction annotation
        higher_is_better = metric_key != "val_lpips"
        direction = "higher is better" if higher_is_better else "lower is better"
        ax.annotate(f"({direction})", xy=(1, 1), xycoords='axes fraction',
                   fontsize=9, color='#666', ha='right', va='top')

    def _draw_combined_quality(self, tests: List[Dict]):
        """Draw combined quality metrics (PSNR, SSIM, LPIPS) in subplots."""
        # Create 3 subplots vertically
        ax1 = self.figure.add_subplot(311)
        ax2 = self.figure.add_subplot(312)
        ax3 = self.figure.add_subplot(313)

        lines_cfg = self._chart_config.get('lines', {})
        linestyle = lines_cfg.get('linestyle', '-')
        linewidth = lines_cfg.get('linewidth', 1.5)
        marker = lines_cfg.get('marker', 'o')
        markersize = lines_cfg.get('markersize', 4)

        for i, test in enumerate(tests):
            curves = test.get("training_curves", {})
            test_name = test.get("name", f"Test {i}")
            color = self.COLORS[i % len(self.COLORS)]

            val_psnr = curves.get("val_psnr", [])
            val_ssim = curves.get("val_ssim", [])
            val_lpips = curves.get("val_lpips", [])

            if val_psnr:
                epochs = range(1, len(val_psnr) + 1)
                ax1.plot(epochs, val_psnr, linestyle=linestyle, marker=marker,
                        label=test_name, color=color, linewidth=linewidth, markersize=markersize)

            if val_ssim:
                epochs = range(1, len(val_ssim) + 1)
                ax2.plot(epochs, val_ssim, linestyle=linestyle, marker=marker,
                        label=test_name, color=color, linewidth=linewidth, markersize=markersize)

            if val_lpips:
                epochs = range(1, len(val_lpips) + 1)
                ax3.plot(epochs, val_lpips, linestyle=linestyle, marker=marker,
                        label=test_name, color=color, linewidth=linewidth, markersize=markersize)

        # Configure axes using config
        axes_cfg = self._chart_config.get('axes', {})
        title_fontsize = axes_cfg.get('title_fontsize', 12)
        ylabel_fontsize = axes_cfg.get('ylabel_fontsize', 10)
        xlabel_fontsize = axes_cfg.get('xlabel_fontsize', 10)

        ax1.set_ylabel("PSNR (dB)", fontsize=ylabel_fontsize)
        ax1.set_title("Quality Metrics Evolution", fontsize=title_fontsize, fontweight='bold')
        self._apply_legend(ax1)
        ax1.grid(True, alpha=0.3)

        ax2.set_ylabel("SSIM", fontsize=ylabel_fontsize)
        ax2.set_ylim(0, 1)
        self._apply_legend(ax2)
        ax2.grid(True, alpha=0.3)

        ax3.set_xlabel("Epoch", fontsize=xlabel_fontsize)
        ax3.set_ylabel("LPIPS", fontsize=ylabel_fontsize)
        ax3.set_ylim(0, 1)
        self._apply_legend(ax3)
        ax3.grid(True, alpha=0.3)

    def _on_export_chart(self):
        """Export the current chart to PNG."""
        if not self._tests:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Chart",
            "training_curves.png",
            "PNG Files (*.png);;PDF Files (*.pdf);;All Files (*.*)"
        )

        if not file_path:
            return

        try:
            self.figure.savefig(file_path, dpi=150, bbox_inches='tight',
                               facecolor='white', edgecolor='none')
            self.logger.info("Exported chart to %s", file_path)
            QMessageBox.information(
                self, "Export Complete",
                f"Chart exported to:\n{file_path}"
            )
        except Exception as e:
            self.logger.error("Failed to export chart: %s", e)
            QMessageBox.warning(self, "Export Error", f"Failed to export chart:\n{e}")

    def clear(self):
        """Clear all data from the view."""
        self._tests = []
        self.figure.clear()
        self.canvas.draw()
        self.info_label.setText("Load experiments to see training curves")
        self.summary_btn.setEnabled(False)
        self.preview_arch_btn.setEnabled(False)
