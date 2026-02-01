"""
Training Summary Popup for Batch Reports Training View.

Displays detailed training information for a selected experiment including
hyperparameters and training curves (Loss Evolution, Quality Metrics Evolution).
"""
import logging
from typing import List, Dict, Any, Optional

import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QComboBox,
    QGroupBox, QGridLayout, QSplitter, QScrollArea, QFrame,
    QSizePolicy, QFileDialog, QMessageBox, QMenu
)
from PyQt5.QtCore import Qt

from ui.custom_widgets.batch_reports.comparison_views.chart_config_popup import (
    ChartConfigPopup, CustomNavigationToolbar
)


class TrainingSummaryPopup(QDialog):
    """
    Popup dialog showing detailed training summary for a selected experiment.

    Features:
    - Experiment selector combobox
    - Hyperparameters table
    - Loss Evolution chart
    - Quality Metrics Evolution chart (PSNR, SSIM, LPIPS)
    """

    def __init__(self, tests: List[Dict[str, Any]], parent=None, logger=None):
        super().__init__(parent)

        self.logger = logger or logging.getLogger(__name__)
        self._tests = [t for t in tests if t.get("training_curves")]

        # Separate chart configurations for each chart (independent)
        self._loss_chart_config = self._create_default_config()
        self._quality_chart_config = self._create_default_config()

        self.setWindowTitle("Training Summary")
        self.setMinimumSize(900, 700)
        self.resize(1000, 750)
        self.setModal(True)

        self._setup_ui()

        # Select first experiment if available
        if self._tests:
            self._on_experiment_changed(0)

    def _setup_ui(self):
        """Setup the popup UI."""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Experiment selector
        selector_layout = QHBoxLayout()
        selector_layout.addWidget(QLabel("Experiment:"))

        self.experiment_combo = QComboBox()
        self.experiment_combo.setMinimumWidth(300)
        for test in self._tests:
            name = test.get("name", "Unknown")
            exp_name = test.get("_experiment_name", "")
            display = f"{name}" + (f" ({exp_name})" if exp_name else "")
            self.experiment_combo.addItem(display)
        self.experiment_combo.currentIndexChanged.connect(self._on_experiment_changed)
        selector_layout.addWidget(self.experiment_combo)
        selector_layout.addStretch()

        main_layout.addLayout(selector_layout)

        # Main content splitter (horizontal: left=hyperparams, right=charts)
        splitter = QSplitter(Qt.Horizontal)

        # Left panel: Hyperparameters
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.hyperparams_group = QGroupBox("Training Hyperparameters")
        self.hyperparams_group.setMinimumWidth(280)
        self.hyperparams_group.setMaximumWidth(350)
        hyperparams_scroll = QScrollArea()
        hyperparams_scroll.setWidgetResizable(True)
        hyperparams_scroll.setFrameShape(QFrame.NoFrame)

        self.hyperparams_widget = QWidget()
        self.hyperparams_layout = QGridLayout(self.hyperparams_widget)
        self.hyperparams_layout.setSpacing(8)
        self.hyperparams_layout.setAlignment(Qt.AlignTop)
        hyperparams_scroll.setWidget(self.hyperparams_widget)

        group_layout = QVBoxLayout(self.hyperparams_group)
        group_layout.addWidget(hyperparams_scroll)

        left_layout.addWidget(self.hyperparams_group)
        splitter.addWidget(left_panel)

        # Right panel: Charts
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(5)

        # Loss Evolution chart
        loss_group = QGroupBox("Loss Evolution")
        loss_layout = QVBoxLayout(loss_group)
        loss_layout.setContentsMargins(5, 5, 5, 5)

        self.loss_figure = Figure(figsize=(8, 3), dpi=100)
        self.loss_canvas = FigureCanvas(self.loss_figure)
        self.loss_canvas.setContextMenuPolicy(Qt.CustomContextMenu)
        self.loss_canvas.customContextMenuRequested.connect(
            lambda pos: self._show_chart_context_menu(pos, self.loss_figure, "loss_evolution"))
        self.loss_toolbar = CustomNavigationToolbar(
            self.loss_canvas, self,
            config_callback=self._on_open_loss_chart_config
        )

        loss_layout.addWidget(self.loss_toolbar)
        loss_layout.addWidget(self.loss_canvas)
        right_layout.addWidget(loss_group)

        # Quality Metrics Evolution chart
        quality_group = QGroupBox("Quality Metrics Evolution")
        quality_layout = QVBoxLayout(quality_group)
        quality_layout.setContentsMargins(5, 5, 5, 5)

        self.quality_figure = Figure(figsize=(8, 4), dpi=100)
        self.quality_canvas = FigureCanvas(self.quality_figure)
        self.quality_canvas.setContextMenuPolicy(Qt.CustomContextMenu)
        self.quality_canvas.customContextMenuRequested.connect(
            lambda pos: self._show_chart_context_menu(pos, self.quality_figure, "quality_metrics"))
        self.quality_toolbar = CustomNavigationToolbar(
            self.quality_canvas, self,
            config_callback=self._on_open_quality_chart_config
        )

        quality_layout.addWidget(self.quality_toolbar)
        quality_layout.addWidget(self.quality_canvas)
        right_layout.addWidget(quality_group)

        splitter.addWidget(right_panel)

        # Set splitter sizes (hyperparams:charts = 1:3)
        splitter.setSizes([300, 700])

        main_layout.addWidget(splitter, 1)

    def _show_chart_context_menu(self, pos, figure: Figure, chart_name: str):
        """Show context menu for saving chart."""
        menu = QMenu(self)
        save_action = menu.addAction("Save chart as...")

        if chart_name == "loss_evolution":
            canvas = self.loss_canvas
        else:
            canvas = self.quality_canvas

        action = menu.exec_(canvas.mapToGlobal(pos))
        if action == save_action:
            self._save_chart(figure, chart_name)

    def _save_chart(self, figure: Figure, default_name: str):
        """Save chart to file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Chart",
            f"{default_name}.png",
            "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg);;All Files (*.*)"
        )

        if not file_path:
            return

        try:
            figure.savefig(file_path, dpi=150, bbox_inches='tight',
                          facecolor='white', edgecolor='none')
            self.logger.info("Saved chart to %s", file_path)
        except Exception as e:
            self.logger.error("Failed to save chart: %s", e)
            QMessageBox.warning(self, "Save Error", f"Failed to save chart:\n{e}")

    def _on_experiment_changed(self, index: int):
        """Handle experiment selection change."""
        if index < 0 or index >= len(self._tests):
            return

        test = self._tests[index]
        self._update_hyperparameters(test)
        self._update_charts(test)

    def _create_default_config(self) -> Dict[str, Any]:
        """Create a default chart configuration."""
        return {
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
                'ymax': 1.0,
            },
            'legend': {
                'position': 0,
                'fontsize': 9,
                'frameon': True,
                'shadow': False,
                'fancybox': True,
                'framealpha': 0.8,
                'ncol': 1,
            },
            'lines': {
                'linestyle': '-',
                'linewidth': 1.5,
                'marker': '',
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

    def _on_open_loss_chart_config(self):
        """Open chart configuration dialog for Loss Evolution chart."""
        popup = ChartConfigPopup(parent=self, logger=self.logger)
        popup.set_config(self._loss_chart_config)

        if popup.exec_() == QDialog.Accepted:
            self._loss_chart_config = popup.get_config()
            self.logger.debug("Loss chart config updated: %s", self._loss_chart_config)
            # Re-draw only loss chart
            index = self.experiment_combo.currentIndex()
            if index >= 0 and index < len(self._tests):
                curves = self._tests[index].get("training_curves", {})
                self._draw_loss_chart(curves)

    def _on_open_quality_chart_config(self):
        """Open chart configuration dialog for Quality Metrics chart."""
        popup = ChartConfigPopup(parent=self, logger=self.logger)
        popup.set_config(self._quality_chart_config)

        if popup.exec_() == QDialog.Accepted:
            self._quality_chart_config = popup.get_config()
            self.logger.debug("Quality chart config updated: %s", self._quality_chart_config)
            # Re-draw only quality chart
            index = self.experiment_combo.currentIndex()
            if index >= 0 and index < len(self._tests):
                curves = self._tests[index].get("training_curves", {})
                self._draw_quality_chart(curves)

    def _update_hyperparameters(self, test: Dict[str, Any]):
        """Update the hyperparameters display."""
        # Clear existing widgets
        while self.hyperparams_layout.count():
            item = self.hyperparams_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Extract hyperparameters from test config
        config = test.get("config", {})
        training_config = config.get("training", {})
        model_config = config.get("model", {})

        # Build hyperparameters dict
        hyperparams = {}

        # Model info
        if model_config:
            hyperparams["Model"] = model_config.get("name", "-")

        # Training params
        if training_config:
            if "epochs" in training_config:
                hyperparams["Epochs"] = training_config["epochs"]
            if "batch_size" in training_config:
                hyperparams["Batch Size"] = training_config["batch_size"]
            if "learning_rate" in training_config or "lr" in training_config:
                lr = training_config.get("learning_rate", training_config.get("lr"))
                hyperparams["Learning Rate"] = f"{lr:.2e}" if lr else "-"
            if "weight_decay" in training_config:
                wd = training_config["weight_decay"]
                hyperparams["Weight Decay"] = f"{wd:.2e}" if wd else "0"
            if "optimizer" in training_config:
                hyperparams["Optimizer"] = training_config["optimizer"]
            if "loss_function" in training_config or "loss" in training_config:
                hyperparams["Loss Function"] = training_config.get("loss_function",
                                                                    training_config.get("loss", "-"))

        # Also check direct test keys for backward compatibility
        direct_keys = [
            ("model_name", "Model"),
            ("epochs", "Epochs"),
            ("batch_size", "Batch Size"),
            ("learning_rate", "Learning Rate"),
            ("lr", "Learning Rate"),
            ("weight_decay", "Weight Decay"),
            ("optimizer", "Optimizer"),
        ]

        for key, display_name in direct_keys:
            if key in test and display_name not in hyperparams:
                value = test[key]
                if key in ["learning_rate", "lr", "weight_decay"] and value:
                    hyperparams[display_name] = f"{value:.2e}"
                else:
                    hyperparams[display_name] = value

        # Training results from curves
        curves = test.get("training_curves", {})
        if curves:
            val_losses = curves.get("val_losses", [])
            if val_losses:
                hyperparams["Final Val Loss"] = f"{val_losses[-1]:.4f}"
                hyperparams["Best Val Loss"] = f"{min(val_losses):.4f}"
                hyperparams["Total Epochs"] = len(val_losses)

            val_psnr = curves.get("val_psnr", [])
            if val_psnr:
                hyperparams["Final PSNR"] = f"{val_psnr[-1]:.2f} dB"
                hyperparams["Best PSNR"] = f"{max(val_psnr):.2f} dB"

            val_ssim = curves.get("val_ssim", [])
            if val_ssim:
                hyperparams["Final SSIM"] = f"{val_ssim[-1]:.4f}"
                hyperparams["Best SSIM"] = f"{max(val_ssim):.4f}"

        # Display hyperparameters
        row = 0
        for name, value in hyperparams.items():
            name_label = QLabel(f"{name}:")
            name_label.setStyleSheet("font-weight: bold; color: #333;")

            value_label = QLabel(str(value))
            value_label.setStyleSheet("color: #555;")
            value_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

            self.hyperparams_layout.addWidget(name_label, row, 0, Qt.AlignRight)
            self.hyperparams_layout.addWidget(value_label, row, 1, Qt.AlignLeft)
            row += 1

        # Add stretch at the end
        self.hyperparams_layout.setRowStretch(row, 1)

    def _update_charts(self, test: Dict[str, Any]):
        """Update both charts for the selected experiment."""
        curves = test.get("training_curves", {})

        self._draw_loss_chart(curves)
        self._draw_quality_chart(curves)

    def _draw_loss_chart(self, curves: Dict[str, Any]):
        """Draw the Loss Evolution chart."""
        self.loss_figure.clear()
        ax = self.loss_figure.add_subplot(111)

        val_losses = curves.get("val_losses", [])
        test_losses = curves.get("test_losses", [])

        if not val_losses and not test_losses:
            ax.text(0.5, 0.5, "No loss data available",
                   ha='center', va='center', fontsize=12, color='#999')
            ax.axis('off')
            self.loss_canvas.draw()
            return

        lines_cfg = self._loss_chart_config.get('lines', {})
        linestyle = lines_cfg.get('linestyle', '-')
        linewidth = lines_cfg.get('linewidth', 1.5)
        marker = lines_cfg.get('marker', '')
        markersize = lines_cfg.get('markersize', 4)

        axes_cfg = self._loss_chart_config.get('axes', {})
        legend_cfg = self._loss_chart_config.get('legend', {})

        if val_losses:
            epochs = range(1, len(val_losses) + 1)
            ax.plot(epochs, val_losses, linestyle=linestyle, marker=marker,
                   label='Validation Loss', color='tab:blue',
                   linewidth=linewidth, markersize=markersize)

        if test_losses:
            epochs = range(1, len(test_losses) + 1)
            ax.plot(epochs, test_losses, linestyle='--', marker=marker,
                   label='Test Loss', color='tab:orange',
                   linewidth=linewidth, markersize=markersize, alpha=0.8)

        ax.set_xlabel('Epoch', fontsize=axes_cfg.get('xlabel_fontsize', 10))
        ax.set_ylabel('Loss', fontsize=axes_cfg.get('ylabel_fontsize', 10))
        ax.set_title('Loss Evolution', fontsize=axes_cfg.get('title_fontsize', 12), fontweight='bold')

        # Apply legend config
        legend_pos = legend_cfg.get('position', 0)
        loc_map = {0: 'upper right', 1: 'upper left', 2: 'lower right', 3: 'lower left'}
        legend_kwargs = {
            'fontsize': legend_cfg.get('fontsize', 9),
            'frameon': legend_cfg.get('frameon', True),
            'shadow': legend_cfg.get('shadow', False),
            'fancybox': legend_cfg.get('fancybox', True),
            'framealpha': legend_cfg.get('framealpha', 0.8),
        }
        if legend_pos in loc_map:
            ax.legend(loc=loc_map[legend_pos], **legend_kwargs)
        elif legend_pos == 4:
            ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), **legend_kwargs)
        else:
            legend_kwargs['frameon'] = False
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), **legend_kwargs)

        ax.grid(True, alpha=0.3)

        self.loss_figure.tight_layout()
        self.loss_canvas.draw()

    def _draw_quality_chart(self, curves: Dict[str, Any]):
        """Draw the Quality Metrics Evolution chart with multiple y-axes."""
        self.quality_figure.clear()

        val_psnr = curves.get("val_psnr", [])
        val_ssim = curves.get("val_ssim", [])
        val_lpips = curves.get("val_lpips", [])

        has_data = val_psnr or val_ssim or val_lpips

        if not has_data:
            ax = self.quality_figure.add_subplot(111)
            ax.text(0.5, 0.5, "No quality metrics data available",
                   ha='center', va='center', fontsize=12, color='#999')
            ax.axis('off')
            self.quality_canvas.draw()
            return

        lines_cfg = self._quality_chart_config.get('lines', {})
        linestyle = lines_cfg.get('linestyle', '-')
        linewidth = lines_cfg.get('linewidth', 1.5)
        marker = lines_cfg.get('marker', '')
        markersize = lines_cfg.get('markersize', 4)

        axes_cfg = self._quality_chart_config.get('axes', {})
        legend_cfg = self._quality_chart_config.get('legend', {})
        colors_cfg = self._quality_chart_config.get('colors', {})

        ax = self.quality_figure.add_subplot(111)
        lines = []
        labels = []

        ylabel_fontsize = axes_cfg.get('ylabel_fontsize', 10)

        # PSNR (primary y-axis)
        if val_psnr:
            epochs = range(1, len(val_psnr) + 1)
            psnr_color = colors_cfg.get('psnr', 'tab:green')
            line, = ax.plot(epochs, val_psnr, linestyle=linestyle, marker=marker,
                           label='PSNR (dB)', color=psnr_color,
                           linewidth=linewidth, markersize=markersize)
            lines.append(line)
            labels.append('PSNR (dB)')
            ax.set_ylabel('PSNR (dB)', color=psnr_color, fontsize=ylabel_fontsize)
            ax.tick_params(axis='y', labelcolor=psnr_color)

        ax.set_xlabel('Epoch', fontsize=axes_cfg.get('xlabel_fontsize', 10))
        ax.set_title('Quality Metrics Evolution', fontsize=axes_cfg.get('title_fontsize', 12), fontweight='bold')
        ax.grid(True, alpha=0.3)

        # SSIM (secondary y-axis)
        if val_ssim:
            ax_ssim = ax.twinx()
            epochs = range(1, len(val_ssim) + 1)
            ssim_color = colors_cfg.get('ssim', 'tab:red')
            line, = ax_ssim.plot(epochs, val_ssim, linestyle='--', marker=marker,
                                label='SSIM', color=ssim_color,
                                linewidth=linewidth, markersize=markersize)
            lines.append(line)
            labels.append('SSIM')
            ax_ssim.set_ylabel('SSIM', color=ssim_color, fontsize=ylabel_fontsize)
            ax_ssim.tick_params(axis='y', labelcolor=ssim_color)
            ax_ssim.set_ylim(0, 1)

        # LPIPS (third y-axis)
        if val_lpips:
            ax_lpips = ax.twinx()
            # Offset the third axis (smaller offset to keep it closer)
            ax_lpips.spines['right'].set_position(('axes', 1.06))
            epochs = range(1, len(val_lpips) + 1)
            lpips_color = colors_cfg.get('lpips', 'tab:purple')
            line, = ax_lpips.plot(epochs, val_lpips, linestyle=':', marker=marker,
                                 label='LPIPS', color=lpips_color,
                                 linewidth=linewidth, markersize=markersize)
            lines.append(line)
            labels.append('LPIPS')
            ax_lpips.set_ylabel('LPIPS', color=lpips_color, fontsize=ylabel_fontsize)
            ax_lpips.tick_params(axis='y', labelcolor=lpips_color)
            ax_lpips.set_ylim(0, 1)

        # Combined legend with config
        legend_pos = legend_cfg.get('position', 0)
        loc_map = {0: 'upper right', 1: 'upper left', 2: 'lower right', 3: 'lower left'}
        legend_kwargs = {
            'fontsize': legend_cfg.get('fontsize', 9),
            'frameon': legend_cfg.get('frameon', True),
            'shadow': legend_cfg.get('shadow', False),
            'fancybox': legend_cfg.get('fancybox', True),
            'framealpha': legend_cfg.get('framealpha', 0.8),
        }
        if legend_pos in loc_map:
            ax.legend(lines, labels, loc=loc_map[legend_pos], **legend_kwargs)
        elif legend_pos == 4:
            # Offset legend to account for multiple y-axes
            ax.legend(lines, labels, loc='center left', bbox_to_anchor=(1.18, 0.5), **legend_kwargs)
        else:
            legend_kwargs['frameon'] = False
            ax.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), **legend_kwargs)

        self.quality_figure.tight_layout()
        self.quality_canvas.draw()
