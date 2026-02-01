"""Quality Metrics page widget with report generation and image preview."""
import logging
import numpy as np
from matplotlib import cm
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QCheckBox, QGridLayout, QSlider, QSizePolicy,
    QGraphicsView, QGraphicsScene, QFrame, QSplitter, QScrollArea,
    QMenu, QFileDialog
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont

from ui.custom_widgets.quality_metrics.quality_report_popup import QualityReportPopup


class QualityMetricsPage(QWidget):
    """
    Quality Metrics page with two main sections:
    - Left: Report generation controls (metric selection + generate button)
    - Right: Image preview with per-image metrics display
    """

    # Signal emitted when analysis is requested
    analysisRequested = pyqtSignal()

    def __init__(self, parent=None, logger=None):
        super().__init__(parent)

        if logger:
            self.logger = logger.getChild("QualityMetricsPage")
        else:
            self.logger = logging.getLogger("SPIm.QualityMetricsPage")

        # Data storage
        self._orig_images = []
        self._noisy_images = []
        self._recon_images = []
        self._psnr_noisy = []
        self._ssim_noisy = []
        self._lpips_noisy = []
        self._psnr_recon = []
        self._ssim_recon = []
        self._lpips_recon = []

        # Thermal colormap for images
        self.cmap = cm.get_cmap('hot')

        self._setup_ui()

    def _setup_ui(self):
        """Setup the main UI layout."""
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(10)

        # Use splitter for resizable sections
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        # Left section: Report Generation
        left_widget = self._create_report_section()
        splitter.addWidget(left_widget)

        # Right section: Image Preview with Metrics
        right_widget = self._create_preview_section()
        splitter.addWidget(right_widget)

        # Set initial sizes (roughly 1:2 ratio)
        splitter.setSizes([350, 700])

        main_layout.addWidget(splitter)

    def _create_report_section(self):
        """Create the left section for report generation."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)

        # Title
        title = QLabel("<h3>Quality Report Generation</h3>")
        layout.addWidget(title)

        # Metrics selection group
        metrics_group = QGroupBox()
        metrics_layout = QVBoxLayout(metrics_group)
        metrics_layout.setSpacing(10)

        self.psnr_checkbox = QCheckBox("PSNR (Peak Signal-to-Noise Ratio) - higher is better")
        self.psnr_checkbox.setChecked(True)
        metrics_layout.addWidget(self.psnr_checkbox)

        self.ssim_checkbox = QCheckBox("SSIM (Structural Similarity Index) - higher is better")
        self.ssim_checkbox.setChecked(True)
        metrics_layout.addWidget(self.ssim_checkbox)

        self.lpips_checkbox = QCheckBox("LPIPS (Learned Perceptual Image Patch Similarity) - lower is better")
        self.lpips_checkbox.setChecked(True)
        metrics_layout.addWidget(self.lpips_checkbox)

        layout.addWidget(metrics_group)

        # Description
        desc_label = QLabel(
            "<i>Select the metrics you want to include in the report. "
            "The report will show per-image values and distributions.</i>"
        )
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #666;")
        layout.addWidget(desc_label)

        # Run analysis button (green - same style as Timing Analysis)
        self.analyze_button = QPushButton("Run Analysis")
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

        # Generate button (blue - same style as Timing Analysis)
        self.generate_button = QPushButton("Generate Quality Report")
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
        self.status_label = QLabel("Load data and run analysis first")
        self.status_label.setStyleSheet("color: #888;")
        layout.addWidget(self.status_label)

        layout.addStretch()

        return container

    def _create_preview_section(self):
        """Create the right section for image preview with metrics."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Title
        title = QLabel("<h3>Quality Metrics Preview</h3>")
        layout.addWidget(title)

        # Images section (top) - make it square-ish by setting larger height
        self.images_group = QGroupBox()
        self.images_group.setMinimumHeight(250)
        self.images_group.setContextMenuPolicy(Qt.CustomContextMenu)
        self.images_group.customContextMenuRequested.connect(self._show_images_context_menu)
        images_layout = QHBoxLayout(self.images_group)
        images_layout.setSpacing(10)

        # Ground-Truth image
        orig_container = QVBoxLayout()
        orig_label = QLabel("Ground-Truth")
        orig_label.setAlignment(Qt.AlignCenter)
        label_font = QFont()
        label_font.setPointSize(10)
        orig_label.setFont(label_font)
        orig_container.addWidget(orig_label)
        self.orig_view = QGraphicsView()
        self.orig_view.setMinimumSize(150, 200)
        self.orig_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.orig_scene = QGraphicsScene()
        self.orig_view.setScene(self.orig_scene)
        orig_container.addWidget(self.orig_view)
        images_layout.addLayout(orig_container)

        # Noisy image
        noisy_container = QVBoxLayout()
        noisy_label = QLabel("Noisy (Input)")
        noisy_label.setAlignment(Qt.AlignCenter)
        noisy_label.setFont(label_font)
        noisy_container.addWidget(noisy_label)
        self.noisy_view = QGraphicsView()
        self.noisy_view.setMinimumSize(150, 200)
        self.noisy_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.noisy_scene = QGraphicsScene()
        self.noisy_view.setScene(self.noisy_scene)
        noisy_container.addWidget(self.noisy_view)
        images_layout.addLayout(noisy_container)

        # Denoised image
        recon_container = QVBoxLayout()
        recon_label = QLabel("Denoised")
        recon_label.setAlignment(Qt.AlignCenter)
        recon_label.setFont(label_font)
        recon_container.addWidget(recon_label)
        self.recon_view = QGraphicsView()
        self.recon_view.setMinimumSize(150, 200)
        self.recon_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.recon_scene = QGraphicsScene()
        self.recon_view.setScene(self.recon_scene)
        recon_container.addWidget(self.recon_view)
        images_layout.addLayout(recon_container)

        layout.addWidget(self.images_group)

        # Slider for image navigation
        slider_layout = QHBoxLayout()
        slider_layout.setSpacing(10)

        self.slider_label = QLabel("Image:")
        slider_layout.addWidget(self.slider_label)

        self.image_slider = QSlider(Qt.Horizontal)
        self.image_slider.setMinimum(0)
        self.image_slider.setMaximum(0)
        self.image_slider.setValue(0)
        self.image_slider.valueChanged.connect(self._on_slider_changed)
        slider_layout.addWidget(self.image_slider, 1)

        self.index_label = QLabel("0  (0 images)")
        self.index_label.setMinimumWidth(100)
        slider_layout.addWidget(self.index_label)

        layout.addLayout(slider_layout)

        # Metrics section with table and bar chart side by side
        self.metrics_group = QGroupBox("Quality Metrics for Current Image")
        self.metrics_group.setContextMenuPolicy(Qt.CustomContextMenu)
        self.metrics_group.customContextMenuRequested.connect(self._show_metrics_context_menu)
        metrics_main_layout = QHBoxLayout(self.metrics_group)
        metrics_main_layout.setSpacing(15)

        # Left side: Table with values
        table_widget = QWidget()
        metrics_layout = QGridLayout(table_widget)
        metrics_layout.setSpacing(10)

        # Headers
        header_font = QFont()
        header_font.setBold(True)

        metrics_layout.addWidget(QLabel(""), 0, 0)
        noisy_header = QLabel("Noisy")
        noisy_header.setFont(header_font)
        noisy_header.setAlignment(Qt.AlignCenter)
        metrics_layout.addWidget(noisy_header, 0, 1)
        recon_header = QLabel("Denoised")
        recon_header.setFont(header_font)
        recon_header.setAlignment(Qt.AlignCenter)
        metrics_layout.addWidget(recon_header, 0, 2)
        change_header = QLabel("Change")
        change_header.setFont(header_font)
        change_header.setAlignment(Qt.AlignCenter)
        metrics_layout.addWidget(change_header, 0, 3)

        # PSNR row (higher is better)
        self.psnr_row_label = QLabel("PSNR (dB) \u2191:")  # ↑ arrow
        self.psnr_row_label.setFont(header_font)
        self.psnr_row_label.setToolTip("Higher is better")
        self.psnr_noisy_display = QLabel("-")
        self.psnr_noisy_display.setAlignment(Qt.AlignCenter)
        self.psnr_noisy_display.setStyleSheet("font-size: 14px;")
        self.psnr_recon_display = QLabel("-")
        self.psnr_recon_display.setAlignment(Qt.AlignCenter)
        self.psnr_recon_display.setStyleSheet("font-size: 14px;")
        self.psnr_change_display = QLabel("-")
        self.psnr_change_display.setAlignment(Qt.AlignCenter)
        self.psnr_change_display.setStyleSheet("font-size: 14px; font-weight: bold;")
        metrics_layout.addWidget(self.psnr_row_label, 1, 0)
        metrics_layout.addWidget(self.psnr_noisy_display, 1, 1)
        metrics_layout.addWidget(self.psnr_recon_display, 1, 2)
        metrics_layout.addWidget(self.psnr_change_display, 1, 3)

        # SSIM row (higher is better)
        self.ssim_row_label = QLabel("SSIM \u2191:")  # ↑ arrow
        self.ssim_row_label.setFont(header_font)
        self.ssim_row_label.setToolTip("Higher is better")
        self.ssim_noisy_display = QLabel("-")
        self.ssim_noisy_display.setAlignment(Qt.AlignCenter)
        self.ssim_noisy_display.setStyleSheet("font-size: 14px;")
        self.ssim_recon_display = QLabel("-")
        self.ssim_recon_display.setAlignment(Qt.AlignCenter)
        self.ssim_recon_display.setStyleSheet("font-size: 14px;")
        self.ssim_change_display = QLabel("-")
        self.ssim_change_display.setAlignment(Qt.AlignCenter)
        self.ssim_change_display.setStyleSheet("font-size: 14px; font-weight: bold;")
        metrics_layout.addWidget(self.ssim_row_label, 2, 0)
        metrics_layout.addWidget(self.ssim_noisy_display, 2, 1)
        metrics_layout.addWidget(self.ssim_recon_display, 2, 2)
        metrics_layout.addWidget(self.ssim_change_display, 2, 3)

        # LPIPS row (lower is better)
        self.lpips_row_label = QLabel("LPIPS \u2193:")  # ↓ arrow
        self.lpips_row_label.setFont(header_font)
        self.lpips_row_label.setToolTip("Lower is better")
        self.lpips_noisy_display = QLabel("-")
        self.lpips_noisy_display.setAlignment(Qt.AlignCenter)
        self.lpips_noisy_display.setStyleSheet("font-size: 14px;")
        self.lpips_recon_display = QLabel("-")
        self.lpips_recon_display.setAlignment(Qt.AlignCenter)
        self.lpips_recon_display.setStyleSheet("font-size: 14px;")
        self.lpips_change_display = QLabel("-")
        self.lpips_change_display.setAlignment(Qt.AlignCenter)
        self.lpips_change_display.setStyleSheet("font-size: 14px; font-weight: bold;")
        metrics_layout.addWidget(self.lpips_row_label, 3, 0)
        metrics_layout.addWidget(self.lpips_noisy_display, 3, 1)
        metrics_layout.addWidget(self.lpips_recon_display, 3, 2)
        metrics_layout.addWidget(self.lpips_change_display, 3, 3)

        metrics_main_layout.addWidget(table_widget)

        # Right side: Bar chart
        self.metrics_bar_figure = Figure(figsize=(4, 3), dpi=100)
        self.metrics_bar_canvas = FigureCanvas(self.metrics_bar_figure)
        self.metrics_bar_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.metrics_bar_canvas.setMinimumSize(200, 150)
        metrics_main_layout.addWidget(self.metrics_bar_canvas)

        layout.addWidget(self.metrics_group)

        # Connect checkboxes to update visibility
        self.psnr_checkbox.toggled.connect(self._update_metrics_visibility)
        self.ssim_checkbox.toggled.connect(self._update_metrics_visibility)
        self.lpips_checkbox.toggled.connect(self._update_metrics_visibility)
        # Also update bar chart when checkboxes change
        self.psnr_checkbox.toggled.connect(self._update_bar_chart)
        self.ssim_checkbox.toggled.connect(self._update_bar_chart)
        self.lpips_checkbox.toggled.connect(self._update_bar_chart)

        layout.addStretch()

        return container

    def _update_metrics_visibility(self):
        """Update visibility of metric rows based on checkbox selection."""
        show_psnr = self.psnr_checkbox.isChecked()
        show_ssim = self.ssim_checkbox.isChecked()
        show_lpips = self.lpips_checkbox.isChecked()

        self.psnr_row_label.setVisible(show_psnr)
        self.psnr_noisy_display.setVisible(show_psnr)
        self.psnr_recon_display.setVisible(show_psnr)
        self.psnr_change_display.setVisible(show_psnr)

        self.ssim_row_label.setVisible(show_ssim)
        self.ssim_noisy_display.setVisible(show_ssim)
        self.ssim_recon_display.setVisible(show_ssim)
        self.ssim_change_display.setVisible(show_ssim)

        self.lpips_row_label.setVisible(show_lpips)
        self.lpips_noisy_display.setVisible(show_lpips)
        self.lpips_recon_display.setVisible(show_lpips)
        self.lpips_change_display.setVisible(show_lpips)

    def set_data(self, orig_images, noisy_images, recon_images,
                 psnr_noisy, ssim_noisy, lpips_noisy,
                 psnr_recon, ssim_recon, lpips_recon):
        """
        Set the image and metrics data.

        Args:
            orig_images: List of original images
            noisy_images: List of noisy/input images
            recon_images: List of reconstructed images
            psnr_noisy: List of PSNR values (noisy vs original)
            ssim_noisy: List of SSIM values (noisy vs original)
            lpips_noisy: List of LPIPS values (noisy vs original)
            psnr_recon: List of PSNR values (reconstructed vs original)
            ssim_recon: List of SSIM values (reconstructed vs original)
            lpips_recon: List of LPIPS values (reconstructed vs original)
        """
        self._orig_images = list(orig_images) if orig_images else []
        self._noisy_images = list(noisy_images) if noisy_images else []
        self._recon_images = list(recon_images) if recon_images else []
        self._psnr_noisy = list(psnr_noisy) if psnr_noisy else []
        self._ssim_noisy = list(ssim_noisy) if ssim_noisy else []
        self._lpips_noisy = list(lpips_noisy) if lpips_noisy else []
        self._psnr_recon = list(psnr_recon) if psnr_recon else []
        self._ssim_recon = list(ssim_recon) if ssim_recon else []
        self._lpips_recon = list(lpips_recon) if lpips_recon else []

        n = len(self._orig_images)
        self.image_slider.setMaximum(max(0, n - 1))
        self.image_slider.setValue(0)
        self.index_label.setText(f"0  ({n} images)")

        if n > 0:
            self.status_label.setText(f"Loaded {n} images with metrics")
            self.status_label.setStyleSheet("color: #080;")
            self.generate_button.setEnabled(True)
            self._on_slider_changed(0)
        else:
            self.status_label.setText("No data available")
            self.status_label.setStyleSheet("color: #888;")
            self.generate_button.setEnabled(False)

        self.logger.debug(f"Data set: {n} images")

    def _format_change(self, noisy_val, recon_val, higher_is_better=True):
        """Format the percentage change between noisy and reconstructed values.

        Args:
            noisy_val: Value for noisy image
            recon_val: Value for reconstructed image
            higher_is_better: If True, positive change is good (PSNR, SSIM).
                              If False, negative change is good (LPIPS).

        Returns:
            Tuple of (text, stylesheet) for the change label.
        """
        if noisy_val is None or recon_val is None or noisy_val == 0:
            return "-", "font-size: 14px; font-weight: bold;"

        # Calculate percentage change
        pct_change = ((recon_val - noisy_val) / abs(noisy_val)) * 100

        # Determine if change is improvement
        if higher_is_better:
            is_improvement = pct_change > 0
        else:
            is_improvement = pct_change < 0

        # Format text and color
        if pct_change >= 0:
            text = f"+{pct_change:.1f}%"
        else:
            text = f"{pct_change:.1f}%"

        if is_improvement:
            style = "font-size: 14px; font-weight: bold; color: #228B22;"  # Green
        else:
            style = "font-size: 14px; font-weight: bold; color: #DC143C;"  # Red

        return text, style

    def _on_slider_changed(self, idx):
        """Handle slider value change to update displayed image and metrics."""
        n = len(self._orig_images)
        self.index_label.setText(f"{idx}  ({n} images)")

        if not (0 <= idx < n):
            return

        # Update images
        self._display_image(self._orig_images[idx], self.orig_scene, self.orig_view)
        self._display_image(self._noisy_images[idx], self.noisy_scene, self.noisy_view)
        if idx < len(self._recon_images):
            self._display_image(self._recon_images[idx], self.recon_scene, self.recon_view)

        # Update PSNR metrics and change
        psnr_n = self._psnr_noisy[idx] if idx < len(self._psnr_noisy) else None
        psnr_r = self._psnr_recon[idx] if idx < len(self._psnr_recon) else None
        self.psnr_noisy_display.setText(f"{psnr_n:.2f}" if psnr_n is not None else "-")
        self.psnr_recon_display.setText(f"{psnr_r:.2f}" if psnr_r is not None else "-")
        psnr_change_text, psnr_change_style = self._format_change(psnr_n, psnr_r, higher_is_better=True)
        self.psnr_change_display.setText(psnr_change_text)
        self.psnr_change_display.setStyleSheet(psnr_change_style)

        # Update SSIM metrics and change
        ssim_n = self._ssim_noisy[idx] if idx < len(self._ssim_noisy) else None
        ssim_r = self._ssim_recon[idx] if idx < len(self._ssim_recon) else None
        self.ssim_noisy_display.setText(f"{ssim_n:.4f}" if ssim_n is not None else "-")
        self.ssim_recon_display.setText(f"{ssim_r:.4f}" if ssim_r is not None else "-")
        ssim_change_text, ssim_change_style = self._format_change(ssim_n, ssim_r, higher_is_better=True)
        self.ssim_change_display.setText(ssim_change_text)
        self.ssim_change_display.setStyleSheet(ssim_change_style)

        # Update LPIPS metrics and change (lower is better)
        lpips_n = self._lpips_noisy[idx] if idx < len(self._lpips_noisy) else None
        lpips_r = self._lpips_recon[idx] if idx < len(self._lpips_recon) else None
        self.lpips_noisy_display.setText(f"{lpips_n:.4f}" if lpips_n is not None else "-")
        self.lpips_recon_display.setText(f"{lpips_r:.4f}" if lpips_r is not None else "-")
        lpips_change_text, lpips_change_style = self._format_change(lpips_n, lpips_r, higher_is_better=False)
        self.lpips_change_display.setText(lpips_change_text)
        self.lpips_change_display.setStyleSheet(lpips_change_style)

        # Update bar chart
        self._update_bar_chart()

    def _update_bar_chart(self):
        """Update the bar chart showing selected metrics for current image.

        Shows a grouped bar chart with two groups (Noisy, Denoised) on X-axis,
        and bars for each selected metric within each group. Values are normalized
        to 0-1 scale for visual comparison (higher = better quality for all).
        """
        self.metrics_bar_figure.clear()

        idx = self.image_slider.value()
        n = len(self._orig_images)

        if not (0 <= idx < n):
            self.metrics_bar_canvas.draw()
            return

        # Collect selected metrics and their values
        show_psnr = self.psnr_checkbox.isChecked()
        show_ssim = self.ssim_checkbox.isChecked()
        show_lpips = self.lpips_checkbox.isChecked()

        # Count how many metrics are selected
        n_metrics = sum([show_psnr, show_ssim, show_lpips])
        if n_metrics == 0:
            self.metrics_bar_canvas.draw()
            return

        # Collect metric data: (name, noisy_val, recon_val, noisy_norm, recon_norm, color, format_str)
        metrics = []

        if show_psnr:
            psnr_n = self._psnr_noisy[idx] if idx < len(self._psnr_noisy) else 0
            psnr_r = self._psnr_recon[idx] if idx < len(self._psnr_recon) else 0
            # Normalize PSNR to 0-1 (assuming 0-50 dB range)
            psnr_n_norm = min(max(psnr_n / 50.0, 0), 1)
            psnr_r_norm = min(max(psnr_r / 50.0, 0), 1)
            metrics.append(('PSNR \u2191', psnr_n, psnr_r, psnr_n_norm, psnr_r_norm, '#1f77b4', '{:.1f}'))

        if show_ssim:
            ssim_n = self._ssim_noisy[idx] if idx < len(self._ssim_noisy) else 0
            ssim_r = self._ssim_recon[idx] if idx < len(self._ssim_recon) else 0
            # SSIM is already 0-1, higher is better
            metrics.append(('SSIM \u2191', ssim_n, ssim_r, ssim_n, ssim_r, '#2ca02c', '{:.3f}'))

        if show_lpips:
            lpips_n = self._lpips_noisy[idx] if idx < len(self._lpips_noisy) else 0
            lpips_r = self._lpips_recon[idx] if idx < len(self._lpips_recon) else 0
            # LPIPS: lower is better, so invert for visualization (1 - LPIPS)
            lpips_n_norm = 1.0 - min(max(lpips_n, 0), 1)
            lpips_r_norm = 1.0 - min(max(lpips_r, 0), 1)
            metrics.append(('LPIPS \u2193', lpips_n, lpips_r, lpips_n_norm, lpips_r_norm, '#d62728', '{:.3f}'))

        # Create single grouped bar chart
        ax = self.metrics_bar_figure.add_subplot(111)

        # X positions for groups
        x = np.arange(2)  # Two groups: Noisy, Denoised
        width = 0.8 / n_metrics  # Width of each bar

        # Plot bars for each metric
        for i, (name, noisy_val, recon_val, noisy_norm, recon_norm, color, fmt) in enumerate(metrics):
            offset = (i - (n_metrics - 1) / 2) * width
            bars = ax.bar(x + offset, [noisy_norm, recon_norm], width * 0.9,
                         label=name, color=color, alpha=0.8)

            # Add value labels on bars (show actual values, not normalized)
            ax.text(bars[0].get_x() + bars[0].get_width()/2, bars[0].get_height() + 0.02,
                   fmt.format(noisy_val), ha='center', va='bottom', fontsize=7)
            ax.text(bars[1].get_x() + bars[1].get_width()/2, bars[1].get_height() + 0.02,
                   fmt.format(recon_val), ha='center', va='bottom', fontsize=7)

        # Configure axes
        ax.set_xticks(x)
        ax.set_xticklabels(['Noisy', 'Denoised'], fontsize=9)
        ax.set_ylabel('Quality Score (normalized)', fontsize=9)
        ax.set_ylim(0, 1.15)  # Leave room for labels
        ax.set_title('Quality Metrics Comparison', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

        # Place legend outside the plot box (below the chart)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                  ncol=n_metrics, fontsize=8, frameon=False)

        # Use fixed margins instead of tight_layout to avoid deformation on redraw
        self.metrics_bar_figure.subplots_adjust(left=0.15, right=0.95, top=0.88, bottom=0.25)
        self.metrics_bar_canvas.draw()

    def _display_image(self, arr, scene, view):
        """Display an image in a graphics view with thermal colormap (no interpolation)."""
        scene.clear()
        arr = np.array(arr, copy=False)

        if arr.ndim == 0 or arr.size == 0:
            return

        # Normalize to [0, 1]
        amin, amax = arr.min(), arr.max()
        norm = (arr - amin) / (amax - amin) if amax > amin else np.zeros_like(arr, dtype=float)

        # Apply thermal colormap
        rgba = self.cmap(norm)
        rgb = (rgba[..., :3] * 255).astype(np.uint8)

        h, w = rgb.shape[:2]
        data = rgb.tobytes()
        bytes_per_line = 3 * w
        qimg = QImage(data, w, h, bytes_per_line, QImage.Format_RGB888)

        pix = QPixmap.fromImage(qimg)
        vw = view.viewport().width()
        vh = view.viewport().height()
        # Use FastTransformation to preserve pixels without interpolation
        pix = pix.scaled(vw, vh, Qt.KeepAspectRatio, Qt.FastTransformation)

        scene.addPixmap(pix)
        scene.setSceneRect(0, 0, pix.width(), pix.height())

    def _on_generate_report(self):
        """Generate and show the quality report popup."""
        if not self._orig_images:
            self.logger.warning("No data available for report generation")
            return

        # Get selected metrics
        selected = []
        if self.psnr_checkbox.isChecked():
            selected.append('psnr')
        if self.ssim_checkbox.isChecked():
            selected.append('ssim')
        if self.lpips_checkbox.isChecked():
            selected.append('lpips')

        if not selected:
            self.logger.warning("No metrics selected for report")
            return

        # Prepare metrics data
        metrics_data = {
            'psnr_noisy': self._psnr_noisy if self._psnr_noisy else None,
            'psnr_recon': self._psnr_recon if self._psnr_recon else None,
            'ssim_noisy': self._ssim_noisy if self._ssim_noisy else None,
            'ssim_recon': self._ssim_recon if self._ssim_recon else None,
            'lpips_noisy': self._lpips_noisy if self._lpips_noisy else None,
            'lpips_recon': self._lpips_recon if self._lpips_recon else None,
        }

        # Show popup
        popup = QualityReportPopup(self, logger=self.logger)
        popup.set_data(metrics_data, selected)
        popup.exec_()

        self.logger.info("Quality report displayed")

    def _show_images_context_menu(self, pos):
        """Show context menu for saving the 3 preview images."""
        menu = QMenu(self)
        save_action = menu.addAction("Save as...")
        save_action.triggered.connect(self._save_preview_images)
        menu.exec_(self.images_group.mapToGlobal(pos))

    def _show_metrics_context_menu(self, pos):
        """Show context menu for saving the metrics plot and table."""
        menu = QMenu(self)
        save_action = menu.addAction("Save as...")
        save_action.triggered.connect(self._save_metrics_plot)
        menu.exec_(self.metrics_group.mapToGlobal(pos))

    def _save_preview_images(self):
        """Save the 3 preview images (Ground-Truth, Noisy, Denoised) in one plot."""
        idx = self.image_slider.value()
        n = len(self._orig_images)

        if not (0 <= idx < n):
            self.logger.warning("No images to save")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Preview Images", "",
            "PNG Files (*.png);;PDF Files (*.pdf);;All Files (*.*)"
        )

        if not file_path:
            return

        try:
            # Create a figure with 3 subplots
            fig = Figure(figsize=(12, 4), dpi=150)

            # Get images
            orig_img = self._orig_images[idx] if idx < len(self._orig_images) else None
            noisy_img = self._noisy_images[idx] if idx < len(self._noisy_images) else None
            recon_img = self._recon_images[idx] if idx < len(self._recon_images) else None

            images = [
                ("Ground-Truth", orig_img),
                ("Noisy (Input)", noisy_img),
                ("Denoised", recon_img)
            ]

            for i, (title, img) in enumerate(images):
                ax = fig.add_subplot(1, 3, i + 1)
                if img is not None:
                    ax.imshow(img, cmap='hot')
                ax.set_title(title, fontsize=12)
                ax.axis('off')

            fig.suptitle(f"Image {idx}", fontsize=14)
            fig.tight_layout()
            fig.savefig(file_path, dpi=150, bbox_inches='tight')

            self.logger.info(f"Preview images saved to {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to save preview images: {e}")

    def _save_metrics_plot(self):
        """Save the metrics plot and table values."""
        idx = self.image_slider.value()
        n = len(self._orig_images)

        if not (0 <= idx < n):
            self.logger.warning("No metrics to save")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Metrics", "",
            "PNG Files (*.png);;PDF Files (*.pdf);;All Files (*.*)"
        )

        if not file_path:
            return

        try:
            # Create a figure with the bar chart and table
            fig = Figure(figsize=(10, 6), dpi=150)

            # Get current metric values
            psnr_n = self._psnr_noisy[idx] if idx < len(self._psnr_noisy) else None
            psnr_r = self._psnr_recon[idx] if idx < len(self._psnr_recon) else None
            ssim_n = self._ssim_noisy[idx] if idx < len(self._ssim_noisy) else None
            ssim_r = self._ssim_recon[idx] if idx < len(self._ssim_recon) else None
            lpips_n = self._lpips_noisy[idx] if idx < len(self._lpips_noisy) else None
            lpips_r = self._lpips_recon[idx] if idx < len(self._lpips_recon) else None

            # Left: Bar chart (recreate the bar chart)
            ax1 = fig.add_subplot(1, 2, 1)

            show_psnr = self.psnr_checkbox.isChecked()
            show_ssim = self.ssim_checkbox.isChecked()
            show_lpips = self.lpips_checkbox.isChecked()

            metrics = []
            if show_psnr and psnr_n is not None:
                psnr_n_norm = min(max(psnr_n / 50.0, 0), 1)
                psnr_r_norm = min(max(psnr_r / 50.0, 0), 1) if psnr_r else 0
                metrics.append(('PSNR \u2191', psnr_n, psnr_r, psnr_n_norm, psnr_r_norm, '#1f77b4', '{:.1f}'))
            if show_ssim and ssim_n is not None:
                metrics.append(('SSIM \u2191', ssim_n, ssim_r, ssim_n, ssim_r if ssim_r else 0, '#2ca02c', '{:.3f}'))
            if show_lpips and lpips_n is not None:
                lpips_n_norm = 1.0 - min(max(lpips_n, 0), 1)
                lpips_r_norm = 1.0 - min(max(lpips_r, 0), 1) if lpips_r else 0
                metrics.append(('LPIPS \u2193', lpips_n, lpips_r, lpips_n_norm, lpips_r_norm, '#d62728', '{:.3f}'))

            if metrics:
                n_metrics = len(metrics)
                x = np.arange(2)
                width = 0.8 / n_metrics

                for i, (name, noisy_val, recon_val, noisy_norm, recon_norm, color, fmt) in enumerate(metrics):
                    offset = (i - (n_metrics - 1) / 2) * width
                    bars = ax1.bar(x + offset, [noisy_norm, recon_norm], width * 0.9,
                                  label=name, color=color, alpha=0.8)
                    ax1.text(bars[0].get_x() + bars[0].get_width()/2, bars[0].get_height() + 0.02,
                            fmt.format(noisy_val), ha='center', va='bottom', fontsize=8)
                    if recon_val:
                        ax1.text(bars[1].get_x() + bars[1].get_width()/2, bars[1].get_height() + 0.02,
                                fmt.format(recon_val), ha='center', va='bottom', fontsize=8)

                ax1.set_xticks(x)
                ax1.set_xticklabels(['Noisy', 'Denoised'])
                ax1.set_ylabel('Quality Score (normalized)')
                ax1.set_ylim(0, 1.15)
                ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=n_metrics, fontsize=9)
                ax1.set_title('Quality Metrics Comparison')
                ax1.grid(True, alpha=0.3, axis='y')

            # Right: Table with values
            ax2 = fig.add_subplot(1, 2, 2)
            ax2.axis('off')

            # Build table data
            table_data = []
            row_labels = []

            if show_psnr:
                psnr_change = ((psnr_r - psnr_n) / abs(psnr_n) * 100) if psnr_n and psnr_r and psnr_n != 0 else 0
                change_str = f"+{psnr_change:.1f}%" if psnr_change >= 0 else f"{psnr_change:.1f}%"
                table_data.append([f"{psnr_n:.2f}" if psnr_n else "-",
                                   f"{psnr_r:.2f}" if psnr_r else "-",
                                   change_str])
                row_labels.append("PSNR (dB) \u2191")

            if show_ssim:
                ssim_change = ((ssim_r - ssim_n) / abs(ssim_n) * 100) if ssim_n and ssim_r and ssim_n != 0 else 0
                change_str = f"+{ssim_change:.1f}%" if ssim_change >= 0 else f"{ssim_change:.1f}%"
                table_data.append([f"{ssim_n:.4f}" if ssim_n else "-",
                                   f"{ssim_r:.4f}" if ssim_r else "-",
                                   change_str])
                row_labels.append("SSIM \u2191")

            if show_lpips:
                lpips_change = ((lpips_r - lpips_n) / abs(lpips_n) * 100) if lpips_n and lpips_r and lpips_n != 0 else 0
                change_str = f"+{lpips_change:.1f}%" if lpips_change >= 0 else f"{lpips_change:.1f}%"
                table_data.append([f"{lpips_n:.4f}" if lpips_n else "-",
                                   f"{lpips_r:.4f}" if lpips_r else "-",
                                   change_str])
                row_labels.append("LPIPS \u2193")

            if table_data:
                table = ax2.table(cellText=table_data,
                                  rowLabels=row_labels,
                                  colLabels=['Noisy', 'Denoised', 'Change'],
                                  loc='center',
                                  cellLoc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1.2, 1.5)

            ax2.set_title('Metric Values', fontsize=12, pad=20)

            fig.suptitle(f"Quality Metrics for Image {idx}", fontsize=14)
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            fig.savefig(file_path, dpi=150, bbox_inches='tight')

            self.logger.info(f"Metrics saved to {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to save metrics: {e}")
