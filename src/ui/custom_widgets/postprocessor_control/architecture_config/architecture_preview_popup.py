"""
Popup dialog for previewing neural network architecture visualization.

Uses PlotNeuralNet (TikZ/LaTeX) for publication-quality output.
"""
import logging
import os
import shutil
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFileDialog, QScrollArea, QWidget,
    QSizePolicy, QGroupBox, QGridLayout, QFrame, QMessageBox,
    QProgressBar, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QCheckBox, QSpinBox, QComboBox, QApplication
)
from pathlib import Path
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap, QImage, QWheelEvent, QPainter

import torch.nn as nn

from .network_visualizer import NetworkVisualizer, SemanticBlock
from .plotneuralnet_generator import PlotNeuralNetGenerator, PDFLATEX_AVAILABLE
from .latex_worker import LaTeXCompilationWorker

# Import MODEL_REGISTRY for batch test mode
try:
    from simulation_engine._4_postprocessor.postprocessor_nn import MODEL_REGISTRY
    MODEL_REGISTRY_AVAILABLE = True
except ImportError:
    MODEL_REGISTRY = {}
    MODEL_REGISTRY_AVAILABLE = False

# Check if pdf2image is available
try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError as e:
    PDF2IMAGE_AVAILABLE = False
    logging.getLogger(__name__).warning(f"pdf2image not available: {e}")

# Log availability at module load time
_logger = logging.getLogger(__name__)
_logger.debug(f"PlotNeuralNet dependencies: PDFLATEX={PDFLATEX_AVAILABLE}, PDF2IMAGE={PDF2IMAGE_AVAILABLE}")


class ZoomableImageView(QGraphicsView):
    """
    A QGraphicsView that supports mouse wheel zoom and pan.

    Features:
    - Initial fit to window
    - Zoom in/out with mouse wheel
    - Pan by dragging with mouse
    - Emits zoom level changes
    """

    zoom_changed = pyqtSignal(float)  # Emits zoom percentage (100 = 100%)

    MIN_ZOOM = 0.1   # 10%
    MAX_ZOOM = 10.0  # 1000%
    ZOOM_STEP = 1.15  # 15% per wheel step

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        # Setup scene
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        # Setup view properties
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setBackgroundBrush(Qt.white)
        self.setFrameShape(QFrame.NoFrame)

        # State
        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._zoom_level = 1.0
        self._fit_zoom = 1.0  # Zoom level that fits the image to view

    def set_pixmap(self, pixmap: QPixmap):
        """Set the image to display and fit it to the view."""
        # Clear previous content
        self._scene.clear()

        # Add new pixmap
        self._pixmap_item = QGraphicsPixmapItem(pixmap)
        self._pixmap_item.setTransformationMode(Qt.SmoothTransformation)
        self._scene.addItem(self._pixmap_item)

        # Set scene rect to image size
        self._scene.setSceneRect(self._pixmap_item.boundingRect())

        # Fit to view
        self.fit_to_view()

    def fit_to_view(self):
        """Fit the image to the current view size."""
        if self._pixmap_item is None:
            return

        # Reset any existing transform
        self.resetTransform()

        # Calculate scale to fit
        view_rect = self.viewport().rect()
        scene_rect = self._scene.sceneRect()

        if scene_rect.width() <= 0 or scene_rect.height() <= 0:
            return

        x_ratio = view_rect.width() / scene_rect.width()
        y_ratio = view_rect.height() / scene_rect.height()

        # Use the smaller ratio to ensure entire image fits
        self._fit_zoom = min(x_ratio, y_ratio) * 0.95  # 95% to leave small margin
        self._zoom_level = self._fit_zoom

        self.scale(self._fit_zoom, self._fit_zoom)
        self.centerOn(self._pixmap_item)

        self.zoom_changed.emit(self._zoom_level * 100)

    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel for zooming."""
        if self._pixmap_item is None:
            return

        # Get zoom direction
        if event.angleDelta().y() > 0:
            # Zoom in
            factor = self.ZOOM_STEP
        else:
            # Zoom out
            factor = 1.0 / self.ZOOM_STEP

        # Calculate new zoom level
        new_zoom = self._zoom_level * factor

        # Clamp to min/max
        if new_zoom < self.MIN_ZOOM:
            factor = self.MIN_ZOOM / self._zoom_level
            new_zoom = self.MIN_ZOOM
        elif new_zoom > self.MAX_ZOOM:
            factor = self.MAX_ZOOM / self._zoom_level
            new_zoom = self.MAX_ZOOM

        # Apply zoom
        self._zoom_level = new_zoom
        self.scale(factor, factor)

        self.zoom_changed.emit(self._zoom_level * 100)

    def reset_zoom(self):
        """Reset zoom to fit the image in view."""
        self.fit_to_view()

    def zoom_in(self):
        """Zoom in by one step."""
        if self._pixmap_item is None:
            return

        factor = self.ZOOM_STEP
        new_zoom = self._zoom_level * factor

        if new_zoom <= self.MAX_ZOOM:
            self._zoom_level = new_zoom
            self.scale(factor, factor)
            self.zoom_changed.emit(self._zoom_level * 100)

    def zoom_out(self):
        """Zoom out by one step."""
        if self._pixmap_item is None:
            return

        factor = 1.0 / self.ZOOM_STEP
        new_zoom = self._zoom_level * factor

        if new_zoom >= self.MIN_ZOOM:
            self._zoom_level = new_zoom
            self.scale(factor, factor)
            self.zoom_changed.emit(self._zoom_level * 100)

    def resizeEvent(self, event):
        """Handle resize to maintain fit if at fit zoom level."""
        super().resizeEvent(event)

        # If we're close to the fit zoom, re-fit on resize
        if self._pixmap_item is not None and abs(self._zoom_level - self._fit_zoom) < 0.01:
            self.fit_to_view()


class ArchitecturePreviewPopup(QDialog):
    """
    Modal dialog displaying neural network architecture visualization.

    Features:
    - PlotNeuralNet (TikZ/LaTeX) publication-quality rendering
    - Model information panel (parameters, input size, type)
    - Zoomable visualization area with mouse wheel zoom
    - Save to PNG/PDF/TEX functionality
    - Optional input/output image embedding
    """

    # Image type options for combo boxes
    IMAGE_TYPE_GROUND_TRUTH = "Ground Truth"
    IMAGE_TYPE_NOISY = "Noisy"
    IMAGE_TYPE_DENOISED = "Denoised"

    # Colormap options for grayscale images
    COLORMAP_GRAY = "Gray"
    COLORMAP_VIRIDIS = "Viridis"
    COLORMAP_JET = "Jet"
    COLORMAP_HOT = "Hot"
    COLORMAP_INFERNO = "Inferno"
    COLORMAP_PLASMA = "Plasma"

    def __init__(self, model: nn.Module, model_name: str,
                 input_size: int = 32, config: Optional[Dict[str, Any]] = None,
                 ground_truth_images: Optional[np.ndarray] = None,
                 noisy_images: Optional[np.ndarray] = None,
                 denoised_images: Optional[np.ndarray] = None,
                 parent: Optional[QWidget] = None,
                 logger: Optional[logging.Logger] = None,
                 # Batch test mode parameters
                 tests: Optional[List[Dict[str, Any]]] = None,
                 test_name: Optional[str] = None):
        """
        Initialize the architecture preview popup.

        Args:
            model: PyTorch model to visualize
            model_name: Display name for the model
            input_size: Input spatial size (assumes square input)
            config: Architecture configuration dictionary
            ground_truth_images: Original clean images (N, H, W)
            noisy_images: Noisy/reconstructed images (N, H, W)
            denoised_images: Post-processed images (N, H, W) - only after training
            parent: Parent widget
            logger: Logger instance
            tests: Optional list of batch tests for test selector (Batch Reports mode)
            test_name: Optional test name to display in title (Batch Reports mode)
        """
        super().__init__(parent)

        # Batch test mode
        self._tests = tests or []
        self._current_test_idx = 0
        self._test_name = test_name

        # Build window title
        if test_name:
            self.setWindowTitle(f"Architecture Preview: {test_name} - {model_name}")
        else:
            self.setWindowTitle(f"Architecture Preview: {model_name}")

        self.setMinimumSize(1100, 700)
        self.resize(1400, 850)
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint)

        self.model = model
        self.model_name = model_name
        self.input_size = input_size
        self.config = config or {}

        # Image datasets for visualization
        self.ground_truth_images = ground_truth_images
        self.noisy_images = noisy_images
        self.denoised_images = denoised_images

        # State for image selection
        self._show_images = False
        self._selected_image_idx = 0
        self._input_image_type = self.IMAGE_TYPE_NOISY  # Default matches combo box
        self._output_image_type = self.IMAGE_TYPE_DENOISED
        self._colormap = self.COLORMAP_HOT  # Default colormap

        # Check if post-processing results are available
        self._has_postprocessing = denoised_images is not None and len(denoised_images) > 0

        if logger:
            self.logger = logger.getChild("ArchitecturePreviewPopup")
        else:
            self.logger = logging.getLogger("SPIm.ArchitecturePreviewPopup")

        # State
        self._pdf_path: Optional[str] = None
        self._tex_source: Optional[str] = None
        self._latex_worker: Optional[LaTeXCompilationWorker] = None
        self._original_pixmap: Optional[QPixmap] = None

        # Extract blocks for visualization
        self._visualizer = NetworkVisualizer(figsize=(18, 9), dpi=100, logger=self.logger)
        self._blocks: List[SemanticBlock] = []
        self._skip_connections: List[Tuple[int, int]] = []
        self._extract_blocks()

        self._setup_ui()

        # Start PlotNeuralNet rendering
        self._start_tikz_compilation()

    def _extract_blocks(self):
        """Extract semantic blocks from model."""
        try:
            arch_type = self._visualizer.analyze_architecture(self.model)
            if arch_type == 'unet':
                self._blocks, self._skip_connections = self._visualizer.extract_unet_blocks(
                    self.model, self.input_size
                )
            else:
                self._blocks, self._skip_connections = self._visualizer.extract_unet_blocks(
                    self.model, self.input_size
                )
            self.logger.debug(f"Extracted {len(self._blocks)} blocks, "
                            f"{len(self._skip_connections)} skip connections")
        except Exception as e:
            self.logger.error(f"Failed to extract blocks: {e}", exc_info=True)
            self._blocks = []
            self._skip_connections = []

    def _setup_model_info_panel(self):
        """Setup or update the model information panel."""
        # Clear existing labels
        for label in self._info_value_labels.values():
            label.deleteLater()
        self._info_value_labels.clear()

        # Remove all items from layout
        while self._info_layout.count():
            item = self._info_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Calculate parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters()
                              if p.requires_grad)
        non_trainable = total_params - trainable_params

        # Format parameter counts
        def format_params(n: int) -> str:
            if n >= 1_000_000:
                return f"{n:,} ({n / 1_000_000:.2f}M)"
            elif n >= 1_000:
                return f"{n:,} ({n / 1_000:.1f}K)"
            return f"{n:,}"

        # Create info labels
        font_bold = QFont()
        font_bold.setBold(True)

        info_items = [
            ("Total Parameters:", format_params(total_params)),
            ("Trainable:", format_params(trainable_params)),
            ("Non-trainable:", format_params(non_trainable)),
            ("Input Size:", f"{self.input_size} x {self.input_size} x 1"),
            ("Model Class:", type(self.model).__name__),
        ]

        for i, (label_text, value_text) in enumerate(info_items):
            row = i // 3
            col = (i % 3) * 2

            label = QLabel(label_text)
            label.setFont(font_bold)
            value = QLabel(value_text)
            self._info_value_labels[label_text] = value

            self._info_layout.addWidget(label, row, col)
            self._info_layout.addWidget(value, row, col + 1)

        # Add configuration info if available
        if self.config:
            config_str = ", ".join(f"{k}={v}" for k, v in self.config.items())
            config_label = QLabel("Configuration:")
            config_label.setFont(font_bold)
            config_value = QLabel(config_str)
            config_value.setWordWrap(True)
            self._info_value_labels["Configuration:"] = config_value

            row = len(info_items) // 3 + 1
            self._info_layout.addWidget(config_label, row, 0)
            self._info_layout.addWidget(config_value, row, 1, 1, 5)

    def _on_test_changed(self, index: int):
        """Handle test selection change in Batch Reports mode."""
        if index < 0 or index >= len(self._tests):
            return

        self._current_test_idx = index
        test = self._tests[index]

        # Get model name and test name
        model_name = test.get("model_name")
        if not model_name:
            config = test.get("config", {})
            model_name = config.get("model_name", "u-net")

        test_name = test.get("name", "Unknown")
        self._test_name = test_name

        # Get model entry from registry
        model_entry = MODEL_REGISTRY.get(model_name.lower())
        if not model_entry:
            model_entry = MODEL_REGISTRY.get(model_name)
        if not model_entry:
            QMessageBox.warning(
                self,
                "Model Not Found",
                f"Model '{model_name}' not found in MODEL_REGISTRY."
            )
            return

        # Get input size
        input_size = test.get("img_size", 64)
        if not input_size:
            config = test.get("config", {})
            input_size = config.get("img_size", 64)

        # Create model instance
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)

            model_cls = model_entry["cls"]
            defaults = model_entry.get("defaults", {}).copy()
            if "img_size" in defaults:
                defaults["img_size"] = input_size

            self.model = model_cls(**defaults)
            self.model_name = model_name
            self.input_size = input_size
            self.config = model_entry.get("defaults", {})

            self.logger.info("Switched to model: %s for test: %s", model_name, test_name)
        except Exception as e:
            QApplication.restoreOverrideCursor()
            self.logger.error("Failed to create model: %s", e, exc_info=True)
            QMessageBox.warning(
                self,
                "Model Creation Failed",
                f"Failed to create model '{model_name}':\n{e}"
            )
            return

        # Load images from exported dataset
        self.ground_truth_images = None
        self.noisy_images = None
        self.denoised_images = None

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
                        self.ground_truth_images = data["originals"]
                    if "reconstructions" in data:
                        self.noisy_images = data["reconstructions"]
                    if "denoised" in data:
                        self.denoised_images = data["denoised"]
                    self.logger.info("Loaded images from %s", test_images_path)
                except Exception as e:
                    self.logger.warning("Failed to load images: %s", e)

        # Update UI
        self._has_postprocessing = self.denoised_images is not None and len(self.denoised_images) > 0

        # Update window title
        self.setWindowTitle(f"Architecture Preview: {test_name} - {model_name}")

        # Update title label
        self.title_label.setText(f"<h2>{test_name} - {model_name} Architecture</h2>")

        # Update model info panel
        self._setup_model_info_panel()

        # Re-extract blocks
        self._extract_blocks()

        # Reset image state
        self._show_images = False
        self._selected_image_idx = 0

        # Update image controls visibility/state
        self._update_image_controls_state()

        # Restart visualization
        self._regenerate_with_images()

        QApplication.restoreOverrideCursor()

    def _update_image_controls_state(self):
        """Update image controls based on current postprocessing state."""
        if hasattr(self, 'show_images_checkbox') and self.show_images_checkbox:
            self.show_images_checkbox.setChecked(False)
            # Update visibility based on whether we have images
            has_images = self._has_postprocessing
            self.show_images_checkbox.setEnabled(has_images)

            if hasattr(self, 'image_idx_spinbox') and self.image_idx_spinbox:
                if has_images:
                    self._num_images = min(
                        len(self.ground_truth_images) if self.ground_truth_images is not None else 0,
                        len(self.noisy_images) if self.noisy_images is not None else 0,
                        len(self.denoised_images) if self.denoised_images is not None else 0
                    )
                    self.image_idx_spinbox.setRange(0, max(0, self._num_images - 1))
                    self.image_idx_spinbox.setValue(0)
                    self._max_image_idx = max(0, self._num_images - 1)
                    if hasattr(self, 'image_total_label') and self.image_total_label:
                        self.image_total_label.setText(f"/ {self._max_image_idx}")

    def _setup_ui(self):
        """Setup the dialog UI layout."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(12)

        # Test selector (only in Batch Reports mode)
        if self._tests and len(self._tests) > 1:
            test_selector_layout = QHBoxLayout()
            test_selector_layout.setSpacing(10)

            test_label = QLabel("Test:")
            test_label.setStyleSheet("font-weight: bold; font-size: 12px;")
            test_selector_layout.addWidget(test_label)

            self.test_combo = QComboBox()
            self.test_combo.setMinimumWidth(300)
            self.test_combo.setStyleSheet("""
                QComboBox {
                    padding: 5px 10px;
                    font-size: 12px;
                    border: 1px solid #ccc;
                    border-radius: 4px;
                }
            """)
            for test in self._tests:
                t_name = test.get("name", "Unknown")
                m_name = test.get("model_name", test.get("config", {}).get("model_name", "Unknown"))
                self.test_combo.addItem(f"{t_name} ({m_name})")
            self.test_combo.currentIndexChanged.connect(self._on_test_changed)
            test_selector_layout.addWidget(self.test_combo)

            test_selector_layout.addStretch()
            main_layout.addLayout(test_selector_layout)
        else:
            self.test_combo = None

        # Title
        if self._test_name:
            self.title_label = QLabel(f"<h2>{self._test_name} - {self.model_name} Architecture</h2>")
        else:
            self.title_label = QLabel(f"<h2>{self.model_name} Architecture</h2>")
        self.title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.title_label)

        # Model information panel
        info_group = QGroupBox("Model Information")
        self._info_layout = QGridLayout(info_group)
        self._info_layout.setSpacing(15)

        # Store references to value labels for dynamic updates
        self._info_value_labels = {}
        self._setup_model_info_panel()

        main_layout.addWidget(info_group)

        # Status bar with progress and zoom info
        status_layout = QHBoxLayout()

        self.progress_label = QLabel("Generating visualization...")
        self.progress_label.setStyleSheet("color: #666;")
        status_layout.addWidget(self.progress_label)

        status_layout.addStretch()

        # Image visualization controls (only if post-processing results available)
        if self._has_postprocessing:
            # Checkbox to enable/disable image display
            self.show_images_checkbox = QCheckBox("Show I/O images")
            self.show_images_checkbox.setToolTip("Display input and output images in the visualization")
            self.show_images_checkbox.stateChanged.connect(self._on_show_images_changed)
            status_layout.addWidget(self.show_images_checkbox)

            # Calculate number of images (use minimum across all available sets)
            self._num_images = min(
                len(self.ground_truth_images) if self.ground_truth_images is not None else 0,
                len(self.noisy_images) if self.noisy_images is not None else 0,
                len(self.denoised_images) if self.denoised_images is not None else 0
            )

            # Image index selector with total count
            self.image_idx_label = QLabel("Image:")
            self.image_idx_label.setStyleSheet("margin-left: 8px;")
            self.image_idx_label.setEnabled(False)
            status_layout.addWidget(self.image_idx_label)

            self.image_idx_spinbox = QSpinBox()
            self.image_idx_spinbox.setRange(0, max(0, self._num_images - 1))
            self.image_idx_spinbox.setValue(0)
            self.image_idx_spinbox.setMaximumWidth(70)
            self.image_idx_spinbox.setEnabled(False)
            self.image_idx_spinbox.valueChanged.connect(self._on_image_idx_changed)
            status_layout.addWidget(self.image_idx_spinbox)

            # Max index label (show maximum selectable index, matching spinbox range)
            self._max_image_idx = max(0, self._num_images - 1)
            self.image_total_label = QLabel(f"/ {self._max_image_idx}")
            self.image_total_label.setStyleSheet("color: #666;")
            self.image_total_label.setEnabled(False)
            status_layout.addWidget(self.image_total_label)

            # Colormap selector
            self.colormap_label = QLabel("Colormap:")
            self.colormap_label.setStyleSheet("margin-left: 8px;")
            self.colormap_label.setEnabled(False)
            status_layout.addWidget(self.colormap_label)

            self.colormap_combo = QComboBox()
            self.colormap_combo.addItems([
                self.COLORMAP_GRAY,
                self.COLORMAP_VIRIDIS,
                self.COLORMAP_JET,
                self.COLORMAP_HOT,
                self.COLORMAP_INFERNO,
                self.COLORMAP_PLASMA
            ])
            self.colormap_combo.setCurrentText(self.COLORMAP_HOT)
            self.colormap_combo.setMaximumWidth(85)
            self.colormap_combo.setEnabled(False)
            self.colormap_combo.currentTextChanged.connect(self._on_colormap_changed)
            status_layout.addWidget(self.colormap_combo)

            # Input image type selector
            self.input_type_label = QLabel("Input:")
            self.input_type_label.setStyleSheet("margin-left: 8px;")
            self.input_type_label.setEnabled(False)
            status_layout.addWidget(self.input_type_label)

            self.input_type_combo = QComboBox()
            self.input_type_combo.addItems([
                self.IMAGE_TYPE_GROUND_TRUTH,
                self.IMAGE_TYPE_NOISY,
                self.IMAGE_TYPE_DENOISED
            ])
            self.input_type_combo.setCurrentText(self.IMAGE_TYPE_NOISY)
            self.input_type_combo.setMaximumWidth(100)
            self.input_type_combo.setEnabled(False)
            self.input_type_combo.currentTextChanged.connect(self._on_input_type_changed)
            status_layout.addWidget(self.input_type_combo)

            # Output image type selector
            self.output_type_label = QLabel("Output:")
            self.output_type_label.setStyleSheet("margin-left: 8px;")
            self.output_type_label.setEnabled(False)
            status_layout.addWidget(self.output_type_label)

            self.output_type_combo = QComboBox()
            self.output_type_combo.addItems([
                self.IMAGE_TYPE_GROUND_TRUTH,
                self.IMAGE_TYPE_NOISY,
                self.IMAGE_TYPE_DENOISED
            ])
            self.output_type_combo.setCurrentText(self.IMAGE_TYPE_DENOISED)
            self.output_type_combo.setMaximumWidth(100)
            self.output_type_combo.setEnabled(False)
            self.output_type_combo.currentTextChanged.connect(self._on_output_type_changed)
            status_layout.addWidget(self.output_type_combo)

            status_layout.addSpacing(10)
        else:
            # No post-processing results - show info note
            self.show_images_checkbox = None
            self.image_idx_spinbox = None
            self.image_idx_label = None
            self.image_total_label = None
            self.colormap_combo = None
            self.colormap_label = None
            self.input_type_combo = None
            self.input_type_label = None
            self.output_type_combo = None
            self.output_type_label = None
            self._num_images = 0

            # Info label
            info_label = QLabel("ℹ️ Train model and apply post-processing to enable I/O image preview")
            info_label.setStyleSheet("color: #888; font-style: italic; margin-left: 10px;")
            info_label.setToolTip(
                "To visualize input/output images:\n"
                "1. Load a dataset\n"
                "2. Generate masks and apply reconstruction\n"
                "3. Train the neural network\n"
                "4. Apply post-processing to generate denoised images"
            )
            status_layout.addWidget(info_label)

        status_layout.addStretch()

        # Zoom controls
        self.zoom_label = QLabel("Zoom: --")
        self.zoom_label.setStyleSheet("color: #666; margin-right: 10px;")
        status_layout.addWidget(self.zoom_label)

        self.fit_btn = QPushButton("Fit")
        self.fit_btn.setMaximumWidth(50)
        self.fit_btn.setToolTip("Fit image to window (also double-click)")
        self.fit_btn.clicked.connect(self._on_fit_clicked)
        self.fit_btn.setEnabled(False)
        status_layout.addWidget(self.fit_btn)

        self.zoom_in_btn = QPushButton("+")
        self.zoom_in_btn.setMaximumWidth(30)
        self.zoom_in_btn.setToolTip("Zoom in")
        self.zoom_in_btn.clicked.connect(self._on_zoom_in)
        self.zoom_in_btn.setEnabled(False)
        status_layout.addWidget(self.zoom_in_btn)

        self.zoom_out_btn = QPushButton("-")
        self.zoom_out_btn.setMaximumWidth(30)
        self.zoom_out_btn.setToolTip("Zoom out")
        self.zoom_out_btn.clicked.connect(self._on_zoom_out)
        self.zoom_out_btn.setEnabled(False)
        status_layout.addWidget(self.zoom_out_btn)

        status_layout.addSpacing(20)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        status_layout.addWidget(self.progress_bar)

        main_layout.addLayout(status_layout)

        # Visualization area with zoomable view
        viz_group = QGroupBox("Architecture Diagram (PlotNeuralNet) - Scroll to zoom, drag to pan")
        viz_layout = QVBoxLayout(viz_group)

        # Zoomable image view
        self.image_view = ZoomableImageView()
        self.image_view.zoom_changed.connect(self._on_zoom_changed)
        self.image_view.setMinimumHeight(400)

        # Loading widget (shown initially)
        self.loading_widget = QWidget()
        loading_layout = QVBoxLayout(self.loading_widget)
        loading_label = QLabel("Compiling LaTeX... This may take a few seconds.")
        loading_label.setAlignment(Qt.AlignCenter)
        loading_label.setStyleSheet("color: #666; font-size: 14px;")
        loading_layout.addWidget(loading_label)

        # Stack loading widget on top initially
        viz_layout.addWidget(self.loading_widget)
        viz_layout.addWidget(self.image_view)
        self.image_view.hide()

        main_layout.addWidget(viz_group, 1)  # Stretch factor 1

        # Buttons
        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch()

        # Save buttons
        self.save_png_btn = QPushButton("Save as PNG")
        self.save_png_btn.clicked.connect(lambda: self._on_save("png"))
        self.save_png_btn.setEnabled(False)
        buttons_layout.addWidget(self.save_png_btn)

        self.save_pdf_btn = QPushButton("Save as PDF")
        self.save_pdf_btn.clicked.connect(lambda: self._on_save("pdf"))
        self.save_pdf_btn.setEnabled(False)
        buttons_layout.addWidget(self.save_pdf_btn)

        self.save_tex_btn = QPushButton("Save as TEX")
        self.save_tex_btn.clicked.connect(lambda: self._on_save("tex"))
        self.save_tex_btn.setToolTip("Save LaTeX/TikZ source code for manual editing")
        self.save_tex_btn.setEnabled(False)
        buttons_layout.addWidget(self.save_tex_btn)

        buttons_layout.addSpacing(20)

        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close)
        buttons_layout.addWidget(self.close_button)

        main_layout.addLayout(buttons_layout)

    def _on_zoom_changed(self, zoom_percent: float):
        """Update zoom label when zoom changes."""
        self.zoom_label.setText(f"Zoom: {zoom_percent:.0f}%")

    def _on_fit_clicked(self):
        """Handle fit button click."""
        self.image_view.fit_to_view()

    def _on_zoom_in(self):
        """Handle zoom in button click."""
        self.image_view.zoom_in()

    def _on_zoom_out(self):
        """Handle zoom out button click."""
        self.image_view.zoom_out()

    def _on_show_images_changed(self, state: int):
        """Handle show images checkbox change."""
        self._show_images = (state == Qt.Checked)

        # Enable/disable all image controls
        enabled = self._show_images
        if self.image_idx_spinbox:
            self.image_idx_spinbox.setEnabled(enabled)
        if self.image_idx_label:
            self.image_idx_label.setEnabled(enabled)
        if self.image_total_label:
            self.image_total_label.setEnabled(enabled)
        if self.colormap_combo:
            self.colormap_combo.setEnabled(enabled)
        if self.colormap_label:
            self.colormap_label.setEnabled(enabled)
        if self.input_type_combo:
            self.input_type_combo.setEnabled(enabled)
        if self.input_type_label:
            self.input_type_label.setEnabled(enabled)
        if self.output_type_combo:
            self.output_type_combo.setEnabled(enabled)
        if self.output_type_label:
            self.output_type_label.setEnabled(enabled)

        # Regenerate visualization
        self._regenerate_with_images()

    def _on_image_idx_changed(self, value: int):
        """Handle image index spinbox change."""
        self._selected_image_idx = value

        # Only regenerate if showing images
        if self._show_images:
            self._regenerate_with_images()

    def _on_input_type_changed(self, text: str):
        """Handle input image type combo change."""
        self._input_image_type = text

        # Only regenerate if showing images
        if self._show_images:
            self._regenerate_with_images()

    def _on_output_type_changed(self, text: str):
        """Handle output image type combo change."""
        self._output_image_type = text

        # Only regenerate if showing images
        if self._show_images:
            self._regenerate_with_images()

    def _on_colormap_changed(self, text: str):
        """Handle colormap combo change."""
        self._colormap = text

        # Only regenerate if showing images
        if self._show_images:
            self._regenerate_with_images()

    def _get_image_by_type(self, image_type: str) -> Optional[np.ndarray]:
        """Get image array by type name."""
        if image_type == self.IMAGE_TYPE_GROUND_TRUTH:
            images = self.ground_truth_images
        elif image_type == self.IMAGE_TYPE_NOISY:
            images = self.noisy_images
        elif image_type == self.IMAGE_TYPE_DENOISED:
            images = self.denoised_images
        else:
            return None

        if images is None or self._selected_image_idx >= len(images):
            return None

        img = images[self._selected_image_idx]

        # Handle different array shapes - convert to 2D
        if img.ndim == 3:
            # (C, H, W) -> (H, W) for single channel
            if img.shape[0] == 1:
                img = img[0]
            # (H, W, C) with single channel
            elif img.shape[-1] == 1:
                img = img[:, :, 0]
            elif img.shape[0] in [1, 3]:
                img = img[0] if img.shape[0] == 1 else np.transpose(img, (1, 2, 0))

        return img

    def _get_selected_input_image(self) -> Optional[np.ndarray]:
        """Get the currently selected input image."""
        if not self._show_images:
            return None
        return self._get_image_by_type(self._input_image_type)

    def _get_selected_output_image(self) -> Optional[np.ndarray]:
        """Get the currently selected output image."""
        if not self._show_images:
            return None
        return self._get_image_by_type(self._output_image_type)

    def _regenerate_with_images(self):
        """Regenerate visualization with/without input/output images."""
        # Stop any running compilation
        if self._latex_worker is not None and self._latex_worker.isRunning():
            self._latex_worker.terminate()
            self._latex_worker.wait(1000)
            self._latex_worker.cleanup()

        # Reset UI state
        self.progress_bar.show()
        self.progress_bar.setRange(0, 0)
        self.progress_label.setText("Regenerating visualization...")

        self.loading_widget.show()
        self.image_view.hide()

        # Disable buttons during regeneration
        self.save_png_btn.setEnabled(False)
        self.save_pdf_btn.setEnabled(False)
        self.fit_btn.setEnabled(False)
        self.zoom_in_btn.setEnabled(False)
        self.zoom_out_btn.setEnabled(False)

        # Get input and output images if enabled
        input_img = self._get_selected_input_image()
        output_img = self._get_selected_output_image()

        # Start new compilation with colormap
        self._start_tikz_compilation(
            input_image_override=input_img,
            output_image_override=output_img,
            colormap=self._colormap if self._show_images else None
        )

    def _start_tikz_compilation(self,
                                input_image_override: Optional[np.ndarray] = None,
                                output_image_override: Optional[np.ndarray] = None,
                                colormap: Optional[str] = None):
        """Start background TikZ/LaTeX compilation.

        Args:
            input_image_override: If provided, use this as input image
            output_image_override: If provided, use this as output image
            colormap: Colormap name for grayscale images (e.g., 'Gray', 'Viridis')
        """
        if not PDFLATEX_AVAILABLE:
            self._show_error_widget(
                "pdflatex not found.\n\n"
                "Install with:\n"
                "  apt install texlive-latex-base texlive-latex-extra"
            )
            self.progress_bar.hide()
            self.progress_label.setText("Error: pdflatex not available")
            return

        if not PDF2IMAGE_AVAILABLE:
            self._show_error_widget(
                "pdf2image not found.\n\n"
                "Install with:\n"
                "  pip install pdf2image\n"
                "  apt install poppler-utils"
            )
            self.progress_bar.hide()
            self.progress_label.setText("Error: pdf2image not available")
            return

        if self._latex_worker is not None and self._latex_worker.isRunning():
            self.logger.warning("TikZ compilation already in progress")
            return

        # Create and start worker
        self._latex_worker = LaTeXCompilationWorker(
            blocks=self._blocks,
            skip_connections=self._skip_connections,
            model_name=self.model_name,
            input_image=input_image_override,
            output_image=output_image_override,
            colormap=colormap,
            logger=self.logger,
            parent=self
        )

        self._latex_worker.progress.connect(self._on_tikz_progress)
        self._latex_worker.tex_ready.connect(self._on_tex_ready)
        self._latex_worker.finished.connect(self._on_tikz_finished)
        self._latex_worker.error.connect(self._on_tikz_error)

        self._latex_worker.start()

    def _on_tikz_progress(self, message: str):
        """Handle TikZ compilation progress update."""
        self.progress_label.setText(message)

    def _on_tex_ready(self, tex_source: str):
        """Handle .tex source code ready."""
        self._tex_source = tex_source
        self.save_tex_btn.setEnabled(True)

    def _on_tikz_finished(self, pdf_path: str):
        """Handle TikZ compilation success."""
        self._pdf_path = pdf_path
        self.progress_label.setText("Done!")
        self.progress_bar.hide()

        # Enable save buttons
        self.save_png_btn.setEnabled(True)
        self.save_pdf_btn.setEnabled(True)

        # Enable zoom controls
        self.fit_btn.setEnabled(True)
        self.zoom_in_btn.setEnabled(True)
        self.zoom_out_btn.setEnabled(True)

        # Convert PDF to image and display
        try:
            # Use higher DPI for better quality when zooming
            images = convert_from_path(pdf_path, dpi=200)
            if images:
                # Convert PIL Image to QPixmap
                pil_img = images[0]
                # Convert to RGB if necessary
                if pil_img.mode != 'RGB':
                    pil_img = pil_img.convert('RGB')

                # Convert to QImage
                data = pil_img.tobytes('raw', 'RGB')
                qimg = QImage(data, pil_img.width, pil_img.height,
                            pil_img.width * 3, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimg)

                # Store original pixmap
                self._original_pixmap = pixmap

                # Hide loading, show image view
                self.loading_widget.hide()
                self.image_view.show()

                # Set pixmap (will auto-fit to view)
                self.image_view.set_pixmap(pixmap)

                self.logger.info(f"TikZ visualization displayed from {pdf_path}")
            else:
                self._show_error_widget("No pages in generated PDF")

        except Exception as e:
            self.logger.error(f"Failed to display PDF: {e}", exc_info=True)
            self._show_error_widget(str(e))

    def _on_tikz_error(self, error_message: str):
        """Handle TikZ compilation error."""
        self.progress_label.setText("Error")
        self.progress_bar.hide()

        self._show_error_widget(error_message)
        self.logger.error(f"TikZ compilation failed: {error_message}")

    def _show_error_widget(self, error_message: str):
        """Show error message in the visualization area."""
        # Hide image view, show loading widget with error
        self.image_view.hide()
        self.loading_widget.show()

        # Clear loading widget and add error
        layout = self.loading_widget.layout()
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        error_label = QLabel(
            f"<h3>Error rendering visualization</h3>"
            f"<p style='color: red;'>{error_message}</p>"
            f"<p>The model structure could not be visualized.</p>"
        )
        error_label.setAlignment(Qt.AlignCenter)
        error_label.setWordWrap(True)
        layout.addWidget(error_label)

    def _on_save(self, format_type: str):
        """
        Save the visualization to a file.

        Args:
            format_type: File format ('png', 'pdf', or 'tex')
        """
        if format_type == 'tex':
            self._save_tex()
            return

        if self._pdf_path is None:
            QMessageBox.warning(self, "No Visualization",
                              "No visualization available to save.")
            return

        format_filters = {
            'png': "PNG Image (*.png)",
            'pdf': "PDF Document (*.pdf)",
        }

        default_name = f"{self.model_name.replace(' ', '_')}_architecture.{format_type}"
        file_filter = format_filters.get(format_type, "All Files (*)")

        file_path, _ = QFileDialog.getSaveFileName(
            self, f"Save Architecture as {format_type.upper()}",
            default_name, file_filter
        )

        if not file_path:
            return

        if not file_path.lower().endswith(f'.{format_type}'):
            file_path += f'.{format_type}'

        try:
            if format_type == 'pdf':
                # Copy the generated PDF
                shutil.copy2(self._pdf_path, file_path)
            elif format_type == 'png':
                # Convert PDF to PNG at high resolution
                images = convert_from_path(self._pdf_path, dpi=300)
                if images:
                    images[0].save(file_path, 'PNG')

            self.logger.info(f"Architecture diagram saved to {file_path}")
            QMessageBox.information(
                self, "Saved Successfully",
                f"Architecture diagram saved to:\n{file_path}"
            )

        except Exception as e:
            self.logger.error(f"Failed to save: {e}")
            QMessageBox.critical(self, "Save Failed", f"Failed to save:\n{str(e)}")

    def _save_tex(self):
        """Save the TikZ/LaTeX source code."""
        if self._tex_source is None:
            # Generate TEX without compiling
            gen = PlotNeuralNetGenerator(logger=self.logger)
            self._tex_source = gen.get_tex_source(
                self._blocks, self._skip_connections, self.model_name
            )

        default_name = f"{self.model_name.replace(' ', '_')}_architecture.tex"
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save LaTeX Source",
            default_name, "LaTeX Files (*.tex)"
        )

        if not file_path:
            return

        if not file_path.lower().endswith('.tex'):
            file_path += '.tex'

        try:
            with open(file_path, 'w') as f:
                f.write(self._tex_source)

            self.logger.info(f"LaTeX source saved to {file_path}")
            QMessageBox.information(
                self, "Saved Successfully",
                f"LaTeX source saved to:\n{file_path}\n\n"
                f"Compile with: pdflatex {os.path.basename(file_path)}"
            )
        except Exception as e:
            self.logger.error(f"Failed to save TEX: {e}")
            QMessageBox.critical(self, "Save Failed", f"Failed to save:\n{str(e)}")

    def get_pdf_path(self) -> Optional[str]:
        """Get the TikZ-generated PDF path (for external use)."""
        return self._pdf_path

    def get_tex_source(self) -> Optional[str]:
        """Get the TikZ/LaTeX source code (for external use)."""
        return self._tex_source

    def closeEvent(self, event):
        """Clean up worker thread and temp files on close."""
        if self._latex_worker is not None:
            if self._latex_worker.isRunning():
                self._latex_worker.terminate()
                self._latex_worker.wait(1000)
            # Clean up temp files
            self._latex_worker.cleanup()
        super().closeEvent(event)
