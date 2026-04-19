"""
Popup dialog for previewing neural network architecture visualization.

Uses PlotNeuralNet (TikZ/LaTeX) for publication-quality output.
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch.nn as nn
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QLabel,
    QMessageBox,
    QWidget,
)

from ui.custom_widgets.postprocessor_control.architecture_config.architecture_preview_popup._ui_builder import (
    build_ui,
    populate_model_info_panel,
)
from ui.custom_widgets.postprocessor_control.architecture_config.latex_worker import (
    LaTeXCompilationWorker,
)
from ui.custom_widgets.postprocessor_control.architecture_config.network_visualizer import (
    NetworkVisualizer,
    SemanticBlock,
)
from ui.custom_widgets.postprocessor_control.architecture_config.plotneuralnet_generator import (
    PDFLATEX_AVAILABLE,
    PlotNeuralNetGenerator,
)

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
                 input_size: int = 32, config: Optional[dict[str, Any]] = None,
                 ground_truth_images: Optional[np.ndarray] = None,
                 noisy_images: Optional[np.ndarray] = None,
                 denoised_images: Optional[np.ndarray] = None,
                 parent: Optional[QWidget] = None,
                 logger: Optional[logging.Logger] = None,
                 # Batch test mode parameters
                 tests: Optional[list[dict[str, Any]]] = None,
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
        self._blocks: list[SemanticBlock] = []
        self._skip_connections: list[tuple[int, int]] = []
        self._extract_blocks()

        build_ui(self)

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
            # Apply per-test architecture overrides (features, depth, ...) so
            # the preview reflects what was actually trained.
            arch_overrides = test.get("architecture_config") or {}
            defaults.update(arch_overrides)
            if "img_size" in defaults:
                defaults["img_size"] = input_size
            # Filter to kwargs the model actually accepts, to stay robust to
            # stale overrides from older batch reports.
            import inspect
            sig = inspect.signature(model_cls.__init__)
            valid = set(sig.parameters) - {"self"}
            kwargs = {k: v for k, v in defaults.items() if k in valid}

            self.model = model_cls(**kwargs)
            self.model_name = model_name
            self.input_size = input_size
            self.config = kwargs

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
        populate_model_info_panel(self)

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
