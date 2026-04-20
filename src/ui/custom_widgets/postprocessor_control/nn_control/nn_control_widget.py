import logging
import inspect
from typing import Dict, Any, Optional

import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QSizePolicy, QMessageBox, QGroupBox, QVBoxLayout
from PyQt5.QtCore import pyqtSignal

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

from ui.custom_widgets.postprocessor_control.nn_control.nn_control import Ui_nn_control
from ui.custom_widgets.postprocessor_control.architecture_config import (
    ArchitectureConfigWidget, ArchitecturePreviewPopup
)
from ui.custom_widgets.common.button_styles import BUTTON_STYLE_GREEN, apply_button_style

# Import the generic postprocessor engine
from simulation_engine._4_postprocessor.postprocessor_nn import (
    MODEL_REGISTRY,
    PostprocessorNN,
    display_to_key,
)

class NNControlWidget(QtWidgets.QWidget, Ui_nn_control):
    """
    Generic Neural Network control widget for postprocessing.
    Allows configuration of hyperparameters and launching training in a background thread.
    Model selection is done via external menu.
    """
    # Signals
    trainProgress   = pyqtSignal(int)             # training progress percentage
    trainFinished   = pyqtSignal()                # emitted when training is done
    imagesReady     = pyqtSignal(list, list, list)  # original, noisy, reconstructed
    # batch, lr, epochs, model_name, use_gpu, weight_decay, dropout, loss_function, optimizer, arch_config
    trainRequested  = pyqtSignal(int, float, int, str, bool, float, float, str, str, dict)

    def __init__(self, parent=None, simulation=None, logger=None):
        super().__init__(parent)
        self.setupUi(self)

        # Set up logger: use provided or default ASPIR logger
        if logger is not None:
            self.logger = logger.getChild("NNControlWidget")
        else:
            self.logger = logging.getLogger("ASPIR.NNControlWidget")
        self.logger.setLevel(logging.DEBUG)
        self.logger.debug("Initializing NNControlWidget")

        self.simulation = simulation

        # Current selected model (set externally via menu)
        self._current_model = "Autoencoder"

        # Set expanding size policy to fill available space
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        # Check GPU availability and update status display
        self._gpu_available = self._check_gpu_availability()
        self._update_gpu_status_display()

        # Connect GPU checkbox to update logic
        self.enable_gpu_checkbox.toggled.connect(self._on_gpu_checkbox_toggled)

        # Add Architecture Configuration Widget after GPU checkbox
        self._setup_architecture_config()

        # Connect and style train button
        self.train_button.clicked.connect(self._on_button_clicked)
        apply_button_style(self.train_button, BUTTON_STYLE_GREEN)

        # When training finishes, handle post-training actions
        self.trainFinished.connect(self._on_train_finished)

    def _setup_architecture_config(self):
        """Setup the architecture configuration widget."""
        # Create the architecture config widget
        self.arch_config_widget = ArchitectureConfigWidget(self, logger=self.logger)

        # Insert it into the layout after the GPU checkbox (index 2)
        # Layout order: [0] gpu_status_label, [1] GPU checkbox, [2] arch_config, [3] formLayout, [4] train_button
        self.main_layout.insertWidget(2, self.arch_config_widget)

        # Connect the preview button signal
        self.arch_config_widget.previewRequested.connect(self._on_preview_architecture)

        # Initialize with default model
        self.arch_config_widget.set_model(self._current_model)

    def set_model(self, model_name: str):
        """Set the current model name (called from external menu)."""
        self._current_model = model_name
        # Update architecture config widget
        self.arch_config_widget.set_model(model_name)
        self.logger.debug("Model set to: %s", model_name)

    def get_model(self) -> str:
        """Get the current model name."""
        return self._current_model

    def get_architecture_config(self) -> Dict[str, Any]:
        """Get the current architecture configuration."""
        return self.arch_config_widget.get_config()

    def _on_preview_architecture(self):
        """Show architecture preview popup with the current configuration."""
        model_key = display_to_key(self._current_model)
        entry = MODEL_REGISTRY.get(model_key)

        if not entry:
            QMessageBox.warning(
                self, "Model Not Found",
                f"Model '{self._current_model}' not found in registry."
            )
            return

        try:
            # Get architecture config
            arch_config = self.arch_config_widget.get_config()

            # Merge with defaults
            merged_config = {**entry['defaults'], **arch_config}

            # Get image size from simulation if available
            img_size = 32
            if self.simulation and hasattr(self.simulation, 'dataset') and self.simulation.dataset:
                img_size = getattr(self.simulation.dataset, 'img_size', 32)
            merged_config['img_size'] = img_size

            # Filter kwargs to match model constructor signature
            model_cls = entry['cls']
            sig = inspect.signature(model_cls.__init__)
            valid_params = set(sig.parameters.keys()) - {'self'}
            filtered_config = {k: v for k, v in merged_config.items() if k in valid_params}

            # Instantiate model for visualization
            self.logger.debug(f"Creating preview model with config: {filtered_config}")
            model = model_cls(**filtered_config)

            # Get postprocessing results if available (after training)
            ground_truth_images = None
            noisy_images = None
            denoised_images = None

            if (self.simulation and
                hasattr(self.simulation, 'postprocessor') and
                self.simulation.postprocessor is not None):
                try:
                    # Try to get test results - this only works after training
                    orig, noise, recon = self.simulation.postprocessor.test_dataset()
                    if orig and noise and recon:
                        ground_truth_images = np.array(orig) if isinstance(orig, list) else orig
                        noisy_images = np.array(noise) if isinstance(noise, list) else noise
                        denoised_images = np.array(recon) if isinstance(recon, list) else recon
                        self.logger.debug(f"Loaded postprocessing results: {len(orig)} images")
                except Exception as e:
                    # Model not trained yet or no results available
                    self.logger.debug(f"No postprocessing results available: {e}")

            # Show preview popup
            popup = ArchitecturePreviewPopup(
                model=model,
                model_name=self._current_model,
                input_size=img_size,
                config=arch_config,
                ground_truth_images=ground_truth_images,
                noisy_images=noisy_images,
                denoised_images=denoised_images,
                parent=self,
                logger=self.logger
            )
            popup.exec_()

        except Exception as e:
            self.logger.error(f"Failed to create architecture preview: {e}", exc_info=True)
            QMessageBox.critical(
                self, "Preview Error",
                f"Failed to create architecture preview:\n{str(e)}"
            )

    def _on_button_clicked(self):
        """
        Handle train button click: read parameters, log them, and emit trainRequested.
        """
        batch = self.batch_size_spinbox.value()
        lr    = self.learning_rate_dspinbox.value()
        epochs = self.epochs_spinbox.value()
        model = self._current_model
        use_gpu = self.enable_gpu_checkbox.isChecked()
        weight_decay = self.weight_decay_dspinbox.value()
        dropout = self.dropout_dspinbox.value()
        loss_function = self.loss_function_combo.currentText()
        optimizer = self.optimizer_combo.currentText()
        arch_config = self.arch_config_widget.get_config()

        self.logger.info(
            "Train requested: batch_size=%d, lr=%.5f, epochs=%d, model=%s, use_gpu=%s, "
            "weight_decay=%.6f, dropout=%.2f, loss_function=%s, optimizer=%s, arch_config=%s",
            batch, lr, epochs, model, use_gpu, weight_decay, dropout, loss_function, optimizer, arch_config
        )
        self.trainRequested.emit(batch, lr, epochs, model, use_gpu, weight_decay, dropout,
                                 loss_function, optimizer, arch_config)

    def is_gpu_enabled(self) -> bool:
        """Return whether GPU training is enabled."""
        return self.enable_gpu_checkbox.isChecked() and self._gpu_available

    def _check_gpu_availability(self) -> bool:
        """Check if CUDA GPU is available."""
        if not TORCH_AVAILABLE:
            self.logger.warning("PyTorch not available, GPU disabled")
            return False
        available = torch.cuda.is_available()
        if available:
            device_name = torch.cuda.get_device_name(0)
            self.logger.info(f"GPU available: {device_name}")
        else:
            self.logger.info("No CUDA GPU available")
        return available

    def _update_gpu_status_display(self):
        """Update the GPU status label with color indicator."""
        if self._gpu_available:
            device_name = torch.cuda.get_device_name(0) if TORCH_AVAILABLE else "Unknown"
            self.gpu_status_label.setText(f"GPU Available: {device_name}")
            self.gpu_status_label.setStyleSheet(
                "QLabel { color: #228B22; background-color: #E8F5E9; "
                "padding: 4px 8px; border-radius: 4px; border: 1px solid #A5D6A7; }"
            )
            self.enable_gpu_checkbox.setEnabled(True)
        else:
            self.gpu_status_label.setText("GPU Not Available (CPU only)")
            self.gpu_status_label.setStyleSheet(
                "QLabel { color: #B22222; background-color: #FFEBEE; "
                "padding: 4px 8px; border-radius: 4px; border: 1px solid #EF9A9A; }"
            )
            self.enable_gpu_checkbox.setEnabled(False)
            self.enable_gpu_checkbox.setChecked(False)

    def _on_gpu_checkbox_toggled(self, checked: bool):
        """Handle GPU checkbox state change."""
        if checked and not self._gpu_available:
            self.enable_gpu_checkbox.setChecked(False)
            self.logger.warning("Cannot enable GPU - not available")
            return
        self.logger.debug(f"GPU training {'enabled' if checked else 'disabled'}")

    def refresh_gpu_status(self):
        """Refresh GPU availability status (call if hardware may have changed)."""
        self._gpu_available = self._check_gpu_availability()
        self._update_gpu_status_display()

    @property
    def gpu_available(self) -> bool:
        """Return True if GPU is available."""
        return self._gpu_available

    def _on_train_finished(self):
        """
        Called when training finishes: retrieve images and emit imagesReady.
        """
        self.logger.debug("Training finished signal received, fetching results")
        try:
            orig, noise, recon = self.simulation.postprocessor.test_dataset()
            self.logger.info(
                "Emitting imagesReady: orig=%d, noise=%d, recon=%d",
                len(orig), len(noise), len(recon)
            )
            self.imagesReady.emit(orig, noise, recon)
        except Exception as e:
            self.logger.error("Failed to retrieve postprocessing results: %s", e, exc_info=True)
            QMessageBox.critical(self, "Error", f"Could not load results: {e}")
