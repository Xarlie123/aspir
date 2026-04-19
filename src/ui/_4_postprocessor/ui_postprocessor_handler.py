import logging
from PyQt5.QtWidgets import (QMessageBox, QWidget, QVBoxLayout,
                              QListWidget, QLabel, QSizePolicy, QFrame,
                              QPushButton, QFileDialog)
from PyQt5.QtCore import pyqtSignal, QObject, Qt

from ui.utils.worker_launcher import WorkerLauncher
from ui.utils.file_formats import MODELS_DIR
from ui._4_postprocessor.postprocessor_worker import PostprocesadoWorker
from ui.custom_widgets.postprocessor_control.nn_control.nn_control_widget import NNControlWidget
from ui.custom_widgets.visualizers.visual_postprocessor.visual_postprocessor_widget import VisualPostprocessorWidget
from ui.custom_widgets.common.dataset_split_widget import DatasetSplitWidget

from simulation_engine._4_postprocessor.postprocessor_nn import (
    MODEL_DISPLAY_NAMES,
    PostprocessorNN,
    display_to_key,
)


class UIPostprocessorHandler(QObject):
    images_ready = pyqtSignal(list, list, list)
    training_finished = pyqtSignal()  # Emitted when training completes successfully

    # Available neural network models — display names come from the single
    # source of truth in postprocessor_nn.MODEL_DISPLAY_NAMES so the menu
    # and the registry can never drift.
    MODELS = list(MODEL_DISPLAY_NAMES.values())

    def __init__(self, ui, simulation, logger, status_manager=None):
        super().__init__()
        self.ui = ui
        self.simulation = simulation
        self.status_manager = status_manager
        self.logger = logger.getChild("UIPostprocessorHandler")
        self.logger.setLevel(logging.DEBUG)
        self.logger.debug("Initializing UIPostprocessorHandler")

        # Create NNControlWidget first (before embedding in menu interface)
        self.nn_control = NNControlWidget(
            parent=None,
            simulation=self.simulation,
            logger=self.logger
        )
        self.nn_control.trainRequested.connect(self.start_postprocessing)
        self.logger.debug("NNControlWidget created")

        # Create DatasetSplitWidget
        self.dataset_split = DatasetSplitWidget(parent=None, logger=self.logger)
        self.dataset_split.splitChanged.connect(self._on_split_changed)
        self.logger.debug("DatasetSplitWidget created")

        # Create VisualPostprocessorWidget (before menu interface setup)
        self.visual_pp = VisualPostprocessorWidget(parent=None, logger=self.logger)
        self.logger.debug("VisualPostprocessorWidget created")

        # Setup menu-based interface (uses the widgets created above)
        self._setup_menu_interface()

        # Connections
        self.nn_control.trainProgress.connect(self.visual_pp.progress_bar.setValue)
        self.nn_control.imagesReady.connect(self._on_images_ready)

    def _setup_menu_interface(self):
        """Setup menu-based interface with QListWidget, matching dataset/mask tabs."""
        self.logger.debug("Setting up menu-based interface for neural networks")

        # Create QListWidget for menu
        self.model_menu = QListWidget()
        for model_name in self.MODELS:
            self.model_menu.addItem(model_name)
        self.model_menu.setCurrentRow(0)
        self.model_menu.currentRowChanged.connect(self._on_model_selection_changed)

        # Style the menu (same as dataset/mask tabs)
        self.model_menu.setStyleSheet("""
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

        # Remove scroll bar from menu
        self.model_menu.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.model_menu.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Common style for content panels
        panel_style = """
            QWidget#contentPanel {
                background-color: #f5f5f5;
                border: 1px solid #ccc;
                border-radius: 4px;
            }
        """

        # Embed menu into placeholder
        self.logger.debug("Embedding menu into postprocessor_menu_placeholder")
        menu_layout = QVBoxLayout(self.ui.postprocessor_menu_placeholder)
        menu_layout.setContentsMargins(0, 0, 0, 0)
        menu_layout.addWidget(self.model_menu)

        # Create content panel with nn_control
        self.logger.debug("Embedding controls into postprocessor_content_placeholder")
        content_panel = QWidget()
        content_panel.setObjectName("contentPanel")
        content_panel.setStyleSheet(panel_style)

        content_layout = QVBoxLayout(content_panel)
        content_layout.setContentsMargins(10, 10, 10, 10)
        content_layout.setSpacing(10)

        # Dataset split widget at top
        self.dataset_split.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        content_layout.addWidget(self.dataset_split)

        # Title label
        title_label = QLabel("<h3>Training Parameters</h3>")
        content_layout.addWidget(title_label)

        # Ensure nn_control has proper size policy
        self.nn_control.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.nn_control.setMinimumHeight(100)
        content_layout.addWidget(self.nn_control)

        # Export buttons layout (horizontal)
        from PyQt5.QtWidgets import QHBoxLayout
        export_layout = QHBoxLayout()
        export_layout.setSpacing(8)

        # Export model weights button (.pt)
        self.export_button = QPushButton("Export (.pt)")
        self.export_button.setMinimumHeight(36)
        self.export_button.setStyleSheet("""
            QPushButton {
                background-color: #607D8B;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #546E7A;
            }
            QPushButton:pressed {
                background-color: #455A64;
            }
            QPushButton:disabled {
                background-color: #ccc;
                color: #666;
            }
        """)
        self.export_button.setToolTip(
            "Export trained model weights in PyTorch format (.pt)\n"
            "The model must be trained first before exporting.\n"
            "To load: model.load_state_dict(torch.load('model.pt'))"
        )
        self.export_button.clicked.connect(self._on_export_model)
        self.export_button.setEnabled(False)
        export_layout.addWidget(self.export_button)

        # Export ONNX button
        self.export_onnx_button = QPushButton("Export (.onnx)")
        self.export_onnx_button.setMinimumHeight(36)
        self.export_onnx_button.setStyleSheet("""
            QPushButton {
                background-color: #5E35B1;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #512DA8;
            }
            QPushButton:pressed {
                background-color: #4527A0;
            }
            QPushButton:disabled {
                background-color: #ccc;
                color: #666;
            }
        """)
        self.export_onnx_button.setToolTip(
            "Export model in ONNX format for deployment\n"
            "(FPGA with Vitis AI, TensorRT, OpenVINO, etc.)\n"
            "The model must be trained first before exporting."
        )
        self.export_onnx_button.clicked.connect(self._on_export_onnx)
        self.export_onnx_button.setEnabled(False)
        export_layout.addWidget(self.export_onnx_button)

        content_layout.addLayout(export_layout)

        # Add stretch to push content to top
        content_layout.addStretch()

        # Embed content panel into placeholder
        placeholder_layout = QVBoxLayout(self.ui.postprocessor_content_placeholder)
        placeholder_layout.setContentsMargins(0, 0, 0, 0)
        placeholder_layout.addWidget(content_panel)

        # Embed preview widget into postprocessor_preview_placeholder
        self.logger.debug("Embedding preview widget into postprocessor_preview_placeholder")

        # Style the preview placeholder with same panel style
        self.ui.postprocessor_preview_placeholder.setStyleSheet("""
            QWidget#postprocessor_preview_placeholder {
                background-color: #f5f5f5;
                border: 1px solid #ccc;
                border-radius: 4px;
            }
        """)
        self.ui.postprocessor_preview_placeholder.setObjectName("postprocessor_preview_placeholder")

        preview_layout = QVBoxLayout(self.ui.postprocessor_preview_placeholder)
        preview_layout.setContentsMargins(10, 10, 10, 10)

        preview_label = QLabel("<h3>Post-processing preview:</h3>")
        preview_label.setAlignment(Qt.AlignCenter)
        preview_layout.addWidget(preview_label)

        # Ensure visual widget expands
        self.visual_pp.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        preview_layout.addWidget(self.visual_pp, 1)

        self.logger.info("Menu-based interface setup complete")

    def _on_model_selection_changed(self, index):
        """Handle model selection from menu."""
        if 0 <= index < len(self.MODELS):
            model_name = self.MODELS[index]
            self.nn_control.set_model(model_name)
            self.logger.debug("Model selection changed to: %s", model_name)

    def get_current_model(self) -> str:
        """Get the currently selected model name."""
        return self.nn_control.get_model()

    def start_postprocessing(self, batch_size, lr, num_epochs, model_name, use_gpu=True,
                              weight_decay=1e-5, dropout=0.0, loss_function="MSE",
                              optimizer_name="Adam", arch_overrides=None):
        # Prevent duplicate threads
        if getattr(self, 'worker_thread', None) and self.worker_thread.isRunning():
            self.logger.warning("Postprocessing already running, ignoring request")
            QMessageBox.warning(None, "Attention", "Postprocessing is already running.")
            return

        self.logger.info(
            "Starting postprocessing: batch_size=%d, lr=%f, epochs=%d, model=%s, use_gpu=%s, "
            "weight_decay=%f, dropout=%.2f, loss_function=%s, optimizer=%s, arch_overrides=%s",
            batch_size, lr, num_epochs, model_name, use_gpu, weight_decay, dropout,
            loss_function, optimizer_name, arch_overrides
        )

        # Notify status manager that task is starting
        if self.status_manager:
            self.status_manager.start_task("Model training")

        # Clear previous loss lists in the visualizer
        if hasattr(self.visual_pp, 'val_losses'):
            self.visual_pp.val_losses.clear()
        if hasattr(self.visual_pp, 'test_losses'):
            self.visual_pp.test_losses.clear()

        # Prepare overrides from dataset and architecture config
        overrides = {"img_size": self.simulation.dataset.img_size}
        if arch_overrides:
            overrides.update(arch_overrides)

        # Get split ratios from widget
        split_ratios = self.get_split_ratios()
        self.logger.debug(
            "Dataset split ratios: train=%.2f, val=%.2f, test=%.2f",
            split_ratios['train'], split_ratios['validation'], split_ratios['test']
        )

        # Configure engine
        try:
            self.simulation.set_postprocessor(
                self.simulation.dataset,
                self.simulation.mask,
                self.simulation.applicator,
                postprocessor_cls = PostprocessorNN,
                model_name         = display_to_key(model_name),
                model_overrides    = overrides,
                batch_size         = batch_size,
                lr                 = lr,
                weight_decay       = weight_decay,
                dropout            = dropout,
                loss_function      = loss_function.lower(),
                optimizer_name     = optimizer_name.lower(),
                use_gpu            = use_gpu,
                train_ratio        = split_ratios['train'],
                val_ratio          = split_ratios['validation'],
                test_ratio         = split_ratios['test']
            )
            self.logger.debug("Postprocessor configured successfully (use_gpu=%s)", use_gpu)
        except Exception as e:
            self.logger.error("Error configuring postprocessor: %s", e, exc_info=True)
            QMessageBox.critical(None, "Error configuring postprocessor", str(e))
            if self.status_manager:
                self.status_manager.error_task(str(e)[:50])
            return

        # Create and launch worker
        self._worker = PostprocesadoWorker(
            self.simulation.postprocessor,
            mode="train",
            num_epochs=num_epochs,
            logger=self.logger
        )
        self._worker.result.connect(self.nn_control.imagesReady.emit)
        self._worker.metrics.connect(self._on_metrics_ready)

        # Connect phase-specific signals
        self._worker.phase_started.connect(self.visual_pp.on_phase_started)
        self._worker.phase_progress.connect(self.visual_pp.on_phase_progress)
        self._worker.phase_completed.connect(self.visual_pp.on_phase_completed)

        # Reset progress before starting
        self.visual_pp.reset_progress()

        self.logger.debug("Launching PostprocesadoWorker…")
        self.worker_thread = WorkerLauncher.launch(
            self._worker,
            on_progress=self.nn_control.trainProgress.emit,
            on_finished=self._on_training_finished,
            on_error=lambda e: self._on_training_error(e)
        )
        self.worker_thread.finished.connect(
            lambda: setattr(self, 'worker_thread', None)
        )
        self.logger.info("PostprocesadoWorker thread started")

    def _on_training_finished(self):
        """Callback when training finishes."""
        self.logger.info("Training finished")
        self.nn_control.trainFinished.emit()

        # Enable export buttons now that model is trained
        self.export_button.setEnabled(True)
        self.export_onnx_button.setEnabled(True)

        # Notify status manager that task is finished
        if self.status_manager:
            self.status_manager.finish_task()

        # Emit signal for step completion (used by stepper)
        self.training_finished.emit()

    def _on_training_error(self, e):
        """Handle training worker error."""
        self.logger.error("Error in PostprocesadoWorker: %s", e, exc_info=True)
        QMessageBox.critical(None, "Error in postprocessing", str(e))
        if self.status_manager:
            self.status_manager.error_task(str(e)[:50])

    def _on_export_model(self):
        """Export trained model weights to a .pt file."""
        if self.simulation.postprocessor is None:
            QMessageBox.warning(None, "No Model", "No trained model available to export.")
            return

        # Get model from postprocessor
        model = getattr(self.simulation.postprocessor, 'model', None)
        if model is None:
            QMessageBox.warning(None, "No Model", "Postprocessor has no model to export.")
            return

        # Build filesystem-safe filename from the canonical registry key
        model_name = display_to_key(self.nn_control.get_model()).replace("-", "_")
        default_filename = f"{model_name}_weights.pt"

        # Ensure models directory exists
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        # Open file dialog
        file_path, _ = QFileDialog.getSaveFileName(
            None,
            "Export Model Weights",
            str(MODELS_DIR / default_filename),
            "PyTorch Model (*.pt);;All Files (*.*)"
        )

        if not file_path:
            return  # User cancelled

        try:
            import torch
            # Save model state dict (recommended) or full model
            torch.save(model.state_dict(), file_path)
            self.logger.info("Model weights exported to: %s", file_path)
            QMessageBox.information(
                None, "Export Successful",
                f"Model weights exported successfully to:\n{file_path}\n\n"
                "To load: model.load_state_dict(torch.load('path.pt'))"
            )
        except Exception as e:
            self.logger.error("Failed to export model: %s", e, exc_info=True)
            QMessageBox.critical(None, "Export Failed", f"Failed to export model:\n{e}")

    def _on_export_onnx(self):
        """Export trained model to ONNX format for deployment (FPGA, TensorRT, etc.)."""
        import sys
        import importlib.util

        # Check for onnxscript dependency (required by PyTorch 2.x)
        spec = importlib.util.find_spec("onnxscript")
        if spec is None:
            QMessageBox.critical(
                None, "Missing Dependency",
                "ONNX export requires the 'onnxscript' package.\n\n"
                "Install it with:\n"
                "pip install onnxscript"
            )
            return

        if self.simulation.postprocessor is None:
            QMessageBox.warning(None, "No Model", "No trained model available to export.")
            return

        # Get model from postprocessor
        model = getattr(self.simulation.postprocessor, 'model', None)
        if model is None:
            QMessageBox.warning(None, "No Model", "Postprocessor has no model to export.")
            return

        # Get image size from dataset
        img_size = 64  # Default
        if self.simulation.dataset:
            img_size = getattr(self.simulation.dataset, 'img_size', 64)

        # Build filesystem-safe filename from the canonical registry key
        model_name = display_to_key(self.nn_control.get_model()).replace("-", "_")
        default_filename = f"{model_name}_{img_size}x{img_size}.onnx"

        # Ensure models directory exists
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        # Open file dialog
        file_path, _ = QFileDialog.getSaveFileName(
            None,
            "Export Model to ONNX",
            str(MODELS_DIR / default_filename),
            "ONNX Model (*.onnx);;All Files (*.*)"
        )

        if not file_path:
            return  # User cancelled

        try:
            import torch

            # Set model to evaluation mode
            model.eval()
            device = next(model.parameters()).device

            # Create dummy input with correct dimensions (batch=1, channels=1, H, W)
            dummy_input = torch.randn(1, 1, img_size, img_size, device=device)

            # Export to ONNX (opset 18 is minimum for PyTorch 2.x)
            # Note: Not using dynamic_axes since FPGA deployment uses fixed batch size
            torch.onnx.export(
                model,
                dummy_input,
                file_path,
                export_params=True,
                opset_version=18,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output']
            )

            self.logger.info("Model exported to ONNX: %s", file_path)
            QMessageBox.information(
                None, "ONNX Export Successful",
                f"Model exported successfully to:\n{file_path}\n\n"
                f"Input shape: (1, 1, {img_size}, {img_size})\n"
                f"Opset version: 18\n\n"
                "Next steps for FPGA (Vitis AI):\n"
                "1. Quantize: vai_q_onnx quantize --model model.onnx\n"
                "2. Compile: vai_c_xir -x model_quantized.onnx -a arch.json"
            )
        except Exception as e:
            self.logger.error("Failed to export ONNX: %s", e, exc_info=True)
            QMessageBox.critical(None, "ONNX Export Failed", f"Failed to export model to ONNX:\n{e}")

    def _on_images_ready(self, orig, recons, denoised):
        # Store validation preview triplets
        self.logger.debug(
            "Images ready: original=%d, recons=%d, denoised=%d",
            len(orig), len(recons), len(denoised)
        )
        self.simulation.validation_results = {
            'original': orig,
            'recons':   recons,
            'denoised': denoised
        }

        # Read model param count if present
        n_params = getattr(self.simulation.postprocessor, 'n_params', None)

        # Refresh preview widget
        self.visual_pp.set_images(orig, recons, denoised)
        self.visual_pp.update_info(
            num_images         = len(denoised),
            img_size           = self.simulation.dataset.img_size,
            dataset_type       = self.simulation.dataset.dataset_type,
            mask_type          = type(self.simulation.mask).__name__,
            postprocessor_type = self.nn_control.get_model(),
            n_params           = n_params
        )
        self.visual_pp.image_slider_value.setValue(0)

    def _on_metrics_ready(self, val_losses, test_losses, val_psnr, val_ssim, val_lpips):
        """Receive full metrics including PSNR/SSIM/LPIPS and plot."""
        self.logger.debug(
            "Metrics ready: val_losses=%d points, test_losses=%d points, PSNR=%d points, SSIM=%d points, LPIPS=%d points",
            len(val_losses), len(test_losses), len(val_psnr), len(val_ssim), len(val_lpips)
        )
        self.visual_pp.val_losses = val_losses
        self.visual_pp.test_losses = test_losses
        self.visual_pp.val_psnr = val_psnr
        self.visual_pp.val_ssim = val_ssim
        self.visual_pp.val_lpips = val_lpips
        self.visual_pp.plot_losses()

    def refresh_preview_from_state(self):
        """
        If the simulation already has validation_results (e.g., after loading an
        experiment), push them into the preview widget so the user sees them
        without retraining.
        """
        vr = getattr(self.simulation, "validation_results", None)
        if not vr:
            return
        n_params = getattr(getattr(self.simulation, "postprocessor", None), "n_params", None)
        self.visual_pp.set_images(vr["original"], vr["recons"], vr["denoised"])
        model_txt = self.nn_control.get_model()
        self.visual_pp.update_info(
            num_images=len(vr["denoised"]),
            img_size=getattr(self.simulation.dataset, "img_size", 0),
            dataset_type=getattr(self.simulation.dataset, "dataset_type", ""),
            mask_type=type(getattr(self.simulation, "mask", object())).__name__,
            postprocessor_type=model_txt,
            n_params=n_params
        )
        self.visual_pp.image_slider_value.setValue(0)

    def _on_split_changed(self, split: dict):
        """Handle dataset split change."""
        self.logger.debug(f"Dataset split changed: {split}")

    def update_dataset_info(self):
        """
        Update the dataset split widget with current dataset size.
        Call this when the dataset changes.
        """
        if self.simulation and self.simulation.dataset:
            # Dataset stores images in .data attribute
            n_images = len(getattr(self.simulation.dataset, 'data', []))
            self.dataset_split.set_total_images(n_images)
            self.logger.debug(f"Updated dataset split widget: {n_images} images")

    def get_split_ratios(self) -> dict:
        """Get the current train/validation/test split ratios."""
        return {
            'train': self.dataset_split.get_train_ratio(),
            'validation': self.dataset_split.get_validation_ratio(),
            'test': self.dataset_split.get_test_ratio(),
        }

    def get_content_widget(self):
        """Return the main container widget for stepper integration."""
        return self.ui.postprocessor_main_container
