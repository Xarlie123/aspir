import logging
from PySide6.QtCore import QObject, Signal
import time  # For simulated preprocessing


class PostprocessorWorker(QObject):
    # Overall progress
    progress = Signal(int)               # % of progress
    finished = Signal()                  # task finished
    error    = Signal(Exception)         # on error
    result   = Signal(list, list, list)  # orig, noise, recon
    metrics  = Signal(list, list, list, list, list)  # val_losses, test_losses, val_psnr, val_ssim, val_lpips

    # Phase-specific signals
    phase_started = Signal(str)        # Phase name started
    phase_progress = Signal(str, int)  # Phase name, progress 0-100
    phase_completed = Signal(str)      # Phase name completed

    # Phase name constants
    PHASE_RECONSTRUCTION = "Reconstruction"
    PHASE_TRAINING = "Training"
    PHASE_INFERENCE = "Inference"

    def __init__(self, postprocessor, mode, num_epochs=None, index=None, logger=None):
        """
        Initialize the worker with the given postprocessor, mode, optional epoch count
        and an optional logger.
        """
        super().__init__()
        self.postprocessor = postprocessor
        self.mode = mode
        self.num_epochs = num_epochs
        self.index = index

        # Set up logger: use provided logger or create a new one
        if logger is not None:
            self.logger = logger.getChild("PostprocessorWorker")
        else:
            self.logger = logging.getLogger("ASPIR.PostprocessorWorker")
        self.logger.setLevel(logging.DEBUG)
        self.logger.debug("Initializing PostprocessorWorker")

    def run(self):
        """Execute training or inference with progress callbacks."""
        model = self.postprocessor.model
        num_params = sum(p.numel() for p in model.parameters())
        self.logger.info(
            "Using model %s with %d parameters",
            model.__class__.__name__, num_params
        )
        self.logger.debug("run() started")

        try:
            if self.mode == "train":
                # Phase 1: Reconstruction/Preprocessing
                self.phase_started.emit(self.PHASE_RECONSTRUCTION)
                dataset_steps = 10
                for i in range(dataset_steps):
                    time.sleep(0.1)
                    pct = int(((i + 1) / dataset_steps) * 100)
                    self.phase_progress.emit(self.PHASE_RECONSTRUCTION, pct)
                self.phase_completed.emit(self.PHASE_RECONSTRUCTION)

                self.logger.debug("Preprocessing loop done, starting train_with_metrics()")

                # Phase 2: Training
                self.phase_started.emit(self.PHASE_TRAINING)

                def training_progress_callback(epoch, total_epochs):
                    """Emit both phase progress and metrics signals."""
                    pct = int((epoch / total_epochs) * 100)
                    self.phase_progress.emit(self.PHASE_TRAINING, pct)

                # Training with metrics callback (val_losses, test_losses, val_psnr, val_ssim, val_lpips)
                self.postprocessor.train_with_metrics(
                    num_epochs=self.num_epochs,
                    progress_callback=training_progress_callback,
                    metrics_callback=lambda v, t, p, s, l: self.metrics.emit(v, t, p, s, l)
                )
                self.phase_completed.emit(self.PHASE_TRAINING)

                # Compute and log final validation & test losses
                val_loss  = self.postprocessor.validate()
                test_loss = self.postprocessor.test_loss()
                self.logger.info(
                    "Training finished: val_loss=%.4f, test_loss=%.4f",
                    val_loss, test_loss
                )

                # Emit finished signal
                self.finished.emit()

            elif self.mode == "infer":
                # Phase: Inference
                self.phase_started.emit(self.PHASE_INFERENCE)
                self.phase_progress.emit(self.PHASE_INFERENCE, 0)

                # Run inference
                orig, noise, recon = self.postprocessor.test_dataset()
                self.logger.debug(
                    "Inference results ready: orig=%d, noise=%d, recon=%d",
                    len(orig), len(noise), len(recon)
                )
                self.result.emit(orig, noise, recon)
                self.phase_progress.emit(self.PHASE_INFERENCE, 100)
                self.phase_completed.emit(self.PHASE_INFERENCE)
                self.finished.emit()

        except Exception as e:
            # Log exception with traceback
            self.logger.error("Exception in run(): %s", e, exc_info=True)
            self.error.emit(e)

        self.logger.debug("run() exiting")

    def _report(self, current, total):
        """Calculate percentage and emit progress signal."""
        pct = int((current / total) * 100) if total else 0
        self.logger.debug("Progress: %d%% (%d/%d)", pct, current, total)
        self.progress.emit(pct)
