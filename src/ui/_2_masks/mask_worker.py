import logging
from PySide6.QtCore import QObject, Signal

class MaskWorker(QObject):
    """
    Worker object that executes mask generation in a separate thread.
    Reports progress via signals and logs key events.
    """
    progress = Signal(int)      # Signal to report progress (percentage)
    finished = Signal()         # Signal emitted when generation is complete
    error = Signal(Exception)   # Signal emitted on error

    def __init__(self, mask, *args, **kwargs):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.mask = mask
        self.args = args
        self.kwargs = kwargs
        self.logger.debug(
            "MaskWorker initialized with %s, args=%s, kwargs=%s",
            type(mask).__name__, args, kwargs
        )

    def run(self):
        """
        Execute the mask generation method with progress callback,
        ensuring not to pass a 'logger' parameter.
        """
        self.logger.info("MaskWorker started for %s", type(self.mask).__name__)
        try:
            safe_kwargs = {k: v for k, v in self.kwargs.items() if k != 'logger'}

            self.mask.generate_masks(
                *self.args,
                progress_callback=self.report_progress,
                **safe_kwargs
            )

            self.logger.info(
                "MaskWorker finished successfully for %s",
                type(self.mask).__name__
            )
            self.finished.emit()
        except Exception as e:
            self.logger.error(
                "MaskWorker encountered error: %s", e, exc_info=True
            )
            self.error.emit(e)

    def report_progress(self, current, total):
        """
        Callback passed to the mask generator to report progress.
        Converts (current, total) to a percentage and emits signal.
        """
        percent = int((current / total) * 100) if total else 0
        self.logger.debug("Progress: %d/%d (%d%%)", current, total, percent)
        self.progress.emit(percent)
