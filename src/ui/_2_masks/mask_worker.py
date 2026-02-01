# File: ui/_2_mascaras/mascara_worker.py
import logging
from PyQt5.QtCore import QObject, pyqtSignal

class MascaraWorker(QObject):
    """
    Worker object that executes mask generation in a separate thread.
    Reports progress via signals and logs key events.
    """
    progress = pyqtSignal(int)      # Signal to report progress (percentage)
    finished = pyqtSignal()         # Signal emitted when generation is complete
    error = pyqtSignal(Exception)   # Signal emitted on error

    def __init__(self, mascara, *args, **kwargs):
        super().__init__()
        # Initialize logger for this worker
        self.logger = logging.getLogger(self.__class__.__name__)
        self.mascara = mascara
        self.args = args
        self.kwargs = kwargs
        self.logger.debug(
            "MascaraWorker initialized with %s, args=%s, kwargs=%s",
            type(mascara).__name__, args, kwargs
        )

    def run(self):
        """
        Execute the mask generation method with progress callback,
        ensuring not to pass a 'logger' parameter.
        """
        self.logger.info("MascaraWorker started for %s", type(self.mascara).__name__)
        try:
            # Filtramos cualquier parámetro 'logger' en kwargs
            safe_kwargs = {k: v for k, v in self.kwargs.items() if k != 'logger'}

            # Llama a generar_mascaras sólo con args y progress_callback
            self.mascara.generate_masks(
                *self.args,
                progress_callback=self.report_progress,
                **safe_kwargs
            )

            self.logger.info(
                "MascaraWorker finished successfully for %s",
                type(self.mascara).__name__
            )
            self.finished.emit()
        except Exception as e:
            self.logger.error(
                "MascaraWorker encountered error: %s", e, exc_info=True
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
