# File: ui/_1_dataset/dataset_worker.py

import logging
from PyQt5.QtCore import QObject, pyqtSignal

class DatasetWorker(QObject):
    """Worker to generate the dataset in a separate thread with progress signal."""
    progress = pyqtSignal(int)      # Signal for progress percentage
    finished = pyqtSignal()         # Signal for successful completion
    error = pyqtSignal(Exception)   # Signal for error

    def __init__(self, dataset, logger=None):
        super().__init__()
        self.dataset = dataset

        # Logger
        if logger is None:
            self.logger = logging.getLogger(self.__class__.__name__)
        else:
            self.logger = logger.getChild(self.__class__.__name__)
        self.logger.debug("Initializing DatasetWorker for dataset '%s'",
                          getattr(dataset, 'name', repr(dataset)))

    def run(self):
        """Executes load_data passing the progress callback."""
        self.logger.info("Starting dataset data loading")
        try:
            self.dataset.load_data(progress_callback=self.report_progress)
            self.logger.info("Dataset loaded successfully in memory")
            self.finished.emit()
        except Exception as e:
            self.logger.error("Error loading dataset: %s", e, exc_info=True)
            self.error.emit(e)
        finally:
            self.logger.debug("DatasetWorker.run() finished")

    def report_progress(self, current, total):
        """
        Progress callback that converts (current, total) to percentage
        and emits the corresponding signal.
        """
        percent = int((current / total) * 100) if total else 0
        self.logger.debug("Dataset progress: %d/%d (%d%%)", current, total, percent)
        self.progress.emit(percent)
