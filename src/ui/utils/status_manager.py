"""
Status Manager for centralized task state management.
Provides signals for task start/finish/error events to update the status bar.
"""
import logging
from PyQt5.QtCore import QObject, pyqtSignal


class StatusManager(QObject):
    """
    Centralized manager for application task status.
    Emits signals when tasks start, finish, or error.
    """

    # Signals
    task_started = pyqtSignal(str)   # Emits task name when a task starts
    task_finished = pyqtSignal()     # Emits when current task finishes
    task_error = pyqtSignal(str)     # Emits error message when task fails

    def __init__(self, logger: logging.Logger = None):
        super().__init__()
        if logger is not None:
            self.logger = logger.getChild("StatusManager")
        else:
            self.logger = logging.getLogger("SPIm.StatusManager")

        self._current_task = None
        self._is_busy = False
        self.logger.debug("StatusManager initialized")

    @property
    def is_busy(self) -> bool:
        """Returns True if a task is currently running."""
        return self._is_busy

    @property
    def current_task(self) -> str:
        """Returns the name of the current task, or None if idle."""
        return self._current_task

    def start_task(self, task_name: str):
        """
        Signal that a new task has started.

        Args:
            task_name: Human-readable name of the task (e.g., "Dataset generation")
        """
        self.logger.debug("Task started: %s", task_name)
        self._current_task = task_name
        self._is_busy = True
        self.task_started.emit(task_name)

    def finish_task(self):
        """Signal that the current task has finished successfully."""
        self.logger.debug("Task finished: %s", self._current_task)
        self._current_task = None
        self._is_busy = False
        self.task_finished.emit()

    def error_task(self, error_message: str):
        """
        Signal that the current task has failed with an error.

        Args:
            error_message: Description of the error
        """
        self.logger.error("Task error (%s): %s", self._current_task, error_message)
        self._current_task = None
        self._is_busy = False
        self.task_error.emit(error_message)
