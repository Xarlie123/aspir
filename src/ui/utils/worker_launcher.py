# ui/utils/worker_launcher.py

from PyQt5.QtCore import QThread


class WorkerLauncher:
    @staticmethod
    def launch(worker, *,
               on_progress=None,
               on_finished=None,
               on_error=None):
        """
        Moves `worker` (a QObject with signals progress(int), finished(), error(Exception))
        to a new thread, connects its signals and starts it.

        :param worker: instance of QObject with run() defined
        :param on_progress: function(int) called with each progress update
        :param on_finished: function() called when finished
        :param on_error:   function(Exception) called when an error occurs
        :returns: the running QThread
        """
        thread = QThread()
        worker.moveToThread(thread)

        if on_progress is not None:
            worker.progress.connect(on_progress)
        if on_finished is not None:
            worker.finished.connect(on_finished)
        if on_error is not None:
            worker.error.connect(on_error)

        # automatic cleanup
        worker.finished.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        thread.started.connect(worker.run)
        thread.start()
        return thread
