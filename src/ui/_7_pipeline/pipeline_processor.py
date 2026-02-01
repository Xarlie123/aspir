# File: ui/_5_pipeline/pipeline_processor.py

import webbrowser
from PyQt5.QtCore import QObject, QThread, Qt
from PyQt5.QtWidgets import QMessageBox, QGraphicsScene
from PyQt5.QtGui import QPixmap

from simulation_engine._6_pipeline.pipeline_reporter import PipelineReporter
from ui._7_pipeline.pipeline_worker import PipelineWorker

class PipelineProcessor(QObject):
    """
    Handles pipeline execution, progress updates, and report generation.
    """

    def __init__(self, ui, tests):
        super().__init__()
        self.ui = ui
        self.tests = tests
        self.reporter = PipelineReporter()
        self._led_paths = {'green': 'ui/img/green.png', 'red': 'ui/img/red.png'}
        self._last_report_path = None

    def run_pipeline(self):
        """Start the pipeline execution in a background thread."""
        if not self.tests:
            QMessageBox.warning(None, "Warning", "No tests loaded.")
            return

        self._set_led('red')
        self.ui.results_progress_bar.setRange(0, 100)
        self.ui.results_total_progress_bar.setRange(0, 100)

        self.thread = QThread()
        self.worker = PipelineWorker(self.tests)
        self.worker.moveToThread(self.thread)
        self.worker.progress_task.connect(self.ui.results_progress_bar.setValue)
        self.worker.progress_overall.connect(self.ui.results_total_progress_bar.setValue)
        self.worker.finished.connect(self.on_pipeline_finished)
        self.worker.error.connect(self.on_pipeline_error)
        self.thread.started.connect(self.worker.run)
        self.thread.start()

    def on_pipeline_finished(self, results):
        """Handle pipeline completion: generate report and cleanup."""
        self._set_led('green')
        try:
            report_path = self.reporter.generate_report(results)
            self._last_report_path = report_path
            QMessageBox.information(None, "Report ready",
                                    "Click 'Open report' to view the report.")
        except Exception as e:
            QMessageBox.critical(None, "Error generating report", str(e))

        # Ensure progress bars are complete
        self.ui.results_progress_bar.setValue(100)
        self.ui.results_total_progress_bar.setValue(100)
        self.thread.quit()
        self.thread.wait()

    def on_pipeline_error(self, traceback_str):
        """Handle pipeline errors and reset UI state."""
        self._set_led('green')
        QMessageBox.critical(None, "Pipeline error", traceback_str)
        self.thread.quit()
        self.thread.wait()

    def open_report(self):
        """Open the last generated report in the default browser."""
        if not self._last_report_path:
            QMessageBox.warning(None, "No report", "No report has been generated yet.")
            return
        webbrowser.open(f"file://{self._last_report_path}")

    def _set_led(self, color: str):
        """Update the LED icon to indicate pipeline status."""
        path = self._led_paths.get(color)
        if not path:
            return
        scene = QGraphicsScene()
        pixmap = QPixmap(path)
        scene.addPixmap(pixmap)
        self.ui.led_pipeline_processing_image.setScene(scene)
        self.ui.led_pipeline_processing_image.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)
