# File: ui/_5_pipeline/pipeline_worker.py

from PySide6.QtCore import QObject, Signal, Slot
import traceback
from simulation_engine._6_pipeline.pipeline_executor import execute_pipeline

class PipelineWorker(QObject):
    progress_task = Signal(int)
    progress_overall = Signal(int)
    finished = Signal(list)
    error = Signal(str)

    def __init__(self, tests):
        super().__init__()
        self.tests = tests

    @Slot()
    def run(self):
        try:
            # Validate size_px for ir_beam before execution
            for cfg in self.tests:
                ds = cfg.get('dataset', {})
                if ds.get('type') == 'ir_beam':
                    size = ds.get('size_px')
                    if size is None or size <= 0:
                        raise ValueError(
                            f"'size_px' debe ser > 0 para test '{cfg.get('name', '')}'"
                        )

            results = execute_pipeline(
                self.tests,
                progress_per_task=lambda curr, total: self.progress_task.emit(int(curr/total*100)),
                progress_overall=lambda idx, total: self.progress_overall.emit(int(idx/total*100))
            )
            self.finished.emit(results)
        except Exception:
            tb = traceback.format_exc()
            self.error.emit(tb)