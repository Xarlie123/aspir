"""Re-measurement of timing/energy on previously executed batch experiments.

Usage flow: the user picks one or more loaded experiments in Batch Reports →
right-click → "Re-measure timing & energy…". Each experiment's
``.batch_analysis_report`` is copied with a ``_reexecuted_<device>_<ts>``
suffix; the fresh copy gets its timing/energy fields rewritten with values
measured on the current host (typically a Jetson). Original files are never
modified.
"""
from ui.custom_widgets.batch_reports.remeasure.remeasure_dialog import (
    RemeasureDialog,
)
from ui.custom_widgets.batch_reports.remeasure.remeasure_worker import (
    RemeasureConfig,
    RemeasureWorker,
)

__all__ = ["RemeasureConfig", "RemeasureDialog", "RemeasureWorker"]
