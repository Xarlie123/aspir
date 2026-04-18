"""Popup dialogs for exporting publication-quality figures from batch reports."""
from ui.custom_widgets.batch_reports.comparison_views.figure_export_popups.interactive_html import (
    InteractiveHTMLPopup,
)
from ui.custom_widgets.batch_reports.comparison_views.figure_export_popups.quality_sampling_ratio import (
    QualitySamplingRatioPopup,
)
from ui.custom_widgets.batch_reports.comparison_views.figure_export_popups.samples_grid import (
    SamplesGridPopup,
)
from ui.custom_widgets.batch_reports.comparison_views.figure_export_popups.visual_comparison import (
    VisualComparisonPopup,
)

__all__ = [
    "InteractiveHTMLPopup",
    "QualitySamplingRatioPopup",
    "SamplesGridPopup",
    "VisualComparisonPopup",
]
