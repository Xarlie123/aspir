"""Architecture preview popup — publication-quality TikZ/LaTeX rendering."""
# Re-export PDFLATEX_AVAILABLE at the package top level — external callers
# (e.g. training_view.py) historically imported it from this module.
from ui.custom_widgets.postprocessor_control.architecture_config.architecture_preview_popup.popup import (
    ArchitecturePreviewPopup,
)
from ui.custom_widgets.postprocessor_control.architecture_config.plotneuralnet_generator import (
    PDFLATEX_AVAILABLE,
)

__all__ = ["ArchitecturePreviewPopup", "PDFLATEX_AVAILABLE"]
