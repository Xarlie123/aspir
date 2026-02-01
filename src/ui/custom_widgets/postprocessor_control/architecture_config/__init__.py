"""
Architecture configuration widgets package.

Provides UI components for configuring and visualizing neural network architectures:
- ArchitectureConfigWidget: Dynamic parameter configuration panel
- ArchitecturePreviewPopup: 3D visualization popup dialog (dual renderer)
- NetworkVisualizer: Matplotlib-based rendering engine
- PlotNeuralNetGenerator: TikZ/LaTeX publication-quality output
"""
from .architecture_config_widget import ArchitectureConfigWidget
from .architecture_preview_popup import ArchitecturePreviewPopup
from .network_visualizer import NetworkVisualizer, SemanticBlock, BlockType
from .plotneuralnet_generator import PlotNeuralNetGenerator, PDFLATEX_AVAILABLE

__all__ = [
    'ArchitectureConfigWidget',
    'ArchitecturePreviewPopup',
    'NetworkVisualizer',
    'SemanticBlock',
    'BlockType',
    'PlotNeuralNetGenerator',
    'PDFLATEX_AVAILABLE',
]
