"""High-level façade — the :class:`NetworkVisualizer` used by the UI."""
from __future__ import annotations

import logging
from typing import Optional

import torch.nn as nn
from matplotlib.figure import Figure

from ui.custom_widgets.postprocessor_control.architecture_config.network_visualizer._extractors import (
    extract_unet_blocks,
)
from ui.custom_widgets.postprocessor_control.architecture_config.network_visualizer._renderers import (
    render_sequential_style,
    render_unet_style,
)
from ui.custom_widgets.postprocessor_control.architecture_config.network_visualizer._types import (
    SemanticBlock,
)


class NetworkVisualizer:
    """
    Renders neural network architecture as PlotNeuralNet-style 3D block diagram.

    Supports:
    - Semantic grouping of layers (Conv+BN+ReLU → single block)
    - U-Net style visualization with skip connections
    - Proper 3D perspective with depth representing channels
    """

    def __init__(self, figsize: tuple[int, int] = (16, 10), dpi: int = 100,
                 logger: Optional[logging.Logger] = None):
        self.figsize = figsize
        self.dpi = dpi
        self.logger = logger or logging.getLogger(__name__)

    def analyze_architecture(self, model: nn.Module) -> str:
        """Detect the architecture type of a model."""
        model_name = type(model).__name__.lower()

        if 'unet' in model_name or hasattr(model, 'enc_blocks'):
            return 'unet'
        elif 'autoencoder' in model_name:
            return 'autoencoder'
        elif 'gan' in model_name or 'cgan' in model_name:
            return 'gan'
        else:
            return 'sequential'

    def extract_unet_blocks(self, model: nn.Module, input_size: int = 32) -> tuple[list[SemanticBlock], list[tuple[int, int]]]:
        """
        Extract semantic blocks from a U-Net architecture.
        Returns blocks and skip connection pairs (encoder_idx, decoder_idx).
        """
        return extract_unet_blocks(model, input_size)

    def render(self, model: nn.Module, input_size: int = 32,
               model_name: str = "Neural Network") -> Figure:
        """Render the network architecture as a matplotlib figure."""
        arch_type = self.analyze_architecture(model)

        if arch_type == 'unet':
            return render_unet_style(model, input_size, model_name, self.figsize, self.dpi)
        else:
            return render_sequential_style(model, input_size, model_name, self.figsize, self.dpi)
