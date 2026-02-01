"""
PlotNeuralNet-style 3D visualization of neural network architectures.
Renders PyTorch models as colored 3D blocks using matplotlib with proper
semantic grouping and skip connection visualization for U-Net architectures.
"""
import logging
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch, Polygon, FancyArrowPatch
from matplotlib.colors import to_rgba
import matplotlib.patches as mpatches
import torch.nn as nn


class BlockType(Enum):
    """Types of semantic blocks in neural networks."""
    INPUT = "input"
    CONV_BLOCK = "conv_block"      # Conv + optional BN + optional activation
    POOL = "pool"
    UPSAMPLE = "upsample"
    BOTTLENECK = "bottleneck"
    OUTPUT = "output"
    FC = "fc"
    RESIDUAL = "residual"
    ATTENTION = "attention"


# Color scheme for different block types
BLOCK_COLORS = {
    BlockType.INPUT: '#3498db',        # Blue
    BlockType.CONV_BLOCK: '#3498db',   # Blue
    BlockType.POOL: '#e74c3c',         # Red
    BlockType.UPSAMPLE: '#9b59b6',     # Purple
    BlockType.BOTTLENECK: '#f39c12',   # Orange
    BlockType.OUTPUT: '#2ecc71',       # Green
    BlockType.FC: '#1abc9c',           # Teal
    BlockType.RESIDUAL: '#3498db',     # Blue
    BlockType.ATTENTION: '#e91e63',    # Pink
}


@dataclass
class SemanticBlock:
    """Represents a semantic block in the network (grouped layers)."""
    name: str
    block_type: BlockType
    in_channels: int
    out_channels: int
    spatial_size: int = 32
    params: int = 0
    level: int = 0  # Encoder/decoder level (0 = input, increases with depth)
    is_encoder: bool = True

    def get_color(self) -> str:
        return BLOCK_COLORS.get(self.block_type, '#7f8c8d')


class NetworkVisualizer:
    """
    Renders neural network architecture as PlotNeuralNet-style 3D block diagram.

    Supports:
    - Semantic grouping of layers (Conv+BN+ReLU → single block)
    - U-Net style visualization with skip connections
    - Proper 3D perspective with depth representing channels
    """

    def __init__(self, figsize: Tuple[int, int] = (16, 10), dpi: int = 100,
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

    def extract_unet_blocks(self, model: nn.Module, input_size: int = 32) -> Tuple[List[SemanticBlock], List[Tuple[int, int]]]:
        """
        Extract semantic blocks from a U-Net architecture.
        Returns blocks and skip connection pairs (encoder_idx, decoder_idx).

        Args:
            model: The U-Net model
            input_size: Either an int (spatial size) or tuple (B, C, H, W)
        """
        blocks = []
        skip_connections = []
        encoder_block_indices = []  # Track indices of encoder conv blocks

        # Handle tuple input_size (B, C, H, W) -> extract spatial dimension
        if isinstance(input_size, (tuple, list)):
            input_size = input_size[-1]  # Use last dimension (W or H)

        # Input block
        blocks.append(SemanticBlock(
            name="Input",
            block_type=BlockType.INPUT,
            in_channels=1,
            out_channels=1,
            spatial_size=input_size,
            level=0,
            is_encoder=True
        ))

        current_spatial = input_size

        # Check for our UNet pattern: downs, pools, bottleneck, ups, up_convs
        if hasattr(model, 'downs') and hasattr(model, 'ups'):
            downs = list(model.downs)
            pools = list(model.pools) if hasattr(model, 'pools') else []
            ups = list(model.ups) if hasattr(model, 'ups') else []
            up_convs = list(model.up_convs) if hasattr(model, 'up_convs') else []

            # Encoder blocks
            for i, down in enumerate(downs):
                out_ch = self._get_output_channels(down)
                level = i + 1

                blocks.append(SemanticBlock(
                    name=f"Enc{i+1}",
                    block_type=BlockType.CONV_BLOCK,
                    in_channels=1 if i == 0 else self._get_output_channels(downs[i-1]),
                    out_channels=out_ch,
                    spatial_size=current_spatial,
                    params=sum(p.numel() for p in down.parameters()),
                    level=level,
                    is_encoder=True
                ))
                encoder_block_indices.append(len(blocks) - 1)

                # Pool after each encoder block
                if i < len(pools):
                    current_spatial = max(1, current_spatial // 2)
                    blocks.append(SemanticBlock(
                        name=f"Pool",
                        block_type=BlockType.POOL,
                        in_channels=out_ch,
                        out_channels=out_ch,
                        spatial_size=current_spatial,
                        level=level,
                        is_encoder=True
                    ))

            # Bottleneck
            if hasattr(model, 'bottleneck'):
                bot_in = self._get_output_channels(downs[-1]) if downs else 64
                bot_out = self._get_output_channels(model.bottleneck)
                max_level = len(downs) + 1
                blocks.append(SemanticBlock(
                    name="Bottleneck",
                    block_type=BlockType.BOTTLENECK,
                    in_channels=bot_in,
                    out_channels=bot_out,
                    spatial_size=current_spatial,
                    params=sum(p.numel() for p in model.bottleneck.parameters()),
                    level=max_level,
                    is_encoder=True
                ))

            # Decoder blocks
            num_levels = len(downs)
            for i, (up, up_conv) in enumerate(zip(ups, up_convs)):
                level = num_levels - i
                current_spatial = min(input_size, current_spatial * 2)

                # Upsample (ConvTranspose)
                up_out_ch = self._get_output_channels_transposed(up)
                blocks.append(SemanticBlock(
                    name=f"Up{i+1}",
                    block_type=BlockType.UPSAMPLE,
                    in_channels=blocks[-1].out_channels,
                    out_channels=up_out_ch,
                    spatial_size=current_spatial,
                    params=sum(p.numel() for p in up.parameters()),
                    level=level,
                    is_encoder=False
                ))

                # Decoder conv block (after concatenation with skip)
                dec_out_ch = self._get_output_channels(up_conv)
                dec_block = SemanticBlock(
                    name=f"Dec{i+1}",
                    block_type=BlockType.CONV_BLOCK,
                    in_channels=up_out_ch * 2,  # concat with skip
                    out_channels=dec_out_ch,
                    spatial_size=current_spatial,
                    params=sum(p.numel() for p in up_conv.parameters()),
                    level=level,
                    is_encoder=False
                )
                blocks.append(dec_block)

                # Skip connection: encoder level (num_levels - i) to this decoder
                enc_level_idx = num_levels - i - 1
                if 0 <= enc_level_idx < len(encoder_block_indices):
                    enc_idx = encoder_block_indices[enc_level_idx]
                    dec_idx = len(blocks) - 1
                    skip_connections.append((enc_idx, dec_idx))

            # Output
            if hasattr(model, 'final_conv'):
                out_params = sum(p.numel() for p in model.final_conv.parameters())
            else:
                out_params = 0

            blocks.append(SemanticBlock(
                name="Output",
                block_type=BlockType.OUTPUT,
                in_channels=blocks[-1].out_channels if blocks else 1,
                out_channels=1,
                spatial_size=input_size,
                params=out_params,
                level=0,
                is_encoder=False
            ))

        elif hasattr(model, 'enc_blocks') and hasattr(model, 'dec_blocks'):
            # Alternative pattern with enc_blocks and dec_blocks
            return self._extract_enc_dec_blocks(model, input_size)

        elif hasattr(model, 'encoder') and hasattr(model, 'decoder'):
            # Fully connected autoencoder pattern (encoder/decoder Sequential)
            return self._extract_fc_autoencoder(model, input_size)

        else:
            # Fallback: analyze module structure
            blocks = self._extract_from_modules(model, input_size)
            skip_connections = self._infer_skip_connections(blocks)

        return blocks, skip_connections

    def _extract_fc_autoencoder(self, model: nn.Module, input_size: int) -> Tuple[List[SemanticBlock], List[Tuple[int, int]]]:
        """
        Extract blocks from fully connected autoencoder with encoder/decoder Sequential.

        This handles models like:
            encoder = nn.Sequential(Linear, ReLU, Linear, ReLU, ...)
            decoder = nn.Sequential(Linear, ReLU, Linear, ReLU, ...)
        """
        blocks = []
        skip_connections = []  # FC autoencoders typically don't have skip connections

        # Input block
        blocks.append(SemanticBlock(
            name="Input",
            block_type=BlockType.INPUT,
            in_channels=1,
            out_channels=input_size * input_size,
            spatial_size=input_size,
            level=0,
            is_encoder=True
        ))

        # Extract encoder Linear layers
        encoder_layers = []
        for module in model.encoder.modules():
            if isinstance(module, nn.Linear):
                encoder_layers.append(module)

        # Add encoder FC blocks
        for i, layer in enumerate(encoder_layers):
            is_bottleneck = (i == len(encoder_layers) - 1)
            blocks.append(SemanticBlock(
                name="Bottleneck" if is_bottleneck else f"FC{i+1}",
                block_type=BlockType.BOTTLENECK if is_bottleneck else BlockType.FC,
                in_channels=layer.in_features,
                out_channels=layer.out_features,
                spatial_size=1,  # FC layers have no spatial dimension
                params=sum(p.numel() for p in layer.parameters()),
                level=i + 1,
                is_encoder=True
            ))

        # Extract decoder Linear layers
        decoder_layers = []
        for module in model.decoder.modules():
            if isinstance(module, nn.Linear):
                decoder_layers.append(module)

        # Add decoder FC blocks
        for i, layer in enumerate(decoder_layers):
            is_output = (i == len(decoder_layers) - 1)
            blocks.append(SemanticBlock(
                name="Output" if is_output else f"FC{len(encoder_layers) + i + 1}",
                block_type=BlockType.OUTPUT if is_output else BlockType.FC,
                in_channels=layer.in_features,
                out_channels=layer.out_features,
                spatial_size=input_size if is_output else 1,
                params=sum(p.numel() for p in layer.parameters()),
                level=len(encoder_layers) - i if not is_output else 0,
                is_encoder=False
            ))

        return blocks, skip_connections

    def _extract_enc_dec_blocks(self, model: nn.Module, input_size: int) -> Tuple[List[SemanticBlock], List[Tuple[int, int]]]:
        """Extract blocks from enc_blocks/dec_blocks pattern."""
        blocks = []
        skip_connections = []
        encoder_block_indices = []

        # Handle tuple input_size (B, C, H, W) -> extract spatial dimension
        if isinstance(input_size, (tuple, list)):
            input_size = input_size[-1]
        current_spatial = input_size

        blocks.append(SemanticBlock(
            name="Input",
            block_type=BlockType.INPUT,
            in_channels=1,
            out_channels=1,
            spatial_size=input_size,
            level=0,
            is_encoder=True
        ))

        enc_blocks = list(model.enc_blocks)
        for i, enc in enumerate(enc_blocks):
            out_ch = self._get_output_channels(enc)
            level = i + 1

            blocks.append(SemanticBlock(
                name=f"Enc{i+1}",
                block_type=BlockType.CONV_BLOCK,
                in_channels=1 if i == 0 else self._get_output_channels(enc_blocks[i-1]),
                out_channels=out_ch,
                spatial_size=current_spatial,
                params=sum(p.numel() for p in enc.parameters()),
                level=level,
                is_encoder=True
            ))
            encoder_block_indices.append(len(blocks) - 1)
            current_spatial = max(1, current_spatial // 2)

        if hasattr(model, 'bottleneck'):
            blocks.append(SemanticBlock(
                name="Bottleneck",
                block_type=BlockType.BOTTLENECK,
                in_channels=blocks[-1].out_channels,
                out_channels=self._get_output_channels(model.bottleneck),
                spatial_size=current_spatial,
                params=sum(p.numel() for p in model.bottleneck.parameters()),
                level=len(enc_blocks) + 1,
                is_encoder=True
            ))

        dec_blocks = list(model.dec_blocks) if hasattr(model, 'dec_blocks') else []
        for i, dec in enumerate(dec_blocks):
            level = len(enc_blocks) - i
            current_spatial = min(input_size, current_spatial * 2)

            blocks.append(SemanticBlock(
                name=f"Up{i+1}",
                block_type=BlockType.UPSAMPLE,
                in_channels=blocks[-1].out_channels,
                out_channels=blocks[-1].out_channels,
                spatial_size=current_spatial,
                level=level,
                is_encoder=False
            ))

            out_ch = self._get_output_channels(dec)
            blocks.append(SemanticBlock(
                name=f"Dec{i+1}",
                block_type=BlockType.CONV_BLOCK,
                in_channels=blocks[-1].out_channels,
                out_channels=out_ch,
                spatial_size=current_spatial,
                params=sum(p.numel() for p in dec.parameters()),
                level=level,
                is_encoder=False
            ))

            enc_idx = len(enc_blocks) - i - 1
            if 0 <= enc_idx < len(encoder_block_indices):
                skip_connections.append((encoder_block_indices[enc_idx], len(blocks) - 1))

        blocks.append(SemanticBlock(
            name="Output",
            block_type=BlockType.OUTPUT,
            in_channels=blocks[-1].out_channels if blocks else 1,
            out_channels=1,
            spatial_size=input_size,
            level=0,
            is_encoder=False
        ))

        return blocks, skip_connections

    def _get_output_channels_transposed(self, module: nn.Module) -> int:
        """Get output channels from ConvTranspose2d."""
        if isinstance(module, nn.ConvTranspose2d):
            return module.out_channels
        for child in module.modules():
            if isinstance(child, nn.ConvTranspose2d):
                return child.out_channels
        return self._get_output_channels(module)

    def _get_output_channels(self, module: nn.Module) -> int:
        """Get the output channels of a module."""
        # Try various common patterns
        for child in module.modules():
            if isinstance(child, (nn.Conv2d, nn.ConvTranspose2d)):
                return child.out_channels
            elif isinstance(child, nn.BatchNorm2d):
                return child.num_features
            elif isinstance(child, nn.GroupNorm):
                return child.num_channels

        # Check direct attributes
        if hasattr(module, 'out_channels'):
            return module.out_channels
        if hasattr(module, 'out_features'):
            return module.out_features

        return 64  # Default

    def _extract_from_modules(self, model: nn.Module, input_size: int) -> List[SemanticBlock]:
        """Extract blocks by analyzing module structure directly."""
        blocks = []
        current_channels = 1

        # Handle tuple input_size (B, C, H, W) -> extract spatial dimension
        if isinstance(input_size, (tuple, list)):
            input_size = input_size[-1]
        current_spatial = input_size

        blocks.append(SemanticBlock(
            name="Input",
            block_type=BlockType.INPUT,
            in_channels=1,
            out_channels=1,
            spatial_size=input_size,
            level=0,
            is_encoder=True
        ))

        # Group consecutive layers into semantic blocks
        pending_conv = None
        pending_params = 0
        block_idx = 0
        level = 0
        is_encoder = True
        seen_upsample = False

        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                # Flush previous pending conv as a block
                if pending_conv is not None:
                    block_idx += 1
                    blocks.append(SemanticBlock(
                        name=f"Conv{block_idx}",
                        block_type=BlockType.CONV_BLOCK,
                        in_channels=pending_conv.in_channels,
                        out_channels=pending_conv.out_channels,
                        spatial_size=current_spatial,
                        params=pending_params,
                        level=level,
                        is_encoder=is_encoder
                    ))
                    current_channels = pending_conv.out_channels

                pending_conv = module
                pending_params = sum(p.numel() for p in module.parameters())

                if module.stride[0] > 1:
                    current_spatial = max(1, current_spatial // module.stride[0])
                    level += 1

            elif isinstance(module, nn.ConvTranspose2d):
                # Flush pending
                if pending_conv is not None:
                    block_idx += 1
                    blocks.append(SemanticBlock(
                        name=f"Conv{block_idx}",
                        block_type=BlockType.CONV_BLOCK,
                        in_channels=pending_conv.in_channels,
                        out_channels=pending_conv.out_channels,
                        spatial_size=current_spatial,
                        params=pending_params,
                        level=level,
                        is_encoder=is_encoder
                    ))
                    current_channels = pending_conv.out_channels
                    pending_conv = None

                # Add upsample block
                if not seen_upsample:
                    seen_upsample = True
                    is_encoder = False

                current_spatial = min(input_size, current_spatial * module.stride[0])
                level = max(0, level - 1)

                block_idx += 1
                blocks.append(SemanticBlock(
                    name=f"Deconv{block_idx}",
                    block_type=BlockType.UPSAMPLE,
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    spatial_size=current_spatial,
                    params=sum(p.numel() for p in module.parameters()),
                    level=level,
                    is_encoder=False
                ))
                current_channels = module.out_channels

            elif isinstance(module, (nn.MaxPool2d, nn.AvgPool2d)):
                # Flush pending
                if pending_conv is not None:
                    block_idx += 1
                    blocks.append(SemanticBlock(
                        name=f"Conv{block_idx}",
                        block_type=BlockType.CONV_BLOCK,
                        in_channels=pending_conv.in_channels,
                        out_channels=pending_conv.out_channels,
                        spatial_size=current_spatial,
                        params=pending_params,
                        level=level,
                        is_encoder=is_encoder
                    ))
                    current_channels = pending_conv.out_channels
                    pending_conv = None

                stride = module.stride if isinstance(module.stride, int) else module.stride[0]
                current_spatial = max(1, current_spatial // stride)
                level += 1

                blocks.append(SemanticBlock(
                    name=f"Pool",
                    block_type=BlockType.POOL,
                    in_channels=current_channels,
                    out_channels=current_channels,
                    spatial_size=current_spatial,
                    level=level,
                    is_encoder=True
                ))

            elif isinstance(module, nn.Upsample):
                if pending_conv is not None:
                    block_idx += 1
                    blocks.append(SemanticBlock(
                        name=f"Conv{block_idx}",
                        block_type=BlockType.CONV_BLOCK,
                        in_channels=pending_conv.in_channels,
                        out_channels=pending_conv.out_channels,
                        spatial_size=current_spatial,
                        params=pending_params,
                        level=level,
                        is_encoder=is_encoder
                    ))
                    current_channels = pending_conv.out_channels
                    pending_conv = None

                if not seen_upsample:
                    seen_upsample = True
                    is_encoder = False

                scale = module.scale_factor if module.scale_factor else 2
                current_spatial = min(input_size, int(current_spatial * scale))
                level = max(0, level - 1)

                blocks.append(SemanticBlock(
                    name=f"Up",
                    block_type=BlockType.UPSAMPLE,
                    in_channels=current_channels,
                    out_channels=current_channels,
                    spatial_size=current_spatial,
                    level=level,
                    is_encoder=False
                ))

            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                # Add params to pending conv
                pending_params += sum(p.numel() for p in module.parameters())

            elif isinstance(module, nn.Linear):
                if pending_conv is not None:
                    block_idx += 1
                    blocks.append(SemanticBlock(
                        name=f"Conv{block_idx}",
                        block_type=BlockType.CONV_BLOCK,
                        in_channels=pending_conv.in_channels,
                        out_channels=pending_conv.out_channels,
                        spatial_size=current_spatial,
                        params=pending_params,
                        level=level,
                        is_encoder=is_encoder
                    ))
                    pending_conv = None

                blocks.append(SemanticBlock(
                    name="FC",
                    block_type=BlockType.FC,
                    in_channels=module.in_features,
                    out_channels=module.out_features,
                    spatial_size=1,
                    params=sum(p.numel() for p in module.parameters()),
                    level=0,
                    is_encoder=False
                ))

        # Flush any remaining pending conv
        if pending_conv is not None:
            block_idx += 1
            blocks.append(SemanticBlock(
                name=f"Conv{block_idx}",
                block_type=BlockType.CONV_BLOCK,
                in_channels=pending_conv.in_channels,
                out_channels=pending_conv.out_channels,
                spatial_size=current_spatial,
                params=pending_params,
                level=level,
                is_encoder=is_encoder
            ))

        # Add output block
        blocks.append(SemanticBlock(
            name="Output",
            block_type=BlockType.OUTPUT,
            in_channels=blocks[-1].out_channels if blocks else 1,
            out_channels=1,
            spatial_size=input_size,
            level=0,
            is_encoder=False
        ))

        return blocks

    def _extract_generic_unet(self, model: nn.Module, input_size: int) -> List[SemanticBlock]:
        """Extract blocks from generic U-Net-like structure."""
        return self._extract_from_modules(model, input_size)

    def _infer_skip_connections(self, blocks: List[SemanticBlock]) -> List[Tuple[int, int]]:
        """Infer skip connections based on block structure."""
        skip_connections = []

        # Find encoder blocks and matching decoder blocks by level
        encoder_blocks = [(i, b) for i, b in enumerate(blocks)
                         if b.is_encoder and b.block_type == BlockType.CONV_BLOCK]
        decoder_blocks = [(i, b) for i, b in enumerate(blocks)
                         if not b.is_encoder and b.block_type in (BlockType.CONV_BLOCK, BlockType.UPSAMPLE)]

        # Match by level
        for enc_idx, enc_block in encoder_blocks:
            for dec_idx, dec_block in decoder_blocks:
                if enc_block.level == dec_block.level and enc_block.level > 0:
                    skip_connections.append((enc_idx, dec_idx))
                    break

        return skip_connections

    def render(self, model: nn.Module, input_size: int = 32,
               model_name: str = "Neural Network") -> Figure:
        """Render the network architecture as a matplotlib figure."""
        arch_type = self.analyze_architecture(model)

        if arch_type == 'unet':
            return self._render_unet_style(model, input_size, model_name)
        else:
            return self._render_sequential_style(model, input_size, model_name)

    def _render_unet_style(self, model: nn.Module, input_size: int,
                          model_name: str) -> Figure:
        """Render U-Net style architecture with skip connections."""
        blocks, skip_connections = self.extract_unet_blocks(model, input_size)
        total_params = sum(p.numel() for p in model.parameters())

        fig = Figure(figsize=self.figsize, dpi=self.dpi, facecolor='white')
        ax = fig.add_subplot(111)
        ax.set_facecolor('#fafafa')

        if not blocks:
            ax.text(0.5, 0.5, "No blocks to visualize",
                    ha='center', va='center', fontsize=14,
                    transform=ax.transAxes)
            return fig

        # Find max level for U-shape layout
        max_level = max(b.level for b in blocks)

        # Separate encoder and decoder blocks
        encoder_blocks = [b for b in blocks if b.is_encoder]
        decoder_blocks = [b for b in blocks if not b.is_encoder]

        # Layout parameters
        block_width = 1.2
        block_spacing_h = 2.0
        level_spacing_v = 2.5
        max_height = 4.0
        max_depth = 1.5

        # Calculate positions
        positions = {}
        block_dims = {}

        # Max channels for scaling
        max_channels = max(max(b.in_channels, b.out_channels) for b in blocks)

        # Position encoder blocks (left side, going down)
        x = 0
        for i, block in enumerate(encoder_blocks):
            y = -block.level * level_spacing_v

            # Block dimensions
            height = min(block.out_channels / max(max_channels, 1) * max_height, max_height)
            height = max(height, 0.8)
            depth = min(block.spatial_size / input_size * max_depth, max_depth)
            depth = max(depth, 0.4)

            positions[id(block)] = (x, y)
            block_dims[id(block)] = (block_width, height, depth)

            if block.block_type != BlockType.POOL:
                x += block_spacing_h

        # Position decoder blocks (right side, going up)
        # Start from the rightmost encoder position
        x = max(p[0] for p in positions.values()) + block_spacing_h * 2

        for i, block in enumerate(decoder_blocks):
            y = -block.level * level_spacing_v

            height = min(block.out_channels / max(max_channels, 1) * max_height, max_height)
            height = max(height, 0.8)
            depth = min(block.spatial_size / input_size * max_depth, max_depth)
            depth = max(depth, 0.4)

            positions[id(block)] = (x, y)
            block_dims[id(block)] = (block_width, height, depth)

            x += block_spacing_h

        # Draw skip connections first (behind blocks)
        for enc_idx, dec_idx in skip_connections:
            if enc_idx < len(blocks) and dec_idx < len(blocks):
                enc_block = blocks[enc_idx]
                dec_block = blocks[dec_idx]

                if id(enc_block) in positions and id(dec_block) in positions:
                    enc_pos = positions[id(enc_block)]
                    dec_pos = positions[id(dec_block)]
                    enc_dims = block_dims[id(enc_block)]
                    dec_dims = block_dims[id(dec_block)]

                    # Draw curved arrow for skip connection
                    start_x = enc_pos[0] + enc_dims[0] / 2
                    start_y = enc_pos[1] + enc_dims[1] / 2
                    end_x = dec_pos[0] - dec_dims[0] / 2
                    end_y = dec_pos[1] + dec_dims[1] / 2

                    # Draw as a bezier curve
                    arrow = FancyArrowPatch(
                        (start_x, start_y), (end_x, end_y),
                        connectionstyle=f"arc3,rad=0.3",
                        arrowstyle='-|>',
                        mutation_scale=15,
                        lw=2,
                        color='#888',
                        alpha=0.7,
                        zorder=1
                    )
                    ax.add_patch(arrow)

        # Draw blocks
        for block in blocks:
            if id(block) in positions:
                x, y = positions[id(block)]
                width, height, depth = block_dims[id(block)]

                self._draw_3d_block(ax, x, y, width, height, depth,
                                   block.get_color(), block)

        # Draw sequential connections (arrows between consecutive blocks)
        all_blocks = encoder_blocks + decoder_blocks
        for i in range(len(all_blocks) - 1):
            b1, b2 = all_blocks[i], all_blocks[i + 1]
            if id(b1) in positions and id(b2) in positions:
                p1 = positions[id(b1)]
                p2 = positions[id(b2)]
                d1 = block_dims[id(b1)]
                d2 = block_dims[id(b2)]

                # Skip if same position (encoder to decoder transition handled by skip)
                if abs(p1[0] - p2[0]) < 0.1:
                    continue

                ax.annotate('',
                           xy=(p2[0] - d2[0]/2 - 0.1, p2[1] + d2[1]/2),
                           xytext=(p1[0] + d1[0]/2 + d1[2]*0.4 + 0.1, p1[1] + d1[1]/2),
                           arrowprops=dict(arrowstyle='->', color='#555', lw=1.5),
                           zorder=2)

        # Add legend
        self._add_legend(ax, blocks)

        # Title
        if total_params >= 1_000_000:
            param_str = f"{total_params / 1_000_000:.2f}M"
        elif total_params >= 1_000:
            param_str = f"{total_params / 1_000:.1f}K"
        else:
            param_str = str(total_params)

        ax.set_title(f"{model_name}\nTotal Parameters: {param_str}",
                    fontsize=14, fontweight='bold', pad=15)

        # Auto-scale axes
        all_x = [p[0] for p in positions.values()]
        all_y = [p[1] for p in positions.values()]
        margin = 3
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin + 2)
        ax.set_ylim(min(all_y) - margin - 2, max(all_y) + margin + max_height)

        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')

        fig.tight_layout()
        return fig

    def _render_sequential_style(self, model: nn.Module, input_size: int,
                                 model_name: str) -> Figure:
        """Render sequential architecture (DnCNN, ResNet, etc.)."""
        blocks = self._extract_from_modules(model, input_size)
        total_params = sum(p.numel() for p in model.parameters())

        fig = Figure(figsize=self.figsize, dpi=self.dpi, facecolor='white')
        ax = fig.add_subplot(111)
        ax.set_facecolor('#fafafa')

        if not blocks:
            ax.text(0.5, 0.5, "No blocks to visualize",
                    ha='center', va='center', fontsize=14,
                    transform=ax.transAxes)
            return fig

        # Merge similar consecutive blocks for cleaner visualization
        merged_blocks = self._merge_similar_blocks(blocks)

        # Layout parameters
        block_width = 1.0
        block_spacing = 1.8
        max_height = 5.0
        max_depth = 2.0

        max_channels = max(max(b.in_channels, b.out_channels) for b in merged_blocks)

        x_offset = 0
        positions = []

        for block in merged_blocks:
            height = min(block.out_channels / max(max_channels, 1) * max_height, max_height)
            height = max(height, 0.6)
            depth = min(block.spatial_size / input_size * max_depth, max_depth)
            depth = max(depth, 0.3)

            self._draw_3d_block(ax, x_offset, 0, block_width, height, depth,
                               block.get_color(), block)

            positions.append((x_offset + block_width/2, height/2))
            x_offset += block_width + block_spacing

        # Draw arrows
        for i in range(len(positions) - 1):
            ax.annotate('',
                       xy=(positions[i+1][0] - block_width/2 - 0.1, positions[i+1][1]),
                       xytext=(positions[i][0] + block_width/2 + 0.3, positions[i][1]),
                       arrowprops=dict(arrowstyle='->', color='#555', lw=1.5))

        # Legend
        self._add_legend(ax, merged_blocks)

        # Title
        if total_params >= 1_000_000:
            param_str = f"{total_params / 1_000_000:.2f}M"
        elif total_params >= 1_000:
            param_str = f"{total_params / 1_000:.1f}K"
        else:
            param_str = str(total_params)

        ax.set_title(f"{model_name}\nTotal Parameters: {param_str}",
                    fontsize=14, fontweight='bold', pad=15)

        ax.set_xlim(-1, x_offset + 1)
        ax.set_ylim(-3, max_height + max_depth + 1)
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')

        fig.tight_layout()
        return fig

    def _merge_similar_blocks(self, blocks: List[SemanticBlock]) -> List[SemanticBlock]:
        """Merge consecutive similar blocks for cleaner visualization."""
        if not blocks:
            return blocks

        merged = [blocks[0]]

        for block in blocks[1:]:
            prev = merged[-1]

            # Merge consecutive conv blocks at same level
            if (block.block_type == prev.block_type == BlockType.CONV_BLOCK and
                block.level == prev.level and
                abs(block.out_channels - prev.out_channels) < prev.out_channels * 0.5):
                # Update previous block to include this one
                merged[-1] = SemanticBlock(
                    name=f"{prev.name}+",
                    block_type=prev.block_type,
                    in_channels=prev.in_channels,
                    out_channels=block.out_channels,
                    spatial_size=block.spatial_size,
                    params=prev.params + block.params,
                    level=prev.level,
                    is_encoder=prev.is_encoder
                )
            else:
                merged.append(block)

        return merged

    def _draw_3d_block(self, ax, x: float, y: float, width: float,
                       height: float, depth: float, color: str,
                       block: SemanticBlock):
        """Draw a 3D-ish block with proper perspective."""
        # Perspective offset
        dx = depth * 0.4
        dy = depth * 0.4

        # Colors
        base_rgba = to_rgba(color)
        light_color = tuple(min(1.0, c + 0.2) for c in base_rgba[:3]) + (base_rgba[3],)
        dark_color = tuple(max(0.0, c - 0.2) for c in base_rgba[:3]) + (base_rgba[3],)

        # Front face
        front = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=color,
            edgecolor='#333',
            linewidth=1.5,
            alpha=0.95,
            zorder=3
        )
        ax.add_patch(front)

        # Top face
        top_x = [x, x + dx, x + width + dx, x + width]
        top_y = [y + height, y + height + dy, y + height + dy, y + height]
        top = Polygon(
            list(zip(top_x, top_y)),
            facecolor=light_color,
            edgecolor='#333',
            linewidth=1,
            alpha=0.9,
            zorder=2
        )
        ax.add_patch(top)

        # Right side
        side_x = [x + width, x + width + dx, x + width + dx, x + width]
        side_y = [y, y + dy, y + height + dy, y + height]
        side = Polygon(
            list(zip(side_x, side_y)),
            facecolor=dark_color,
            edgecolor='#333',
            linewidth=1,
            alpha=0.9,
            zorder=2
        )
        ax.add_patch(side)

        # Labels
        text_color = 'white' if self._is_dark(color) else 'black'

        # Block name
        ax.text(x + width/2, y + height/2 + 0.15, block.name,
                ha='center', va='center',
                fontsize=9, fontweight='bold',
                color=text_color, zorder=4)

        # Channels
        ax.text(x + width/2, y + height/2 - 0.25, f"{block.out_channels}ch",
                ha='center', va='center',
                fontsize=7,
                color=text_color, alpha=0.9, zorder=4)

        # Spatial size below block
        ax.text(x + width/2, y - 0.3, f"{block.spatial_size}×{block.spatial_size}",
                ha='center', va='top',
                fontsize=6, color='#555')

        # Parameter count (if significant)
        if block.params >= 1000:
            if block.params >= 1_000_000:
                param_text = f"{block.params / 1_000_000:.1f}M"
            else:
                param_text = f"{block.params / 1_000:.0f}K"
            ax.text(x + width/2, y - 0.55, param_text,
                    ha='center', va='top',
                    fontsize=5, color='#888')

    def _is_dark(self, color: str) -> bool:
        """Check if color is dark for text contrast."""
        rgba = to_rgba(color)
        luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
        return luminance < 0.5

    def _add_legend(self, ax, blocks: List[SemanticBlock]):
        """Add color legend for block types."""
        unique_types = []
        seen = set()

        for block in blocks:
            if block.block_type not in seen:
                unique_types.append((block.block_type, block.get_color()))
                seen.add(block.block_type)

        if not unique_types:
            return

        # Create legend handles
        handles = []
        labels = []

        type_names = {
            BlockType.INPUT: "Input",
            BlockType.CONV_BLOCK: "Conv Block",
            BlockType.POOL: "Pooling",
            BlockType.UPSAMPLE: "Upsample",
            BlockType.BOTTLENECK: "Bottleneck",
            BlockType.OUTPUT: "Output",
            BlockType.FC: "Fully Connected",
            BlockType.RESIDUAL: "Residual",
            BlockType.ATTENTION: "Attention",
        }

        for block_type, color in unique_types:
            handles.append(mpatches.Patch(facecolor=color, edgecolor='#333',
                                         linewidth=1))
            labels.append(type_names.get(block_type, str(block_type)))

        # Add skip connection to legend
        handles.append(FancyArrowPatch((0, 0), (1, 0), arrowstyle='-|>',
                                       color='#888', lw=2))
        labels.append("Skip Connection")

        ax.legend(handles, labels, loc='upper right', fontsize=8,
                 framealpha=0.9, edgecolor='#ccc')
