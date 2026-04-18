"""Matplotlib renderers that turn semantic blocks into publication-quality figures."""
from __future__ import annotations

import matplotlib.patches as mpatches
import torch.nn as nn
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

from ui.custom_widgets.postprocessor_control.architecture_config.network_visualizer._extractors import (
    extract_from_modules,
    extract_unet_blocks,
)
from ui.custom_widgets.postprocessor_control.architecture_config.network_visualizer._types import (
    BlockType,
    SemanticBlock,
)


def is_dark(color: str) -> bool:
    """Check if color is dark for text contrast."""
    rgba = to_rgba(color)
    luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
    return luminance < 0.5


def draw_3d_block(ax, x: float, y: float, width: float,
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
    text_color = 'white' if is_dark(color) else 'black'

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


def add_legend(ax, blocks: list[SemanticBlock]):
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


def merge_similar_blocks(blocks: list[SemanticBlock]) -> list[SemanticBlock]:
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


def render_unet_style(model: nn.Module, input_size: int,
                      model_name: str, figsize: tuple[int, int],
                      dpi: int) -> Figure:
    """Render U-Net style architecture with skip connections."""
    blocks, skip_connections = extract_unet_blocks(model, input_size)
    total_params = sum(p.numel() for p in model.parameters())

    fig = Figure(figsize=figsize, dpi=dpi, facecolor='white')
    ax = fig.add_subplot(111)
    ax.set_facecolor('#fafafa')

    if not blocks:
        ax.text(0.5, 0.5, "No blocks to visualize",
                ha='center', va='center', fontsize=14,
                transform=ax.transAxes)
        return fig

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
    for block in encoder_blocks:
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

    for block in decoder_blocks:
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
                    connectionstyle="arc3,rad=0.3",
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

            draw_3d_block(ax, x, y, width, height, depth,
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
    add_legend(ax, blocks)

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


def render_sequential_style(model: nn.Module, input_size: int,
                            model_name: str, figsize: tuple[int, int],
                            dpi: int) -> Figure:
    """Render sequential architecture (DnCNN, ResNet, etc.)."""
    blocks = extract_from_modules(model, input_size)
    total_params = sum(p.numel() for p in model.parameters())

    fig = Figure(figsize=figsize, dpi=dpi, facecolor='white')
    ax = fig.add_subplot(111)
    ax.set_facecolor('#fafafa')

    if not blocks:
        ax.text(0.5, 0.5, "No blocks to visualize",
                ha='center', va='center', fontsize=14,
                transform=ax.transAxes)
        return fig

    # Merge similar consecutive blocks for cleaner visualization
    merged_blocks = merge_similar_blocks(blocks)

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

        draw_3d_block(ax, x_offset, 0, block_width, height, depth,
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
    add_legend(ax, merged_blocks)

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
