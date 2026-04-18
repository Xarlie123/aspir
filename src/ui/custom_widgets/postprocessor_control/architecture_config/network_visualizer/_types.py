"""Types and colour scheme used by the network visualizer."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


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
