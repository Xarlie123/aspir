"""Block extraction — analyse PyTorch modules and build semantic block lists."""
from __future__ import annotations

import torch.nn as nn

from ui.custom_widgets.postprocessor_control.architecture_config.network_visualizer._types import (
    BlockType,
    SemanticBlock,
)


def get_output_channels_transposed(module: nn.Module) -> int:
    """Get output channels from ConvTranspose2d."""
    if isinstance(module, nn.ConvTranspose2d):
        return module.out_channels
    for child in module.modules():
        if isinstance(child, nn.ConvTranspose2d):
            return child.out_channels
    return get_output_channels(module)


def get_output_channels(module: nn.Module) -> int:
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


def infer_skip_connections(blocks: list[SemanticBlock]) -> list[tuple[int, int]]:
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


def extract_fc_autoencoder(model: nn.Module, input_size: int) -> tuple[list[SemanticBlock], list[tuple[int, int]]]:
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


def extract_enc_dec_blocks(model: nn.Module, input_size: int) -> tuple[list[SemanticBlock], list[tuple[int, int]]]:
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
        out_ch = get_output_channels(enc)
        level = i + 1

        blocks.append(SemanticBlock(
            name=f"Enc{i+1}",
            block_type=BlockType.CONV_BLOCK,
            in_channels=1 if i == 0 else get_output_channels(enc_blocks[i-1]),
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
            out_channels=get_output_channels(model.bottleneck),
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

        out_ch = get_output_channels(dec)
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


def extract_from_modules(model: nn.Module, input_size: int) -> list[SemanticBlock]:
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

    for _, module in model.named_modules():
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
                name="Pool",
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
                name="Up",
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


def extract_generic_unet(model: nn.Module, input_size: int) -> list[SemanticBlock]:
    """Extract blocks from generic U-Net-like structure."""
    return extract_from_modules(model, input_size)


def extract_unet_blocks(model: nn.Module, input_size: int = 32) -> tuple[list[SemanticBlock], list[tuple[int, int]]]:
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
            out_ch = get_output_channels(down)
            level = i + 1

            blocks.append(SemanticBlock(
                name=f"Enc{i+1}",
                block_type=BlockType.CONV_BLOCK,
                in_channels=1 if i == 0 else get_output_channels(downs[i-1]),
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
                    name="Pool",
                    block_type=BlockType.POOL,
                    in_channels=out_ch,
                    out_channels=out_ch,
                    spatial_size=current_spatial,
                    level=level,
                    is_encoder=True
                ))

        # Bottleneck
        if hasattr(model, 'bottleneck'):
            bot_in = get_output_channels(downs[-1]) if downs else 64
            bot_out = get_output_channels(model.bottleneck)
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
            up_out_ch = get_output_channels_transposed(up)
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
            dec_out_ch = get_output_channels(up_conv)
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
        return extract_enc_dec_blocks(model, input_size)

    elif hasattr(model, 'encoder') and hasattr(model, 'decoder'):
        # Fully connected autoencoder pattern (encoder/decoder Sequential)
        return extract_fc_autoencoder(model, input_size)

    else:
        # Fallback: analyze module structure
        blocks = extract_from_modules(model, input_size)
        skip_connections = infer_skip_connections(blocks)

    return blocks, skip_connections
