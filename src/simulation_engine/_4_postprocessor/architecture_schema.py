"""
Schema definitions for neural network architecture parameters.
Defines configurable parameters for each model type in the post-processor.
"""
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field


@dataclass
class ParamSpec:
    """Specification for a configurable architecture parameter."""
    name: str
    param_type: str  # 'int', 'float', 'bool', 'list_int', 'list_float', 'choice'
    default: Any
    min_val: Optional[Any] = None
    max_val: Optional[Any] = None
    step: Optional[Any] = None
    choices: Optional[List[Any]] = None
    tooltip: str = ""
    display_name: Optional[str] = None  # User-friendly name for UI

    def get_display_name(self) -> str:
        """Return display name or formatted param name."""
        if self.display_name:
            return self.display_name
        # Convert snake_case to Title Case
        return self.name.replace('_', ' ').title()


# Architecture parameter schema for each model
# Keys must match MODEL_REGISTRY keys in postprocessor_nn.py (lowercase)
ARCHITECTURE_SCHEMA: Dict[str, List[ParamSpec]] = {
    "autoencoder": [
        ParamSpec(
            name="img_size",
            param_type="int",
            default=32,
            min_val=8,
            max_val=256,
            step=8,
            tooltip="Input image size (image will be flattened to img_size²)",
            display_name="Image Size"
        ),
    ],

    "dncnn": [
        ParamSpec(
            name="features",
            param_type="int",
            default=128,
            min_val=16,
            max_val=256,
            step=16,
            tooltip="Number of feature channels in hidden layers",
            display_name="Feature Channels"
        ),
        ParamSpec(
            name="depth",
            param_type="int",
            default=5,
            min_val=3,
            max_val=30,
            step=1,
            tooltip="Total number of convolutional layers (paper uses 17)",
            display_name="Network Depth"
        ),
    ],

    "dilatedcnn": [
        ParamSpec(
            name="features",
            param_type="int",
            default=128,
            min_val=16,
            max_val=256,
            step=16,
            tooltip="Number of feature channels",
            display_name="Feature Channels"
        ),
        ParamSpec(
            name="dilation_rates",
            param_type="list_int",
            default=[1, 2, 4, 8],
            tooltip="Dilation rates for each layer (e.g., 1, 2, 4, 8)",
            display_name="Dilation Rates"
        ),
    ],

    "residual_cnn": [
        ParamSpec(
            name="features",
            param_type="int",
            default=64,
            min_val=16,
            max_val=256,
            step=16,
            tooltip="Feature channels per residual block",
            display_name="Feature Channels"
        ),
        ParamSpec(
            name="num_blocks",
            param_type="int",
            default=8,
            min_val=1,
            max_val=20,
            step=1,
            tooltip="Number of residual blocks",
            display_name="Residual Blocks"
        ),
    ],

    "u-net": [
        ParamSpec(
            name="features",
            param_type="list_int",
            default=[8, 16, 32, 64],
            tooltip="Channel widths per encoder level (e.g., 8, 16, 32, 64)",
            display_name="Encoder Channels"
        ),
    ],

    "u-net-residual-attention": [
        # Note: The actual model uses 'widths' but registry uses 'features'
        # We use 'widths' here as that's what the model expects
        ParamSpec(
            name="widths",
            param_type="list_int",
            default=[32, 64, 128, 256],
            tooltip="Channel widths per encoder level (e.g., 32, 64, 128, 256)",
            display_name="Encoder Widths"
        ),
        ParamSpec(
            name="dropout",
            param_type="float",
            default=0.1,
            min_val=0.0,
            max_val=0.5,
            step=0.05,
            tooltip="Dropout probability in residual blocks",
            display_name="Dropout"
        ),
        ParamSpec(
            name="use_se",
            param_type="bool",
            default=True,
            tooltip="Use Squeeze-Excitation (channel attention) blocks",
            display_name="SE Blocks"
        ),
        ParamSpec(
            name="use_attn",
            param_type="bool",
            default=True,
            tooltip="Use attention gates on skip connections",
            display_name="Attention Gates"
        ),
    ],

    "noise2void": [
        ParamSpec(
            name="features",
            param_type="list_int",
            default=[8, 16, 32, 64],
            tooltip="UNet backbone channel widths per level",
            display_name="Backbone Channels"
        ),
    ],

    "mobilenet_denoising": [
        ParamSpec(
            name="features",
            param_type="list_int",
            default=[16, 32, 64, 128],
            tooltip="Channel widths for depthwise separable blocks",
            display_name="Block Channels"
        ),
    ],

    "cgan denoising": [
        ParamSpec(
            name="stem_channels",
            param_type="int",
            default=96,
            min_val=32,
            max_val=256,
            step=16,
            tooltip="Multi-scale stem output channels",
            display_name="Stem Channels"
        ),
        ParamSpec(
            name="denoise_channels",
            param_type="int",
            default=64,
            min_val=32,
            max_val=256,
            step=16,
            tooltip="Feature-domain denoising trunk width",
            display_name="Denoise Channels"
        ),
        ParamSpec(
            name="denoise_depth",
            param_type="int",
            default=8,
            min_val=2,
            max_val=16,
            step=1,
            tooltip="Number of Conv-BN-ReLU layers in denoiser",
            display_name="Denoise Depth"
        ),
        ParamSpec(
            name="hl_blocks",
            param_type="int",
            default=4,
            min_val=1,
            max_val=8,
            step=1,
            tooltip="Number of (CooperativeAttention + ResBlock) stacks",
            display_name="HL Blocks"
        ),
        ParamSpec(
            name="final_act",
            param_type="choice",
            default="sigmoid",
            choices=["sigmoid", "tanh", "none"],
            tooltip="Output activation: sigmoid (0..1), tanh (-1..1), or none",
            display_name="Output Activation"
        ),
    ],
}


def get_schema_for_model(model_name: str) -> List[ParamSpec]:
    """
    Get the parameter schema for a given model.

    Args:
        model_name: Model name (case-insensitive, matches MODEL_REGISTRY keys)

    Returns:
        List of ParamSpec for the model, or empty list if not found
    """
    return ARCHITECTURE_SCHEMA.get(model_name.lower(), [])


def get_default_config(model_name: str) -> Dict[str, Any]:
    """
    Get default configuration values for a model.

    Args:
        model_name: Model name (case-insensitive)

    Returns:
        Dictionary of parameter names to default values
    """
    schema = get_schema_for_model(model_name)
    return {spec.name: spec.default for spec in schema}
