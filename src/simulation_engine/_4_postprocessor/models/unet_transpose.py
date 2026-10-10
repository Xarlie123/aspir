import torch.nn as nn

from simulation_engine._4_postprocessor.models.unet import UNet


class UNetTranspose(UNet):
    """U-Net with the decoder ASPIR used before 2026-05-16 (commit 4eb6f44).

    Identical to :class:`UNet` except that each decoder step upsamples with a
    ``ConvTranspose2d(kernel_size=2, stride=2)`` instead of ``Upsample`` +
    ``Conv2d``. It exists only so checkpoints saved with that decoder (for
    example the Use Case 2 models) still load, for re-measurement and for
    inference; new models are trained with :class:`UNet`.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, features: list = [4, 8, 16, 32]):  # noqa: B006
        super().__init__(in_channels, out_channels, features)
        self.ups = nn.ModuleList()
        prev = features[-1] * 2
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(prev, feature, kernel_size=2, stride=2))
            prev = feature


def is_transpose_unet_state(state: dict) -> bool:
    """True if a U-Net ``state_dict`` was saved with the ConvTranspose decoder."""
    w = state.get("ups.0.weight")
    return w is not None and getattr(w, "ndim", 0) == 4 and "ups.0.1.weight" not in state
