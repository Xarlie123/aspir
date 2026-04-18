"""
Noise2Void — self-supervised blind-spot denoising (Krull et al., CVPR 2019).

Canonical idea: train a network to predict each pixel from its neighbourhood,
*without ever seeing the pixel itself* (blind-spot). Because the network cannot
observe the pixel it is asked to estimate, it cannot learn the identity mapping
and must rely on spatial context — which removes the zero-mean noise component.

This module implements the **mask-based training scheme** from the original N2V
reference implementation:

  1. For each training sample pick a small fraction of pixels at random.
  2. Replace the value at those positions with the value of a random neighbour
     inside a small window (the "blind-spot replacement").
  3. Run the network on the masked input.
  4. Compute the loss only at the masked positions, against the *original*
     (unmodified) noisy pixel values.

The architectural receptive field is therefore not restricted: we rely on the
masking scheme to prevent identity learning, exactly like the reference
TensorFlow / CSBDeep implementation.

Because N2V is self-supervised, the ``clean`` tensor passed to
:meth:`training_step` is **ignored**; the noisy input is used both as input and
as target.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from simulation_engine._4_postprocessor.models.unet import UNet


class Noise2Void(nn.Module):
    """Self-supervised blind-spot denoiser built on a plain U-Net backbone."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: list = [8, 16, 32, 64],
        mask_ratio: float = 0.01,
        neighbor_radius: int = 5,
    ):
        super().__init__()
        self.unet = UNet(in_channels=in_channels,
                         out_channels=out_channels,
                         features=features)
        self.mask_ratio = float(mask_ratio)         # ≈1 % of pixels per image (paper uses 0.5–2%)
        self.neighbor_radius = int(neighbor_radius)  # random-neighbour window radius in pixels

    # ---------------------------------------------------------------- forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Direct prediction. Used for validation / inference (no masking)."""
        return self.unet(x)

    # ------------------------------------------------------------ self-sup step

    def training_step(
        self,
        noisy: torch.Tensor,
        clean: torch.Tensor,  # intentionally unused — N2V is self-supervised
        criterion: nn.Module,  # noqa: ARG002 — replaced by masked-MSE below
        optimizer: torch.optim.Optimizer,
    ) -> torch.Tensor:
        """
        One self-supervised blind-spot training step.

        Returns the masked-MSE loss on this batch (for progress logging).
        """
        mask = self._blind_spot_mask(noisy.shape, noisy.device)
        blinded = self._replace_with_random_neighbors(noisy, mask)

        prediction = self.unet(blinded)

        # Masked MSE: only on positions the network *could not see*, compared
        # against the untouched noisy value. For zero-mean noise this converges
        # to predicting the clean pixel.
        diff2 = (prediction - noisy) ** 2 * mask
        loss = diff2.sum() / (mask.sum() + 1e-8)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    # -------------------------------------------------------------- internals

    def _blind_spot_mask(self, shape: tuple, device: torch.device) -> torch.Tensor:
        """Binary mask (1 where the pixel is blinded, 0 elsewhere)."""
        return (torch.rand(shape, device=device) < self.mask_ratio).float()

    def _replace_with_random_neighbors(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Replace every masked pixel with the value of a random neighbour inside
        a (2r+1)×(2r+1) window. Non-masked pixels are kept untouched.
        """
        b, c, h, w = x.shape
        r = self.neighbor_radius
        offs_y = torch.randint(-r, r + 1, (b, c, h, w), device=x.device)
        offs_x = torch.randint(-r, r + 1, (b, c, h, w), device=x.device)

        y_idx = torch.arange(h, device=x.device).view(1, 1, h, 1).expand(b, c, h, w)
        x_idx = torch.arange(w, device=x.device).view(1, 1, 1, w).expand(b, c, h, w)
        src_y = (y_idx + offs_y).clamp_(0, h - 1)
        src_x = (x_idx + offs_x).clamp_(0, w - 1)

        b_idx = torch.arange(b, device=x.device).view(b, 1, 1, 1).expand(b, c, h, w)
        c_idx = torch.arange(c, device=x.device).view(1, c, 1, 1).expand(b, c, h, w)
        neighbour = x[b_idx, c_idx, src_y, src_x]

        return torch.where(mask > 0.5, neighbour, x)
