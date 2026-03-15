import torch
import torch.nn as nn
from simulation_engine._4_postprocessor.models.unet import UNet

class Noise2Void(nn.Module):
    """
    Noise2Void: self-supervised blind-spot denoising model.
    Internally utiliza una U-Net; durante el entrenamiento se deben
    enmascarar píxeles de la entrada (blind-spot strategy).
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: list = [64, 128, 256, 512]
    ):
        super().__init__()
        # Constructor U-Net para mapear ruido
        self.unet = UNet(in_channels=in_channels,
                         out_channels=out_channels,
                         features=features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]
        Devuelve: x - R(x), donde R(x) es el ruido estimado por la U-Net.
        En el pipeline de entrenamiento, aplicar máscara aleatoria a x antes
        de pasarlo por la red.
        """
        residual = self.unet(x)
        return x - residual
