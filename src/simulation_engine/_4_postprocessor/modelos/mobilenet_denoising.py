# File: Simulacion/postprocesador/modelos/mobilenet_denoising.py
import torch
import torch.nn as nn


class MobileNetDenoising(nn.Module):
    """
    MobileNet ligero para denoising usando convoluciones separables.
    - Convolución separable seguida de convolución punto a punto para cambiar canales.
    Output: x - R(x), donde R(x) es el ruido estimado.
    """

    def __init__(
            self,
            in_channels: int = 1,
            out_channels: int = 1,
            features: list = [32, 64, 128]
    ):
        super().__init__()
        # Capa inicial: conv estándar
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True)
        )

        # Bloques separables: depthwise + pointwise, usando progresión de canales
        blocks = []
        prev_channels = features[0]
        for feature in features[1:]:
            # Depthwise conv sobre prev_channels
            blocks.append(nn.Sequential(
                nn.Conv2d(prev_channels, prev_channels, kernel_size=3, padding=1, groups=prev_channels, bias=False),
                nn.BatchNorm2d(prev_channels),
                nn.ReLU(inplace=True),
                # Pointwise conv para mapear a 'feature' canales
                nn.Conv2d(prev_channels, feature, kernel_size=1, bias=False),
                nn.BatchNorm2d(feature),
                nn.ReLU(inplace=True)
            ))
            prev_channels = feature

        self.blocks = nn.Sequential(*blocks)
        # Capa final para estimar ruido
        self.tail = nn.Conv2d(prev_channels, out_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]
        devuelve: x - R(x)
        """
        out = self.head(x)  # [B, f0, H, W]
        out = self.blocks(out)  # [B, last_feature, H, W]
        residual = self.tail(out)  # [B, out_channels, H, W]
        return x - residual
