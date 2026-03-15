import torch
import torch.nn as nn

class DilatedCNN(nn.Module):
    """
    DilatedCNN: red convolucional para denoising usando dilataciones.
    - Capa inicial conv+ReLU (dilatación=1)
    - Varios bloques Conv+BN+ReLU con diferentes tasas de dilatación
    - Capa final conv para estimar el ruido
    Output: x - R(x)
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: int = 64,
        dilation_rates: list = [1, 2, 4, 8]
    ):
        super().__init__()
        layers = []
        # Capa inicial: conv + ReLU
        layers.append(nn.Conv2d(in_channels, features, kernel_size=3, padding=1, dilation=1, bias=False))
        layers.append(nn.ReLU(inplace=True))
        # Capas con dilatación variable
        for rate in dilation_rates:
            layers.append(nn.Conv2d(features, features, kernel_size=3, padding=rate, dilation=rate, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        # Capa final sin activación
        layers.append(nn.Conv2d(features, out_channels, kernel_size=3, padding=1, bias=False))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        residual = self.net(x)
        return x - residual
