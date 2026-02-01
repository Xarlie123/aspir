# File: Simulacion/postprocesador/modelos/residual_cnn.py
import torch
import torch.nn as nn

class ResidualCNN(nn.Module):
    """
    Residual CNN para denoising:
    - conv inicial
    - N bloques residuales (Conv → BN → ReLU → Conv → BN + skip)
    - conv final para estimar el mapa de ruido
    Output: x - R(x)
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: int = 64,
        num_blocks: int = 8
    ):
        super().__init__()
        # Capa inicial
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

        # Bloques residuales
        blocks = []
        for _ in range(num_blocks):
            blocks.append(nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True),
                nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(features)
            ))
        self.res_blocks = nn.ModuleList(blocks)
        self.relu = nn.ReLU(inplace=True)

        # Capa final (sin activación)
        self.tail = nn.Conv2d(features, out_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        out = self.head(x)
        for block in self.res_blocks:
            residual = block(out)
            out = self.relu(out + residual)  # skip + activación
        noise = self.tail(out)
        return x - noise
