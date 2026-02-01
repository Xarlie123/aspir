import torch
import torch.nn as nn

class DnCNN(nn.Module):
    """
    DnCNN: convolutional denoising network with residual learning.
    Architecture: conv+ReLU + (Conv+BN+ReLU)* (depth-2) + conv
    Output: residual map, se resta de la entrada para obtener la imagen limpia.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1, features: int = 64, depth: int = 17):
        super().__init__()
        layers = []
        # 1st layer: no BN
        layers.append(nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False))
        layers.append(nn.ReLU(inplace=True))
        # middle layers: Conv+BN+ReLU
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        # last layer: no activation
        layers.append(nn.Conv2d(features, out_channels, kernel_size=3, padding=1, bias=False))

        self.dncnn = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W], valores en [0,1] (float32)
        devuelve: x - R(x), donde R(x) es el ruido estimado
        """
        residual = self.dncnn(x)
        return x - residual
