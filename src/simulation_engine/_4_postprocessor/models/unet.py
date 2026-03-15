# File: Simulacion/postprocesador/models/unet.py
import torch
import torch.nn as nn

class UNet(nn.Module):
    """
    U-Net: red de convoluciones con conexiones de salto para denoising.
    Arquitectura simétrica de encoder-decoder con skip connections.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1, features: list = [4, 8, 16, 32]):
        super().__init__()
        self.features = features

        # Encoder: bloques de doble conv + pooling
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev_channels = in_channels
        for feature in features:
            self.downs.append(nn.Sequential(
                nn.Conv2d(prev_channels, feature, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(feature),
                nn.ReLU(inplace=True),
                nn.Conv2d(feature, feature, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(feature),
                nn.ReLU(inplace=True)
            ))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            prev_channels = feature

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(prev_channels, prev_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(prev_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(prev_channels * 2, prev_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(prev_channels * 2),
            nn.ReLU(inplace=True)
        )

        # Decoder: transposed conv + doble conv
        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        # Iniciar canales para el decoder
        in_decoder_channels = features[-1] * 2  # output del bottleneck
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(in_decoder_channels, feature, kernel_size=2, stride=2)
            )
            self.up_convs.append(nn.Sequential(
                nn.Conv2d(feature * 2, feature, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(feature),
                nn.ReLU(inplace=True),
                nn.Conv2d(feature, feature, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(feature),
                nn.ReLU(inplace=True)
            ))
            in_decoder_channels = feature

        # Capa final
        self.final_conv = nn.Conv2d(self.features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []
        # Encoder forward
        for down, pool in zip(self.downs, self.pools):
            x = down(x)
            skip_connections.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Decoder forward
        for up, up_conv, skip in zip(self.ups, self.up_convs, skip_connections):
            x = up(x)
            # Ajustar tamaño si difiere de skip
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:])
            # Concatenar skip connection
            x = torch.cat((skip, x), dim=1)
            x = up_conv(x)

        return self.final_conv(x)
