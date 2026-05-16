import torch
import torch.nn as nn


class UNetRes(nn.Module):
    """
    ResUNet variant of UNet for FINN deployment.

    Replaces channel-concatenation skip connections with residual additions
    (Zhang et al. 2018, "Road Extraction by Deep Residual U-Net"). Motivation:
    FINN v0.10.1 cannot map channel-axis Concat to HW (issue #329, open since
    2023); add-based skips are the only skip pattern with native HW support
    in FINN (ResNet-50 is the reference). The rest of the deployment targets
    (NEON / DPU / HLS4ML / Jetson) keep using the vanilla UNet.

    Compared to UNet:
        - up_convs first Conv2d: feature*2 -> feature input channels
        - forward: x = skip + x  (was torch.cat((skip, x), dim=1))

    Everything else — encoder, bottleneck, Upsample+Conv decoder, BatchNorm,
    ReLU, MaxPool, final_conv — matches UNet exactly.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, features: list = [4, 8, 16, 32]):
        super().__init__()
        self.features = features

        # Encoder: doble conv + pooling (idéntico a UNet)
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

        # Bottleneck (idéntico a UNet)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(prev_channels, prev_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(prev_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(prev_channels * 2, prev_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(prev_channels * 2),
            nn.ReLU(inplace=True)
        )

        # Decoder: Upsample(×2 nearest) + Conv2d(3×3) + doble conv.
        # Diferencia clave vs UNet: el primer conv de up_convs recibe `feature`
        # canales (suma), no `feature * 2` (concat).
        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        in_decoder_channels = features[-1] * 2  # output del bottleneck
        for feature in reversed(features):
            self.ups.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_decoder_channels, feature, kernel_size=3, padding=1, bias=True),
            ))
            self.up_convs.append(nn.Sequential(
                nn.Conv2d(feature, feature, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(feature),
                nn.ReLU(inplace=True),
                nn.Conv2d(feature, feature, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(feature),
                nn.ReLU(inplace=True)
            ))
            in_decoder_channels = feature

        # Capa final (idéntica a UNet)
        self.final_conv = nn.Conv2d(self.features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []
        for down, pool in zip(self.downs, self.pools):
            x = down(x)
            skip_connections.append(x)
            x = pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for up, up_conv, skip in zip(self.ups, self.up_convs, skip_connections):
            x = up(x)
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:])
            # Residual skip: add en lugar de concat (FINN-compatible)
            x = skip + x
            x = up_conv(x)

        return self.final_conv(x)
