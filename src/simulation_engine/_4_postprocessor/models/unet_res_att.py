# Always comment Python code in English
import torch
import torch.nn as nn

def kaiming_init(module: nn.Module):
    """He initialization for Conv/ConvTranspose; set BN/GN weights to 1, biases to 0."""
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
        if hasattr(module, "weight") and module.weight is not None:
            nn.init.ones_(module.weight)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.zeros_(module.bias)


class SqueezeExcite(nn.Module):
    """Channel-wise attention (SE block)."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(self.avg(x))
        return x * w


class ResBlock(nn.Module):
    """Residual 2×Conv block with GroupNorm + SiLU and optional SE + Dropout."""
    def __init__(self, in_ch: int, out_ch: int, groups: int = 8,
                 se: bool = True, p_drop: float = 0.1):
        super().__init__()
        g1 = min(groups, out_ch)
        g2 = min(groups, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.gn1   = nn.GroupNorm(num_groups=g1, num_channels=out_ch)
        self.act   = nn.SiLU(inplace=True)
        self.drop  = nn.Dropout2d(p_drop) if p_drop > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.gn2   = nn.GroupNorm(num_groups=g2, num_channels=out_ch)
        self.se    = SqueezeExcite(out_ch) if se else nn.Identity()
        self.skip  = (nn.Identity() if in_ch == out_ch
                      else nn.Conv2d(in_ch, out_ch, 1, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.skip(x)
        x = self.conv1(x); x = self.gn1(x); x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x); x = self.gn2(x)
        x = self.se(x)
        x = x + res
        x = self.act(x)
        return x


class AttnGate(nn.Module):
    """
    Attention gate for skip features (Attention U-Net):
    Uses decoder's gating signal g and encoder skip x to compute an attention map.
    """
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=False),
            nn.GroupNorm( min(8, F_int), F_int )
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=False),
            nn.GroupNorm( min(8, F_int), F_int )
        )
        self.psi = nn.Sequential(
            nn.SiLU(inplace=True),
            nn.Conv2d(F_int, 1, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # g: decoder features (coarse); x: encoder skip (same spatial size)
        return x * self.psi(self.W_g(g) + self.W_x(x))


class UNetResAttn(nn.Module):
    """
    A stronger U-Net for 32×32 inputs:
    - Residual blocks with GroupNorm + SiLU
    - SE in each block
    - Attention gates on decoder skips
    - Bilinear upsampling + 1×1 conv
    - Dilated bottleneck for larger receptive field
    """
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 widths=(32, 64, 128, 256),
                 dropout: float = 0.1,
                 use_se: bool = True,
                 use_attn: bool = True):
        super().__init__()
        self.widths = list(widths)
        self.use_attn = use_attn

        # Encoder
        self.enc_blocks = nn.ModuleList()
        self.pools      = nn.ModuleList()
        c_prev = in_channels
        for c in self.widths:
            self.enc_blocks.append(ResBlock(c_prev, c, se=use_se, p_drop=dropout))
            self.pools.append(nn.MaxPool2d(2))
            c_prev = c

        # Bottleneck with dilations (1,2,4)
        c_bot = self.widths[-1]
        self.bottleneck = nn.Sequential(
            nn.Conv2d(c_bot, c_bot*2, 3, padding=1, dilation=1, bias=False),
            nn.GroupNorm(min(8, c_bot*2), c_bot*2),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_bot*2, c_bot*2, 3, padding=2, dilation=2, bias=False),
            nn.GroupNorm(min(8, c_bot*2), c_bot*2),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_bot*2, c_bot*2, 3, padding=4, dilation=4, bias=False),
            nn.GroupNorm(min(8, c_bot*2), c_bot*2),
            nn.SiLU(inplace=True),
        )

        # Decoder
        self.up_blocks = nn.ModuleList()
        self.attn_gates = nn.ModuleList() if use_attn else None
        c_dec_in = c_bot * 2
        for c_skip in reversed(self.widths):
            self.up_blocks.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(c_dec_in, c_skip, 1, bias=False)  # channel reducer after upsampling
            ))
            if use_attn:
                self.attn_gates.append(AttnGate(F_g=c_skip, F_l=c_skip, F_int=max(c_skip // 2, 8)))
            # After concat(skip, up), fuse with a residual block
            # Input channels: c_skip (from skip) + c_skip (from up) = 2*c_skip
            setattr(self, f"dec_fuse_{c_skip}",
                    ResBlock(2 * c_skip, c_skip, se=use_se, p_drop=dropout))
            c_dec_in = c_skip

        # Final projection
        self.out_conv = nn.Conv2d(self.widths[0], out_channels, kernel_size=1)

        # Init
        self.apply(kaiming_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder pathway
        skips = []
        for enc, pool in zip(self.enc_blocks, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder pathway
        skips = list(reversed(skips))
        attn_idx = 0
        for up, c_skip in zip(self.up_blocks, reversed(self.widths)):
            x = up(x)  # upsample + channel reduce
            skip = skips[attn_idx]
            attn_idx += 1

            # Spatial attention gate on the skip connection
            if self.use_attn:
                x = getattr(self, f"_align_{c_skip}", nn.Identity())(x) if hasattr(self, f"_align_{c_skip}") else x
                gate = self.attn_gates[attn_idx - 1]
                skip = gate(g=x, x=skip)

            # Concatenate and fuse
            x = torch.cat([skip, x], dim=1)
            fuse_block: nn.Module = getattr(self, f"dec_fuse_{c_skip}")
            x = fuse_block(x)

        return self.out_conv(x)
