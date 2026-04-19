# -*- coding: utf-8 -*-
# Standalone generator-first cGAN (RCA-GAN-like) for denoising.
# Compatible with PostprocessorNN (expects a single nn.Module with forward(noisy)->clean).
# Paper reference: RCA-GAN (residual backbone + cooperative attention). See class docstring.

import torch
import torch.nn as nn
import torch.nn.functional as F


def _kaiming_init(m: nn.Module):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        if getattr(m, "weight", None) is not None:
            nn.init.ones_(m.weight)
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)


class _ResBlockBN(nn.Module):
    """Residual block: Conv-BN-ReLU-Conv-BN + skip."""
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(ch)

    def forward(self, x):
        r = x
        x = self.conv1(x); x = self.bn1(x); x = F.relu(x, inplace=True)
        x = self.conv2(x); x = self.bn2(x)
        x = F.relu(x + r, inplace=True)
        return x


class _ChannelAttentionBN(nn.Module):
    """
    Channel attention without reduction (as described: rely on BN stats + learnable scaling).
    BN -> 1x1 conv (C->C, no reduction) -> sigmoid -> reweight.
    """
    def __init__(self, ch: int):
        super().__init__()
        self.bn   = nn.BatchNorm2d(ch, affine=True)  # affine learns gamma/beta por canal
        self.conv = nn.Conv2d(ch, ch, kernel_size=1, bias=True)

    def forward(self, x):
        # BN already normalizes per-channel using batch stats; conv learns per-channel mixing
        w = torch.sigmoid(self.conv(self.bn(x)))
        return x * w


class _SpatialAttention(nn.Module):
    """Spatial attention: concat(avg,max over C) -> 7x7 conv -> sigmoid -> reweight."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=True)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        a = torch.cat([avg, mx], dim=1)
        w = torch.sigmoid(self.conv(a))
        return x * w


class _CooperativeAttention(nn.Module):
    """Channel attention (no reduction) + spatial attention in sequence."""
    def __init__(self, ch: int):
        super().__init__()
        self.ca = _ChannelAttentionBN(ch)
        self.sa = _SpatialAttention()

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


class _MultiScaleStem(nn.Module):
    """
    Multi-scale feature extraction (1x1, 3x3, 5x5, 7x7) -> concat -> 1x1 fuse.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        # split out_ch across branches (ensure at least 8 channels per branch)
        bch = max(8, out_ch // 4)
        self.b1 = nn.Conv2d(in_ch, bch, kernel_size=1, padding=0, bias=False)
        self.b2 = nn.Conv2d(in_ch, bch, kernel_size=3, padding=1, bias=False)
        self.b3 = nn.Conv2d(in_ch, bch, kernel_size=5, padding=2, bias=False)
        self.b4 = nn.Conv2d(in_ch, bch, kernel_size=7, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(bch * 4)
        self.fuse = nn.Conv2d(bch * 4, out_ch, kernel_size=1, bias=False)

    def forward(self, x):
        x = torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        x = self.fuse(x)
        x = F.relu(x, inplace=True)
        return x


class _FeatureDomainDenoiser(nn.Module):
    """Stack of Conv-BN-ReLU layers (8 by default) at constant channels."""
    def __init__(self, ch: int, depth: int = 8):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers += [nn.Conv2d(ch, ch, 3, padding=1, bias=False),
                       nn.BatchNorm2d(ch),
                       nn.ReLU(inplace=True)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class _HighLevelExtractor(nn.Module):
    """
    Fuse + (CooperativeAttention -> ResBlock) × N.
    """
    def __init__(self, ch: int, num_blocks: int = 4):
        super().__init__()
        self.fuse = nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False)
        self.bn   = nn.BatchNorm2d(ch)
        blocks = []
        for _ in range(num_blocks):
            blocks += [ _CooperativeAttention(ch), _ResBlockBN(ch) ]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.fuse(x); x = self.bn(x); x = F.relu(x, inplace=True)
        x = self.blocks(x)
        return x


class _DimReduceFuse(nn.Module):
    """
    Five-layer 3x3 Conv stack to fuse and reduce channels to out_ch.
    """
    def __init__(self, in_ch: int, out_ch: int, mid: int = None):
        super().__init__()
        c = in_ch if mid is None else mid
        layers = [
            nn.Conv2d(in_ch, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1, bias=False),     nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1, bias=False),     nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1, bias=False),     nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, out_ch, 3, padding=1, bias=True)
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class cGAN(nn.Module):
    """
    RCA-GAN-style generator-first module for denoising (drop-in with PostprocessorNN).
    - Forward returns only generator output (no discriminator involvement).
    - Works with grayscale (in_channels=1) or RGB (in_channels=3).
    - Designed for 32×32, 64×64, etc.

    Args:
        in_channels:     input channels (default 1)
        out_channels:    output channels (default 1)
        stem_channels:   channels after multi-scale stem
        denoise_channels: working width in the denoising trunk
        denoise_depth:   number of Conv-BN-ReLU layers in feature-domain denoiser (paper: 8)
        hl_blocks:       number of (CoopAttention + ResBlock) stacks (paper: several; default 4)
        final_act:       'sigmoid' (0..1) | 'tanh' (-1..1) | None
    """
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 stem_channels: int = 64,
                 denoise_channels: int = 64,
                 denoise_depth: int = 8,
                 hl_blocks: int = 4,
                 final_act: str = "sigmoid",
                 # Adversarial-training hyperparameters (WGAN-GP)
                 d_lr: float = 1e-4,
                 d_betas: tuple = (0.5, 0.9),
                 lambda_gp: float = 10.0,
                 lambda_sup: float = 100.0,
                 d_steps_per_g: int = 1):
        super().__init__()

        # 1) Multi-scale feature extraction
        self.stem = _MultiScaleStem(in_channels, stem_channels)

        # 2) Project to denoising width (if needed)
        self.proj = (nn.Identity() if stem_channels == denoise_channels
                     else nn.Conv2d(stem_channels, denoise_channels, 1, bias=False))

        # 3) Feature-domain denoising (stack)
        self.trunk = _FeatureDomainDenoiser(denoise_channels, depth=denoise_depth)

        # 4) High-level extractor with Cooperative Attention + residuals
        self.hlex = _HighLevelExtractor(denoise_channels, num_blocks=hl_blocks)

        # 5) Dimensionality-reduction fusion to out_channels
        self.head = _DimReduceFuse(denoise_channels, out_channels, mid=denoise_channels)

        # Output activation
        self.final_act = final_act
        self.apply(_kaiming_init)

        # Discriminator — used in the adversarial training_step (WGAN-GP)
        self.discriminator = _RcaDiscriminator(in_channels=out_channels)

        # Learnable global skip to preserve low-frequency content
        self.skip_proj = (nn.Identity() if in_channels == out_channels
                          else nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False))

        # Adversarial-training configuration
        self._d_lr = d_lr
        self._d_betas = d_betas
        self._lambda_gp = lambda_gp
        self._lambda_sup = lambda_sup
        self._d_steps_per_g = max(1, int(d_steps_per_g))
        self._d_optimizer = None  # lazily built on first training_step (after .to(device))
        self._step_counter = 0

    # Always comment Python code in English
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x  # keep input for the global skip connection

        # Feature extraction trunk
        x = self.stem(x)
        x = self.proj(x)
        x = self.trunk(x)
        x = self.hlex(x)
        y = self.head(x)

        # --- Global learnable skip (Identity if C_in==C_out, else 1x1 Conv) ---
        skip = self.skip_proj(x_in)
        y = y + skip

        # Output activation
        if self.final_act == "sigmoid":
            y = torch.sigmoid(y)
        elif self.final_act == "tanh":
            y = torch.tanh(y)
        return y

    # ------------------------------------------------- Adversarial training

    def generator_parameters(self):
        """
        Parameters that belong to the generator only (excludes the discriminator).

        Used by PostprocessorNN to build the "main" optimizer so that its
        ``step()`` never updates the discriminator — the discriminator has
        its own Adam optimizer managed internally by :meth:`training_step`.
        """
        disc_ids = {id(p) for p in self.discriminator.parameters()}
        return [p for p in self.parameters() if id(p) not in disc_ids]

    def _build_d_optimizer_if_needed(self):
        if self._d_optimizer is None:
            self._d_optimizer = torch.optim.Adam(
                self.discriminator.parameters(),
                lr=self._d_lr,
                betas=self._d_betas,
            )

    def training_step(
        self,
        noisy: torch.Tensor,
        clean: torch.Tensor,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> torch.Tensor:
        """
        One WGAN-GP + supervised training step.

          D step:  maximize E[D(clean)] - E[D(G(noisy))] with gradient penalty.
          G step:  minimize λ_sup·supervised(G(noisy), clean) - E[D(G(noisy))].

        The ``optimizer`` passed in is the generator optimizer (owned by
        PostprocessorNN); the discriminator optimizer is owned by the module
        and built lazily.

        Returns the generator loss (for logging).
        """
        self._build_d_optimizer_if_needed()
        self._step_counter += 1

        # ---------- Discriminator step(s) ----------
        for _ in range(self._d_steps_per_g):
            with torch.no_grad():
                fake_det = self(noisy)
            d_real = self.discriminator(clean)
            d_fake = self.discriminator(fake_det)
            gp = self.wgan_gp_gradient_penalty(self.discriminator, clean, fake_det)
            d_loss = self.wgan_gp_critic_loss(d_real, d_fake, gp, lambda_gp=self._lambda_gp)

            self._d_optimizer.zero_grad()
            d_loss.backward()
            self._d_optimizer.step()

        # ---------- Generator step ----------
        fake = self(noisy)
        d_fake_for_g = self.discriminator(fake)
        g_adv = -d_fake_for_g.mean()
        g_sup = criterion(fake, clean)
        g_loss = self._lambda_sup * g_sup + g_adv

        optimizer.zero_grad()
        g_loss.backward()
        optimizer.step()
        return g_loss

    # ----------- WGAN-GP loss helpers (used by training_step, also re-usable) ----

    @staticmethod
    def wgan_gp_critic_loss(d_real: torch.Tensor, d_fake: torch.Tensor, gp: torch.Tensor, lambda_gp: float = 10.0):
        """WGAN-GP critic loss: E[D(fake)] - E[D(real)] + λ*GP"""
        return d_fake.mean() - d_real.mean() + lambda_gp * gp

    @staticmethod
    def wgan_gp_gradient_penalty(critic: nn.Module, real: torch.Tensor, fake: torch.Tensor):
        """Gradient penalty for WGAN-GP."""
        bs = real.size(0)
        eps = torch.rand(bs, 1, 1, 1, device=real.device)
        x_hat = eps * real + (1 - eps) * fake
        x_hat.requires_grad_(True)
        d_hat = critic(x_hat)
        grads = torch.autograd.grad(
            outputs=d_hat, inputs=x_hat,
            grad_outputs=torch.ones_like(d_hat),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        gp = ((grads.view(bs, -1).norm(2, dim=1) - 1.0) ** 2).mean()
        return gp


class _RcaDiscriminator(nn.Module):
    """
    Discriminator per paper: 6 conv (BN+LReLU) with max-pooling between stages + 2 FC.
    Output is a scalar (WGAN-GP friendly).
    *No* se usa en el `forward` del cGAN (compatibilidad con tu pipeline), pero disponible si lo necesitas.
    """
    def __init__(self, in_channels: int = 1, base: int = 32):
        super().__init__()
        c1, c2, c3, c4, c5, c6 = base, base*2, base*2, base*4, base*4, base*8

        def block(ic, oc, k=3, s=1, p=1):
            return nn.Sequential(
                nn.Conv2d(ic, oc, kernel_size=k, stride=s, padding=p, bias=False),
                nn.BatchNorm2d(oc),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.features = nn.Sequential(
            block(in_channels, c1, k=3, s=1, p=1),
            nn.MaxPool2d(2),
            block(c1, c2, k=3, s=1, p=1),
            nn.MaxPool2d(2),
            block(c2, c3, k=5, s=1, p=2),
            nn.MaxPool2d(2),
            block(c3, c4, k=3, s=1, p=1),
            block(c4, c5, k=5, s=1, p=2),
            nn.MaxPool2d(2),
            block(c5, c6, k=3, s=1, p=1),
        )

        # Se aplanará dinámicamente en forward (para soportar 32x32 / 64x64)
        self.fc1 = nn.Linear(c6 * 2 * 2, 1024)  # para 32x32 -> 2x2; para 64x64 será 4x4 (se ajusta abajo)
        self.fc2 = nn.Linear(1024, 1)
        self.apply(_kaiming_init)

    def forward(self, x):
        f = self.features(x)
        b, c, h, w = f.shape
        # Ajuste robusto del primer FC al tamaño espacial actual
        if self.fc1.in_features != c * h * w:
            self.fc1 = nn.Linear(c * h * w, 1024).to(f.device)
        f = f.view(b, -1)
        f = F.leaky_relu(self.fc1(f), 0.2, inplace=True)
        out = self.fc2(f)
        return out
