import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import inspect
from typing import Dict, Any, Optional
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim


class SSIMLoss(nn.Module):
    """SSIM-based loss (1 - SSIM for minimization)."""
    def __init__(self, window_size: int = 11, data_range: float = 1.0):
        super().__init__()
        self.window_size = window_size
        self.data_range = data_range

    def _gaussian_window(self, size: int, sigma: float = 1.5) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32)
        coords -= size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g.unsqueeze(0).unsqueeze(0)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Handle different input dimensions
        if pred.dim() == 2:
            pred = pred.unsqueeze(0).unsqueeze(0)
            target = target.unsqueeze(0).unsqueeze(0)
        elif pred.dim() == 3:
            pred = pred.unsqueeze(1)
            target = target.unsqueeze(1)

        C1 = (0.01 * self.data_range) ** 2
        C2 = (0.03 * self.data_range) ** 2

        # Create gaussian window
        window_1d = self._gaussian_window(self.window_size).to(pred.device)
        window = window_1d.T @ window_1d
        window = window.expand(pred.size(1), 1, self.window_size, self.window_size)

        mu1 = nn.functional.conv2d(pred, window, padding=self.window_size // 2, groups=pred.size(1))
        mu2 = nn.functional.conv2d(target, window, padding=self.window_size // 2, groups=target.size(1))

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = nn.functional.conv2d(pred ** 2, window, padding=self.window_size // 2, groups=pred.size(1)) - mu1_sq
        sigma2_sq = nn.functional.conv2d(target ** 2, window, padding=self.window_size // 2, groups=target.size(1)) - mu2_sq
        sigma12 = nn.functional.conv2d(pred * target, window, padding=self.window_size // 2, groups=pred.size(1)) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return 1 - ssim_map.mean()


class PSNRLoss(nn.Module):
    """PSNR-based loss (-PSNR for minimization)."""
    def __init__(self, max_val: float = 1.0):
        super().__init__()
        self.max_val = max_val

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = torch.mean((pred - target) ** 2)
        if mse == 0:
            return torch.tensor(0.0, device=pred.device)
        psnr = 10 * torch.log10((self.max_val ** 2) / mse)
        return -psnr  # Negative so minimizing loss maximizes PSNR


class CombinedLoss(nn.Module):
    """Combined loss: weighted sum of MSE, SSIM, and optionally LPIPS."""
    def __init__(self, device: torch.device, mse_weight: float = 0.5,
                 ssim_weight: float = 0.3, lpips_weight: float = 0.2):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.ssim_loss = SSIMLoss()
        self.mse_weight = mse_weight
        self.ssim_weight = ssim_weight
        self.lpips_weight = lpips_weight
        self.device = device

        # Lazy-load LPIPS
        self._lpips_model = None
        if lpips_weight > 0:
            try:
                import lpips
                self._lpips_model = lpips.LPIPS(net='alex').to(device)
                self._lpips_model.eval()
            except ImportError:
                pass

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.mse_weight * self.mse_loss(pred, target)
        loss += self.ssim_weight * self.ssim_loss(pred, target)

        if self._lpips_model is not None and self.lpips_weight > 0:
            # LPIPS expects 3-channel images
            if pred.dim() == 4 and pred.shape[1] == 1:
                pred_3ch = pred.repeat(1, 3, 1, 1)
                target_3ch = target.repeat(1, 3, 1, 1)
            else:
                pred_3ch = pred
                target_3ch = target
            # Normalize to [-1, 1]
            pred_norm = pred_3ch * 2 - 1
            target_norm = target_3ch * 2 - 1
            lpips_val = self._lpips_model(pred_norm, target_norm).mean()
            loss += self.lpips_weight * lpips_val

        return loss


class LPIPSLoss(nn.Module):
    """Wrapper for LPIPS perceptual loss."""
    def __init__(self, lpips_model):
        super().__init__()
        self.lpips_model = lpips_model

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Handle grayscale by repeating to 3 channels
        if pred.dim() == 4 and pred.shape[1] == 1:
            pred_3ch = pred.repeat(1, 3, 1, 1)
            target_3ch = target.repeat(1, 3, 1, 1)
        elif pred.dim() == 3:
            pred_3ch = pred.unsqueeze(1).repeat(1, 3, 1, 1)
            target_3ch = target.unsqueeze(1).repeat(1, 3, 1, 1)
        else:
            pred_3ch = pred
            target_3ch = target

        # Normalize to [-1, 1]
        pred_norm = pred_3ch * 2 - 1
        target_norm = target_3ch * 2 - 1

        return self.lpips_model(pred_norm, target_norm).mean()


from simulation_engine._4_postprocessor.postprocessor import Postprocessor
from simulation_engine._4_postprocessor.models.autoencoder import Autoencoder
from simulation_engine._4_postprocessor.models.dncnn import DnCNN
from simulation_engine._4_postprocessor.models.unet import UNet
from simulation_engine._4_postprocessor.models.unet_res import UNetRes
from simulation_engine._4_postprocessor.models.unet_res_att import UNetResAttn
from simulation_engine._4_postprocessor.models.residual_cnn import ResidualCNN
from simulation_engine._4_postprocessor.models.noise2void import Noise2Void
from simulation_engine._4_postprocessor.models.mobilenet_denoising import MobileNetDenoising
from simulation_engine._4_postprocessor.models.dilated_cnn import DilatedCNN
from simulation_engine._4_postprocessor.models.cgan import cGAN

# Data format constants
DATA_FORMAT_FP32 = "FP32"
DATA_FORMAT_INT8 = "INT8"
DATA_FORMAT_INT4 = "INT4"

# Default seed for reproducibility (can be overridden per dataset)
DEFAULT_SEED = 42

# Registry of models with convolution flag
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "autoencoder": {
        "cls": Autoencoder,
        "defaults": {"img_size": None},
        "conv": False
    },
    "cgan-denoising": {
        "cls": cGAN,
        # Defaults mirror your other conv models; img_size is accepted and ignored internally
        "defaults": {"in_channels": 1, "out_channels": 1, "stem_channels": 96,
                     "denoise_channels": 64, "denoise_depth": 8, "hl_blocks": 4,
                     "final_act": "sigmoid"},
        "conv": True
    },
    "dncnn": {
        "cls": DnCNN,
        # depth=5 is a deliberately shallow default for edge-device deployment;
        # the original paper (Zhang et al. 2017) uses depth=17, features=64. The
        # GUI exposes 'depth' up to 30 so users can reproduce the paper setup.
        "defaults": {"in_channels": 1, "out_channels": 1, "features": 128, "depth": 5},
        "conv": True
    },
    "u-net": {
        "cls": UNet,
        "defaults": {"in_channels": 1, "out_channels": 1, "features": [8, 16, 32, 64]},
        "conv": True
    },
    "u-net-residual": {
        # ResUNet variant: add-skips instead of concat-skips, for FINN HW.
        # Defaults mirror "u-net" so val_psnr is directly comparable.
        "cls": UNetRes,
        "defaults": {"in_channels": 1, "out_channels": 1, "features": [8, 16, 32, 64]},
        "conv": True
    },
    "u-net-residual-attention": {
        "cls": UNetResAttn,
        "defaults": {"in_channels": 1, "out_channels": 1, "features": [8, 16, 32, 64]},
        "conv": True
    },
    "residual-cnn": {
        "cls": ResidualCNN,
        "defaults": {"in_channels": 1, "out_channels": 1, "features": 64, "num_blocks": 8},
        "conv": True
    },
    "noise2void": {
        "cls": Noise2Void,
        "defaults": {"in_channels": 1, "out_channels": 1, "features": [8, 16, 32, 64]},
        "conv": True
    },
    "mobilenet-denoising": {
        "cls": MobileNetDenoising,
        "defaults": {"in_channels": 1, "out_channels": 1, "features": [16, 32, 64, 128]},
        "conv": True
    },
    "dilated-cnn": {
        "cls": DilatedCNN,
        "defaults": {"in_channels": 1, "out_channels": 1, "features": 128, "dilation_rates": [1, 2, 4, 8]},
        "conv": True
    },
}


# Legacy key aliases — keep old experiment configs loadable after the
# kebab-case unification. New code should use the canonical keys above.
_LEGACY_MODEL_ALIASES: Dict[str, str] = {
    "cgan denoising":       "cgan-denoising",
    "residual_cnn":         "residual-cnn",
    "mobilenet_denoising":  "mobilenet-denoising",
    "dilatedcnn":           "dilated-cnn",
}


def resolve_model_name(model_name: str) -> str:
    """Map a legacy model name to its canonical key (pass-through if unknown)."""
    if model_name in MODEL_REGISTRY:
        return model_name
    return _LEGACY_MODEL_ALIASES.get(model_name, model_name)


# ---------------------------------------------------------------------------
# GUI display names
# ---------------------------------------------------------------------------
# The GUI shows user-facing labels that don't match the canonical keys
# (mixed capitalisation, spaces, underscores). Keep them verbatim — changing
# the visible strings would surprise users — and provide a direct translation
# table so call sites don't have to rely on the legacy-alias fallback.
#
# The order is the order used by the model-selection menu.
MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "autoencoder":              "Autoencoder",
    "cgan-denoising":           "cGAN Denoising",
    "dncnn":                    "DnCNN",
    "u-net":                    "U-Net",
    "u-net-residual":           "U-Net-Residual",
    "u-net-residual-attention": "U-Net-Residual-Attention",
    "residual-cnn":             "Residual_CNN",
    "noise2void":               "Noise2Void",
    "mobilenet-denoising":      "MobileNet_Denoising",
    "dilated-cnn":              "DilatedCNN",
}

# Reverse index built once; guards against silent drift if an entry is added
# to one dict but not the other.
_DISPLAY_TO_KEY: Dict[str, str] = {v: k for k, v in MODEL_DISPLAY_NAMES.items()}
assert set(MODEL_DISPLAY_NAMES.keys()) == set(MODEL_REGISTRY.keys()), (
    "MODEL_DISPLAY_NAMES and MODEL_REGISTRY must have matching keys"
)


def display_to_key(display: str) -> str:
    """
    Map a GUI display name to its canonical MODEL_REGISTRY key.

    Accepts display names verbatim (e.g. ``'cGAN Denoising'``,
    ``'Residual_CNN'``) and returns the kebab-case canonical key
    (e.g. ``'cgan-denoising'``, ``'residual-cnn'``).

    Falls through to :func:`resolve_model_name` so raw or legacy strings
    already stored on disk (pre-unification experiment manifests) keep working.
    """
    if display in _DISPLAY_TO_KEY:
        return _DISPLAY_TO_KEY[display]
    return resolve_model_name(display.lower())

class PostprocessorNN(Postprocessor):
    """
    Generic training/inference engine that takes:
      - model_name: key in MODEL_REGISTRY
      - model_overrides: parameters to override defaults
      - loss_function: 'mse', 'lpips', 'psnr', 'ssim', or 'combined'
      - optimizer_name: 'adam', 'adamw', 'sgd', or 'rmsprop'
    Supports different data formats (FP32, INT8, INT4) for embedded system testing.
    """
    def __init__(
        self,
        model_name: str,
        model_overrides: Dict[str, Any],
        dataset: Any,
        applicator: Any,
        batch_size: int = 16,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        loss_function: str = 'mse',
        optimizer_name: str = 'adam',
        use_gpu: bool = True,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        logger: logging.Logger = None,
        reconstructed_data: np.ndarray = None
    ):
        super().__init__()

        # Set up logger: use provided or root ASPIR logger
        if logger is not None:
            self.logger = logger.getChild("PostprocessorNN")
        else:
            self.logger = logging.getLogger("ASPIR.PostprocessorNN")
        self.logger.setLevel(logging.DEBUG)
        self.logger.debug("Initializing PostprocessorNN")

        self.img_size = dataset.img_size
        self.batch_size = batch_size

        # Get data format from dataset (default to FP32)
        self.data_format = getattr(dataset, 'data_format', DATA_FORMAT_FP32)
        self.logger.info("Data format from dataset: %s", self.data_format)

        # Device selection: use GPU only if requested AND available
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.logger.info("Using device: %s (use_gpu=%s, cuda_available=%s)",
                         self.device, use_gpu, torch.cuda.is_available())

        # 1) Prepare data arrays - handle different data formats
        clean_np = np.stack(dataset.data).astype(np.float32)

        # Use pre-reconstructed data if provided, otherwise run reconstruction
        if reconstructed_data is not None:
            self.logger.debug("Using pre-reconstructed data (shape: %s)", reconstructed_data.shape)
            noisy_np = reconstructed_data.astype(np.float32).reshape(-1, self.img_size, self.img_size)
        else:
            self.logger.debug("Running reconstruction via applicator.process_dataset()")
            noisy_df = applicator.process_dataset()
            noisy_np = noisy_df.to_numpy().astype(np.float32).reshape(-1, self.img_size, self.img_size)

        # Example global min-max (or your percentile scaling):
        min_val = float(noisy_np.min())
        max_val = float(noisy_np.max())
        noisy_np = (noisy_np - min_val) / (max_val - min_val + 1e-8)

        # Keep the normalization constants: reproducing a reconstruction outside
        # ASPIR (INT8 deployment) needs the exact lo/span applied here.
        self.norm_lo = min_val
        self.norm_span = (max_val - min_val) + 1e-8

        # Apply format-specific processing
        clean_np, noisy_np = self._apply_data_format_to_arrays(clean_np, noisy_np)

        self.logger.debug(
            "Data prepared: clean shape %s (dtype: %s), noisy shape %s (dtype: %s), normalization range [%.4f, %.4f]",
            clean_np.shape, clean_np.dtype, noisy_np.shape, noisy_np.dtype, min_val, max_val
        )

        # 2) Select model configuration (accept legacy names for old experiments)
        canonical_name = resolve_model_name(model_name)
        if canonical_name != model_name:
            self.logger.info("Resolved legacy model name %r -> %r", model_name, canonical_name)
        entry = MODEL_REGISTRY.get(canonical_name)
        if entry is None:
            msg = f"Unknown model: {model_name}. Options: {list(MODEL_REGISTRY.keys())}"
            self.logger.error(msg)
            raise ValueError(msg)
        model_name = canonical_name
        model_cls = entry["cls"]
        self.is_conv = entry.get("conv", False)
        self.logger.info(
            "Selected model '%s' (conv=%s)", model_name, self.is_conv
        )

        # 3) Create tensors
        if self.is_conv:
            clean_ts = torch.from_numpy(clean_np).unsqueeze(1)
            noisy_ts = torch.from_numpy(noisy_np).unsqueeze(1)
        else:
            clean_ts = torch.from_numpy(clean_np).view(-1, self.img_size * self.img_size)
            noisy_ts = torch.from_numpy(noisy_np).view(-1, self.img_size * self.img_size)

        # 4) Split into train/val/test with seed from dataset for reproducibility
        # Use dataset's seed if available, otherwise use default
        self.dataset_seed = getattr(dataset, 'seed', None) or DEFAULT_SEED
        self.logger.info("Using dataset seed for train/val/test split: %d", self.dataset_seed)

        # Set seeds for reproducibility
        torch.manual_seed(self.dataset_seed)
        np.random.seed(self.dataset_seed)

        full_ds = TensorDataset(noisy_ts, clean_ts)
        total = len(full_ds)

        # Normalize ratios to ensure they sum to 1.0
        ratio_sum = train_ratio + val_ratio + test_ratio
        if ratio_sum != 1.0:
            train_ratio = train_ratio / ratio_sum
            val_ratio = val_ratio / ratio_sum
            test_ratio = test_ratio / ratio_sum

        n_train = int(train_ratio * total)
        n_val = int(val_ratio * total)
        n_test = total - n_train - n_val  # Remainder goes to test to avoid rounding issues

        split_generator = torch.Generator().manual_seed(self.dataset_seed)
        train_ds, val_ds, test_ds = random_split(
            full_ds,
            [n_train, n_val, n_test],
            generator=split_generator
        )
        # Frame indices behind each split. Exported with the calibration set as
        # documentary proof that calibration and evaluation frames are disjoint.
        self.train_indices = list(train_ds.indices)
        self.test_indices = list(test_ds.indices)

        self.loaders = {
            "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            "val":   DataLoader(val_ds,   batch_size=batch_size, shuffle=False),
            "test":  DataLoader(test_ds,  batch_size=batch_size, shuffle=False)
        }
        self.logger.debug(
            "Dataset split: train=%d, val=%d, test=%d (seed=%d)",
            n_train, n_val, n_test, self.dataset_seed
        )

        # 5) Instantiate model with filtered kwargs
        raw_kwargs = {**entry["defaults"], **model_overrides}
        if raw_kwargs.get("img_size") is None:
            raw_kwargs["img_size"] = self.img_size
        sig = inspect.signature(model_cls.__init__)
        valid = set(sig.parameters) - {"self", "*args", "**kwargs"}
        filtered_kwargs = {k: v for k, v in raw_kwargs.items() if k in valid}
        self.model = model_cls(**filtered_kwargs).to(self.device)

        # ----> CALCULATE AND STORE THE NUMBER OF PARAMETERS <----
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info("Total model parameters: %d", self.n_params)

        self.logger.info(
            "Model %s instantiated with args %s", model_cls.__name__, filtered_kwargs
        )

        # 6) Set up loss function based on selection
        self.loss_function_name = loss_function.lower()
        self.criterion = self._create_loss_function(self.loss_function_name)
        self.logger.info("Using loss function: %s", self.loss_function_name)

        # 7) Set up optimizer based on selection
        self.optimizer_name = optimizer_name.lower()
        self.optimizer = self._create_optimizer(self.optimizer_name, lr, weight_decay)
        self.logger.info("Using optimizer: %s (lr=%f, weight_decay=%f)",
                        self.optimizer_name, lr, weight_decay)

        self.trained = False
        self.logger.debug("Optimizer and loss function configured")

    def _create_loss_function(self, loss_name: str) -> nn.Module:
        """Create loss function based on name."""
        if loss_name == 'mse':
            return nn.MSELoss()
        elif loss_name == 'lpips':
            try:
                import lpips
                lpips_model = lpips.LPIPS(net='alex').to(self.device)
                lpips_model.eval()
                return LPIPSLoss(lpips_model)
            except ImportError:
                self.logger.warning("LPIPS not available, falling back to MSE")
                return nn.MSELoss()
        elif loss_name == 'psnr':
            return PSNRLoss()
        elif loss_name == 'ssim':
            return SSIMLoss()
        elif loss_name == 'combined':
            return CombinedLoss(self.device)
        else:
            self.logger.warning("Unknown loss function '%s', using MSE", loss_name)
            return nn.MSELoss()

    def _create_optimizer(self, opt_name: str, lr: float, weight_decay: float) -> optim.Optimizer:
        """Create optimizer based on name.

        If the model exposes ``generator_parameters()`` (e.g. cGAN with an
        internal discriminator), only those parameters are optimized here —
        the model is then responsible for updating the rest inside its own
        ``training_step``.
        """
        if hasattr(self.model, 'generator_parameters'):
            params = self.model.generator_parameters()
            self.logger.debug("Using model.generator_parameters() for main optimizer")
        else:
            params = self.model.parameters()
        if opt_name == 'adam':
            return optim.Adam(params, lr=lr, weight_decay=weight_decay)
        elif opt_name == 'adamw':
            return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        elif opt_name == 'sgd':
            return optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
        elif opt_name == 'rmsprop':
            return optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
        else:
            self.logger.warning("Unknown optimizer '%s', using Adam", opt_name)
            return optim.Adam(params, lr=lr, weight_decay=weight_decay)

    def _apply_data_format_to_arrays(self, clean_np: np.ndarray, noisy_np: np.ndarray) -> tuple:
        """
        Apply data format conversions to numpy arrays based on self.data_format.
        For training, we keep data in float32 but simulate quantization effects.

        Supported formats:
            - FP32: Full 32-bit float precision (for computers)
            - INT8: 8-bit integer quantization with 256 levels (for embedded systems)
            - INT4: 4-bit integer quantization with 16 levels (for FPGA)

        Args:
            clean_np: Clean images array
            noisy_np: Noisy images array

        Returns:
            Tuple of (clean_np, noisy_np) with format applied
        """
        if self.data_format == DATA_FORMAT_FP32:
            # Standard float32 - no changes needed
            return clean_np.astype(np.float32), noisy_np.astype(np.float32)

        elif self.data_format == DATA_FORMAT_INT8:
            # Quantize to 256 levels (8-bit)
            def quantize_int8(arr):
                num_levels = 256
                min_v, max_v = arr.min(), arr.max()
                if max_v - min_v > 0:
                    normalized = (arr - min_v) / (max_v - min_v)
                    quantized = np.round(normalized * (num_levels - 1)) / (num_levels - 1)
                    return (quantized * (max_v - min_v) + min_v).astype(np.float32)
                return arr.astype(np.float32)

            self.logger.debug("Applied INT8 quantization simulation (256 levels)")
            return quantize_int8(clean_np), quantize_int8(noisy_np)

        elif self.data_format == DATA_FORMAT_INT4:
            # Quantize to 16 levels (4-bit) - typical for FPGA deployment
            def quantize_int4(arr):
                num_levels = 16
                min_v, max_v = arr.min(), arr.max()
                if max_v - min_v > 0:
                    normalized = (arr - min_v) / (max_v - min_v)
                    quantized = np.round(normalized * (num_levels - 1)) / (num_levels - 1)
                    return (quantized * (max_v - min_v) + min_v).astype(np.float32)
                return arr.astype(np.float32)

            self.logger.debug("Applied INT4 quantization simulation (16 levels)")
            return quantize_int4(clean_np), quantize_int4(noisy_np)

        else:
            self.logger.warning("Unknown data format '%s', using FP32", self.data_format)
            return clean_np.astype(np.float32), noisy_np.astype(np.float32)

    def get_torch_dtype(self) -> torch.dtype:
        """Get the appropriate PyTorch dtype for the current data format."""
        # All formats use float32 for training - quantization is simulated
        dtype_map = {
            DATA_FORMAT_FP32: torch.float32,
            DATA_FORMAT_INT8: torch.float32,  # INT8 simulated in float32 for training
            DATA_FORMAT_INT4: torch.float32,  # INT4 simulated in float32 for training
        }
        return dtype_map.get(self.data_format, torch.float32)

    def _run_training_batch(self, noisy: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
        """
        One optimizer step on a batch.

        If the model defines ``training_step(noisy, clean, criterion, optimizer)``
        (e.g. Noise2Void for blind-spot self-supervision, or cGAN for WGAN-GP
        alternating updates), that method fully owns the forward / backward /
        step logic and returns the loss it minimized on this batch.

        Otherwise, the default supervised path is used: forward -> criterion ->
        backward -> step.
        """
        custom_step = getattr(self.model, "training_step", None)
        if callable(custom_step):
            return custom_step(noisy, clean, self.criterion, self.optimizer)

        self.optimizer.zero_grad()
        outputs = self.model(noisy)
        loss = self.criterion(outputs, clean)
        loss.backward()
        self.optimizer.step()
        return loss

    def train_with_metrics(self, num_epochs: int, progress_callback=None, metrics_callback=None):
        """
        Train and collect validation & test losses plus PSNR/SSIM/LPIPS, then call metrics_callback.
        """
        self.logger.info("Starting train_with_metrics for %d epochs", num_epochs)
        val_losses = []
        test_losses = []
        val_psnr = []
        val_ssim = []
        val_lpips = []

        for epoch in range(1, num_epochs + 1):
            self.model.train()
            for noisy, clean in self.loaders["train"]:
                noisy, clean = noisy.to(self.device), clean.to(self.device)
                self._run_training_batch(noisy, clean)

            if progress_callback:
                progress_callback(epoch, num_epochs)
            val_loss = self.validate()
            test_loss = self.test_loss()
            psnr, ssim, lpips_val = self._compute_val_metrics()
            val_losses.append(val_loss)
            test_losses.append(test_loss)
            val_psnr.append(psnr)
            val_ssim.append(ssim)
            val_lpips.append(lpips_val)
            self.logger.debug(
                "Epoch %d/%d — val_loss=%.4f, test_loss=%.4f, PSNR=%.2f, SSIM=%.4f, LPIPS=%.4f",
                epoch, num_epochs, val_loss, test_loss, psnr, ssim, lpips_val
            )

        self.trained = True
        if metrics_callback:
            metrics_callback(val_losses, test_losses, val_psnr, val_ssim, val_lpips)
        self.logger.info("Training complete, metrics callback emitted")

    def train(self, num_epochs: int, progress_callback=None):
        """Train model without collecting full metrics."""
        self.logger.info("Starting train for %d epochs", num_epochs)
        for epoch in range(1, num_epochs + 1):
            self.model.train()
            for noisy, clean in self.loaders["train"]:
                noisy, clean = noisy.to(self.device), clean.to(self.device)
                self._run_training_batch(noisy, clean)
            if progress_callback:
                progress_callback(epoch, num_epochs)
            self.logger.debug("Completed epoch %d/%d", epoch, num_epochs)
        self.trained = True
        self.logger.info("Train finished")

    def validate(self):
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for noisy, clean in self.loaders["val"]:
                noisy, clean = noisy.to(self.device), clean.to(self.device)
                outputs = self.model(noisy)
                total_loss += self.criterion(outputs, clean).item()
        avg = total_loss / len(self.loaders["val"])
        self.logger.debug("Validation loss: %.4f", avg)
        return avg

    def test_loss(self) -> float:
        """Evaluate on test set."""
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for noisy, clean in self.loaders["test"]:
                noisy, clean = noisy.to(self.device), clean.to(self.device)
                outputs = self.model(noisy)
                total_loss += self.criterion(outputs, clean).item()
        avg = total_loss / len(self.loaders["test"])
        self.logger.debug("Test loss: %.4f", avg)
        return avg

    def _compute_val_metrics(self) -> tuple:
        """Compute average PSNR, SSIM, and LPIPS on validation set."""
        self.model.eval()
        psnr_values = []
        ssim_values = []
        lpips_values = []

        # Lazy-load LPIPS model to avoid overhead if not used
        if not hasattr(self, '_lpips_model'):
            try:
                import lpips
                self._lpips_model = lpips.LPIPS(net='alex').to(self.device)
                self._lpips_model.eval()
            except ImportError:
                self._lpips_model = None
                self.logger.warning("LPIPS not available - install lpips package")

        with torch.no_grad():
            for noisy, clean in self.loaders["val"]:
                noisy, clean = noisy.to(self.device), clean.to(self.device)
                outputs = self.model(noisy)

                # Convert to numpy for skimage metrics
                outputs_np = outputs.cpu().numpy()
                clean_np = clean.cpu().numpy()

                for i in range(outputs_np.shape[0]):
                    # Remove channel dimension if present (B, C, H, W) -> (H, W)
                    out_img = outputs_np[i].squeeze()
                    clean_img = clean_np[i].squeeze()
                    # Compute PSNR and SSIM
                    psnr = compute_psnr(clean_img, out_img, data_range=clean_img.max() - clean_img.min())
                    ssim = compute_ssim(clean_img, out_img, data_range=clean_img.max() - clean_img.min())
                    psnr_values.append(psnr)
                    ssim_values.append(ssim)

                # Compute LPIPS (batch-wise for efficiency)
                if self._lpips_model is not None:
                    try:
                        # Reshape flat outputs to images for non-conv models
                        if not self.is_conv:
                            out_imgs = outputs.view(-1, 1, self.img_size, self.img_size)
                            clean_imgs = clean.view(-1, 1, self.img_size, self.img_size)
                        elif outputs.dim() == 4 and outputs.shape[1] == 1:
                            out_imgs = outputs
                            clean_imgs = clean
                        else:
                            out_imgs = outputs.unsqueeze(1) if outputs.dim() == 3 else outputs
                            clean_imgs = clean.unsqueeze(1) if clean.dim() == 3 else clean

                        # LPIPS requires minimum 64x64 images - upsample if needed
                        if self.img_size < 64:
                            out_imgs = nn.functional.interpolate(out_imgs, size=(64, 64), mode='bilinear', align_corners=False)
                            clean_imgs = nn.functional.interpolate(clean_imgs, size=(64, 64), mode='bilinear', align_corners=False)

                        # LPIPS expects 3-channel images
                        out_3ch = out_imgs.repeat(1, 3, 1, 1)
                        clean_3ch = clean_imgs.repeat(1, 3, 1, 1)

                        # Normalize to [-1, 1]
                        out_norm = out_3ch * 2 - 1
                        clean_norm = clean_3ch * 2 - 1

                        lpips_batch = self._lpips_model(out_norm, clean_norm)
                        lpips_values.extend(lpips_batch.cpu().numpy().flatten().tolist())
                    except Exception as e:
                        self.logger.warning("LPIPS computation failed: %s", e)

        avg_psnr = float(np.mean(psnr_values)) if psnr_values else 0.0
        avg_ssim = float(np.mean(ssim_values)) if ssim_values else 0.0
        avg_lpips = float(np.mean(lpips_values)) if lpips_values else 0.0
        return avg_psnr, avg_ssim, avg_lpips

    def save_model(self, path: str):
        """Save model weights to disk."""
        torch.save(self.model.state_dict(), path)
        self.logger.info("Model saved to %s", path)

    def load_model(self, path: str):
        """Load model weights and prepare for inference."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.trained = True
        self.logger.info("Model loaded from %s", path)

    def test_dataset(self):
        """Return lists of orig, noisy, recon from test set."""
        self.logger.debug("Starting test_dataset inference on test set")
        orig, recon, denoised = [], [], []
        self.model.eval()
        with torch.no_grad():
            for noisy, clean in self.loaders["test"]:
                noisy, clean = noisy.to(self.device), clean.to(self.device)
                outputs = self.model(noisy)
                if self.is_conv:
                    o = clean.cpu().numpy().squeeze(1)
                    r = noisy.cpu().numpy().squeeze(1)
                    d = outputs.cpu().numpy().squeeze(1)
                else:
                    o = clean.cpu().numpy().reshape(-1, self.img_size, self.img_size)
                    r = noisy.cpu().numpy().reshape(-1, self.img_size, self.img_size)
                    d = outputs.cpu().numpy().reshape(-1, self.img_size, self.img_size)
                orig.extend(list(o))
                recon.extend(list(r))
                denoised.extend(list(d))
        self.logger.info(
            "test_dataset complete: orig=%d, recon=%d, denoised=%d",
            len(orig), len(recon), len(denoised)
        )
        return orig, recon, denoised

    def train_dataset(self, limit: int | None = None):
        """Return lists of orig, recon from the train set (no model inference).

        Post-training INT8 quantization observes activation ranges by pushing a
        few hundred representative inputs through the network. Those inputs must
        come from the train split: calibrating on test frames leaks the
        evaluation distribution into the quantized model and inflates the PSNR
        reported afterwards.

        Only the network inputs are needed, so the model is never run here -
        that makes this pass much cheaper than test_dataset().

        Args:
            limit: Stop after this many frames. None exports the whole split.

        Returns:
            Tuple of (orig, recon) lists of (H, W) arrays. Element i corresponds
            to self.train_indices[i].
        """
        self.logger.debug("Starting train_dataset pass (limit=%s)", limit)

        # Iterate the split in its stored order: the train loader shuffles, which
        # would break the pairing with self.train_indices and would also consume
        # the global RNG that training depends on.
        train_ds = self.loaders["train"].dataset
        loader = DataLoader(
            train_ds,
            batch_size=self.loaders["train"].batch_size,
            shuffle=False,
        )

        orig, recon = [], []
        with torch.no_grad():
            for noisy, clean in loader:
                if self.is_conv:
                    o = clean.numpy().squeeze(1)
                    r = noisy.numpy().squeeze(1)
                else:
                    o = clean.numpy().reshape(-1, self.img_size, self.img_size)
                    r = noisy.numpy().reshape(-1, self.img_size, self.img_size)
                orig.extend(list(o))
                recon.extend(list(r))
                if limit is not None and len(recon) >= limit:
                    break

        if limit is not None:
            orig, recon = orig[:limit], recon[:limit]

        self.logger.info(
            "train_dataset complete: orig=%d, recon=%d", len(orig), len(recon)
        )
        return orig, recon
