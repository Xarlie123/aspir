"""Device and precision resolution for the reconstruction stage."""

import logging
import os

import torch

DEVICE_CHOICES = ("auto", "cpu", "cuda")
DTYPE_CHOICES = ("float32", "float64")

_log = logging.getLogger("ReconstructionDevice")
_auto_logged = False
_fallback_logged = False


def resolve_device(device="auto", logger=None) -> torch.device:
    """Map ``"auto" | "cpu" | "cuda"`` to a ``torch.device``.

    ``"auto"`` picks CUDA when available and CPU otherwise, and logs the
    choice once per process. ``"cuda"`` on a machine without CUDA falls back to
    CPU with a single warning instead of failing: a batch config written on a
    GPU host must still run on a CPU-only one, and the device actually used is
    recorded next to every timing result.
    """
    global _auto_logged, _fallback_logged
    log = logger or _log

    if isinstance(device, torch.device):
        return device
    key = "auto" if device is None else str(device).strip().lower()
    if key not in DEVICE_CHOICES:
        raise ValueError(f"device must be one of {DEVICE_CHOICES}, got {device!r}")

    cuda_ok = torch.cuda.is_available()
    if key == "cpu":
        return torch.device("cpu")
    if key == "cuda":
        if cuda_ok:
            return torch.device("cuda")
        if not _fallback_logged:
            log.warning("Reconstruction device 'cuda' requested but CUDA is not "
                        "available; falling back to CPU.")
            _fallback_logged = True
        return torch.device("cpu")

    resolved = torch.device("cuda") if cuda_ok else torch.device("cpu")
    if not _auto_logged:
        log.info("Reconstruction device 'auto' resolved to %s", describe_device(resolved))
        _auto_logged = True
    return resolved


def resolve_dtype(dtype="float32") -> torch.dtype:
    """Map ``"float32" | "float64"`` (or a torch dtype) to a ``torch.dtype``."""
    if isinstance(dtype, torch.dtype):
        return dtype
    key = str(dtype).strip().lower()
    if key not in DTYPE_CHOICES:
        raise ValueError(f"dtype must be one of {DTYPE_CHOICES}, got {dtype!r}")
    return torch.float32 if key == "float32" else torch.float64


def describe_device(device: torch.device) -> str:
    """Human-readable device name, e.g. ``cuda (NVIDIA GeForce GTX 1650)``."""
    if device.type == "cuda":
        try:
            return f"cuda ({torch.cuda.get_device_name(device)})"
        except Exception:
            return "cuda"
    return "cpu"


def available_bytes(device: torch.device) -> int:
    """Free memory on ``device`` in bytes (host RAM for the CPU)."""
    if device.type == "cuda":
        free, _total = torch.cuda.mem_get_info(device)
        return int(free)
    try:
        return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (ValueError, OSError, AttributeError):
        return 8 * 1024 ** 3


def synchronize(device: torch.device):
    """Block until queued kernels finish; a no-op on the CPU."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
