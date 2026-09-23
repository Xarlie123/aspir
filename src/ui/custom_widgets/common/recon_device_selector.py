"""Reconstruction-device dropdown shared by every mask-control widget."""

from PySide6.QtWidgets import QComboBox

# Display label -> device key understood by the reconstruction backend.
DEVICE_ITEMS = (("Auto", "auto"), ("CPU", "cpu"), ("CUDA", "cuda"))


def create_recon_device_combo(parent=None) -> QComboBox:
    """Build the Auto / CPU / CUDA selector for the reconstruction step.

    "CUDA" is disabled when no CUDA device is present; "Auto" then resolves to
    the CPU, so a CPU-only install keeps working with the default.
    """
    combo = QComboBox(parent)
    for text, key in DEVICE_ITEMS:
        combo.addItem(text, key)
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_ok else None
    except Exception:
        cuda_ok, gpu_name = False, None
    if not cuda_ok:
        item = combo.model().item(combo.findData("cuda"))
        if item is not None:
            item.setEnabled(False)
    combo.setToolTip(
        "Where the reconstruction runs.\n"
        "Auto = CUDA when available, otherwise CPU.\n"
        + (f"CUDA device: {gpu_name}" if cuda_ok else "No CUDA device detected.")
    )
    return combo


def get_recon_device(combo) -> str:
    """Device key of the current selection (``"auto"`` if unset)."""
    if combo is None:
        return "auto"
    return combo.currentData() or "auto"


def set_recon_device(combo, key):
    """Select ``key`` ("auto" / "cpu" / "cuda") if present."""
    if combo is None:
        return
    idx = combo.findData(str(key or "auto").lower())
    if idx >= 0:
        combo.setCurrentIndex(idx)
