"""Controls for the modelled acquisition: projections per pattern and the grey-level DMD rate.

Shared by Batch Test, Re-measure and Single Test -> Timing. The model itself is
``simulation_engine/_5_analyzer/acquisition.py``.
"""
from __future__ import annotations

from PySide6.QtWidgets import QComboBox, QDoubleSpinBox

from simulation_engine._5_analyzer.acquisition import SCHEME_LABELS, SCHEMES, allowed_schemes

PROJECTION_TIP = (
    "Physical projections per stored pattern, for the modelled acquisition time.\n"
    "Family default: differential (P\u207a and P\u207b, 2 per pattern) for bipolar Hadamard\n"
    "patterns (natural, Walsh-Paley, Cake-Cutting, Cal-Sal), since a DMD cannot project -1;\n"
    "one per pattern for binary patterns (Scatter, Sweep, binary FSPI) and for grey-level\n"
    "FSPI, whose count already includes every phase step.\n"
    "P\u207a + one all-on frame (M + 1) gives the same values only without noise.\n"
    "One per pattern on Hadamard is the count of ASPIR 1.2 and earlier.\n"
    "The sampling ratio M/N does not change; only the frame count does."
)
GREY_RATE_TIP = (
    "DMD rate for grey-level patterns (grey-level FSPI). A DMD renders grey levels by\n"
    "pulse-width modulation, far below its binary rate. Default: the binary sampling rate,\n"
    "which is optimistic for grey-level patterns."
)


def make_projection_combo() -> QComboBox:
    """Family default plus the three schemes; item data is the scheme (None = default)."""
    combo = QComboBox()
    combo.addItem("Family default", None)
    for scheme in SCHEMES:
        combo.addItem(SCHEME_LABELS[scheme], scheme)
    combo.setToolTip(PROJECTION_TIP)
    return combo


def make_grey_rate_spin() -> QDoubleSpinBox:
    """Grey-level DMD rate in kHz; 0 shows as Default (the binary rate)."""
    spin = QDoubleSpinBox()
    spin.setRange(0.0, 1000.0)
    spin.setDecimals(3)
    spin.setSuffix(" kHz")
    spin.setSpecialValueText("Default")
    spin.setValue(0.0)
    spin.setToolTip(GREY_RATE_TIP)
    return spin


def restrict_projection_combo(combo: QComboBox, mask_type: str, binarization=None) -> None:
    """Enable only the schemes that apply to ``mask_type``; fall back to the default if needed."""
    allowed = allowed_schemes(mask_type, binarization)
    model = combo.model()
    for i in range(combo.count()):
        data = combo.itemData(i)
        model.item(i).setEnabled(data is None or data in allowed)
    if combo.currentData() is not None and combo.currentData() not in allowed:
        combo.setCurrentIndex(0)


def set_projection_combo(combo: QComboBox, scheme) -> None:
    """Select ``scheme`` (None = Family default)."""
    index = combo.findData(scheme)
    combo.setCurrentIndex(index if index >= 0 else 0)
