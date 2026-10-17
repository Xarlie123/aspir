"""Modelled acquisition time: how many physical projections the stored patterns cost.

ASPIR simulates each stored pattern as one noiseless measurement. A digital micromirror device (DMD)
projects binary frames, so the frames an acquisition needs depend on the patterns' values:

* **Binary {0, 1}** (Scatter, Sweep, binary FSPI): one frame per pattern.
* **Bipolar ±1** (Hadamard natural, Walsh–Paley, Cake-Cutting, Cal-Sal): a DMD cannot project −1.
  - ``differential``, the default: project P⁺ and P⁻ and subtract, two frames per pattern. This is what
    real systems do; SPIHIM records 8192 frames for 4096 coefficients.
  - ``reference``: P⁺ plus one all-on frame, M + 1 frames. It gives the same values only without noise.
  - ``per_pattern``: one frame per pattern, the count of ASPIR 1.2 and earlier.
* **Grey levels** (grey-level FSPI): one frame per pattern, already counting every phase step. A DMD
  renders grey levels by pulse-width modulation, far below its binary rate. ``grey_level_rate_khz`` sets
  that rate; unset, the binary rate is used, which is optimistic.

The sampling ratio M / N keeps each family's own meaning (for Hadamard, the number of coefficients);
only the frame count and the acquisition time built on it depend on the scheme. Measured data are not
modelled. Reports from earlier versions have no ``timing_projection_scheme`` and counted one frame per
pattern (:func:`report_scheme`).
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

PER_PATTERN, DIFFERENTIAL, REFERENCE, NOT_MODELLED = "per_pattern", "differential", "reference", "not_modelled"
SCHEMES = (PER_PATTERN, DIFFERENTIAL, REFERENCE)
SCHEME_LABELS = {
    PER_PATTERN: "One per pattern",
    DIFFERENTIAL: "Differential (P⁺ and P⁻)",
    REFERENCE: "P⁺ + one all-on frame (M + 1)",
}
BIPOLAR, BINARY, GREY = "bipolar", "binary", "grey"
ALLOWED = {BIPOLAR: SCHEMES, BINARY: (PER_PATTERN,), GREY: (PER_PATTERN,)}
DEFAULT = {BIPOLAR: DIFFERENTIAL, BINARY: PER_PATTERN, GREY: PER_PATTERN}

# Pattern values of each mask family, for reports whose patterns were not saved
_TYPE_POLARITY = {
    "hadamard": BIPOLAR, "hadamard_natural": BIPOLAR, "hadamard_cake_cutting": BIPOLAR,
    "hadamard_walsh_paley": BIPOLAR, "cal_sal": BIPOLAR, "scatter": BINARY, "sweep": BINARY,
}

_log = logging.getLogger(__name__)


def polarity(masks) -> str:
    """``bipolar`` (any value below 0), ``binary`` (only 0 and 1) or ``grey``."""
    arr = np.asarray(masks)
    if arr.size == 0:
        return BINARY
    if float(arr.min()) < 0:
        return BIPOLAR
    return BINARY if bool(np.all((arr == 0) | (arr == 1))) else GREY


def polarity_of_type(mask_type: str, binarization: Optional[str] = None) -> str:
    """The pattern values a mask family produces, when the patterns themselves are not at hand."""
    if mask_type == "fourier":
        return BINARY if binarization not in (None, "", "none") else GREY
    return _TYPE_POLARITY.get(mask_type, BINARY)


def allowed_schemes(mask_type: str, binarization: Optional[str] = None) -> tuple[str, ...]:
    """Schemes that make sense for a mask family (for the GUI)."""
    return ALLOWED[polarity_of_type(mask_type, binarization)]


def resolve_scheme(pol: str, requested: Optional[str], logger: Optional[logging.Logger] = None,
                   name: str = "") -> str:
    """The scheme to apply: the family default when unset, or when the request does not apply.

    A scheme that does not apply (for example ``differential`` on binary patterns) is not an error: the
    family default is used and a warning is logged, so a hand-written configuration does not stop a batch.
    """
    if requested in (None, "", "default"):
        return DEFAULT[pol]
    if requested not in ALLOWED[pol]:
        (logger or _log).warning("Projection scheme %r does not apply to %s patterns%s; using %r",
                                 requested, pol, f" ({name})" if name else "", DEFAULT[pol])
        return DEFAULT[pol]
    return requested


def projection_count(n_patterns: int, scheme: str) -> int:
    """Physical frames for ``n_patterns`` stored patterns under ``scheme``."""
    n = int(n_patterns)
    if scheme == DIFFERENTIAL:
        return 2 * n
    if scheme == REFERENCE:
        return n + 1
    return n


def acquisition_fields(n_patterns: int, pol: str, rate_khz: float, scheme: Optional[str] = None,
                       grey_level_rate_khz: Optional[float] = None, measured: bool = False,
                       logger: Optional[logging.Logger] = None, name: str = "") -> dict[str, Any]:
    """Report fields of the modelled acquisition (``timing_acquisition_ms`` and how it was counted)."""
    if measured:
        return {"timing_acquisition_ms": 0.0, "timing_num_projections": None,
                "timing_projection_scheme": NOT_MODELLED, "timing_pattern_polarity": pol}
    applied = resolve_scheme(pol, scheme, logger, name)
    frames = projection_count(n_patterns, applied)
    rate = grey_level_rate_khz if (pol == GREY and grey_level_rate_khz) else rate_khz
    out = {
        "timing_acquisition_ms": float(frames / rate) if rate and rate > 0 else 0.0,
        "timing_num_projections": frames,
        "timing_projection_scheme": applied,
        "timing_pattern_polarity": pol,
    }
    if grey_level_rate_khz:
        out["timing_grey_level_rate_khz"] = float(grey_level_rate_khz)
    return out


def acquisition_for_mask(mask, n_patterns: int, rate_khz: float, scheme: Optional[str] = None,
                         grey_level_rate_khz: Optional[float] = None, mask_type: str = "",
                         logger: Optional[logging.Logger] = None, name: str = "") -> dict[str, Any]:
    """:func:`acquisition_fields` for a mask object, with the polarity read from its patterns."""
    from simulation_engine._2_mask_gen.mask_measured import MaskMeasured

    masks = getattr(mask, "masks", None) if mask is not None else None
    if masks is not None and len(masks):
        pol = polarity(masks)
    else:
        pol = polarity_of_type(mask_type, getattr(mask, "binarization", None))
    return acquisition_fields(n_patterns, pol, rate_khz, scheme, grey_level_rate_khz,
                              measured=isinstance(mask, MaskMeasured), logger=logger, name=name)


def report_scheme(result: dict[str, Any]) -> str:
    """The scheme a report entry was counted with; reports before ASPIR 1.3 counted one per pattern."""
    return result.get("timing_projection_scheme") or PER_PATTERN
