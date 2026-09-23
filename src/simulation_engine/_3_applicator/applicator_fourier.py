"""Native Fourier Single-Pixel Imaging reconstruction (inverse FFT)."""

import numpy as np
import pandas as pd

from simulation_engine._3_applicator.applicator import ApplicatorABC

# Bucket values are computed in blocks of this many patterns to bound the
# temporary float64 sensing matrix at a few tens of MB.
_BLOCK = 512

# Pattern amplitude b in P = a + b*cos(...). Declared here rather than imported
# from MaskFourier because .importlinter keeps the pipeline stages _1.._5
# mutually independent; it must stay in step with
# simulation_engine._2_mask_gen.mask_fourier.B_AMPLITUDE.
_B_AMPLITUDE = 0.5


class ApplicatorFourier(ApplicatorABC):
    """Fourier Single-Pixel Imaging with phase-shifting demodulation.

    Each spectrum coefficient is acquired with ``phase_steps`` sinusoidal
    patterns. The bucket values ``D_phi`` are combined into one complex
    Fourier coefficient, the conjugate-symmetric half-plane is filled, the
    unsampled entries are left at zero, and the image is recovered with
    ``numpy.fft.ifft2`` (real part).

    Demodulation
    ------------
    Four-step (``phi = 0, pi/2, pi, 3pi/2``)::

        F_raw = (D_0 - D_pi) + j*(D_pi/2 - D_3pi/2)          = 2*b * F

    Three-step (``phi = 0, 2pi/3, 4pi/3``)::

        F_raw = (2*D_0 - D_2pi/3 - D_4pi/3)
                + j*sqrt(3)*(D_2pi/3 - D_4pi/3)              = 3*b * F

    where ``b`` is the pattern amplitude (``0.5``) and ``F`` the true DFT
    coefficient under NumPy's ``fft2`` sign convention. Dividing by ``2*b``
    (four-step) or ``3*b`` (three-step) makes the recovered coefficient equal
    to ``fft2(image)[u, v]``, so a noiseless full-spectrum acquisition returns
    the ground truth at **unit scale and zero offset** — no rescaling needed
    before the quality metrics.

    The ``a`` offset of the patterns cancels in every difference above: the
    phase-shift subtraction *is* the differential measurement, which is why
    Fourier patterns need no complementary ``P+``/``P-`` projection the way
    bipolar Hadamard patterns do.

    Partial mask ranges
    -------------------
    ``apply_mask_range`` may be handed a slice that cuts a coefficient's
    phase group in half. Incomplete groups cannot be demodulated, so they are
    skipped (and counted in ``last_skipped_groups``).
    """

    # Reconstruction method identifier for reports
    RECONSTRUCTION_METHOD = "Fourier Inverse"

    def __init__(self, dataset, mask):
        super().__init__(dataset, mask)
        self.dataset = dataset
        self.mask = mask
        self.reconstructed_image = None
        self.reconstructed_dataset = None
        self.last_skipped_groups = 0
        self.last_spectrum = None

    # ------------------------------------------------------------------
    def _require_metadata(self):
        """Fetch the pattern -> (coefficient, phase) metadata."""
        masks = getattr(self.mask, "masks", None)
        if masks is None or len(masks) == 0:
            if hasattr(self.mask, "generate_masks"):
                self.mask.generate_masks()
            masks = getattr(self.mask, "masks", None)
        if masks is None or len(masks) == 0:
            raise ValueError(
                "The mask has no available patterns in 'masks'. "
                "Make sure to generate them before using the applicator."
            )

        freq = getattr(self.mask, "freq_indices", None)
        phases = getattr(self.mask, "phases", None)
        coeff_ids = getattr(self.mask, "coeff_ids", None)
        if freq is None or phases is None or coeff_ids is None:
            raise ValueError(
                "ApplicatorFourier needs the Fourier metadata (freq_indices, "
                "phases, coeff_ids) produced by MaskFourier.generate_masks(). "
                "Use a mask-agnostic solver (pseudoinverse / FISTA / TV) for "
                "patterns without it."
            )
        return masks, freq, phases, coeff_ids

    @staticmethod
    def _bucket_values(image_flat, masks):
        """Simulate the single-pixel detector: D_i = <image, P_i>."""
        out = np.empty(len(masks), dtype=np.float64)
        for start in range(0, len(masks), _BLOCK):
            block = np.asarray(masks[start:start + _BLOCK], dtype=np.float64)
            out[start:start + len(block)] = block.reshape(len(block), -1) @ image_flat
        return out

    @staticmethod
    def _demodulate(bucket, phases):
        """Combine one coefficient's bucket values into a complex amplitude.

        ``bucket`` and ``phases`` hold the group's measurements in any order;
        both canonical phase sets are ascending, so sorting by phase recovers
        the (0, pi/2, pi, 3pi/2) / (0, 2pi/3, 4pi/3) positions the formulas
        below index.
        """
        d = np.asarray(bucket, dtype=np.float64)[np.argsort(np.asarray(phases))]
        if len(d) == 4:
            raw = complex(d[0] - d[2], d[1] - d[3])
            scale = 2.0 * _B_AMPLITUDE
        elif len(d) == 3:
            raw = complex(2.0 * d[0] - d[1] - d[2],
                          np.sqrt(3.0) * (d[1] - d[2]))
            scale = 3.0 * _B_AMPLITUDE
        else:
            raise ValueError(f"Unsupported phase group size: {len(d)}")
        return raw / scale

    # ------------------------------------------------------------------
    def apply_mask_range(self, idx_mask_min, idx_mask_max, idx_image):
        """Acquire a range of patterns on one image and inverse-FFT it back."""
        masks, freq, phases, coeff_ids = self._require_metadata()

        image = np.asarray(self.dataset.data[idx_image], dtype=np.float64)
        h, w = image.shape

        idx_max = min(idx_mask_max, len(masks))
        idx_min = max(0, idx_mask_min)
        if idx_max <= idx_min:
            self.reconstructed_image = np.zeros_like(image)
            return self.reconstructed_image

        sel = slice(idx_min, idx_max)
        phase_steps = int(getattr(self.mask, "phase_steps", 4))
        bucket = self._bucket_values(image.ravel(), masks[sel])
        phi = np.asarray(phases[sel], dtype=np.float64)
        cids = np.asarray(coeff_ids[sel])
        freqs = np.asarray(freq[sel])

        spectrum = np.zeros((h, w), dtype=np.complex128)
        skipped = 0
        for cid in np.unique(cids):
            member = cids == cid
            if int(member.sum()) != phase_steps:
                skipped += 1  # phase group cut by the requested range
                continue
            group_phases = phi[member]
            if len(np.unique(np.round(group_phases, 9))) != phase_steps:
                skipped += 1
                continue

            value = self._demodulate(bucket[member], group_phases)
            u, v = (int(x) for x in freqs[member][0])
            pu, pv = (-u) % h, (-v) % w
            if (pu, pv) == (u, v):
                # Self-conjugate entry (DC and the Nyquist corners): real.
                spectrum[u, v] = value.real
            else:
                spectrum[u, v] = value
                spectrum[pu, pv] = np.conj(value)

        self.last_skipped_groups = skipped
        logger = getattr(self.mask, "logger", None)
        if skipped and logger is not None:
            logger.debug(
                "Fourier reconstruction skipped %d incomplete phase group(s)", skipped
            )

        # Kept for inspection: the conjugate-symmetric spectrum that produced
        # this image (used by the tests and handy for spectrum figures).
        self.last_spectrum = spectrum

        reconstructed = np.fft.ifft2(spectrum).real
        self.reconstructed_image = reconstructed
        return reconstructed

    def process_image(self, idx):
        """Reconstruct a single image with the full pattern set."""
        masks, _, _, _ = self._require_metadata()
        return self.apply_mask_range(0, len(masks), idx)

    def process_dataset(self, idx_mask_min=0, idx_mask_max=None):
        """Reconstruct every image; one flattened reconstruction per row."""
        masks, _, _, _ = self._require_metadata()
        if idx_mask_max is None:
            idx_mask_max = len(masks)

        reconstructed_images = []
        for idx in range(len(self.dataset.data)):
            rec = self.apply_mask_range(idx_mask_min, idx_mask_max, idx)
            reconstructed_images.append(rec.flatten())

        df = pd.DataFrame(reconstructed_images)
        self.reconstructed_dataset = df
        return df
