"""Fourier Single-Pixel Imaging (FSPI) mask family.

Reference
---------
Zhang, Ma & Zhong, "Single-pixel imaging by means of Fourier spectrum
acquisition", Nature Communications 6, 6225 (2015) — four-step phase shifting.
Zhang et al., "Fast Fourier single-pixel imaging via binary illumination",
Scientific Reports 7, 12029 (2017) — three-step phase shifting.
"""

import numpy as np

from simulation_engine._2_mask_gen.mask import MaskABC

# Phase shifts per scheme, in radians. Order matters: the applicator's
# reconstruction formulas index these positions.
PHASE_SETS = {
    3: (0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0),
    4: (0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0),
}

SPECTRUM_ORDERINGS = ("circular", "square")

# Pattern offset and amplitude: P = A_OFFSET + B_AMPLITUDE * cos(...), chosen so
# patterns live in [0, 1] (grayscale, non-negative — projectable without a
# complementary P+/P- split).
A_OFFSET = 0.5
B_AMPLITUDE = 0.5


class MaskFourier(MaskABC):
    """Sinusoidal Fourier patterns with phase shifting (FSPI).

    For every sampled spatial frequency ``(fx, fy)`` and phase shift ``phi``
    the illumination pattern is::

        P(x, y) = a + b * cos(2*pi*(fx*x + fy*y) + phi)

    with ``a = b = 0.5`` so values lie in ``[0, 1]``. Frequencies live on the
    discrete FFT grid of the ``img_size x img_size`` image: ``fx = u / H`` and
    ``fy = v / W`` for integer FFT indices ``(u, v)``.

    Sampling-ratio accounting
    -------------------------
    ``sampling_ratio`` keeps the meaning it has everywhere else in ASPIR:
    ``M / N`` with ``M`` the number of *patterns projected* (one bucket
    measurement each) and ``N = img_size**2`` the number of pixels. Each
    Fourier coefficient costs ``phase_steps`` patterns, so a ratio ``r``
    buys ``r*N/phase_steps`` complex coefficients, which conjugate symmetry
    then expands to ``2*r*N/phase_steps`` spectrum entries. Coefficients are
    *not* counted as patterns.

    Because ``M`` must be a whole number of complete phase groups, the
    requested ``round(sampling_ratio * N)`` is rounded to the nearest multiple
    of ``phase_steps``; the realised ratio is available as
    ``effective_sampling_ratio``.

    Spectrum sampling
    -----------------
    The spectrum of a real image is conjugate-symmetric, so only one
    half-plane of coefficients is acquired and the other half is filled by
    conjugation at reconstruction time. Coefficients are taken
    low-frequency-first, which is what makes FSPI competitive at low sampling
    ratios (the same rationale as the Cake-Cutting Hadamard ordering):

    * ``"circular"`` (default) — ascending Euclidean radius from DC.
    * ``"square"`` — ascending Chebyshev radius, i.e. growing square
      low-pass blocks, scanned row-major inside each ring.

    Differential acquisition
    ------------------------
    Fourier patterns are already non-negative, so the ``P+``/``P-``
    complementary split used for bipolar Hadamard patterns is unnecessary.
    The phase-shift differences (``D_0 - D_pi`` and friends) *are* the
    differential step: they cancel the ``a`` offset and with it the
    background/DC drift of the bucket detector.

    Parameters
    ----------
    img_size : int
        Side of the square image. ``N = img_size**2``.
    sampling_ratio : float
        Target ``M/N``. Values above ``phase_steps * K / N`` (all available
        half-plane coefficients, ``K = (N + 4) / 2`` for even sizes) are
        clamped to a full-spectrum acquisition.
    phase_steps : int
        ``4`` (default) for four-step phase shifting, ``3`` for three-step.
    spectrum_ordering : str
        ``"circular"`` (default) or ``"square"``.
    seed : int, optional
        Accepted for signature symmetry with the randomised families
        (``MaskScatter``) and stored, but unused: FSPI sampling is fully
        deterministic given the parameters above.
    """

    def __init__(self,
                 img_size=64,
                 sampling_ratio=0.25,
                 phase_steps=4,
                 spectrum_ordering="circular",
                 seed=None,
                 logger=None):
        super().__init__(img_size, logger)

        self.validate_size()

        phase_steps = int(phase_steps)
        if phase_steps not in PHASE_SETS:
            raise ValueError(
                f"phase_steps must be one of {sorted(PHASE_SETS)}, got {phase_steps}"
            )

        ordering = str(spectrum_ordering).strip().lower()
        if ordering not in SPECTRUM_ORDERINGS:
            raise ValueError(
                f"spectrum_ordering must be one of {SPECTRUM_ORDERINGS}, got {spectrum_ordering!r}"
            )

        sampling_ratio = float(sampling_ratio)
        if sampling_ratio <= 0:
            raise ValueError(f"sampling_ratio must be > 0, got {sampling_ratio}")

        self.sampling_ratio = sampling_ratio
        self.phase_steps = phase_steps
        self.spectrum_ordering = ordering
        self.seed = seed

        # Filled by generate_masks(); the applicator needs them to know which
        # coefficient and phase each projected pattern belongs to.
        self.freq_indices = None   # (M, 2) int32 FFT indices (u, v)
        self.phases = None         # (M,) float64 phase shift in radians
        self.coeff_ids = None      # (M,) int32 coefficient group id

        self.logger.debug(
            "Initializing MaskFourier: img_size=%d, sampling_ratio=%.4f, "
            "phase_steps=%d, ordering=%s",
            img_size, sampling_ratio, phase_steps, ordering
        )

    # ------------------------------------------------------------------
    # Spectrum sampling
    # ------------------------------------------------------------------
    def _half_plane_representatives(self):
        """Return one FFT index per conjugate-symmetric pair.

        ``F[u, v] = conj(F[-u, -v])`` for a real image, so only one member of
        each pair carries information. Self-conjugate points (``(0,0)`` and,
        for even sizes, the Nyquist corners) are their own partner and are
        purely real. For an even ``H = W = n`` this yields ``(n**2 + 4) / 2``
        representatives.
        """
        h = w = self.img_size
        seen = set()
        reps = []
        for u in range(h):
            for v in range(w):
                partner = ((-u) % h, (-v) % w)
                key = min((u, v), partner)
                if key not in seen:
                    seen.add(key)
                    reps.append(key)
        return np.asarray(reps, dtype=np.int64)

    def _centered(self, reps):
        """Map FFT indices to signed frequencies in ``[-n/2, n/2)``."""
        h = w = self.img_size
        fu = np.where(reps[:, 0] <= h // 2, reps[:, 0], reps[:, 0] - h)
        fv = np.where(reps[:, 1] <= w // 2, reps[:, 1], reps[:, 1] - w)
        return fu, fv

    def _order_representatives(self, reps):
        """Sort half-plane coefficients low-frequency first."""
        fu, fv = self._centered(reps)
        if self.spectrum_ordering == "circular":
            primary = fu.astype(np.float64) ** 2 + fv.astype(np.float64) ** 2
        else:  # "square": growing square low-pass blocks
            primary = np.maximum(np.abs(fu), np.abs(fv)).astype(np.float64)
        # Deterministic tie-break: row-major within a ring.
        order = np.lexsort((reps[:, 1], reps[:, 0], np.abs(fv), np.abs(fu), primary))
        return reps[order]

    def n_available_coefficients(self):
        """Number of independent (half-plane) spectrum coefficients."""
        n = self.img_size
        return (n * n + 4) // 2 if n % 2 == 0 else (n * n + 1) // 2

    def _resolve_counts(self):
        """Map ``sampling_ratio`` to a whole number of phase groups."""
        n_pixels = self.img_size ** 2
        n_coeff = round(self.sampling_ratio * n_pixels / self.phase_steps)
        n_coeff = max(1, min(n_coeff, self.n_available_coefficients()))
        return n_coeff, n_coeff * self.phase_steps

    @property
    def effective_sampling_ratio(self):
        """Realised ``M/N`` after rounding to complete phase groups."""
        _, n_patterns = self._resolve_counts()
        return n_patterns / float(self.img_size ** 2)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def generate_masks(self, progress_callback=None):
        """Build the ``(M, H, W)`` float32 pattern stack in ``[0, 1]``."""
        self.logger.info(
            "Generating Fourier masks (ratio=%.4f, phase_steps=%d, ordering=%s)",
            self.sampling_ratio, self.phase_steps, self.spectrum_ordering
        )
        self.validate_size()

        h = w = self.img_size
        n_coeff, n_patterns = self._resolve_counts()

        reps = self._order_representatives(self._half_plane_representatives())
        selected = reps[:n_coeff]

        phases = PHASE_SETS[self.phase_steps]
        yy, xx = np.meshgrid(np.arange(w), np.arange(h))  # xx: rows, yy: cols

        masks = np.empty((n_patterns, h, w), dtype=np.float32)
        freq_indices = np.empty((n_patterns, 2), dtype=np.int32)
        phase_arr = np.empty(n_patterns, dtype=np.float64)
        coeff_ids = np.empty(n_patterns, dtype=np.int32)

        k = 0
        for c, (u, v) in enumerate(selected):
            theta = 2.0 * np.pi * (u * xx / float(h) + v * yy / float(w))
            for phi in phases:
                masks[k] = (A_OFFSET + B_AMPLITUDE * np.cos(theta + phi)).astype(np.float32)
                freq_indices[k] = (u, v)
                phase_arr[k] = phi
                coeff_ids[k] = c
                k += 1
            if progress_callback:
                progress_callback(c + 1, n_coeff)

        self.masks = masks
        self.freq_indices = freq_indices
        self.phases = phase_arr
        self.coeff_ids = coeff_ids
        self.num_patterns = int(masks.shape[0])

        self.logger.info(
            "Generated %d Fourier patterns (%d coefficients, effective M/N=%.4f)",
            self.num_patterns, n_coeff, self.effective_sampling_ratio
        )

    # ------------------------------------------------------------------
    # Persistence — same .npz path as the other families, plus the metadata
    # the inverse-FFT reconstruction needs.
    # ------------------------------------------------------------------
    def save_masks(self, path: str | None = None, compress: bool = True) -> bool:
        if self.masks is None:
            self.logger.error("No masks to save: 'self.masks' is None.")
            return False
        if path is None:
            path = f"masks_{self.mask_type}_{self.img_size}.npz"
        if (self.masks.ndim != 3
                or self.masks.shape[1] != self.img_size
                or self.masks.shape[2] != self.img_size):
            self.logger.error(
                "Masks must have shape (N, %d, %d). Got %s",
                self.img_size, self.img_size, self.masks.shape
            )
            return False

        try:
            self.num_patterns = int(self.masks.shape[0])
            saver = np.savez_compressed if compress else np.savez
            saver(
                path,
                masks=self.masks,
                img_size=np.int64(self.img_size),
                num_patterns=np.int64(self.num_patterns),
                mask_type=np.array(self.mask_type),
                freq_indices=self.freq_indices,
                phases=self.phases,
                coeff_ids=self.coeff_ids,
                phase_steps=np.int64(self.phase_steps),
                spectrum_ordering=np.array(self.spectrum_ordering),
                sampling_ratio=np.float64(self.sampling_ratio),
            )
            self.logger.info("Saved %d Fourier masks to %s", self.num_patterns, path)
            return True
        except Exception as e:
            self.logger.exception("Failed to save masks: %s", e)
            return False

    def load_masks(self, path: str | None = None, progress_callback=None,
                   mmap_mode: str | None = None) -> bool:
        if not super().load_masks(path=path, progress_callback=progress_callback,
                                  mmap_mode=mmap_mode):
            return False
        if path is None:
            path = f"masks_{self.mask_type}_{self.img_size}.npz"
        try:
            with np.load(path, allow_pickle=False, mmap_mode=mmap_mode) as npz:
                missing = [k for k in ("freq_indices", "phases", "coeff_ids")
                           if k not in npz.files]
                if missing:
                    self.logger.warning(
                        "NPZ %s lacks Fourier metadata %s; the native inverse-FFT "
                        "reconstruction will not be available.", path, missing
                    )
                    return True
                self.freq_indices = npz["freq_indices"]
                self.phases = npz["phases"]
                self.coeff_ids = npz["coeff_ids"]
                if "phase_steps" in npz.files:
                    self.phase_steps = int(npz["phase_steps"])
                if "spectrum_ordering" in npz.files:
                    self.spectrum_ordering = str(npz["spectrum_ordering"])
                if "sampling_ratio" in npz.files:
                    self.sampling_ratio = float(npz["sampling_ratio"])
            return True
        except Exception as e:
            self.logger.exception("Failed to load Fourier metadata from %s: %s", path, e)
            return False
