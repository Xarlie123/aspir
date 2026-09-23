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

# Binarisation of the sinusoidal patterns for DMD projection. A DMD switches
# binary frames roughly two orders of magnitude faster than it can render
# grayscale by pulse-width modulation, so binary FSPI trades pattern fidelity
# for frame rate (Zhang et al. 2017).
BINARIZATION_MODES = ("none", "floyd_steinberg", "bayer")

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

    Binary illumination for DMD projection
    --------------------------------------
    ``binarization`` dithers each sinusoid down to two levels so a DMD can
    switch it at binary frame rates instead of rendering grayscale by
    pulse-width modulation. ``dmd_upsampling`` (U) renders the pattern on a
    ``U*H x U*W`` mirror grid first, which is what buys back the grey levels
    the dither gives up: the scene is still modelled at ``img_size``, so each
    image pixel integrates the ``U x U`` mirrors covering it and ends up with
    ``U**2 + 1`` effective levels.

    ``masks`` therefore always holds the *effective* pattern seen by the
    scene, ``(M, H, W)`` in ``[0, 1]`` — quantised rather than continuous when
    binarisation is on. The binary frame actually sent to the DMD is
    recomputed on demand by :meth:`dmd_pattern`, which keeps a full binary
    stack out of memory.

    ``"floyd_steinberg"`` is the error-diffusion dither used by Zhang et al.
    (2017); it preserves the local mean, which is what keeps the demodulation
    unbiased. ``"bayer"`` is an ordered dither: fully vectorised and far
    faster to generate, with structured rather than noise-like artefacts.

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
    binarization : str
        ``"none"`` (default, grayscale patterns), ``"floyd_steinberg"`` or
        ``"bayer"``.
    dmd_upsampling : int
        Mirror-grid magnification U used before dithering (default 1, no
        magnification). Ignored when ``binarization`` is ``"none"``.
    seed : int, optional
        Accepted for signature symmetry with the randomised families
        (``MaskScatter``) and stored, but unused: FSPI sampling is fully
        deterministic given the parameters above, dithering included.
    """

    def __init__(self,
                 img_size=64,
                 sampling_ratio=0.25,
                 phase_steps=4,
                 spectrum_ordering="circular",
                 binarization="none",
                 dmd_upsampling=1,
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

        binarization = str(binarization).strip().lower()
        if binarization not in BINARIZATION_MODES:
            raise ValueError(
                f"binarization must be one of {BINARIZATION_MODES}, got {binarization!r}"
            )

        dmd_upsampling = int(dmd_upsampling)
        if dmd_upsampling < 1:
            raise ValueError(f"dmd_upsampling must be >= 1, got {dmd_upsampling}")

        self.sampling_ratio = sampling_ratio
        self.phase_steps = phase_steps
        self.spectrum_ordering = ordering
        self.binarization = binarization
        self.dmd_upsampling = dmd_upsampling
        self.seed = seed

        # Filled by generate_masks(); the applicator needs them to know which
        # coefficient and phase each projected pattern belongs to.
        self.freq_indices = None   # (M, 2) int32 FFT indices (u, v)
        self.phases = None         # (M,) float64 phase shift in radians
        self.coeff_ids = None      # (M,) int32 coefficient group id

        self.logger.debug(
            "Initializing MaskFourier: img_size=%d, sampling_ratio=%.4f, "
            "phase_steps=%d, ordering=%s, binarization=%s, dmd_upsampling=%d",
            img_size, sampling_ratio, phase_steps, ordering,
            binarization, dmd_upsampling
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
    # ------------------------------------------------------------------
    # Binary illumination (DMD)
    # ------------------------------------------------------------------
    @staticmethod
    def _bayer_matrix(order=3):
        """Recursive Bayer threshold tile, ``2**order`` on a side, in (0, 1)."""
        m = np.array([[0.0, 2.0], [3.0, 1.0]])
        for _ in range(order - 1):
            m = np.block([[4 * m, 4 * m + 2], [4 * m + 3, 4 * m + 1]])
        n = m.shape[0]
        return (m + 0.5) / (n * n)

    def _bayer_binarize(self, values):
        """Ordered dither: compare against a tiled threshold matrix."""
        thr = self._bayer_matrix()
        n = thr.shape[0]
        h, w = values.shape
        tiled = np.tile(thr, ((h + n - 1) // n, (w + n - 1) // n))[:h, :w]
        return (values > tiled).astype(np.uint8)

    @staticmethod
    def _floyd_steinberg_binarize(values):
        """Error-diffusion dither (Floyd-Steinberg), the scheme in the 2017 paper.

        Sequential by construction: the quantisation error of each mirror is
        pushed onto neighbours that have not been decided yet, which is what
        preserves the local mean and keeps the demodulation unbiased. Rows are
        carried as 1-D arrays to keep the inner loop cheap.
        """
        h, w = values.shape
        out = np.empty((h, w), dtype=np.uint8)
        cur = values[0].astype(np.float64, copy=True)
        for y in range(h):
            nxt = values[y + 1].astype(np.float64, copy=True) if y + 1 < h else None
            for x in range(w):
                old = cur[x]
                new = 1.0 if old >= 0.5 else 0.0
                out[y, x] = int(new)
                err = old - new
                if x + 1 < w:
                    cur[x + 1] += err * 0.4375        # 7/16
                if nxt is not None:
                    if x > 0:
                        nxt[x - 1] += err * 0.1875    # 3/16
                    nxt[x] += err * 0.3125            # 5/16
                    if x + 1 < w:
                        nxt[x + 1] += err * 0.0625    # 1/16
            cur = nxt
        return out

    def _binarize(self, values):
        if self.binarization == "bayer":
            return self._bayer_binarize(values)
        return self._floyd_steinberg_binarize(values)

    def _dmd_frame(self, u, v, phi):
        """Binary ``(U*H, U*W)`` frame the DMD projects for one pattern.

        The sinusoid is evaluated on the mirror grid — a mirror at column
        ``X`` sits at image coordinate ``X / U`` — and then dithered.

        The FFT index must be folded to its signed frequency first. On the
        integer pixel grid ``v = W-1`` and ``v = -1`` are the same pattern, but
        on a finer mirror grid they are not: the raw index would render a
        near-Nyquist grating instead of the intended low-frequency fringe, and
        block-averaging it back would return almost nothing at the coefficient
        being measured. The physical system projects the lowest-frequency
        member of the aliasing class, which is the signed one.
        """
        n_up = self.dmd_upsampling
        size = self.img_size
        f_u = u if u <= size // 2 else u - size
        f_v = v if v <= size // 2 else v - size
        hi = size * n_up
        yy, xx = np.meshgrid(np.arange(hi), np.arange(hi))
        theta = 2.0 * np.pi * (f_u * xx + f_v * yy) / float(hi)
        gray = A_OFFSET + B_AMPLITUDE * np.cos(theta + phi)
        return self._binarize(gray)

    def _effective_from_binary(self, binary):
        """Average each ``U x U`` mirror block back onto the image grid."""
        n_up = self.dmd_upsampling
        if n_up == 1:
            return binary.astype(np.float64)
        h = w = self.img_size
        return binary.astype(np.float64).reshape(h, n_up, w, n_up).mean(axis=(1, 3))

    def dmd_pattern(self, index):
        """Recompute the binary frame sent to the DMD for pattern ``index``.

        Returns a ``(U*H, U*W)`` uint8 array of zeros and ones. Recomputed
        rather than stored: a full binary stack at U = 4 runs to tens of
        megabytes and is only needed when driving real hardware.
        """
        if self.binarization == "none":
            raise ValueError(
                "This mask was generated with binarization='none'; its patterns "
                "are grayscale and have no DMD binary frame."
            )
        if self.freq_indices is None:
            raise ValueError("Call generate_masks() before requesting a DMD frame.")
        u, v = (int(x) for x in self.freq_indices[index])
        return self._dmd_frame(u, v, float(self.phases[index]))

    def generate_masks(self, progress_callback=None):
        """Build the ``(M, H, W)`` float32 pattern stack in ``[0, 1]``.

        With ``binarization`` enabled the stored pattern is the *effective*
        one: the dithered binary frame averaged back over each ``U x U``
        mirror block, which is what the scene actually integrates.
        """
        self.logger.info(
            "Generating Fourier masks (ratio=%.4f, phase_steps=%d, ordering=%s, "
            "binarization=%s, dmd_upsampling=%d)",
            self.sampling_ratio, self.phase_steps, self.spectrum_ordering,
            self.binarization, self.dmd_upsampling
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

        binary_mode = self.binarization != "none"

        k = 0
        for c, (u, v) in enumerate(selected):
            theta = 2.0 * np.pi * (u * xx / float(h) + v * yy / float(w))
            for phi in phases:
                if binary_mode:
                    effective = self._effective_from_binary(self._dmd_frame(u, v, phi))
                else:
                    effective = A_OFFSET + B_AMPLITUDE * np.cos(theta + phi)
                masks[k] = effective.astype(np.float32)
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

        levels = (self.dmd_upsampling ** 2 + 1) if binary_mode else None
        self.logger.info(
            "Generated %d Fourier patterns (%d coefficients, effective M/N=%.4f%s)",
            self.num_patterns, n_coeff, self.effective_sampling_ratio,
            f", {self.binarization} dither, {levels} effective levels" if binary_mode else ""
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
                binarization=np.array(self.binarization),
                dmd_upsampling=np.int64(self.dmd_upsampling),
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
                if "binarization" in npz.files:
                    self.binarization = str(npz["binarization"])
                if "dmd_upsampling" in npz.files:
                    self.dmd_upsampling = int(npz["dmd_upsampling"])
            return True
        except Exception as e:
            self.logger.exception("Failed to load Fourier metadata from %s: %s", path, e)
            return False
