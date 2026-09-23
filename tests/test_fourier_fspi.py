"""Checks for the Fourier Single-Pixel Imaging family (MaskFourier + ApplicatorFourier).

The repository has no test framework (see CLAUDE.md), so this file is a
self-contained script: run it directly with the project venv.

    .venv/bin/python tests/test_fourier_fspi.py

It also works unchanged under pytest if that is ever added, since every check
is a plain ``test_*`` function using bare ``assert``.
"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from simulation_engine._2_mask_gen.mask_fourier import MaskFourier
from simulation_engine._3_applicator.applicator_fista import ApplicatorFISTA
from simulation_engine._3_applicator.applicator_fourier import ApplicatorFourier
from simulation_engine._3_applicator.applicator_pseudoinverse import (
    ApplicatorPseudoinverse,
)
from simulation_engine._3_applicator.applicator_tv_norm import ApplicatorTV

IMG = 64
N = IMG * IMG
# Large enough that _resolve_counts() clamps to the full half-plane, i.e. every
# independent spectrum coefficient is acquired.
FULL = 4.0


class _Dataset:
    """Minimal stand-in for DatasetABC: the applicators only touch these."""

    def __init__(self, images):
        self.name = "fspi_test"
        self.img_size = images[0].shape[0]
        self.data = images


def _psnr(reference, test):
    mse = float(np.mean((reference - test) ** 2))
    if mse == 0:
        return float("inf")
    peak = float(reference.max() - reference.min())
    return 10.0 * np.log10(peak ** 2 / mse)


def _random_image(size=IMG, seed=0):
    return np.random.default_rng(seed).random((size, size))


# ---------------------------------------------------------------------------
def test_full_spectrum_reconstruction_is_exact():
    """A noiseless full-spectrum acquisition returns the ground truth."""
    image = _random_image()
    dataset = _Dataset([image])
    for phase_steps in (4, 3):
        mask = MaskFourier(img_size=IMG, sampling_ratio=FULL, phase_steps=phase_steps)
        mask.generate_masks()
        recon = ApplicatorFourier(dataset, mask).process_image(0)
        psnr = _psnr(image, recon)
        assert psnr > 40.0, f"phase_steps={phase_steps}: PSNR {psnr:.2f} dB <= 40 dB"
        # The implemented normalisation is unit scale, zero offset.
        scale = np.polyfit(image.ravel(), recon.ravel(), 1)[0]
        assert abs(scale - 1.0) < 1e-6, f"phase_steps={phase_steps}: scale {scale}"
        print(f"  full spectrum, phase_steps={phase_steps}: "
              f"M={mask.num_patterns}, PSNR={psnr:.1f} dB, scale={scale:.6f}")


def test_conjugate_fill_gives_a_real_image():
    """The conjugate-symmetric half-plane fill leaves no imaginary residual."""
    image = _random_image(seed=1)
    dataset = _Dataset([image])
    for phase_steps in (4, 3):
        for ratio in (0.1, FULL):
            mask = MaskFourier(img_size=IMG, sampling_ratio=ratio, phase_steps=phase_steps)
            mask.generate_masks()
            app = ApplicatorFourier(dataset, mask)
            app.process_image(0)
            residual = float(np.abs(np.fft.ifft2(app.last_spectrum).imag).max())
            peak = float(np.abs(np.fft.ifft2(app.last_spectrum).real).max())
            assert residual < 1e-9 * max(peak, 1.0), (
                f"phase_steps={phase_steps}, ratio={ratio}: imaginary residual {residual:.2e}"
            )
    print("  imaginary residual below 1e-9 relative in every case")


def test_pattern_count_matches_sampling_ratio():
    """M tracks round(sampling_ratio * N), snapped to whole phase groups."""
    for phase_steps in (4, 3):
        for ratio in (0.04, 0.08, 0.16, 0.24, 0.32, 0.5):
            mask = MaskFourier(img_size=IMG, sampling_ratio=ratio, phase_steps=phase_steps)
            mask.generate_masks()
            m = mask.num_patterns
            target = round(ratio * N)
            assert m % phase_steps == 0, f"M={m} is not a multiple of {phase_steps}"
            # Exact equality is only possible when round(ratio*N) happens to be
            # a multiple of phase_steps; otherwise M is the nearest such value.
            assert abs(m - target) < phase_steps, (
                f"ratio={ratio}, phase_steps={phase_steps}: M={m} vs target {target}"
            )
            assert mask.masks.shape == (m, IMG, IMG)
            assert mask.masks.dtype == np.float32
            assert mask.masks.min() >= 0.0 and mask.masks.max() <= 1.0
    print("  pattern counts consistent for 6 ratios x 2 phase schemes")


def test_mask_agnostic_solvers_accept_fourier_masks():
    """Pseudoinverse / FISTA / TV run on the rank-deficient Fourier matrix."""
    size = 16
    image = _random_image(size=size, seed=2)
    dataset = _Dataset([image])
    mask = MaskFourier(img_size=size, sampling_ratio=0.25, phase_steps=4)
    mask.generate_masks()
    for cls in (ApplicatorPseudoinverse, ApplicatorFISTA, ApplicatorTV):
        recon = cls(dataset, mask).process_image(0)
        assert recon.shape == (size, size), f"{cls.__name__}: shape {recon.shape}"
        assert np.isfinite(recon).all(), f"{cls.__name__}: non-finite output"
    print("  pseudoinverse, FISTA and TV all produced finite images")


def test_npz_round_trip_preserves_patterns_and_metadata():
    """save_masks / load_masks keep the patterns and the FSPI metadata."""
    mask = MaskFourier(img_size=32, sampling_ratio=0.2, phase_steps=3,
                       spectrum_ordering="square")
    mask.generate_masks()

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "masks_fourier.npz")
        assert mask.save_masks(path=path) is True

        loaded = MaskFourier(img_size=32, sampling_ratio=0.99, phase_steps=4)
        assert loaded.load_masks(path=path) is True

        assert np.array_equal(loaded.masks, mask.masks)
        assert np.array_equal(loaded.freq_indices, mask.freq_indices)
        assert np.allclose(loaded.phases, mask.phases)
        assert np.array_equal(loaded.coeff_ids, mask.coeff_ids)
        assert loaded.num_patterns == mask.num_patterns
        assert loaded.phase_steps == 3
        assert loaded.spectrum_ordering == "square"
        assert abs(loaded.sampling_ratio - 0.2) < 1e-12

        # The reloaded mask must still reconstruct through the native path.
        image = _random_image(size=32, seed=3)
        dataset = _Dataset([image])
        recon = ApplicatorFourier(dataset, loaded).process_image(0)
        assert np.isfinite(recon).all()
    print("  npz round trip preserved patterns, metadata and reconstruction")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for fn in tests:
        print(f"\n{fn.__name__}")
        try:
            fn()
            print("  PASS")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
