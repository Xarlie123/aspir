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


def test_dmd_frames_are_binary_and_upsampled():
    """The frame sent to the DMD is strictly two-level at U x the image size."""
    size = 32
    for mode in ("floyd_steinberg", "bayer"):
        for upsampling in (1, 2, 3):
            mask = MaskFourier(img_size=size, sampling_ratio=0.02, phase_steps=4,
                               binarization=mode, dmd_upsampling=upsampling)
            mask.generate_masks()
            frame = mask.dmd_pattern(len(mask.masks) // 2)
            assert frame.shape == (size * upsampling, size * upsampling), frame.shape
            assert set(np.unique(frame)).issubset({0, 1}), np.unique(frame)
            # The effective pattern the scene integrates has U**2 + 1 levels.
            levels = np.unique(np.round(mask.masks, 6))
            assert len(levels) <= upsampling ** 2 + 1, len(levels)
            assert levels.min() >= 0.0 and levels.max() <= 1.0
    print("  binary frames, correct shape, and <= U^2+1 effective levels")


def test_grayscale_masks_have_no_dmd_frame():
    """Asking a grayscale mask for a DMD frame is an error, not a silent guess."""
    mask = MaskFourier(img_size=32, sampling_ratio=0.02)
    mask.generate_masks()
    try:
        mask.dmd_pattern(0)
    except ValueError:
        print("  grayscale mask rejects dmd_pattern() as expected")
    else:
        raise AssertionError("dmd_pattern() should raise for binarization='none'")


def test_dither_preserves_the_pattern_mean():
    """Error diffusion keeps the local mean, which keeps demodulation unbiased."""
    size = 32
    mask = MaskFourier(img_size=size, sampling_ratio=0.05, phase_steps=4,
                       binarization="floyd_steinberg", dmd_upsampling=2)
    mask.generate_masks()
    yy, xx = np.meshgrid(np.arange(size), np.arange(size))
    for i in range(0, len(mask.masks), 4):
        u, v = (int(t) for t in mask.freq_indices[i])
        f_u = u if u <= size // 2 else u - size
        f_v = v if v <= size // 2 else v - size
        ideal = 0.5 + 0.5 * np.cos(2 * np.pi * (f_u * xx + f_v * yy) / size + mask.phases[i])
        assert abs(mask.masks[i].mean() - ideal.mean()) < 0.01, (
            f"(u,v)=({u},{v}): mean {mask.masks[i].mean():.4f} vs {ideal.mean():.4f}"
        )
    print("  dithered patterns match the ideal mean within 0.01")


def test_negative_frequencies_render_the_signed_fringe():
    """Regression: FFT indices must be folded to signed frequencies on the mirror grid.

    On the integer pixel grid v = W-1 and v = -1 describe the same pattern, but
    on a finer mirror grid they do not. Rendering the raw index produced a
    near-Nyquist grating instead of the intended low-frequency fringe; block
    averaging it returned almost nothing, and those coefficients came back at
    ~3 % of their true value. Checked on the pattern itself so the assertion
    does not depend on any image.
    """
    size = 32
    mask = MaskFourier(img_size=size, sampling_ratio=0.05, phase_steps=4,
                       binarization="floyd_steinberg", dmd_upsampling=3)
    yy, xx = np.meshgrid(np.arange(size), np.arange(size))

    for u, v in [(1, size - 1), (2, size - 2), (1, size - 3), (3, size - 1), (1, 1)]:
        effective = mask._effective_from_binary(mask._dmd_frame(u, v, 0.0))
        f_u = u if u <= size // 2 else u - size
        f_v = v if v <= size // 2 else v - size
        ideal = 0.5 + 0.5 * np.cos(2 * np.pi * (f_u * xx + f_v * yy) / size)

        a = effective - effective.mean()
        b = ideal - ideal.mean()
        corr = float(np.sum(a * b) / np.sqrt(np.sum(a ** 2) * np.sum(b ** 2)))
        assert corr > 0.9, f"(u,v)=({u},{v}) renders at correlation {corr:.3f}"
    print("  signed fringes render at correlation > 0.9, negative indices included")


def test_binary_fspi_reconstructs_close_to_grayscale():
    """Dithering costs quality but must not break the reconstruction."""
    size = 32
    image = _random_image(size=size, seed=8)
    # Smooth the noise so the low-frequency budget can actually represent it.
    kernel = np.outer(*(np.hanning(9),) * 2)
    kernel /= kernel.sum()
    padded = np.pad(image, 4, mode="wrap")
    image = np.array([[np.sum(padded[y:y + 9, x:x + 9] * kernel)
                       for x in range(size)] for y in range(size)])
    dataset = _Dataset([image])

    psnrs = {}
    for mode, upsampling in (("none", 1), ("floyd_steinberg", 1), ("bayer", 2)):
        mask = MaskFourier(img_size=size, sampling_ratio=0.2, phase_steps=4,
                           binarization=mode, dmd_upsampling=upsampling)
        mask.generate_masks()
        recon = ApplicatorFourier(dataset, mask).process_image(0)
        recon = (recon - recon.min()) / (recon.max() - recon.min() + 1e-12)
        ref = (image - image.min()) / (image.max() - image.min() + 1e-12)
        psnrs[mode] = _psnr(ref, recon)

    for mode in ("floyd_steinberg", "bayer"):
        assert psnrs[mode] > 15.0, f"{mode}: {psnrs[mode]:.2f} dB"
        assert psnrs[mode] > psnrs["none"] - 12.0, (
            f"{mode} {psnrs[mode]:.2f} dB is far below grayscale {psnrs['none']:.2f} dB"
        )
    print("  " + ", ".join(f"{k}={v:.1f} dB" for k, v in psnrs.items()))


def test_npz_round_trip_preserves_patterns_and_metadata():
    """save_masks / load_masks keep the patterns and the FSPI metadata."""
    mask = MaskFourier(img_size=32, sampling_ratio=0.2, phase_steps=3,
                       spectrum_ordering="square", binarization="bayer",
                       dmd_upsampling=2)
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
        assert loaded.binarization == "bayer"
        assert loaded.dmd_upsampling == 2
        assert abs(loaded.sampling_ratio - 0.2) < 1e-12

        # The reloaded mask must still reconstruct through the native path.
        image = _random_image(size=32, seed=3)
        dataset = _Dataset([image])
        recon = ApplicatorFourier(dataset, loaded).process_image(0)
        assert np.isfinite(recon).all()
    print("  npz round trip preserved patterns, metadata and reconstruction")


def test_remeasure_rebuilds_the_metadata_of_old_exports():
    """Re-measure reconstructs FSPI tests whose masks.npz holds only the patterns.

    Exports before v1.3 saved the patterns without each one's coefficient and
    phase, so re-measure could not time the Fourier reconstruction. It now
    regenerates the metadata and accepts it only if the saved patterns match.
    """
    import json
    import logging

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])  # noqa: F841
    from ui.custom_widgets.batch_reports.remeasure.remeasure_worker import (
        RemeasureConfig,
        _build_remeasure_pipeline,
        _make_test_config,
        _strip_old_metrics,
    )

    log = logging.getLogger("test")
    images = [_random_image(IMG, seed=s) for s in range(3)]
    entry = {"name": "FSPI 16%", "mask_type": "fourier", "reconstruction_method": "Fourier Inverse"}
    for steps, ordering, binarization in ((4, "circular", "none"), (3, "square", "none"),
                                          (4, "circular", "bayer")):
        mask = MaskFourier(img_size=IMG, sampling_ratio=0.16, phase_steps=steps,
                           spectrum_ordering=ordering, binarization=binarization)
        mask.generate_masks()
        direct = ApplicatorFourier(_Dataset(images), mask, device="cpu").process_image(1)
        for with_meta in (False, True):
            with tempfile.TemporaryDirectory() as tmp:
                test_dir = os.path.join(tmp, "data", "FSPI_16%")
                os.makedirs(test_dir)
                extra = ({"freq_indices": mask.freq_indices, "phases": mask.phases,
                          "coeff_ids": mask.coeff_ids, "phase_steps": np.array(mask.phase_steps),
                          "binarization": np.array(mask.binarization)} if with_meta else {})
                np.savez(os.path.join(test_dir, "masks.npz"), masks=mask.masks, **extra)
                np.savez(os.path.join(test_dir, "test_images.npz"), originals=np.stack(images))
                cfg = _make_test_config(entry, RemeasureConfig(use_gpu=False))
                from pathlib import Path
                _, applicator = _build_remeasure_pipeline(entry, Path(tmp), IMG, cfg, log)
                assert isinstance(applicator, ApplicatorFourier), type(applicator)
                got = np.asarray(applicator.process_image(1))
                err = np.abs(got - direct).max()
                assert err < 1e-5, (steps, ordering, binarization, with_meta, err)

    # Patterns that no FSPI configuration produces are refused, not mis-reconstructed
    with tempfile.TemporaryDirectory() as tmp:
        test_dir = os.path.join(tmp, "data", "FSPI_16%")
        os.makedirs(test_dir)
        np.savez(os.path.join(test_dir, "masks.npz"),
                 masks=np.random.default_rng(0).random((656, IMG, IMG)).astype(np.float32))
        np.savez(os.path.join(test_dir, "test_images.npz"), originals=np.stack(images))
        from pathlib import Path
        _, applicator = _build_remeasure_pipeline(entry, Path(tmp), IMG,
                                                  _make_test_config(entry, RemeasureConfig(use_gpu=False)), log)
        assert applicator is None

    # Re-measured energy drops the source report's dynamic_* fields
    stale = {"energy_mean_mj": 1.0, "dynamic_reconstruction_energy_pt_mj": 34.0, "dynamic_power_W": 2.0,
             "timing_mean_ms": 3.0, "psnr_denoised": 25.0}
    _strip_old_metrics(stale, kill_timing=False, kill_energy=True)
    assert stale == {"timing_mean_ms": 3.0, "psnr_denoised": 25.0}, json.dumps(stale)
    print("  patterns-only and full exports both reconstruct (4-step, 3-step, Bayer); "
          "foreign patterns refused; stale dynamic_* dropped")


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
