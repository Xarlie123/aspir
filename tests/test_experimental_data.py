"""Checks for the experimental-measurement path (measured.py, spihim.py, experimental.py).

The repository has no test framework (see CLAUDE.md), so this file is a
self-contained script: run it directly with the project venv.

    .venv/bin/python tests/test_experimental_data.py

No download is needed: the SPIHIM adapter is exercised on a tiny synthetic
acquisition and PNG bank written to a temporary directory.
"""

import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch

from simulation_engine import experimental as ex
from simulation_engine._1_dataset_gen.measured import (
    DatasetFromMeasurements,
    MeasuredData,
    load_measurements,
    save_measurements,
)
from simulation_engine._1_dataset_gen.spihim import (
    SATURATION,
    HadamardPngBank,
    load_spihim,
    orient_patterns,
)
from simulation_engine._2_mask_gen.mask_fourier import MaskFourier
from simulation_engine._2_mask_gen.mask_hadamard import MaskHadamard
from simulation_engine._2_mask_gen.mask_hadamard_cake_cutting import (
    MaskHadamardCakeCutting,
)
from simulation_engine._2_mask_gen.mask_scatter import MaskScatter
from simulation_engine._2_mask_gen.mask_sweep import MaskSweep
from simulation_engine._3_applicator.applicator_fourier import ApplicatorFourier
from simulation_engine._3_applicator.applicator_hadamard import ApplicatorHadamard
from simulation_engine._3_applicator.applicator_scatter import ApplicatorScatter
from simulation_engine._3_applicator.applicator_sweep import ApplicatorSweep

logging.disable(logging.INFO)
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _sylvester(n):
    h = np.array([[1.0]])
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h


def _psnr(ref, x):
    mse = np.mean((ref - x) ** 2)
    return 10 * np.log10((ref.max() - ref.min()) ** 2 / mse) if mse > 0 else np.inf


def _synthetic_spihim(tmp, side=8, seed=0):
    """A PNG bank of ``side**2`` bipolar rows and one raw acquisition of a scene."""
    import scipy.io as sio
    from PIL import Image

    rng = np.random.default_rng(seed)
    n = side * side
    H = _sylvester(n).reshape(n, side, side)
    bank_dir = Path(tmp) / "Hadamard_64x64"
    bank_dir.mkdir()
    for k in range(n):
        Image.fromarray(((H[k] > 0) * 255).astype(np.uint8)).save(bank_dir / f"Hadamard_64x64_{2 * k + 1}.png")
        Image.fromarray(((H[k] < 0) * 255).astype(np.uint8)).save(bank_dir / f"Hadamard_64x64_{2 * k + 2}.png")

    scene = rng.uniform(0.2, 1.0, (side, side))
    weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    lam = np.array([400.0, 500.0, 600.0, 700.0, 800.0])
    dark = 800.0
    rows = rng.permutation(n)
    png_idx = []
    for r in rows:                       # acquisition order; negative first for odd rows
        pair = [2 * r + 1, 2 * r + 2]
        png_idx += pair[::-1] if r % 2 else pair
    png_idx = np.array(png_idx)
    binary = np.where(png_idx % 2 == 1, 1, 0)[:, None, None] * (H[(png_idx - 1) // 2] > 0) + \
        np.where(png_idx % 2 == 0, 1, 0)[:, None, None] * (H[(png_idx - 1) // 2] < 0)
    light = np.tensordot(binary.astype(float), scene, axes=([1, 2], [0, 1]))
    M = np.round(dark + light[:, None] * weights[None, :] * 10).astype(np.float64)
    M[3, 4] = SATURATION                 # one clipped value in the last bin
    sio.savemat(Path(tmp) / "synthetic_raw.mat", {
        "M": M.astype(np.uint16),
        "spec": {"lambda_vec": lam},
        "opt": {"N": side, "pattern_name": "Hadamard", "type": "variance", "cr": 1.0, "method": "PosNeg"},
        "par": {"pattern_indices": png_idx, "CT": 0.004, "DMD_fr": 250.0, "camera": "SPC"},
    })
    return Path(tmp), H, scene, rows, weights


def test_spihim_adapter_on_synthetic_mat():
    with tempfile.TemporaryDirectory() as tmp:
        root, H, scene, rows, w = _synthetic_spihim(tmp)
        bank = HadamardPngBank(root)
        assert np.array_equal(bank.bipolar(), H.astype(np.int8)), "bank = PNG(2k+1) - PNG(2k+2)"

        ident = {"rot90": 0, "flip": False}
        d = load_spihim(root / "synthetic_raw.mat", bank, orientation=ident)
        assert d.metadata["bins_used"] == 4 and d.metadata["bins_dropped_saturated"] == 1
        assert np.array_equal(d.patterns, H[rows]), "rows follow the acquisition order"
        expected = 10 * w[:4].sum() * np.tensordot(H[rows], scene, axes=([1, 2], [0, 1]))
        # Counts are integers: at most 0.5 rounding per bin and half, 4 bins x 2 halves.
        assert np.allclose(d.y[0], expected, atol=4.0), "y = sum over bins of pos - neg (dark cancels)"

        b = load_spihim(root / "synthetic_raw.mat", bank, spectral="bin", bin_index=1, orientation=ident)
        assert np.allclose(b.y[0], 10 * w[1] * expected / (10 * w[:4].sum()), atol=1.0)
        band = load_spihim(root / "synthetic_raw.mat", bank, spectral="band", band=(450, 650),
                           orientation=ident)
        assert band.metadata["bins_used"] == 2
        assert band.metadata["wavelength_range_used_nm"] == [500.0, 600.0]

        o = load_spihim(root / "synthetic_raw.mat", bank, orientation={"rot90": 1, "flip": True})
        assert np.array_equal(o.patterns, orient_patterns(H[rows], 1, True))
        x = ex.reconstruct_measured(o, "native", "cpu")[0]
        truth = np.rot90(scene, 1)[:, ::-1] * 10 * w[:4].sum()
        # 40 dB, not 60: the synthetic counts are rounded to integers.
        assert _psnr(truth, x) > 40, "orientation of the patterns orients the reconstruction"
        assert _psnr(scene * 10 * w[:4].sum(), x) < 20, "and differs from the unoriented one"
    print("  pairing, pattern mapping, saturated bin, spectral modes and orientation OK")


def test_noiseless_measured_file_reconstructs_exactly():
    rng = np.random.default_rng(1)
    scene = rng.uniform(0, 1, (64, 64))
    mask = MaskHadamard(img_size=64)
    mask.generate_masks()
    P = mask.masks.astype(np.int8)
    y = P.reshape(len(P), -1).astype(np.float64) @ scene.ravel()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "m.npz"
        save_measurements(path, MeasuredData(y=y, patterns=P, img_size=64, reference=scene,
                                             metadata={"units": "a.u."}))
        data = load_measurements(path)
        assert data.patterns.dtype == np.int8 and data.metadata["units"] == "a.u."
        ds = DatasetFromMeasurements(path=str(path))
        ds.load_data()
        assert ds.is_experimental and np.allclose(ds.data[0], scene)
    for device in DEVICES:
        x = ex.reconstruct_measured(data, "native", device)[0]
        p = _psnr(scene, x)
        assert p > 60, f"{device}: PSNR {p:.1f} dB"
        print(f"  {device}: Hadamard Linear PSNR {p:.1f} dB")


def test_cake_cutting_subsampling_matches_aspir_mask():
    rng = np.random.default_rng(2)
    H = _sylvester(4096)
    order = rng.permutation(4096)
    signs = rng.choice([-1, 1], 4096)
    P = (H[order] * signs[:, None]).reshape(4096, 64, 64).astype(np.int8)
    scene = rng.uniform(0, 1, (64, 64))
    data = MeasuredData(y=P.reshape(4096, -1).astype(np.float64) @ scene.ravel(), patterns=P, img_size=64)

    ratio = 0.08
    m = ex.n_measurements(ratio, 4096)
    mask = MaskHadamardCakeCutting(img_size=64, min_idx=0, max_idx=m)
    mask.generate_masks()
    sub = ex.subsample(data, "cake_cutting", ratio)
    assert sub.n_patterns == m == 327
    assert np.array_equal(sub.patterns.astype(np.float64), mask.masks), "same rows, order and signs"
    assert np.allclose(sub.y[0], sub.patterns.reshape(m, -1).astype(np.float64) @ scene.ravel()), \
        "y follows the sign flips"
    again = ex.subsample_as(data, mask.masks)
    assert np.array_equal(again.patterns, sub.patterns) and np.allclose(again.y, sub.y)
    print(f"  {m} rows equal MaskHadamardCakeCutting's at M/N = {ratio}")


def _simulated_y(app, m):
    """The detector values the simulated path computes for image 0."""
    return app._operator(0, m).forward(app._images([0]))[0].cpu().numpy()


def test_measured_path_equals_simulated_path():
    rng = np.random.default_rng(3)
    scene = rng.uniform(0, 1, (32, 32))
    mask = MaskHadamard(img_size=32, min_idx=0, max_idx=300)
    mask.generate_masks()
    data = MeasuredData(y=np.zeros(300), patterns=mask.masks.astype(np.int8), img_size=32)
    for device in DEVICES:
        for method in ("native", "pseudoinverse", "fista", "tv_norm"):
            app = ex.measured_simulation(data, method, device, scene=scene).applicator
            simulated = app.apply_mask_range(0, 300, 0)
            measured = app.reconstruct(_simulated_y(app, 300))
            assert np.array_equal(simulated, measured), f"{method}/{device} differ"
    # The remaining native algorithms, on their own masks.
    others = [(MaskScatter(img_size=32, point_density=10.0, num_patterns=300, seed=1), ApplicatorScatter),
              (MaskSweep(img_size=32, parametros=[{"angle": a, "bar_width": 1, "stride": 1}
                                                  for a in (0.0, 45.0, 90.0, 135.0)]), ApplicatorSweep),
              (MaskFourier(32, sampling_ratio=0.3), ApplicatorFourier)]
    for mask_o, cls in others:
        mask_o.generate_masks()
        ds = DatasetFromMeasurements(measurements=MeasuredData(
            y=np.zeros(len(mask_o.masks)), patterns=np.zeros((len(mask_o.masks), 32, 32)),
            img_size=32, reference=scene))
        for device in DEVICES:
            app = cls(ds, mask_o, device=device)
            simulated = app.apply_mask_range(0, len(mask_o.masks), 0)
            measured = app.reconstruct(_simulated_y(app, len(mask_o.masks)))
            assert np.array_equal(simulated, measured), f"{cls.__name__}/{device} differ"
    print(f"  seven algorithms x {DEVICES}: reconstruct(y) == simulated path")


def test_legacy_backend_refuses_measured_data():
    mask = MaskHadamard(img_size=16)
    mask.generate_masks()
    data = MeasuredData(y=np.zeros(256), patterns=mask.masks.astype(np.int8), img_size=16)
    ds = DatasetFromMeasurements(measurements=data)
    app = ApplicatorHadamard(ds, mask, backend="legacy")
    try:
        app.reconstruct(np.zeros(256))
    except NotImplementedError:
        print("  legacy backend raises NotImplementedError")
        return
    raise AssertionError("legacy backend accepted measured data")


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
