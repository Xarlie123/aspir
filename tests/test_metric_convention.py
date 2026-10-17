"""Quality-metric convention 2 and the stable Cake-Cutting order (v1.3).

Self-contained script in the style of tests/test_solver_parameters.py:

    .venv/bin/python tests/test_metric_convention.py

Before v1.3 the "reconstructed" quality level scored the network input (the
linear output min-max normalised over the dataset), Hadamard Linear returned the
pseudoinverse times N/M, PSNR / SSIM used each reference image's own range and
estimates were not clipped (the ASPIR paper's analysis has the diagnosis).
These checks pin the convention that replaced it.
"""

import json
import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pandas as pd

from simulation_engine._1_dataset_gen.dataset import DatasetABC
from simulation_engine._2_mask_gen.mask_hadamard_cake_cutting import MaskHadamardCakeCutting
from simulation_engine._2_mask_gen.mask_sweep import MaskSweep
from simulation_engine._4_postprocessor.postprocessor_nn import PostprocessorNN
from simulation_engine._5_analyzer.analyzer_noise import METRIC_CONVENTION, NoiseAnalyzer
from simulation_engine.simulation import Simulation

LOG = logging.getLogger("test_metric_convention")
SIZE = 32                      # LPIPS (AlexNet) needs at least 32 x 32


class ArrayDataset(DatasetABC):
    def __init__(self, images):
        super().__init__("arrays", images.shape[1], LOG)
        self.data = [np.asarray(im, dtype=np.float32) for im in images]
        self.num_images = len(self.data)
        self.seed = 10

    def load_data(self, progress_callback=None):
        return True

    def to_dataframe(self):
        return pd.DataFrame([im.ravel() for im in self.data])


def _images(n=40, seed=0):
    """Smooth random images in [0, 1] (8-bit levels, as a real dataset)."""
    rng = np.random.default_rng(seed)
    x = rng.random((n, SIZE // 4, SIZE // 4))
    x = np.kron(x, np.ones((4, 4)))
    return np.round(x * 255) / 255


def _applicator(images, mask, method="native", backend="torch"):
    sim = Simulation(LOG)
    sim.set_dataset(ArrayDataset(images))
    mask.generate_masks()
    sim.mask = mask
    sim.set_applicator(reconstruction_method=method, device="cpu")
    if backend != "torch":
        sim.applicator.backend = backend
    return sim.applicator


def _pinv(images, masks):
    S = masks.reshape(len(masks), -1).astype(np.float64)
    X = images.reshape(len(images), -1)
    return (np.linalg.pinv(S) @ (S @ X.T)).T.reshape(images.shape)


def test_hadamard_linear_is_the_pseudoinverse():
    X = _images(6)
    for backend in ("torch", "legacy"):
        app = _applicator(X, MaskHadamardCakeCutting(img_size=SIZE, max_idx=40, logger=LOG), backend=backend)
        out = np.asarray(app.process_dataset(), dtype=np.float64).reshape(X.shape)
        err = np.abs(out - _pinv(X, app.mask.masks)).max()
        assert err < 1e-5, (backend, err)
        print(f"  {backend}: M = 40 of {SIZE * SIZE}, max |native - pinv| = {err:.1e}")


def test_full_sampling_returns_the_image():
    X = _images(3)
    app = _applicator(X, MaskHadamardCakeCutting(img_size=SIZE, logger=LOG))
    out = np.asarray(app.process_dataset(), dtype=np.float64).reshape(X.shape)
    assert np.abs(out - X).max() < 1e-5
    print("  M = N: the image itself")


def test_psnr_ssim_convention_on_known_arrays():
    rng = np.random.default_rng(1)
    ref = (rng.random((4, SIZE, SIZE)) * 0.8).astype(np.float32)        # range < 1 on purpose
    shifted = ref + 0.1                                                    # MSE = 0.01
    over = ref + 5.0                                                       # clipped to 1
    a = NoiseAnalyzer(list(ref), list(shifted), list(over))
    a.analyze()
    assert abs(a.psnr_noise_mean - 20.0) < 1e-3, a.psnr_noise_mean          # data range 1, not ref's
    expected = np.mean([10 * np.log10(1 / np.mean((1 - r.astype(np.float64)) ** 2)) for r in ref])
    assert abs(a.psnr_rec_mean - expected) < 1e-3, (a.psnr_rec_mean, expected)
    assert METRIC_CONVENTION == 2
    print(f"  +0.1 offset -> {a.psnr_noise_mean:.3f} dB; values > 1 clipped -> {a.psnr_rec_mean:.2f} dB")


def _postprocessor(mask, n=40):
    X = _images(n)
    app = _applicator(X, mask)
    pp = PostprocessorNN("u-net", {"img_size": SIZE, "features": [4, 8]}, app.dataset, app,
                         batch_size=8, use_gpu=False, logger=LOG)
    return X, pp, app


def test_linear_stage_is_scored_on_the_image_scale():
    X, pp, app = _postprocessor(MaskHadamardCakeCutting(img_size=SIZE, max_idx=64, logger=LOG))
    test = np.asarray(pp.test_linear())
    expected = _pinv(X[pp.test_indices], np.asarray(app.mask.masks))
    assert np.abs(test - expected).max() < 1e-5
    assert pp.linear_calibration is None
    _, net_in, _ = pp.test_dataset()
    assert min(float(np.min(v)) for v in net_in) >= 0 and max(float(np.max(v)) for v in net_in) <= 1
    print("  Hadamard: test_linear = pseudoinverse of the test images; network input stays in [0, 1]")


def test_correlation_estimators_are_calibrated_on_the_train_split():
    params = [{"angle": a, "bar_width": 2, "stride": 2} for a in (0.0, 90.0)]
    X, pp, _ = _postprocessor(MaskSweep(SIZE, params, LOG))
    cal = pp.linear_calibration
    assert cal and cal["fitted_on"] == "train split", cal
    test = np.asarray(pp.test_linear())
    raw_mean_err = abs(float(np.mean(test)) - float(np.mean(X[pp.test_indices])))
    assert raw_mean_err < 0.05, raw_mean_err
    print(f"  Sweep: gain {cal['gain']:.3g}, offset {cal['offset']:.3g}; test mean within {raw_mean_err:.3f}")


def test_batch_quality_fields():
    from ui.custom_widgets.batch_test.batch_test_runner._quality import quality_results

    _, pp, _ = _postprocessor(MaskHadamardCakeCutting(img_size=SIZE, max_idx=64, logger=LOG))
    results, (orig, _, den) = quality_results(pp, ["quality"], LOG)
    ref = NoiseAnalyzer(orig, pp.test_linear(), den)
    ref.analyze()
    assert results["metric_convention"] == 2
    assert abs(results["psnr_recons"] - ref.psnr_noise_mean) < 1e-9
    assert "psnr_network_input" in results and results["linear_calibration"] is None
    print(f"  psnr_recons {results['psnr_recons']:.2f} dB (linear), network input "
          f"{results['psnr_network_input']:.2f} dB")


def test_cake_cutting_ties_are_stable():
    m = MaskHadamardCakeCutting(img_size=SIZE, logger=LOG)
    H = np.kron(*(2 * [m.walsh_hadamard_transform(int(np.log2(SIZE)))]))
    _, order, blocks = m.cake_cutting(H, int(np.log2(SIZE)))
    for value in np.unique(blocks):
        members = order[blocks[order] == value]
        assert np.all(np.diff(members) > 0), value                         # ties in Sylvester order
    _, again, _ = m.cake_cutting(H, int(np.log2(SIZE)))
    assert np.array_equal(order, again)
    print(f"  {len(np.unique(blocks))} block counts; ties kept in Sylvester order")


def test_export_records_the_convention():
    from ui.custom_widgets.batch_test.batch_test_runner._export import export_datasets
    from ui.custom_widgets.batch_test.test_config_model import TestConfiguration

    _, pp, app = _postprocessor(MaskHadamardCakeCutting(img_size=SIZE, max_idx=64, logger=LOG))
    orig, net_in, den = pp.test_dataset()
    entry = {"mask": app.mask, "applicator": app, "originals": orig, "reconstructions": net_in, "denoised": den,
             "linear": pp.test_linear(), "config": TestConfiguration(name="t"), "norm_lo": pp.norm_lo,
             "norm_span": pp.norm_span}
    with tempfile.TemporaryDirectory() as tmp:
        export_datasets({0: entry}, LOG, tmp)
        d = os.path.join(tmp, "data", "t")
        with np.load(os.path.join(d, "test_images.npz")) as z:
            assert "linear" in z.files and "reconstructions" in z.files
        with open(os.path.join(d, "test_config.json")) as fh:
            cfg = json.load(fh)
    assert cfg["metric_convention"] == 2 and cfg["normalization"]["linear_divisor"] == "pixels", cfg
    print("  test_images.npz has 'linear'; test_config.json records the convention and divisor")


def test_mixed_conventions_are_detected():
    from PySide6.QtWidgets import QApplication

    from ui.custom_widgets.batch_reports.batch_report_model import BatchReportModel

    _app = QApplication.instance() or QApplication([])
    model = BatchReportModel(logger=LOG)
    with tempfile.TemporaryDirectory() as tmp:
        for name, meta in (("old", {}), ("new", {"metric_convention": 2})):
            path = os.path.join(tmp, f"{name}.batch_analysis_report")
            with open(path, "w") as fh:
                json.dump({"metadata": meta, "summary": {}, "results": []}, fh)
            model.load_experiment(Path(path))
    assert model.metric_conventions() == [1, 2], model.metric_conventions()
    print("  reports without the field count as convention 1; mixing is detected")


if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_")]
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
