"""Network-input normalisation fitted on the training split (v1.3).

Self-contained script in the style of tests/test_solver_parameters.py:

    .venv/bin/python tests/test_train_normalisation.py

The linear output is min-max normalised before it enters the network. Until
v1.3 the min / max were taken over the whole dataset, test images included, and
the INT8 / INT4 grid spanned the whole array too. They now come from the
training split only, and the integer formats quantise the network input on the
fixed [0, 1] grid. The train / val / test split itself must not change, so that
the same test images are scored as before.
"""

import json
import logging
import os
import sys
import tempfile
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import torch
from torch.utils.data import TensorDataset, random_split

from simulation_engine._4_postprocessor.postprocessor_nn import PostprocessorNN

LOG = logging.getLogger("test_train_normalisation")
N, SIZE, SEED = 50, 32, 10


def _postprocessor(data_format="FP32", outlier=None):
    """Random images; the network input is the image itself, optionally with one extreme image."""
    rng = np.random.default_rng(0)
    clean = (np.round(rng.random((N, SIZE, SIZE)) * 255) / 255).astype(np.float32)
    linear = clean.copy()
    if outlier is not None:
        linear[outlier] = 50.0                                       # far outside the others' range
    ds = SimpleNamespace(data=list(clean), img_size=SIZE, seed=SEED, data_format=data_format)
    pp = PostprocessorNN("u-net", {"img_size": SIZE, "features": [4, 8]}, ds, None, use_gpu=False,
                         logger=LOG, reconstructed_data=linear)
    return clean, linear, pp


def _old_split():
    """The split as drawn before v1.3 (random_split over the dataset itself)."""
    full = TensorDataset(torch.zeros(N, 1))
    parts = random_split(full, [int(0.8 * N), int(0.1 * N), N - int(0.8 * N) - int(0.1 * N)],
                         generator=torch.Generator().manual_seed(SEED))
    return [list(p.indices) for p in parts]


def test_split_is_unchanged():
    _, _, pp = _postprocessor()
    assert [pp.train_indices, pp.val_indices, pp.test_indices] == _old_split()
    print(f"  train / val / test = {len(pp.train_indices)} / {len(pp.val_indices)} / {len(pp.test_indices)}, "
          "same indices as before")


def test_normalisation_uses_training_images_only():
    test_image = _old_split()[2][0]
    _, linear, pp = _postprocessor(outlier=test_image)
    train = linear[pp.train_indices]
    assert pp.norm_fitted_on == "train split"
    assert abs(pp.norm_lo - float(train.min())) < 1e-7, pp.norm_lo
    assert abs(pp.norm_span - (float(train.max() - train.min()) + 1e-8)) < 1e-7, pp.norm_span
    print(f"  extreme test image ignored: lo {pp.norm_lo:.3f}, span {pp.norm_span:.3f} (from the train split)")


def test_integer_formats_use_the_unit_grid():
    test_image = _old_split()[2][0]
    _, _, pp = _postprocessor("INT8", outlier=test_image)
    _, net_in, _ = pp.test_dataset()
    x = np.asarray(net_in)
    assert x.min() >= 0.0 and x.max() <= 1.0, (x.min(), x.max())
    assert np.abs(x * 255 - np.round(x * 255)).max() < 1e-4                # on the 1/255 grid
    assert np.allclose(x[pp.test_indices.index(test_image)], 1.0)          # clipped, not rescaled
    print("  INT8 network input on the [0, 1] grid; values above the training range clip to 1")


def test_fp32_input_is_not_clipped():
    test_image = _old_split()[2][0]
    _, _, pp = _postprocessor("FP32", outlier=test_image)
    _, net_in, _ = pp.test_dataset()
    assert float(np.asarray(net_in)[pp.test_indices.index(test_image)].max()) > 1.0
    print("  FP32 network input keeps values outside [0, 1]")


def test_references_are_unchanged():
    clean, _, pp = _postprocessor("INT8")
    orig, _, _ = pp.test_dataset()
    assert np.array_equal(np.asarray(orig), clean[pp.test_indices])       # 8-bit data: lossless as before
    print("  reference images identical to the stored 8-bit originals")


def test_export_records_where_the_normalisation_was_fitted():
    from ui.custom_widgets.batch_test.batch_test_runner._export import export_datasets
    from ui.custom_widgets.batch_test.test_config_model import TestConfiguration

    _, _, pp = _postprocessor()
    orig, net_in, den = pp.test_dataset()
    entry = {"mask": None, "applicator": None, "originals": orig, "reconstructions": net_in, "denoised": den,
             "config": TestConfiguration(name="t"), "norm_lo": pp.norm_lo, "norm_span": pp.norm_span,
             "norm_fitted_on": pp.norm_fitted_on}
    with tempfile.TemporaryDirectory() as tmp:
        export_datasets({0: entry}, LOG, tmp)
        with open(os.path.join(tmp, "data", "t", "test_config.json")) as fh:
            cfg = json.load(fh)
    assert cfg["normalization"]["fitted_on"] == "train split", cfg["normalization"]
    print("  test_config.json: normalization.fitted_on = 'train split'")


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
