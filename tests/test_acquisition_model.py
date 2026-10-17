"""Checks for the modelled acquisition: projections per pattern, by pattern values and scheme.

The repository has no test framework (see CLAUDE.md), so this file is a
self-contained script: run it directly with the project venv.

    .venv/bin/python tests/test_acquisition_model.py
"""

import json
import logging
import math
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import torch
from PySide6.QtWidgets import QApplication

import simulation_engine._5_analyzer.acquisition as acq
from simulation_engine._2_mask_gen.mask_cal_sal import MaskCalSal
from simulation_engine._2_mask_gen.mask_fourier import MaskFourier
from simulation_engine._2_mask_gen.mask_hadamard import MaskHadamard
from simulation_engine._2_mask_gen.mask_hadamard_cake_cutting import MaskHadamardCakeCutting
from simulation_engine._2_mask_gen.mask_hadamard_walsh_paley import MaskHadamardWalshPaley
from simulation_engine._2_mask_gen.mask_measured import MaskMeasured
from simulation_engine._2_mask_gen.mask_scatter import MaskScatter
from simulation_engine._2_mask_gen.mask_sweep import MaskSweep

# The ui package sets matplotlib's Qt backend on import. Without a display (over
# SSH, e.g. on a Jetson) matplotlib accepts it only once a Qt application exists.
_APP = QApplication.instance() or QApplication([])

from ui.custom_widgets.batch_reports.batch_report_model import BatchReportModel
from ui.custom_widgets.batch_reports.remeasure.remeasure_worker import RemeasureConfig, _make_test_config
from ui.custom_widgets.batch_test.batch_test_runner._timing import measure_timing
from ui.custom_widgets.batch_test.test_config_model import BatchTestConfig, TestConfiguration

LOG = logging.getLogger("test_acquisition_model")
RATE = 10.752
N = 32


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.records = []

    def emit(self, record):
        self.records.append(record)


def _masks():
    sweep = MaskSweep(img_size=N, parametros=[{"angle": 0.0, "bar_width": 2, "stride": 4}], logger=LOG)
    built = {
        "hadamard_natural": MaskHadamard(N, 0, 40, logger=LOG),
        "hadamard_cake_cutting": MaskHadamardCakeCutting(N, 0, 40, logger=LOG),
        "hadamard_walsh_paley": MaskHadamardWalshPaley(N, 0, 40, logger=LOG),
        "cal_sal": MaskCalSal(img_size=N, logger=LOG),
        "scatter": MaskScatter(img_size=N, point_density=10.0, num_patterns=40, seed=1, logger=LOG),
        "sweep": sweep,
        "fourier": MaskFourier(img_size=N, sampling_ratio=0.08),
        "fourier_binary": MaskFourier(img_size=N, sampling_ratio=0.08, binarization="bayer"),
    }
    for m in built.values():
        m.generate_masks()
    return built


def test_counts_by_pattern_values_and_scheme():
    for name, mask in _masks().items():
        m = len(mask.masks)
        out = acq.acquisition_for_mask(mask, m, RATE, logger=LOG, name=name)
        bipolar = name.startswith("hadamard") or name == "cal_sal"
        expected_scheme = acq.DIFFERENTIAL if bipolar else acq.PER_PATTERN
        expected_frames = 2 * m if bipolar else m
        assert out["timing_projection_scheme"] == expected_scheme, (name, out)
        assert out["timing_num_projections"] == expected_frames, (name, out)
        assert math.isclose(out["timing_acquisition_ms"], expected_frames / RATE), (name, out)
        assert out["timing_pattern_polarity"] == ("bipolar" if bipolar else
                                                  "grey" if name == "fourier" else "binary"), (name, out)
        if bipolar:
            for scheme, frames in ((acq.PER_PATTERN, m), (acq.REFERENCE, m + 1), (acq.DIFFERENTIAL, 2 * m)):
                got = acq.acquisition_for_mask(mask, m, RATE, scheme)["timing_num_projections"]
                assert got == frames, (name, scheme, got)
    print("  bipolar Hadamard families: differential (2M) by default, M and M + 1 on request; "
          "Scatter, Sweep, binary and grey FSPI: M")


def test_a_scheme_that_does_not_apply_falls_back_with_a_warning():
    cap = _Capture()
    log = logging.getLogger("test_acquisition_model.fallback")
    log.addHandler(cap)
    log.setLevel(logging.WARNING)
    sweep = _masks()["sweep"]
    out = acq.acquisition_for_mask(sweep, len(sweep.masks), RATE, acq.DIFFERENTIAL, logger=log, name="Sweep 1")
    assert out["timing_projection_scheme"] == acq.PER_PATTERN and out["timing_num_projections"] == len(sweep.masks)
    assert len(cap.records) == 1 and "does not apply" in cap.records[0].getMessage()
    print("  'differential' on Sweep: one per pattern, one warning")


def test_grey_level_rate_applies_to_grey_patterns_only():
    masks = _masks()
    grey, binary = masks["fourier"], masks["fourier_binary"]
    g = acq.acquisition_for_mask(grey, len(grey.masks), RATE, grey_level_rate_khz=0.1)
    b = acq.acquisition_for_mask(binary, len(binary.masks), RATE, grey_level_rate_khz=0.1)
    assert math.isclose(g["timing_acquisition_ms"], len(grey.masks) / 0.1)
    assert math.isclose(b["timing_acquisition_ms"], len(binary.masks) / RATE)
    assert g["timing_grey_level_rate_khz"] == 0.1
    print(f"  grey FSPI at 0.1 kHz: {g['timing_acquisition_ms']:.0f} ms; binary FSPI keeps {RATE} kHz")


def test_measured_data_are_not_modelled():
    patterns = np.where(np.random.default_rng(0).random((8, N, N)) > 0.5, 1.0, -1.0)
    out = acq.acquisition_for_mask(MaskMeasured(patterns), 8, RATE)
    assert out["timing_projection_scheme"] == acq.NOT_MODELLED and out["timing_acquisition_ms"] == 0.0
    print("  measured patterns: not_modelled, no acquisition time")


def test_configuration_round_trip():
    assert TestConfiguration().projection_scheme is None and TestConfiguration.from_dict({}).projection_scheme is None
    cfg = TestConfiguration(name="t", mask_type="hadamard_cake_cutting", projection_scheme="reference",
                            grey_level_rate_khz=1.5)
    back = TestConfiguration.from_dict(json.loads(json.dumps(cfg.to_dict())))
    assert (back.projection_scheme, back.grey_level_rate_khz) == ("reference", 1.5)
    try:
        TestConfiguration.from_dict({"projection_scheme": "twice"})
        raise AssertionError("an unknown scheme was accepted")
    except ValueError:
        pass
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "b.batch_config")
        BatchTestConfig(name="b", tests=[cfg]).save(path)
        loaded = BatchTestConfig.load(path).tests[0]
    assert (loaded.projection_scheme, loaded.grey_level_rate_khz) == ("reference", 1.5)
    rc = _make_test_config({"name": "x"}, RemeasureConfig(projection_scheme="per_pattern", grey_level_rate_khz=2.0))
    assert (rc.projection_scheme, rc.grey_level_rate_khz) == ("per_pattern", 2.0)
    print("  .batch_config and re-measure carry the scheme and the grey-level rate; unset = family default")


def test_old_reports_read_as_one_per_pattern_and_mixing_is_detected():
    assert acq.report_scheme({}) == acq.PER_PATTERN
    old = {"name": "CC 16%", "mask_type": "hadamard_cake_cutting", "timing_acquisition_ms": 60.92,
           "timing_num_patterns": 655}
    new = dict(old, timing_acquisition_ms=121.84, timing_num_projections=1310, timing_projection_scheme="differential")
    model = BatchReportModel()
    with tempfile.TemporaryDirectory() as tmp:
        for i, entry in enumerate((old, new)):
            path = Path(tmp) / f"r{i}.batch_analysis_report"
            path.write_text(json.dumps({"metadata": {"batch_name": f"r{i}"}, "results": [entry]}))
            assert model.load_experiment(path) is not None
        assert model.bipolar_projection_schemes() == ["differential", "per_pattern"], model.bipolar_projection_schemes()
    print("  a report without the new fields reads as per_pattern; Batch Reports sees the two schemes")


def test_remeasure_applies_the_scheme_it_is_given():
    mask = _masks()["hadamard_cake_cutting"]
    m = len(mask.masks)
    net = torch.nn.Conv2d(1, 1, 3, padding=1).eval()
    post = SimpleNamespace(model=net, img_size=N, is_conv=True, device=torch.device("cpu"), applicator=None)
    entry = {"name": "CC", "mask_type": "hadamard_cake_cutting"}
    for scheme, frames in ((None, 2 * m), ("per_pattern", m), ("reference", m + 1)):
        cfg = _make_test_config(entry, RemeasureConfig(use_gpu=False, projection_scheme=scheme))
        cfg.timing_warmup_runs, cfg.timing_measurement_runs = 1, 2
        res = measure_timing(post, cfg, SimpleNamespace(data=[]), LOG,
                             applicator=SimpleNamespace(mask=mask, reconstruction_device="cpu",
                                                        process_image=lambda i: None),
                             warmup_runs=1, measurement_runs=2, sampling_rate_khz=RATE, max_recon_samples=1)
        assert res["timing_num_projections"] == frames and res["timing_num_patterns"] == m, (scheme, res)
        assert math.isclose(res["timing_acquisition_ms"], frames / RATE)
    # A report whose patterns were not saved: the values follow from the family
    assert acq.polarity_of_type("hadamard_cake_cutting") == acq.BIPOLAR
    assert acq.polarity_of_type("fourier", "none") == acq.GREY and acq.polarity_of_type("sweep") == acq.BINARY
    print(f"  Cake-Cutting M = {m}: default {2 * m}, per_pattern {m}, reference {m + 1} projections")


def test_the_gui_offers_only_the_schemes_that_apply():
    from ui.custom_widgets.batch_test.test_config_widget import TestConfigWidget

    w = TestConfigWidget()
    w.set_config(TestConfiguration(name="t", mask_type="hadamard_cake_cutting", projection_scheme="reference"))
    assert w.projection_scheme_combo.currentData() == "reference"
    enabled = {w.projection_scheme_combo.itemData(i) for i in range(w.projection_scheme_combo.count())
               if w.projection_scheme_combo.model().item(i).isEnabled()}
    assert enabled == {None, "per_pattern", "differential", "reference"}, enabled
    w.mask_type_combo.setCurrentText("sweep")
    enabled = {w.projection_scheme_combo.itemData(i) for i in range(w.projection_scheme_combo.count())
               if w.projection_scheme_combo.model().item(i).isEnabled()}
    assert enabled == {None, "per_pattern"}, enabled
    assert w.projection_scheme_combo.currentData() is None, "a scheme that no longer applies must reset"
    assert w._config.projection_scheme is None
    print("  Hadamard: all three schemes; Sweep: family default or one per pattern")


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
