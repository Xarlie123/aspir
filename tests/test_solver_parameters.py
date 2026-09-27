"""FISTA / TV parameters reach the solvers (R1.3a).

Self-contained script in the style of tests/test_fourier_fspi.py:

    .venv/bin/python tests/test_solver_parameters.py

Before R1.3a the batch runner set attributes the solvers never read, so FISTA
always ran with lambda = 1e-3 and 500 iterations and TV with lambda = 0.1 and
100 iterations. These checks pin both halves of the fix: user values are
applied, and default settings still reproduce what always ran.
"""

import json
import logging
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from scipy.ndimage import gaussian_filter

from simulation_engine._2_mask_gen.mask_scatter import MaskScatter
from simulation_engine._3_applicator.applicator_fista import ApplicatorFISTA
from simulation_engine._3_applicator.applicator_tv_norm import ApplicatorTV
from simulation_engine.simulation import Simulation
from ui.custom_widgets.batch_reports.remeasure.remeasure_worker import (
    RemeasureConfig,
    _make_test_config,
)
from ui.custom_widgets.batch_test.batch_test_runner._pipeline import (
    create_applicator,
    solver_params_of,
)
from ui.custom_widgets.batch_test.test_config_model import BatchTestConfig, TestConfiguration

logging.basicConfig(level=logging.ERROR)
LOG = logging.getLogger("test")
SIZE = 32

HISTORICAL = {"fista": (1e-3, 500), "tv": (0.1, 100)}


class _Dataset:
    def __init__(self, images):
        self.name = "solver_params"
        self.img_size = images[0].shape[0]
        self.data = images


def _image(seed=0):
    img = gaussian_filter(np.random.default_rng(seed).random((SIZE, SIZE)), 1.5)
    return (img - img.min()) / (img.max() - img.min())


class _FixedMask(MaskScatter):
    """A scatter mask whose patterns are generated once and kept.

    ``create_applicator`` calls ``generate_masks()``, which for scatter draws new
    random patterns; comparisons need the same patterns on both sides.
    """

    def generate_masks(self, progress_callback=None):
        if self.masks is None:
            super().generate_masks(progress_callback)


def _mask():
    np.random.seed(3)
    mask = _FixedMask(img_size=SIZE, point_density=10.0, num_patterns=164, seed=3, logger=LOG)
    mask.generate_masks()
    return mask


def _rel(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / np.linalg.norm(np.asarray(b)))


# ---------------------------------------------------------------------------
def test_defaults_are_the_historical_values():
    """A default config builds the solvers with the values that always ran."""
    ds = _Dataset([_image()])
    mask = _mask()
    cfg = TestConfiguration()
    assert (cfg.fista_lambda, cfg.fista_iterations) == HISTORICAL["fista"]
    assert (cfg.tv_lambda, cfg.tv_iterations) == HISTORICAL["tv"]

    for method, cls, attr in (("fista", ApplicatorFISTA, "maxit"), ("tv_norm", ApplicatorTV, "niter")):
        app = create_applicator(TestConfiguration(reconstruction_method=method), mask, ds)
        assert isinstance(app, cls)
        key = "fista" if method == "fista" else "tv"
        assert (app.lam, getattr(app, attr)) == HISTORICAL[key], (app.lam, getattr(app, attr))
        # Legacy backend with those values == legacy backend with the class
        # defaults: default settings reproduce what always ran. TV is bit for
        # bit. Legacy FISTA is not reproducible bit for bit even against itself:
        # ARPACK's random start vector moves the last bits of the Lipschitz
        # constant, so two identical legacy runs differ by ~1e-13. That run-to-run
        # floor is the tightest bound that can hold.
        kwargs = {"lam": app.lam, attr: getattr(app, attr), "backend": "legacy"}
        a = cls(ds, mask, **kwargs).process_image(0)
        b = cls(ds, mask, backend="legacy").process_image(0)
        if method == "tv_norm":
            assert np.array_equal(a, b), f"{method}: legacy result changed"
            how = "bit for bit"
        else:
            err = _rel(a, b)
            assert err < 1e-10, f"{method}: legacy result changed ({err:.1e})"
            how = f"{err:.0e} (legacy run-to-run floor)"
        print(f"  {method:8s} lam={app.lam:g} iterations={getattr(app, attr)}; legacy {how}")


def test_non_default_values_reach_the_solver():
    """User values change the result and match a direct construction, on both backends."""
    ds = _Dataset([_image(seed=1)])
    mask = _mask()
    cases = [("fista", ApplicatorFISTA, "maxit", {"fista_lambda": 2.0, "fista_iterations": 20}),
             ("tv_norm", ApplicatorTV, "niter", {"tv_lambda": 0.4, "tv_iterations": 15})]
    for method, cls, attr, values in cases:
        lam, it = list(values.values())
        cfg = TestConfiguration(reconstruction_method=method, reconstruction_device="cpu", **values)
        app = create_applicator(cfg, mask, ds)
        assert (app.lam, getattr(app, attr)) == (lam, it)
        via_config = app.process_image(0)
        direct = cls(ds, mask, lam=lam, device="cpu", **{attr: it}).process_image(0)
        default = cls(ds, mask, device="cpu").process_image(0)
        assert _rel(via_config, direct) < 1e-6, "config path differs from direct construction"
        change = _rel(via_config, default)
        assert change > 1e-2, f"{method}: non-default values did not change the result ({change:.1e})"

        legacy = cls(ds, mask, lam=lam, backend="legacy", **{attr: it}).process_image(0)
        legacy_default = cls(ds, mask, backend="legacy").process_image(0)
        assert _rel(legacy, legacy_default) > 1e-2, f"{method}: legacy ignores the values"
        parity = _rel(via_config, legacy)
        assert parity < 1e-2, f"{method}: torch vs legacy with the same values {parity:.1e}"
        print(f"  {method:8s} lam={lam:g} it={it}: change vs default {change:.2f}, "
              f"torch vs legacy {parity:.1e}")


def test_old_configs_migrate_to_the_values_that_ran():
    """A 2.1 config loads with the historical values and warns; 2.2 keeps its own."""
    stored = TestConfiguration(name="t", reconstruction_method="fista").to_dict()
    stored.update({"fista_lambda": 0.01, "fista_iterations": 100, "tv_lambda": 0.1, "tv_iterations": 50})

    class _Capture(logging.Handler):
        def __init__(self):
            super().__init__()
            self.records = []

        def emit(self, record):
            self.records.append(record)

    cap = _Capture()
    logging.getLogger("BatchTestConfig").addHandler(cap)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            results = {}
            for version in ("2.1", "2.2", None):
                data = {"name": "b", "tests": [stored]}
                if version:
                    data["format_version"] = version
                path = os.path.join(tmp, f"v{version}.batch_config")
                with open(path, "w") as fh:
                    json.dump(data, fh)
                cap.records.clear()
                t = BatchTestConfig.load(path).tests[0]
                results[version] = ((t.fista_lambda, t.fista_iterations, t.tv_lambda, t.tv_iterations),
                                    len([r for r in cap.records if r.levelno == logging.WARNING]))
    finally:
        logging.getLogger("BatchTestConfig").removeHandler(cap)

    migrated = (*HISTORICAL["fista"], *HISTORICAL["tv"])
    assert results["2.1"] == (migrated, 1), results["2.1"]
    assert results[None] == (migrated, 1), results[None]
    assert results["2.2"] == ((0.01, 100, 0.1, 50), 0), results["2.2"]
    # Saving writes the current format.
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "new.batch_config")
        BatchTestConfig(name="b", tests=[TestConfiguration()]).save(path)
        with open(path) as fh:
            assert json.load(fh)["format_version"] == "2.2"
    print(f"  2.1 and unversioned -> {migrated} with one warning; 2.2 keeps (0.01, 100, 0.1, 50)")


def test_remeasure_uses_the_recorded_solver():
    """Re-measure builds TV for a TV test and applies the recorded parameters."""
    ds = _Dataset([_image(seed=2)])
    mask = _mask()
    rc = RemeasureConfig(use_gpu=False)
    # A report written before the fix: display name, no reconstruction_params.
    old = {"name": "t", "mask_type": "scatter", "reconstruction_method": "TV Norm"}
    app = create_applicator(_make_test_config(old, rc), mask, ds)
    assert isinstance(app, ApplicatorTV), type(app).__name__
    assert (app.lam, app.niter) == HISTORICAL["tv"]
    # A report written after the fix.
    new = dict(old, reconstruction_params={"lam": 0.3, "iterations": 7})
    app = create_applicator(_make_test_config(new, rc), mask, ds)
    assert isinstance(app, ApplicatorTV) and (app.lam, app.niter) == (0.3, 7)
    fista = {"name": "f", "mask_type": "scatter", "reconstruction_method": "FISTA",
             "reconstruction_params": {"lam": 0.02, "iterations": 40}}
    app = create_applicator(_make_test_config(fista, rc), mask, ds)
    assert isinstance(app, ApplicatorFISTA) and (app.lam, app.maxit) == (0.02, 40)
    # What the runner records is what re-measure reads back.
    assert solver_params_of(app) == {"lam": 0.02, "iterations": 40}
    print("  'TV Norm' -> ApplicatorTV; recorded params applied; absent params -> historical")


def test_simulation_and_single_test_panels():
    """Simulation.set_applicator and the Single Test panels pass the values through."""
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from simulation_engine._2_mask_gen.mask_hadamard import MaskHadamard
    from ui.custom_widgets.mask_control.fourier_control.fourier_control_widget import (
        FourierControlWidget,
    )
    from ui.custom_widgets.mask_control.hadamard_control.hadamard_control_widget import (
        HadamardControlWidget,
    )
    from ui.custom_widgets.mask_control.scatter_control.scatter_control_widget import (
        ScatterControlWidget,
    )
    from ui.custom_widgets.mask_control.sweep_control.sweep_control_widget import SweepControlWidget

    ds = _Dataset([_image(seed=4)])
    sim = Simulation(LOG)
    sim.set_dataset(ds)
    mask = _mask()
    sim.mask = mask
    sim.set_applicator(reconstruction_method="fista", device="cpu",
                       solver_params={"lam": 0.07, "iterations": 11})
    assert (sim.applicator.lam, sim.applicator.maxit) == (0.07, 11)
    mask.applicator_type_scatter = "TV-norm"
    mask.solver_params = {"lam": 0.2, "iterations": 9}
    sim.set_mask(mask)
    assert isinstance(sim.applicator, ApplicatorTV) and (sim.applicator.lam, sim.applicator.niter) == (0.2, 9)

    panels = {
        "scatter": (ScatterControlWidget(logger=LOG), "select_applicator_scatter_list"),
        "sweep": (SweepControlWidget(logger=LOG), "reconstruction_method_combo"),
        "hadamard": (HadamardControlWidget(mask_cls=MaskHadamard, logger=LOG), "reconstruction_method_combo"),
        "fourier": (FourierControlWidget(logger=LOG), "reconstruction_method_combo"),
    }
    for name, (w, combo_name) in panels.items():
        combo = getattr(w, combo_name)
        spw = w.solver_params_widget
        combo.setCurrentIndex(0)                       # native method
        assert spw.isHidden() and spw.params() is None, name
        combo.setCurrentText("FISTA")
        assert not spw.isHidden(), name
        assert spw.params() == {"lam": 1e-3, "iterations": 500}, (name, spw.params())
        combo.setCurrentText("TV-norm")
        assert spw.params() == {"lam": 0.1, "iterations": 100}, (name, spw.params())
        spw.set_params(0.25, 30)
        w.set_img_size(SIZE)
        got = []
        w.maskReady.connect(got.append)
        w.generate_masks_button.click()
        assert got and got[0].solver_params == {"lam": 0.25, "iterations": 30}, name
    # In Batch Test the panels' controls are suppressed (the form has its own).
    spw = panels["fourier"][0].solver_params_widget
    spw.suppress()
    panels["fourier"][0].reconstruction_method_combo.setCurrentText("FISTA")
    assert spw.isHidden()
    print("  set_applicator/set_mask apply solver_params; all four panels show, default and emit them")


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
