"""Checks for the propagation of the idle-baseline uncertainty (R2.2).

The repository has no test framework (see CLAUDE.md), so this file is a
self-contained script: run it directly with the project venv.

    .venv/bin/python tests/test_energy_uncertainty.py

No energy counter is read: every check uses synthetic readings.
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from simulation_engine._5_analyzer.baseline_capture import (
    BackendBaseline,
    BaselineResult,
    add_dynamic_uncertainty,
    derive_dynamic_metrics,
    dynamic_energy_uncertainty,
)
from simulation_engine._5_analyzer.energy_backends._rapl import RAPLBackend

GPU, CPU = "NVIDIA GeForce GTX 1650", "AMD Ryzen 7 4800H with Radeon Graphics"


def _baseline(gpu_samples, cpu_samples):
    def one(name, vals):
        vals = list(map(float, vals))
        return BackendBaseline(backend=name, power_W_mean=float(np.mean(vals)),
                               power_W_std=float(np.std(vals)), duration_s=float(len(vals)),
                               n_samples=len(vals), samples_W=vals)
    return BaselineResult(per_backend={GPU: one(GPU, gpu_samples), CPU: one(CPU, cpu_samples)},
                          duration_s=60.0, requested_duration_s=60.0)


def test_idle_term_uses_the_standard_deviation():
    idle = {"mean_W": 6.0, "std_W": 0.2, "sem_W": 0.2 / math.sqrt(60), "n": 60}
    P = np.full(10, 7.0)
    t = np.full(10, 8.0)
    out = dynamic_energy_uncertainty(P * t, P, t, idle)
    assert math.isclose(out["dynamic_energy_pt_mj"], (7.0 - 6.0) * 8.0)
    assert math.isclose(out["dynamic_energy_sigma_idle_mj"], 0.2 * 8.0)
    assert math.isclose(out["dynamic_energy_sigma_idle_sem_mj"], 0.2 / math.sqrt(60) * 8.0)
    assert out["dynamic_energy_sigma_blocks_mj"] == 0.0, "identical blocks have no spread"
    assert math.isclose(out["dynamic_energy_sigma_mj"], 1.6)
    print("  idle term = sigma_idle * t, not the standard error")


def test_block_term_matches_the_analytical_value():
    rng = np.random.default_rng(1)
    P = 7.0 + rng.normal(0, 0.1, 10)
    t = 8.0 + 0.5 * (P - 7.0) + rng.normal(0, 0.05, 10)     # correlated with P
    idle = {"mean_W": 6.0, "std_W": 0.0, "sem_W": 0.0, "n": 60}
    out = dynamic_energy_uncertainty(P * t, P, t, idle)
    B, pb, tb = len(P), P.mean(), t.mean()
    sp2, st2 = P.var(ddof=1), t.var(ddof=1)
    spt = np.cov(P, t, ddof=1)[0, 1]
    pd = pb - 6.0
    expected = math.sqrt((tb ** 2 * sp2 + pd ** 2 * st2 + 2 * tb * pd * spt) / B)
    assert math.isclose(out["dynamic_energy_sigma_blocks_mj"], expected, rel_tol=1e-12)
    print(f"  delta-method block term {expected:.4f} mJ reproduced")


def test_sigma_matches_known_variance():
    """Synthetic readings with known variance: the reported sigma is the true spread."""
    rng = np.random.default_rng(2)
    mu_p, s_p, t0, B = 7.0, 0.3, 8.0, 10
    p_idle, s_idle = 6.0, 0.1
    analytical = math.sqrt((s_idle * t0) ** 2 + (t0 * s_p / math.sqrt(B)) ** 2)
    reported, estimates = [], []
    for _ in range(4000):
        P = rng.normal(mu_p, s_p, B)
        t = np.full(B, t0)
        idle_drawn = rng.normal(p_idle, s_idle)      # the baseline itself fluctuates
        estimates.append((P.mean() - idle_drawn) * t0)
        out = dynamic_energy_uncertainty(P * t, P, t, {"mean_W": p_idle, "std_W": s_idle,
                                                       "sem_W": 0.0, "n": 60}, draws=200)
        reported.append(out["dynamic_energy_sigma_mj"])
    observed = float(np.std(estimates))
    assert abs(observed / analytical - 1) < 0.05, (observed, analytical)
    # The block term is estimated from 10 blocks, so its average is slightly low.
    assert abs(np.mean(reported) / analytical - 1) < 0.05, (np.mean(reported), analytical)
    print(f"  true spread {observed:.4f}, analytical {analytical:.4f}, "
          f"mean reported {np.mean(reported):.4f} mJ")


def test_bootstrap_interval_covers_about_95_percent():
    rng = np.random.default_rng(3)
    true = (7.0 - 6.0) * 8.0
    hits, n = 0, 300
    for _ in range(n):
        P = rng.normal(7.0, 0.2, 10)
        t = np.full(10, 8.0)
        idle_drawn = rng.normal(6.0, 0.1)
        out = dynamic_energy_uncertainty(P * t, P, t, {"mean_W": idle_drawn, "std_W": 0.1,
                                                       "sem_W": 0.0, "n": 60}, draws=1000)
        lo, hi = out["dynamic_energy_ci95_mj"]
        hits += lo <= true <= hi
    assert 0.88 <= hits / n <= 0.99, hits / n
    print(f"  95 % interval covers the true value in {100 * hits / n:.0f} % of runs")


def test_matches_paper_side_analysis():
    """The R0 Jetson GPU blocks at 16 % give the paper's dynamic energy."""
    E = [56.038, 56.779, 57.311, 56.250, 55.944, 56.753, 56.261, 56.488, 56.760, 56.169]
    P = [7.00, 7.04, 7.16, 7.04, 7.00, 7.08, 7.08, 7.06, 7.06, 7.08]
    t = [8.01, 8.07, 8.01, 7.99, 8.00, 8.02, 7.95, 8.00, 8.04, 7.94]
    out = dynamic_energy_uncertainty(E, P, t, {"mean_W": 6.16, "std_W": 0.09, "sem_W": 0.0, "n": 1})
    expected = (np.mean(P) - 6.16) * np.mean(t)
    assert math.isclose(out["dynamic_energy_pt_mj"], expected, rel_tol=1e-12)
    assert math.isclose(out["dynamic_energy_sigma_idle_mj"], 0.09 * np.mean(t), rel_tol=1e-12)
    assert 0.85 < out["idle_share_of_energy"] < 0.89, "paper: idle is ~87 % of GPU energy"
    print(f"  E_dyn {expected:.3f} mJ, idle share {out['idle_share_of_energy']:.0%}, as in R2.2")


def test_missing_baseline_leaves_dynamic_fields_empty():
    blocks = {"energy_mj": [56.0, 57.0], "power_W": [7.0, 7.1], "time_ms": [8.0, 8.03]}
    results = {"energy_mean_mj": 56.5, "energy_mean_watts": 7.05, "energy_std_mj": 0.5,
               "energy_blocks": blocks, "energy_backends": [GPU]}
    before = dict(results)
    added = add_dynamic_uncertainty(results, None, "inference")
    assert not any(k.startswith("dynamic_") for k in results), "no dynamic field without baseline"
    for k, v in before.items():
        assert results[k] == v, f"{k} changed"
    assert set(added) == {"energy_n_blocks", "energy_block_mean_mj", "energy_block_sem_mj"}
    assert all(v is None for v in derive_dynamic_metrics(56.5, 7.05, None).values())
    print("  totals untouched, dynamic fields absent, block statistics kept")


def test_each_stage_subtracts_its_own_backends():
    base = _baseline([10.0, 10.2, 9.8, 10.0], [5.0, 5.4, 4.6, 5.0])
    cpu_only = base.idle_for([CPU])
    both = base.idle_for([GPU, CPU])
    assert math.isclose(cpu_only["mean_W"], 5.0) and math.isclose(both["mean_W"], 15.0)
    assert math.isclose(both["std_W"], math.sqrt(np.var([10, 10.2, 9.8, 10], ddof=1)
                                                 + np.var([5, 5.4, 4.6, 5], ddof=1)))
    results = {
        "energy_reconstruction_blocks": {"energy_mj": [60.0] * 3, "power_W": [6.0] * 3,
                                         "time_ms": [10.0] * 3},
        "energy_reconstruction_backend_names": [CPU],
    }
    add_dynamic_uncertainty(results, base, "reconstruction")
    assert math.isclose(results["dynamic_reconstruction_energy_pt_mj"], (6.0 - 5.0) * 10.0)
    assert results["energy_reconstruction_idle_backends"] == [CPU]
    meta = base.to_dict()
    assert meta["n_idle_samples"] == 4 and meta["P_idle_std_w"] > 0
    assert meta["per_backend"][0]["P_idle_sem_w"] == meta["per_backend"][0]["P_idle_std_w"] / 2
    print("  a CPU reconstruction subtracts only the CPU idle power")


def test_rapl_counter_wrap():
    rapl = RAPLBackend()
    rapl._max_energy = {"package-0": 65532610987}
    assert rapl._delta_uj("package-0", 100, 300) == 200
    assert rapl._delta_uj("package-0", 65532610900, 13) == 87 + 13
    print("  increments across the counter wrap are positive")


def test_remeasure_loads_pre_refactor_unet():
    """Re-measure rebuilds U-Nets saved with the ConvTranspose decoder (before 4eb6f44)."""
    import logging
    import tempfile
    from pathlib import Path

    import torch

    # The ui package sets matplotlib's Qt backend on import. Without a display (over
    # SSH, e.g. on a Jetson) matplotlib accepts it only once a Qt application exists.
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])

    from simulation_engine._4_postprocessor.models.unet import UNet
    from simulation_engine._4_postprocessor.models.unet_transpose import UNetTranspose
    from ui.custom_widgets.batch_reports.remeasure.remeasure_worker import _build_model_facade

    log = logging.getLogger("test")
    entry = {"name": "Old 16%", "model_name": "u-net",
             "architecture_config": {"features": [8, 16, 32, 64]}}
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "models").mkdir()
        for cls in (UNetTranspose, UNet):
            net = cls(features=[8, 16, 32, 64]).eval()
            torch.save(net.state_dict(), Path(tmp) / "models" / "Old_16%.pt")
            facade = _build_model_facade(entry, 64, Path(tmp), use_gpu=False, logger=log)
            assert type(facade.model) is cls, type(facade.model)
            x = torch.rand(1, 1, 64, 64)
            with torch.no_grad():
                assert torch.allclose(facade.model(x), net(x))
    print("  both U-Net decoders load and give the saved network's output")


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
