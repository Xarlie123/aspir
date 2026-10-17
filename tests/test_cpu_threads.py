"""Checks for the CPU thread-pool setting (torch and BLAS threads of the CPU passes).

The repository has no test framework (see CLAUDE.md), so this file is a
self-contained script: run it directly with the project venv.

    .venv/bin/python tests/test_cpu_threads.py

It runs on any machine: the BLAS pool it sets is whichever the process has
loaded (numpy's OpenBLAS copy on a desktop; the system OpenBLAS that torch
calls on a Jetson, as well).
"""

import json
import logging
import os
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import torch
from PySide6.QtWidgets import QApplication

import simulation_engine._5_analyzer.cpu_threads as ct
from simulation_engine._5_analyzer.platform_state import platform_state

# The ui package sets matplotlib's Qt backend on import. Without a display (over
# SSH, e.g. on a Jetson) matplotlib accepts it only once a Qt application exists.
_APP = QApplication.instance() or QApplication([])

from ui.custom_widgets.batch_reports.remeasure.remeasure_worker import RemeasureConfig, _make_test_config
from ui.custom_widgets.batch_test.batch_test_runner._timing import measure_timing
from ui.custom_widgets.batch_test.test_config_model import BatchTestConfig, TestConfiguration

LOG = logging.getLogger("test_cpu_threads")


def _blas_threads():
    return [p["threads"] for p in ct.blas_pools() if p["user_api"] == "blas"]


def test_configuration_round_trip():
    assert (TestConfiguration().cpu_torch_threads, TestConfiguration().cpu_blas_threads) == (None, None)
    assert TestConfiguration.from_dict({}).cpu_torch_threads is None, "unset must mean the library default"
    cfg = TestConfiguration(name="t", cpu_torch_threads=1, cpu_blas_threads=8)
    back = TestConfiguration.from_dict(json.loads(json.dumps(cfg.to_dict())))
    assert (back.cpu_torch_threads, back.cpu_blas_threads) == (1, 8)
    assert TestConfiguration.from_dict({"cpu_torch_threads": 0}).cpu_torch_threads is None
    try:
        TestConfiguration.from_dict({"cpu_blas_threads": -2})
        raise AssertionError("a negative thread count was accepted")
    except ValueError:
        pass
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "b.batch_config")
        BatchTestConfig(name="b", tests=[cfg]).save(path)
        loaded = BatchTestConfig.load(path).tests[0]
    assert (loaded.cpu_torch_threads, loaded.cpu_blas_threads) == (1, 8)
    remeasured = _make_test_config({"name": "x"}, RemeasureConfig(cpu_torch_threads=2, cpu_blas_threads=1))
    assert (remeasured.cpu_torch_threads, remeasured.cpu_blas_threads) == (2, 1)
    print("  .batch_config and re-measure carry the setting; unset and 0 mean the library default")


def test_the_gui_saves_and_shows_the_setting():
    from ui.custom_widgets.batch_test.test_config_widget import TestConfigWidget

    widget = TestConfigWidget()
    widget.set_config(TestConfiguration(name="t", cpu_torch_threads=1, cpu_blas_threads=4))
    assert (widget.cpu_torch_threads_spin.value(), widget.cpu_blas_threads_spin.value()) == (1, 4)
    widget.cpu_torch_threads_spin.setValue(0)
    widget.cpu_blas_threads_spin.setValue(2)
    cfg = widget._config
    assert (cfg.cpu_torch_threads, cfg.cpu_blas_threads) == (None, 2), (cfg.cpu_torch_threads, cfg.cpu_blas_threads)
    assert widget.cpu_torch_threads_spin.text() == "Default"

    from ui.custom_widgets.batch_reports.remeasure.remeasure_dialog import _thread_spin
    from ui.custom_widgets.timing_analysis.timing_analysis_page import TimingAnalysisPage
    spin = _thread_spin("tip")
    assert spin.value() == 0 and spin.text() == "Default"
    page = TimingAnalysisPage()
    assert (page.cpu_torch_threads, page.cpu_blas_threads) == (None, None)
    page.cpu_torch_threads_spinbox.setValue(1)
    page.cpu_blas_threads_spinbox.setValue(8)
    assert (page.cpu_torch_threads, page.cpu_blas_threads) == (1, 8)
    print("  Batch Test widget, Re-measure dialog and Single Test timing: both counts, 0 shows as Default")


def test_settings_apply_inside_and_restore_after():
    before_torch, before_blas = torch.get_num_threads(), _blas_threads()
    assert before_blas, "no BLAS pool visible to threadpoolctl"
    with ct.cpu_threads(torch_threads=1, blas_threads=1, logger=LOG) as state:
        assert torch.get_num_threads() == 1
        assert set(_blas_threads()) == {1}, _blas_threads()
        assert state["torch_intra_op_threads"] == 1
        assert {p["threads"] for p in state["blas_pools"] if p["user_api"] == "blas"} == {1}
        assert state["requested"] == {"torch_threads": 1, "blas_threads": 1}
    assert torch.get_num_threads() == before_torch
    assert _blas_threads() == before_blas, (_blas_threads(), before_blas)
    with ct.cpu_threads() as state:   # unset: nothing changes, the state is still recorded
        assert torch.get_num_threads() == before_torch and _blas_threads() == before_blas
        assert state["requested"] == {"torch_threads": None, "blas_threads": None}
    print(f"  torch {before_torch} -> 1 -> {before_torch}; BLAS {before_blas} -> 1 -> {before_blas}")


def test_a_blas_setting_that_cannot_apply_fails_loudly():
    saved = ct.threadpoolctl
    ct.threadpoolctl = None
    try:
        with ct.cpu_threads(blas_threads=1):
            pass
        raise AssertionError("a BLAS count was reported without threadpoolctl")
    except RuntimeError:
        pass
    finally:
        ct.threadpoolctl = saved
    print("  without threadpoolctl a BLAS count raises instead of being recorded")


def test_oversubscription_rule():
    def pool(threads, library="openblas", layer="pthreads", torch_uses=True):
        return {"user_api": "blas", "library": library, "version": "0.3.20", "threads": threads,
                "threading_layer": layer, "path": "/usr/lib/libopenblasp-r0.3.20.so", "used_by_torch": torch_uses}

    rule = ct.oversubscription
    # Jetson torch wheel (BLAS_INFO=open): R1.3e 8 x 8, 8 x 4, 8 x 2 slow; 8 x 1, 1 x 8, 1 x 1 not
    for torch_threads, blas_threads, expected in ((8, 8, True), (4, 8, True), (2, 8, True),
                                                  (1, 8, False), (8, 1, False), (1, 1, False)):
        got = rule(torch_threads, [pool(blas_threads)], 8, "open") is not None
        assert got == expected, (torch_threads, blas_threads, got)
    assert rule(8, [pool(16, torch_uses=False)], 16, "mkl") is None, "numpy's own copy is not torch's"
    assert rule(8, [pool(8, layer="openmp")], 8, "open") is None, "an OpenMP OpenBLAS shares torch's pool"
    assert rule(8, [], 4, "mkl") is not None, "more torch threads than usable cores"
    print("  flags torch > 1 with a separate pthreads OpenBLAS > 1, or more threads than cores")


def test_platform_state_records_the_pools():
    s = platform_state()["cpu_threading"]
    for key in ("torch_intra_op_threads", "torch_inter_op_threads", "torch_blas", "blas_pools", "environment",
                "usable_cores", "oversubscribed", "threadpoolctl"):
        assert key in s, key
    assert set(s["environment"]) == {"OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"}
    print(f"  torch BLAS {s['torch_blas']}, {len(s['blas_pools'])} pools, {s['usable_cores']} cores, "
          f"oversubscribed {s['oversubscribed']}")


def test_cpu_timing_runs_with_the_setting_and_reports_it():
    net = torch.nn.Sequential(torch.nn.Conv2d(1, 4, 3, padding=1), torch.nn.ReLU(),
                              torch.nn.Conv2d(4, 1, 3, padding=1)).eval()
    post = SimpleNamespace(model=net, img_size=32, is_conv=True, device=torch.device("cpu"), applicator=None)
    before_torch, before_blas = torch.get_num_threads(), _blas_threads()
    cfg = TestConfiguration(name="t", use_gpu=False, cpu_torch_threads=1, cpu_blas_threads=1)
    res = measure_timing(post, cfg, SimpleNamespace(data=[]), LOG, warmup_runs=2, measurement_runs=5)
    st = res["cpu_threads"]
    assert st["torch_intra_op_threads"] == 1 and {p["threads"] for p in st["blas_pools"]
                                                  if p["user_api"] == "blas"} == {1}
    assert res["cpu_oversubscribed"] is False
    assert torch.get_num_threads() == before_torch and _blas_threads() == before_blas, "not restored"
    print(f"  timing pass ran with torch 1 x BLAS 1 ({res['timing_cpu_mean_ms']:.2f} ms) and restored the pools")


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
