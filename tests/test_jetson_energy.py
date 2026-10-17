"""Checks for the Jetson INA3221 energy backend and the platform-state record.

The repository has no test framework (see CLAUDE.md), so this file is a
self-contained script: run it directly with the project venv.

    .venv/bin/python tests/test_jetson_energy.py

No real sensor is read: the INA3221 sysfs tree is a temporary directory whose
voltage and current files a writer thread rewrites in place.
"""

import logging
import math
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch

from simulation_engine._5_analyzer.analyzer_energy import EnergyAnalyzer
from simulation_engine._5_analyzer.baseline_capture import capture_idle_baseline
from simulation_engine._5_analyzer.energy_backends._jetson import JetsonSysfsBackend
from simulation_engine._5_analyzer.energy_backends._monitor import EnergyMonitor
from simulation_engine._5_analyzer.platform_state import platform_state

LOG = logging.getLogger("test")


class FakeINA3221:
    """A sysfs-like INA3221 tree: ``<root>/1-0040/hwmon/hwmon9/in{N}_*`` and ``curr{N}_input``."""

    def __init__(self, root: Path, rails: dict[str, tuple[int, int]]):
        self.dir = root / "1-0040" / "hwmon" / "hwmon9"
        self.dir.mkdir(parents=True)
        self.fds = {}
        for n, (label, (mv, ma)) in enumerate(rails.items(), start=1):
            (self.dir / f"in{n}_label").write_text(label + "\n")
            for name, value in ((f"in{n}_input", mv), (f"curr{n}_input", ma)):
                path = self.dir / name
                path.write_text("")
                self.fds[(label, name[:2])] = os.open(path, os.O_RDWR)
                self.set(label, name[:2], value)

    def set(self, label: str, kind: str, value: int) -> None:
        """Rewrite one value in place (fixed width, so a reader never sees a short file)."""
        os.pwrite(self.fds[(label, kind)], b"%-12d\n" % value, 0)

    def close(self):
        for fd in self.fds.values():
            os.close(fd)


def _backend(root: Path) -> JetsonSysfsBackend:
    b = JetsonSysfsBackend(logger=LOG, driver_paths=[root], require_jetson=False)
    assert b.initialize(), "the fake tree was not found"
    return b


def _monitor(backend) -> EnergyMonitor:
    m = EnergyMonitor(logger=LOG)
    m._backends = [backend]
    m._is_initialized = True
    return m


def test_module_input_rail_is_integrated_alone():
    with tempfile.TemporaryDirectory() as tmp:
        fake = FakeINA3221(Path(tmp), {"VDD_IN": (5000, 1200), "VDD_CPU_GPU_CV": (5000, 200),
                                       "VDD_SOC": (5000, 300)})
        b = _backend(Path(tmp))
        try:
            assert b.rail_label == "VDD_IN", b.rail_label
            assert b.rails_available == ["VDD_CPU_GPU_CV", "VDD_IN", "VDD_SOC"], b.rails_available
            assert math.isclose(b.get_current_power(), 6.0), "sub-rails must not be added to VDD_IN"
        finally:
            b.shutdown()
            fake.close()
    print("  VDD_IN chosen; the sub-rails are listed but not added")


def test_without_an_input_rail_the_channels_are_summed():
    with tempfile.TemporaryDirectory() as tmp:
        fake = FakeINA3221(Path(tmp), {"VDD_GPU_SOC": (5000, 400), "VDD_CPU_CV": (5000, 600)})
        b = _backend(Path(tmp))
        try:
            assert b.rail_label == "VDD_GPU_SOC+VDD_CPU_CV" or b.rail_label == "VDD_CPU_CV+VDD_GPU_SOC"
            assert math.isclose(b.get_current_power(), 5.0)
        finally:
            b.shutdown()
            fake.close()
    print("  no VDD_IN: the channels are summed and the label says so")


def test_constant_power_integrates_with_many_samples():
    with tempfile.TemporaryDirectory() as tmp:
        fake = FakeINA3221(Path(tmp), {"VDD_IN": (5000, 1200)})
        b = _backend(Path(tmp))
        try:
            b.start_measurement()
            time.sleep(0.5)
            r = b.stop_measurement()
        finally:
            b.shutdown()
            fake.close()
    assert math.isclose(r.avg_power_watts, 6.0, rel_tol=1e-9), r.avg_power_watts
    assert math.isclose(r.energy_joules, 6.0 * r.duration_seconds, rel_tol=1e-9)
    assert r.n_samples >= 50, f"only {r.n_samples} samples in 0.5 s"
    assert 100 <= r.sampling_hz <= 210, r.sampling_hz
    print(f"  6 W over {r.duration_seconds:.3f} s: {r.n_samples} samples, {r.sampling_hz:.0f} Hz")


def test_a_power_step_inside_the_window_is_resolved():
    # The two-point estimate of the jtop backend would give (6 + 10) / 2 = 8 W
    # whatever the step's position; the sampled integral must follow the step.
    with tempfile.TemporaryDirectory() as tmp:
        fake = FakeINA3221(Path(tmp), {"VDD_IN": (5000, 1200)})
        b = _backend(Path(tmp))
        try:
            b.start_measurement()
            time.sleep(0.15)
            fake.set("VDD_IN", "cu", 2000)
            time.sleep(0.45)
            r = b.stop_measurement()
        finally:
            b.shutdown()
            fake.close()
    expected = (6.0 * 0.15 + 10.0 * 0.45) / 0.60
    assert abs(r.avg_power_watts - expected) < 0.25, (r.avg_power_watts, expected)
    print(f"  6 W then 10 W: mean {r.avg_power_watts:.2f} W (expected {expected:.2f}, two-point 8.00)")


def test_the_sampler_pauses_between_measurements():
    with tempfile.TemporaryDirectory() as tmp:
        fake = FakeINA3221(Path(tmp), {"VDD_IN": (5000, 1200)})
        b = _backend(Path(tmp))
        try:
            assert not b._run.is_set(), "sampling before any measurement"
            b.start_measurement()
            assert b._run.is_set()
            b.stop_measurement()
            assert not b._run.is_set(), "still sampling after the measurement"
        finally:
            b.shutdown()
            fake.close()
    print("  the thread runs only inside a measurement (not during the timing pass)")


def test_baseline_samples_are_one_second_means():
    with tempfile.TemporaryDirectory() as tmp:
        fake = FakeINA3221(Path(tmp), {"VDD_IN": (5000, 1200)})
        b = _backend(Path(tmp))
        stop = threading.Event()

        def flicker():   # 5 W and 7 W alternating every 10 ms: mean 6 W
            high = False
            while not stop.is_set():
                fake.set("VDD_IN", "cu", 1400 if high else 1000)
                high = not high
                time.sleep(0.01)

        writer = threading.Thread(target=flicker, daemon=True)
        writer.start()
        try:
            base = capture_idle_baseline(_monitor(b), duration_s=3, logger=LOG)
        finally:
            stop.set()
            writer.join()
            b.shutdown()
            fake.close()
    vals = base.per_backend[b.device_name].samples_W
    assert len(vals) == 3, vals
    assert all(abs(v - 6.0) < 0.15 for v in vals), vals
    assert not b._run.is_set(), "the baseline left the sampler running"
    print(f"  3 s baseline of a 5/7 W flicker: {[round(v, 3) for v in vals]} W "
          f"(instantaneous readings would scatter by 1 W)")


def test_inference_blocks_record_their_samples():
    with tempfile.TemporaryDirectory() as tmp:
        fake = FakeINA3221(Path(tmp), {"VDD_IN": (5000, 1200)})
        b = _backend(Path(tmp))
        try:
            net = torch.nn.Conv2d(1, 1, 3, padding=1).eval()
            an = EnergyAnalyzer(model=net, device="cpu", warmup_runs=2, measurement_runs=20, logger=LOG)
            an._energy_monitor = _monitor(b)
            an._is_initialized = True
            x = torch.rand(1, 1, 32, 32)

            def slow(_):   # ~5 ms per call, so a block of 20 lasts ~0.1 s
                time.sleep(0.005)
                return net(x)

            an.model = slow
            an.model.eval = lambda: None
            res = an.analyze_inference([x] * 3, n_runs=20, warmup_runs=2, return_per_image=False)
        finally:
            b.shutdown()
            fake.close()
    assert res.samples_per_block is not None and len(res.samples_per_block) == 3, res.samples_per_block
    assert all(n >= 8 for n in res.samples_per_block), res.samples_per_block
    assert res.sampling_hz is not None and res.sampling_hz > 80, res.sampling_hz
    # Energy scaled to the timed loop: the mean power is the sensor's, exactly
    assert abs(res.mean_power_watts - 6.0) < 1e-9, res.mean_power_watts
    print(f"  3 blocks of ~0.1 s: {res.samples_per_block} samples, {res.sampling_hz:.0f} Hz")


def test_platform_state_snapshot():
    s = platform_state()
    for key in ("time", "hostname", "kernel", "python", "cpu", "temperatures_c"):
        assert key in s, key
    assert "policies" in s["cpu"]
    if Path("/etc/nv_tegra_release").exists():
        assert "jetson" in s and "nvpmodel" in s["jetson"], s.get("jetson")
    print(f"  {s['hostname']}: {len(s['cpu']['policies'])} CPU policies, "
          f"{len(s['temperatures_c'])} thermal zones, "
          f"{'Jetson ' + str(s['jetson'].get('nvpmodel')) if 'jetson' in s else 'not a Jetson'}")


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
