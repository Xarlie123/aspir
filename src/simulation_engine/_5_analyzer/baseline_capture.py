"""Idle-power baseline capture for energy measurements.

Single-pass measurements on the Jetson rail (or any other backend
that reads a shared SoC power sensor) include the system's idle
overhead — desktop, GUI, background services — which contaminates
the per-test "energy of inference" reading. To get a fair "dynamic
energy" we capture a short window of idle power before the batch
starts and subtract it from each test's average power × inference
time. This module owns the capture logic; the subtraction lives in
the report serializer so the totals on disk stay untouched.

The module is hardware-agnostic: it consumes an already-initialised
:class:`EnergyMonitor` and asks each backend for its instantaneous
power reading on a 1 Hz schedule. Backends sample at their own
native rate (jtop ~1 Hz, NVML ~10 Hz, RAPL ~1 kHz internally) — the
1 Hz outer poll is a deliberate compromise: high enough to catch
short transients, low enough that the polling python loop itself
stays well under 1 % CPU and doesn't bias the very baseline it's
trying to measure.

The "10.752 kHz" figure that lives elsewhere in the codebase is
the **DMD sampling rate** for the simulated single-pixel
acquisition; it has nothing to do with how the energy sensor is
read and must not be reused here.
"""
from __future__ import annotations

import logging
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from simulation_engine._5_analyzer.energy_backends._monitor import EnergyMonitor


# 1 Hz outer poll — see module docstring for the rationale.
_BASELINE_POLL_INTERVAL_S = 1.0


@dataclass
class BackendBaseline:
    """Baseline statistics for one energy backend."""

    backend: str                     # device_name as the backend reported it
    power_W_mean: float
    power_W_std: float
    duration_s: float
    n_samples: int
    samples_W: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form for the batch report metadata."""
        return {
            "backend":      self.backend,
            "power_W_mean": float(self.power_W_mean),
            "power_W_std":  float(self.power_W_std),
            "duration_s":   float(self.duration_s),
            "n_samples":    int(self.n_samples),
        }


@dataclass
class BaselineResult:
    """Aggregated baseline result for the whole pre-batch window.

    ``per_backend`` maps device-name → :class:`BackendBaseline`. The
    ``total_*`` fields sum the per-backend means / quadrature-add the
    stds, so callers that want a single representative number for the
    "system idle" (e.g. the dynamic-power column on the per-test rows)
    don't have to walk the dict.
    """

    per_backend: dict[str, BackendBaseline]
    duration_s: float
    requested_duration_s: float

    @property
    def total_power_W(self) -> float:
        return float(sum(b.power_W_mean for b in self.per_backend.values()))

    @property
    def total_power_std_W(self) -> float:
        # Independent stds add in quadrature.
        return float(
            sum(b.power_W_std ** 2 for b in self.per_backend.values()) ** 0.5
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_power_W":         self.total_power_W,
            "total_power_std_W":     self.total_power_std_W,
            "duration_s":            self.duration_s,
            "requested_duration_s":  self.requested_duration_s,
            "per_backend": [b.to_dict() for b in self.per_backend.values()],
        }


def capture_idle_baseline(
    monitor: EnergyMonitor,
    duration_s: float,
    logger: Optional[logging.Logger] = None,
    progress_callback: Optional[Callable[[float, float], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Optional[BaselineResult]:
    """Sample the monitor's instantaneous power for ``duration_s``.

    Parameters
    ----------
    monitor:
        An ``EnergyMonitor`` that has already been ``initialize()``-d.
        If ``not monitor.is_initialized`` the function returns ``None``
        and logs a WARNING — the caller writes ``baseline = None`` to
        the report and the batch keeps running with ``dynamic_*``
        columns blank.
    duration_s:
        How long to sample, in seconds. Clamped to >= 1 s to avoid
        pathological zero windows.
    progress_callback:
        Optional ``(elapsed_s, total_s)`` callback fired once per
        outer poll; useful for the GUI progress bar.
    cancel_check:
        Optional ``() -> bool`` polled each tick. When it returns
        truthy the loop stops early and whatever samples were
        collected so far are returned (so a user-aborted baseline
        still produces useful numbers).

    Returns
    -------
    A :class:`BaselineResult` on success, ``None`` if the monitor is
    not initialised. A baseline that ran but collected zero samples
    (e.g. all backends returned 0 W or NaN) is still returned — its
    means will be 0 and the consumer is expected to surface that as
    a WARNING, not silently substitute a value.
    """
    log = logger or logging.getLogger(__name__)
    if not monitor.is_initialized:
        log.warning(
            "Idle baseline skipped: EnergyMonitor not initialised "
            "(no backend available)."
        )
        return None

    duration_s = max(1.0, float(duration_s))
    log.info(
        "Capturing idle baseline for %.1f s on backends: %s",
        duration_s, monitor.available_backends,
    )

    # Bucket sample lists per backend up-front so we don't lose data
    # if a backend reports for the first sample but fails on later
    # ones (the dict keeps every key the monitor yielded).
    samples: dict[str, list[float]] = {
        name: [] for name in monitor.available_backends
    }
    t0 = time.perf_counter()
    next_tick = t0
    while True:
        if cancel_check is not None and cancel_check():
            log.info("Idle baseline cancelled by caller after %.1f s",
                     time.perf_counter() - t0)
            break
        readings = monitor.get_current_power()
        for name, watts in readings.items():
            samples.setdefault(name, []).append(float(watts))
        elapsed = time.perf_counter() - t0
        if progress_callback is not None:
            progress_callback(elapsed, duration_s)
        if elapsed >= duration_s:
            break
        # Sleep until the next tick. Using a target time rather than a
        # fixed sleep keeps the cadence even under jitter from the
        # backend reads themselves.
        next_tick += _BASELINE_POLL_INTERVAL_S
        delay = max(0.0, next_tick - time.perf_counter())
        time.sleep(delay)

    real_duration = time.perf_counter() - t0
    per_backend: dict[str, BackendBaseline] = {}
    for name, vals in samples.items():
        if not vals:
            log.warning(
                "Idle baseline: backend '%s' produced no samples", name
            )
            mean = 0.0
            std = 0.0
        else:
            mean = float(sum(vals) / len(vals))
            std = float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0
        per_backend[name] = BackendBaseline(
            backend=name,
            power_W_mean=mean,
            power_W_std=std,
            duration_s=real_duration,
            n_samples=len(vals),
            samples_W=list(vals),
        )

    result = BaselineResult(
        per_backend=per_backend,
        duration_s=real_duration,
        requested_duration_s=duration_s,
    )
    log.info(
        "Idle baseline captured: total %.2f W ± %.2f W (sum across %d backend(s))",
        result.total_power_W, result.total_power_std_W, len(per_backend),
    )
    return result


def derive_dynamic_metrics(
    energy_mean_mj: Optional[float],
    energy_mean_watts: Optional[float],
    baseline_power_W: Optional[float],
) -> dict[str, Optional[float]]:
    """Compute the dynamic-power / dynamic-energy / dynamic-efficiency
    triple for a single test row.

    The inference time used here is derived from the energy phase's
    own integrated values (``time = energy_mj / energy_mean_watts``)
    so the subtraction is consistent with the window in which the
    energy was integrated — *assumes timing and energy share the
    iteration window*. The Timing analyzer reports
    ``timing_*_mean_ms`` from a separate loop in ``_timing.py``;
    using that here would mix two different windows and bias the
    result, so this function deliberately avoids it.

    Returns ``{"dynamic_power_W": …, "dynamic_energy_mj": …,
    "dynamic_efficiency_imgs_per_J": …}`` with values ``None`` for
    fields that can't be computed (e.g. baseline missing, or
    ``energy_mean_watts == 0``). Negative dynamics are returned as-is
    — they're a real diagnostic signal (test power < baseline,
    typically meaning thermal drift).
    """
    out: dict[str, Optional[float]] = {
        "dynamic_power_W": None,
        "dynamic_energy_mj": None,
        "dynamic_efficiency_imgs_per_J": None,
    }
    if baseline_power_W is None:
        return out
    if energy_mean_watts is None or energy_mean_mj is None:
        return out
    try:
        avg_w = float(energy_mean_watts)
        e_mj = float(energy_mean_mj)
        b_w = float(baseline_power_W)
    except (TypeError, ValueError):
        return out

    out["dynamic_power_W"] = avg_w - b_w

    # inference_time_s from the energy phase itself: same window as
    # the integrated energy, so the subtraction below is honest.
    if avg_w > 0:
        inference_time_s = e_mj / 1000.0 / avg_w
    else:
        inference_time_s = None

    if inference_time_s is not None:
        dyn_e = e_mj - b_w * inference_time_s * 1000.0
        out["dynamic_energy_mj"] = dyn_e
        if dyn_e > 0:
            out["dynamic_efficiency_imgs_per_J"] = 1000.0 / dyn_e
        # else: leave efficiency as None — float("nan") would propagate
        # into the report JSON as a non-standard NaN literal that
        # downstream loaders would choke on.

    return out
