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
:class:`EnergyMonitor` and asks each backend for its power reading on
a 1 Hz schedule. Backends sample at their own native rate (NVML
~10 Hz, RAPL ~1 kHz internally) — the 1 Hz outer poll is a deliberate
compromise: high enough to catch short transients, low enough that
the polling python loop itself stays well under 1 % CPU and doesn't
bias the very baseline it's trying to measure. A backend that samples
on its own (the Jetson INA3221 backend, 200 Hz) keeps sampling for the
whole window and returns the mean since the previous poll, so each of
its baseline samples is a 1 s mean; the first poll then waits one
interval.

The "10.752 kHz" figure that lives elsewhere in the codebase is
the **DMD sampling rate** for the simulated single-pixel
acquisition; it has nothing to do with how the energy sensor is
read and must not be reused here.
"""
from __future__ import annotations

import logging
import math
import statistics
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

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

    @property
    def sample_std_W(self) -> float:
        """Sample standard deviation (n - 1) of the idle readings.

        ``power_W_std`` is the population form and keeps its historical
        meaning; the uncertainty propagation uses this one.
        """
        if len(self.samples_W) > 1:
            return float(statistics.stdev(self.samples_W))
        return float(self.power_W_std)

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form for the batch report metadata."""
        n = int(self.n_samples)
        std = self.sample_std_W
        return {
            "backend":      self.backend,
            "power_W_mean": float(self.power_W_mean),
            "power_W_std":  float(self.power_W_std),
            "duration_s":   float(self.duration_s),
            "n_samples":    n,
            # Uncertainty fields (R2.2): sample std of the idle power, its
            # standard error, and the sample count.
            "P_idle_mean_w":  float(self.power_W_mean),
            "P_idle_std_w":   std,
            "P_idle_sem_w":   std / math.sqrt(n) if n > 0 else None,
            "n_idle_samples": n,
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

    def idle_for(self, backends: Iterable[str] | None = None) -> dict[str, float] | None:
        """Idle power of the backends that took part in a measurement.

        A measurement must subtract the idle draw of exactly the backends
        it integrated: a reconstruction on the CPU is read by RAPL alone and
        must not lose the GPU's idle power. ``backends`` are device names as
        the energy backends report them; ``None`` means all of them. Means
        add, sample standard deviations add in quadrature (independent
        sensors), and ``n`` is the smallest sample count. Returns ``None``
        when none of the requested backends was captured.
        """
        if backends is None:
            chosen = list(self.per_backend.values())
        else:
            wanted = set(backends)
            chosen = [b for name, b in self.per_backend.items() if name in wanted]
        if not chosen:
            return None
        mean = float(sum(b.power_W_mean for b in chosen))
        std = float(math.sqrt(sum(b.sample_std_W ** 2 for b in chosen)))
        n = int(min(b.n_samples for b in chosen))
        return {
            "mean_W": mean,
            "std_W": std,
            "sem_W": std / math.sqrt(n) if n > 0 else float("nan"),
            "n": n,
            "backends": [b.backend for b in chosen],
        }

    def to_dict(self) -> dict[str, Any]:
        total = self.idle_for(None) or {}
        return {
            "total_power_W":         self.total_power_W,
            "total_power_std_W":     self.total_power_std_W,
            "duration_s":            self.duration_s,
            "requested_duration_s":  self.requested_duration_s,
            "per_backend": [b.to_dict() for b in self.per_backend.values()],
            # Uncertainty fields (R2.2), summed over backends.
            "P_idle_mean_w":  total.get("mean_W"),
            "P_idle_std_w":   total.get("std_W"),
            "P_idle_sem_w":   total.get("sem_W"),
            "n_idle_samples": total.get("n"),
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
    continuous = monitor.start_sampling()
    t0 = time.perf_counter()
    # A continuously sampling backend reports the mean since the previous
    # poll: the first poll comes one interval in, so it averages a full one.
    next_tick = t0 + (_BASELINE_POLL_INTERVAL_S if continuous else 0.0)
    try:
        while True:
            delay = max(0.0, next_tick - time.perf_counter())
            time.sleep(delay)
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
    finally:
        monitor.stop_sampling()

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


# ---------------------------------------------------------------------------
# Uncertainty of the dynamic energy (R2.2)
# ---------------------------------------------------------------------------

BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 0


def dynamic_energy_uncertainty(
    block_energy_mj,
    block_power_w,
    block_time_ms,
    idle: dict[str, float] | None,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    r"""Dynamic energy per image with its uncertainty, from per-block readings.

    The measurement is split into :math:`B` blocks. Block :math:`b` gives a
    mean power :math:`P_b` and a time per image :math:`t_b`
    (:math:`E_b = P_b t_b` is its energy per image). With
    :math:`\bar P` and :math:`\bar t` the block means and
    :math:`P_\text{idle} \pm \sigma_\text{idle}` the idle baseline of the
    backends that took the measurement,

    .. math::

        E_\text{dyn} = (\bar P - P_\text{idle})\,\bar t .

    Two uncertainty components are reported separately and combined in
    quadrature (they are independent):

    * **idle baseline**: :math:`\sigma_{E,\text{idle}} = \sigma_\text{idle}\,\bar t`,
      with :math:`\sigma_\text{idle}` the *standard deviation* of the idle
      readings, not their standard error. The concern is the baseline drifting
      between the idle window and the run, which the standard error would
      hide. The standard-error version, :math:`\sigma_\text{idle}/\sqrt{n}\,\bar t`,
      is stored as well for reference;
    * **blocks**: first-order (delta-method) propagation of the sampling error
      of :math:`\bar P` and :math:`\bar t`, including their covariance,

      .. math::

          \sigma_{E,\text{blocks}}^2 = \frac{1}{B}\left[\bar t^{\,2} s_P^2
          + (\bar P - P_\text{idle})^2 s_t^2
          + 2\,\bar t\,(\bar P - P_\text{idle})\,s_{Pt}\right],

      with sample (co)variances over the blocks.

    :math:`\sigma_E = \sqrt{\sigma_{E,\text{idle}}^2 + \sigma_{E,\text{blocks}}^2}`.
    A 95 % interval comes from a bootstrap that resamples the blocks with
    replacement and draws :math:`P_\text{idle} \sim N(P_\text{idle},
    \sigma_\text{idle})` each time; this is the procedure of the paper-side
    analysis, so both give the same numbers. The seed is fixed.

    Energies in mJ, powers in W, times in ms. Without a baseline only the
    block statistics of the total energy are returned.
    """
    E = np.asarray(block_energy_mj, dtype=float)
    P = np.asarray(block_power_w, dtype=float)
    t = np.asarray(block_time_ms, dtype=float)
    B = int(min(E.size, P.size, t.size))
    out: dict[str, Any] = {"n_blocks": B}
    if B == 0:
        return out
    E, P, t = E[:B], P[:B], t[:B]
    out["energy_block_mean_mj"] = float(E.mean())
    out["energy_block_sem_mj"] = float(E.std(ddof=1) / math.sqrt(B)) if B > 1 else None
    if idle is None:
        return out

    p_idle, s_idle = float(idle["mean_W"]), float(idle["std_W"])
    p_bar, t_bar = float(P.mean()), float(t.mean())
    p_dyn = p_bar - p_idle
    e_dyn = p_dyn * t_bar
    s_idle_term = s_idle * t_bar
    sem_idle = float(idle.get("sem_W", float("nan")))
    if B > 1:
        cov = np.cov(np.vstack([P, t]), ddof=1)
        var_blocks = (t_bar ** 2 * cov[0, 0] + p_dyn ** 2 * cov[1, 1]
                      + 2 * t_bar * p_dyn * cov[0, 1]) / B
        s_blocks = float(math.sqrt(max(var_blocks, 0.0)))
        s_p_blocks = float(math.sqrt(cov[0, 0] / B))
    else:
        s_blocks, s_p_blocks = 0.0, 0.0

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, B, size=(draws, B))
    idle_draws = rng.normal(p_idle, s_idle, size=draws)
    boot = (P[idx].mean(axis=1) - idle_draws) * t[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])

    out.update({
        "idle_power_W": p_idle,
        "idle_power_std_W": s_idle,
        "idle_power_sem_W": sem_idle,
        "n_idle_samples": int(idle.get("n", 0)),
        "idle_backends": list(idle.get("backends", [])),
        "dynamic_power_block_W": p_dyn,
        "dynamic_power_sigma_W": float(math.sqrt(s_idle ** 2 + s_p_blocks ** 2)),
        "dynamic_energy_pt_mj": e_dyn,
        "dynamic_energy_sigma_idle_mj": s_idle_term,
        "dynamic_energy_sigma_idle_sem_mj": sem_idle * t_bar,
        "dynamic_energy_sigma_blocks_mj": s_blocks,
        "dynamic_energy_sigma_mj": float(math.sqrt(s_idle_term ** 2 + s_blocks ** 2)),
        "dynamic_energy_ci95_mj": [float(lo), float(hi)],
        "idle_share_of_energy": float(p_idle * t_bar / E.mean()) if E.mean() > 0 else None,
    })
    return out


def uncertainty_fields(stage: str, stats: dict[str, Any]) -> dict[str, Any]:
    """Report names for the output of :func:`dynamic_energy_uncertainty`.

    ``stage`` is ``"inference"`` (``dynamic_energy_sigma_mj``,
    ``energy_idle_power_W``, ...) or ``"reconstruction"``
    (``dynamic_reconstruction_energy_sigma_mj``,
    ``energy_reconstruction_idle_power_W``, ...), so the two stages never
    overwrite each other and no existing field changes.
    """
    lead = "energy_" if stage == "inference" else f"energy_{stage}_"
    out = {}
    for key, value in stats.items():
        if key.startswith("dynamic_"):
            name = key if stage == "inference" else f"dynamic_{stage}_" + key[len("dynamic_"):]
        elif key.startswith("energy_"):
            name = key if stage == "inference" else lead + key[len("energy_"):]
        else:
            name = lead + key
        out[name] = value
    return out


def add_dynamic_uncertainty(results: dict[str, Any], baseline: BaselineResult | None,
                            stage: str = "inference") -> dict[str, Any]:
    """Add the uncertainty fields of one stage to a test's ``results``.

    Uses the per-block readings the energy measurement stored
    (``energy_blocks`` / ``energy_reconstruction_blocks``) and the idle
    baseline of the backends that took that measurement. Without a baseline
    the dynamic fields are left out and the total fields are untouched.
    Returns the fields that were added.
    """
    key, names = (("energy_blocks", "energy_backends") if stage == "inference"
                  else ("energy_reconstruction_blocks", "energy_reconstruction_backend_names"))
    blocks = results.get(key)
    if not blocks:
        return {}
    idle = baseline.idle_for(results.get(names)) if baseline is not None else None
    stats = dynamic_energy_uncertainty(blocks.get("energy_mj", []), blocks.get("power_W", []),
                                       blocks.get("time_ms", []), idle)
    fields = uncertainty_fields(stage, stats)
    results.update(fields)
    return fields
