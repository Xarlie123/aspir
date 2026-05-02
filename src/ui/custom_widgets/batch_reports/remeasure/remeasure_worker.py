"""Background worker that re-runs inference on saved experiments.

The worker iterates the tests of one or more ``.batch_analysis_report``
files, rebuilds each trained model from its checkpoint + architecture
metadata, and re-measures DNN inference timing and/or energy with the
existing :func:`measure_timing` / :func:`measure_energy` primitives. The
output is written to a *copy* of the report (the suffix is decided by the
caller); the originals are left untouched.

The full timing breakdown (acquisition + reconstruction + inference)
needs the saved masks and test images, so it only works on
``ALL_DATA`` exports. On ``REPORTS_AND_MODELS`` exports we still
re-measure inference timing and energy; acquisition time is
back-filled from the original ``timing_num_patterns`` so the table
isn't blank, and reconstruction is left at zero with a log warning.
"""
from __future__ import annotations

import copy
import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from PyQt5.QtCore import QObject, pyqtSignal

from simulation_engine._4_postprocessor.postprocessor_nn import (
    MODEL_REGISTRY,
    resolve_model_name,
)
from simulation_engine._5_analyzer.analyzer_energy import EnergyAnalyzer
from simulation_engine._5_analyzer.baseline_capture import (
    BaselineResult,
    capture_idle_baseline,
    derive_dynamic_metrics,
)
from ui.custom_widgets.batch_test.batch_test_runner._energy import measure_energy
from ui.custom_widgets.batch_test.batch_test_runner._pipeline import create_applicator
from ui.custom_widgets.batch_test.batch_test_runner._timing import measure_timing
from ui.custom_widgets.batch_test.test_config_model import TestConfiguration
from ui.utils.file_formats import safe_test_dirname


@dataclass
class RemeasureConfig:
    """User-controllable knobs for a re-measurement run."""

    measure_timing: bool = True
    measure_energy: bool = True
    use_gpu: bool = True
    # Unified app-wide defaults (Single Test → Timing, Batch Test
    # and the Re-measure dialog all start from these).
    warmup_runs: int = 20
    measurement_runs: int = 800
    sampling_rate_khz: float = 10.752
    device_label: str = "jetson"  # appears in the output filename
    # Idle-baseline capture before the first job. When False the
    # ``baseline_*`` fields stay blank in the output report and the
    # per-test ``dynamic_*`` columns come back as None, but every
    # other column is identical to a non-baseline run — the feature
    # only adds information.
    capture_baseline: bool = True
    baseline_duration_s: int = 60
    # Inference batch size for the timing / energy lazo. Default 1
    # mirrors the historical "single-image latency" measurement; B>1
    # is closer to a throughput-oriented deployment. The measurement
    # primitives divide per-call results by B internally so the
    # ``_mean_ms`` / ``_mean_mj`` columns stay per-image regardless.
    inference_batch_size: int = 1


@dataclass
class RemeasureJob:
    """Single experiment to re-measure."""

    source_report: Path           # original .batch_analysis_report
    destination_report: Path      # copy where new metrics will be written
    experiment_index: int = -1    # index in the BatchReportModel (for UI feedback)


@dataclass
class RemeasureOutcome:
    """Result for a single :class:`RemeasureJob` after the worker finishes."""

    job: RemeasureJob
    success: bool
    written_path: Optional[Path] = None
    error: str = ""
    skipped_tests: list[str] = field(default_factory=list)


def _build_model_facade(
    test_entry: dict,
    img_size: int,
    batch_dir: Path,
    use_gpu: bool,
    logger: logging.Logger,
):
    """Reconstruct a `PostprocessorNN`-compatible facade from a saved test.

    Returns a :class:`SimpleNamespace` exposing the attributes that
    ``measure_timing`` / ``measure_energy`` consume: ``model``, ``img_size``,
    ``is_conv``, ``device``, ``batch_size``, ``applicator``.
    """
    import torch

    raw_name = test_entry.get("model_name", "")
    canonical = resolve_model_name(raw_name)
    entry = MODEL_REGISTRY.get(canonical)
    if entry is None:
        raise RuntimeError(
            f"Model '{raw_name}' is not in MODEL_REGISTRY (resolved to '{canonical}')"
        )

    # Defaults + saved overrides — same recipe as PostprocessorNN.__init__.
    overrides = dict(test_entry.get("architecture_config") or {})
    if test_entry.get("dropout", 0):
        overrides.setdefault("dropout", test_entry["dropout"])
    kwargs = {**(entry.get("defaults") or {}), **overrides}
    if "img_size" in kwargs and kwargs["img_size"] is None:
        kwargs["img_size"] = img_size

    model_cls = entry["cls"]
    try:
        model = model_cls(**kwargs)
    except TypeError:
        # Some models ignore img_size; retry without it if it slipped in.
        kwargs.pop("img_size", None)
        model = model_cls(**kwargs)

    weights_path = batch_dir / "models" / f"{safe_test_dirname(test_entry.get('name', ''))}.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {weights_path}")

    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
    model.to(device)

    logger.debug(
        "Rebuilt model '%s' from %s on %s", canonical, weights_path.name, device
    )

    return SimpleNamespace(
        model=model,
        img_size=img_size,
        is_conv=entry.get("conv", False),
        device=device,
        batch_size=int(test_entry.get("batch_size", 16)),
        applicator=None,  # no reconstruction timing on re-measurement
    )


def _make_test_config(test_entry: dict, cfg: RemeasureConfig) -> TestConfiguration:
    """Build a :class:`TestConfiguration` with only the fields the
    measurement primitives read; everything else stays at the dataclass
    default and is harmless.

    ``test_split`` is forced to 100 because the data we have on disk
    *is* the original test set — there is nothing else to split off,
    and reconstruction timing should average across every available
    image instead of just the 10 % that originally fed the loss.
    """
    return TestConfiguration(
        name=test_entry.get("name", "test"),
        mask_type=test_entry.get("mask_type", "scatter"),
        reconstruction_method=test_entry.get("reconstruction_method", "conventional"),
        use_gpu=cfg.use_gpu,
        timing_warmup_runs=cfg.warmup_runs,
        timing_measurement_runs=cfg.measurement_runs,
        timing_sampling_rate_khz=cfg.sampling_rate_khz,
        test_split=100,
    )


# Map saved ``mask_type`` strings → the concrete class we instantiate so the
# isinstance() checks in :func:`create_applicator` (and its native dispatch)
# pick the correct algorithm. Imported lazily inside the helper to keep this
# module's startup cost small.
_MASK_TYPE_CLASSES: dict[str, str] = {
    "scatter":              "simulation_engine._2_mask_gen.mask_scatter:MaskScatter",
    "sweep":                "simulation_engine._2_mask_gen.mask_sweep:MaskSweep",
    "hadamard_natural":     "simulation_engine._2_mask_gen.mask_hadamard:MaskHadamard",
    "hadamard_cake_cutting":"simulation_engine._2_mask_gen.mask_hadamard_cake_cutting:MaskHadamardCakeCutting",
    "hadamard_walsh_paley": "simulation_engine._2_mask_gen.mask_hadamard_walsh_paley:MaskHadamardWalshPaley",
    "cal_sal":              "simulation_engine._2_mask_gen.mask_cal_sal:MaskCalSal",
}


def _instantiate_mask_facade(mask_type: str, img_size: int, n_patterns: int,
                             logger: logging.Logger):
    """Instantiate the right :class:`MaskABC` subclass with throwaway init
    parameters. The caller overwrites ``masks`` / ``num_patterns`` after,
    so the values we pass here only need to keep the constructor happy."""
    spec = _MASK_TYPE_CLASSES.get(mask_type)
    if spec is None:
        raise ValueError(f"Unsupported mask_type for re-measure: {mask_type!r}")
    module_name, _, class_name = spec.partition(":")
    import importlib
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)

    if mask_type == "scatter":
        return cls(img_size=img_size, point_density=10.0,
                   num_patterns=n_patterns, seed=0, logger=logger)
    if mask_type == "sweep":
        # parametros structure must match what MaskSweep expects, even
        # though we replace ``masks`` immediately afterwards.
        dummy = [{"angle": 0.0, "bar_width": 2, "stride": 4}]
        return cls(img_size=img_size, parametros=dummy, logger=logger)
    if mask_type in ("hadamard_natural", "hadamard_cake_cutting",
                     "hadamard_walsh_paley"):
        return cls(img_size=img_size, min_idx=0, max_idx=n_patterns,
                   logger=logger)
    # cal_sal: only needs img_size
    return cls(img_size=img_size, logger=logger)


def _build_remeasure_pipeline(test_entry: dict, batch_dir: Path, img_size: int,
                              test_config: TestConfiguration,
                              logger: logging.Logger):
    """Reload the test set + mask from disk and assemble an applicator.

    Only works for ``ALL_DATA`` exports — we need both
    ``data/<test>/masks.npz`` and ``data/<test>/test_images.npz`` to be
    present. Returns ``(dataset_facade, applicator)`` on success or
    ``(None, None)`` if anything is missing; the caller falls back to a
    timing run without reconstruction.
    """
    import numpy as np

    safe = safe_test_dirname(test_entry.get("name", ""))
    test_dir = batch_dir / "data" / safe
    masks_path = test_dir / "masks.npz"
    images_path = test_dir / "test_images.npz"
    if not (masks_path.exists() and images_path.exists()):
        logger.debug("No saved masks/images for '%s' — skipping reconstruction "
                     "timing (export level was not ALL_DATA?)", safe)
        return None, None

    masks_arr = np.load(str(masks_path))["masks"]
    with np.load(str(images_path)) as data:
        if "originals" not in data.files:
            logger.warning("test_images.npz for '%s' has no 'originals' key", safe)
            return None, None
        originals = np.array(data["originals"])

    # Dataset facade: applicators only read ``.data`` (indexable list of
    # 2-D arrays) and ``.img_size``.
    dataset = SimpleNamespace(
        data=[np.asarray(originals[i]) for i in range(originals.shape[0])],
        img_size=int(img_size),
        name="remeasure",
    )

    mask_type = test_entry.get("mask_type", "")
    n_patterns = int(masks_arr.shape[0])
    try:
        mask = _instantiate_mask_facade(mask_type, int(img_size), n_patterns, logger)
    except Exception as exc:
        logger.warning("Cannot instantiate mask facade for '%s' (%s): %s",
                       test_entry.get("name", ""), mask_type, exc)
        return dataset, None

    # Replace whatever the constructor pre-computed with the saved patterns
    # and disable regeneration — ``create_applicator`` would otherwise call
    # ``mask.generate_masks()`` and overwrite our loaded array with a fresh
    # synthesis (potentially in a different ordering).
    mask.masks = masks_arr
    mask.num_patterns = n_patterns
    mask.generate_masks = lambda *args, **kwargs: None  # noqa: E731

    try:
        applicator = create_applicator(test_config, mask, dataset)
    except Exception as exc:
        logger.warning("Cannot build applicator for '%s' (method=%s): %s",
                       test_entry.get("name", ""),
                       test_config.reconstruction_method, exc)
        return dataset, None
    return dataset, applicator


def _strip_old_metrics(test_entry: dict, kill_timing: bool, kill_energy: bool) -> None:
    """Drop stale keys before writing fresh measurements."""
    if kill_timing:
        for key in list(test_entry.keys()):
            if key.startswith("timing_") or key == "profiler_results":
                test_entry.pop(key, None)
        test_entry.pop("timing_metrics", None)
    if kill_energy:
        for key in list(test_entry.keys()):
            if key.startswith("energy_") or key.startswith("efficiency_"):
                test_entry.pop(key, None)


class RemeasureWorker(QObject):
    """QObject worker that processes a list of :class:`RemeasureJob`.

    Move it onto a :class:`QThread` and connect ``progress`` / ``job_done`` /
    ``finished`` / ``error`` to the dialog. Cancellation is cooperative —
    call :meth:`cancel` between tests.
    """

    progress = pyqtSignal(int, int, str)        # current, total, message
    job_done = pyqtSignal(object)               # RemeasureOutcome
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, jobs: list[RemeasureJob], cfg: RemeasureConfig,
                 logger: Optional[logging.Logger] = None, parent=None):
        super().__init__(parent)
        self.jobs = jobs
        self.cfg = cfg
        self.logger = (logger or logging.getLogger("RemeasureWorker"))
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def _total_steps(self) -> int:
        steps = 0
        for job in self.jobs:
            try:
                with open(job.source_report, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                steps += len(data.get("results", []))
            except Exception:
                steps += 1
        return max(steps, 1)

    def run(self) -> None:
        # Build a single EnergyAnalyzer up-front and reuse it for every
        # test of every job in this pass. On Jetson the jtop daemon
        # gets flaky after several open/close cycles (silent freezes
        # in start_measurement) — sharing one connection eliminates
        # that source of churn entirely. The analyzer is initialised
        # lazily so we don't pay the cost when ``measure_energy`` is
        # disabled.
        shared_analyzer: Optional[EnergyAnalyzer] = None
        baseline: Optional[BaselineResult] = None
        try:
            total = self._total_steps()
            done = 0
            if self.cfg.measure_energy:
                shared_analyzer = EnergyAnalyzer(
                    model=None,                 # set per-test inside measure_energy
                    device="cpu",               # ditto
                    warmup_runs=self.cfg.warmup_runs,
                    measurement_runs=self.cfg.measurement_runs,
                    enable_gpu_energy=self.cfg.use_gpu,
                    enable_cpu_energy=True,
                    logger=self.logger,
                )
                if not shared_analyzer.initialize():
                    self.logger.warning(
                        "Shared energy analyzer could not be initialised — "
                        "falling back to per-test analyzers."
                    )
                    shared_analyzer = None

                # Capture the idle baseline before the first test so
                # the dynamic-energy column can subtract the system
                # pedestal from each test's measurement. Reuses the
                # already-initialised monitor — no extra jtop spin-up.
                if (self.cfg.capture_baseline
                        and shared_analyzer is not None
                        and shared_analyzer.is_initialized):
                    baseline = capture_idle_baseline(
                        shared_analyzer._energy_monitor,
                        duration_s=self.cfg.baseline_duration_s,
                        logger=self.logger,
                        progress_callback=self._on_baseline_tick,
                        cancel_check=lambda: self._cancel,
                    )

            for job in self.jobs:
                if self._cancel:
                    break
                done = self._run_job(job, done, total, shared_analyzer, baseline)
            self.finished.emit()
        except Exception as exc:
            self.logger.exception("Re-measurement worker crashed")
            self.error.emit(str(exc))
            self.finished.emit()
        finally:
            if shared_analyzer is not None:
                try:
                    shared_analyzer.shutdown()
                except Exception:
                    self.logger.debug("Shared analyzer shutdown raised", exc_info=True)

    def _on_baseline_tick(self, elapsed_s: float, total_s: float) -> None:
        """Surface the baseline countdown through the existing progress
        signal. ``current=0, total=0`` is a sentinel the dialog
        recognises as "we're in the baseline phase, show a cooldown-
        style message instead of the per-test ratio".
        """
        msg = f"Capturing idle baseline ({int(elapsed_s)} / {int(total_s)} s)"
        self.progress.emit(0, 0, msg)

    def _run_job(self, job: RemeasureJob, done: int, total: int,
                 shared_analyzer: Optional[EnergyAnalyzer] = None,
                 baseline: Optional[BaselineResult] = None) -> int:
        outcome = RemeasureOutcome(job=job, success=False)
        try:
            with open(job.source_report, "r", encoding="utf-8") as fh:
                report = json.load(fh)
        except Exception as exc:
            outcome.error = f"Failed to read report: {exc}"
            self.job_done.emit(outcome)
            return done

        metadata = report.get("metadata", {}) or {}
        img_size = int(metadata.get("dataset_info", {}).get("img_size", 0))
        if img_size <= 0:
            outcome.error = "Report does not declare dataset_info.img_size"
            self.job_done.emit(outcome)
            return done

        batch_dir = job.source_report.parent
        new_results = []
        for test_entry in report.get("results", []):
            if self._cancel:
                break
            done += 1
            test_name = test_entry.get("name", "test")
            self.progress.emit(done, total, f"{job.source_report.stem} · {test_name}")
            # Banner line at the start of every test so the log is
            # self-contained — the reader can see at a glance which
            # compute path, batch size, sub-block layout and baseline
            # state were in effect for the numbers that follow.
            self.logger.info(
                "Starting measurement: test=%r compute_path=%s "
                "device_label=%s batch_size=%d warmup=%d measurements=%d "
                "(blocks of %d) baseline=%s",
                test_name,
                "GPU run" if self.cfg.use_gpu else "CPU run",
                self.cfg.device_label,
                int(self.cfg.inference_batch_size),
                int(self.cfg.warmup_runs),
                int(self.cfg.measurement_runs),
                max(1, int(self.cfg.measurement_runs)
                    // max(1, min(10, self.cfg.measurement_runs))),
                ("ON" if self.cfg.capture_baseline else "OFF"),
            )

            try:
                facade = _build_model_facade(
                    test_entry, img_size, batch_dir,
                    use_gpu=self.cfg.use_gpu, logger=self.logger,
                )
            except Exception as exc:
                self.logger.warning("Skipping '%s': %s", test_name, exc)
                outcome.skipped_tests.append(f"{test_name}: {exc}")
                new_results.append(test_entry)
                continue

            updated = copy.deepcopy(test_entry)
            _strip_old_metrics(updated, self.cfg.measure_timing, self.cfg.measure_energy)
            test_config = _make_test_config(test_entry, self.cfg)

            if self.cfg.measure_timing:
                # Try to rebuild the dataset + applicator so measure_timing
                # can populate acquisition + reconstruction times. Falls
                # back to inference-only timing if the export didn't include
                # masks/images.
                dataset_facade, applicator = _build_remeasure_pipeline(
                    test_entry, batch_dir, img_size, test_config, self.logger,
                )
                try:
                    timing = measure_timing(
                        facade, test_config, dataset=dataset_facade,
                        logger=self.logger,
                        applicator=applicator,
                        warmup_runs=self.cfg.warmup_runs,
                        measurement_runs=self.cfg.measurement_runs,
                        sampling_rate_khz=self.cfg.sampling_rate_khz,
                        inference_batch_size=self.cfg.inference_batch_size,
                        # Reconstruction with NumPy on Jetson CPU is
                        # roughly linear in pattern count and dominates
                        # the wall-clock of a re-measurement run when
                        # a test has 1000+ patterns. Cap the loop at
                        # 3 samples — enough for a stable mean, fast
                        # enough that the user doesn't think the dialog
                        # has hung.
                        max_recon_samples=3,
                    )
                    # When the export had no data/ folder we couldn't pass
                    # a mask, so num_patterns defaulted to 1 and acquisition
                    # came out wrong (1 / sampling_rate_khz). Carry forward
                    # the pattern count from the original report so the
                    # acquisition column at least matches reality.
                    if applicator is None:
                        prev = test_entry.get("timing_num_patterns")
                        if prev and self.cfg.sampling_rate_khz > 0:
                            timing["timing_num_patterns"] = int(prev)
                            timing["timing_acquisition_ms"] = (
                                float(prev) / self.cfg.sampling_rate_khz
                            )
                    updated.update(timing)
                    timing_metrics = updated.get("timing_metrics") or {}
                    timing_metrics["inference_time_ms"] = timing.get("timing_mean_ms", 0.0)
                    updated["timing_metrics"] = timing_metrics
                except Exception as exc:
                    self.logger.warning("Timing failed for '%s': %s", test_name, exc)
                    updated["timing_error"] = str(exc)

            if self.cfg.measure_energy:
                try:
                    energy = measure_energy(
                        facade, test_config, self.logger,
                        warmup_runs=self.cfg.warmup_runs,
                        measurement_runs=self.cfg.measurement_runs,
                        analyzer=shared_analyzer,
                        inference_batch_size=self.cfg.inference_batch_size,
                    )
                    updated.update(energy)
                except Exception as exc:
                    self.logger.warning("Energy failed for '%s': %s", test_name, exc)
                    updated["energy_error"] = str(exc)

                # Derive dynamic-power columns from the just-measured
                # energy plus the shared baseline. The helper keeps
                # the timing-window assumption consistent (energy and
                # the inference time used to subtract the pedestal
                # both come from the same energy phase, see
                # ``derive_dynamic_metrics``).
                if baseline is not None:
                    dyn = derive_dynamic_metrics(
                        energy_mean_mj=updated.get("energy_mean_mj"),
                        energy_mean_watts=updated.get("energy_mean_watts"),
                        baseline_power_W=baseline.total_power_W,
                    )
                    updated.update(dyn)
                    if dyn.get("dynamic_power_W") is not None and dyn["dynamic_power_W"] < 0:
                        self.logger.warning(
                            "Dynamic power negative for '%s': baseline %.2f W "
                            "> test power %.2f W — check thermal state or "
                            "sampling protocol.",
                            test_name, baseline.total_power_W,
                            float(updated.get("energy_mean_watts") or 0.0),
                        )

            updated["_remeasured"] = True
            updated["_remeasured_at"] = datetime.now().isoformat()
            updated["_remeasured_device"] = self.cfg.device_label
            new_results.append(updated)

            # Free the model before moving to the next test (Jetson VRAM is
            # tight; without this two big U-Nets would coexist on device).
            del facade
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        report["results"] = new_results
        report.setdefault("metadata", {})
        report["metadata"]["remeasured_at"] = datetime.now().isoformat()
        report["metadata"]["remeasured_device"] = self.cfg.device_label
        report["metadata"]["batch_name"] = (
            f"{report['metadata'].get('batch_name', job.source_report.stem)} "
            f"(re-executed on {self.cfg.device_label})"
        )
        # Stash the baseline at metadata level (once per report). When
        # ``capture_baseline`` was off this stays absent, which is the
        # cue the Energy view uses to disable the "Subtract idle
        # baseline" toggle for that experiment.
        if baseline is not None:
            report["metadata"]["idle_baseline"] = baseline.to_dict()

        try:
            self._write_report(job.destination_report, job.source_report, report)
        except Exception as exc:
            outcome.error = f"Failed to write copy: {exc}"
            self.job_done.emit(outcome)
            return done

        outcome.success = True
        outcome.written_path = job.destination_report
        self.job_done.emit(outcome)
        return done

    @staticmethod
    def _write_report(dest: Path, src: Path, report: dict) -> None:
        # Use shutil.copy first so the destination inherits permissions, then
        # overwrite the JSON payload with the updated dict.
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        with open(dest, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False, default=str)
