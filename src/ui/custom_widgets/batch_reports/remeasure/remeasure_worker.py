"""Background worker that re-runs inference on saved experiments.

The worker iterates the tests of one or more ``.batch_analysis_report``
files, rebuilds each trained model from its checkpoint + architecture
metadata, and re-measures DNN inference timing and/or energy with the
existing :func:`measure_timing` / :func:`measure_energy` primitives. The
output is written to a *copy* of the report (the suffix is decided by the
caller); the originals are left untouched.

We only need the model weights and ``img_size``; the random-tensor
benchmark inside ``measure_timing``/``measure_energy`` does not require
the original test dataset, so re-measurement also works for experiments
exported with ``REPORTS_AND_MODELS`` (no ``data/`` directory).
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
from ui.custom_widgets.batch_test.batch_test_runner._energy import measure_energy
from ui.custom_widgets.batch_test.batch_test_runner._timing import measure_timing
from ui.custom_widgets.batch_test.test_config_model import TestConfiguration
from ui.utils.file_formats import safe_test_dirname


@dataclass
class RemeasureConfig:
    """User-controllable knobs for a re-measurement run."""

    measure_timing: bool = True
    measure_energy: bool = True
    use_gpu: bool = True
    warmup_runs: int = 5
    measurement_runs: int = 200
    sampling_rate_khz: float = 10.752
    device_label: str = "jetson"  # appears in the output filename


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
    default and is harmless."""
    return TestConfiguration(
        name=test_entry.get("name", "test"),
        use_gpu=cfg.use_gpu,
        timing_warmup_runs=cfg.warmup_runs,
        timing_measurement_runs=cfg.measurement_runs,
        timing_sampling_rate_khz=cfg.sampling_rate_khz,
        test_split=int(test_entry.get("test_split", 10)),
    )


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
        try:
            total = self._total_steps()
            done = 0
            for job in self.jobs:
                if self._cancel:
                    break
                done = self._run_job(job, done, total)
            self.finished.emit()
        except Exception as exc:
            self.logger.exception("Re-measurement worker crashed")
            self.error.emit(str(exc))
            self.finished.emit()

    def _run_job(self, job: RemeasureJob, done: int, total: int) -> int:
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
                try:
                    timing = measure_timing(
                        facade, test_config, dataset=None, logger=self.logger,
                        applicator=None,
                        warmup_runs=self.cfg.warmup_runs,
                        measurement_runs=self.cfg.measurement_runs,
                        sampling_rate_khz=self.cfg.sampling_rate_khz,
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
                    )
                    updated.update(energy)
                except Exception as exc:
                    self.logger.warning("Energy failed for '%s': %s", test_name, exc)
                    updated["energy_error"] = str(exc)

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
