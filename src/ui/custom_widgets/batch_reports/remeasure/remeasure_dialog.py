"""Modal dialog that configures and runs the re-measurement worker.

The dialog has two visual states wired to a :class:`QStackedWidget`:

1. *Config* — the user picks warmup runs, iterations, backends, etc.
2. *Progress* — a progress bar + status label drive while the worker runs.

When the worker emits ``finished`` the bottom buttons swap to a single
*Close* button. Cancellation is cooperative: pressing *Cancel* during the
run sets a flag on the worker; the in-flight test still finishes (timing
loops are short and atomicity matters for accuracy), but no new test is
started.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import QThread, Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ui.custom_widgets.batch_reports.remeasure.remeasure_worker import (
    RemeasureConfig,
    RemeasureJob,
    RemeasureOutcome,
    RemeasureWorker,
)


def _detect_device_label() -> str:
    """Return a short, file-system-friendly tag for the current host.

    Order: Jetson (via /etc/nv_tegra_release) → first word of the CUDA
    device name → ``cpu``.
    """
    if Path("/etc/nv_tegra_release").exists():
        return "jetson"
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0).lower()
            return name.split()[0].replace("/", "-")
    except Exception:
        pass
    return "cpu"


def _suffix_for(source: Path, device_label: str) -> Path:
    """Build the ``_reexecuted_<device>_<timestamp>`` sibling path.

    We append a counter if the natural name already exists so two
    consecutive runs on the same minute don't collide.
    """
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    base = f"{source.stem}_reexecuted_{device_label}_{stamp}"
    candidate = source.with_name(base + source.suffix)
    counter = 1
    while candidate.exists():
        candidate = source.with_name(f"{base}_{counter}{source.suffix}")
        counter += 1
    return candidate


class RemeasureDialog(QDialog):
    """Configure + run re-measurement for one or more loaded experiments.

    Parameters
    ----------
    sources : list[tuple[int, Path]]
        ``(experiment_index, report_path)`` pairs. The dialog will copy
        each report to a sibling file with the ``_reexecuted_…`` suffix
        and rewrite the timing/energy fields in the copy only.
    """

    def __init__(self, sources: list[tuple[int, Path]],
                 logger: Optional[logging.Logger] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Re-measure timing & energy")
        self.setModal(True)
        self.resize(540, 420)

        self.logger = (logger or logging.getLogger("RemeasureDialog"))
        self._sources = sources
        self._thread: Optional[QThread] = None
        self._worker: Optional[RemeasureWorker] = None
        self._outputs: list[Path] = []
        self._failures: list[str] = []

        self._build_ui()

    # ------------------------------------------------------------------
    # Public — what the caller reads after exec_()
    # ------------------------------------------------------------------
    def written_paths(self) -> list[Path]:
        return list(self._outputs)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        n_exp = len(self._sources)
        header = QLabel(
            f"Re-measure {n_exp} experiment{'s' if n_exp != 1 else ''} "
            f"on this machine. The original report file is left untouched; "
            f"a copy with the suffix <i>_reexecuted_&lt;device&gt;_&lt;timestamp&gt;</i> "
            f"will be written next to it."
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        self._stack = QStackedWidget()
        self._stack.addWidget(self._build_config_page())
        self._stack.addWidget(self._build_progress_page())
        layout.addWidget(self._stack, 1)

        self._buttons = QDialogButtonBox()
        self._start_btn = self._buttons.addButton("Start", QDialogButtonBox.AcceptRole)
        self._cancel_btn = self._buttons.addButton(QDialogButtonBox.Cancel)
        self._start_btn.clicked.connect(self._on_start)
        self._cancel_btn.clicked.connect(self._on_cancel)
        layout.addWidget(self._buttons)

    def _build_config_page(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)

        what_box = QGroupBox("What to measure")
        what_form = QFormLayout(what_box)
        self._chk_timing = QCheckBox("Inference timing (CPU + GPU)")
        self._chk_timing.setChecked(True)
        self._chk_energy = QCheckBox("Energy consumption")
        self._chk_energy.setChecked(True)
        what_form.addRow(self._chk_timing)
        what_form.addRow(self._chk_energy)
        outer.addWidget(what_box)

        knobs_box = QGroupBox("Measurement parameters")
        knobs_form = QFormLayout(knobs_box)

        self._spn_warmup = QSpinBox()
        self._spn_warmup.setRange(0, 1000)
        self._spn_warmup.setValue(5)
        knobs_form.addRow("Warmup runs:", self._spn_warmup)

        self._spn_runs = QSpinBox()
        self._spn_runs.setRange(1, 10000)
        self._spn_runs.setValue(200)
        self._spn_runs.setToolTip(
            "Higher values give more stable energy and timing estimates "
            "(the original batch test default is 800)."
        )
        knobs_form.addRow("Measurement runs:", self._spn_runs)

        self._spn_sampling = QDoubleSpinBox()
        self._spn_sampling.setRange(0.001, 1000.0)
        self._spn_sampling.setDecimals(3)
        self._spn_sampling.setValue(10.752)
        self._spn_sampling.setToolTip("Pattern sampling rate; only affects acquisition-time math.")
        knobs_form.addRow("Sampling rate (kHz):", self._spn_sampling)

        self._chk_gpu = QCheckBox("Use GPU when available")
        self._chk_gpu.setChecked(True)
        knobs_form.addRow(self._chk_gpu)

        self._txt_device = QLineEdit(_detect_device_label())
        self._txt_device.setToolTip(
            "Short tag added to the output filename "
            "(e.g. <name>_reexecuted_<tag>_<timestamp>)."
        )
        knobs_form.addRow("Device label:", self._txt_device)

        outer.addWidget(knobs_box)
        outer.addStretch(1)
        return page

    def _build_progress_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        self._status = QLabel("Preparing…")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        self._progress = QProgressBar()
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        layout.addWidget(self._progress)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setStyleSheet("font-family: monospace; font-size: 10px;")
        layout.addWidget(self._log, 1)
        return page

    # ------------------------------------------------------------------
    # Worker lifecycle
    # ------------------------------------------------------------------
    def _on_start(self) -> None:
        cfg = RemeasureConfig(
            measure_timing=self._chk_timing.isChecked(),
            measure_energy=self._chk_energy.isChecked(),
            use_gpu=self._chk_gpu.isChecked(),
            warmup_runs=int(self._spn_warmup.value()),
            measurement_runs=int(self._spn_runs.value()),
            sampling_rate_khz=float(self._spn_sampling.value()),
            device_label=(self._txt_device.text().strip() or "device"),
        )
        if not (cfg.measure_timing or cfg.measure_energy):
            self._append_log("Nothing to measure — pick at least one of timing/energy.")
            return

        jobs = []
        for exp_idx, src in self._sources:
            dest = _suffix_for(Path(src), cfg.device_label)
            jobs.append(RemeasureJob(
                source_report=Path(src),
                destination_report=dest,
                experiment_index=exp_idx,
            ))

        self._stack.setCurrentIndex(1)
        self._start_btn.setEnabled(False)
        self._cancel_btn.setText("Cancel")

        self._thread = QThread(self)
        self._worker = RemeasureWorker(jobs, cfg, logger=self.logger)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.job_done.connect(self._on_job_done)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)

        for job in jobs:
            self._append_log(f"queued: {job.source_report.name} → {job.destination_report.name}")
        self._thread.start()

    def _on_cancel(self) -> None:
        if self._worker is not None and self._thread is not None and self._thread.isRunning():
            self._append_log("Cancellation requested — current test will finish.")
            self._worker.cancel()
            self._cancel_btn.setEnabled(False)
            return
        self.reject()

    # ------------------------------------------------------------------
    # Worker signal handlers
    # ------------------------------------------------------------------
    def _on_progress(self, current: int, total: int, message: str) -> None:
        if self._progress.maximum() != total:
            self._progress.setRange(0, total)
        self._progress.setValue(current)
        self._status.setText(f"[{current}/{total}] {message}")

    def _on_job_done(self, outcome: RemeasureOutcome) -> None:
        if outcome.success and outcome.written_path:
            self._outputs.append(outcome.written_path)
            self._append_log(f"OK   {outcome.written_path.name}")
            for skipped in outcome.skipped_tests:
                self._append_log(f"  ! skipped {skipped}")
        else:
            self._failures.append(f"{outcome.job.source_report.name}: {outcome.error}")
            self._append_log(f"FAIL {outcome.job.source_report.name}: {outcome.error}")

    def _on_finished(self) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
            self._thread = None
        self._worker = None

        if self._failures:
            self._status.setText(
                f"Done with {len(self._failures)} failure(s) — see log below."
            )
        else:
            self._status.setText(
                f"Done. {len(self._outputs)} new report(s) written."
            )

        self._cancel_btn.setEnabled(True)
        self._cancel_btn.setText("Close")
        try:
            self._cancel_btn.clicked.disconnect()
        except TypeError:
            pass
        self._cancel_btn.clicked.connect(self.accept)
        self._start_btn.setVisible(False)

    def _on_error(self, msg: str) -> None:
        self._append_log(f"WORKER ERROR: {msg}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _append_log(self, line: str) -> None:
        self._log.append(line)

    def closeEvent(self, event):  # noqa: N802 - Qt signature
        # If the user closes the window while a run is in progress,
        # cancel cooperatively and wait for the thread to settle so we
        # don't leave a worker running on a dead QObject.
        if self._thread is not None and self._thread.isRunning():
            if self._worker is not None:
                self._worker.cancel()
            self._thread.quit()
            self._thread.wait()
        super().closeEvent(event)
