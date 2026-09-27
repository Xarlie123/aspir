# File: ui/custom_widgets/dataset_control/load_measurements/load_measurements_widget.py

import logging
import os
from pathlib import Path

from PySide6 import QtWidgets
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QCheckBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton

from simulation_engine import experimental
from simulation_engine._1_dataset_gen.measured import DatasetFromMeasurements, load_measurements
from ui.custom_widgets.common.button_styles import BUTTON_STYLE_BLUE, BUTTON_STYLE_GREEN, apply_button_style

FILE_FILTER = ("Measurements (*.npz *_raw.mat);;ASPIR measurement file (*.npz);;"
               "SPIHIM raw acquisition (*_raw.mat)")


class LoadMeasurementsWidget(QtWidgets.QWidget):
    """Select a file of *measured* single-pixel data and emit a DatasetFromMeasurements.

    Accepts ASPIR measurement files (``.npz``, see ``measured.py``) and SPIHIM
    raw acquisitions (``*_raw.mat``; the Hadamard PNG bank is looked up next to
    the session folder, as ``scripts/download_spihim.py`` lays it out).
    """
    datasetReady = Signal(object)  # noqa: N815 - same signal name as the other dataset widgets

    def __init__(self, parent=None, logger=None):
        super().__init__(parent)
        self.logger = (logging.getLogger(self.__class__.__name__) if logger is None
                       else logger.getChild(self.__class__.__name__))
        self._path = ""

        layout = QtWidgets.QVBoxLayout(self)
        intro = QLabel(
            "Load bucket-detector values acquired on real hardware. The acquisition "
            "step is skipped: the patterns come from the file and the Test Masks step "
            "reconstructs from the measured values. Network training and reports "
            "need ground-truth images and are not available for measured data.")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        row = QHBoxLayout()
        self.path_input = QLineEdit()
        self.path_input.setReadOnly(True)
        self.path_input.setPlaceholderText("No file selected")
        self.browse_button = QPushButton("Select file...")
        row.addWidget(self.path_input, 1)
        row.addWidget(self.browse_button)
        layout.addLayout(row)

        self.average_frames = QCheckBox("Average repeated frames")
        self.average_frames.setChecked(True)
        self.unit_peak = QCheckBox("Scale values to a unit-peak image (FISTA / TV defaults assume [0, 1])")
        self.unit_peak.setChecked(True)
        layout.addWidget(self.average_frames)
        layout.addWidget(self.unit_peak)

        self.info_label = QLabel("")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        self.load_button = QPushButton("Load measurements")
        layout.addWidget(self.load_button)
        apply_button_style(self.browse_button, BUTTON_STYLE_BLUE)
        apply_button_style(self.load_button, BUTTON_STYLE_GREEN)

        self.browse_button.clicked.connect(self._select_file)
        self.load_button.clicked.connect(self._on_load)

    @staticmethod
    def _datasets_dir():
        return str(Path(__file__).resolve().parents[5] / "datasets")

    def _select_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select measurement file", self._datasets_dir(), FILE_FILTER)
        if not path:
            return
        self._path = path
        self.path_input.setText(path)
        self.info_label.setText(self._describe(path))

    def _describe(self, path):
        if path.endswith(".npz"):
            try:
                data = load_measurements(path)
            except Exception as exc:  # shown to the user
                return f"Not a readable ASPIR measurement file: {exc}"
            src = data.metadata.get("source", "unknown source")
            return (f"{data.n_patterns} patterns, {data.n_frames} frame(s), "
                    f"{data.img_size}x{data.img_size} ({src}).")
        bank = self._find_spihim_bank(path)
        return ("SPIHIM acquisition (OpenSpyrit, CC-BY 4.0; cite Beneti Martins et al., "
                "Opt. Express 31(10), 15599, 2023). Pattern bank: "
                + (str(bank) if bank else "not found, you will be asked for it."))

    @staticmethod
    def _find_spihim_bank(path):
        for base in Path(path).resolve().parents[:3]:
            candidate = base / "Hadamard_64x64"
            if candidate.is_dir():
                return candidate
        return None

    def _loader(self, path):
        average, unit_peak = self.average_frames.isChecked(), self.unit_peak.isChecked()
        if path.endswith(".npz"):
            def read():
                return load_measurements(path)
        else:
            from simulation_engine._1_dataset_gen.spihim import load_spihim

            bank = self._find_spihim_bank(path)
            if bank is None:
                chosen = QFileDialog.getExistingDirectory(self, "Select the Hadamard_64x64 pattern bank")
                if not chosen:
                    return None
                bank = Path(chosen)

            def read():
                return load_spihim(path, bank)

        def load():
            data = read()
            if average and data.n_frames > 1:
                data = data.mean_frame()
            return experimental.scale_to_unit_peak(data) if unit_peak else data
        return load

    def _on_load(self):
        if not self._path or not os.path.isfile(self._path):
            QMessageBox.warning(self, "Warning", "Please select a measurement file first.")
            return
        loader = self._loader(self._path)
        if loader is None:
            return
        name = Path(self._path).name.removesuffix(".npz").removesuffix("_raw.mat")
        ds = DatasetFromMeasurements(path=self._path, loader=loader, logger=self.logger, name=name)
        self.logger.info("Emitting datasetReady for measurements %s", self._path)
        self.datasetReady.emit(ds)
