import logging

from PySide6 import QtWidgets
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
)

from simulation_engine._2_mask_gen.mask_fourier import MaskFourier
from ui.custom_widgets.common.button_styles import BUTTON_STYLE_GREEN, apply_button_style

# First label names the Fourier-specific algorithm; the rest match the generic
# iterative solvers exposed by every other mask widget.
RECON_METHOD_CHOICES = ("Fourier Inverse", "Pseudoinverse", "FISTA", "TV-norm")

PHASE_STEP_CHOICES = (("4 (four-step)", 4), ("3 (three-step)", 3))
ORDERING_CHOICES = (("Circular (radial)", "circular"), ("Square (low-pass block)", "square"))

BINARIZATION_CHOICES = (
    ("Grayscale", "none"),
    ("Binary - Floyd-Steinberg", "floyd_steinberg"),
    ("Binary - Bayer", "bayer"),
)


class FourierControlWidget(QtWidgets.QWidget):
    """Control panel for the Fourier Single-Pixel Imaging mask family.

    Built in code rather than from a Designer ``.ui`` file because the Fourier
    parameters (sampling ratio, phase steps, spectrum ordering) share nothing
    with the index-range slider the Hadamard widget provides.

    Emits ``maskReady(mask)`` with a configured ``MaskFourier`` instance.
    """

    # Qt signal naming: camelCase matches the sibling mask-control widgets.
    maskReady = Signal(object)  # noqa: N815

    def __init__(self, parent=None, logger=None):
        super().__init__(parent)
        if logger is None:
            self.logger = logging.getLogger(self.__class__.__name__)
        else:
            self.logger = logger.getChild(self.__class__.__name__)
        self.logger.debug("Initializing FourierControlWidget")

        self._mask = None
        self._img_size = None

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setSpacing(8)

        # --- sampling ratio -------------------------------------------------
        ratio_row = QHBoxLayout()
        ratio_row.setSpacing(10)
        ratio_label = QLabel("Sampling ratio M/N:")
        ratio_label.setStyleSheet("font-size: 11px;")
        ratio_row.addWidget(ratio_label)
        self.sampling_ratio_spinbox = QDoubleSpinBox()
        self.sampling_ratio_spinbox.setRange(0.1, 200.0)
        self.sampling_ratio_spinbox.setDecimals(1)
        self.sampling_ratio_spinbox.setSingleStep(5.0)
        self.sampling_ratio_spinbox.setValue(25.0)
        self.sampling_ratio_spinbox.setSuffix(" %")
        self.sampling_ratio_spinbox.setFixedWidth(90)
        self.sampling_ratio_spinbox.setToolTip(
            "Patterns projected as a percentage of the pixel count N.\n"
            "Each Fourier coefficient costs 'phase steps' patterns, so the\n"
            "coefficient budget is M / phase_steps. Above ~200 % the whole\n"
            "half-plane is already covered and the value is clamped."
        )
        self.sampling_ratio_spinbox.valueChanged.connect(self._update_count_label)
        ratio_row.addWidget(self.sampling_ratio_spinbox)
        ratio_row.addStretch()
        self.main_layout.addLayout(ratio_row)

        # --- phase steps ----------------------------------------------------
        phase_row = QHBoxLayout()
        phase_row.setSpacing(10)
        phase_label = QLabel("Phase steps:")
        phase_label.setStyleSheet("font-size: 11px;")
        phase_row.addWidget(phase_label)
        self.phase_steps_combo = QComboBox()
        for text, _ in PHASE_STEP_CHOICES:
            self.phase_steps_combo.addItem(text)
        self.phase_steps_combo.setToolTip(
            "Phase-shifting scheme. Four-step (Zhang et al. 2015) is the\n"
            "canonical FSPI sequence; three-step (Zhang et al. 2017) trades\n"
            "25 % fewer patterns per coefficient for slightly more noise."
        )
        self.phase_steps_combo.currentIndexChanged.connect(self._update_count_label)
        phase_row.addWidget(self.phase_steps_combo)
        phase_row.addStretch()
        self.main_layout.addLayout(phase_row)

        # --- spectrum ordering ----------------------------------------------
        order_row = QHBoxLayout()
        order_row.setSpacing(10)
        order_label = QLabel("Spectrum ordering:")
        order_label.setStyleSheet("font-size: 11px;")
        order_row.addWidget(order_label)
        self.spectrum_ordering_combo = QComboBox()
        for text, _ in ORDERING_CHOICES:
            self.spectrum_ordering_combo.addItem(text)
        self.spectrum_ordering_combo.setToolTip(
            "Order in which spectrum coefficients are acquired, always\n"
            "low-frequency first: circular takes growing radial rings,\n"
            "square takes growing square low-pass blocks."
        )
        order_row.addWidget(self.spectrum_ordering_combo)
        order_row.addStretch()
        self.main_layout.addLayout(order_row)

        # --- DMD illumination -------------------------------------------------
        binar_row = QHBoxLayout()
        binar_row.setSpacing(10)
        binar_label = QLabel("DMD illumination:")
        binar_label.setStyleSheet("font-size: 11px;")
        binar_row.addWidget(binar_label)
        self.binarization_combo = QComboBox()
        for text, _ in BINARIZATION_CHOICES:
            self.binarization_combo.addItem(text)
        self.binarization_combo.setToolTip(
            "Grayscale patterns need pulse-width modulation on a DMD, which is\n"
            "roughly two orders of magnitude slower than binary switching.\n"
            "Dithering trades pattern fidelity for that frame rate:\n"
            "Floyd-Steinberg diffuses the error and keeps the local mean;\n"
            "Bayer is an ordered dither, far faster to generate but with\n"
            "structured artefacts that need upsampling to push out of band."
        )
        self.binarization_combo.currentIndexChanged.connect(self._on_binarization_changed)
        binar_row.addWidget(self.binarization_combo)
        up_label = QLabel("upsampling:")
        up_label.setStyleSheet("font-size: 11px;")
        binar_row.addWidget(up_label)
        self.dmd_upsampling_spinbox = QSpinBox()
        self.dmd_upsampling_spinbox.setRange(1, 8)
        self.dmd_upsampling_spinbox.setValue(1)
        self.dmd_upsampling_spinbox.setPrefix("x ")
        self.dmd_upsampling_spinbox.setFixedWidth(70)
        self.dmd_upsampling_spinbox.setEnabled(False)
        self.dmd_upsampling_spinbox.setToolTip(
            "Mirror-grid magnification U used before dithering. The scene is\n"
            "still modelled at the image size, so each pixel integrates the\n"
            "U x U mirrors covering it and recovers U^2 + 1 grey levels.\n"
            "Generation cost grows as U^2 with Floyd-Steinberg."
        )
        self.dmd_upsampling_spinbox.valueChanged.connect(self._update_count_label)
        binar_row.addWidget(self.dmd_upsampling_spinbox)
        binar_row.addStretch()
        self.main_layout.addLayout(binar_row)

        # --- live pattern budget --------------------------------------------
        self.selected_count_label = QLabel("(patterns: -)")
        self.selected_count_label.setStyleSheet("font-size: 11px; color: #666;")
        self.main_layout.addWidget(self.selected_count_label)

        # --- reconstruction method -------------------------------------------
        recon_row = QHBoxLayout()
        recon_row.setSpacing(10)
        recon_label = QLabel("Reconstruction method:")
        recon_label.setStyleSheet("font-size: 11px;")
        recon_row.addWidget(recon_label)
        self.reconstruction_method_combo = QComboBox()
        self.reconstruction_method_combo.addItems(RECON_METHOD_CHOICES)
        self.reconstruction_method_combo.setToolTip(
            "Fourier Inverse = phase-shift demodulation plus inverse FFT.\n"
            "Other options use the generic iterative solvers."
        )
        recon_row.addWidget(self.reconstruction_method_combo)
        recon_row.addStretch()
        self.main_layout.addLayout(recon_row)

        # --- generate --------------------------------------------------------
        self.generate_masks_button = QPushButton("Generate masks")
        self.generate_masks_button.clicked.connect(self._on_generate_mask)
        apply_button_style(self.generate_masks_button, BUTTON_STYLE_GREEN)
        self.main_layout.addWidget(self.generate_masks_button)

        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.adjustSize()
        self._update_count_label()

    # ------------------------------------------------------------------
    # Parameter accessors (mirrors the other mask-control widgets)
    # ------------------------------------------------------------------
    def get_sampling_ratio(self) -> float:
        """Sampling ratio as a fraction (the spinbox shows a percentage)."""
        return float(self.sampling_ratio_spinbox.value()) / 100.0

    def set_sampling_ratio(self, value: float):
        self.sampling_ratio_spinbox.setValue(float(value) * 100.0)

    def get_phase_steps(self) -> int:
        return PHASE_STEP_CHOICES[self.phase_steps_combo.currentIndex()][1]

    def set_phase_steps(self, value: int):
        for idx, (_, steps) in enumerate(PHASE_STEP_CHOICES):
            if steps == int(value):
                self.phase_steps_combo.setCurrentIndex(idx)
                return

    def get_spectrum_ordering(self) -> str:
        return ORDERING_CHOICES[self.spectrum_ordering_combo.currentIndex()][1]

    def set_spectrum_ordering(self, value: str):
        for idx, (_, key) in enumerate(ORDERING_CHOICES):
            if key == str(value).lower():
                self.spectrum_ordering_combo.setCurrentIndex(idx)
                return

    def get_binarization(self) -> str:
        return BINARIZATION_CHOICES[self.binarization_combo.currentIndex()][1]

    def set_binarization(self, value: str):
        for idx, (_, key) in enumerate(BINARIZATION_CHOICES):
            if key == str(value).lower():
                self.binarization_combo.setCurrentIndex(idx)
                return

    def get_dmd_upsampling(self) -> int:
        return int(self.dmd_upsampling_spinbox.value())

    def set_dmd_upsampling(self, value: int):
        self.dmd_upsampling_spinbox.setValue(int(value))

    def get_parameters(self) -> dict:
        return {
            "sampling_ratio": self.get_sampling_ratio(),
            "phase_steps": self.get_phase_steps(),
            "spectrum_ordering": self.get_spectrum_ordering(),
            "binarization": self.get_binarization(),
            "dmd_upsampling": self.get_dmd_upsampling(),
        }

    def _on_binarization_changed(self, *_):
        """Upsampling only applies when the patterns are actually dithered."""
        self.dmd_upsampling_spinbox.setEnabled(self.get_binarization() != "none")
        self._update_count_label()

    def get_reconstruction_method(self) -> str:
        return self.reconstruction_method_combo.currentText()

    def set_reconstruction_method(self, value: str):
        idx = self.reconstruction_method_combo.findText(value)
        if idx >= 0:
            self.reconstruction_method_combo.setCurrentIndex(idx)

    # ------------------------------------------------------------------
    def _update_count_label(self, *_):
        """Show the pattern / coefficient budget for the current settings."""
        if not self._img_size:
            self.selected_count_label.setText("(patterns: - ; create a dataset first)")
            return
        probe = MaskFourier(
            img_size=self._img_size,
            sampling_ratio=self.get_sampling_ratio(),
            phase_steps=self.get_phase_steps(),
            spectrum_ordering=self.get_spectrum_ordering(),
            binarization=self.get_binarization(),
            dmd_upsampling=self.get_dmd_upsampling(),
            logger=self.logger,
        )
        n_coeff, n_patterns = probe._resolve_counts()
        text = (f"({n_patterns} patterns = {n_coeff} coefficients x {probe.phase_steps} phases, "
                f"effective M/N = {probe.effective_sampling_ratio * 100:.1f} %")
        if probe.binarization != "none":
            u = probe.dmd_upsampling
            text += (f"; binary {u * self._img_size}x{u * self._img_size} DMD frames, "
                     f"{u * u + 1} grey levels")
        self.selected_count_label.setText(text + ")")

    def set_img_size(self, img_size: int):
        """Set the image size for mask instantiation."""
        self._img_size = img_size
        self.logger.info("Image size set to %d", img_size)
        self._update_count_label()

    def _on_generate_mask(self):
        """Build the MaskFourier instance and hand it to the pipeline."""
        self.logger.debug("_on_generate_mask called; img_size=%s", self._img_size)
        if self._img_size is None:
            QMessageBox.warning(self, "Error", "Create a dataset first.")
            self.logger.warning("Attempted to generate mask without img_size set")
            return
        try:
            mask_obj = MaskFourier(
                img_size=self._img_size,
                sampling_ratio=self.get_sampling_ratio(),
                phase_steps=self.get_phase_steps(),
                spectrum_ordering=self.get_spectrum_ordering(),
                binarization=self.get_binarization(),
                dmd_upsampling=self.get_dmd_upsampling(),
                logger=self.logger,
            )
            mask_obj.applicator_type_scatter = self.get_reconstruction_method()
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))
            self.logger.error("Error instantiating mask: %s", e)
            return
        self._mask = mask_obj
        self.logger.info("Mask generated: MaskFourier (method=%s)",
                         mask_obj.applicator_type_scatter)
        self.maskReady.emit(mask_obj)

    @property
    def mask(self):
        """Last generated mask object."""
        return self._mask
