"""Lambda / iteration controls for the iterative solvers, shared by every mask panel."""

from PySide6.QtWidgets import QDoubleSpinBox, QHBoxLayout, QLabel, QSpinBox, QWidget

from simulation_engine._3_applicator.applicator_fista import (
    DEFAULT_ITERATIONS as FISTA_DEFAULT_ITERATIONS,
)
from simulation_engine._3_applicator.applicator_fista import (
    DEFAULT_LAMBDA as FISTA_DEFAULT_LAMBDA,
)
from simulation_engine._3_applicator.applicator_tv_norm import (
    DEFAULT_ITERATIONS as TV_DEFAULT_ITERATIONS,
)
from simulation_engine._3_applicator.applicator_tv_norm import (
    DEFAULT_LAMBDA as TV_DEFAULT_LAMBDA,
)

# Method key -> (default lambda, default iterations).
DEFAULTS = {
    "fista": (FISTA_DEFAULT_LAMBDA, FISTA_DEFAULT_ITERATIONS),
    "tv_norm": (TV_DEFAULT_LAMBDA, TV_DEFAULT_ITERATIONS),
}


def method_key(display_text: str):
    """"FISTA" -> "fista", "TV-norm" / "TV Norm" -> "tv_norm", anything else -> None."""
    key = (display_text or "").strip().lower().replace(" ", "_").replace("-", "_")
    return key if key in DEFAULTS else None


class SolverParamsWidget(QWidget):
    """Regularisation (lambda) and iteration count for FISTA and TV.

    Hook it to a mask panel's *Reconstruction method* combo with
    ``combo.currentTextChanged.connect(widget.set_method)``. It is visible only
    while an iterative solver is selected, and switching solver loads that
    solver's defaults. ``params()`` returns ``{"lam", "iterations"}`` for the
    selected solver, or None for the direct methods.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)

        lam_label = QLabel("Lambda:")
        lam_label.setStyleSheet("font-size: 11px;")
        row.addWidget(lam_label)
        self.lambda_spin = QDoubleSpinBox()
        self.lambda_spin.setRange(1e-6, 10.0)
        self.lambda_spin.setDecimals(6)
        self.lambda_spin.setSingleStep(1e-3)
        self.lambda_spin.setToolTip(
            "Regularisation strength: weight of the l1 term (FISTA) or of the\n"
            "total variation (TV) against the data-fit term."
        )
        row.addWidget(self.lambda_spin)

        it_label = QLabel("Iterations:")
        it_label.setStyleSheet("font-size: 11px;")
        row.addWidget(it_label)
        self.iterations_spin = QSpinBox()
        self.iterations_spin.setRange(1, 100000)
        self.iterations_spin.setToolTip("Number of solver iterations (there is no early stop).")
        row.addWidget(self.iterations_spin)
        row.addStretch()

        self._method = None
        self._suppressed = False
        self.hide()

    def suppress(self):
        """Never show this widget (for containers with their own controls)."""
        self._suppressed = True
        self.hide()

    def set_method(self, display_text: str):
        key = method_key(display_text)
        if key is not None and key != self._method:
            lam, iterations = DEFAULTS[key]
            self.lambda_spin.setValue(lam)
            self.iterations_spin.setValue(iterations)
        self._method = key
        self.setVisible(key is not None and not self._suppressed)

    def params(self):
        if self._method is None:
            return None
        return {"lam": float(self.lambda_spin.value()),
                "iterations": int(self.iterations_spin.value())}

    def set_params(self, lam=None, iterations=None):
        if lam is not None:
            self.lambda_spin.setValue(float(lam))
        if iterations is not None:
            self.iterations_spin.setValue(int(iterations))
