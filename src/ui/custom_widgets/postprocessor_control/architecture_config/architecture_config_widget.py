"""
Widget for configuring neural network architecture parameters.
Provides model-specific parameter panels that swap based on selection.
"""
import logging
from typing import Dict, Any, Optional

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QGridLayout, QSpinBox,
    QDoubleSpinBox, QCheckBox, QLineEdit, QComboBox,
    QPushButton, QLabel, QGroupBox, QHBoxLayout, QFrame,
    QSizePolicy
)
from PyQt5.QtCore import pyqtSignal, Qt

from simulation_engine._4_postprocessor.architecture_schema import (
    ARCHITECTURE_SCHEMA, ParamSpec, get_schema_for_model
)


class ArchitectureConfigWidget(QWidget):
    """
    Dynamic widget for configuring model-specific architecture parameters.

    Signals:
        configChanged(dict): Emitted when any parameter value changes
        previewRequested(): Emitted when user clicks the preview button
    """

    configChanged = pyqtSignal(dict)
    previewRequested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None, logger: Optional[logging.Logger] = None):
        super().__init__(parent)

        if logger:
            self.logger = logger.getChild("ArchitectureConfigWidget")
        else:
            self.logger = logging.getLogger("ASPIR.ArchitectureConfigWidget")

        self._current_model: Optional[str] = None
        self._param_widgets: Dict[str, tuple] = {}  # name -> (widget, ParamSpec)
        self._setup_ui()

    def _setup_ui(self):
        """Setup the widget UI layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        # Main group box
        self.group_box = QGroupBox("Architecture Configuration")
        group_layout = QVBoxLayout(self.group_box)
        group_layout.setSpacing(4)
        group_layout.setContentsMargins(8, 8, 8, 8)

        # Parameters container with grid layout for two-column display
        self.params_widget = QWidget()
        self.params_layout = QGridLayout(self.params_widget)
        self.params_layout.setHorizontalSpacing(10)
        self.params_layout.setVerticalSpacing(4)
        self.params_layout.setContentsMargins(0, 0, 0, 0)
        # Set column stretch for equal width columns
        self.params_layout.setColumnStretch(1, 1)
        self.params_layout.setColumnStretch(3, 1)
        group_layout.addWidget(self.params_widget)

        # Bottom row with preview button and param count (no separator to save space)
        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(8)
        bottom_layout.setContentsMargins(0, 4, 0, 0)

        self.preview_btn = QPushButton("Preview Architecture")
        self.preview_btn.setToolTip("Show visual diagram of the neural network architecture")
        self.preview_btn.clicked.connect(self.previewRequested.emit)
        bottom_layout.addWidget(self.preview_btn)

        bottom_layout.addStretch()

        self.param_count_label = QLabel("Parameters: -")
        self.param_count_label.setStyleSheet("color: #666; font-style: italic;")
        bottom_layout.addWidget(self.param_count_label)

        group_layout.addLayout(bottom_layout)

        layout.addWidget(self.group_box)

        # Initially show placeholder message
        self._show_placeholder()

    def _show_placeholder(self):
        """Show placeholder when no model is selected."""
        self._clear_param_widgets()
        label = QLabel("Select a model to configure its architecture")
        label.setStyleSheet("color: #888; font-style: italic;")
        label.setAlignment(Qt.AlignCenter)
        self.params_layout.addWidget(label, 0, 0, 1, 4)  # Span all 4 columns

    def set_model(self, model_name: str):
        """
        Update the widget for the selected model.

        Args:
            model_name: Name of the model (case-insensitive)
        """
        # Store the canonical registry key so estimate/schema comparisons
        # can use the new kebab-case names without going through the legacy
        # alias fallback every time.
        if model_name:
            from simulation_engine._4_postprocessor.postprocessor_nn import display_to_key
            self._current_model = display_to_key(model_name)
        else:
            self._current_model = None
        self._clear_param_widgets()

        if not self._current_model:
            self._show_placeholder()
            return

        schema = get_schema_for_model(self._current_model)

        if not schema:
            label = QLabel(f"No configurable parameters for {model_name}")
            label.setStyleSheet("color: #888; font-style: italic;")
            label.setAlignment(Qt.AlignCenter)
            self.params_layout.addWidget(label, 0, 0, 1, 4)  # Span all 4 columns
            self.logger.debug(f"No schema found for model: {model_name}")
            return

        # Create widgets for each parameter in two-column layout
        # Layout: [label0][widget0][label1][widget1]
        #         [label2][widget2][label3][widget3]
        for i, param_spec in enumerate(schema):
            widget = self._create_param_widget(param_spec)
            label = QLabel(f"{param_spec.get_display_name()}:")
            label.setToolTip(param_spec.tooltip)

            # Calculate grid position (2 params per row)
            row = i // 2
            col_offset = (i % 2) * 2  # 0 for left column, 2 for right column

            self.params_layout.addWidget(label, row, col_offset)
            self.params_layout.addWidget(widget, row, col_offset + 1)
            self._param_widgets[param_spec.name] = (widget, param_spec)

        self.logger.debug(f"Configured UI for model: {model_name} with {len(schema)} parameters")
        self._update_param_count_estimate()

    def _create_param_widget(self, spec: ParamSpec) -> QWidget:
        """
        Create the appropriate widget for a parameter type.

        Args:
            spec: Parameter specification

        Returns:
            QWidget configured for the parameter type
        """
        # Expanding size policy for input widgets
        expanding_policy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        if spec.param_type == "int":
            widget = QSpinBox()
            widget.setSizePolicy(expanding_policy)
            widget.setRange(spec.min_val or 1, spec.max_val or 9999)
            if spec.step:
                widget.setSingleStep(spec.step)
            widget.setValue(spec.default)
            widget.valueChanged.connect(self._on_value_changed)

        elif spec.param_type == "float":
            widget = QDoubleSpinBox()
            widget.setSizePolicy(expanding_policy)
            widget.setRange(spec.min_val or 0.0, spec.max_val or 1.0)
            widget.setDecimals(3)
            if spec.step:
                widget.setSingleStep(spec.step)
            widget.setValue(spec.default)
            widget.valueChanged.connect(self._on_value_changed)

        elif spec.param_type == "bool":
            widget = QCheckBox()
            widget.setChecked(spec.default)
            widget.stateChanged.connect(self._on_value_changed)

        elif spec.param_type in ("list_int", "list_float"):
            widget = QLineEdit()
            widget.setSizePolicy(expanding_policy)
            # Convert list to comma-separated string
            if isinstance(spec.default, list):
                widget.setText(", ".join(str(x) for x in spec.default))
            else:
                widget.setText(str(spec.default))
            widget.setPlaceholderText("e.g., 32, 64, 128")
            widget.textChanged.connect(self._on_value_changed)

        elif spec.param_type == "choice":
            widget = QComboBox()
            widget.setSizePolicy(expanding_policy)
            if spec.choices:
                widget.addItems([str(c) for c in spec.choices])
                if spec.default in spec.choices:
                    widget.setCurrentText(str(spec.default))
            widget.currentTextChanged.connect(self._on_value_changed)

        else:
            # Fallback to label
            widget = QLabel(str(spec.default))
            self.logger.warning(f"Unknown param type: {spec.param_type}")

        widget.setToolTip(spec.tooltip)
        return widget

    def get_config(self) -> Dict[str, Any]:
        """
        Get the current architecture configuration.

        Returns:
            Dictionary of parameter names to their current values
        """
        config = {}
        for name, (widget, spec) in self._param_widgets.items():
            try:
                config[name] = self._get_widget_value(widget, spec)
            except ValueError as e:
                self.logger.warning(f"Invalid value for {name}: {e}")
                config[name] = spec.default
        return config

    def set_config(self, config: Dict[str, Any]):
        """
        Set configuration values from a dictionary.

        Args:
            config: Dictionary of parameter names to values
        """
        for name, value in config.items():
            if name in self._param_widgets:
                widget, spec = self._param_widgets[name]
                self._set_widget_value(widget, spec, value)
        # _set_widget_value blocks signals, so recompute the estimate manually.
        self._update_param_count_estimate()

    def _get_widget_value(self, widget: QWidget, spec: ParamSpec) -> Any:
        """Extract value from widget based on parameter type."""
        if spec.param_type == "int":
            return widget.value()

        elif spec.param_type == "float":
            return widget.value()

        elif spec.param_type == "bool":
            return widget.isChecked()

        elif spec.param_type == "list_int":
            text = widget.text().strip()
            if not text:
                return spec.default
            return [int(x.strip()) for x in text.split(",") if x.strip()]

        elif spec.param_type == "list_float":
            text = widget.text().strip()
            if not text:
                return spec.default
            return [float(x.strip()) for x in text.split(",") if x.strip()]

        elif spec.param_type == "choice":
            return widget.currentText()

        return None

    def _set_widget_value(self, widget: QWidget, spec: ParamSpec, value: Any):
        """Set widget value, blocking signals to avoid recursion."""
        widget.blockSignals(True)

        try:
            if spec.param_type == "int":
                widget.setValue(int(value))

            elif spec.param_type == "float":
                widget.setValue(float(value))

            elif spec.param_type == "bool":
                widget.setChecked(bool(value))

            elif spec.param_type in ("list_int", "list_float"):
                if isinstance(value, list):
                    widget.setText(", ".join(str(x) for x in value))
                else:
                    widget.setText(str(value))

            elif spec.param_type == "choice":
                widget.setCurrentText(str(value))

        finally:
            widget.blockSignals(False)

    def _on_value_changed(self, *args):
        """Handle parameter value change."""
        self._update_param_count_estimate()
        self.configChanged.emit(self.get_config())

    def _update_param_count_estimate(self):
        """Update the estimated parameter count label."""
        # For now, show a placeholder. Full calculation would require model instantiation.
        config = self.get_config()

        # Simple estimation based on model type and config
        estimate = self._estimate_params(config)
        if estimate > 0:
            if estimate >= 1_000_000:
                text = f"~{estimate / 1_000_000:.2f}M params"
            elif estimate >= 1_000:
                text = f"~{estimate / 1_000:.1f}K params"
            else:
                text = f"~{estimate} params"
            self.param_count_label.setText(text)
        else:
            self.param_count_label.setText("Parameters: -")

    def _estimate_params(self, config: Dict[str, Any]) -> int:
        """
        Rough parameter count estimation based on model architecture.

        This is an approximation - actual count requires model instantiation.
        """
        if not self._current_model:
            return 0

        model = self._current_model

        if model == "dncnn":
            features = config.get("features", 64)
            depth = config.get("depth", 17)
            # First layer: 1*features*9, middle: (depth-2)*features*features*9, last: features*1*9
            return 1 * features * 9 + (depth - 2) * features * features * 9 + features * 1 * 9

        elif model == "residual-cnn":
            features = config.get("features", 64)
            num_blocks = config.get("num_blocks", 8)
            # Head + blocks + tail
            return 1 * features * 9 + num_blocks * 2 * features * features * 9 + features * 1 * 9

        elif model == "dilated-cnn":
            features = config.get("features", 64)
            dilation_rates = config.get("dilation_rates", [1, 2, 4, 8])
            return 1 * features * 9 + len(dilation_rates) * features * features * 9 + features * 1 * 9

        elif model in ("u-net", "noise2void"):
            features = config.get("features", [8, 16, 32, 64])
            if isinstance(features, list) and len(features) > 0:
                # Rough encoder + decoder estimate
                total = 0
                prev = 1
                for f in features:
                    total += prev * f * 9 * 2  # 2 convs per level
                    prev = f
                return total * 3  # encoder + bottleneck + decoder

        elif model == "u-net-residual-attention":
            widths = config.get("widths", [32, 64, 128, 256])
            if isinstance(widths, list) and len(widths) > 0:
                total = 0
                prev = 1
                for w in widths:
                    total += prev * w * 9 * 4  # ResBlocks have more params
                    prev = w
                return total * 4  # More complex architecture

        elif model == "mobilenet-denoising":
            features = config.get("features", [32, 64, 128])
            if isinstance(features, list) and len(features) > 0:
                # Depthwise separable is more efficient
                total = 0
                prev = 1
                for f in features:
                    total += prev * 9 + prev * f  # Depthwise + Pointwise
                    prev = f
                return total

        elif model == "cgan-denoising":
            stem = config.get("stem_channels", 96)
            denoise_ch = config.get("denoise_channels", 64)
            denoise_depth = config.get("denoise_depth", 8)
            hl_blocks = config.get("hl_blocks", 4)
            # Very rough estimate for cGAN
            return stem * 100 + denoise_ch * denoise_ch * 9 * denoise_depth + hl_blocks * 10000

        elif model == "autoencoder":
            img_size = config.get("img_size", 32)
            # FC layers: input -> 1024 -> 512 -> 256 -> 128 -> 64 -> 128 -> ... -> output
            input_dim = img_size * img_size
            return input_dim * 1024 + 1024 * 512 + 512 * 256 + 256 * 128 + 128 * 64 + \
                   64 * 128 + 128 * 256 + 256 * 512 + 512 * 1024 + 1024 * input_dim

        return 0

    def _clear_param_widgets(self):
        """Remove all parameter widgets from the form."""
        while self.params_layout.count():
            item = self.params_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._param_widgets.clear()

    def get_current_model(self) -> Optional[str]:
        """Get the currently selected model name."""
        return self._current_model
