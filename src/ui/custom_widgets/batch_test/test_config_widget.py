"""
Widget for editing a single test configuration.
Embeds the existing mask control widgets from Single Test mode.
"""
import logging
from typing import Optional, List

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox,
    QGroupBox, QLabel, QScrollArea, QSizePolicy, QStackedWidget,
    QFrame
)
from PyQt5.QtCore import pyqtSignal, Qt

from ui.custom_widgets.batch_test.test_config_model import (
    TestConfiguration, BatchTestConfig
)

# Import existing mask control widgets
from ui.custom_widgets.mask_control.scatter_control.scatter_control_widget import ScatterControlWidget
from ui.custom_widgets.mask_control.hadamard_control.hadamard_control_widget import HadamardControlWidget
from ui.custom_widgets.mask_control.sweep_control.sweep_control_widget import SweepControlWidget

# Import mask classes for Hadamard variants
from simulation_engine._2_mask_gen.mask_hadamard import MaskHadamard


class TestConfigWidget(QWidget):
    """
    Widget for editing a single test configuration.
    Reuses existing mask control widgets from Single Test mode.

    Signals:
        config_changed(): Emitted when any configuration value changes
    """

    config_changed = pyqtSignal()

    def __init__(self, parent=None, logger=None):
        super().__init__(parent)

        if logger:
            self.logger = logger.getChild("TestConfigWidget")
        else:
            self.logger = logging.getLogger("TestConfigWidget")

        self._config: Optional[TestConfiguration] = None
        self._read_only = False
        self._updating = False  # Prevent signal loops
        self._img_size = 64  # Default image size

        self._setup_ui()

    def _setup_ui(self):
        """Setup the UI."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(10)

        # Header
        header = QLabel("Test Configuration")
        header.setStyleSheet("font-weight: bold; font-size: 12px;")
        main_layout.addWidget(header)

        # Scroll area for config options
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(10)

        # Test name
        name_group = QGroupBox("Identification")
        name_layout = QFormLayout(name_group)
        self.name_input = QLineEdit()
        self.name_input.textChanged.connect(self._on_value_changed)
        name_layout.addRow("Test Name:", self.name_input)
        scroll_layout.addWidget(name_group)

        # Mask configuration group
        mask_group = QGroupBox("Mask Configuration")
        mask_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        mask_layout = QVBoxLayout(mask_group)
        mask_layout.setSpacing(5)
        mask_layout.setContentsMargins(10, 10, 10, 10)

        # Mask type selector
        type_row = QHBoxLayout()
        type_row.setSpacing(15)
        type_row.addWidget(QLabel("Mask Type:"))
        self.mask_type_combo = QComboBox()
        self.mask_type_combo.addItems(BatchTestConfig.MASK_TYPES)
        self.mask_type_combo.currentTextChanged.connect(self._on_mask_type_changed)
        type_row.addWidget(self.mask_type_combo)
        type_row.addStretch()
        mask_layout.addLayout(type_row)

        # Stacked widget for mask-specific controls
        self.mask_control_stack = QStackedWidget()

        # Create and configure mask control widgets (hide generate buttons)
        self._setup_mask_controls()

        mask_layout.addWidget(self.mask_control_stack)

        # FISTA/TV parameters (show/hide based on scatter applicator selection)
        self.recon_params_widget = QWidget()
        recon_params_layout = QGridLayout(self.recon_params_widget)
        recon_params_layout.setContentsMargins(0, 5, 0, 0)
        recon_params_layout.setHorizontalSpacing(15)

        recon_params_layout.addWidget(QLabel("Lambda:"), 0, 0)
        self.recon_lambda_spin = QDoubleSpinBox()
        self.recon_lambda_spin.setRange(0.0001, 10.0)
        self.recon_lambda_spin.setDecimals(4)
        self.recon_lambda_spin.setSingleStep(0.01)
        self.recon_lambda_spin.setValue(0.01)
        self.recon_lambda_spin.valueChanged.connect(self._on_value_changed)
        recon_params_layout.addWidget(self.recon_lambda_spin, 0, 1)

        recon_params_layout.addWidget(QLabel("Iterations:"), 0, 2)
        self.recon_iter_spin = QSpinBox()
        self.recon_iter_spin.setRange(1, 1000)
        self.recon_iter_spin.setValue(100)
        self.recon_iter_spin.valueChanged.connect(self._on_value_changed)
        recon_params_layout.addWidget(self.recon_iter_spin, 0, 3)

        mask_layout.addWidget(self.recon_params_widget)
        self.recon_params_widget.hide()  # Hidden for pseudoinverse/direct

        scroll_layout.addWidget(mask_group)

        # DNN configuration
        dnn_group = QGroupBox("DNN Configuration")
        dnn_layout = QGridLayout(dnn_group)
        dnn_layout.setHorizontalSpacing(15)
        dnn_layout.setVerticalSpacing(8)

        # Row 0: Model and Epochs
        dnn_layout.addWidget(QLabel("Model:"), 0, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItems(BatchTestConfig.MODEL_NAMES)
        self.model_combo.currentTextChanged.connect(self._on_value_changed)
        dnn_layout.addWidget(self.model_combo, 0, 1)

        dnn_layout.addWidget(QLabel("Epochs:"), 0, 2)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(50)
        self.epochs_spin.valueChanged.connect(self._on_value_changed)
        dnn_layout.addWidget(self.epochs_spin, 0, 3)

        # Row 1: Batch Size and Learning Rate
        dnn_layout.addWidget(QLabel("Batch Size:"), 1, 0)
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 256)
        self.batch_size_spin.setValue(16)
        self.batch_size_spin.valueChanged.connect(self._on_value_changed)
        dnn_layout.addWidget(self.batch_size_spin, 1, 1)

        dnn_layout.addWidget(QLabel("Learning Rate:"), 1, 2)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.00001, 1.0)
        self.lr_spin.setDecimals(5)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setValue(0.001)
        self.lr_spin.valueChanged.connect(self._on_value_changed)
        dnn_layout.addWidget(self.lr_spin, 1, 3)

        # Row 2: Weight Decay and Dropout
        dnn_layout.addWidget(QLabel("Weight Decay:"), 2, 0)
        self.weight_decay_spin = QDoubleSpinBox()
        self.weight_decay_spin.setRange(0.0, 0.1)
        self.weight_decay_spin.setDecimals(6)
        self.weight_decay_spin.setSingleStep(0.0001)
        self.weight_decay_spin.setValue(0.0001)
        self.weight_decay_spin.valueChanged.connect(self._on_value_changed)
        dnn_layout.addWidget(self.weight_decay_spin, 2, 1)

        dnn_layout.addWidget(QLabel("Dropout:"), 2, 2)
        self.dropout_spin = QDoubleSpinBox()
        self.dropout_spin.setRange(0.0, 0.9)
        self.dropout_spin.setDecimals(2)
        self.dropout_spin.setSingleStep(0.05)
        self.dropout_spin.setValue(0.0)
        self.dropout_spin.valueChanged.connect(self._on_value_changed)
        dnn_layout.addWidget(self.dropout_spin, 2, 3)

        # Row 3: Loss Function and Optimizer
        dnn_layout.addWidget(QLabel("Loss Function:"), 3, 0)
        self.loss_function_combo = QComboBox()
        self.loss_function_combo.addItems(BatchTestConfig.LOSS_FUNCTIONS)
        self.loss_function_combo.currentTextChanged.connect(self._on_value_changed)
        dnn_layout.addWidget(self.loss_function_combo, 3, 1)

        dnn_layout.addWidget(QLabel("Optimizer:"), 3, 2)
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(BatchTestConfig.OPTIMIZERS)
        self.optimizer_combo.currentTextChanged.connect(self._on_value_changed)
        dnn_layout.addWidget(self.optimizer_combo, 3, 3)

        # Row 4: GPU checkbox
        self.use_gpu_checkbox = QCheckBox("Use GPU (if available)")
        self.use_gpu_checkbox.setChecked(True)
        self.use_gpu_checkbox.stateChanged.connect(self._on_value_changed)
        dnn_layout.addWidget(self.use_gpu_checkbox, 4, 0, 1, 4)

        dnn_layout.setColumnStretch(1, 1)
        dnn_layout.setColumnStretch(3, 1)

        scroll_layout.addWidget(dnn_group)

        # Dataset Split configuration
        split_group = QGroupBox("Dataset Split")
        split_layout = QHBoxLayout(split_group)
        split_layout.setSpacing(15)

        # Train split
        split_layout.addWidget(QLabel("Train:"))
        self.train_split_spin = QSpinBox()
        self.train_split_spin.setRange(1, 98)
        self.train_split_spin.setValue(80)
        self.train_split_spin.setSuffix("%")
        self.train_split_spin.setToolTip("Percentage of dataset for training")
        self.train_split_spin.valueChanged.connect(self._on_split_changed)
        split_layout.addWidget(self.train_split_spin)

        # Validation split
        split_layout.addWidget(QLabel("Validation:"))
        self.val_split_spin = QSpinBox()
        self.val_split_spin.setRange(1, 98)
        self.val_split_spin.setValue(10)
        self.val_split_spin.setSuffix("%")
        self.val_split_spin.setToolTip("Percentage of dataset for validation")
        self.val_split_spin.valueChanged.connect(self._on_split_changed)
        split_layout.addWidget(self.val_split_spin)

        # Test split
        split_layout.addWidget(QLabel("Test:"))
        self.test_split_spin = QSpinBox()
        self.test_split_spin.setRange(1, 98)
        self.test_split_spin.setValue(10)
        self.test_split_spin.setSuffix("%")
        self.test_split_spin.setToolTip("Percentage of dataset for testing")
        self.test_split_spin.valueChanged.connect(self._on_split_changed)
        split_layout.addWidget(self.test_split_spin)

        # Total label
        self.split_total_label = QLabel("= 100%")
        self.split_total_label.setStyleSheet("color: green; font-weight: bold;")
        split_layout.addWidget(self.split_total_label)

        split_layout.addStretch()
        scroll_layout.addWidget(split_group)

        # Reports configuration
        reports_group = QGroupBox("Reports to Generate")
        reports_layout = QVBoxLayout(reports_group)
        reports_layout.setSpacing(10)

        # Checkboxes row - report types with display labels
        checkboxes_row = QHBoxLayout()
        checkboxes_row.setSpacing(15)

        self.report_checkboxes = {}
        report_labels = {
            "training_curves": "Training Curves",
            "quality": "Quality",
            "timing": "Timing",
            "energy": "Energy"
        }
        for report_type in BatchTestConfig.REPORT_TYPES:
            label = report_labels.get(report_type, report_type.upper())
            cb = QCheckBox(label)
            cb.setChecked(True)  # All checked by default
            cb.stateChanged.connect(self._on_value_changed)
            if report_type == "timing":
                cb.stateChanged.connect(self._on_timing_checkbox_changed)
            self.report_checkboxes[report_type] = cb
            checkboxes_row.addWidget(cb)

        checkboxes_row.addStretch()
        reports_layout.addLayout(checkboxes_row)

        # Include datasets checkbox (separate from report types)
        self.include_datasets_checkbox = QCheckBox("Include datasets (training, validation, test images)")
        self.include_datasets_checkbox.setChecked(True)
        self.include_datasets_checkbox.setToolTip(
            "Export original images, reconstructions, and denoised outputs"
        )
        self.include_datasets_checkbox.stateChanged.connect(self._on_value_changed)
        reports_layout.addWidget(self.include_datasets_checkbox)

        # Timing analysis parameters (shown when timing checkbox is checked)
        self.timing_params_widget = QWidget()
        timing_params_layout = QGridLayout(self.timing_params_widget)
        timing_params_layout.setContentsMargins(20, 5, 0, 0)
        timing_params_layout.setHorizontalSpacing(15)
        timing_params_layout.setVerticalSpacing(8)

        timing_params_layout.addWidget(QLabel("Warmup runs:"), 0, 0)
        self.timing_warmup_spin = QSpinBox()
        self.timing_warmup_spin.setRange(0, 100)
        self.timing_warmup_spin.setValue(5)
        self.timing_warmup_spin.setToolTip("Number of warmup runs before timing measurements")
        self.timing_warmup_spin.valueChanged.connect(self._on_value_changed)
        timing_params_layout.addWidget(self.timing_warmup_spin, 0, 1)

        timing_params_layout.addWidget(QLabel("Measurement runs:"), 0, 2)
        self.timing_measurement_spin = QSpinBox()
        self.timing_measurement_spin.setRange(1, 2000)
        self.timing_measurement_spin.setValue(800)
        self.timing_measurement_spin.setToolTip("Number of runs for timing/energy measurement (higher = more accurate energy readings)")
        self.timing_measurement_spin.valueChanged.connect(self._on_value_changed)
        timing_params_layout.addWidget(self.timing_measurement_spin, 0, 3)

        timing_params_layout.addWidget(QLabel("Sampling rate:"), 1, 0)
        self.timing_sampling_spin = QDoubleSpinBox()
        self.timing_sampling_spin.setRange(0.001, 1000.0)
        self.timing_sampling_spin.setDecimals(3)
        self.timing_sampling_spin.setValue(10.752)
        self.timing_sampling_spin.setSuffix(" kHz")
        self.timing_sampling_spin.setToolTip("Hardware sampling rate for acquisition time calculation")
        self.timing_sampling_spin.valueChanged.connect(self._on_value_changed)
        timing_params_layout.addWidget(self.timing_sampling_spin, 1, 1)

        timing_params_layout.setColumnStretch(1, 1)
        timing_params_layout.setColumnStretch(3, 1)

        reports_layout.addWidget(self.timing_params_widget)

        scroll_layout.addWidget(reports_group)

        # Spacer
        scroll_layout.addStretch()

        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll, 1)

        # Placeholder when no test selected
        self.placeholder = QLabel("Select a test to view/edit its configuration")
        self.placeholder.setStyleSheet("color: #888; font-style: italic;")
        self.placeholder.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.placeholder)

        # Initially show placeholder
        scroll.hide()
        self.scroll = scroll

        # Set initial states
        self._on_mask_type_changed("scatter")
        self._on_timing_checkbox_changed()  # Show/hide timing params

    def _setup_mask_controls(self):
        """Setup the embedded mask control widgets."""
        # Make the stack widget compact - fixed height based on largest needed (sweep with table)
        self.mask_control_stack.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.mask_control_stack.setFixedHeight(150)  # Enough for sweep table

        # Scatter control (2 rows ~60px)
        self.scatter_control = ScatterControlWidget(logger=self.logger)
        self.scatter_control.generate_masks_button.hide()
        self.scatter_control.setMinimumHeight(0)
        self.scatter_control.setMaximumHeight(80)
        self._connect_scatter_signals()
        self.mask_control_stack.addWidget(self.scatter_control)

        # Hadamard control (grid + slider + percentage ~110px)
        self.hadamard_control = HadamardControlWidget(mask_cls=MaskHadamard, logger=self.logger)
        self.hadamard_control.generate_masks_button.hide()
        self.hadamard_control.setMinimumHeight(0)
        self.hadamard_control.setMaximumHeight(130)
        self._connect_hadamard_signals()
        self.mask_control_stack.addWidget(self.hadamard_control)

        # Sweep control (table widget with buttons on right)
        self.sweep_control = SweepControlWidget(logger=self.logger)
        self.sweep_control.generate_masks_button.hide()
        self.sweep_control.setMinimumHeight(100)
        self.sweep_control.setMaximumHeight(150)
        self._connect_sweep_signals()
        self.mask_control_stack.addWidget(self.sweep_control)

        # Cal-Sal placeholder
        cal_sal_widget = QWidget()
        cal_sal_layout = QHBoxLayout(cal_sal_widget)
        cal_sal_layout.setContentsMargins(0, 5, 0, 5)
        cal_sal_label = QLabel("Uses default Cal-Sal patterns")
        cal_sal_label.setStyleSheet("color: #666; font-style: italic;")
        cal_sal_layout.addWidget(cal_sal_label)
        cal_sal_layout.addStretch()
        self.mask_control_stack.addWidget(cal_sal_widget)

    def _connect_scatter_signals(self):
        """Connect scatter control signals to config updates."""
        self.scatter_control.point_density_value.valueChanged.connect(self._on_value_changed)
        self.scatter_control.number_patterns_scatter_value.valueChanged.connect(self._on_value_changed)
        self.scatter_control.random_seed_scatter_value.valueChanged.connect(self._on_value_changed)
        self.scatter_control.select_applicator_scatter_list.currentTextChanged.connect(
            self._on_scatter_applicator_changed
        )

    def _connect_hadamard_signals(self):
        """Connect hadamard control signals to config updates."""
        self.hadamard_control.hadamard_slider.valueChanged.connect(self._on_value_changed)

    def _connect_sweep_signals(self):
        """Connect sweep control signals to config updates."""
        # Sweep parameters are in a table, connect cell change signal
        self.sweep_control.sweep_parameters_table.cellChanged.connect(self._on_value_changed)

    def _on_scatter_applicator_changed(self, text: str):
        """Update reconstruction parameters visibility when scatter applicator changes."""
        if self._updating:
            return

        # Map scatter applicator to reconstruction method
        method_map = {
            "Conventional": "conventional",
            "Pseudoinverse": "pseudoinverse",
            "FISTA": "fista",
            "TV-norm": "tv_norm"
        }
        method = method_map.get(text, "conventional")

        # Show/hide FISTA/TV parameters
        show_params = method in ("fista", "tv_norm")
        self.recon_params_widget.setVisible(show_params)

        # Update default values based on method
        if method == "fista":
            self.recon_lambda_spin.setValue(0.01)
            self.recon_iter_spin.setValue(100)
        elif method == "tv_norm":
            self.recon_lambda_spin.setValue(0.1)
            self.recon_iter_spin.setValue(50)

        if self._config:
            self._on_value_changed()

    def _on_mask_type_changed(self, mask_type: str):
        """Handle mask type selection change."""
        # Map mask types to stack indices (all Hadamard variants use same controls)
        type_to_index = {
            "scatter": 0,
            "hadamard_natural": 1,
            "hadamard_scramble": 1,
            "hadamard_cake_cutting": 1,
            "hadamard_walsh_paley": 1,
            "sweep": 2,
            "cal_sal": 3,
        }
        index = type_to_index.get(mask_type, 0)
        self.mask_control_stack.setCurrentIndex(index)

        # Also trigger value change
        self._on_value_changed()


    def _on_timing_checkbox_changed(self):
        """Show/hide timing parameters based on timing checkbox."""
        timing_enabled = self.report_checkboxes.get("timing", None)
        if timing_enabled:
            self.timing_params_widget.setVisible(timing_enabled.isChecked())

    def _on_split_changed(self):
        """Handle dataset split value changes."""
        total = (self.train_split_spin.value() +
                 self.val_split_spin.value() +
                 self.test_split_spin.value())

        # Update total label with color indication
        if total == 100:
            self.split_total_label.setText("= 100%")
            self.split_total_label.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.split_total_label.setText(f"= {total}%")
            self.split_total_label.setStyleSheet("color: red; font-weight: bold;")

        # Trigger general value change
        self._on_value_changed()

    def set_img_size(self, img_size: int):
        """Set image size for mask controls."""
        self._img_size = img_size
        self.scatter_control.set_img_size(img_size)
        self.hadamard_control.set_img_size(img_size)
        self.sweep_control.set_img_size(img_size)

        # Update Hadamard slider range
        max_patterns = img_size * img_size
        self.hadamard_control.hadamard_slider.set_range(0, max_patterns)
        self.hadamard_control.number_patterns_max_hadamard_value.setText(str(max_patterns))

    def set_config(self, config: Optional[TestConfiguration]):
        """Set the configuration to display/edit."""
        self._config = config
        self._updating = True

        if config is None:
            self.scroll.hide()
            self.placeholder.show()
        else:
            self.placeholder.hide()
            self.scroll.show()

            # Populate fields
            self.name_input.setText(config.name)
            self.mask_type_combo.setCurrentText(config.mask_type)

            # Scatter params
            self.scatter_control.point_density_value.setValue(int(config.scatter_point_density))
            self.scatter_control.number_patterns_scatter_value.setValue(config.scatter_num_patterns)
            self.scatter_control.random_seed_scatter_value.setValue(config.mask_seed)

            # Hadamard params
            self.hadamard_control.hadamard_slider.set_values(
                config.hadamard_min_idx, config.hadamard_max_idx
            )

            # Sweep params - update table
            self._update_sweep_table(config.sweep_angles, config.sweep_bar_width, config.sweep_stride)

            # Reconstruction method - set via scatter applicator
            applicator_map = {
                "conventional": "Conventional",
                "pseudoinverse": "Pseudoinverse",
                "fista": "FISTA",
                "tv_norm": "TV-norm"
            }
            applicator_text = applicator_map.get(config.reconstruction_method, "Conventional")
            self.scatter_control.select_applicator_scatter_list.setCurrentText(applicator_text)

            # FISTA/TV parameters
            if config.reconstruction_method == "fista":
                self.recon_lambda_spin.setValue(config.fista_lambda)
                self.recon_iter_spin.setValue(config.fista_iterations)
                self.recon_params_widget.show()
            elif config.reconstruction_method == "tv_norm":
                self.recon_lambda_spin.setValue(config.tv_lambda)
                self.recon_iter_spin.setValue(config.tv_iterations)
                self.recon_params_widget.show()
            else:
                self.recon_params_widget.hide()

            # DNN params
            self.model_combo.setCurrentText(config.model_name)
            self.epochs_spin.setValue(config.epochs)
            self.batch_size_spin.setValue(config.batch_size)
            self.lr_spin.setValue(config.learning_rate)
            self.weight_decay_spin.setValue(config.weight_decay)
            self.dropout_spin.setValue(config.dropout)
            self.loss_function_combo.setCurrentText(config.loss_function)
            self.optimizer_combo.setCurrentText(config.optimizer)
            self.use_gpu_checkbox.setChecked(config.use_gpu)

            # Dataset split
            self.train_split_spin.setValue(config.train_split)
            self.val_split_spin.setValue(config.val_split)
            self.test_split_spin.setValue(config.test_split)
            self._on_split_changed()  # Update total label

            # Reports
            for report_type, cb in self.report_checkboxes.items():
                cb.setChecked(report_type in config.reports)

            # Include datasets checkbox
            self.include_datasets_checkbox.setChecked(config.include_datasets)

            # Timing analysis params
            self.timing_warmup_spin.setValue(config.timing_warmup_runs)
            self.timing_measurement_spin.setValue(config.timing_measurement_runs)
            self.timing_sampling_spin.setValue(config.timing_sampling_rate_khz)

            # Update stack widgets
            self._on_mask_type_changed(config.mask_type)
            self._on_timing_checkbox_changed()

        self._updating = False

    def _update_sweep_table(self, angles: List[float], bar_width: int, stride: int):
        """Update sweep parameters table."""
        table = self.sweep_control.sweep_parameters_table
        table.setRowCount(0)
        for angle in angles:
            row = table.rowCount()
            table.insertRow(row)
            from PyQt5 import QtWidgets
            table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(angle)))
            table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(bar_width)))
            table.setItem(row, 2, QtWidgets.QTableWidgetItem(str(stride)))

    def get_config(self) -> Optional[TestConfiguration]:
        """Get the current configuration."""
        return self._config

    def set_read_only(self, read_only: bool):
        """Set read-only mode."""
        self._read_only = read_only

        self.name_input.setReadOnly(read_only)
        self.mask_type_combo.setEnabled(not read_only)

        # Scatter controls
        self.scatter_control.point_density_value.setReadOnly(read_only)
        self.scatter_control.number_patterns_scatter_value.setReadOnly(read_only)
        self.scatter_control.random_seed_scatter_value.setReadOnly(read_only)
        self.scatter_control.select_applicator_scatter_list.setEnabled(not read_only)

        # Hadamard controls
        self.hadamard_control.hadamard_slider.setEnabled(not read_only)
        self.hadamard_control.percentage_spinbox.setReadOnly(read_only)

        # Sweep controls
        self.sweep_control.sweep_parameters_table.setEnabled(not read_only)
        self.sweep_control.add_row_button.setEnabled(not read_only)
        self.sweep_control.remove_row_button.setEnabled(not read_only)

        # Reconstruction params (FISTA/TV)
        self.recon_lambda_spin.setReadOnly(read_only)
        self.recon_iter_spin.setReadOnly(read_only)

        # DNN
        self.model_combo.setEnabled(not read_only)
        self.epochs_spin.setReadOnly(read_only)
        self.batch_size_spin.setReadOnly(read_only)
        self.lr_spin.setReadOnly(read_only)
        self.weight_decay_spin.setReadOnly(read_only)
        self.dropout_spin.setReadOnly(read_only)
        self.loss_function_combo.setEnabled(not read_only)
        self.optimizer_combo.setEnabled(not read_only)
        self.use_gpu_checkbox.setEnabled(not read_only)

        # Dataset split
        self.train_split_spin.setReadOnly(read_only)
        self.val_split_spin.setReadOnly(read_only)
        self.test_split_spin.setReadOnly(read_only)

        for cb in self.report_checkboxes.values():
            cb.setEnabled(not read_only)

        # Include datasets
        self.include_datasets_checkbox.setEnabled(not read_only)

        # Timing params
        self.timing_warmup_spin.setReadOnly(read_only)
        self.timing_measurement_spin.setReadOnly(read_only)
        self.timing_sampling_spin.setReadOnly(read_only)

    def _on_value_changed(self):
        """Handle value change in any field."""
        if self._updating or self._config is None or self._read_only:
            return

        # Update config from UI
        self._config.name = self.name_input.text()
        self._config.mask_type = self.mask_type_combo.currentText()

        # Scatter params
        self._config.scatter_point_density = float(self.scatter_control.point_density_value.value())
        self._config.scatter_num_patterns = self.scatter_control.number_patterns_scatter_value.value()
        self._config.mask_seed = self.scatter_control.random_seed_scatter_value.value()

        # Hadamard params
        self._config.hadamard_min_idx = self.hadamard_control.hadamard_slider.low_value
        self._config.hadamard_max_idx = self.hadamard_control.hadamard_slider.high_value

        # Sweep params - read from table
        try:
            params = self.sweep_control.get_parameters()
            if params:
                self._config.sweep_angles = [p['angle'] for p in params]
                self._config.sweep_bar_width = params[0]['bar_width']
                self._config.sweep_stride = params[0]['stride']
        except (ValueError, IndexError):
            pass  # Keep existing values if parsing fails

        # Reconstruction - get method from scatter applicator
        applicator_text = self.scatter_control.select_applicator_scatter_list.currentText()
        method_map = {
            "Conventional": "conventional",
            "Pseudoinverse": "pseudoinverse",
            "FISTA": "fista",
            "TV-norm": "tv_norm"
        }
        self._config.reconstruction_method = method_map.get(applicator_text, "conventional")
        if self._config.reconstruction_method == "fista":
            self._config.fista_lambda = self.recon_lambda_spin.value()
            self._config.fista_iterations = self.recon_iter_spin.value()
        elif self._config.reconstruction_method == "tv_norm":
            self._config.tv_lambda = self.recon_lambda_spin.value()
            self._config.tv_iterations = self.recon_iter_spin.value()

        # DNN params
        self._config.model_name = self.model_combo.currentText()
        self._config.epochs = self.epochs_spin.value()
        self._config.batch_size = self.batch_size_spin.value()
        self._config.learning_rate = self.lr_spin.value()
        self._config.weight_decay = self.weight_decay_spin.value()
        self._config.dropout = self.dropout_spin.value()
        self._config.loss_function = self.loss_function_combo.currentText()
        self._config.optimizer = self.optimizer_combo.currentText()
        self._config.use_gpu = self.use_gpu_checkbox.isChecked()

        # Dataset split
        self._config.train_split = self.train_split_spin.value()
        self._config.val_split = self.val_split_spin.value()
        self._config.test_split = self.test_split_spin.value()

        # Reports
        self._config.reports = [
            report_type for report_type, cb in self.report_checkboxes.items()
            if cb.isChecked()
        ]

        # Include datasets
        self._config.include_datasets = self.include_datasets_checkbox.isChecked()

        # Timing params
        self._config.timing_warmup_runs = self.timing_warmup_spin.value()
        self._config.timing_measurement_runs = self.timing_measurement_spin.value()
        self._config.timing_sampling_rate_khz = self.timing_sampling_spin.value()

        self.config_changed.emit()
