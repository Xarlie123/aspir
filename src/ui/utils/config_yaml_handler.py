"""
Handler for saving and loading UI configuration.

Supports both the new JSON format (.single_test_config) and legacy YAML format.
"""
import json
import yaml
import os
import logging
from pathlib import Path
from PyQt5.QtWidgets import (
    QWidget, QSpinBox, QDoubleSpinBox, QLineEdit,
    QCheckBox, QComboBox, QTableWidgetItem, QMessageBox
)
from ui.custom_widgets.mask_control.sweep_control.sweep_control_widget import SweepControlWidget
from ui.custom_widgets.mask_control.scatter_control.scatter_control_widget import ScatterControlWidget
from ui.custom_widgets.common.data_format_selector import DataFormatSelector
from ui.utils.file_formats import (
    FileExtensions, SingleTestConfig, SINGLE_TESTS_DIR
)


class ConfigYamlHandler:
    """
    Handler for saving and loading UI configuration.

    Now uses JSON format with .single_test_config extension by default.
    Maintains backward compatibility with legacy .yaml files.
    """

    def __init__(self, ui, mascara_handler, logger=None, dataset_handler=None):
        self.ui = ui
        self.mascara_handler = mascara_handler
        self.dataset_handler = dataset_handler
        if logger is None:
            self.logger = logging.getLogger(self.__class__.__name__)
        else:
            self.logger = logger.getChild(self.__class__.__name__)
        self.logger.debug("ConfigYamlHandler initialized")

    def _collect_config_data(self) -> dict:
        """Collect all configuration data from widgets."""
        data = {
            "format_version": "2.0",
            "widgets": {},
            "sweep_params": [],
            "scatter": {},
            "hadamard": [],
            "data_formats": {},
        }

        # 1) Basic values of standard widgets
        for widget in self.ui.centralwidget.findChildren(
                (QSpinBox, QDoubleSpinBox, QLineEdit, QCheckBox, QComboBox)):
            name = widget.objectName()
            if not name:
                continue
            # Skip internal QLineEdit of spinboxes
            if isinstance(widget, QLineEdit) and isinstance(widget.parent(), (QSpinBox, QDoubleSpinBox)):
                continue

            if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                data["widgets"][name] = widget.value()
            elif isinstance(widget, QLineEdit):
                data["widgets"][name] = widget.text()
            elif isinstance(widget, QCheckBox):
                data["widgets"][name] = widget.isChecked()
            elif isinstance(widget, QComboBox):
                data["widgets"][name] = widget.currentIndex()
            self.logger.debug(f"Collected widget {name}: {data['widgets'].get(name)}")

        # 2) Sweep table
        sweep: SweepControlWidget = getattr(self.mascara_handler, 'sweep_control', None)
        if sweep:
            data['sweep_params'] = sweep.get_parameters()
            self.logger.debug(f"Collected sweep_params: {data['sweep_params']}")

        # 3) ScatterControlWidget
        scatter: ScatterControlWidget = getattr(self.mascara_handler, 'scatter_control', None)
        if scatter:
            data['scatter'] = {
                'point_density': scatter.point_density_value.value(),
                'num_patterns': scatter.number_patterns_scatter_value.value(),
                'seed': scatter.random_seed_scatter_value.value()
            }
            self.logger.debug(f"Collected scatter: {data['scatter']}")

        # 4) Hadamard sliders
        for ctrl in getattr(self.mascara_handler, 'hadamard_controls', []):
            entry = {
                'low': ctrl.hadamard_slider.low_value,
                'high': ctrl.hadamard_slider.high_value,
                'mask_cls': ctrl._mask_cls.__name__
            }
            data['hadamard'].append(entry)
            self.logger.debug(f"Collected hadamard ctrl: {entry}")

        # 5) Data format selectors from dataset widgets
        if self.dataset_handler is not None:
            if hasattr(self.dataset_handler, 'ir_widget') and hasattr(self.dataset_handler.ir_widget, 'data_format_selector'):
                data['data_formats']['ir_profile'] = self.dataset_handler.ir_widget.data_format_selector.get_format()
            if hasattr(self.dataset_handler, 'selecciona_imagen_widget') and hasattr(self.dataset_handler.selecciona_imagen_widget, 'data_format_selector'):
                data['data_formats']['single_image'] = self.dataset_handler.selecciona_imagen_widget.data_format_selector.get_format()
            if hasattr(self.dataset_handler, 'selecciona_directorio_imagen_widget') and hasattr(self.dataset_handler.selecciona_directorio_imagen_widget, 'data_format_selector'):
                data['data_formats']['folder'] = self.dataset_handler.selecciona_directorio_imagen_widget.data_format_selector.get_format()
            if hasattr(self.dataset_handler, 'internet_widget') and hasattr(self.dataset_handler.internet_widget, 'data_format_selector'):
                data['data_formats']['internet'] = self.dataset_handler.internet_widget.data_format_selector.get_format()
            self.logger.debug(f"Collected data_formats: {data['data_formats']}")

        return data

    def _apply_config_data(self, data: dict):
        """Apply configuration data to widgets."""
        # Handle both new format (with 'widgets' key) and legacy format (flat structure)
        widgets_data = data.get('widgets', data)  # Fallback to flat structure for legacy

        # 1) Restore standard widgets
        for widget in self.ui.centralwidget.findChildren(
                (QSpinBox, QDoubleSpinBox, QLineEdit, QCheckBox, QComboBox)):
            name = widget.objectName()
            if name in widgets_data:
                val = widgets_data[name]
                if isinstance(widget, (QSpinBox, QDoubleSpinBox)) and isinstance(val, (int, float)):
                    widget.setValue(val)
                elif isinstance(widget, QLineEdit) and isinstance(val, str):
                    widget.setText(val)
                elif isinstance(widget, QCheckBox) and isinstance(val, bool):
                    widget.setChecked(val)
                elif isinstance(widget, QComboBox) and isinstance(val, int):
                    widget.setCurrentIndex(val)
                self.logger.debug(f"Restored widget {name}: {val}")

        # 2) SweepControlWidget
        sweep: SweepControlWidget = getattr(self.mascara_handler, 'sweep_control', None)
        if sweep and 'sweep_params' in data:
            sweep.sweep_parameters_table.setRowCount(0)
            for params in data['sweep_params']:
                sweep.add_row()
                row = sweep.sweep_parameters_table.rowCount() - 1
                # Support both old Spanish keys and new English keys
                angle = params.get('angle') if 'angle' in params else params.get('ángulo')
                bar_width = params.get('bar_width') if 'bar_width' in params else params.get('ancho_barra')
                stride = params.get('stride', 0)

                if angle is not None:
                    sweep.sweep_parameters_table.item(row, 0).setText(str(angle))
                if bar_width is not None:
                    sweep.sweep_parameters_table.item(row, 1).setText(str(bar_width))
                if stride is not None:
                    sweep.sweep_parameters_table.item(row, 2).setText(str(stride))
                self.logger.debug(f"Restored sweep row {row}: angle={angle}, bar_width={bar_width}, stride={stride}")

        # 3) ScatterControlWidget
        scatter: ScatterControlWidget = getattr(self.mascara_handler, 'scatter_control', None)
        if scatter and 'scatter' in data:
            sc = data['scatter']
            # Support both old Spanish keys and new English keys
            scatter.point_density_value.setValue(
                sc.get('point_density', sc.get('densidad', 0))
            )
            scatter.number_patterns_scatter_value.setValue(
                sc.get('num_patterns', sc.get('n_patrones', 0))
            )
            scatter.random_seed_scatter_value.setValue(
                sc.get('seed', sc.get('semilla', 0))
            )
            self.logger.debug(f"Restored scatter: {sc}")

        # 4) Hadamard sliders
        had_data = data.get('hadamard', [])
        for ctrl, vals in zip(getattr(self.mascara_handler, 'hadamard_controls', []), had_data):
            ctrl.hadamard_slider.low_value = vals.get('low', 0)
            ctrl.hadamard_slider.high_value = vals.get('high', ctrl.hadamard_slider.max_val)
            ctrl.hadamard_slider.update()
            self.logger.debug(f"Restored hadamard ctrl: {vals}")

        # 5) Data format selectors
        if self.dataset_handler is not None and 'data_formats' in data:
            formats = data['data_formats']
            if 'ir_profile' in formats and hasattr(self.dataset_handler, 'ir_widget'):
                if hasattr(self.dataset_handler.ir_widget, 'data_format_selector'):
                    self.dataset_handler.ir_widget.data_format_selector.set_format(formats['ir_profile'])
            if 'single_image' in formats and hasattr(self.dataset_handler, 'selecciona_imagen_widget'):
                if hasattr(self.dataset_handler.selecciona_imagen_widget, 'data_format_selector'):
                    self.dataset_handler.selecciona_imagen_widget.data_format_selector.set_format(formats['single_image'])
            if 'folder' in formats and hasattr(self.dataset_handler, 'selecciona_directorio_imagen_widget'):
                if hasattr(self.dataset_handler.selecciona_directorio_imagen_widget, 'data_format_selector'):
                    self.dataset_handler.selecciona_directorio_imagen_widget.data_format_selector.set_format(formats['folder'])
            if 'internet' in formats and hasattr(self.dataset_handler, 'internet_widget'):
                if hasattr(self.dataset_handler.internet_widget, 'data_format_selector'):
                    self.dataset_handler.internet_widget.data_format_selector.set_format(formats['internet'])

    def save_config(self, file_path: str) -> bool:
        """
        Save configuration to JSON file with .single_test_config extension.

        Args:
            file_path: Path to save the configuration

        Returns:
            True if successful, False otherwise
        """
        self.logger.info(f"Saving configuration to '{file_path}'")

        # Ensure correct extension
        path = Path(file_path)
        if path.suffix != FileExtensions.SINGLE_TEST_CONFIG:
            path = path.with_suffix(FileExtensions.SINGLE_TEST_CONFIG)
            self.logger.debug(f"Extension changed to {FileExtensions.SINGLE_TEST_CONFIG}")

        try:
            data = self._collect_config_data()
            SingleTestConfig.save(data, path)
            self.logger.info(f"Configuration saved successfully to '{path}'")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}", exc_info=True)
            return False

    def load_config(self, file_path: str) -> bool:
        """
        Load configuration from JSON or YAML file.

        Supports both new .single_test_config (JSON) and legacy .yaml formats.

        Args:
            file_path: Path to the configuration file

        Returns:
            True if successful, False otherwise
        """
        self.logger.info(f"Loading configuration from '{file_path}'")

        if not file_path or not os.path.exists(file_path):
            self.logger.error(f"File not found: '{file_path}'")
            QMessageBox.warning(None, "Error", f"File not found:\n{file_path}")
            return False

        try:
            path = Path(file_path)

            # Determine format based on extension
            if path.suffix == FileExtensions.SINGLE_TEST_CONFIG or path.suffix == '.json':
                data = SingleTestConfig.load(path)
            elif path.suffix in ('.yaml', '.yml'):
                # Legacy YAML format
                with open(path, 'r') as f:
                    data = yaml.safe_load(f) or {}
            else:
                # Try JSON first, fallback to YAML
                try:
                    data = SingleTestConfig.load(path)
                except json.JSONDecodeError:
                    with open(path, 'r') as f:
                        data = yaml.safe_load(f) or {}

            self._apply_config_data(data)
            self.logger.info(f"Configuration loaded successfully from '{file_path}'")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}", exc_info=True)
            QMessageBox.warning(None, "Error", f"Failed to load configuration:\n{e}")
            return False

    # Legacy method names for backward compatibility
    def save_to_yaml(self, file_path: str):
        """Legacy method - redirects to save_config with JSON format."""
        # Change extension from .yaml to .single_test_config
        path = Path(file_path)
        if path.suffix in ('.yaml', '.yml'):
            path = path.with_suffix(FileExtensions.SINGLE_TEST_CONFIG)
        return self.save_config(str(path))

    def load_from_yaml(self, file_path: str):
        """Legacy method - redirects to load_config (supports both formats)."""
        return self.load_config(file_path)

    @staticmethod
    def get_default_directory() -> Path:
        """Get the default directory for single test configs."""
        SINGLE_TESTS_DIR.mkdir(parents=True, exist_ok=True)
        return SINGLE_TESTS_DIR

    @staticmethod
    def get_file_filter() -> str:
        """Get file filter for open/save dialogs."""
        return (
            f"Single Test Config (*{FileExtensions.SINGLE_TEST_CONFIG});;"
            "Legacy YAML (*.yaml *.yml);;"
            "All Files (*.*)"
        )
