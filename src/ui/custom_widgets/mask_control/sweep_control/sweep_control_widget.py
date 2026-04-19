import logging
from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QMessageBox
from ui.custom_widgets.mask_control.sweep_control.ui_sweep_control import Ui_Sweep_Control
from ui.custom_widgets.common.button_styles import (
    BUTTON_STYLE_GREEN, BUTTON_STYLE_ORANGE, BUTTON_STYLE_RED, apply_button_style
)
from simulation_engine._2_mask_gen.mask_sweep import MaskSweep

class SweepControlWidget(QtWidgets.QWidget, Ui_Sweep_Control):
    """
    Widget to manage sweep mask parameters, with table operations (add/remove rows)
    and creation of the mask object, with logging.
    Emits:
        maskReady(mask): when a MaskSweep is created.
    """
    maskReady = pyqtSignal(object)

    def __init__(self, parent=None, logger=None):
        super().__init__(parent)
        self.setupUi(self)
        # Initialize logger
        if logger is None:
            self.logger = logging.getLogger(self.__class__.__name__)
        else:
            self.logger = logger.getChild(self.__class__.__name__)
        self.logger.debug("Initializing SweepControlWidget")

        # internal storage for image size
        self._img_size = None

        # connect UI buttons
        self.add_row_button.clicked.connect(self.add_row)
        self.logger.debug("Connected add_row_button to add_row")
        self.remove_row_button.clicked.connect(self.remove_row)
        self.logger.debug("Connected remove_row_button to remove_row")
        self.generate_masks_button.clicked.connect(self.on_create_clicked)
        self.logger.debug("Connected generate_masks_button to on_create_clicked")

        # Apply button styles - smaller circular buttons for add/remove
        self.add_row_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 14px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #45a049; }
            QPushButton:pressed { background-color: #3d8b40; }
        """)
        self.remove_row_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                border-radius: 14px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #da190b; }
            QPushButton:pressed { background-color: #c1170a; }
        """)
        apply_button_style(self.generate_masks_button, BUTTON_STYLE_GREEN)

        # configure table for dynamic rows
        self.sweep_parameters_table.setColumnCount(3)
        self.sweep_parameters_table.setHorizontalHeaderLabels(['Angle', 'Bar Width', 'Stride'])
        self.sweep_parameters_table.horizontalHeader().setStretchLastSection(True)

        # Add default sweep parameters (4 angles: 0°, 90°, 45°, -45° with bar_width=5, stride=2)
        self._add_default_rows()

        # allow widget to use its preferred size
        self.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        self.adjustSize()

    def set_img_size(self, img_size: int):
        """Set image size for mask instantiation."""
        self._img_size = img_size
        self.logger.info("Image size set to %d for sweep", img_size)

    def _add_default_rows(self):
        """Add default sweep parameters: 4 angles (0°, 90°, 45°, -45°) with bar_width=5, stride=2."""
        default_params = [
            {'angle': 0.0, 'bar_width': 5, 'stride': 2},
            {'angle': 90.0, 'bar_width': 5, 'stride': 2},
            {'angle': 45.0, 'bar_width': 5, 'stride': 2},
            {'angle': -45.0, 'bar_width': 5, 'stride': 2},
        ]
        # Clear existing rows first
        self.sweep_parameters_table.setRowCount(0)
        for params in default_params:
            row = self.sweep_parameters_table.rowCount()
            self.sweep_parameters_table.insertRow(row)
            self.sweep_parameters_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(params['angle'])))
            self.sweep_parameters_table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(params['bar_width'])))
            self.sweep_parameters_table.setItem(row, 2, QtWidgets.QTableWidgetItem(str(params['stride'])))
        self.logger.debug("Added %d default sweep rows", len(default_params))

    def add_row(self):
        """Insert a new empty row into the table."""
        row = self.sweep_parameters_table.rowCount()
        self.sweep_parameters_table.insertRow(row)
        # default values
        self.sweep_parameters_table.setItem(row, 0, QtWidgets.QTableWidgetItem('0.0'))
        self.sweep_parameters_table.setItem(row, 1, QtWidgets.QTableWidgetItem('1'))
        self.sweep_parameters_table.setItem(row, 2, QtWidgets.QTableWidgetItem('1'))
        self.logger.debug("Added row %d with default values", row)

    def remove_row(self):
        """Remove the currently selected row or last row if none selected."""
        row = self.sweep_parameters_table.currentRow()
        if row < 0:
            row = self.sweep_parameters_table.rowCount() - 1
        if row >= 0:
            self.sweep_parameters_table.removeRow(row)
            self.logger.debug("Removed row %d", row)

    def get_parameters(self):
        """Read table rows and return list of parameter dicts."""
        params = []
        for row in range(self.sweep_parameters_table.rowCount()):
            try:
                angle = float(self.sweep_parameters_table.item(row, 0).text())
                bar_width = int(self.sweep_parameters_table.item(row, 1).text())
                stride = int(self.sweep_parameters_table.item(row, 2).text())
                params.append({'angle': angle, 'bar_width': bar_width, 'stride': stride})
            except Exception as e:
                self.logger.error("Invalid values in row %d: %s", row, e)
                raise ValueError(f"Invalid values in row {row + 1}")
        self.logger.debug("Collected parameters: %s", params)
        return params

    def on_create_clicked(self):
        """Create a MaskSweep from current table and emit maskReady."""
        self.logger.debug("on_create_clicked called; img_size=%s", self._img_size)
        if self._img_size is None:
            QMessageBox.warning(self, "Error", "Image size not set. Cannot create mask.")
            self.logger.warning("Attempt to create sweep mask without img_size set")
            return
        try:
            params = self.get_parameters()
        except ValueError as e:
            QMessageBox.warning(self, "Error", str(e))
            return

        try:
            mask = MaskSweep(self._img_size, params)
            self.logger.info("MaskSweep created with %d parameters", len(params))
        except Exception as e:
            QMessageBox.critical(self, "Error creating sweep mask", str(e))
            self.logger.error("Error creating MaskSweep: %s", e, exc_info=True)
            return

        self.maskReady.emit(mask)
        self.logger.debug("maskReady emitted with %s", type(mask).__name__)
