# File: ui/custom_widgets/common/data_format_selector.py
"""
Data format selector widget for choosing quantization precision.
Used across all dataset generator widgets to test different data formats
for embedded system deployment (FPGA, microcontrollers, etc.).
"""

import logging
from enum import Enum
from PyQt5 import QtWidgets, QtCore
import numpy as np
import torch


class DataFormat(Enum):
    """Supported data formats for quantization testing."""
    FP32 = "FP32"   # 32-bit floating point (default, for computers)
    INT8 = "INT8"   # 8-bit integer quantization (for embedded systems)
    INT4 = "INT4"   # 4-bit integer quantization (for FPGA)

    @property
    def numpy_dtype(self):
        """Return the numpy dtype for this format."""
        # Note: INT4 and INT8 are stored as float32 after quantization simulation
        # to maintain consistent value range [0, 1]
        mapping = {
            DataFormat.FP32: np.float32,
            DataFormat.INT8: np.float32,  # Stored as float32 after quantization
            DataFormat.INT4: np.float32,  # Stored as float32 after quantization
        }
        return mapping[self]

    @property
    def torch_dtype(self):
        """Return the PyTorch dtype for this format."""
        mapping = {
            DataFormat.FP32: torch.float32,
            DataFormat.INT8: torch.float32,  # Quantized but stored as float32
            DataFormat.INT4: torch.float32,  # Quantized but stored as float32
        }
        return mapping[self]

    @property
    def quantization_levels(self):
        """Return the number of quantization levels for this format."""
        levels = {
            DataFormat.FP32: None,  # Continuous (no quantization)
            DataFormat.INT8: 256,   # 2^8 levels
            DataFormat.INT4: 16,    # 2^4 levels
        }
        return levels[self]

    @property
    def description(self):
        """Return a human-readable description of this format."""
        descriptions = {
            DataFormat.FP32: "32-bit float (full precision)",
            DataFormat.INT8: "8-bit integer (256 levels)",
            DataFormat.INT4: "4-bit integer (16 levels)",
        }
        return descriptions[self]


class DataFormatSelector(QtWidgets.QWidget):
    """
    A reusable widget for selecting data format (quantization precision).
    Emits formatChanged signal when the selection changes.
    """
    formatChanged = QtCore.pyqtSignal(str)  # Emits the format name as string

    def __init__(self, parent=None, logger=None):
        """
        Initialize the data format selector.

        Args:
            parent: Parent widget
            logger: Logger instance
        """
        super().__init__(parent)

        if logger is None:
            self.logger = logging.getLogger(self.__class__.__name__)
        else:
            self.logger = logger.getChild(self.__class__.__name__)

        self._setup_ui()
        self.logger.debug("DataFormatSelector initialized")

    def _setup_ui(self):
        """Set up the widget UI."""
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Label
        self.label = QtWidgets.QLabel("Data Format:")
        layout.addWidget(self.label)

        # ComboBox
        self.format_combo = QtWidgets.QComboBox()
        self.format_combo.setObjectName("data_format_combobox")
        self.format_combo.setToolTip(
            "Select the data precision format for testing.\n"
            "FP32: Full precision for computers\n"
            "INT8: 8-bit quantization for embedded systems\n"
            "INT4: 4-bit quantization for FPGA deployment"
        )

        # Add format options
        formats = [DataFormat.FP32, DataFormat.INT8, DataFormat.INT4]

        for fmt in formats:
            self.format_combo.addItem(f"{fmt.value} - {fmt.description}", fmt.value)

        # Set default to FP32
        self.format_combo.setCurrentIndex(0)

        # Connect signal
        self.format_combo.currentIndexChanged.connect(self._on_format_changed)

        layout.addWidget(self.format_combo, 1)

    def _on_format_changed(self, index: int):
        """Handle format selection change."""
        fmt = self.get_format()
        self.logger.debug("Data format changed to: %s", fmt)
        self.formatChanged.emit(fmt)

    def get_format(self) -> str:
        """Get the currently selected format as string."""
        return self.format_combo.currentData()

    def get_format_enum(self) -> DataFormat:
        """Get the currently selected format as DataFormat enum."""
        return DataFormat(self.get_format())

    def set_format(self, format_str: str):
        """Set the format by string value."""
        for i in range(self.format_combo.count()):
            if self.format_combo.itemData(i) == format_str:
                self.format_combo.setCurrentIndex(i)
                return
        self.logger.warning("Unknown format: %s", format_str)

    def set_format_index(self, index: int):
        """Set the format by combo index."""
        if 0 <= index < self.format_combo.count():
            self.format_combo.setCurrentIndex(index)

    def get_format_index(self) -> int:
        """Get the current combo index."""
        return self.format_combo.currentIndex()


def convert_to_format(data: np.ndarray, target_format: DataFormat, logger=None) -> np.ndarray:
    """
    Convert numpy array to the specified data format.
    All formats output values in [0, 1] range for consistency.

    Args:
        data: Input numpy array
        target_format: Target DataFormat
        logger: Optional logger

    Returns:
        Converted numpy array (always float32 with quantization applied)
    """
    if logger:
        logger.debug("Converting data from %s to %s", data.dtype, target_format.value)

    # Normalize input to float32 [0, 1] first
    if data.dtype == np.uint8:
        data_f32 = data.astype(np.float32) / 255.0
    else:
        data_f32 = data.astype(np.float32)

    if target_format == DataFormat.FP32:
        return data_f32

    elif target_format == DataFormat.INT8:
        # Quantize to 256 levels (8-bit)
        num_levels = 256
        min_val = data_f32.min()
        max_val = data_f32.max()
        if max_val - min_val > 0:
            normalized = (data_f32 - min_val) / (max_val - min_val)
            quantized = np.round(normalized * (num_levels - 1)) / (num_levels - 1)
            result = quantized * (max_val - min_val) + min_val
        else:
            result = data_f32
        return result.astype(np.float32)

    elif target_format == DataFormat.INT4:
        # Quantize to 16 levels (4-bit)
        num_levels = 16
        min_val = data_f32.min()
        max_val = data_f32.max()
        if max_val - min_val > 0:
            normalized = (data_f32 - min_val) / (max_val - min_val)
            quantized = np.round(normalized * (num_levels - 1)) / (num_levels - 1)
            result = quantized * (max_val - min_val) + min_val
        else:
            result = data_f32
        return result.astype(np.float32)

    else:
        raise ValueError(f"Unsupported format: {target_format}")


def convert_tensor_to_format(tensor: torch.Tensor, target_format: DataFormat,
                              logger=None) -> torch.Tensor:
    """
    Convert PyTorch tensor to the specified data format.
    All formats output values in [0, 1] range for consistency.

    Args:
        tensor: Input PyTorch tensor
        target_format: Target DataFormat
        logger: Optional logger

    Returns:
        Converted PyTorch tensor (always float32 with quantization applied)
    """
    if logger:
        logger.debug("Converting tensor from %s to %s", tensor.dtype, target_format.value)

    # Ensure float32 for processing
    tensor_f32 = tensor.float()

    if target_format == DataFormat.FP32:
        return tensor_f32

    elif target_format == DataFormat.INT8:
        # Quantize to 256 levels (8-bit)
        num_levels = 256
        min_val = tensor_f32.min()
        max_val = tensor_f32.max()
        if max_val - min_val > 0:
            normalized = (tensor_f32 - min_val) / (max_val - min_val)
            quantized = torch.round(normalized * (num_levels - 1)) / (num_levels - 1)
            result = quantized * (max_val - min_val) + min_val
        else:
            result = tensor_f32
        return result.float()

    elif target_format == DataFormat.INT4:
        # Quantize to 16 levels (4-bit)
        num_levels = 16
        min_val = tensor_f32.min()
        max_val = tensor_f32.max()
        if max_val - min_val > 0:
            normalized = (tensor_f32 - min_val) / (max_val - min_val)
            quantized = torch.round(normalized * (num_levels - 1)) / (num_levels - 1)
            result = quantized * (max_val - min_val) + min_val
        else:
            result = tensor_f32
        return result.float()

    else:
        raise ValueError(f"Unsupported format: {target_format}")
