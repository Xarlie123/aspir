"""Popup dialog for NVIDIA Nsight Systems profiling configuration and execution."""
import logging
import subprocess
import shutil
from pathlib import Path
from typing import Optional

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QGridLayout, QFileDialog, QCheckBox, QSpinBox,
    QLineEdit, QMessageBox, QProgressBar, QTextEdit, QApplication,
    QSizePolicy
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QProcess
from PyQt5.QtGui import QFont


class NsightWorker(QThread):
    """Worker thread for running nsys profile."""
    finished = pyqtSignal(bool, str)  # success, message
    output = pyqtSignal(str)  # stdout/stderr output

    def __init__(self, command: list, parent=None):
        super().__init__(parent)
        self.command = command
        self._process = None

    def run(self):
        try:
            self._process = subprocess.Popen(
                self.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            for line in iter(self._process.stdout.readline, ''):
                if line:
                    self.output.emit(line.strip())

            self._process.wait()

            if self._process.returncode == 0:
                self.finished.emit(True, "Profiling completed successfully")
            else:
                self.finished.emit(False, f"Profiling failed with code {self._process.returncode}")

        except Exception as e:
            self.finished.emit(False, str(e))

    def stop(self):
        if self._process:
            self._process.terminate()


class NsightProfilerPopup(QDialog):
    """
    Popup dialog for configuring and running NVIDIA Nsight Systems profiling.
    """

    def __init__(self, parent=None, simulation=None, logger=None):
        super().__init__(parent)
        self.setWindowTitle("NVIDIA Nsight Systems Profiler")
        self.setMinimumSize(700, 600)
        self.resize(750, 650)

        if logger:
            self.logger = logger.getChild("NsightProfilerPopup")
        else:
            self.logger = logging.getLogger("SPIm.NsightProfilerPopup")

        self.simulation = simulation
        self._worker = None
        self._output_file = None

        # Check if nsys is available
        self._nsys_available = self._check_nsys()

        self._setup_ui()

    def _check_nsys(self) -> bool:
        """Check if nsys is available in PATH."""
        return shutil.which("nsys") is not None

    def _setup_ui(self):
        """Setup the popup UI layout."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # Title
        title = QLabel("<h2>NVIDIA Nsight Systems Profiler</h2>")
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)

        # Nsys availability check
        if not self._nsys_available:
            warning = QLabel(
                "<span style='color: #F44336;'>⚠ nsys not found in PATH</span><br>"
                "<small>Install NVIDIA Nsight Systems or add it to PATH</small>"
            )
            warning.setAlignment(Qt.AlignCenter)
            main_layout.addWidget(warning)

        # Description
        desc = QLabel(
            "Profile your DNN inference with detailed CPU↔GPU analysis including "
            "memory transfers, kernel launches, and synchronization points."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #666; padding: 5px;")
        main_layout.addWidget(desc)

        # Configuration section
        config_group = QGroupBox("Profiling Options")
        config_layout = QGridLayout(config_group)
        config_layout.setSpacing(10)

        row = 0

        # Trace options
        config_layout.addWidget(QLabel("<b>Trace:</b>"), row, 0)
        row += 1

        self.trace_cuda = QCheckBox("CUDA (kernels, memcpy)")
        self.trace_cuda.setChecked(True)
        self.trace_cuda.setToolTip("Trace CUDA API calls and kernel executions")
        config_layout.addWidget(self.trace_cuda, row, 0)

        self.trace_nvtx = QCheckBox("NVTX (markers)")
        self.trace_nvtx.setChecked(True)
        self.trace_nvtx.setToolTip("Trace NVTX annotations for code regions")
        config_layout.addWidget(self.trace_nvtx, row, 1)

        self.trace_osrt = QCheckBox("OS Runtime")
        self.trace_osrt.setChecked(True)
        self.trace_osrt.setToolTip("Trace OS runtime calls (threads, sync)")
        config_layout.addWidget(self.trace_osrt, row, 2)
        row += 1

        self.trace_cudnn = QCheckBox("cuDNN")
        self.trace_cudnn.setChecked(False)
        self.trace_cudnn.setToolTip("Trace cuDNN library calls")
        config_layout.addWidget(self.trace_cudnn, row, 0)

        self.trace_cublas = QCheckBox("cuBLAS")
        self.trace_cublas.setChecked(False)
        self.trace_cublas.setToolTip("Trace cuBLAS library calls")
        config_layout.addWidget(self.trace_cublas, row, 1)
        row += 1

        # Memory options
        config_layout.addWidget(QLabel("<b>Memory:</b>"), row, 0)
        row += 1

        self.cuda_memory = QCheckBox("Track CUDA memory usage")
        self.cuda_memory.setChecked(True)
        self.cuda_memory.setToolTip("Track GPU memory allocations and usage")
        config_layout.addWidget(self.cuda_memory, row, 0, 1, 2)
        row += 1

        # Number of images
        config_layout.addWidget(QLabel("<b>Images to profile:</b>"), row, 0)
        self.num_images_spin = QSpinBox()
        self.num_images_spin.setRange(1, 100)
        self.num_images_spin.setValue(10)
        self.num_images_spin.setToolTip("Number of images to run through the model")
        config_layout.addWidget(self.num_images_spin, row, 1)
        row += 1

        # Output directory
        config_layout.addWidget(QLabel("<b>Output directory:</b>"), row, 0)
        output_layout = QHBoxLayout()
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setText(str(Path.home() / "nsight_profiles"))
        self.output_dir_edit.setToolTip("Directory to save profiling results")
        output_layout.addWidget(self.output_dir_edit)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_output_dir)
        output_layout.addWidget(browse_btn)
        config_layout.addLayout(output_layout, row, 1, 1, 2)

        main_layout.addWidget(config_group)

        # Progress section (expandable)
        progress_group = QGroupBox("Progress")
        progress_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        progress_layout = QVBoxLayout(progress_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar)

        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setFont(QFont("Monospace", 9))
        self.output_text.setMinimumHeight(150)
        self.output_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.output_text.setPlaceholderText("Profiling output will appear here...")
        progress_layout.addWidget(self.output_text)

        main_layout.addWidget(progress_group, 1)  # stretch factor 1 to expand

        # Buttons
        buttons_layout = QHBoxLayout()

        self.run_button = QPushButton("Run Profiling")
        self.run_button.setMinimumHeight(40)
        self.run_button.setStyleSheet("""
            QPushButton {
                background-color: #76B900;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover { background-color: #5A8F00; }
            QPushButton:disabled { background-color: #ccc; color: #666; }
        """)
        self.run_button.clicked.connect(self._on_run_profiling)
        self.run_button.setEnabled(self._nsys_available)
        buttons_layout.addWidget(self.run_button)

        self.open_button = QPushButton("Open Results")
        self.open_button.setMinimumHeight(40)
        self.open_button.setEnabled(False)
        self.open_button.clicked.connect(self._on_open_results)
        self.open_button.setToolTip("Open the .nsys-rep file in Nsight Systems GUI")
        buttons_layout.addWidget(self.open_button)

        buttons_layout.addStretch()

        self.close_button = QPushButton("Close")
        self.close_button.setMinimumHeight(40)
        self.close_button.clicked.connect(self.close)
        buttons_layout.addWidget(self.close_button)

        main_layout.addLayout(buttons_layout)

    def _browse_output_dir(self):
        """Browse for output directory."""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Output Directory",
            self.output_dir_edit.text()
        )
        if dir_path:
            self.output_dir_edit.setText(dir_path)

    def _on_run_profiling(self):
        """Start the Nsight profiling."""
        if self.simulation is None:
            QMessageBox.warning(self, "Error", "No simulation available")
            return

        post = getattr(self.simulation, 'postprocessor', None)
        if post is None or not getattr(post, 'trained', False):
            QMessageBox.warning(self, "Error", "No trained model available")
            return

        # Create output directory
        output_dir = Path(self.output_dir_edit.text())
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate profiling script
        script_path = output_dir / "nsight_inference_script.py"
        self._generate_profiling_script(script_path)

        # Build nsys command
        cmd = self._build_nsys_command(script_path, output_dir)

        self.logger.info(f"Running: {' '.join(cmd)}")
        self.output_text.clear()
        self.output_text.append(f"Command: {' '.join(cmd)}\n")
        self.output_text.append("-" * 50 + "\n")

        # Start worker
        self.run_button.setEnabled(False)
        self.progress_bar.setVisible(True)

        self._worker = NsightWorker(cmd)
        self._worker.output.connect(self._on_worker_output)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def _build_nsys_command(self, script_path: Path, output_dir: Path) -> list:
        """Build the nsys command with selected options."""
        import sys
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_name = output_dir / f"inference_profile_{timestamp}"
        self._output_file = str(output_name) + ".nsys-rep"

        cmd = [
            "nsys", "profile",
            "--stats=true",
            f"--output={output_name}",
            "--force-overwrite=true",
        ]

        # Trace options
        traces = []
        if self.trace_cuda.isChecked():
            traces.append("cuda")
        if self.trace_nvtx.isChecked():
            traces.append("nvtx")
        if self.trace_osrt.isChecked():
            traces.append("osrt")
        if self.trace_cudnn.isChecked():
            traces.append("cudnn")
        if self.trace_cublas.isChecked():
            traces.append("cublas")

        if traces:
            cmd.append(f"--trace={','.join(traces)}")

        # Memory option
        if self.cuda_memory.isChecked():
            cmd.append("--cuda-memory-usage=true")

        # Use the same Python interpreter that's running this application (virtualenv)
        python_executable = sys.executable
        cmd.extend([python_executable, str(script_path)])

        return cmd

    def _generate_profiling_script(self, script_path: Path):
        """Generate Python script for profiling with the actual trained model."""
        import torch

        num_images = self.num_images_spin.value()
        output_dir = script_path.parent

        # Get model and input size from postprocessor
        post = self.simulation.postprocessor
        input_size = 128  # Default
        model_path = output_dir / "model_for_profiling.pt"
        model_class_name = "Unknown"

        if hasattr(post, 'model') and post.model is not None:
            model = post.model
            model_class_name = model.__class__.__name__

            # Try to infer input size from dataset
            if hasattr(post, 'loaders') and 'val' in post.loaders:
                val_loader = post.loaders['val']
                for batch, _ in val_loader:
                    input_size = batch.shape[-1]
                    break

            # Export the model to a file
            try:
                # Move to CPU before saving to avoid GPU memory issues
                model_cpu = model.cpu()
                torch.save(model_cpu, model_path)
                # Move back to original device
                if hasattr(post, 'device'):
                    model.to(post.device)
                self.logger.info(f"Exported model to: {model_path}")
            except Exception as e:
                self.logger.error(f"Failed to export model: {e}")
                model_path = None
        else:
            model_path = None
            self.logger.warning("No trained model found, will use placeholder")

        # Generate the profiling script
        src_path = Path.cwd() / 'src'
        model_path_str = str(model_path) if model_path else "None"

        script = f'''#!/usr/bin/env python3
"""
Auto-generated Nsight Systems profiling script.
Generated by SPIm Profiler

Model: {model_class_name}
Input size: {input_size}x{input_size}
Images to profile: {num_images}
"""
import sys
sys.path.insert(0, "{src_path}")

import torch
import numpy as np

def main():
    print("=" * 60)
    print("NSIGHT SYSTEMS PROFILING")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        sys.exit(1)

    device = torch.device("cuda")
    print(f"Device: {{torch.cuda.get_device_name(0)}}")

    # Configuration
    num_images = {num_images}
    input_size = {input_size}
    model_path = "{model_path_str}"

    # Load the trained model
    print(f"\\nLoading model from: {{model_path}}")
    if model_path != "None" and model_path:
        try:
            model = torch.load(model_path, map_location='cpu', weights_only=False)
            model = model.to(device)
            model.eval()
            print(f"Model loaded: {{model.__class__.__name__}}")
        except Exception as e:
            print(f"Failed to load model: {{e}}")
            print("Using placeholder model instead")
            model = torch.nn.Sequential(
                torch.nn.Conv2d(1, 32, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 1, 3, padding=1)
            ).to(device)
            model.eval()
    else:
        print("No model file provided, using placeholder")
        model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 1, 3, padding=1)
        ).to(device)
        model.eval()

    # Create test input
    print(f"\\nCreating {{num_images}} test images ({{input_size}}x{{input_size}})...")
    x = torch.randn(num_images, 1, input_size, input_size, device=device)
    print(f"Input shape: {{x.shape}}")

    # Warmup runs (important for accurate GPU profiling)
    print("\\nWarmup runs (5 iterations)...")
    with torch.no_grad():
        for _ in range(5):
            _ = model(x[:1])
            torch.cuda.synchronize()

    # Profiled inference with NVTX markers for timeline visualization
    print(f"\\nProfiling {{num_images}} images...")
    torch.cuda.nvtx.range_push("inference_loop")

    with torch.no_grad():
        for i in range(num_images):
            torch.cuda.nvtx.range_push(f"image_{{i}}")
            _ = model(x[i:i+1])
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_pop()

    # Also profile batch inference
    print(f"\\nProfiling batch inference ({{num_images}} images at once)...")
    torch.cuda.nvtx.range_push("batch_inference")
    with torch.no_grad():
        _ = model(x)
        torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    print("\\n" + "=" * 60)
    print("PROFILING COMPLETE!")
    print("=" * 60)
    print("Open the .nsys-rep file in Nsight Systems GUI to analyze:")
    print("  - CUDA kernel execution times")
    print("  - Memory transfers (CPU <-> GPU)")
    print("  - NVTX markers for inference regions")

if __name__ == "__main__":
    main()
'''
        with open(script_path, 'w') as f:
            f.write(script)

        self.logger.info(f"Generated profiling script: {script_path}")

    def _on_worker_output(self, text: str):
        """Handle worker output."""
        self.output_text.append(text)
        # Auto-scroll
        scrollbar = self.output_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        QApplication.processEvents()

    def _on_worker_finished(self, success: bool, message: str):
        """Handle worker completion."""
        self.progress_bar.setVisible(False)
        self.run_button.setEnabled(True)

        if success:
            self.output_text.append(f"\n✓ {message}")
            self.output_text.append(f"\nOutput: {self._output_file}")
            self.open_button.setEnabled(True)
            self.logger.info(f"Nsight profiling completed: {self._output_file}")
        else:
            self.output_text.append(f"\n✗ Error: {message}")
            self.logger.error(f"Nsight profiling failed: {message}")

    def _on_open_results(self):
        """Open the results in Nsight Systems GUI."""
        if not self._output_file or not Path(self._output_file).exists():
            QMessageBox.warning(self, "Error", "Output file not found")
            return

        # Try to open with nsys-ui
        nsys_ui = shutil.which("nsys-ui") or shutil.which("nsight-sys")

        if nsys_ui:
            try:
                subprocess.Popen([nsys_ui, self._output_file])
                self.logger.info(f"Opened {self._output_file} in Nsight Systems")
            except Exception as e:
                self.logger.error(f"Failed to open Nsight: {e}")
                QMessageBox.warning(
                    self, "Error",
                    f"Failed to open Nsight Systems:\n{e}\n\n"
                    f"File location:\n{self._output_file}"
                )
        else:
            QMessageBox.information(
                self, "Open Results",
                f"Nsight Systems GUI not found in PATH.\n\n"
                f"Open manually:\n{self._output_file}"
            )

    def closeEvent(self, event):
        """Handle close event."""
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait()
        event.accept()
