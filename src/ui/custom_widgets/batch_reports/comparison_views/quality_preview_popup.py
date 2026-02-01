"""Quality Metrics Preview popup for Batch Reports."""
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
from matplotlib import cm
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QSlider, QGroupBox, QGridLayout, QSizePolicy,
    QWidget, QPushButton, QApplication, QMessageBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QFont

# Try to import pylops and pyproximal for FISTA and TV Norm
try:
    import pylops
    import pyproximal
    ITERATIVE_METHODS_AVAILABLE = True
except ImportError:
    ITERATIVE_METHODS_AVAILABLE = False


class MaskApplicationPopup(QDialog):
    """
    Popup showing mask application evolution.

    Shows how masks are progressively applied to create the noisy image
    using the actual reconstruction algorithm.

    Supported methods:
    - Ghost Imaging (Conventional): x̂ = (1/N) * Σ(B_i - B_avg) * M_i
    - Pseudoinverse: x = S^+ @ y
    - FISTA: minimize ||y - S@x||² + λ||x||₁ (slow, iterative)
    - TV Norm: minimize ||y - S@x||² + λ TV(x) (slow, iterative)
    """

    # Fixed display size for images
    DISPLAY_SIZE = 250

    # Slow methods that show a warning
    SLOW_METHODS = {"FISTA", "TV Norm"}

    def __init__(self, original: np.ndarray, reconstructed: np.ndarray,
                 masks: Optional[np.ndarray] = None, image_idx: int = 0,
                 test_name: str = "", reconstruction_method: str = "Ghost Imaging",
                 logger=None, parent=None):
        super().__init__(parent)

        self.logger = logger.getChild("MaskApplicationPopup") if logger else logging.getLogger("MaskApplicationPopup")
        self._original = original.astype(np.float64) if original is not None else None
        self._reconstructed = reconstructed
        self._masks = masks
        self._image_idx = image_idx
        self._test_name = test_name
        self._reconstruction_method = reconstruction_method
        self._current_mask_idx = 0

        # Determine effective method for preview
        self._effective_method = self._get_effective_method()

        # Precompute data needed for reconstruction
        self._measurements = None
        self._masks_matrix = None
        if self._original is not None and self._masks is not None and len(self._masks) > 0:
            self._precompute_data()

        self.cmap = cm.get_cmap('hot')
        self._setup_ui()
        self._update_display()

    def _get_effective_method(self) -> str:
        """Determine which reconstruction method to use."""
        method_lower = self._reconstruction_method.lower()

        if "pseudoinverse" in method_lower:
            return "Pseudoinverse"
        elif "fista" in method_lower:
            if ITERATIVE_METHODS_AVAILABLE:
                return "FISTA"
            else:
                self.logger.warning("pylops/pyproximal not available, falling back to Ghost Imaging")
                return "Ghost Imaging"
        elif "tv" in method_lower:
            if ITERATIVE_METHODS_AVAILABLE:
                return "TV Norm"
            else:
                self.logger.warning("pylops/pyproximal not available, falling back to Ghost Imaging")
                return "Ghost Imaging"
        elif "ghost" in method_lower or "conventional" in method_lower:
            return "Ghost Imaging"
        else:
            # Default to Ghost Imaging
            return "Ghost Imaging"

    def _precompute_data(self):
        """Precompute data needed for reconstruction methods."""
        # Precompute measurements for all methods
        self._measurements = []
        for mask in self._masks:
            mask_float = mask.astype(np.float64)
            measurement = (self._original * mask_float).sum()
            self._measurements.append(measurement)

        # For methods that need the masks matrix, precompute it
        if self._effective_method in {"Pseudoinverse", "FISTA", "TV Norm"}:
            self._masks_matrix = np.array([
                mask.flatten().astype(np.float64) for mask in self._masks
            ], dtype=np.float64)

    def _setup_ui(self):
        self.setWindowTitle(f"Mask Application - {self._test_name} - Image {self._image_idx}")
        self.setMinimumSize(850, 500)
        self.resize(950, 550)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Title
        title = QLabel("<h3>Mask Application Visualization</h3>")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Images side by side
        images_layout = QHBoxLayout()
        images_layout.setSpacing(15)

        # Original image
        orig_container = QVBoxLayout()
        orig_title = QLabel("Ground-Truth")
        orig_title.setAlignment(Qt.AlignCenter)
        orig_title.setStyleSheet("font-weight: bold;")
        orig_container.addWidget(orig_title)
        self.orig_label = QLabel()
        self.orig_label.setAlignment(Qt.AlignCenter)
        self.orig_label.setFixedSize(self.DISPLAY_SIZE, self.DISPLAY_SIZE)
        self.orig_label.setStyleSheet("background-color: #2a2a2a; border: 1px solid #555;")
        orig_container.addWidget(self.orig_label)
        orig_container.addStretch()
        images_layout.addLayout(orig_container)

        # Current mask preview (if available)
        mask_container = QVBoxLayout()
        self.mask_title = QLabel("Mask Pattern (0)")
        self.mask_title.setAlignment(Qt.AlignCenter)
        self.mask_title.setStyleSheet("font-weight: bold;")
        mask_container.addWidget(self.mask_title)
        self.mask_label = QLabel()
        self.mask_label.setAlignment(Qt.AlignCenter)
        self.mask_label.setFixedSize(self.DISPLAY_SIZE, self.DISPLAY_SIZE)
        self.mask_label.setStyleSheet("background-color: #2a2a2a; border: 1px solid #555;")
        mask_container.addWidget(self.mask_label)
        mask_container.addStretch()
        images_layout.addLayout(mask_container)

        # Progressive reconstruction using actual reconstruction method
        recon_container = QVBoxLayout()
        self.recon_title = QLabel("Reconstruction (0 masks)")
        self.recon_title.setAlignment(Qt.AlignCenter)
        self.recon_title.setStyleSheet("font-weight: bold;")
        recon_container.addWidget(self.recon_title)
        self.recon_label = QLabel()
        self.recon_label.setAlignment(Qt.AlignCenter)
        self.recon_label.setFixedSize(self.DISPLAY_SIZE, self.DISPLAY_SIZE)
        self.recon_label.setStyleSheet("background-color: #2a2a2a; border: 1px solid #555;")
        recon_container.addWidget(self.recon_label)
        recon_container.addStretch()
        images_layout.addLayout(recon_container)

        layout.addLayout(images_layout, 1)

        # Mask slider
        slider_group = QGroupBox("Mask Navigation")
        slider_layout = QHBoxLayout(slider_group)
        slider_layout.setSpacing(10)

        slider_label = QLabel("Masks applied:")
        slider_layout.addWidget(slider_label)

        self.mask_slider = QSlider(Qt.Horizontal)
        self.mask_slider.setMinimum(0)
        self.mask_slider.setMaximum(0)
        self.mask_slider.setValue(0)
        self.mask_slider.valueChanged.connect(self._on_mask_slider_changed)
        slider_layout.addWidget(self.mask_slider, 1)

        self.mask_idx_label = QLabel("0 / 0")
        self.mask_idx_label.setMinimumWidth(80)
        slider_layout.addWidget(self.mask_idx_label)

        layout.addWidget(slider_group)

        # Info label with warning for slow methods
        info_text = f"Method: {self._effective_method}"
        if self._masks is not None:
            info_text += f" | {len(self._masks)} masks"

        self.info_label = QLabel(info_text)
        self.info_label.setAlignment(Qt.AlignCenter)

        # Show warning for slow methods
        if self._effective_method in self.SLOW_METHODS:
            self.info_label.setStyleSheet("color: #FF6600; font-size: 11px; font-weight: bold;")
            warning_label = QLabel("⚠️ Slow method - reconstruction may take time when moving the slider")
            warning_label.setStyleSheet("color: #FF6600; font-size: 10px;")
            warning_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(warning_label)
        else:
            self.info_label.setStyleSheet("color: #666; font-size: 11px;")

        layout.addWidget(self.info_label)

        # Close button
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

        # Setup slider range
        if self._masks is not None and len(self._masks) > 0:
            self.mask_slider.setMaximum(len(self._masks) - 1)
            self.mask_idx_label.setText(f"0 / {len(self._masks) - 1}")
        else:
            self.mask_label.setText("No masks\navailable")
            self.recon_label.setText("No masks\navailable")

    def _on_mask_slider_changed(self, idx: int):
        """Handle mask slider change."""
        self._current_mask_idx = idx
        n_masks = len(self._masks) if self._masks is not None else 0
        self.mask_idx_label.setText(f"{idx} / {n_masks - 1 if n_masks > 0 else 0}")
        self.mask_title.setText(f"Mask Pattern ({idx})")
        self.recon_title.setText(f"Reconstruction ({idx + 1} masks)")
        self._update_mask_display()
        self._update_progressive_display()

    def _update_display(self):
        """Update all image displays."""
        if self._original is not None:
            self._display_image(self._original, self.orig_label)
        else:
            self.orig_label.setText("Not available")

        self._update_mask_display()
        self._update_progressive_display()

    def _update_mask_display(self):
        """Update the current mask pattern display."""
        if self._masks is not None and self._current_mask_idx < len(self._masks):
            mask = self._masks[self._current_mask_idx]
            self._display_image(mask, self.mask_label, use_grayscale=True)
        else:
            self.mask_label.setText("No mask")

    def _update_progressive_display(self):
        """Update the progressive reconstruction display using the appropriate method."""
        if self._original is None:
            self.recon_label.setText("No original")
            return

        if self._measurements is None or self._masks is None:
            self.recon_label.setText("No masks")
            return

        idx_max = self._current_mask_idx + 1  # inclusive (masks 0 to current)
        if idx_max <= 0:
            self.recon_label.setText("Move slider")
            return

        # Show busy cursor for slow methods
        if self._effective_method in self.SLOW_METHODS:
            QApplication.setOverrideCursor(Qt.WaitCursor)

        try:
            if self._effective_method == "Pseudoinverse":
                reconstructed = self._reconstruct_pseudoinverse(idx_max)
            elif self._effective_method == "FISTA":
                reconstructed = self._reconstruct_fista(idx_max)
            elif self._effective_method == "TV Norm":
                reconstructed = self._reconstruct_tv_norm(idx_max)
            else:
                # Ghost Imaging (default)
                reconstructed = self._reconstruct_ghost_imaging(idx_max)

            if reconstructed is not None:
                self._display_image(reconstructed, self.recon_label)
            else:
                self.recon_label.setText("Reconstruction failed")
        finally:
            # Restore cursor for slow methods
            if self._effective_method in self.SLOW_METHODS:
                QApplication.restoreOverrideCursor()

    def _reconstruct_ghost_imaging(self, idx_max: int) -> Optional[np.ndarray]:
        """
        Reconstruct using Ghost Imaging formula.

        Formula: x̂ = (1/N) * Σ(B_i - B_avg) * M_i
        Where B_i is the measurement, B_avg is the DC offset.
        """
        measurements_subset = self._measurements[:idx_max]
        N = len(measurements_subset)

        if N == 0:
            return None

        # Calculate average measurement (DC offset removal)
        measurement_avg = np.mean(measurements_subset)

        # Accumulate weighted masks
        accumulated = np.zeros(self._original.shape, dtype=np.float64)
        for i in range(idx_max):
            mask = self._masks[i].astype(np.float64)
            accumulated += (self._measurements[i] - measurement_avg) * mask

        # Normalize by number of measurements
        return accumulated / N

    def _reconstruct_pseudoinverse(self, idx_max: int) -> Optional[np.ndarray]:
        """
        Reconstruct using Moore-Penrose Pseudoinverse.

        Formula: x = S^+ @ y
        Where S is the measurement matrix, y is the measurements vector.
        """
        if self._masks_matrix is None:
            return self._reconstruct_ghost_imaging(idx_max)

        # Extract subset of masks and measurements
        S = self._masks_matrix[:idx_max]  # Shape: (M, N_pixels)
        y = np.array(self._measurements[:idx_max], dtype=np.float64)

        if len(S) == 0:
            return None

        try:
            # Compute Moore-Penrose pseudoinverse: x = S^+ @ y
            S_pinv = np.linalg.pinv(S)
            img_vec = S_pinv @ y
            return img_vec.reshape(self._original.shape)
        except Exception as e:
            self.logger.warning("Pseudoinverse failed: %s, falling back to Ghost Imaging", e)
            return self._reconstruct_ghost_imaging(idx_max)

    def _reconstruct_fista(self, idx_max: int) -> Optional[np.ndarray]:
        """
        Reconstruct using FISTA (Fast Iterative Shrinkage-Thresholding Algorithm).

        Solves: minimize ||y - S@x||² + λ||x||₁
        """
        if not ITERATIVE_METHODS_AVAILABLE:
            self.logger.warning("pylops/pyproximal not available for FISTA")
            return self._reconstruct_ghost_imaging(idx_max)

        if self._masks_matrix is None:
            return self._reconstruct_ghost_imaging(idx_max)

        # Extract subset of masks and measurements
        S = self._masks_matrix[:idx_max]
        y = np.array(self._measurements[:idx_max], dtype=np.float64)

        if len(S) == 0:
            return None

        try:
            M, N_pix = S.shape

            # Wrap measurement matrix with pylops
            Sop = pylops.MatrixMult(S)

            # Define data fidelity: f(x) = 1/2 ||S@x - y||²
            l2 = pyproximal.proximal.L2(Op=Sop, b=y)

            # Define L1 regularizer: g(x) = λ||x||₁
            lam = 1e-3  # Regularization parameter
            l1 = lam * pyproximal.proximal.L1()

            # Compute step size
            L_val = np.abs((Sop.H * Sop).eigs(1)[0])
            tau = 0.95 / L_val

            # Initial guess
            x0 = np.zeros(N_pix, dtype=np.float64)

            # Run FISTA (reduced iterations for preview)
            maxit = 100  # Reduced for real-time preview
            opt = pyproximal.optimization.primal.ProximalGradient(
                l2, l1, tau=tau, x0=x0, niter=maxit,
                acceleration='fista', show=False
            )

            # Extract solution
            if hasattr(opt, 'run'):
                x_fista = opt.run()
            elif hasattr(opt, 'solve'):
                opt.solve()
                x_fista = opt.x
            else:
                x_fista = opt

            return x_fista.reshape(self._original.shape)

        except Exception as e:
            self.logger.warning("FISTA failed: %s, falling back to Ghost Imaging", e)
            return self._reconstruct_ghost_imaging(idx_max)

    def _reconstruct_tv_norm(self, idx_max: int) -> Optional[np.ndarray]:
        """
        Reconstruct using TV-norm regularization.

        Solves: minimize ||y - S@x||² + λ TV(x)
        """
        if not ITERATIVE_METHODS_AVAILABLE:
            self.logger.warning("pylops/pyproximal not available for TV Norm")
            return self._reconstruct_ghost_imaging(idx_max)

        if self._masks_matrix is None:
            return self._reconstruct_ghost_imaging(idx_max)

        # Extract subset of masks and measurements
        S = self._masks_matrix[:idx_max]
        y = np.array(self._measurements[:idx_max], dtype=np.float64)

        if len(S) == 0:
            return None

        try:
            M, N_pix = S.shape

            # Wrap measurement matrix with pylops
            Sop = pylops.MatrixMult(S)

            # Build gradient operator for TV
            Gop = pylops.Gradient(dims=self._original.shape, sampling=1.0,
                                  edge=False, kind='forward', dtype='float64')

            # Define L2 data fidelity
            l2 = pyproximal.proximal.L2(Op=Sop, b=y)

            # Define L21 regularizer for isotropic TV
            lam = 1e-1  # TV regularization strength
            l21 = lam * pyproximal.proximal.L21(ndim=2)

            # Primal-dual step sizes
            L_tv = 8.0  # Conservative estimate
            tau_tv = 1.0 / np.sqrt(L_tv)
            mu_tv = 1.0 / (tau_tv * L_tv)

            # Run primal-dual algorithm (reduced iterations for preview)
            niter = 50  # Reduced for real-time preview
            pd_solver = pyproximal.optimization.primaldual.PrimalDual(
                l2, l21, Gop,
                tau=tau_tv, mu=mu_tv, theta=1.0,
                x0=np.zeros(N_pix, dtype=np.float64),
                niter=niter, show=False
            )

            # Extract solution
            if hasattr(pd_solver, 'x'):
                x_tv = pd_solver.x
            else:
                x_tv = pd_solver

            return x_tv.reshape(self._original.shape)

        except Exception as e:
            self.logger.warning("TV Norm failed: %s, falling back to Ghost Imaging", e)
            return self._reconstruct_ghost_imaging(idx_max)

    def _display_image(self, arr: np.ndarray, label: QLabel, use_grayscale: bool = False):
        """Display an image without interpolation using fixed size."""
        arr = np.array(arr, copy=False)

        if arr.ndim == 0 or arr.size == 0:
            label.setText("No image")
            return

        # Normalize to [0, 1]
        amin, amax = arr.min(), arr.max()
        norm = (arr - amin) / (amax - amin) if amax > amin else np.zeros_like(arr, dtype=float)

        if use_grayscale:
            # Grayscale for masks
            gray = (norm * 255).astype(np.uint8)
            h, w = gray.shape
            qimg = QImage(gray.data, w, h, w, QImage.Format_Grayscale8)
        else:
            # Thermal colormap for images
            rgba = self.cmap(norm)
            rgb = (rgba[..., :3] * 255).astype(np.uint8)
            h, w = rgb.shape[:2]
            data = rgb.tobytes()
            bytes_per_line = 3 * w
            qimg = QImage(data, w, h, bytes_per_line, QImage.Format_RGB888)

        pix = QPixmap.fromImage(qimg)

        # Scale to fixed size without interpolation (FastTransformation = nearest neighbor)
        display_size = self.DISPLAY_SIZE - 4
        pix = pix.scaled(display_size, display_size, Qt.KeepAspectRatio, Qt.FastTransformation)

        label.setPixmap(pix)


class QualityPreviewPopup(QDialog):
    """
    Popup dialog for previewing per-image quality metrics in Batch Reports.

    Features:
    - Test selection dropdown
    - Image slider for navigating through images
    - Side-by-side display of Ground-Truth, Noisy, Denoised images
    - Per-image metrics table with change percentages
    - Bar chart showing normalized quality scores
    - Button to view mask application visualization
    """

    # Fixed display size for images
    IMAGE_DISPLAY_SIZE = 220

    def __init__(self, tests: List[Dict[str, Any]], logger=None, parent=None):
        super().__init__(parent)

        if logger:
            self.logger = logger.getChild("QualityPreviewPopup")
        else:
            self.logger = logging.getLogger("QualityPreviewPopup")

        self._tests = tests
        self._current_test = None
        self._current_per_image = {}
        self._current_idx = 0

        # Cached images for current test
        self._originals = None
        self._reconstructions = None
        self._denoised = None
        self._masks = None

        # Thermal colormap for images
        self.cmap = cm.get_cmap('hot')

        self._setup_ui()
        self._populate_tests()

    def _setup_ui(self):
        """Setup the popup UI."""
        self.setWindowTitle("Quality Metrics Preview")
        self.setMinimumSize(950, 700)
        self.resize(1050, 750)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)

        # Test selection row
        selection_layout = QHBoxLayout()

        test_label = QLabel("Select Test:")
        test_label.setStyleSheet("font-weight: bold;")
        selection_layout.addWidget(test_label)

        self.test_combo = QComboBox()
        self.test_combo.setMinimumWidth(300)
        self.test_combo.currentIndexChanged.connect(self._on_test_changed)
        selection_layout.addWidget(self.test_combo)

        selection_layout.addStretch()

        # Data status label
        self.data_status_label = QLabel("")
        self.data_status_label.setStyleSheet("color: #666; font-style: italic;")
        selection_layout.addWidget(self.data_status_label)

        main_layout.addLayout(selection_layout)

        # Images section with slider right below
        images_group = QGroupBox("Image Comparison")
        images_main_layout = QVBoxLayout(images_group)
        images_main_layout.setSpacing(8)

        # Images row
        images_layout = QHBoxLayout()
        images_layout.setSpacing(15)

        label_font = QFont()
        label_font.setPointSize(10)

        # Ground-Truth image
        orig_container = QVBoxLayout()
        orig_label = QLabel("Ground-Truth")
        orig_label.setAlignment(Qt.AlignCenter)
        orig_label.setFont(label_font)
        orig_container.addWidget(orig_label)
        self.orig_image = QLabel()
        self.orig_image.setAlignment(Qt.AlignCenter)
        self.orig_image.setFixedSize(self.IMAGE_DISPLAY_SIZE, self.IMAGE_DISPLAY_SIZE)
        self.orig_image.setStyleSheet("background-color: #2a2a2a; border: 1px solid #555;")
        orig_container.addWidget(self.orig_image)
        orig_container.addStretch()
        images_layout.addLayout(orig_container)

        # Noisy image
        noisy_container = QVBoxLayout()
        noisy_label = QLabel("Noisy (Reconstructed)")
        noisy_label.setAlignment(Qt.AlignCenter)
        noisy_label.setFont(label_font)
        noisy_container.addWidget(noisy_label)
        self.noisy_image = QLabel()
        self.noisy_image.setAlignment(Qt.AlignCenter)
        self.noisy_image.setFixedSize(self.IMAGE_DISPLAY_SIZE, self.IMAGE_DISPLAY_SIZE)
        self.noisy_image.setStyleSheet("background-color: #2a2a2a; border: 1px solid #555;")
        noisy_container.addWidget(self.noisy_image)
        noisy_container.addStretch()
        images_layout.addLayout(noisy_container)

        # Denoised image
        recon_container = QVBoxLayout()
        recon_label = QLabel("Denoised")
        recon_label.setAlignment(Qt.AlignCenter)
        recon_label.setFont(label_font)
        recon_container.addWidget(recon_label)
        self.recon_image = QLabel()
        self.recon_image.setAlignment(Qt.AlignCenter)
        self.recon_image.setFixedSize(self.IMAGE_DISPLAY_SIZE, self.IMAGE_DISPLAY_SIZE)
        self.recon_image.setStyleSheet("background-color: #2a2a2a; border: 1px solid #555;")
        recon_container.addWidget(self.recon_image)
        recon_container.addStretch()
        images_layout.addLayout(recon_container)

        images_main_layout.addLayout(images_layout, 1)

        # Image slider (right below images)
        slider_layout = QHBoxLayout()
        slider_layout.setSpacing(10)

        self.slider_label = QLabel("Image:")
        slider_layout.addWidget(self.slider_label)

        self.image_slider = QSlider(Qt.Horizontal)
        self.image_slider.setMinimum(0)
        self.image_slider.setMaximum(0)
        self.image_slider.setValue(0)
        self.image_slider.valueChanged.connect(self._on_slider_changed)
        slider_layout.addWidget(self.image_slider, 1)

        self.index_label = QLabel("0  (0 images)")
        self.index_label.setMinimumWidth(100)
        slider_layout.addWidget(self.index_label)

        # View Mask Application button
        self.mask_btn = QPushButton("View Mask Application")
        self.mask_btn.setEnabled(False)
        self.mask_btn.setToolTip("View how masks are applied to create the noisy image")
        self.mask_btn.clicked.connect(self._on_view_mask_application)
        self.mask_btn.setStyleSheet("""
            QPushButton {
                background-color: #5C6BC0;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: bold;
            }
            QPushButton:hover:enabled {
                background-color: #3F51B5;
            }
            QPushButton:disabled {
                background-color: #ccc;
                color: #888;
            }
        """)
        slider_layout.addWidget(self.mask_btn)

        images_main_layout.addLayout(slider_layout)

        main_layout.addWidget(images_group)

        # Metrics section
        metrics_group = QGroupBox("Quality Metrics for Current Image")
        metrics_main_layout = QHBoxLayout(metrics_group)
        metrics_main_layout.setSpacing(15)

        # Left: Metrics table
        table_widget = QWidget()
        table_layout = QGridLayout(table_widget)
        table_layout.setSpacing(8)

        header_font = QFont()
        header_font.setBold(True)

        # Headers
        table_layout.addWidget(QLabel(""), 0, 0)
        noisy_header = QLabel("Noisy")
        noisy_header.setFont(header_font)
        noisy_header.setAlignment(Qt.AlignCenter)
        table_layout.addWidget(noisy_header, 0, 1)
        recon_header = QLabel("Denoised")
        recon_header.setFont(header_font)
        recon_header.setAlignment(Qt.AlignCenter)
        table_layout.addWidget(recon_header, 0, 2)
        change_header = QLabel("Change")
        change_header.setFont(header_font)
        change_header.setAlignment(Qt.AlignCenter)
        table_layout.addWidget(change_header, 0, 3)

        # PSNR row
        self.psnr_label = QLabel("PSNR (dB) \u2191:")
        self.psnr_label.setFont(header_font)
        self.psnr_label.setToolTip("Higher is better")
        self.psnr_noisy_display = QLabel("-")
        self.psnr_noisy_display.setAlignment(Qt.AlignCenter)
        self.psnr_noisy_display.setStyleSheet("font-size: 13px;")
        self.psnr_recon_display = QLabel("-")
        self.psnr_recon_display.setAlignment(Qt.AlignCenter)
        self.psnr_recon_display.setStyleSheet("font-size: 13px;")
        self.psnr_change_display = QLabel("-")
        self.psnr_change_display.setAlignment(Qt.AlignCenter)
        self.psnr_change_display.setStyleSheet("font-size: 13px; font-weight: bold;")
        table_layout.addWidget(self.psnr_label, 1, 0)
        table_layout.addWidget(self.psnr_noisy_display, 1, 1)
        table_layout.addWidget(self.psnr_recon_display, 1, 2)
        table_layout.addWidget(self.psnr_change_display, 1, 3)

        # SSIM row
        self.ssim_label = QLabel("SSIM \u2191:")
        self.ssim_label.setFont(header_font)
        self.ssim_label.setToolTip("Higher is better")
        self.ssim_noisy_display = QLabel("-")
        self.ssim_noisy_display.setAlignment(Qt.AlignCenter)
        self.ssim_noisy_display.setStyleSheet("font-size: 13px;")
        self.ssim_recon_display = QLabel("-")
        self.ssim_recon_display.setAlignment(Qt.AlignCenter)
        self.ssim_recon_display.setStyleSheet("font-size: 13px;")
        self.ssim_change_display = QLabel("-")
        self.ssim_change_display.setAlignment(Qt.AlignCenter)
        self.ssim_change_display.setStyleSheet("font-size: 13px; font-weight: bold;")
        table_layout.addWidget(self.ssim_label, 2, 0)
        table_layout.addWidget(self.ssim_noisy_display, 2, 1)
        table_layout.addWidget(self.ssim_recon_display, 2, 2)
        table_layout.addWidget(self.ssim_change_display, 2, 3)

        # LPIPS row
        self.lpips_label = QLabel("LPIPS \u2193:")
        self.lpips_label.setFont(header_font)
        self.lpips_label.setToolTip("Lower is better")
        self.lpips_noisy_display = QLabel("-")
        self.lpips_noisy_display.setAlignment(Qt.AlignCenter)
        self.lpips_noisy_display.setStyleSheet("font-size: 13px;")
        self.lpips_recon_display = QLabel("-")
        self.lpips_recon_display.setAlignment(Qt.AlignCenter)
        self.lpips_recon_display.setStyleSheet("font-size: 13px;")
        self.lpips_change_display = QLabel("-")
        self.lpips_change_display.setAlignment(Qt.AlignCenter)
        self.lpips_change_display.setStyleSheet("font-size: 13px; font-weight: bold;")
        table_layout.addWidget(self.lpips_label, 3, 0)
        table_layout.addWidget(self.lpips_noisy_display, 3, 1)
        table_layout.addWidget(self.lpips_recon_display, 3, 2)
        table_layout.addWidget(self.lpips_change_display, 3, 3)

        metrics_main_layout.addWidget(table_widget)

        # Right: Bar chart
        self.bar_figure = Figure(figsize=(4, 2.5), dpi=100)
        self.bar_canvas = FigureCanvas(self.bar_figure)
        self.bar_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.bar_canvas.setMinimumSize(250, 150)
        metrics_main_layout.addWidget(self.bar_canvas)

        main_layout.addWidget(metrics_group)

        # Info label
        self.info_label = QLabel("Select a test to preview quality metrics")
        self.info_label.setStyleSheet("color: #666; font-size: 11px;")
        self.info_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.info_label)

    def _populate_tests(self):
        """Populate the test selection dropdown."""
        self.test_combo.clear()

        for test in self._tests:
            per_image = test.get("quality_per_image", {})
            if per_image:
                test_name = test.get("name", "Unknown Test")
                n_images = len(per_image.get("psnr_noisy", []))
                self.test_combo.addItem(f"{test_name} ({n_images} images)", test)

        if self.test_combo.count() > 0:
            self.test_combo.setCurrentIndex(0)
        else:
            self.info_label.setText("No tests with per-image data available")

    def _load_test_images(self, test: Dict[str, Any]) -> bool:
        """Load test images from the NPZ file if available."""
        self._originals = None
        self._reconstructions = None
        self._denoised = None
        self._masks = None

        experiment_path = test.get("_experiment_path")
        if not experiment_path:
            self.logger.debug("No experiment path in test data")
            return False

        report_path = Path(experiment_path)
        batch_dir = report_path.parent

        test_name = test.get("name", "Unknown")
        safe_name = test_name.replace(" ", "_").replace("/", "-")

        test_images_path = batch_dir / "data" / safe_name / "test_images.npz"
        masks_path = batch_dir / "data" / safe_name / "masks.npz"

        self.logger.debug("Looking for images at: %s", test_images_path)

        if test_images_path.exists():
            try:
                data = np.load(str(test_images_path))
                self._originals = data.get("originals")
                self._reconstructions = data.get("reconstructions")
                self._denoised = data.get("denoised")

                self.logger.info(
                    "Loaded test images: originals=%s, reconstructions=%s, denoised=%s",
                    self._originals.shape if self._originals is not None else None,
                    self._reconstructions.shape if self._reconstructions is not None else None,
                    self._denoised.shape if self._denoised is not None else None
                )

                if masks_path.exists():
                    try:
                        mask_data = np.load(str(masks_path))
                        self._masks = mask_data.get("masks")
                        self.logger.debug("Loaded masks: %s", self._masks.shape if self._masks is not None else None)
                    except Exception as e:
                        self.logger.warning("Failed to load masks: %s", e)

                return True

            except Exception as e:
                self.logger.error("Failed to load test images: %s", e)
                return False
        else:
            self.logger.debug("Test images file not found: %s", test_images_path)
            return False

    def _on_test_changed(self, index: int):
        """Handle test selection change."""
        if index < 0:
            return

        test = self.test_combo.itemData(index)
        if not test:
            return

        self._current_test = test
        self._current_per_image = test.get("quality_per_image", {})

        images_loaded = self._load_test_images(test)

        if images_loaded:
            self.data_status_label.setText("Images loaded")
            self.data_status_label.setStyleSheet("color: #228B22; font-style: italic;")
            # Enable mask button if masks are available
            self.mask_btn.setEnabled(self._masks is not None and len(self._masks) > 0)
        else:
            self.data_status_label.setText("Images not available (metrics only)")
            self.data_status_label.setStyleSheet("color: #666; font-style: italic;")
            self.mask_btn.setEnabled(False)

        n_images = len(self._current_per_image.get("psnr_noisy", []))
        self.image_slider.setMaximum(max(0, n_images - 1))
        self.image_slider.setValue(0)
        self.index_label.setText(f"0  ({n_images} images)")

        if n_images > 0:
            self._on_slider_changed(0)
            self.info_label.setText(f"Showing {n_images} images from {test.get('name', 'test')}")
        else:
            self._clear_displays()
            self.info_label.setText("No images available for this test")

    def _on_view_mask_application(self):
        """Open the mask application visualization popup."""
        if self._originals is None or self._reconstructions is None:
            return

        idx = self._current_idx
        if idx >= len(self._originals) or idx >= len(self._reconstructions):
            return

        test_name = self._current_test.get("name", "Unknown") if self._current_test else "Unknown"
        reconstruction_method = self._current_test.get("reconstruction_method", "Ghost Imaging") if self._current_test else "Ghost Imaging"

        popup = MaskApplicationPopup(
            original=self._originals[idx],
            reconstructed=self._reconstructions[idx],
            masks=self._masks,
            image_idx=idx,
            test_name=test_name,
            reconstruction_method=reconstruction_method,
            logger=self.logger,
            parent=self
        )
        popup.exec_()

    def _on_slider_changed(self, idx: int):
        """Handle image slider change."""
        self._current_idx = idx
        per_image = self._current_per_image
        n_images = len(per_image.get("psnr_noisy", []))

        self.index_label.setText(f"{idx}  ({n_images} images)")

        if not (0 <= idx < n_images):
            return

        self._update_metrics(idx)
        self._update_bar_chart(idx)
        self._update_images(idx)

    def _update_metrics(self, idx: int):
        """Update the metrics display for the current image."""
        per_image = self._current_per_image

        psnr_noisy = per_image.get("psnr_noisy", [])
        psnr_denoised = per_image.get("psnr_denoised", [])
        psnr_n = psnr_noisy[idx] if idx < len(psnr_noisy) else None
        psnr_d = psnr_denoised[idx] if idx < len(psnr_denoised) else None
        self.psnr_noisy_display.setText(f"{psnr_n:.2f}" if psnr_n is not None else "-")
        self.psnr_recon_display.setText(f"{psnr_d:.2f}" if psnr_d is not None else "-")
        self._set_change_label(self.psnr_change_display, psnr_n, psnr_d, higher_is_better=True)

        ssim_noisy = per_image.get("ssim_noisy", [])
        ssim_denoised = per_image.get("ssim_denoised", [])
        ssim_n = ssim_noisy[idx] if idx < len(ssim_noisy) else None
        ssim_d = ssim_denoised[idx] if idx < len(ssim_denoised) else None
        self.ssim_noisy_display.setText(f"{ssim_n:.4f}" if ssim_n is not None else "-")
        self.ssim_recon_display.setText(f"{ssim_d:.4f}" if ssim_d is not None else "-")
        self._set_change_label(self.ssim_change_display, ssim_n, ssim_d, higher_is_better=True)

        lpips_noisy = per_image.get("lpips_noisy", [])
        lpips_denoised = per_image.get("lpips_denoised", [])
        lpips_n = lpips_noisy[idx] if idx < len(lpips_noisy) else None
        lpips_d = lpips_denoised[idx] if idx < len(lpips_denoised) else None
        self.lpips_noisy_display.setText(f"{lpips_n:.4f}" if lpips_n is not None else "-")
        self.lpips_recon_display.setText(f"{lpips_d:.4f}" if lpips_d is not None else "-")
        self._set_change_label(self.lpips_change_display, lpips_n, lpips_d, higher_is_better=False)

    def _set_change_label(self, label: QLabel, noisy_val: Optional[float],
                          recon_val: Optional[float], higher_is_better: bool):
        """Set the change label with color based on improvement."""
        if noisy_val is None or recon_val is None or noisy_val == 0:
            label.setText("-")
            label.setStyleSheet("font-size: 13px; font-weight: bold;")
            return

        pct_change = ((recon_val - noisy_val) / abs(noisy_val)) * 100
        is_improvement = pct_change > 0 if higher_is_better else pct_change < 0
        text = f"+{pct_change:.1f}%" if pct_change >= 0 else f"{pct_change:.1f}%"

        if is_improvement:
            style = "font-size: 13px; font-weight: bold; color: #228B22;"
        else:
            style = "font-size: 13px; font-weight: bold; color: #DC143C;"

        label.setText(text)
        label.setStyleSheet(style)

    def _update_bar_chart(self, idx: int):
        """Update the bar chart for the current image."""
        self.bar_figure.clear()
        per_image = self._current_per_image

        psnr_n = per_image.get("psnr_noisy", [])[idx] if idx < len(per_image.get("psnr_noisy", [])) else 0
        psnr_d = per_image.get("psnr_denoised", [])[idx] if idx < len(per_image.get("psnr_denoised", [])) else 0
        ssim_n = per_image.get("ssim_noisy", [])[idx] if idx < len(per_image.get("ssim_noisy", [])) else 0
        ssim_d = per_image.get("ssim_denoised", [])[idx] if idx < len(per_image.get("ssim_denoised", [])) else 0
        lpips_n = per_image.get("lpips_noisy", [])[idx] if idx < len(per_image.get("lpips_noisy", [])) else 0
        lpips_d = per_image.get("lpips_denoised", [])[idx] if idx < len(per_image.get("lpips_denoised", [])) else 0

        metrics = [
            ('PSNR \u2191', psnr_n, psnr_d, psnr_n / 50.0, psnr_d / 50.0, '#1f77b4', '{:.1f}'),
            ('SSIM \u2191', ssim_n, ssim_d, ssim_n, ssim_d, '#2ca02c', '{:.3f}'),
            ('LPIPS \u2193', lpips_n, lpips_d, 1.0 - lpips_n, 1.0 - lpips_d, '#d62728', '{:.3f}'),
        ]

        ax = self.bar_figure.add_subplot(111)
        x = np.arange(2)
        n_metrics = 3
        width = 0.8 / n_metrics

        for i, (name, noisy_val, recon_val, noisy_norm, recon_norm, color, fmt) in enumerate(metrics):
            offset = (i - (n_metrics - 1) / 2) * width
            bars = ax.bar(x + offset, [noisy_norm, recon_norm], width * 0.9,
                         label=name, color=color, alpha=0.8)
            ax.text(bars[0].get_x() + bars[0].get_width()/2, bars[0].get_height() + 0.02,
                   fmt.format(noisy_val), ha='center', va='bottom', fontsize=7)
            ax.text(bars[1].get_x() + bars[1].get_width()/2, bars[1].get_height() + 0.02,
                   fmt.format(recon_val), ha='center', va='bottom', fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(['Noisy', 'Denoised'], fontsize=9)
        ax.set_ylabel('Quality Score', fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.set_title(f'Image {idx}', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
                  ncol=3, fontsize=7, frameon=False)

        self.bar_figure.subplots_adjust(left=0.12, right=0.95, top=0.88, bottom=0.22)
        self.bar_canvas.draw()

    def _update_images(self, idx: int):
        """Update the image displays."""
        if self._originals is not None and idx < len(self._originals):
            self._display_image(self._originals[idx], self.orig_image)
        else:
            self.orig_image.setText("Not available")
            self.orig_image.setStyleSheet("background-color: #2a2a2a; border: 1px solid #555; color: #888;")

        if self._reconstructions is not None and idx < len(self._reconstructions):
            self._display_image(self._reconstructions[idx], self.noisy_image)
        else:
            self.noisy_image.setText("Not available")
            self.noisy_image.setStyleSheet("background-color: #2a2a2a; border: 1px solid #555; color: #888;")

        if self._denoised is not None and idx < len(self._denoised):
            self._display_image(self._denoised[idx], self.recon_image)
        else:
            self.recon_image.setText("Not available")
            self.recon_image.setStyleSheet("background-color: #2a2a2a; border: 1px solid #555; color: #888;")

    def _display_image(self, arr, label: QLabel):
        """Display an image with thermal colormap without interpolation using fixed size."""
        arr = np.array(arr, copy=False)

        if arr.ndim == 0 or arr.size == 0:
            label.setText("Empty")
            return

        amin, amax = arr.min(), arr.max()
        norm = (arr - amin) / (amax - amin) if amax > amin else np.zeros_like(arr, dtype=float)

        rgba = self.cmap(norm)
        rgb = (rgba[..., :3] * 255).astype(np.uint8)

        h, w = rgb.shape[:2]
        data = rgb.tobytes()
        bytes_per_line = 3 * w
        qimg = QImage(data, w, h, bytes_per_line, QImage.Format_RGB888)

        pix = QPixmap.fromImage(qimg)
        # Use fixed display size (FastTransformation = nearest neighbor, no interpolation)
        display_size = self.IMAGE_DISPLAY_SIZE - 4
        pix = pix.scaled(display_size, display_size, Qt.KeepAspectRatio, Qt.FastTransformation)
        label.setPixmap(pix)

    def _clear_displays(self):
        """Clear all displays."""
        self.orig_image.clear()
        self.noisy_image.clear()
        self.recon_image.clear()
        self.bar_figure.clear()
        self.bar_canvas.draw()

        self.psnr_noisy_display.setText("-")
        self.psnr_recon_display.setText("-")
        self.psnr_change_display.setText("-")
        self.ssim_noisy_display.setText("-")
        self.ssim_recon_display.setText("-")
        self.ssim_change_display.setText("-")
        self.lpips_noisy_display.setText("-")
        self.lpips_recon_display.setText("-")
        self.lpips_change_display.setText("-")
