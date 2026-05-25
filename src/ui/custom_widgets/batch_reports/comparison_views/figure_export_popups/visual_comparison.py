"""Visual Comparison popup (Fig 9 style)."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
)

from ui.utils.file_formats import safe_test_dirname
from ui.custom_widgets.batch_reports.comparison_views.figure_export_popups._base import (
    BaseFigureExportPopup,
)
from ui.custom_widgets.batch_reports.comparison_views.figure_export_popups._column_config import (
    ColumnConfig,
    ColumnListWidget,
)

# Try to import pylops for iterative reconstruction methods
try:
    import pylops
    PYLOPS_AVAILABLE = True
except ImportError:
    PYLOPS_AVAILABLE = False


class VisualComparisonPopup(BaseFigureExportPopup):
    """
    Popup for generating Visual Comparison figure.
    Shows configurable columns with: Ground Truth | Linear Recon | Iterative CS | DNN Output
    With configurable PSNR and timing metrics below each image.
    """

    def __init__(self, tests: list[dict[str, Any]], logger=None, parent=None):
        super().__init__(tests, logger, parent)
        self.setWindowTitle("Visual Comparison Figure")
        self.setMinimumSize(1100, 800)
        self.resize(1200, 850)

        self._figure = None
        self._canvas = None
        self._images_cache = {}  # Cache loaded images
        self._setup_ui()

    def _setup_ui(self):
        """Setup the popup UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)

        # Title
        title = QLabel("Visual Comparison of Reconstruction Methods")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(title)

        desc = QLabel("Double-click on a column card to configure it. Drag cards to reorder.")
        desc.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(desc)

        # Column configuration area
        config_group = QGroupBox("Column Configuration (drag to reorder, double-click to edit)")
        config_layout = QVBoxLayout(config_group)
        config_layout.setContentsMargins(10, 15, 10, 10)

        self.column_list = ColumnListWidget(self._tests)
        self.column_list.columns_changed.connect(self._update_preview)
        config_layout.addWidget(self.column_list)

        layout.addWidget(config_group)

        # Options row
        options_layout = QHBoxLayout()

        # Test selection (common for all columns)
        options_layout.addWidget(QLabel("Source Test:"))
        self.test_combo = QComboBox()
        for i, test in enumerate(self._tests):
            name = test.get("name", f"Test {i+1}")
            exp_name = test.get("_experiment_name", "")
            display = f"{name} ({exp_name})" if exp_name else name
            self.test_combo.addItem(display, i)
        self.test_combo.currentIndexChanged.connect(self._update_preview)
        options_layout.addWidget(self.test_combo)

        options_layout.addSpacing(20)

        # Image selection
        options_layout.addWidget(QLabel("Image index:"))
        self.image_spin = QSpinBox()
        self.image_spin.setMinimum(0)
        max_images = self._get_max_num_images()
        self.image_spin.setMaximum(max(0, max_images - 1))
        self.image_spin.setValue(0)
        self.image_spin.valueChanged.connect(self._update_preview)
        options_layout.addWidget(self.image_spin)

        options_layout.addSpacing(20)

        # Colormap
        options_layout.addWidget(QLabel("Colormap:"))
        self.cmap_combo = self._create_colormap_combo()
        self.cmap_combo.currentTextChanged.connect(self._update_preview)
        options_layout.addWidget(self.cmap_combo)

        options_layout.addStretch()

        # Save button
        self.save_btn = QPushButton("Save Figure...")
        self.save_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 8px 20px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.save_btn.clicked.connect(self._on_save)
        options_layout.addWidget(self.save_btn)

        layout.addLayout(options_layout)

        # Preview
        preview_label = QLabel("Preview:")
        preview_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(preview_label)

        self._figure = Figure(figsize=(12, 4), dpi=100)
        self._canvas = FigureCanvas(self._figure)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self._canvas, 1)

        # Close button
        close_layout = QHBoxLayout()
        close_layout.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        close_layout.addWidget(close_btn)
        layout.addLayout(close_layout)

        # Initial preview
        self._update_preview()

    def _load_masks(self, test: dict) -> Optional[np.ndarray]:
        """Load masks from NPZ file for a test."""
        batch_dir = test.get("_batch_dir")
        # Use _original_name for file path (survives renames in UI)
        original_name = test.get("_original_name", test.get("name", ""))

        if not batch_dir:
            return None

        batch_dir = Path(batch_dir)
        safe_name = safe_test_dirname(original_name)
        masks_path = batch_dir / "data" / safe_name / "masks.npz"

        if not masks_path.exists():
            self.logger.warning("Masks not found: %s", masks_path)
            return None

        try:
            data = np.load(str(masks_path))
            masks = data.get("masks")
            self.logger.debug("Loaded masks from %s", masks_path)
            return masks
        except Exception as e:
            self.logger.error("Failed to load masks: %s", e)
            return None

    def _compute_pseudoinverse_reconstruction(self, original: np.ndarray, masks: np.ndarray) -> np.ndarray:
        """Compute pseudoinverse (linear) reconstruction from ground truth and masks."""
        H, W = original.shape[:2]

        # Cast to float64 and rescale masks to [0, 1] so the sensing matrix is
        # numerically well-conditioned regardless of whether masks are uint8
        # 0/255 or already floats. Without this, uint8 masks make the pinv-based
        # reconstruction come back in a very compressed dynamic range.
        original_f = original.astype(np.float64)
        masks_f = masks.astype(np.float64)
        mmax = masks_f.max()
        if mmax > 1.5:
            masks_f = masks_f / mmax

        n_masks = masks_f.shape[0]
        measurements = (masks_f.reshape(n_masks, -1) @ original_f.reshape(-1))

        S = masks_f.reshape(n_masks, -1)

        try:
            S_pinv = np.linalg.pinv(S)
            reconstructed = (S_pinv @ measurements).reshape(H, W)

            # With an underdetermined sensing matrix (typical for sweep), the
            # minimum-norm solution lands in a smaller range than the ground
            # truth. Rescale to [0, 1] so the image fills the colormap instead
            # of appearing washed out after the popup's vmin/vmax=1 imshow.
            rmin = float(reconstructed.min())
            rmax = float(reconstructed.max())
            if rmax > rmin:
                reconstructed = (reconstructed - rmin) / (rmax - rmin)
            else:
                reconstructed = np.zeros_like(reconstructed)
            return reconstructed
        except Exception as e:
            self.logger.error("Pseudoinverse failed: %s", e)
            return original

    def _compute_fista_reconstruction(self, original: np.ndarray, masks: np.ndarray) -> np.ndarray:
        """
        Compute TV-regularized reconstruction from ground truth and masks.
        Uses Total Variation norm which is more appropriate for images than L1.

        Solves: minimize ||y - S @ x||_2^2 + λ TV(x)
        """
        self.logger.info("_compute_fista_reconstruction called: original shape=%s, masks shape=%s",
                        original.shape, masks.shape)

        if not PYLOPS_AVAILABLE:
            self.logger.warning("pylops not available, falling back to pseudoinverse")
            return self._compute_pseudoinverse_reconstruction(original, masks)

        try:
            import pyproximal  # noqa: F401
        except ImportError:
            self.logger.warning("pyproximal not available, falling back to pseudoinverse")
            return self._compute_pseudoinverse_reconstruction(original, masks)

        H, W = original.shape[:2]
        n_masks = masks.shape[0]

        # Compute measurements from ground truth
        y = np.array([np.sum(original.astype(np.float64) * masks[i].astype(np.float64))
                      for i in range(n_masks)], dtype=np.float64)

        # Flatten masks to sensing matrix S: (n_masks, H*W)
        S = np.array([m.flatten().astype(np.float64) for m in masks], dtype=np.float64)

        try:
            # Create pylops LinearOperator
            Sop = pylops.MatrixMult(S)

            # Use Split Bregman for TV-regularized reconstruction
            # This is more appropriate for images than L1 on pixels
            from pylops.optimization.sparsity import splitbregman

            # Create 2D gradient operator for TV norm
            Gop = pylops.Gradient(dims=(H, W), edge=True, kind='forward')

            # TV regularization parameter - higher values = smoother result
            mu = 0.1  # Data fidelity weight
            lamda = 0.05  # TV regularization weight

            self.logger.info("Running Split Bregman TV reconstruction...")

            # Run Split Bregman algorithm for TV minimization
            # Returns (x, niter, cost)
            x_tv, niter, _cost = splitbregman(
                Sop,
                y,
                [Gop],
                niter_outer=20,
                niter_inner=5,
                mu=mu,
                epsRL1s=[lamda],
                tol=1e-4,
                show=False
            )

            self.logger.info("Split Bregman completed in %d iterations", niter)

            # Reshape back to image
            reconstructed = x_tv.reshape(H, W)

            # Normalize to [0, 1]
            reconstructed = np.clip(reconstructed, 0, 1)
            return reconstructed.astype(np.float32)

        except Exception as e:
            self.logger.error("TV reconstruction failed: %s, trying L1-FISTA", e)
            # Fallback to L1-FISTA with higher regularization
            return self._compute_fista_l1_fallback(original, masks, S, y, H, W)

    def _compute_fista_l1_fallback(self, original: np.ndarray, masks: np.ndarray,
                                    S: np.ndarray, y: np.ndarray, H: int, W: int) -> np.ndarray:
        """Fallback L1-FISTA with higher regularization for visible difference."""
        try:
            import pyproximal

            Sop = pylops.MatrixMult(S)

            # Higher regularization for visible difference
            l2 = pyproximal.proximal.L2(Op=Sop, b=y)
            lam = 0.1  # Higher lambda for visible sparsity effect
            l1 = lam * pyproximal.proximal.L1()

            L_val = np.abs((Sop.H * Sop).eigs(1)[0])
            tau = 0.95 / L_val

            x0 = np.zeros(H * W, dtype=np.float64)

            opt = pyproximal.optimization.primal.ProximalGradient(
                l2, l1,
                tau=tau,
                x0=x0,
                niter=200,
                acceleration='fista',
                show=False
            )

            x_fista = opt if isinstance(opt, np.ndarray) else (opt.run() if hasattr(opt, 'run') else opt.x)
            reconstructed = x_fista.reshape(H, W)
            reconstructed = np.clip(reconstructed, 0, 1)
            return reconstructed.astype(np.float32)

        except Exception as e:
            self.logger.error("L1-FISTA fallback failed: %s, using pseudoinverse", e)
            return self._compute_pseudoinverse_reconstruction(original, masks)

    def _compute_fista_alternative(self, original: np.ndarray, masks: np.ndarray,
                                    S: np.ndarray, measurements: np.ndarray,
                                    H: int, W: int) -> np.ndarray:
        """Alternative FISTA implementation using scipy (backup method)."""
        from scipy.optimize import minimize

        lambd = 1e-3

        def objective(x):
            residual = S @ x - measurements
            return 0.5 * np.sum(residual ** 2) + lambd * np.sum(np.abs(x))

        def gradient(x):
            residual = S @ x - measurements
            return S.T @ residual + lambd * np.sign(x)

        # Initial guess from pseudoinverse
        x0 = np.linalg.lstsq(S, measurements, rcond=None)[0]

        try:
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                jac=gradient,
                options={'maxiter': 100, 'disp': False}
            )
            reconstructed = result.x.reshape(H, W)
            reconstructed = np.clip(reconstructed, 0, 1)
            return reconstructed.astype(np.float32)
        except Exception as e:
            self.logger.error("Alternative FISTA failed: %s", e)
            return self._compute_pseudoinverse_reconstruction(original, masks)

    def _compute_psnr(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """Compute PSNR between original and reconstructed images."""
        original = np.asarray(original, dtype=np.float64)
        reconstructed = np.asarray(reconstructed, dtype=np.float64)

        # Ensure same shape
        if original.shape != reconstructed.shape:
            self.logger.warning("Shape mismatch: original=%s, reconstructed=%s",
                              original.shape, reconstructed.shape)
            return 0.0

        # Compute MSE
        mse = np.mean((original - reconstructed) ** 2)
        if mse == 0:
            return float('inf')

        # Assume data range is [0, 1]
        max_val = 1.0
        psnr = 10.0 * np.log10((max_val ** 2) / mse)
        return psnr

    def _get_image_for_column(self, config: ColumnConfig, test_idx: int, image_idx: int):
        """Get the appropriate image for a column configuration."""
        if test_idx >= len(self._tests):
            return None

        test = self._tests[test_idx]
        test_name = test.get("name", "")

        # Check cache for loaded images
        cache_key = (test_name, test_idx)
        if cache_key not in self._images_cache:
            originals, reconstructions, denoised = self._load_test_images(test)
            self._images_cache[cache_key] = {
                'originals': originals,
                'reconstructions': reconstructions,
                'denoised': denoised,
                'masks': None,  # Loaded on demand
                'fista_cache': {},  # Cache computed FISTA results (images)
                'pinv_cache': {},   # Cache computed pseudoinverse results (images)
                'fista_metrics': {},  # Cache metrics: {image_idx: {'psnr': ..., 'time_ms': ...}}
                'pinv_metrics': {},   # Cache metrics for pseudoinverse
            }

        cached = self._images_cache[cache_key]
        originals = cached['originals']
        reconstructions = cached['reconstructions']
        denoised = cached['denoised']

        # Select appropriate image based on type
        if config.col_type == ColumnConfig.TYPE_GROUND_TRUTH:
            if originals is not None and image_idx < len(originals):
                return originals[image_idx]

        elif config.col_type == ColumnConfig.TYPE_LINEAR_RECON:
            # Compute pseudoinverse reconstruction on-the-fly
            self.logger.info("LINEAR_RECON: originals=%s, image_idx=%d",
                           originals is not None, image_idx)
            if originals is not None and image_idx < len(originals):
                if image_idx in cached['pinv_cache']:
                    self.logger.info("LINEAR_RECON: Returning cached pinv result")
                    return cached['pinv_cache'][image_idx]

                # Load masks if not loaded
                if cached['masks'] is None:
                    self.logger.info("LINEAR_RECON: Loading masks...")
                    cached['masks'] = self._load_masks(test)

                self.logger.info("LINEAR_RECON: masks loaded = %s",
                               cached['masks'] is not None)
                if cached['masks'] is not None:
                    self.logger.info("LINEAR_RECON: Computing pseudoinverse...")
                    QApplication.setOverrideCursor(Qt.WaitCursor)
                    try:
                        original = originals[image_idx]
                        t_start = time.perf_counter()
                        recon = self._compute_pseudoinverse_reconstruction(
                            original, cached['masks']
                        )
                        t_end = time.perf_counter()
                        time_ms = (t_end - t_start) * 1000

                        # Compute PSNR
                        psnr = self._compute_psnr(original, recon)

                        # Cache results and metrics
                        cached['pinv_cache'][image_idx] = recon
                        cached['pinv_metrics'][image_idx] = {
                            'psnr': psnr,
                            'time_ms': time_ms
                        }
                        self.logger.info("LINEAR_RECON: Pseudoinverse complete, PSNR=%.2f dB, time=%.1f ms",
                                       psnr, time_ms)
                        return recon
                    finally:
                        QApplication.restoreOverrideCursor()
                elif reconstructions is not None and image_idx < len(reconstructions):
                    # Fallback to stored reconstructions. Applicators (e.g.
                    # sweep ghost imaging) often leave a non-zero DC baseline,
                    # which with imshow(vmin=0, vmax=1) would render a washed
                    # grey image. Rescale to [0, 1] so the full dynamic range
                    # is visible.
                    self.logger.info("LINEAR_RECON: Using stored reconstructions (fallback)")
                    recon = np.asarray(reconstructions[image_idx], dtype=np.float32)
                    rmin = float(recon.min())
                    rmax = float(recon.max())
                    if rmax > rmin:
                        recon = (recon - rmin) / (rmax - rmin)
                    return recon

        elif config.col_type == ColumnConfig.TYPE_ITERATIVE_CS:
            # Compute TV-norm reconstruction on-the-fly
            self.logger.info("ITERATIVE_CS: originals=%s, image_idx=%d",
                           originals is not None, image_idx)
            if originals is not None and image_idx < len(originals):
                if image_idx in cached['fista_cache']:
                    self.logger.info("ITERATIVE_CS: Returning cached TV-norm result")
                    return cached['fista_cache'][image_idx]

                # Load masks if not loaded
                if cached['masks'] is None:
                    self.logger.info("ITERATIVE_CS: Loading masks...")
                    cached['masks'] = self._load_masks(test)

                self.logger.info("ITERATIVE_CS: masks loaded = %s",
                               cached['masks'] is not None)
                if cached['masks'] is not None:
                    self.logger.info("ITERATIVE_CS: Computing TV-norm with %d masks...",
                                   len(cached['masks']))
                    QApplication.setOverrideCursor(Qt.WaitCursor)
                    try:
                        original = originals[image_idx]
                        t_start = time.perf_counter()
                        recon = self._compute_fista_reconstruction(
                            original, cached['masks']
                        )
                        t_end = time.perf_counter()
                        time_ms = (t_end - t_start) * 1000

                        # Compute PSNR
                        psnr = self._compute_psnr(original, recon)

                        # Cache results and metrics
                        cached['fista_cache'][image_idx] = recon
                        cached['fista_metrics'][image_idx] = {
                            'psnr': psnr,
                            'time_ms': time_ms
                        }
                        self.logger.info("ITERATIVE_CS: TV-norm complete, PSNR=%.2f dB, time=%.1f ms",
                                       psnr, time_ms)
                        return recon
                    finally:
                        QApplication.restoreOverrideCursor()
                else:
                    self.logger.warning("ITERATIVE_CS: No masks available, cannot compute TV-norm")

        elif config.col_type == ColumnConfig.TYPE_LINEAR_RECON_DNN:
            if denoised is not None and image_idx < len(denoised):
                return denoised[image_idx]

        return None

    def _get_bottom_text(self, config: ColumnConfig, test_idx: int, image_idx: int = 0) -> str:
        """Get the bottom text for a column based on checkbox selections."""
        parts = []

        # Custom text (always shown if provided)
        if config.custom_text:
            parts.append(config.custom_text)

        # Ground Truth only shows custom text
        if config.col_type == ColumnConfig.TYPE_GROUND_TRUTH:
            return "\n".join(parts) if parts else ""

        if test_idx >= len(self._tests):
            return "\n".join(parts) if parts else ""

        test = self._tests[test_idx]
        test_name = test.get("name", "")

        # Check if we have computed metrics in the cache
        cache_key = (test_name, test_idx)
        cached = self._images_cache.get(cache_key, {})

        # Get computed metrics for Linear Recon and Iterative CS
        computed_metrics = None
        if config.col_type == ColumnConfig.TYPE_LINEAR_RECON:
            computed_metrics = cached.get('pinv_metrics', {}).get(image_idx)
        elif config.col_type == ColumnConfig.TYPE_ITERATIVE_CS:
            computed_metrics = cached.get('fista_metrics', {}).get(image_idx)
        elif config.col_type == ColumnConfig.TYPE_LINEAR_RECON_DNN:
            # Use the on-demand Linear Recon time so "Linear Recon + U-Net"
            # stays consistent with the "Linear Recon" column on the same row.
            computed_metrics = cached.get('pinv_metrics', {}).get(image_idx)

        # Add time if checkbox is checked
        if config.show_time:
            time_text = self._get_time_text(config, test, computed_metrics)
            if time_text:
                parts.append(time_text)

        # Add PSNR if checkbox is checked
        if config.show_psnr:
            if config.col_type == ColumnConfig.TYPE_LINEAR_RECON_DNN:
                psnr = test.get("psnr_denoised")
            elif computed_metrics:
                psnr = computed_metrics.get('psnr')
            else:
                psnr = test.get("psnr_recons")

            if psnr is not None:
                parts.append(f"PSNR: {psnr:.2f} dB")

        # Add SSIM if checkbox is checked
        if config.show_ssim:
            if config.col_type == ColumnConfig.TYPE_LINEAR_RECON_DNN:
                ssim = test.get("ssim_denoised")
            elif computed_metrics:
                # Compute SSIM if we have the images
                ssim = self._get_computed_ssim(cached, image_idx, config.col_type)
            else:
                ssim = test.get("ssim_recons")

            if ssim is not None:
                parts.append(f"SSIM: {ssim:.4f}")

        # Add LPIPS if checkbox is checked
        if config.show_lpips:
            if config.col_type == ColumnConfig.TYPE_LINEAR_RECON_DNN:
                lpips = test.get("lpips_denoised")
            else:
                lpips = test.get("lpips_recons")

            if lpips is not None:
                parts.append(f"LPIPS: {lpips:.4f}")

        return "\n".join(parts)

    def _get_time_text(self, config: ColumnConfig, test: dict, computed_metrics: dict = None) -> str:
        """Get appropriate time text based on column type (automatic)."""
        if config.col_type == ColumnConfig.TYPE_GROUND_TRUTH:
            return ""
        elif config.col_type == ColumnConfig.TYPE_LINEAR_RECON:
            # Use computed metrics if available
            if computed_metrics and 'time_ms' in computed_metrics:
                return f"Time: {computed_metrics['time_ms']:.1f} ms (CPU)"
            recon_time = test.get("timing_reconstruction_ms")
            if recon_time is not None:
                return f"Time: {recon_time:.1f} ms (CPU)"
        elif config.col_type == ColumnConfig.TYPE_ITERATIVE_CS:
            # Use computed metrics if available (this is the actual TV-norm time)
            if computed_metrics and 'time_ms' in computed_metrics:
                return f"Time: {computed_metrics['time_ms']:.1f} ms (CPU)"
            # Fallback to stored time (but this is usually for pseudoinverse)
            recon_time = test.get("timing_reconstruction_ms")
            if recon_time is not None:
                return f"Time: {recon_time:.1f} ms (CPU)"
        elif config.col_type == ColumnConfig.TYPE_LINEAR_RECON_DNN:
            # Reconstruction (CPU) + U-Net inference (GPU). Prefer the on-demand
            # pseudoinverse time measured in the Linear Recon column so both
            # cards agree; fall back to the batch-run timing otherwise.
            if computed_metrics and 'time_ms' in computed_metrics:
                recon_time = computed_metrics['time_ms']
            else:
                recon_time = test.get("timing_reconstruction_ms", 0) or 0
            gpu_time = test.get("timing_gpu_mean_ms", 0) or 0
            total = recon_time + gpu_time
            return f"Time: {total:.1f} ms (CPU+GPU)"
        return ""

    def _get_computed_ssim(self, cached: dict, image_idx: int, col_type: str) -> Optional[float]:
        """Get or compute SSIM for a reconstructed image."""
        # Check if we already have it cached
        if col_type == ColumnConfig.TYPE_LINEAR_RECON:
            metrics = cached.get('pinv_metrics', {}).get(image_idx, {})
            if 'ssim' in metrics:
                return metrics['ssim']
            # Compute if we have the images
            recon = cached.get('pinv_cache', {}).get(image_idx)
        elif col_type == ColumnConfig.TYPE_ITERATIVE_CS:
            metrics = cached.get('fista_metrics', {}).get(image_idx, {})
            if 'ssim' in metrics:
                return metrics['ssim']
            recon = cached.get('fista_cache', {}).get(image_idx)
        else:
            return None

        originals = cached.get('originals')
        if recon is None or originals is None or image_idx >= len(originals):
            return None

        try:
            from skimage.metrics import structural_similarity as ssim_func
            original = np.asarray(originals[image_idx], dtype=np.float64)
            reconstructed = np.asarray(recon, dtype=np.float64)

            # Ensure 2D
            if original.ndim == 3:
                original = original.squeeze()
            if reconstructed.ndim == 3:
                reconstructed = reconstructed.squeeze()

            ssim_val = ssim_func(original, reconstructed, data_range=1.0)

            # Cache the result
            if col_type == ColumnConfig.TYPE_LINEAR_RECON:
                if image_idx not in cached.get('pinv_metrics', {}):
                    cached.setdefault('pinv_metrics', {})[image_idx] = {}
                cached['pinv_metrics'][image_idx]['ssim'] = ssim_val
            elif col_type == ColumnConfig.TYPE_ITERATIVE_CS:
                if image_idx not in cached.get('fista_metrics', {}):
                    cached.setdefault('fista_metrics', {})[image_idx] = {}
                cached['fista_metrics'][image_idx]['ssim'] = ssim_val

            return ssim_val
        except ImportError:
            self.logger.warning("skimage not available for SSIM computation")
            return None
        except Exception as e:
            self.logger.error("Failed to compute SSIM: %s", e)
            return None

    def _update_preview(self):
        """Update the preview figure."""
        self._figure.clear()

        columns = self.column_list.get_columns()
        if not columns:
            ax = self._figure.add_subplot(111)
            ax.text(0.5, 0.5, "Add columns to preview",
                   ha='center', va='center', fontsize=14, color='#999')
            ax.axis('off')
            self._canvas.draw()
            return

        n_cols = len(columns)
        cmap = self.cmap_combo.currentText()
        image_idx = self.image_spin.value()
        test_idx = self.test_combo.currentData() if self.test_combo.count() > 0 else 0

        # Create subplots with more bottom margin for text
        self._figure.subplots_adjust(bottom=0.15, top=0.88, left=0.02, right=0.98, wspace=0.1)
        axes = self._figure.subplots(1, n_cols)
        if n_cols == 1:
            axes = [axes]

        for col, config in enumerate(columns):
            ax = axes[col]

            # Get image using the common test index
            img = self._get_image_for_column(config, test_idx, image_idx)

            if img is not None:
                img = np.array(img)
                if img.ndim == 3 and img.shape[-1] == 1:
                    img = img.squeeze(-1)
                ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
            else:
                ax.text(0.5, 0.5, "No image", ha='center', va='center',
                       fontsize=10, color='#999')
                ax.set_facecolor('#f0f0f0')

            ax.axis('off')

            # Title
            ax.set_title(config.title, fontsize=11, fontweight='bold', pad=8)

            # Bottom text (pass image_idx to get metrics for the correct image)
            bottom_text = self._get_bottom_text(config, test_idx, image_idx)
            if bottom_text:
                ax.text(0.5, -0.08, bottom_text,
                       ha='center', va='top', transform=ax.transAxes,
                       fontsize=9)

        self._canvas.draw()

    def _on_save(self):
        """Save the current figure."""
        self._save_figure(self._figure, "visual_comparison.png")
