"""Reconstruction algorithms used by the mask-application preview popup.

All four methods fall back to Ghost Imaging on failure or missing deps, so
the popup always produces *something* to display.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

# Try to import pylops and pyproximal for FISTA and TV Norm
try:
    import pylops
    import pyproximal
    ITERATIVE_METHODS_AVAILABLE = True
except ImportError:
    ITERATIVE_METHODS_AVAILABLE = False


def reconstruct_ghost_imaging(idx_max: int, measurements, masks,
                              original_shape) -> Optional[np.ndarray]:
    """
    Reconstruct using Ghost Imaging formula.

    Formula: x̂ = (1/N) * Σ(B_i - B_avg) * M_i
    Where B_i is the measurement, B_avg is the DC offset.
    """
    measurements_subset = measurements[:idx_max]
    n = len(measurements_subset)

    if n == 0:
        return None

    # Calculate average measurement (DC offset removal)
    measurement_avg = np.mean(measurements_subset)

    # Accumulate weighted masks
    accumulated = np.zeros(original_shape, dtype=np.float64)
    for i in range(idx_max):
        mask = masks[i].astype(np.float64)
        accumulated += (measurements[i] - measurement_avg) * mask

    # Normalize by number of measurements
    return accumulated / n


def reconstruct_pseudoinverse(idx_max: int, measurements, masks, masks_matrix,
                              original_shape, logger) -> Optional[np.ndarray]:
    """
    Reconstruct using Moore-Penrose Pseudoinverse.

    Formula: x = S^+ @ y
    Where S is the measurement matrix, y is the measurements vector.
    """
    if masks_matrix is None:
        return reconstruct_ghost_imaging(idx_max, measurements, masks, original_shape)

    # Extract subset of masks and measurements
    s_matrix = masks_matrix[:idx_max]  # Shape: (M, N_pixels)
    y = np.array(measurements[:idx_max], dtype=np.float64)

    if len(s_matrix) == 0:
        return None

    try:
        # Compute Moore-Penrose pseudoinverse: x = S^+ @ y
        s_pinv = np.linalg.pinv(s_matrix)
        img_vec = s_pinv @ y
        return img_vec.reshape(original_shape)
    except Exception as e:
        logger.warning("Pseudoinverse failed: %s, falling back to Ghost Imaging", e)
        return reconstruct_ghost_imaging(idx_max, measurements, masks, original_shape)


def reconstruct_fista(idx_max: int, measurements, masks, masks_matrix,
                      original_shape, logger) -> Optional[np.ndarray]:
    """
    Reconstruct using FISTA (Fast Iterative Shrinkage-Thresholding Algorithm).

    Solves: minimize ||y - S@x||² + λ||x||₁
    """
    if not ITERATIVE_METHODS_AVAILABLE:
        logger.warning("pylops/pyproximal not available for FISTA")
        return reconstruct_ghost_imaging(idx_max, measurements, masks, original_shape)

    if masks_matrix is None:
        return reconstruct_ghost_imaging(idx_max, measurements, masks, original_shape)

    # Extract subset of masks and measurements
    s_matrix = masks_matrix[:idx_max]
    y = np.array(measurements[:idx_max], dtype=np.float64)

    if len(s_matrix) == 0:
        return None

    try:
        _m, n_pix = s_matrix.shape

        # Wrap measurement matrix with pylops
        Sop = pylops.MatrixMult(s_matrix)

        # Define data fidelity: f(x) = 1/2 ||S@x - y||²
        l2 = pyproximal.proximal.L2(Op=Sop, b=y)

        # Define L1 regularizer: g(x) = λ||x||₁
        lam = 1e-3  # Regularization parameter
        l1 = lam * pyproximal.proximal.L1()

        # Compute step size
        L_val = np.abs((Sop.H * Sop).eigs(1)[0])
        tau = 0.95 / L_val

        # Initial guess
        x0 = np.zeros(n_pix, dtype=np.float64)

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

        return x_fista.reshape(original_shape)

    except Exception as e:
        logger.warning("FISTA failed: %s, falling back to Ghost Imaging", e)
        return reconstruct_ghost_imaging(idx_max, measurements, masks, original_shape)


def reconstruct_tv_norm(idx_max: int, measurements, masks, masks_matrix,
                        original_shape, logger) -> Optional[np.ndarray]:
    """
    Reconstruct using TV-norm regularization.

    Solves: minimize ||y - S@x||² + λ TV(x)
    """
    if not ITERATIVE_METHODS_AVAILABLE:
        logger.warning("pylops/pyproximal not available for TV Norm")
        return reconstruct_ghost_imaging(idx_max, measurements, masks, original_shape)

    if masks_matrix is None:
        return reconstruct_ghost_imaging(idx_max, measurements, masks, original_shape)

    # Extract subset of masks and measurements
    s_matrix = masks_matrix[:idx_max]
    y = np.array(measurements[:idx_max], dtype=np.float64)

    if len(s_matrix) == 0:
        return None

    try:
        _m, n_pix = s_matrix.shape

        # Wrap measurement matrix with pylops
        Sop = pylops.MatrixMult(s_matrix)

        # Build gradient operator for TV
        Gop = pylops.Gradient(dims=original_shape, sampling=1.0,
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
            x0=np.zeros(n_pix, dtype=np.float64),
            niter=niter, show=False
        )

        # Extract solution
        if hasattr(pd_solver, 'x'):
            x_tv = pd_solver.x
        else:
            x_tv = pd_solver

        return x_tv.reshape(original_shape)

    except Exception as e:
        logger.warning("TV Norm failed: %s, falling back to Ghost Imaging", e)
        return reconstruct_ghost_imaging(idx_max, measurements, masks, original_shape)
