"""Shared helpers for the SPyRiT side (run with the SPyRiT venv, .venv/bin/python).

Only SPyRiT's public API and numpy are used here; nothing from ASPIR is imported.
"""

from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
UC2 = REPO / "experiments/batch_tests/uc2_batch_test_celeb_3000_hadamard_results"
RATIOS = (0.04, 0.08, 0.16, 0.24, 0.32)
H = 64
N = H * H


def n_measurements(ratio):
    return int(np.floor(ratio * N + 1e-9))


def sylvester(n):
    h = np.array([[1.0]])
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h


def uc2_masks(ratio):
    with np.load(UC2 / "data" / f"Celeb_-_Cake_Cutting_{round(ratio * 100)}%" / "masks.npz") as z:
        return z["masks"]


def order_for_rows(sylvester_rows):
    """SPyRiT order matrix that selects ASPIR's Sylvester rows, first to last.

    A 64x64 Sylvester-of-Sylvester pattern k is outer(h_a, h_b) with a = k // 64, b = k % 64, and SPyRiT's
    HadamSplit2d measures outer(w_i, w_j), with w the rows of its 1D Walsh matrix, at flat index i * 64 + j.
    Each h_a is one of the w_i (same rows, another order), so pattern k sits at (p[a], p[b]). The order
    matrix gives the requested rows decreasing scores, so that SPyRiT's stable descending sort returns them
    in the same sequence; the remaining positions get lower scores.
    """
    from spyrit.core.torch import walsh_matrix

    W = walsh_matrix(H).numpy()
    S = sylvester(H)
    p = np.array([int(np.flatnonzero(np.all(S[a] == W, axis=1))[0]) for a in range(H)])
    order = np.zeros((H, H), dtype=np.float64)
    score = float(N)
    for k in sylvester_rows:
        a, b = divmod(int(k), H)
        order[p[a], p[b]] = score
        score -= 1.0
    return torch.tensor(order, dtype=torch.float32)
