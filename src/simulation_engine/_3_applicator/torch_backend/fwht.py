"""Fast Walsh-Hadamard transform in natural (Sylvester) order."""

import torch

# Largest dense Hadamard block applied per stage. H_N factors as
# H_{k1} (x) H_{k2} (x) ... over the row-major digits of the index, so the
# transform is a handful of small matrix products (2 stages at 64x64, 3 at
# 256x256) instead of log2(N) butterfly passes. The arithmetic grows from
# N log2 N to about N * RADIX per stage, but each stage is one GEMM: far fewer
# kernel launches, which is what dominates at small N and on the GPU.
RADIX = 64

_BLOCKS = {}


def _hadamard_block(k, dtype, device):
    key = (k, dtype, str(device))
    H = _BLOCKS.get(key)
    if H is None:
        H = torch.ones((1, 1), dtype=dtype, device=device)
        while H.shape[0] < k:
            H = torch.cat((torch.cat((H, H), 1), torch.cat((H, -H), 1)), 0)
        _BLOCKS[key] = H
    return H


def fwht(x: torch.Tensor) -> torch.Tensor:
    """Unnormalised Walsh-Hadamard transform ``H x`` along the last axis.

    ``x`` has shape ``(..., N)`` with ``N`` a power of two. The result is in
    natural (Sylvester) order, ``H_{2n} = [[H_n, H_n], [H_n, -H_n]]``, so for
    a 2-D Hadamard pattern flattened row-major (a row of
    ``kron(H_s, H_s) = H_{s*s}``) the transform of the image picks out exactly
    that pattern's inner product. ``H`` is symmetric and ``H H = N I``.
    """
    n = x.shape[-1]
    if n & (n - 1):
        raise ValueError(f"FWHT length must be a power of two, got {n}")
    lead = x.shape[:-1]
    y = x.reshape(-1, n)
    b = y.shape[0]
    left, rem = 1, n
    while rem > 1:
        k = min(RADIX, rem)
        rem //= k
        right = n // (left * k)
        H = _hadamard_block(k, y.dtype, y.device)
        # Apply H_k along this digit of the index (H is symmetric).
        y = torch.einsum("blkr,kj->bljr", y.reshape(b, left, k, right), H)
        left *= k
    return y.reshape(*lead, n)
