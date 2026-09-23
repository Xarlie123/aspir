"""Linear sensing operators for the reconstruction backend.

Every solver is written against ``forward`` (``S x``), ``adjoint`` (``S^T y``)
and ``normal`` (``S^T (S x - y)`` plus the residual), batched over images:
``X`` is ``(B, N)`` and ``Y`` is ``(B, M)``. The concrete operator decides how
those products are computed:

* ``ExplicitOperator`` keeps the dense ``(M, N)`` matrix resident on the device
  when it fits, and otherwise streams it from host memory in row blocks sized
  from the free device memory. Every mask family works this way at any size,
  only slower.
* ``HadamardOperator`` never materialises ``S``: patterns are rows of the
  Sylvester Hadamard matrix, so ``S x`` is a fast Walsh-Hadamard transform
  followed by a gather of the family's own rows.
* ``FourierOperator`` does the same with ``torch.fft`` for grayscale Fourier
  patterns ``a + b cos(2 pi (u p / H + v q / W) + phi)``.
"""

import contextlib
import logging

import numpy as np
import torch

from simulation_engine._3_applicator.torch_backend.device import available_bytes
from simulation_engine._3_applicator.torch_backend.fwht import fwht

_log = logging.getLogger("SensingOperator")

# Fraction of the free device memory an explicit matrix may occupy before the
# operator switches to streaming it in row blocks.
_RESIDENT_FRACTION = 0.5

# Fourier pattern offset and amplitude, P = a + b cos(...). Declared here rather
# than imported because .importlinter keeps the pipeline stages mutually
# independent; they must match simulation_engine._2_mask_gen.mask_fourier.
_FOURIER_A = 0.5
_FOURIER_B = 0.5


class SensingOperator:
    """Common interface. Subclasses set ``kind``, ``shape``, ``device``, ``dtype``."""

    kind = "abstract"
    #: ``c`` when ``S S^T = c I`` exactly (orthogonal rows of equal norm), else None.
    gram_scalar = None

    def __init__(self, shape, device, dtype):
        self.shape = (int(shape[0]), int(shape[1]))
        self.device = device
        self.dtype = dtype

    @property
    def matrix_free(self) -> bool:
        return self.kind in ("fwht", "fft")

    def forward(self, X):
        raise NotImplementedError

    def adjoint(self, Y):
        raise NotImplementedError

    def normal(self, X, Y):
        """Return ``(S^T (S X - Y), 0.5 * ||S X - Y||^2 per image)``."""
        R = self.forward(X) - Y
        return self.adjoint(R), 0.5 * (R * R).sum(dim=1)

    def dense(self):
        """The explicit matrix on the device, or None if it is not resident."""
        return None


# ---------------------------------------------------------------------------
class ExplicitOperator(SensingOperator):
    """Dense ``(M, N)`` matrix, resident or streamed in row blocks."""

    def __init__(self, S_host: torch.Tensor, device, dtype, max_device_bytes=None, logger=None):
        super().__init__(S_host.shape, device, dtype)
        log = logger or _log
        m, n = self.shape
        elem = torch.tensor([], dtype=dtype).element_size()
        need = m * n * elem
        self.S = None
        self.S_host = S_host.to(dtype) if S_host.dtype != dtype else S_host

        # On the CPU the host matrix *is* the resident matrix; it is streamed
        # only when the caller caps the memory explicitly (tests, or bounding
        # the working set on a shared host).
        if device.type == "cpu" and (max_device_bytes is None or need <= max_device_bytes):
            self.S = self.S_host
            self.kind = "explicit"
            return

        budget = int(max_device_bytes) if max_device_bytes is not None else \
            int(_RESIDENT_FRACTION * available_bytes(device))
        if need <= budget and device.type != "cpu":
            try:
                self.S = self.S_host.to(device)
                self.kind = "explicit"
                return
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                log.warning("Sensing matrix (%.2f GB) did not fit on %s despite the "
                            "estimate; falling back to chunked streaming.", need / 1e9, device)

        # Chunked: one reusable device block, host rows copied in per pass.
        self.kind = "explicit-chunked"
        rows = max(1, min(m, budget // max(1, n * elem)))
        self.rows_per_chunk = int(rows)
        # Pinned memory speeds up the host-to-device copies, but pin_memory()
        # makes a second copy of the matrix; only do it when that is cheap.
        if need < 0.25 * available_bytes(torch.device("cpu")):
            with contextlib.suppress(RuntimeError):
                self.S_host = self.S_host.pin_memory()
        self._block = torch.empty((self.rows_per_chunk, n), dtype=dtype, device=device)
        log.info("Sensing matrix %.2f GB exceeds the device budget %.2f GB: streaming "
                 "in %d-row blocks (%d passes per product).",
                 need / 1e9, budget / 1e9, self.rows_per_chunk,
                 -(-m // self.rows_per_chunk))

    def _chunks(self):
        m = self.shape[0]
        for r0 in range(0, m, self.rows_per_chunk):
            r1 = min(m, r0 + self.rows_per_chunk)
            blk = self._block[: r1 - r0]
            blk.copy_(self.S_host[r0:r1], non_blocking=True)
            yield r0, r1, blk

    def forward(self, X):
        if self.S is not None:
            return X @ self.S.T
        out = torch.empty((X.shape[0], self.shape[0]), dtype=self.dtype, device=self.device)
        for r0, r1, blk in self._chunks():
            out[:, r0:r1] = X @ blk.T
        return out

    def adjoint(self, Y):
        if self.S is not None:
            return Y @ self.S
        out = torch.zeros((Y.shape[0], self.shape[1]), dtype=self.dtype, device=self.device)
        for r0, r1, blk in self._chunks():
            out += Y[:, r0:r1] @ blk
        return out

    def normal(self, X, Y):
        if self.S is not None:
            return super().normal(X, Y)
        # One streaming pass: S^T S x = sum over blocks of S_b^T (S_b x).
        grad = torch.zeros((X.shape[0], self.shape[1]), dtype=self.dtype, device=self.device)
        resid = torch.zeros(X.shape[0], dtype=self.dtype, device=self.device)
        for r0, r1, blk in self._chunks():
            R = X @ blk.T - Y[:, r0:r1]
            grad += R @ blk
            resid += (R * R).sum(dim=1)
        return grad, 0.5 * resid

    def dense(self):
        return self.S


# ---------------------------------------------------------------------------
class HadamardOperator(SensingOperator):
    """Rows of the natural-order Sylvester Hadamard matrix, applied by FWHT."""

    kind = "fwht"

    def __init__(self, indices, signs, n_pixels, device, dtype):
        super().__init__((len(indices), n_pixels), device, dtype)
        self.idx = torch.as_tensor(np.asarray(indices), dtype=torch.long, device=device)
        self.signs = torch.as_tensor(np.asarray(signs), dtype=dtype, device=device)
        unique = torch.unique(self.idx).numel() == self.idx.numel()
        # Distinct rows of H are orthogonal with squared norm N, so S S^T = N I.
        self.gram_scalar = float(n_pixels) if unique else None
        self._unique = unique

    def forward(self, X):
        return fwht(X)[:, self.idx] * self.signs

    def adjoint(self, Y):
        Z = torch.zeros((Y.shape[0], self.shape[1]), dtype=self.dtype, device=self.device)
        vals = Y * self.signs
        if self._unique:
            Z[:, self.idx] = vals
        else:
            Z.index_add_(1, self.idx, vals)
        return fwht(Z)


def detect_hadamard(masks_2d: np.ndarray, device, dtype, batch=512):
    """Recover ``(indices, signs)`` if every pattern is a +-row of Sylvester H.

    ``fwht`` of a Sylvester row ``h_k`` is ``N e_k`` because ``H H = N I``, so
    one transform per pattern returns its row index and sign directly — no
    knowledge of the family's ordering (natural, Walsh-Paley, Cake-Cutting,
    Cal-Sal) is needed, and the mask generators stay untouched. Returns None as
    soon as a pattern is not an exact +-1 Sylvester row.
    """
    m, n = masks_2d.shape
    if n & (n - 1):
        return None
    # Cheap rejection first: non-bipolar families fail on the first pattern.
    first = np.asarray(masks_2d[0])
    if not np.all(np.abs(first) == 1):
        return None

    idx = np.empty(m, dtype=np.int64)
    sgn = np.empty(m, dtype=np.float64)
    work = torch.float64 if dtype == torch.float64 else torch.float32
    for s in range(0, m, batch):
        P = torch.as_tensor(np.asarray(masks_2d[s:s + batch]), dtype=work, device=device)
        T = fwht(P)
        k = T.abs().argmax(dim=1)
        peak = T.gather(1, k[:, None]).squeeze(1)
        if not torch.allclose(peak.abs(), torch.full_like(peak, float(n)), rtol=1e-4):
            return None
        rest = T.abs().sum(dim=1) - peak.abs()
        if float(rest.max()) > 1e-3 * n:
            return None
        idx[s:s + len(k)] = k.cpu().numpy()
        sgn[s:s + len(k)] = torch.sign(peak).cpu().numpy()
    return idx, sgn


# ---------------------------------------------------------------------------
class FourierOperator(SensingOperator):
    """Grayscale sinusoidal patterns applied through ``torch.fft``.

    With ``X = fft2(x)`` (NumPy sign convention) the inner product with
    ``a + b cos(2 pi (u p/H + v q/W) + phi)`` is
    ``a * X[0,0] + b * Re(exp(j phi) * conj(X[u, v]))``, and the adjoint is
    ``a * sum(y) + b * Re(H W * ifft2(C))`` with ``C[u, v]`` accumulating
    ``y_i exp(j phi_i)``.
    """

    kind = "fft"

    def __init__(self, freq_indices, phases, img_size, device, dtype,
                 a=_FOURIER_A, b=_FOURIER_B):
        n = img_size * img_size
        super().__init__((len(phases), n), device, dtype)
        self.h = self.w = int(img_size)
        freq = np.asarray(freq_indices, dtype=np.int64)
        self.u = torch.as_tensor(freq[:, 0], device=device)
        self.v = torch.as_tensor(freq[:, 1], device=device)
        cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64
        phi = torch.as_tensor(np.asarray(phases, dtype=np.float64), device=device)
        self.rot = torch.polar(torch.ones_like(phi), phi).to(cdtype)
        self.cdtype = cdtype
        self.a = float(a)
        self.b = float(b)
        self._flat = self.u * self.w + self.v

    def forward(self, X):
        B = X.shape[0]
        F = torch.fft.fft2(X.reshape(B, self.h, self.w).to(self.cdtype))
        vals = F.reshape(B, -1)[:, self._flat]
        dc = F[:, 0, 0].real[:, None]
        return (self.a * dc + self.b * (self.rot[None, :] * vals.conj()).real).to(self.dtype)

    def adjoint(self, Y):
        B = Y.shape[0]
        C = torch.zeros((B, self.h * self.w), dtype=self.cdtype, device=self.device)
        C.index_add_(1, self._flat, Y.to(self.cdtype) * self.rot[None, :])
        spatial = torch.fft.ifft2(C.reshape(B, self.h, self.w)).real * (self.h * self.w)
        out = self.b * spatial + self.a * Y.sum(dim=1)[:, None, None]
        return out.reshape(B, -1).to(self.dtype)


# ---------------------------------------------------------------------------
def build_operator(mask, idx_min, idx_max, device, dtype, mode="auto",
                   max_device_bytes=None, logger=None):
    """Pick the cheapest exact operator for ``mask`` rows ``[idx_min, idx_max)``.

    ``mode``: ``"auto"`` (structured when possible, explicit otherwise),
    ``"explicit"`` (always dense), ``"fwht"`` / ``"fft"`` (require that
    structure, error if absent).
    """
    log = logger or _log
    img = int(mask.img_size)
    n = img * img
    sl = slice(idx_min, idx_max)

    # Structured metadata supplied directly (lets callers skip materialising S).
    if mode in ("auto", "fwht") and getattr(mask, "hadamard_indices", None) is not None:
        idx = np.asarray(mask.hadamard_indices)[sl]
        sgn = np.asarray(getattr(mask, "hadamard_signs", np.ones(len(mask.hadamard_indices))))[sl]
        return HadamardOperator(idx, sgn, n, device, dtype)

    fourier_ok = (getattr(mask, "freq_indices", None) is not None
                  and getattr(mask, "phases", None) is not None
                  and str(getattr(mask, "binarization", "none")) == "none")
    if mode in ("auto", "fft") and fourier_ok:
        return FourierOperator(np.asarray(mask.freq_indices)[sl], np.asarray(mask.phases)[sl],
                               img, device, dtype)
    if mode == "fft":
        raise ValueError("FFT operator requested but the mask has no grayscale Fourier metadata")

    masks = getattr(mask, "masks", None)
    if masks is None:
        raise ValueError("The mask has no patterns in 'masks' and no structured metadata.")
    masks_2d = np.asarray(masks[sl]).reshape(-1, n)

    if mode in ("auto", "fwht"):
        found = detect_hadamard(masks_2d, device, dtype)
        if found is not None:
            return HadamardOperator(found[0], found[1], n, device, dtype)
        if mode == "fwht":
            raise ValueError("FWHT operator requested but the patterns are not Sylvester rows")

    np_dtype = np.float64 if dtype == torch.float64 else np.float32
    S_host = torch.from_numpy(np.ascontiguousarray(masks_2d, dtype=np_dtype))
    return ExplicitOperator(S_host, device, dtype, max_device_bytes=max_device_bytes, logger=log)
