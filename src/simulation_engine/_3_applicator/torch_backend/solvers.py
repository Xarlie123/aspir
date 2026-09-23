"""Reconstruction algorithms, one torch implementation each, any device.

Batched over images: measurements ``Y`` are ``(B, M)`` and reconstructions
``(B, N)``. Every function mirrors the legacy NumPy / pylops / pyproximal code
it replaces step for step, so the two can be compared at equal iteration count
and parameters; the only differences are the operator (possibly matrix-free)
and the working precision.
"""

import math

import numpy as np
import torch

# Pattern amplitude b of the Fourier family; see operators._FOURIER_B.
_FOURIER_B = 0.5


# ---------------------------------------------------------------------------
# Direct methods
# ---------------------------------------------------------------------------
def correlation(op, Y):
    """Ghost imaging / Sweep Linear: ``x = S^T (y - mean(y)) / M``."""
    return op.adjoint(Y - Y.mean(dim=1, keepdim=True)) / op.shape[0]


def hadamard_linear(op, Y):
    """Hadamard Linear: ``x = S^T y / M``."""
    return op.adjoint(Y) / op.shape[0]


def fourier_groups(freq_indices, phases, coeff_ids, idx_min, idx_max, img_size,
                   phase_steps, device):
    """Index tensors for phase-shifting demodulation of rows ``[idx_min, idx_max)``.

    Returns ``(groups, n_skipped)``. Only coefficients whose whole phase group
    lies inside the range can be demodulated; the rest are skipped, exactly as
    the legacy applicator does.
    """
    sl = slice(idx_min, idx_max)
    cids = np.asarray(coeff_ids)[sl]
    phi = np.asarray(phases, dtype=np.float64)[sl]
    freq = np.asarray(freq_indices)[sl]

    uniq, counts = np.unique(cids, return_counts=True)
    steps = int(phase_steps)
    order, uv, skipped = [], [], 0
    for c, cnt in zip(uniq, counts, strict=True):
        members = np.nonzero(cids == c)[0]
        if cnt != steps or len(np.unique(np.round(phi[members], 9))) != steps:
            skipped += 1
            continue
        members = members[np.argsort(phi[members])]   # canonical phase order
        order.append(members)
        uv.append(freq[members[0]])
    if not order:
        return None, skipped

    order = np.stack(order)
    uv = np.stack(uv).astype(np.int64)
    u, v = uv[:, 0], uv[:, 1]
    pu, pv = (-u) % img_size, (-v) % img_size
    selfc = (pu == u) & (pv == v)
    t = lambda a: torch.as_tensor(a, device=device)  # noqa: E731
    groups = {
        "order": t(order), "steps": steps, "size": img_size,
        "u": t(u), "v": t(v), "pu": t(pu), "pv": t(pv), "self": t(selfc),
    }
    return groups, skipped


def fourier_inverse(Y, groups, dtype):
    """Phase-shift demodulation + conjugate fill + ``ifft2`` (real part).

    Divides the four-step combination by ``2b`` and the three-step one by
    ``3b`` so each recovered value equals ``fft2(image)[u, v]``: unit scale,
    zero offset. Returns ``(images (B, N), spectrum (B, H, W))``.
    """
    B = Y.shape[0]
    n = groups["size"]
    cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    D = Y[:, groups["order"]]                       # (B, G, steps)
    if groups["steps"] == 4:
        F = torch.complex(D[..., 0] - D[..., 2], D[..., 1] - D[..., 3]) / (2 * _FOURIER_B)
    else:
        F = torch.complex(2 * D[..., 0] - D[..., 1] - D[..., 2],
                          math.sqrt(3.0) * (D[..., 1] - D[..., 2])) / (3 * _FOURIER_B)
    F = F.to(cdtype)
    spec = torch.zeros((B, n, n), dtype=cdtype, device=Y.device)
    spec[:, groups["u"], groups["v"]] = F
    spec[:, groups["pu"], groups["pv"]] = F.conj()
    sc = groups["self"]
    if bool(sc.any()):
        spec[:, groups["u"][sc], groups["v"][sc]] = F[:, sc].real.to(cdtype)
    return torch.fft.ifft2(spec).real.reshape(B, -1).to(dtype), spec


# ---------------------------------------------------------------------------
# Pseudoinverse
# ---------------------------------------------------------------------------
def pinv_matrix(op, rcond=None):
    """``S^+`` for a resident explicit operator (cache it per mask set).

    ``rcond`` defaults to NumPy's ``pinv`` cut-off (1e-15, relative) in float64
    and to torch's ``max(M, N) * eps`` in float32, where a 1e-15 cut-off would
    keep round-off singular values and amplify them.
    """
    S = op.dense()
    if S is None:
        return None
    if rcond is None:
        rcond = 1e-15 if S.dtype == torch.float64 else None
    return torch.linalg.pinv(S, rtol=rcond) if rcond is not None else torch.linalg.pinv(S)


def cg(apply, rhs, x0=None, tol=1e-6, maxit=200):
    """Batched conjugate gradient for SPD ``apply``; returns ``(x, iterations)``."""
    x = torch.zeros_like(rhs) if x0 is None else x0.clone()
    r = rhs - apply(x)
    p = r.clone()
    rs = (r * r).sum(dim=1)
    target = (tol * rhs.norm(dim=1)) ** 2
    it = 0
    for it in range(1, maxit + 1):
        if bool((rs <= target).all()):
            it -= 1
            break
        Ap = apply(p)
        denom = (p * Ap).sum(dim=1)
        alpha = torch.where(denom > 0, rs / denom.clamp_min(torch.finfo(rs.dtype).tiny),
                            torch.zeros_like(rs))
        x = x + alpha[:, None] * p
        r = r - alpha[:, None] * Ap
        rs_new = (r * r).sum(dim=1)
        beta = torch.where(rs > 0, rs_new / rs.clamp_min(torch.finfo(rs.dtype).tiny),
                           torch.zeros_like(rs))
        p = r + beta[:, None] * p
        rs = rs_new
    return x, it


def pseudoinverse(op, Y, Spinv=None, tol=1e-6, maxit=200):
    """Minimum-norm least squares ``x = S^+ y``.

    Uses the cached explicit ``S^+`` when available; otherwise solves
    ``S S^T z = y`` by CG and returns ``x = S^T z`` (CGNE), which reaches the
    same minimum-norm solution without forming ``S``. When ``S S^T = c I``
    (Hadamard rows) the solve is exact in closed form.
    Returns ``(x, info)``.
    """
    if Spinv is not None:
        return Y @ Spinv.T, {"solver": "pinv"}
    if op.gram_scalar is not None:
        return op.adjoint(Y) / op.gram_scalar, {"solver": "closed-form"}
    z, it = cg(lambda Z: op.forward(op.adjoint(Z)), Y, tol=tol, maxit=maxit)
    return op.adjoint(z), {"solver": "cgne", "cg_iterations": it}


# ---------------------------------------------------------------------------
# FISTA (l1)
# ---------------------------------------------------------------------------
def power_iteration(op, iters=200, tol=1e-7, seed=0):
    """Largest eigenvalue of ``S^T S`` (the Lipschitz constant of the gradient)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn((1, op.shape[1]), generator=g, dtype=op.dtype).to(op.device)
    x = x / x.norm()
    lam = 0.0
    for _ in range(iters):
        z = op.adjoint(op.forward(x))
        new = float(z.norm())
        if new == 0.0:
            return 0.0
        x = z / new
        if abs(new - lam) <= tol * new:
            lam = new
            break
        lam = new
    return lam


def _soft(v, t):
    return torch.sign(v) * torch.clamp(v.abs() - t, min=0.0)


def fista(op, Y, lam, maxit, L, record_cost=True):
    """``min 0.5 ||S x - y||^2 + lam ||x||_1`` — pyproximal ProximalGradient, fista.

    Same iteration as the legacy path: ``tau = 0.95 / L``, ``x0 = 0``,
    ``x = soft(y - tau grad f(y), tau lam)``, ``t_k = (1 + sqrt(1 + 4 t^2)) / 2``,
    ``y = x + (t_{k-1} - 1) / t_k (x - x_old)``. Returns ``(x, cost)`` with
    ``cost`` of shape ``(B, iterations)`` (empty if ``record_cost`` is False).
    """
    tau = 0.95 / L
    x = torch.zeros((Y.shape[0], op.shape[1]), dtype=op.dtype, device=op.device)
    yk = x.clone()
    t = 1.0
    costs = []
    for _ in range(int(maxit)):
        x_old = x
        grad, _ = op.normal(yk, Y)
        x = _soft(yk - tau * grad, tau * lam)
        t_old = t
        t = (1.0 + math.sqrt(1.0 + 4.0 * t * t)) / 2.0
        yk = x + ((t_old - 1.0) / t) * (x - x_old)
        if record_cost:
            R = op.forward(x) - Y
            costs.append(0.5 * (R * R).sum(dim=1) + lam * x.abs().sum(dim=1))
    cost = torch.stack(costs, dim=1) if costs else torch.empty((Y.shape[0], 0), device=op.device)
    return x, cost


# ---------------------------------------------------------------------------
# Total variation (Chambolle-Pock)
# ---------------------------------------------------------------------------
def gradient_2d(X):
    """Forward differences with a zero last sample on each axis.

    Matches ``pylops.Gradient(kind='forward', edge=False)``: output stacked as
    ``(d/d axis0, d/d axis1)``. ``X`` is ``(B, H, W)``; returns ``(B, 2, H, W)``.
    """
    g = torch.zeros((X.shape[0], 2, *X.shape[1:]), dtype=X.dtype, device=X.device)
    g[:, 0, :-1, :] = X[:, 1:, :] - X[:, :-1, :]
    g[:, 1, :, :-1] = X[:, :, 1:] - X[:, :, :-1]
    return g


def gradient_2d_adjoint(G):
    """Adjoint of :func:`gradient_2d` (minus the discrete divergence)."""
    q0, q1 = G[:, 0], G[:, 1]
    out = torch.zeros_like(q0)
    out[:, 1:, :] += q0[:, :-1, :]
    out[:, :-1, :] -= q0[:, :-1, :]
    out[:, :, 1:] += q1[:, :, :-1]
    out[:, :, :-1] -= q1[:, :, :-1]
    return out


class ProxF:
    """``prox_{tau f}(v) = (I + tau S^T S)^{-1} (v + tau S^T y)`` for f = 0.5||Sx - y||^2.

    Exact wherever possible — the legacy path solves a dense N x N system at
    every iteration:

    * ``"woodbury"`` when ``S S^T = c I`` (Hadamard rows):
      ``(I + tau S^T S)^{-1} = I - tau / (1 + tau c) S^T S``, matrix-free.
    * ``"cholesky"`` for a resident explicit matrix with ``N <= cholesky_max_n``:
      ``I + tau S^T S`` factorised once in float64.
    * ``"cg"`` otherwise: conjugate gradient warm-started from the previous
      iterate — inexact, the only option that scales without forming ``S^T S``.
    """

    def __init__(self, op, tau, cholesky_max_n=4096, cg_tol=1e-6, cg_maxit=50):
        self.op = op
        self.tau = float(tau)
        self.cg_tol = cg_tol
        self.cg_maxit = cg_maxit
        self.cg_iterations = []
        S = op.dense()
        if op.gram_scalar is not None:
            self.mode = "woodbury"
            self.coef = self.tau / (1.0 + self.tau * op.gram_scalar)
        elif S is not None and op.shape[1] <= cholesky_max_n:
            self.mode = "cholesky"
            S64 = S.to(torch.float64)
            C = torch.eye(op.shape[1], dtype=torch.float64, device=op.device) + self.tau * (S64.T @ S64)
            self.chol = torch.linalg.cholesky(C)
        else:
            self.mode = "cg"

    def __call__(self, V, Y, x_prev=None):
        rhs = V + self.tau * self.op.adjoint(Y)
        if self.mode == "woodbury":
            return rhs - self.coef * self.op.adjoint(self.op.forward(rhs))
        if self.mode == "cholesky":
            sol = torch.cholesky_solve(rhs.T.to(torch.float64), self.chol)
            return sol.T.to(V.dtype)
        x, it = cg(lambda Z: Z + self.tau * self.op.adjoint(self.op.forward(Z)),
                   rhs, x0=x_prev, tol=self.cg_tol, maxit=self.cg_maxit)
        self.cg_iterations.append(it)
        return x


def tv_objective(op, X, Y, lam, img_size):
    R = op.forward(X) - Y
    G = gradient_2d(X.reshape(-1, img_size, img_size))
    return 0.5 * (R * R).sum(dim=1) + lam * torch.sqrt((G * G).sum(dim=1)).sum(dim=(1, 2))


def tv(op, Y, lam, niter, img_size, prox_f=None, record_cost=True):
    """``min 0.5 ||S x - y||^2 + lam TV(x)`` — pyproximal PrimalDual, gfirst=True.

    Chambolle-Pock with the legacy step sizes: ``tau = 1/sqrt(8)``,
    ``mu = 1 / (8 tau)``, ``theta = 1``, ``x0 = 0``, isotropic TV
    (``||(grad x)_ij||_2``). The dual update projects onto the ``lam``-ball
    per pixel, the primal update applies the exact data-term prox.
    Returns ``(x, cost, prox_f)``.
    """
    L_grad = 8.0
    tau = 1.0 / math.sqrt(L_grad)
    mu = 1.0 / (tau * L_grad)
    theta = 1.0
    if prox_f is None:
        prox_f = ProxF(op, tau)

    B = Y.shape[0]
    x = torch.zeros((B, op.shape[1]), dtype=op.dtype, device=op.device)
    xhat = x.clone()
    p = torch.zeros((B, 2, img_size, img_size), dtype=op.dtype, device=op.device)
    costs = []
    for _ in range(int(niter)):
        x_old = x
        p = p + mu * gradient_2d(xhat.reshape(B, img_size, img_size))
        norm = torch.sqrt((p * p).sum(dim=1, keepdim=True))
        p = lam * p / torch.maximum(norm, torch.full_like(norm, lam))
        v = x - tau * gradient_2d_adjoint(p).reshape(B, -1)
        x = prox_f(v, Y, x_prev=x)
        xhat = x + theta * (x - x_old)
        if record_cost:
            costs.append(tv_objective(op, x, Y, lam, img_size))
    cost = torch.stack(costs, dim=1) if costs else torch.empty((B, 0), device=op.device)
    return x, cost, prox_f
