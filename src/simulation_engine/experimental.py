"""Experimental measurements through ASPIR: sub-sampling, reconstruction, simulation.

This module sits above the pipeline stages because it needs two of them at
once: the mask generators (``_2``) define ASPIR's pattern orderings, and the
reconstruction backend (``_3``) recognises which Sylvester Hadamard row each
pattern is. That lets a full measured acquisition be sub-sampled in exactly
the order ASPIR's own masks use, so a network trained in simulation at ratio
``M/N`` sees the same rows on real data.

Typical use::

    data = load_spihim(mat, bank)                        # 4096 measured rows
    sub = subsample(data, "cake_cutting", 0.16)          # ASPIR's first 655 rows
    x = reconstruct_measured(sub, "native")              # Hadamard Linear
    x_sim = reconstruct_simulated(scene, sub, "native")  # same rows, simulated
"""

import logging
from pathlib import Path

import numpy as np

from simulation_engine._1_dataset_gen.measured import DatasetFromMeasurements, MeasuredData
from simulation_engine._2_mask_gen.mask_cal_sal import MaskCalSal
from simulation_engine._2_mask_gen.mask_hadamard import MaskHadamard
from simulation_engine._2_mask_gen.mask_hadamard_cake_cutting import MaskHadamardCakeCutting
from simulation_engine._2_mask_gen.mask_hadamard_walsh_paley import MaskHadamardWalshPaley
from simulation_engine._2_mask_gen.mask_measured import MaskMeasured
from simulation_engine.simulation import Simulation

_log = logging.getLogger("aspir").getChild("Experimental")

# ASPIR's Hadamard orderings, keyed like the batch configs' ``mask_type``.
ORDERINGS = {
    "natural": MaskHadamard,
    "walsh_paley": MaskHadamardWalshPaley,
    "cake_cutting": MaskHadamardCakeCutting,
    "cal_sal": MaskCalSal,
}
# Ratios of Use Case 2 (M = floor(ratio * N), as its masks.npz have).
RATIOS = (0.04, 0.08, 0.16, 0.24, 0.32)

_ORDER_CACHE = {}


def n_measurements(ratio, n_pixels) -> int:
    """``floor(ratio * N)``; the tolerance keeps e.g. 0.24 * 4096 at 983."""
    return int(np.floor(float(ratio) * int(n_pixels) + 1e-9))


def hadamard_rows(patterns):
    """``(indices, signs)`` of +-1 Sylvester patterns, or ValueError if they are not."""
    import torch

    from simulation_engine._3_applicator.torch_backend.operators import detect_hadamard

    patterns = np.asarray(patterns)
    flat = patterns.reshape(patterns.shape[0], -1).astype(np.float64)
    found = detect_hadamard(flat, torch.device("cpu"), torch.float64)
    if found is None:
        raise ValueError("the patterns are not +-1 rows of a Sylvester Hadamard matrix")
    idx, sgn = found
    return np.asarray(idx, dtype=np.int64), np.asarray(sgn, dtype=np.float64)


def aspir_ordering(ordering, img_size=64, cache_dir=None):
    """Sylvester ``(indices, signs)`` of ASPIR's full ``ordering`` mask, in order.

    Generating the Cake-Cutting mask at 64x64 takes ~20 s, so the result is
    kept in memory and, with ``cache_dir``, on disk
    (``hadamard_order_<ordering>_<img_size>.npz``).
    """
    if ordering not in ORDERINGS:
        raise ValueError(f"ordering must be one of {tuple(ORDERINGS)}, got {ordering!r}")
    key = (ordering, int(img_size))
    if key in _ORDER_CACHE:
        return _ORDER_CACHE[key]
    path = None
    if cache_dir is not None:
        path = Path(cache_dir) / f"hadamard_order_{ordering}_{int(img_size)}.npz"
        if path.exists():
            with np.load(path) as npz:
                _ORDER_CACHE[key] = (npz["indices"], npz["signs"])
            return _ORDER_CACHE[key]
    _log.info("Generating ASPIR's %s ordering at %dx%d", ordering, img_size, img_size)
    mask = ORDERINGS[ordering](img_size=int(img_size), min_idx=0, max_idx=int(img_size) ** 2)
    mask.generate_masks()
    _ORDER_CACHE[key] = hadamard_rows(mask.masks)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, indices=_ORDER_CACHE[key][0], signs=_ORDER_CACHE[key][1])
    return _ORDER_CACHE[key]


def subsample(data: MeasuredData, ordering, ratio, cache_dir=None) -> MeasuredData:
    """The first ``floor(ratio * N)`` rows of ASPIR's ``ordering``, taken from ``data``.

    Each measured row is matched to ASPIR's row by its Sylvester index, and
    ``y`` and the pattern are multiplied by the sign that turns the measured
    pattern into ASPIR's one, so the returned patterns are exactly those of
    ASPIR's mask. ``ordering="acquisition"`` keeps the instrument's own order
    (SPIHIM: "variance"), which also serves partial acquisitions.
    """
    n_pix = data.img_size ** 2
    m = n_measurements(ratio, n_pix)
    note = f"{ordering} M/N={ratio} (M={m})"
    if ordering == "acquisition":
        if m > data.n_patterns:
            raise ValueError(f"only {data.n_patterns} rows were measured, {m} requested")
        return data.subset(np.argsort(data.acquisition_order, kind="stable")[:m], note=note)

    a_idx, a_sgn = aspir_ordering(ordering, data.img_size, cache_dir)
    return _take_rows(data, a_idx[:m], a_sgn[:m], note)


def subsample_as(data: MeasuredData, patterns, note=None) -> MeasuredData:
    """The rows of ``data`` that reproduce ``patterns`` (e.g. a saved ``masks.npz``), in order.

    Use it to match a network to the exact rows it was trained with. Before
    v1.3, ASPIR's Cake-Cutting order did not fix the order of equal block
    counts, so masks saved by older versions may hold different rows among
    ties at the cut-off ratio than today's stable order.
    """
    idx, sgn = hadamard_rows(patterns)
    return _take_rows(data, idx, sgn, note or f"as given patterns (M={len(idx)})")


def _take_rows(data, a_idx, a_sgn, note):
    meas_idx, meas_sgn = hadamard_rows(data.patterns)
    where = {int(r): i for i, r in enumerate(meas_idx)}
    missing = [int(r) for r in a_idx if int(r) not in where]
    if missing:
        raise ValueError(f"{len(missing)} of the {len(a_idx)} requested Sylvester rows were "
                         f"not measured (e.g. row {missing[0]})")
    rows = np.array([where[int(r)] for r in a_idx], dtype=np.int64)
    flip = np.asarray(a_sgn, dtype=np.float64) * meas_sgn[rows]
    out = data.subset(rows, note=note)
    out.y = out.y * flip[None, :]
    out.patterns = (out.patterns * flip[:, None, None]).astype(data.patterns.dtype)
    return out


def measured_simulation(data: MeasuredData, method="native", device="auto",
                        solver_params=None, scene=None, logger=None) -> Simulation:
    """A :class:`Simulation` wired to ``data``'s patterns.

    The dataset holds ``scene`` (for the simulated path) or else ``data``'s
    reference; the measured path ignores it.
    """
    logger = logger or _log
    if scene is not None:
        data = MeasuredData(y=data.y, patterns=data.patterns, img_size=data.img_size,
                            acquisition_order=data.acquisition_order, reference=scene,
                            flat_field=data.flat_field, metadata=data.metadata)
    sim = Simulation(logger)
    sim.set_dataset(DatasetFromMeasurements(measurements=data, logger=logger))
    sim.mask = MaskMeasured(data.patterns, logger=logger)
    sim.set_applicator(reconstruction_method=method, device=device, solver_params=solver_params)
    return sim


def reconstruct_measured(data: MeasuredData, method="native", device="auto",
                         solver_params=None, simulation=None) -> np.ndarray:
    """Reconstruct every frame of ``data`` from its measured ``y``: ``(F, H, W)``."""
    sim = simulation or measured_simulation(data, method, device, solver_params)
    return np.atleast_3d(sim.applicator.reconstruct(data.y)).reshape(
        data.n_frames, data.img_size, data.img_size)


def reconstruct_simulated(scene, data: MeasuredData, method="native", device="auto",
                          solver_params=None, simulation=None) -> np.ndarray:
    """ASPIR's simulated acquisition of ``scene`` with ``data``'s patterns: ``(H, W)``.

    This is the ordinary simulated path (detector values computed from the
    scene, no added noise by default), on the same rows as the measurement.
    """
    sim = simulation or measured_simulation(data, method, device, solver_params, scene=scene)
    return sim.applicator.apply_mask_range(0, data.n_patterns, 0)


def scale_to_unit_peak(data: MeasuredData) -> MeasuredData:
    """Divide ``y`` by one constant so the frame-averaged image peaks at 1.

    Measured values are in instrument units (SPIHIM: ~1e7 counts), whereas
    ASPIR's simulated scenes are in [0, 1] and the iterative solvers' default
    regularisation weights were chosen on that scale; on raw counts they have
    no effect. The image used is the minimum-norm one, ``S^T y / N``, which is
    exact for a full orthogonal pattern set. Only the measurements themselves
    are used; the constant goes into ``metadata["y_scale"]``.
    """
    flat = data.patterns.reshape(data.n_patterns, -1).astype(np.float64)
    image = data.y.mean(axis=0) @ flat / flat.shape[1]
    peak = float(image.max())
    if not peak > 0:
        raise ValueError("the reconstructed image has no positive peak to scale by")
    out = data.subset(np.arange(data.n_patterns))
    out.y = data.y / peak
    out.metadata["y_scale"] = 1.0 / peak
    return out
