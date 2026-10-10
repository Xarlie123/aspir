"""Shared pieces of the PSNR-convention diagnostic (R2.1b). Read-only on every stored output.

Two scoring functions, and one parameterised function that walks between them:

* ``aspir_r0``: ASPIR's own analyser (``simulation_engine/_5_analyzer/analyzer_noise.py``), called directly.
  This is the path behind every quality number in the R0 manuscript.
* ``shared``: ``benchmarks/spyrit_comparison/compare.py:score`` (clip to [0, 1], data range 1).
* ``metrics``: one function with the factors as switches, used for the step-by-step walk.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BATCH = REPO / "experiments/batch_tests"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "benchmarks/spyrit_comparison"))

logging.disable(logging.CRITICAL)

MANUSCRIPT = Path("/media/beat4/Dades/PhD/Drive_directors/3_ir_profiler_ML/"
                  "2026_SR_Intelligent_Computing_Software_ASPIR_R1")

# Stored runs. UC2_R0 is the run the manuscript reports (Fig. 6a, text of Section 3.2); UC2_RERUN is a second
# training of the same recipe on the same data, the one the SPyRiT comparison, R1.3b and R1.3c used as "UC2".
RUNS = {
    "UC2_R0": BATCH / "batch_test_celeb_hadamard_results_1",
    "UC2_RERUN": BATCH / "uc2_batch_test_celeb_3000_hadamard_results",
    "UC1_SWEEP": BATCH / "uc1_batch_test_strides_paper_2000_results",
    "UC1_CCH": BATCH / "uc1_batch_test_cake_cutting_hadamard_paper_2000_results",
}
UC2_TESTS = {r: f"Celeb - Cake Cutting {round(r * 100)}%" for r in (0.04, 0.08, 0.16, 0.24, 0.32)}

# Numbers printed in the R0 manuscript (sections/3_use_cases.tex line 97)
MANUSCRIPT_UC2 = {"linear_4": 15.3, "linear_32": 18.8, "unet_4": 21.3, "unet_32": 28.2, "gain_4": 6.0,
                  "gain_32": 9.4}


def test_dir(run, test_name):
    return RUNS[run] / "data" / test_name.replace(" ", "_")


def report(run):
    return json.loads((RUNS[run] / "results.batch_analysis_report").read_text())


def stored(run, test_name):
    """originals, network inputs ("reconstructions") and network outputs ("denoised"), and the masks."""
    d = test_dir(run, test_name)
    with np.load(d / "test_images.npz") as z:
        arrays = {k: z[k].astype(np.float64) for k in ("originals", "reconstructions", "denoised")}
    with np.load(d / "masks.npz") as z:
        arrays["masks"] = z["masks"].astype(np.float64)
    return arrays


# --------------------------------------------------------------------------------------------- scoring
_LPIPS = {}


def lpips_values(ref, est):
    """LPIPS exactly as ASPIR's analyser: AlexNet (v0.1), grey repeated to three channels, x -> 2x - 1."""
    import lpips

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev not in _LPIPS:
        _LPIPS[dev] = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(ref), 64):
            r = torch.tensor(ref[i:i + 64], dtype=torch.float32, device=dev)[:, None].repeat(1, 3, 1, 1)
            e = torch.tensor(est[i:i + 64], dtype=torch.float32, device=dev)[:, None].repeat(1, 3, 1, 1)
            out.append(_LPIPS[dev](2 * r - 1, 2 * e - 1).flatten().cpu().numpy())
    return np.concatenate(out)


def metrics(ref, est, data_range="per_image", clip=False, with_lpips=True):
    """Mean of per-image PSNR / SSIM (/ LPIPS).

    data_range: "per_image" -> ref.max() - ref.min() of each reference image (ASPIR R0); a number -> fixed.
    clip: clip the estimate to [0, 1] before scoring.
    """
    ref = np.asarray(ref, dtype=np.float64)
    est = np.asarray(est, dtype=np.float64)
    if clip:
        est = np.clip(est, 0.0, 1.0)
    psnr, ssim = [], []
    for r, e in zip(ref, est, strict=True):
        dr = (r.max() - r.min()) if data_range == "per_image" else float(data_range)
        dr = dr if dr != 0 else 1.0
        psnr.append(peak_signal_noise_ratio(r, e, data_range=dr))
        ssim.append(structural_similarity(r, e, data_range=dr))
    out = {"psnr": float(np.mean(psnr)), "ssim": float(np.mean(ssim))}
    if with_lpips:
        out["lpips"] = float(np.mean(lpips_values(ref, est)))
    return out


def aspir_r0(ref, est):
    """ASPIR's own NoiseAnalyzer on (ref, est); returns mean PSNR / SSIM / LPIPS of est vs ref."""
    from simulation_engine._5_analyzer.analyzer_noise import NoiseAnalyzer

    a = NoiseAnalyzer(list(np.asarray(ref, np.float32)), list(np.asarray(est, np.float32)),
                      list(np.asarray(est, np.float32)))
    a.analyze()
    return {"psnr": a.psnr_rec_mean, "ssim": a.ssim_rec_mean, "lpips": a.lpips_rec_mean}


def shared(ref, est):
    """The SPyRiT comparison's scoring function, imported unchanged."""
    from compare import score

    s = score(ref, est)
    return {k: float(np.mean(v)) for k, v in s.items()}


# ------------------------------------------------------------------------------- linear reconstructions
def hadamard_linear(X, masks):
    """ASPIR's native Hadamard Linear: x = H_S^T y / M (applicator_hadamard.py), M = number of rows."""
    H = masks.reshape(len(masks), -1)
    Y = X.reshape(len(X), -1) @ H.T
    return (Y @ H / len(H)).reshape(X.shape)


def sweep_linear(X, masks):
    """ASPIR's native Sweep Linear: x = S^T (y - mean(y)) / M (applicator_sweep.py)."""
    S = masks.reshape(len(masks), -1)
    Y = X.reshape(len(X), -1) @ S.T
    Y = Y - Y.mean(axis=1, keepdims=True)
    return (Y @ S / len(S)).reshape(X.shape)


def quantize_int8(arr):
    """PostprocessorNN._apply_data_format_to_arrays, INT8 branch (global min / max of the array)."""
    lo, hi = arr.min(), arr.max()
    if hi - lo > 0:
        return np.round((arr - lo) / (hi - lo) * 255) / 255 * (hi - lo) + lo
    return arr


def network_input_map(L, stored_input, levels=255):
    """Recover the global min-max map ASPIR applied before the network: stored = q(a * L + b).

    The global minimum and maximum were taken over the whole dataset (train + val + test), which is not
    stored, so a and b are fitted by least squares on the test images. ``levels`` is the dataset's
    quantisation (255 for INT8, 15 for INT4). Returns a, b, a L + b, and the largest deviation between
    q(a L + b) and the stored network input, in quantisation steps.
    """
    A = np.stack([L.ravel(), np.ones(L.size)], axis=1)
    (a, b), *_ = np.linalg.lstsq(A, stored_input.ravel(), rcond=None)
    approx = a * L + b
    dev_steps = float(np.abs(np.round(approx * levels) - np.round(stored_input * levels)).max())
    return float(a), float(b), approx, dev_steps


def oracle_affine(L, X):
    """Best global gain and offset of L against the ground truth (least squares over the test set)."""
    A = np.stack([L.ravel(), np.ones(L.size)], axis=1)
    (a, b), *_ = np.linalg.lstsq(A, X.ravel(), rcond=None)
    return float(a), float(b), a * L + b


def md_table(rows, cols, fmt=None):
    fmt = fmt or {}
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            cells.append(fmt.get(c, "{:.2f}").format(v) if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
