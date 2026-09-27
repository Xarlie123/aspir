#!/usr/bin/env python
"""Experimental validation on SPIHIM: real single-pixel measurements through ASPIR.

Reproduces every number and figure of the experimental section from the files
that ``scripts/download_spihim.py`` fetches:

    python scripts/download_spihim.py
    python scripts/experimental_spihim_figure.py

Outputs (``--out``, default ``experiments/experimental_spihim/``):

``star_grid.{pdf,png}``      ten-repeat-mean star: rows M/N, columns reference,
                             linear, FISTA, TV, U-Net CelebA (a) / (b) [, STL-10 (a) / (b)]
``star_sim_grid.{pdf,png}``  the same, ASPIR's simulated acquisition of the reference
``r1l1s2p_grid.{pdf,png}``   Thorlabs R1L1S2P (single acquisition, relative metrics)
``curves.{pdf,png}``         PSNR / SSIM / resolution limit vs M/N: experiment,
                             simulation, simulation with calibrated noise
``metrics.csv / .md``        every metric (object x ratio x method x source)
``noise_floor.csv``          each repeat against the ten-repeat mean
``od_sweep.csv``             R1L1S2P at five optical densities, M/N = 0.16
``stl10.csv``                the STL-10 registration attempt
``noise_model.json``         B4: per-measurement noise estimated from the repeats
``summary.json``             devices, versions, choices

Choices that the numbers depend on
----------------------------------
* Sub-sampling uses the *exact* Cake-Cutting rows of Use Case 2
  (``masks.npz`` of each UC2 test), mapped to measured rows by Sylvester index,
  so each U-Net sees the rows it was trained with.
* Measured values are divided by one constant per acquisition so that the
  full-acquisition image peaks at 1 (``experimental.scale_to_unit_peak``):
  ASPIR's simulated scenes live in [0, 1] and the FISTA / TV default weights
  were chosen on that scale. The simulated scene is that same unit-peak image.
* Measured images have no absolute scale (and Hadamard Linear divides by M
  whereas the pseudoinverse family divides by N). Before PSNR / SSIM every
  image is mapped onto the reference by a least-squares affine fit; the
  reference is min-max normalised to [0, 1]; PSNR and SSIM then follow ASPIR's
  analyzer (``data_range`` = reference range = 1).
* Resolution: Michelson contrast of the spokes along circles around the star
  centre, ``C(r) = 2|c_k(r)| / c_0(r)`` with ``c_k`` the angular Fourier
  coefficient at the spoke frequency ``k`` (a sinusoid fit, robust to pixel
  noise). Scanning inwards from ``R_START`` the limit is the radius where
  ``C`` first falls below ``CONTRAST_THRESHOLD``; the corresponding spatial
  frequency is ``k / (2 pi r)`` cycles/pixel (Nyquist 0.5 at ``r = k / pi``).
* The UC2 U-Nets were trained on CelebA with a global min-max over their
  training reconstructions; those constants were not stored. (a) normalises
  each input to [0, 1]; (b) matches the input's mean and standard deviation
  to those of the stored UC2 test inputs at that ratio.
* If ``scripts/train_stl10_unets.py`` has run, the same recipe trained on
  STL-10 adds two columns: (a) per-image min-max and (b) the training
  constants that ASPIR recorded.
"""

import argparse
import csv
import itertools
import json
import logging
import platform
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from scipy.ndimage import map_coordinates  # noqa: E402
from skimage.metrics import peak_signal_noise_ratio, structural_similarity  # noqa: E402

from simulation_engine import experimental as ex  # noqa: E402
from simulation_engine._1_dataset_gen.measured import MeasuredData  # noqa: E402
from simulation_engine._1_dataset_gen.spihim import (  # noqa: E402
    HadamardPngBank,
    load_spihim,
    read_raw_mat,
    saturated_bins,
)

UC2 = REPO / "experiments/batch_tests/uc2_batch_test_celeb_3000_hadamard_results"
JUN = "spihim_2020-Jun-11"
STAR_REPEATS = [f"starSectorD_2_0.0_{i:02d}_raw.mat" for i in range(1, 11)]
R1L1 = "R1L1S2P_1_{od}_raw.mat"
ODS = ("0.0", "0.3", "0.6", "1.0", "1.3")
OD_RATIO = 0.16

METHODS = (("linear", "native"), ("fista", "fista"), ("tv", "tv_norm"))
# U-Net variants; the STL-10 ones join when scripts/train_stl10_unets.py has run.
UNETS = ["unet_a", "unet_b"]
COLUMNS = ["reference", "linear", "fista", "tv", "unet_a", "unet_b"]
LABELS = {"reference": "Reference", "linear": "Linear", "fista": "FISTA", "tv": "TV",
          "unet_a": "U-Net CelebA (a)", "unet_b": "U-Net CelebA (b)",
          "stl_a": "U-Net STL-10 (a)", "stl_b": "U-Net STL-10 (b)"}
STL_NETS = REPO / "experiments" / "experimental_spihim" / "stl10_unets"

CONTRAST_THRESHOLD = 0.1
R_START = 24.0
RADII = np.arange(2.0, 29.0, 0.5)
TIMING_REPEATS = 5
CAL_REALISATIONS = 10

log = logging.getLogger("aspir")


# ---------------------------------------------------------------------------
# UC2 U-Net. The checkpoints predate commit 4eb6f44, which replaced the
# decoder's ConvTranspose2d by Upsample + Conv2d for FPGA toolchains; this is
# the architecture they were trained with (verified against the exported ONNX
# files and the stored test outputs).
# ---------------------------------------------------------------------------
class UNetUC2(nn.Module):
    def __init__(self, features=(8, 16, 32, 64)):
        super().__init__()

        def double(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

        self.downs, self.pools = nn.ModuleList(), nn.ModuleList()
        prev = 1
        for f in features:
            self.downs.append(double(prev, f))
            self.pools.append(nn.MaxPool2d(2, 2))
            prev = f
        self.bottleneck = double(prev, prev * 2)
        self.ups, self.up_convs = nn.ModuleList(), nn.ModuleList()
        prev = features[-1] * 2
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(prev, f, kernel_size=2, stride=2))
            self.up_convs.append(double(f * 2, f))
            prev = f
        self.final_conv = nn.Conv2d(features[0], 1, kernel_size=1)

    def forward(self, x):
        skips = []
        for down, pool in zip(self.downs, self.pools, strict=True):
            x = down(x)
            skips.append(x)
            x = pool(x)
        x = self.bottleneck(x)
        for up, up_conv, skip in zip(self.ups, self.up_convs, skips[::-1], strict=True):
            x = torch.cat((skip, up(x)), dim=1)
            x = up_conv(x)
        return self.final_conv(x)


def uc2_test(ratio):
    return UC2 / "data" / f"Celeb_-_Cake_Cutting_{round(ratio * 100)}%"


def per_image_minmax(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-12)


def load_uc2(ratio, device):
    """UC2 CelebA net (and the STL-10 net, if trained) with their input normalisations.

    ``nets`` maps a variant name to ``(module, input_fn)``: (a) per-image
    min-max for both; (b) for CelebA, mean / std matched to the stored UC2
    test inputs (the training constants were not stored); for STL-10, the
    training constants ``norm_lo`` / ``norm_span`` recorded by ASPIR.
    """
    net = UNetUC2()
    path = UC2 / "models" / f"Celeb_-_Cake_Cutting_{round(ratio * 100)}%.pt"
    net.load_state_dict(torch.load(path, map_location="cpu"), strict=True)
    net.to(device).eval()
    with np.load(uc2_test(ratio) / "test_images.npz") as z:
        inputs = z["reconstructions"].astype(np.float64)
    with np.load(uc2_test(ratio) / "masks.npz") as z:
        masks = z["masks"]
    mu, sd = float(inputs.mean()), float(inputs.std())
    out = {"path": path, "masks": masks, "in_mean": mu, "in_std": sd, "nets": {
        "unet_a": (net, per_image_minmax),
        "unet_b": (net, lambda x: (x - x.mean()) / (x.std() + 1e-12) * sd + mu)}}
    stl = STL_NETS / f"STL10_-_Cake_Cutting_{round(ratio * 100)}%"
    if stl.with_suffix(".pt").exists():
        from simulation_engine._4_postprocessor.models.unet import UNet

        meta = json.loads(stl.with_suffix(".json").read_text())
        snet = UNet(features=meta["recipe"]["architecture_config"]["features"])
        snet.load_state_dict(torch.load(stl.with_suffix(".pt"), map_location="cpu"), strict=True)
        snet.to(device).eval()
        lo, span, scale = meta["norm_lo"], meta["norm_span"], meta["image_max"]
        # Unit-peak measurements vs training images peaking at image_max.
        out["nets"]["stl_a"] = (snet, per_image_minmax)
        out["nets"]["stl_b"] = (snet, lambda x: (x * scale - lo) / span)
        out["stl_path"] = stl.with_suffix(".pt")
    return out


def run_unet(net, x_in, device):
    t = torch.as_tensor(x_in, dtype=torch.float32, device=device)[None, None]
    with torch.no_grad():
        return net(t)[0, 0].double().cpu().numpy()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def minmax(x):
    return (x - x.min()) / (x.max() - x.min())


def fit_to(x, ref):
    """Least-squares ``a x + b`` onto ``ref``."""
    A = np.c_[x.ravel(), np.ones(x.size)]
    coef, *_ = np.linalg.lstsq(A, ref.ravel(), rcond=None)
    return (A @ coef).reshape(ref.shape)


def psnr_ssim(x, ref):
    f = fit_to(x, ref)
    return (float(peak_signal_noise_ratio(ref, f, data_range=1.0)),
            float(structural_similarity(ref, f, data_range=1.0)))


class Star:
    """Centre and spoke count of a sector star, measured on its reference image."""

    def __init__(self, ref):
        c = (ref.shape[0] - 1) / 2
        amp = np.abs(np.fft.rfft(self._ring(ref, c, c, 20.0)))
        self.k = int(np.argmax(amp[2:80]) + 2)
        best = None
        for cy, cx in itertools.product(np.arange(c - 5, c + 5.01, 0.25), repeat=2):
            s = sum(np.abs(np.fft.rfft(self._ring(ref, cy, cx, r))[self.k]) for r in range(10, 26))
            if best is None or s > best[0]:
                best = (s, cy, cx)
        self.cy, self.cx = best[1], best[2]

    @staticmethod
    def _ring(img, cy, cx, r, n=720):
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return map_coordinates(img, [cy + r * np.sin(t), cx + r * np.cos(t)], order=1)

    def contrast(self, img):
        out = []
        for r in RADII:
            c = np.fft.rfft(self._ring(img, self.cy, self.cx, r)) / 720
            a0 = c[0].real
            out.append(2 * np.abs(c[self.k]) / a0 if a0 > 0 else np.nan)
        return np.array(out)

    def limit(self, img):
        """(radius px, cycles/px) where the contrast first drops below the threshold."""
        C = self.contrast(img)
        start = int(np.searchsorted(RADII, R_START))
        if not C[start] >= CONTRAST_THRESHOLD:
            return float("nan"), float("nan")
        for i in range(start, 0, -1):
            if not C[i - 1] >= CONTRAST_THRESHOLD:
                c1, c0 = C[i], C[i - 1]
                w = (c1 - CONTRAST_THRESHOLD) / (c1 - c0) if np.isfinite(c0) else 1.0
                r = RADII[i] - w * (RADII[i] - RADII[i - 1])
                return float(r), float(self.k / (2 * np.pi * r))
        return float(RADII[0]), float(self.k / (2 * np.pi * RADII[0]))


# ---------------------------------------------------------------------------
# Reconstruction with timing
# ---------------------------------------------------------------------------
def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def timed(fn, device, repeats=TIMING_REPEATS):
    out = fn()                                    # untimed: operator / solver setup
    ts = []
    for _ in range(repeats):
        sync(device)
        t0 = time.perf_counter()
        out = fn()
        sync(device)
        ts.append(time.perf_counter() - t0)
    return out, float(np.median(ts) * 1e3)


def reconstruct_all(sub, uc2, devices, scene=None, y=None):
    """Every method on ``sub`` (measured ``y`` or simulated from ``scene``).

    Returns ``{method: image}`` and ``{(method, device): ms}``; images come from
    the first device.
    """
    images, times = {}, {}
    for d_i, device in enumerate(devices):
        for name, method in METHODS:
            sim = ex.measured_simulation(sub, method, device, scene=scene)
            if scene is not None and y is None:
                fn = lambda sim=sim: sim.applicator.apply_mask_range(0, sub.n_patterns, 0)  # noqa: E731
            else:
                yy = sub.y[0] if y is None else y
                fn = lambda sim=sim, yy=yy: sim.applicator.reconstruct(yy)  # noqa: E731
            img, ms = timed(fn, device)
            times[(name, device)] = ms
            if d_i == 0:
                images[name] = np.asarray(img, dtype=np.float64)
        net_device = torch.device(device if device != "auto" else "cpu")
        for mode in UNETS:
            net, input_fn = uc2["nets"][mode]
            net.to(net_device)
            x_in = input_fn(images["linear"])
            out, ms = timed(lambda net=net, x_in=x_in, nd=net_device: run_unet(net, x_in, nd), device)
            times[(mode, device)] = ms
            if d_i == 0:
                images[mode] = out
    for net, _ in uc2["nets"].values():
        net.to(devices[0])
    return images, times


def score(images, ref, star=None):
    rows = {}
    for name, img in images.items():
        p, s = psnr_ssim(img, ref)
        r_lim, f_lim = star.limit(fit_to(img, ref)) if star is not None else (np.nan, np.nan)
        rows[name] = {"psnr": p, "ssim": s, "res_radius_px": r_lim, "res_cyc_per_px": f_lim}
    return rows


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_star(root, bank):
    files = [root / JUN / f for f in STAR_REPEATS]
    sat = sorted(set().union(*(saturated_bins(read_raw_mat(f)).tolist() for f in files)))
    frames = [load_spihim(f, bank, exclude_bins=sat) for f in files]
    y = np.concatenate([d.y for d in frames])
    base = frames[0]
    meta = dict(base.metadata, source_file=[f.name for f in files],
                bins_dropped_saturated=len(sat), saturated_bins_union=len(sat))
    return ex.scale_to_unit_peak(MeasuredData(y=y, patterns=base.patterns, img_size=base.img_size,
                                              acquisition_order=base.acquisition_order, metadata=meta))


def noise_floor(star_data, devices):
    X = ex.reconstruct_measured(star_data, "native", devices[0])
    mean = X.mean(axis=0)
    ref = minmax(mean)
    star = Star(ref)
    rows = []
    for i, x in enumerate(X):
        loo = minmax((X.sum(axis=0) - x) / (len(X) - 1))
        p, s = psnr_ssim(x, ref)
        pl, sl = psnr_ssim(x, loo)
        r, f = star.limit(fit_to(x, ref))
        rows.append({"repeat": i + 1, "psnr_vs_mean": p, "ssim_vs_mean": s,
                     "psnr_vs_leave_one_out": pl, "ssim_vs_leave_one_out": sl,
                     "res_radius_px": r, "res_cyc_per_px": f,
                     "dc_measurement": float(star_data.y[i, 0])})
    return mean, ref, star, rows


def noise_model(star_data):
    """B4: per-measurement noise across the ten repeats, and its level dependence."""
    Y = star_data.y
    mean, sd = Y.mean(axis=0), Y.std(axis=0, ddof=1)
    dc = Y[:, 0]
    # Remove the frame-to-frame gain drift (tracked by the DC row) and look again.
    Yg = Y / (dc[:, None] / dc.mean())
    sdg = Yg.std(axis=0, ddof=1)
    lvl = np.abs(mean)
    A = np.c_[np.ones_like(lvl), lvl ** 2]
    coef, *_ = np.linalg.lstsq(A[1:], sd[1:] ** 2, rcond=None)
    return {
        "n_repeats": int(Y.shape[0]), "n_measurements": int(Y.shape[1]),
        "dc_measurement_per_repeat": dc.tolist(),
        "dc_relative_drift_pct": float((dc.max() - dc.min()) / dc.mean() * 100),
        "noise_std_median": float(np.median(sd[1:])),
        "noise_std_median_after_gain_correction": float(np.median(sdg[1:])),
        "noise_std_dc_row": float(sd[0]),
        "signal_abs_median": float(np.median(lvl[1:])), "signal_abs_max_non_dc": float(lvl[1:].max()),
        "fit_var_eq_a_plus_b_y2": {"a": float(coef[0]), "b": float(coef[1]),
                                   "sqrt_a": float(np.sqrt(max(coef[0], 0))),
                                   "sqrt_b": float(np.sqrt(max(coef[1], 0)))},
        "corr_std_vs_level": float(np.corrcoef(lvl[1:], sd[1:])[0, 1]),
        "per_row_std": sd,
    }


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
def star_experiment(star_data, ref, star, uc2s, devices, nmodel, rng):
    mean_frame = star_data.mean_frame()
    phys = ex.reconstruct_measured(mean_frame, "native", devices[0])[0]   # full, physical units
    X = ex.reconstruct_measured(star_data, "native", devices[0])
    loo_refs = [minmax((X.sum(axis=0) - x) / (len(X) - 1)) for x in X]
    rows, grids, tim = [], {"exp": {}, "sim": {}}, []
    for ratio in ex.RATIOS:
        uc2 = uc2s[ratio]
        sub = ex.subsample_as(mean_frame, uc2["masks"], note=f"UC2 {ratio}")
        # Experiment: ten-repeat-mean measurements.
        imgs, times = reconstruct_all(sub, uc2, devices)
        grids["exp"][ratio] = imgs
        for m, v in score(imgs, ref, star).items():
            rows.append({"object": "star", "source": "experiment", "ratio": ratio, "M": sub.n_patterns,
                         "method": m, **v})
        tim += [{"object": "star", "ratio": ratio, "M": sub.n_patterns, "method": m, "device": d, "ms": t}
                for (m, d), t in times.items()]
        # Single shot against the leave-one-out mean (independent noise).
        single = {m: [] for m in (*[n for n, _ in METHODS], *UNETS)}
        frames = ex.subsample_as(star_data, uc2["masks"])
        for i, loo in enumerate(loo_refs):
            imgs_i, _ = reconstruct_all(frames, uc2, devices[:1], y=frames.y[i])
            for m, v in score(imgs_i, loo, star).items():
                single[m].append(v)
        for m, vs in single.items():
            rows.append({"object": "star", "source": "experiment_single_vs_loo", "ratio": ratio,
                         "M": sub.n_patterns, "method": m,
                         **{k: float(np.nanmean([v[k] for v in vs])) for k in vs[0]},
                         "psnr_std": float(np.std([v["psnr"] for v in vs]))})
        # Simulation: the reference as scene, same rows, ASPIR's default (noiseless) acquisition.
        imgs_s, _ = reconstruct_all(sub, uc2, devices[:1], scene=phys)
        grids["sim"][ratio] = imgs_s
        for m, v in score(imgs_s, ref, star).items():
            rows.append({"object": "star", "source": "simulation", "ratio": ratio, "M": sub.n_patterns,
                         "method": m, **v})
        # Simulation with the noise calibrated on the repeats (B4).
        idx, _ = ex.hadamard_rows(sub.patterns)
        meas_idx, _ = ex.hadamard_rows(star_data.patterns)
        pos = {int(r): i for i, r in enumerate(meas_idx)}
        sigma = nmodel["per_row_std"][[pos[int(r)] for r in idx]]
        y_clean = sub.patterns.reshape(sub.n_patterns, -1).astype(np.float64) @ phys.ravel()
        acc = {}
        for _ in range(CAL_REALISATIONS):
            y_noisy = y_clean + rng.normal(0.0, sigma)
            imgs_c, _ = reconstruct_all(sub, uc2, devices[:1], y=y_noisy)
            for m, v in score(imgs_c, ref, star).items():
                acc.setdefault(m, []).append(v)
        for m, vs in acc.items():
            rows.append({"object": "star", "source": "simulation_calibrated_noise", "ratio": ratio,
                         "M": sub.n_patterns, "method": m,
                         **{k: float(np.nanmean([v[k] for v in vs])) for k in vs[0]},
                         "psnr_std": float(np.std([v["psnr"] for v in vs]))})
        log.info("star M/N=%.2f done", ratio)
    return rows, grids, tim


def r1l1_experiment(root, bank, uc2s, devices):
    data = ex.scale_to_unit_peak(load_spihim(root / JUN / R1L1.format(od="0.0"), bank))
    ref = minmax(ex.reconstruct_measured(data, "native", devices[0])[0])
    rows, grid, tim = [], {}, []
    for ratio in ex.RATIOS:
        uc2 = uc2s[ratio]
        sub = ex.subsample_as(data, uc2["masks"])
        imgs, times = reconstruct_all(sub, uc2, devices)
        grid[ratio] = imgs
        for m, v in score(imgs, ref).items():
            rows.append({"object": "r1l1s2p", "source": "experiment", "ratio": ratio, "M": sub.n_patterns,
                         "method": m, **v})
        tim += [{"object": "r1l1s2p", "ratio": ratio, "M": sub.n_patterns, "method": m, "device": d, "ms": t}
                for (m, d), t in times.items()]
    return ref, rows, grid, tim, data.metadata


NOISE_REPEATS = [f"noObjectD_1_0.3_0{i}_raw.mat" for i in range(1, 5)]
DC_BAND = (450.0, 700.0)


def sigma_counts_4ms(root, bank):
    """Per-measurement noise (counts) from the four no-object repeats at 4 ms (as R1L1S2P)."""
    Y = np.concatenate([load_spihim(root / JUN / f, bank).y for f in NOISE_REPEATS])
    sd = Y.std(axis=0, ddof=1)
    level = np.abs(Y.mean(axis=0))
    return float(np.median(sd[1:])), float(np.corrcoef(level[1:], sd[1:])[0, 1])


def od_sweep(root, bank, uc2, ref, devices, rng):
    """R1L1S2P at five ODs, M/N = OD_RATIO, against the OD 0.0 reference.

    Next to each measurement: ASPIR's noiseless simulation of the OD 0.0 scene
    (identical for every OD, since only light level changes) and a simulation
    with calibrated noise: the OD 0.0 measurement attenuated by the measured
    transmission (DC row, 450-700 nm, the same bins at every OD) plus additive Gaussian
    noise with the no-object repeats' sigma.
    """
    sigma, corr = sigma_counts_4ms(root, bank)
    raw0 = load_spihim(root / JUN / R1L1.format(od="0.0"), bank)
    peak0 = 1.0 / ex.scale_to_unit_peak(raw0).metadata["y_scale"]
    sub0 = ex.subsample_as(raw0, uc2["masks"])
    # Transmission on one fixed bin set: the band minus OD 0.0's clipped bins.
    sat0 = saturated_bins(read_raw_mat(root / JUN / R1L1.format(od="0.0")))

    def dc_band(od):
        return load_spihim(root / JUN / R1L1.format(od=od), bank, spectral="band", band=DC_BAND,
                           exclude_saturated=False, exclude_bins=sat0).y[0, 0]

    dc0 = dc_band("0.0")
    rows = []
    for od in ODS:
        raw = load_spihim(root / JUN / R1L1.format(od=od), bank)
        data = ex.scale_to_unit_peak(raw)
        full = ex.reconstruct_measured(data, "native", devices[0])[0]
        p_full, s_full = psnr_ssim(full, ref)
        sub = ex.subsample_as(data, uc2["masks"])
        imgs, _ = reconstruct_all(sub, uc2, devices[:1])
        dc = dc_band(od)
        att = float(dc / dc0)
        row = {"od": od, "transmission_450_700nm": att, "expected_10_pow_minus_od": 10 ** -float(od),
               "bins_dropped_saturated": data.metadata["bins_dropped_saturated"],
               "noise_sigma_counts": sigma, "full_linear_psnr": p_full, "full_linear_ssim": s_full}
        for m, v in score(imgs, ref).items():
            row[f"{m}_psnr"] = v["psnr"]
            row[f"{m}_ssim"] = v["ssim"]
        acc = {}
        for _ in range(CAL_REALISATIONS):
            y = (att * sub0.y[0] + rng.normal(0.0, sigma, sub0.n_patterns)) / (att * peak0)
            imgs_c, _ = reconstruct_all(sub0, uc2, devices[:1], y=y)
            for m, v in score(imgs_c, ref).items():
                acc.setdefault(m, []).append(v["psnr"])
        for m, vs in acc.items():
            row[f"{m}_psnr_sim_calibrated"] = float(np.mean(vs))
        rows.append(row)
    base = rows[0]
    for row in rows:                      # noiseless simulation of the OD 0.0 scene
        for name, _ in METHODS:
            row[f"{name}_psnr_sim_noiseless"] = base[f"{name}_psnr"]
        for m in UNETS:
            row[f"{m}_psnr_sim_noiseless"] = base[f"{m}_psnr"]
    return rows, {"sigma_counts_4ms": sigma, "corr_sigma_vs_level": corr}


def stl10_attempt(root, bank, devices):
    """B2.5: flat field, band and a registration search against the digital originals."""
    from PIL import Image
    from scipy.ndimage import gaussian_filter, shift, zoom

    def place(img, scale):
        z = zoom(img, scale, order=1)
        if z.shape[0] >= 64:
            o = (z.shape[0] - 64) // 2
            return z[o:o + 64, o:o + 64]
        p = 64 - z.shape[0]
        return np.pad(z, ((p // 2, p - p // 2),) * 2, mode="edge")

    rows = []
    cases = [("stl10_01_32ms_ps", "stl_test_01.png"), ("stl10_01_64ms_ps", "stl_test_01.png"),
             ("stl10_01_32ms_cp", "stl_test_01.png"), ("stl10_08_32ms_ps", "stl_test_08.png")]
    flat = root / JUN / "noObjectD_1_0.0_raw.mat"
    for name, png in cases:
        target = minmax(np.asarray(Image.open(root / JUN / "stl_images" / png).convert("L"), float))
        for spectral, band in (("sum", None), ("band", (450.0, 700.0))):
            d = load_spihim(root / JUN / f"{name}_raw.mat", bank, spectral=spectral, band=band,
                            orientation={"rot90": 0, "flip": False}, flat_field_mat=flat)
            x = ex.reconstruct_measured(d, "native", devices[0])[0]
            ff = d.flat_field
            ffc = x / np.maximum(ff, 0.05 * ff.max())
            for variant, img in (("raw", x), ("flat_field", ffc), ("flat_field_smooth", gaussian_filter(ffc, 1))):
                best = (-np.inf, None, None)
                for k, fl, inv, sc in itertools.product(range(4), (0, 1), (0, 1), (0.9, 1.0, 1.1)):
                    im = np.rot90(img, k)
                    im = im[:, ::-1] if fl else im
                    im = place(-im if inv else im, sc)
                    for dy, dx in itertools.product(range(-4, 5, 2), repeat=2):
                        f = np.clip(fit_to(shift(im, (dy, dx), order=1, mode="nearest"), target), 0, 1)
                        p = peak_signal_noise_ratio(target, f, data_range=1.0)
                        if p > best[0]:
                            best = (p, structural_similarity(target, f, data_range=1.0),
                                    (k, fl, inv, sc, dy, dx))
                rows.append({"acquisition": name, "target": png, "spectral": spectral,
                             "variant": variant, "best_psnr": float(best[0]), "ssim_at_best": float(best[1]),
                             "rot90_flip_invert_scale_dy_dx": str(best[2])})
    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def save_grid(grid, ref, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ratios = list(grid)
    fig, axes = plt.subplots(len(ratios), len(COLUMNS), figsize=(1.6 * len(COLUMNS), 1.6 * len(ratios) + 0.4))
    for i, ratio in enumerate(ratios):
        for j, col in enumerate(COLUMNS):
            ax = axes[i, j]
            img = ref if col == "reference" else np.clip(fit_to(grid[ratio][col], ref), 0, 1)
            ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(LABELS[col], fontsize=8)
            if j == 0:
                ax.set_ylabel(f"M/N={ratio:.2f}", fontsize=8)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=200)
    plt.close(fig)


def save_curves(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {"experiment": "-", "simulation": "--", "simulation_calibrated_noise": ":"}
    names = (*[n for n, _ in METHODS], *UNETS)
    colors = {m: f"C{i}" for i, m in enumerate(names)}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, key, label in zip(axes, ("psnr", "ssim", "res_cyc_per_px"),
                              ("PSNR (dB)", "SSIM", "resolution limit (cycles/px)"), strict=True):
        for src, ls in styles.items():
            for m, c in colors.items():
                pts = sorted((r["ratio"], r[key]) for r in rows
                             if r["object"] == "star" and r["source"] == src and r["method"] == m)
                if pts:
                    ax.plot(*zip(*pts, strict=True), ls, color=c, marker="o", ms=3,
                            label=f"{LABELS[m]}" if src == "experiment" else None)
        ax.set_xlabel("M/N")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    axes[2].text(0.02, 0.02, "solid: experiment  dashed: simulation\ndotted: simulation, calibrated noise",
                 transform=axes[2].transAxes, fontsize=7)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=200)
    plt.close(fig)


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def res(v, spec):
    """A resolution entry; NaN means the spokes are unresolved even at R_START."""
    return f"unresolved at r = {R_START:g}" if np.isnan(v) else format(v, spec)


def fmt(v):
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def write_md(path, metrics, timings, noise_rows, od_rows, stl_rows, summary):
    lines = ["# SPIHIM experimental validation", "",
             f"Devices: {', '.join(summary['devices'])} ({summary.get('gpu', 'no GPU')}). "
             f"Contrast threshold {CONTRAST_THRESHOLD}; star centre ({summary['star_centre'][0]:.2f}, "
             f"{summary['star_centre'][1]:.2f}) px, k = {summary['star_k']} spoke cycles.", ""]
    for obj in ("star", "r1l1s2p"):
        for src in ("experiment", "experiment_single_vs_loo", "simulation", "simulation_calibrated_noise"):
            sel = [r for r in metrics if r["object"] == obj and r["source"] == src]
            if not sel:
                continue
            lines += [f"## {obj} — {src}", "", "| M/N | M | method | PSNR (dB) | SSIM | res. radius (px) | res. (cyc/px) |",
                      "|---|---|---|---|---|---|---|"]
            lines += [f"| {r['ratio']:.2f} | {r['M']} | {r['method']} | {r['psnr']:.2f} | {r['ssim']:.3f} | "
                      f"{res(r['res_radius_px'], '.2f')} | {res(r['res_cyc_per_px'], '.3f')} |" for r in sel]
            lines.append("")
    lines += ["## Timings (median ms, after one untimed call)", "", "| object | M/N | method | device | ms |",
              "|---|---|---|---|---|"]
    lines += [f"| {t['object']} | {t['ratio']:.2f} | {t['method']} | {t['device']} | {t['ms']:.2f} |"
              for t in timings]
    for title, rows in (("Noise floor (single repeat vs ten-repeat mean)", noise_rows),
                        (f"OD sweep, R1L1S2P, M/N = {OD_RATIO}", od_rows),
                        ("STL-10 registration attempt", stl_rows)):
        if not rows:
            continue
        keys = list(rows[0])
        lines += ["", f"## {title}", "", "| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
        lines += ["| " + " | ".join(fmt(r[k]) for k in keys) + " |" for r in rows]
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", type=Path, default=REPO / "datasets" / "spihim")
    ap.add_argument("--out", type=Path, default=REPO / "experiments" / "experimental_spihim")
    ap.add_argument("--devices", nargs="+", default=None,
                    help="devices to time (default: cuda cpu, or cpu); metrics use the first")
    ap.add_argument("--skip-stl10", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    log.setLevel(logging.INFO)
    devices = args.devices or (["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"])
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    bank = HadamardPngBank(args.data / "Hadamard_64x64")
    uc2s = {r: load_uc2(r, devices[0]) for r in ex.RATIOS}
    if all("stl_a" in u["nets"] for u in uc2s.values()):
        UNETS.extend(["stl_a", "stl_b"])
        COLUMNS.extend(["stl_a", "stl_b"])

    star_data = load_star(args.data, bank)
    _, ref, star, noise_rows = noise_floor(star_data, devices)
    nmodel = noise_model(star_data)
    star_rows, grids, t_star = star_experiment(star_data, ref, star, uc2s, devices, nmodel, rng)
    r_ref, r_rows, r_grid, t_r, r_meta = r1l1_experiment(args.data, bank, uc2s, devices)
    od_rows, od_noise = od_sweep(args.data, bank, uc2s[OD_RATIO], r_ref, devices, rng)
    stl_rows = [] if args.skip_stl10 else stl10_attempt(args.data, bank, devices)
    metrics, timings = star_rows + r_rows, t_star + t_r

    summary = {
        "devices": devices, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cpu": platform.processor() or platform.machine(), "torch": torch.__version__,
        "numpy": np.__version__, "star_centre": [star.cy, star.cx], "star_k": star.k,
        "contrast_threshold": CONTRAST_THRESHOLD, "reference_resolution": star.limit(ref),
        "uc2_checkpoints": {str(r): str(u["path"].relative_to(REPO)) for r, u in uc2s.items()},
        "uc2_input_stats": {str(r): [u["in_mean"], u["in_std"]] for r, u in uc2s.items()},
        "stl10_checkpoints": {str(r): str(u["stl_path"].relative_to(REPO)) for r, u in uc2s.items()
                              if "stl_path" in u},
        "star_metadata": star_data.metadata, "r1l1s2p_metadata": r_meta,
        "seed": args.seed,
    }
    save_grid(grids["exp"], ref, args.out / "star_grid", "starSectorD_2_0.0, ten-repeat mean (experiment)")
    save_grid(grids["sim"], ref, args.out / "star_sim_grid", "starSectorD_2_0.0 reference, ASPIR simulation")
    save_grid(r_grid, r_ref, args.out / "r1l1s2p_grid", "R1L1S2P_1_0.0 (experiment, relative reference)")
    save_curves(metrics, args.out / "curves")
    write_csv(args.out / "metrics.csv", metrics)
    write_csv(args.out / "timings.csv", timings)
    write_csv(args.out / "noise_floor.csv", noise_rows)
    write_csv(args.out / "od_sweep.csv", od_rows)
    if stl_rows:
        write_csv(args.out / "stl10.csv", stl_rows)
    nm = {k: v for k, v in nmodel.items() if k != "per_row_std"}
    nm["units"] = "measurements divided by y_scale^-1 (unit-peak image); multiply by 1/y_scale for counts"
    nm["y_scale"] = star_data.metadata["y_scale"]
    nm["noise_std_median_counts_star_8ms"] = nm["noise_std_median"] / nm["y_scale"]
    nm["noObject_0.3_repeats_4ms"] = od_noise
    (args.out / "noise_model.json").write_text(json.dumps(nm, indent=2))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    write_md(args.out / "metrics.md", metrics, timings, noise_rows, od_rows, stl_rows, summary)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
