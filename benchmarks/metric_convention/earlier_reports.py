#!/usr/bin/env python
"""Task E: does the R1.1 FSPI / Cake-Cutting crossover survive the proposed convention? (ASPIR venv)

    ../../.venv/bin/python earlier_reports.py

R1.1 scored 50 CelebA images with a third convention: every reconstruction AND every ground truth min-max
normalised per image, data range 1. Here both families are scored on the 300 UC2 test images:
* R1.1 convention (per-image min-max of estimate and reference, data range 1, no LPIPS);
* the proposed convention (shared function: estimate on the image's scale, clip, data range 1).
FSPI: MaskFourier (four-step, circular ordering) + ApplicatorFourier (unit scale by construction).
Cake-Cutting: UC2's stored rows, pseudoinverse (native Hadamard Linear x M/N).
Writes earlier_reports.md.
"""

import logging

import numpy as np
from common import HERE, UC2_TESTS, hadamard_linear, md_table, shared, stored
from scales import ArrayDataset
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from simulation_engine._2_mask_gen.mask_fourier import MaskFourier
from simulation_engine.simulation import Simulation

LOG = logging.getLogger("earlier")


def minmax(a):
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


def r11(ref, est):
    p = [peak_signal_noise_ratio(minmax(r), minmax(e), data_range=1.0) for r, e in zip(ref, est, strict=True)]
    s = [structural_similarity(minmax(r), minmax(e), data_range=1.0) for r, e in zip(ref, est, strict=True)]
    return float(np.mean(p)), float(np.mean(s))


def fspi(X, ratio):
    mask = MaskFourier(img_size=X.shape[1], sampling_ratio=ratio, phase_steps=4, spectrum_ordering="circular",
                       logger=LOG)
    mask.generate_masks()
    sim = Simulation(LOG)
    sim.set_dataset(ArrayDataset(X))
    sim.mask = mask
    sim.set_applicator(reconstruction_method="native")
    out = np.asarray(sim.applicator.process_dataset(), dtype=np.float64).reshape(X.shape)
    return out, len(mask.masks)


def main():
    rows = []
    for ratio, test in UC2_TESTS.items():
        a = stored("UC2_R0", test)
        X, masks = a["originals"], a["masks"]
        cc = hadamard_linear(X, masks) * len(masks) / X[0].size
        fo, m_f = fspi(X, ratio)
        f11, c11 = r11(X, fo), r11(X, cc)
        fsh, csh = shared(X, fo), shared(X, cc)
        rows.append({"M/N": ratio, "M FSPI / CC": f"{m_f} / {len(masks)}",
                     "R1.1 conv. FSPI": f11[0], "R1.1 conv. CC": c11[0], "R1.1 conv. FSPI - CC": f11[0] - c11[0],
                     "shared FSPI": fsh["psnr"], "shared CC": csh["psnr"], "shared FSPI - CC": fsh["psnr"] - csh["psnr"],
                     "shared SSIM FSPI": fsh["ssim"], "shared SSIM CC": csh["ssim"]})
        print(rows[-1], flush=True)
    md = ["# Task E — R1.1 FSPI vs Cake-Cutting under two conventions (300 UC2 test images)", "",
          "PSNR in dB. R1.1 convention: estimate and reference min-max normalised per image, data range 1.",
          "Shared: estimate on the image's scale, clipped to [0, 1], data range 1.", "",
          md_table(rows, list(rows[0]), {"R1.1 conv. FSPI - CC": "{:+.2f}", "shared FSPI - CC": "{:+.2f}",
                                         "shared SSIM FSPI": "{:.4f}", "shared SSIM CC": "{:.4f}"}), ""]
    (HERE / "earlier_reports.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
