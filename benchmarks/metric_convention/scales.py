#!/usr/bin/env python
"""Task A.2: the scale of each applicator's output, measured (ASPIR venv).

    ../../.venv/bin/python scales.py

16 UC2 test images (64 x 64, stored originals), every reconstruction method through ASPIR's public API
(Simulation.set_applicator), at about M/N = 0.16:
* Hadamard rows: UC2's stored Cake-Cutting masks (655 rows) -> native (Hadamard Linear), pseudoinverse,
  FISTA, TV (default parameters);
* Fourier: MaskFourier, sampling ratio 0.16 -> native (Fourier inverse);
* Sweep: 4 angles, bar width 5, stride 2 -> native (Sweep Linear) and pseudoinverse;
* Scatter: 655 random patterns -> native (Ghost Imaging).
For each output: DC gain = mean(output) / mean(image); least-squares fit output ~ g * image + c; and the PSNR
of the raw output under the shared function (clip to [0, 1], data range 1), i.e. what a scale-sensitive
metric sees without any normalisation. Writes scales.md.
"""

import logging

import numpy as np
import pandas as pd
from common import HERE, md_table, shared, stored

from simulation_engine._1_dataset_gen.dataset import DatasetABC
from simulation_engine._2_mask_gen.mask_fourier import MaskFourier
from simulation_engine._2_mask_gen.mask_measured import MaskMeasured
from simulation_engine._2_mask_gen.mask_scatter import MaskScatter
from simulation_engine._2_mask_gen.mask_sweep import MaskSweep
from simulation_engine.simulation import Simulation

N_IMAGES = 16
LOG = logging.getLogger("scales")


class ArrayDataset(DatasetABC):
    def __init__(self, images):
        super().__init__("uc2_test", images.shape[1], LOG)
        self.data = [np.asarray(im, dtype=np.float32) for im in images]
        self.num_images = len(self.data)

    def load_data(self, progress_callback=None):
        return True

    def to_dataframe(self):
        return pd.DataFrame([im.ravel() for im in self.data])


def run(images, mask, method):
    sim = Simulation(LOG)
    sim.set_dataset(ArrayDataset(images))
    sim.mask = mask
    if getattr(mask, "masks", None) is None:
        mask.generate_masks()
    sim.set_applicator(reconstruction_method=method)
    out = sim.applicator.process_dataset()
    return np.asarray(out, dtype=np.float64).reshape(images.shape), sim.applicator


def main():
    a = stored("UC2_R0", "Celeb - Cake Cutting 16%")
    X = a["originals"][:N_IMAGES]
    size = X.shape[1]
    cases = [
        ("Hadamard (UC2 rows)", MaskMeasured(a["masks"].astype(np.int8), logger=LOG), ["native", "pseudoinverse", "fista",
                                                                           "tv_norm"]),
        ("Fourier", MaskFourier(img_size=size, sampling_ratio=0.16, logger=LOG), ["native"]),
        ("Sweep", MaskSweep(size, [{"angle": t, "bar_width": 5, "stride": 2} for t in (0.0, 45.0, 90.0, 135.0)], LOG),
         ["native", "pseudoinverse"]),
        ("Scatter", MaskScatter(size, point_density=10.0, num_patterns=655, logger=LOG), ["native"]),
    ]
    rows = []
    for family, mask, methods in cases:
        for method in methods:
            out, app = run(X, mask, method)
            g, c = np.linalg.lstsq(np.stack([X.ravel(), np.ones(X.size)], 1), out.ravel(), rcond=None)[0]
            rows.append({"masks": family, "M": len(mask.masks), "method": method,
                         "applicator": type(app).__name__,
                         "DC gain": float(out.mean() / X.mean()), "fit gain g": float(g), "fit offset c": float(c),
                         "raw PSNR (shared)": shared(X, out)["psnr"]})
            print(rows[-1], flush=True)
    md = ["# Task A.2 — output scale of each applicator (16 UC2 test images, M/N ≈ 0.16)", "",
          "DC gain = mean(output) / mean(image); fit: output ≈ g · image + c (least squares). An output on the",
          "image's scale has DC gain ≈ 1 and c ≈ 0.", "",
          md_table(rows, ["masks", "M", "method", "applicator", "DC gain", "fit gain g", "fit offset c",
                          "raw PSNR (shared)"],
                   {"DC gain": "{:.3f}", "fit gain g": "{:.3f}", "fit offset c": "{:.3f}"}), ""]
    (HERE / "scales.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
