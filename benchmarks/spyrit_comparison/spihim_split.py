#!/usr/bin/env python
"""Split (positive / negative) SPIHIM star measurements for SPyRiT's DC-Net (ASPIR venv).

SPyRiT's experimental pipeline takes the raw positive and negative measurements of each Hadamard row
(prep.UnsplitRescaleEstim), not their difference. This writes them for `starSectorD_2_0.0_01…10`:

* bins: the panchromatic sum minus the union of saturated bins over the ten repeats, as in
  scripts/experimental_spihim_figure.py;
* dark: the measurement of the all-off pattern (negative half of the constant row) is subtracted from
  every measurement, so both halves are proportional to the light;
* orientation: ASPIR's DEFAULT_ORIENTATION (two quarter turns). Rotating a Sylvester pattern can flip its
  sign, which swaps its positive and negative halves; that is applied;
* indexing: by Sylvester row (0…4095), averaged over the ten repeats.

Also writes the reference (ten-repeat-mean full linear reconstruction, min-max normalised), so both
tools are scored against the same image. Output: data/spihim_star_split.npz.
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

from simulation_engine._1_dataset_gen.spihim import (  # noqa: E402
    DEFAULT_ORIENTATION,
    HadamardPngBank,
    orient_patterns,
    read_raw_mat,
    saturated_bins,
)
from simulation_engine.experimental import hadamard_rows  # noqa: E402

ROOT = REPO / "datasets/spihim"
FILES = [ROOT / "spihim_2020-Jun-11" / f"starSectorD_2_0.0_{i:02d}_raw.mat" for i in range(1, 11)]


def main():
    bank = HadamardPngBank(ROOT / "Hadamard_64x64").bipolar()             # (4096, 64, 64) bank rows
    oriented = orient_patterns(bank, **DEFAULT_ORIENTATION)
    syl_idx, syl_sgn = hadamard_rows(oriented)                            # bank row -> Sylvester row, sign
    raws = [read_raw_mat(f) for f in FILES]
    sat = sorted(set().union(*(saturated_bins(r).tolist() for r in raws)))
    keep = np.ones(raws[0]["M"].shape[1], dtype=bool)
    keep[sat] = False

    pos = np.zeros((len(raws), 4096))
    neg = np.zeros((len(raws), 4096))
    for f, raw in enumerate(raws):
        s = raw["M"][:, keep].astype(np.float64).sum(axis=1)
        p = raw["pattern_indices"]
        is_pos = p % 2 == 1
        bank_row = np.where(is_pos, (p - 1) // 2, (p - 2) // 2)
        k = syl_idx[bank_row]
        flip = syl_sgn[bank_row] < 0                       # oriented pattern = -h_k: halves swap
        to_pos = is_pos ^ flip
        pos[f, k[to_pos]] = s[to_pos]
        neg[f, k[~to_pos]] = s[~to_pos]
        dark = neg[f, 0]                                   # constant row: its negative half is all-off
        pos[f] -= dark
        neg[f] -= dark

    pos_m, neg_m = pos.mean(axis=0), neg.mean(axis=0)
    from scipy.linalg import hadamard
    Hs = hadamard(4096).astype(np.float64)                 # Sylvester, natural order
    x = (Hs.T @ (pos_m - neg_m) / 4096).reshape(64, 64)
    ref = (x - x.min()) / (x.max() - x.min())
    np.savez_compressed(HERE / "data" / "spihim_star_split.npz", pos=pos_m, neg=neg_m,
                        pos_frames=pos, neg_frames=neg, reference=ref,
                        n_bins=int(keep.sum()), n_saturated=len(sat))
    print("bins", int(keep.sum()), "saturated", len(sat), "dark/bin-sum", float(np.mean(
        [r["M"][:, keep].astype(float).sum(axis=1).min() for r in raws])))


if __name__ == "__main__":
    main()
