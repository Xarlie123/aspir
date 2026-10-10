#!/usr/bin/env python
"""Rebuild the Use Case 2 CelebA images and split, and check them against UC2. (ASPIR venv)

UC2 (`experiments/batch_tests/uc2_batch_test_celeb_3000_hadamard.batch_config`) used
`DatasetFromCelebrities(img_size=64, num_images=3000, seed=10, data_format="INT8")` and
`PostprocessorNN`'s split: `random_split` with `torch.Generator().manual_seed(10)` into 80/10/10. Both are
reproduced here through ASPIR's own classes, and the test split is checked image by image against the
`originals` UC2 stored, before anything is compared.

Writes `data/uc2_split.npz` (train / val / test images in [0, 1], their indices) and
`data/uc2_rows.npz` (the exact Cake-Cutting rows of each UC2 test, as Sylvester indices and signs).

    ../../.venv/bin/python uc2_data.py
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

from simulation_engine._1_dataset_gen.DatasetFromCelebrities import DatasetFromCelebrities  # noqa: E402
from simulation_engine.experimental import RATIOS, hadamard_rows  # noqa: E402

UC2 = REPO / "experiments/batch_tests/uc2_batch_test_celeb_3000_hadamard_results"
CONFIG = REPO / "experiments/batch_tests/uc2_batch_test_celeb_3000_hadamard.batch_config"


def uc2_test_dir(ratio):
    return UC2 / "data" / f"Celeb_-_Cake_Cutting_{round(ratio * 100)}%"


def quantize_int8(arr):
    """PostprocessorNN._apply_data_format_to_arrays, INT8 branch (global min / max)."""
    lo, hi = arr.min(), arr.max()
    if hi - lo > 0:
        q = np.round((arr - lo) / (hi - lo) * 255) / 255
        return (q * (hi - lo) + lo).astype(np.float32)
    return arr.astype(np.float32)


def main():
    logging.basicConfig(level=logging.WARNING)
    info = json.loads(CONFIG.read_text())["dataset_info"]
    test0 = json.loads(CONFIG.read_text())["tests"][0]
    ds = DatasetFromCelebrities(name=info["name"], img_size=info["img_size"], num_images=info["num_images"],
                                seed=info["seed"], data_format=info["data_format"],
                                download_dir=str(REPO / "datasets/tmp_downloaded_datasets/celebrities"))
    ds.load_data()
    clean = quantize_int8(np.stack(ds.data).astype(np.float32))
    n = len(clean)
    n_train = int(test0["train_split"] / 100 * n)
    n_val = int(test0["val_split"] / 100 * n)
    n_test = n - n_train - n_val
    gen = torch.Generator().manual_seed(info["seed"])
    perm = torch.randperm(n, generator=gen).tolist()        # what random_split draws
    tr, va, te = perm[:n_train], perm[n_train:n_train + n_val], perm[n_train + n_val:]
    # The same split through random_split itself, to be sure.
    sub = torch.utils.data.random_split(range(n), [n_train, n_val, n_test],
                                        generator=torch.Generator().manual_seed(info["seed"]))
    assert list(sub[2].indices) == te and list(sub[0].indices) == tr

    report = {"n_images": n, "n_train": n_train, "n_val": n_val, "n_test": n_test, "seed": info["seed"],
              "data_format": info["data_format"], "per_ratio": {}}
    for ratio in RATIOS:
        with np.load(uc2_test_dir(ratio) / "test_images.npz") as z:
            stored = z["originals"]
        diff = float(np.abs(stored - clean[te]).max())
        report["per_ratio"][str(ratio)] = {"max_abs_diff_test_originals": diff}
        assert diff < 1e-6, f"test split differs from UC2 at {ratio}: {diff}"
    rows = {}
    for ratio in RATIOS:
        with np.load(uc2_test_dir(ratio) / "masks.npz") as z:
            idx, sgn = hadamard_rows(z["masks"])
        rows[f"idx_{ratio}"], rows[f"sgn_{ratio}"] = idx, sgn
        assert np.all(sgn == 1), "UC2 Cake-Cutting rows are positive Sylvester rows"
    out = HERE / "data"
    out.mkdir(exist_ok=True)
    np.savez_compressed(out / "uc2_split.npz", train=clean[tr], val=clean[va], test=clean[te],
                        train_idx=np.array(tr), val_idx=np.array(va), test_idx=np.array(te))
    np.savez_compressed(out / "uc2_rows.npz", **rows)
    (out / "uc2_split.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
