#!/usr/bin/env python
"""Train the Use Case 2 U-Nets on STL-10 instead of CelebA, for the SPIHIM validation.

SPIHIM's own networks (SPyRiT) were trained on STL-10, and the CelebA U-Nets
of Use Case 2 meet a domain shift on the star targets. This script repeats the
UC2 recipe with STL-10 images so the two training sets can be compared on the
same measured data:

    python scripts/train_stl10_unets.py            # downloads ~0.4 GB once, then trains 5 nets
    python scripts/experimental_spihim_figure.py   # picks the nets up automatically

Recipe (from ``uc2_batch_test_celeb_3000_hadamard.batch_config``): 3000 images
at 64x64, INT8 data format, the exact Cake-Cutting rows of each UC2 test
(``masks.npz``), Hadamard Linear reconstruction, U-Net with features
[8, 16, 32, 64] (ASPIR's current U-Net), MSE, Adam, lr 1e-3, weight decay
1e-4, batch 16, 50 epochs, 80/10/10 split. Training runs through ASPIR's
``PostprocessorNN``, which also records the input normalisation constants.

Data: the first 3000 images of the STL-10 *train* split (Coates, Lee & Ng,
AISTATS 2011), converted to greyscale. Only ``train_X.bin`` is extracted; the
archive is streamed and the download stops once it has been read.
"""

import argparse
import hashlib
import json
import logging
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402

from simulation_engine._1_dataset_gen.DatasetFromFolder import DatasetFromFolder  # noqa: E402
from simulation_engine._2_mask_gen.mask_measured import MaskMeasured  # noqa: E402
from simulation_engine.experimental import RATIOS  # noqa: E402
from simulation_engine.simulation import Simulation  # noqa: E402

STL10_URL = "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"
MEMBER = "stl10_binary/train_X.bin"
UC2 = REPO / "experiments/batch_tests/uc2_batch_test_celeb_3000_hadamard_results"
RECIPE = {"num_images": 3000, "img_size": 64, "data_format": "INT8", "model_name": "u-net",
          "architecture_config": {"features": [8, 16, 32, 64]}, "epochs": 50, "batch_size": 16,
          "learning_rate": 1e-3, "weight_decay": 1e-4, "dropout": 0.0, "loss_function": "MSE",
          "optimizer": "Adam", "split": [0.8, 0.1, 0.1], "reconstruction": "Hadamard Linear"}

log = logging.getLogger("aspir")


def fetch_train_x(dest: Path) -> Path:
    """Stream the STL-10 archive and keep only ``train_X.bin``."""
    out = dest / "train_X.bin"
    if out.exists():
        return out
    dest.mkdir(parents=True, exist_ok=True)
    log.info("Streaming %s (stops after %s)", STL10_URL, MEMBER)
    with urllib.request.urlopen(STL10_URL) as resp, tarfile.open(fileobj=resp, mode="r|gz") as tar:
        for member in tar:
            if member.name == MEMBER:
                part = out.with_suffix(".part")
                with tar.extractfile(member) as src, open(part, "wb") as dst:
                    while chunk := src.read(1 << 20):
                        dst.write(chunk)
                part.rename(out)
                break
        else:
            raise RuntimeError(f"{MEMBER} not found in the archive")
    return out


def export_pngs(train_x: Path, folder: Path, n: int) -> str:
    """First ``n`` images as 96x96 greyscale PNGs; returns the SHA-256 of train_X.bin."""
    from PIL import Image

    digest = hashlib.sha256(train_x.read_bytes()).hexdigest()
    folder.mkdir(parents=True, exist_ok=True)
    if len(list(folder.glob("*.png"))) >= n:
        return digest
    # Stored column-major: (N, 3, 96, 96) with the spatial axes transposed.
    raw = np.fromfile(train_x, dtype=np.uint8).reshape(-1, 3, 96, 96).transpose(0, 3, 2, 1)
    for i in range(n):
        Image.fromarray(raw[i]).convert("L").save(folder / f"stl10_train_{i:05d}.png")
    return digest


def train_one(folder: Path, ratio: float, out: Path, device_ok: bool):
    sim = Simulation(log)
    dataset = DatasetFromFolder(RECIPE["img_size"], str(folder), logger=log, data_format=RECIPE["data_format"])
    dataset.load_data()
    dataset.data = dataset.data[:RECIPE["num_images"]]
    dataset.num_images = len(dataset.data)
    sim.set_dataset(dataset)
    with np.load(UC2 / "data" / f"Celeb_-_Cake_Cutting_{round(ratio * 100)}%" / "masks.npz") as z:
        masks = z["masks"]
    sim.mask = MaskMeasured(masks.astype(np.int8), logger=log)
    sim.set_applicator(reconstruction_method="native", device="cuda" if device_ok else "cpu")
    sim.set_postprocessor(
        dataset, sim.mask, sim.applicator, model_name=RECIPE["model_name"],
        model_overrides=dict(RECIPE["architecture_config"]), batch_size=RECIPE["batch_size"],
        lr=RECIPE["learning_rate"], weight_decay=RECIPE["weight_decay"], dropout=RECIPE["dropout"],
        loss_function=RECIPE["loss_function"], optimizer_name=RECIPE["optimizer"], use_gpu=device_ok,
        train_ratio=RECIPE["split"][0], val_ratio=RECIPE["split"][1], test_ratio=RECIPE["split"][2])
    pp = sim.postprocessor
    t0 = time.perf_counter()
    pp.train_with_metrics(num_epochs=RECIPE["epochs"])
    minutes = (time.perf_counter() - t0) / 60
    name = f"STL10_-_Cake_Cutting_{round(ratio * 100)}%"
    pp.save_model(str(out / f"{name}.pt"))
    orig, recon, den = (np.asarray(a, dtype=np.float64) for a in pp.test_dataset())

    def psnr(ref, est):
        """Mean PSNR, peak = each reference's range (as ASPIR's analyzer)."""
        vals = [10 * np.log10(np.ptp(x) ** 2 / max(np.mean((x - y) ** 2), 1e-20))
                for x, y in zip(ref, est, strict=True)]
        return float(np.mean(vals))

    meta = {"ratio": ratio, "M": len(masks), "norm_lo": pp.norm_lo, "norm_span": pp.norm_span,
            "image_max": float(max(np.max(img) for img in dataset.data)),
            "test_psnr_input_db": psnr(orig, recon), "test_psnr_output_db": psnr(orig, den),
            "training_minutes": minutes, "recipe": RECIPE}
    (out / f"{name}.json").write_text(json.dumps(meta, indent=2))
    log.info("%s: test PSNR %.2f -> %.2f dB (%.1f min)", name, meta["test_psnr_input_db"],
             meta["test_psnr_output_db"], minutes)
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", type=Path, default=REPO / "datasets" / "stl10")
    ap.add_argument("--out", type=Path, default=REPO / "experiments" / "experimental_spihim" / "stl10_unets")
    ap.add_argument("--ratios", type=float, nargs="+", default=list(RATIOS))
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    log.setLevel(logging.INFO)
    torch.manual_seed(0)

    train_x = fetch_train_x(args.data)
    folder = args.data / "train_gray"
    digest = export_pngs(train_x, folder, RECIPE["num_images"])
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {"stl10_train_X_sha256": digest, "source": STL10_URL, "nets": []}
    for ratio in args.ratios:
        summary["nets"].append(train_one(folder, ratio, args.out, torch.cuda.is_available()))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
