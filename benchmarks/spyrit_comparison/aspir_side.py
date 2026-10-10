#!/usr/bin/env python
"""ASPIR side of the comparison (ASPIR venv: ../../.venv/bin/python aspir_side.py <part>).

Parts:
  linear   noiseless acquisition + Hadamard Linear and pseudoinverse on the UC2 test images,
           with the exact Cake-Cutting rows of UC2
  learned  the UC2 U-Net checkpoints (not retrained) on the UC2 test images

ASPIR is driven through its public API: a dataset, MaskMeasured with the UC2 patterns, and the applicators
built by Simulation.set_applicator; the UC2 networks are loaded with UNetTranspose (their decoder predates
2026-05-16).
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

from simulation_engine._1_dataset_gen.dataset import DatasetABC  # noqa: E402
from simulation_engine._2_mask_gen.mask_measured import MaskMeasured  # noqa: E402
from simulation_engine._4_postprocessor.models.unet_transpose import UNetTranspose  # noqa: E402
from simulation_engine.experimental import RATIOS  # noqa: E402
from simulation_engine.simulation import Simulation  # noqa: E402

DATA = HERE / "data"
UC2 = REPO / "experiments/batch_tests/uc2_batch_test_celeb_3000_hadamard_results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
log = logging.getLogger("aspir")


class ArrayDataset(DatasetABC):
    """The UC2 test images, already in memory."""

    def __init__(self, images):
        super().__init__("uc2_test", images.shape[1], log)
        self.data = [np.asarray(im, dtype=np.float32) for im in images]
        self.num_images = len(self.data)

    def load_data(self, progress_callback=None):
        return True

    def to_dataframe(self):
        return pd.DataFrame([im.ravel() for im in self.data])


def uc2_dir(ratio):
    return UC2 / "data" / f"Celeb_-_Cake_Cutting_{round(ratio * 100)}%"


def applicator(images, ratio, method):
    with np.load(uc2_dir(ratio) / "masks.npz") as z:
        masks = z["masks"]
    sim = Simulation(log)
    sim.set_dataset(ArrayDataset(images))
    sim.mask = MaskMeasured(masks.astype(np.int8), logger=log)
    sim.set_applicator(reconstruction_method=method, device=DEVICE)
    return sim.applicator


def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def per_image_ms(fn, n, repeats=50):
    for _ in range(20):
        fn(0)
    ts = []
    for i in range(repeats):
        sync()
        t0 = time.perf_counter()
        fn(i % n)
        sync()
        ts.append(time.perf_counter() - t0)
    return 1e3 * float(np.median(ts))


def part_linear():
    with np.load(DATA / "uc2_split.npz") as z:
        test = z["test"]
    out, timing = {}, {}
    for ratio in RATIOS:
        for method, key in (("native", "native"), ("pseudoinverse", "pinv")):
            app = applicator(test, ratio, method)
            rec = app.process_dataset().to_numpy().reshape(test.shape)
            out[f"{key}_{ratio}"] = rec.astype(np.float32)
            timing[f"{key}_{ratio}"] = per_image_ms(lambda i, a=app: a.apply_mask_range(0, None, i), len(test))
    np.savez_compressed(DATA / "aspir_linear.npz", **out)
    (DATA / "aspir_linear_timing.json").write_text(json.dumps(timing, indent=2))
    print("linear done", {k: round(v, 3) for k, v in timing.items()})


def part_learned():
    """UC2 checkpoints on the UC2 test inputs, as UC2 evaluated them.

    UC2 fed the network Hadamard Linear reconstructions normalised by a global min-max over the dataset and
    quantised to 8 bits (INT8 data format); those exact inputs are stored with the test images. The outputs
    are recomputed here and checked against the stored ones.
    """
    out, timing, check = {}, {}, {}
    for ratio in RATIOS:
        net = UNetTranspose(features=[8, 16, 32, 64])
        net.load_state_dict(torch.load(UC2 / "models" / f"Celeb_-_Cake_Cutting_{round(ratio * 100)}%.pt",
                                       map_location="cpu"))
        net.to(DEVICE).eval()
        with np.load(uc2_dir(ratio) / "test_images.npz") as z:
            inputs, stored = z["reconstructions"], z["denoised"]
        x = torch.tensor(inputs, dtype=torch.float32)[:, None].to(DEVICE)
        with torch.no_grad():
            y = net(x)[:, 0].cpu().numpy()
        check[str(ratio)] = float(np.abs(y - stored).max())
        out[f"rec_{ratio}"] = y
        out[f"input_{ratio}"] = inputs
        with torch.no_grad():
            timing[f"unet_{ratio}"] = per_image_ms(lambda i, n=net, x=x: n(x[i:i + 1]), len(x))
    np.savez_compressed(DATA / "aspir_learned.npz", **out)
    (DATA / "aspir_learned_timing.json").write_text(json.dumps({"timing_ms": timing,
                                                                 "max_abs_diff_vs_stored": check}, indent=2))
    print("learned done", check, {k: round(v, 3) for k, v in timing.items()})


def part_retrain(keep_best_val=False):
    """Supplementary: retrain the UC2 recipe with today's ASPIR on this GPU, for training time.

    Same images, split, rows and hyper-parameters as UC2 (batch config), through Simulation /
    PostprocessorNN as Batch Test does; the U-Net is ASPIR's current one (Upsample decoder).
    With keep_best_val (part `retrain_keepbest`), ASPIR's own *Keep best-validation weights* option
    selects the weights, as SPyRiT's train_model does.
    """
    from simulation_engine._1_dataset_gen.DatasetFromCelebrities import DatasetFromCelebrities

    cfg = json.loads((REPO / "experiments/batch_tests/uc2_batch_test_celeb_3000_hadamard.batch_config").read_text())
    info, t0cfg = cfg["dataset_info"], cfg["tests"][0]
    ds = DatasetFromCelebrities(name=info["name"], img_size=info["img_size"], num_images=info["num_images"],
                                seed=info["seed"], data_format=info["data_format"],
                                download_dir=str(REPO / "datasets/tmp_downloaded_datasets/celebrities"))
    ds.load_data()
    with np.load(DATA / "uc2_split.npz") as z:
        test_ref = z["test"]
    out, log_ = {}, {}
    for ratio in RATIOS:
        with np.load(uc2_dir(ratio) / "masks.npz") as z:
            masks = z["masks"]
        sim = Simulation(log)
        sim.set_dataset(ds)
        sim.mask = MaskMeasured(masks.astype(np.int8), logger=log)
        sim.set_applicator(reconstruction_method="native", device=DEVICE)
        torch.manual_seed(0)
        sim.set_postprocessor(ds, sim.mask, sim.applicator, model_name=t0cfg["model_name"],
                              model_overrides=dict(t0cfg["architecture_config"]),
                              batch_size=t0cfg["batch_size"], lr=t0cfg["learning_rate"],
                              weight_decay=t0cfg["weight_decay"], dropout=t0cfg["dropout"],
                              loss_function=t0cfg["loss_function"], optimizer_name=t0cfg["optimizer"],
                              use_gpu=DEVICE == "cuda", train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
        pp = sim.postprocessor
        sync()
        t0 = time.perf_counter()
        pp.train_with_metrics(num_epochs=t0cfg["epochs"], keep_best_val=keep_best_val)
        sync()
        minutes = (time.perf_counter() - t0) / 60
        orig, _, den = (np.asarray(a, dtype=np.float32) for a in pp.test_dataset())
        assert np.abs(orig - test_ref).max() < 1e-6, "test split differs"
        out[f"rec_{ratio}"] = den
        log_[str(ratio)] = {"train_minutes": minutes,
                            "params": sum(p.numel() for p in pp.model.parameters()),
                            "kept_epoch": pp.kept_epoch, "kept_weights": pp.kept_weights}
        print(ratio, log_[str(ratio)], flush=True)
    suffix = "_keepbest" if keep_best_val else ""
    np.savez_compressed(DATA / f"aspir_retrained{suffix}.npz", **out)
    (DATA / f"aspir_retrain{suffix}_log.json").write_text(json.dumps(log_, indent=2))


def part_retrain_bestval():
    """Diagnostic: the same retraining, keeping both the last-epoch and the best-validation weights.

    SPyRiT's train_model returns the weights of the epoch with the lowest validation loss; ASPIR kept the
    last epoch. This trains epoch by epoch (PostprocessorNN.train(1), same optimiser throughout), validates
    after each epoch with PostprocessorNN.validate() and records both, so the effect of that one difference
    is measured within a single run. It predates ASPIR's own option (part `retrain_keepbest`).
    """
    import copy

    from simulation_engine._1_dataset_gen.DatasetFromCelebrities import DatasetFromCelebrities

    cfg = json.loads((REPO / "experiments/batch_tests/uc2_batch_test_celeb_3000_hadamard.batch_config").read_text())
    info, t0cfg = cfg["dataset_info"], cfg["tests"][0]
    ds = DatasetFromCelebrities(name=info["name"], img_size=info["img_size"], num_images=info["num_images"],
                                seed=info["seed"], data_format=info["data_format"],
                                download_dir=str(REPO / "datasets/tmp_downloaded_datasets/celebrities"))
    ds.load_data()
    out, log_ = {}, {}
    for ratio in RATIOS:
        with np.load(uc2_dir(ratio) / "masks.npz") as z:
            masks = z["masks"]
        sim = Simulation(log)
        sim.set_dataset(ds)
        sim.mask = MaskMeasured(masks.astype(np.int8), logger=log)
        sim.set_applicator(reconstruction_method="native", device=DEVICE)
        torch.manual_seed(0)
        sim.set_postprocessor(ds, sim.mask, sim.applicator, model_name=t0cfg["model_name"],
                              model_overrides=dict(t0cfg["architecture_config"]),
                              batch_size=t0cfg["batch_size"], lr=t0cfg["learning_rate"],
                              weight_decay=t0cfg["weight_decay"], dropout=t0cfg["dropout"],
                              loss_function=t0cfg["loss_function"], optimizer_name=t0cfg["optimizer"],
                              use_gpu=DEVICE == "cuda", train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
        pp = sim.postprocessor
        best, best_epoch, val_curve = None, -1, []
        for epoch in range(1, t0cfg["epochs"] + 1):
            pp.train(num_epochs=1)
            v = float(pp.validate())
            val_curve.append(v)
            if best is None or v < min(val_curve[:-1] or [np.inf]):
                best, best_epoch = copy.deepcopy(pp.model.state_dict()), epoch
        _, _, last = (np.asarray(a, dtype=np.float32) for a in pp.test_dataset())
        pp.model.load_state_dict(best)
        _, _, best_out = (np.asarray(a, dtype=np.float32) for a in pp.test_dataset())
        out[f"last_{ratio}"], out[f"best_{ratio}"] = last, best_out
        log_[str(ratio)] = {"best_epoch": best_epoch, "val_last": val_curve[-1], "val_best": min(val_curve),
                            "val_curve": val_curve}
        print(ratio, {k: v for k, v in log_[str(ratio)].items() if k != "val_curve"}, flush=True)
    np.savez_compressed(DATA / "aspir_retrained_bestval.npz", **out)
    (DATA / "aspir_retrain_bestval_log.json").write_text(json.dumps(log_, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["linear", "learned", "retrain", "retrain_keepbest", "retrain_bestval"])
    a = ap.parse_args()
    {"linear": part_linear, "learned": part_learned, "retrain": part_retrain,
     "retrain_keepbest": lambda: part_retrain(keep_best_val=True),
     "retrain_bestval": part_retrain_bestval}[a.part]()
