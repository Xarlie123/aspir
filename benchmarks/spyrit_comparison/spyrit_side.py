#!/usr/bin/env python
"""SPyRiT side of the comparison (SPyRiT venv: .venv/bin/python spyrit_side.py <part>).

Parts:
  linear      noiseless split acquisition + pseudoinverse (PinvNet without denoiser) on the UC2 test images
  train       PinvNet + Unet, one per ratio, trained on the UC2 training images with SPyRiT's train_model
  evaluate    the trained nets on the UC2 test images
  pretrained  part (c): SPyRiT's published 64x64 weights, unchanged

Conventions, following SPyRiT 3's tutorials: images in [0, 1] (SPyRiT 2.4 used [-1, 1]; tuto_05 notes the
change), HadamSplit2d with the positive / negative split, prep.Unsplit (noiseless) before the
pseudoinverse. Use Case 2 also trained on [0, 1] images, and both tools are scored by one function
(compare.py).
"""

import argparse
import json
import platform
import time
from collections import OrderedDict

import numpy as np
import torch
from spyrit_common import HERE, RATIOS, H, n_measurements, order_for_rows

DATA = HERE / "data"
MODELS = HERE / "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_split():
    with np.load(DATA / "uc2_split.npz") as z:
        return {k: z[k] for k in z.files}


def meas_op(ratio):
    from spyrit.core.meas import HadamSplit2d

    rows = np.load(DATA / "uc2_rows.npz")[f"idx_{ratio}"]
    return HadamSplit2d(H, M=n_measurements(ratio), order=order_for_rows(rows), device=DEVICE)


def pinv_net(ratio, denoi=None):
    from spyrit.core.prep import Unsplit
    from spyrit.core.recon import PinvNet

    kwargs = {} if denoi is None else {"denoi": denoi}
    return PinvNet(meas_op(ratio), Unsplit(), device=DEVICE, **kwargs)


def sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def per_image_ms(fn, x, repeats=50):
    """Median per-image latency, batch 1, after warm-up (same protocol as ASPIR's timing)."""
    one = x[:1]
    with torch.no_grad():
        for _ in range(20):
            fn(one)
        ts = []
        for i in range(repeats):
            xi = x[i % len(x):i % len(x) + 1]
            sync()
            t0 = time.perf_counter()
            fn(xi)
            sync()
            ts.append(time.perf_counter() - t0)
    return 1e3 * float(np.median(ts))


def part_linear():
    test = torch.tensor(load_split()["test"], dtype=torch.float32)[:, None].to(DEVICE)
    out, timing = {}, {}
    for ratio in RATIOS:
        net = pinv_net(ratio).eval()
        with torch.no_grad():
            rec = net(test)
        out[f"rec_{ratio}"] = rec.squeeze(1).cpu().numpy()
        timing[str(ratio)] = per_image_ms(lambda x, n=net: n(x), test)
    np.savez_compressed(DATA / "spyrit_linear.npz", **out)
    (DATA / "spyrit_linear_timing.json").write_text(json.dumps(timing, indent=2))
    print("linear done", timing)


# Training budget, matched to Use Case 2 (uc2_batch_test_celeb_3000_hadamard.batch_config).
EPOCHS, BATCH, LR, WEIGHT_DECAY = 50, 16, 1e-3, 1e-4


def loaders(split):
    def ds(name):
        x = torch.tensor(split[name], dtype=torch.float32)[:, None]
        # train_model expects (input, label) pairs; for a FullNet the target is the image itself.
        return torch.utils.data.TensorDataset(x, x)

    g = torch.Generator().manual_seed(0)
    return {"train": torch.utils.data.DataLoader(ds("train"), batch_size=BATCH, shuffle=True, generator=g),
            "val": torch.utils.data.DataLoader(ds("val"), batch_size=BATCH, shuffle=False)}


def part_train(ratios):
    from spyrit.core.nnet import Unet
    from spyrit.core.train import Weight_Decay_Loss, save_net, train_model

    split = load_split()
    MODELS.mkdir(exist_ok=True)
    log = {}
    for ratio in ratios:
        torch.manual_seed(0)
        net = pinv_net(ratio, denoi=torch.nn.Sequential(OrderedDict({"unet": Unet()})))
        criterion = Weight_Decay_Loss(torch.nn.MSELoss())
        optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        # UC2 trained at a constant learning rate; train_model needs a scheduler, so give it a no-op one.
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=EPOCHS, gamma=1.0)
        t0 = time.perf_counter()
        net, info = train_model(net, criterion, optimizer, scheduler, loaders(split), DEVICE,
                                MODELS, num_epochs=EPOCHS, disp=False)
        minutes = (time.perf_counter() - t0) / 60
        save_net(MODELS / f"spyrit_pinvnet_unet_{round(ratio * 100)}.pth", net.denoi)
        log[str(ratio)] = {"train_minutes": minutes,
                           "best_val_loss": float(min(info.get("val", [float("nan")]))),
                           "params_denoiser": sum(p.numel() for p in net.denoi.parameters())}
        print(ratio, log[str(ratio)], flush=True)
    (DATA / "spyrit_train_log.json").write_text(json.dumps(log, indent=2))


def part_evaluate():
    from spyrit.core.nnet import Unet
    from spyrit.core.train import load_net

    test = torch.tensor(load_split()["test"], dtype=torch.float32)[:, None].to(DEVICE)
    out, timing = {}, {}
    for ratio in RATIOS:
        net = pinv_net(ratio, denoi=torch.nn.Sequential(OrderedDict({"unet": Unet()})))
        load_net(MODELS / f"spyrit_pinvnet_unet_{round(ratio * 100)}.pth", net.denoi, DEVICE, False)
        net.eval()
        with torch.no_grad():
            rec = net(test)
        out[f"rec_{ratio}"] = rec.squeeze(1).cpu().numpy()
        timing[str(ratio)] = per_image_ms(lambda x, n=net: n(x), test)
    np.savez_compressed(DATA / "spyrit_learned.npz", **out)
    (DATA / "spyrit_learned_timing.json").write_text(json.dumps(timing, indent=2))
    print("evaluate done", timing)


def versions():
    import spyrit
    return {"spyrit": spyrit.__version__ if hasattr(spyrit, "__version__") else "see pip",
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "python": platform.python_version()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["linear", "train", "evaluate", "pretrained", "versions"])
    ap.add_argument("--ratios", type=float, nargs="+", default=list(RATIOS))
    a = ap.parse_args()
    if a.part == "linear":
        part_linear()
    elif a.part == "train":
        part_train(a.ratios)
    elif a.part == "evaluate":
        part_evaluate()
    elif a.part == "versions":
        print(json.dumps(versions(), indent=2))
    else:
        raise SystemExit("pretrained: see pretrained.py")
