# Use Case 2 in SPyRiT 3.1, end to end, following the tutorials (tuto_01_c, tuto_03, tuto_04_b, tuto_05).
# CelebA 64x64, Cake-Cutting Hadamard, five ratios, PinvNet + Unet, PSNR / SSIM / timing per ratio.
# Run with the SPyRiT venv:  .venv/bin/python workflow_spyrit.py <celeba_folder> [--epochs 50]
import argparse
import time
from collections import OrderedDict

import numpy as np
import torch
import torchvision
from spyrit.core.meas import HadamSplit2d
from spyrit.core.nnet import Unet
from spyrit.core.prep import Unsplit
from spyrit.core.recon import PinvNet
from spyrit.core.train import Weight_Decay_Loss, save_net, train_model
from spyrit.misc.metrics import batch_psnr_, batch_ssim
from spyrit.misc.sampling import define_order
from spyrit.misc.statistics import transform_gray_norm
from torch.utils.data import DataLoader, Subset, TensorDataset, random_split

ap = argparse.ArgumentParser()
ap.add_argument("celeba")
ap.add_argument("--epochs", type=int, default=50)
ap.add_argument("--images", type=int, default=3000)
args = ap.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset: grayscale 64x64 images in [0, 1]; 3000 of them, split 80 / 10 / 10.
dataset = torchvision.datasets.ImageFolder(args.celeba, transform=transform_gray_norm(64, normalize=False))
g = torch.Generator().manual_seed(10)
dataset = Subset(dataset, torch.randperm(len(dataset), generator=g)[: args.images].tolist())
images = torch.stack([x for x, _ in dataset])
train, val, test = random_split(TensorDataset(images, images), [0.8, 0.1, 0.1], generator=g)
loaders = {"train": DataLoader(train, batch_size=16, shuffle=True), "val": DataLoader(val, batch_size=16)}
x_test = torch.stack([x for x, _ in test]).to(device)

order = define_order(64, "CC", pdf=True)            # Cake-Cutting; scores, higher = measured first
results = {}
for ratio in (0.04, 0.08, 0.16, 0.24, 0.32):
    meas_op = HadamSplit2d(64, M=int(ratio * 64 * 64), order=order, device=device)   # noiseless
    net = PinvNet(meas_op, Unsplit(), denoi=torch.nn.Sequential(OrderedDict({"unet": Unet()})), device=device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs, gamma=1.0)
    net, _ = train_model(net, Weight_Decay_Loss(torch.nn.MSELoss()), optimizer, scheduler, loaders, device,
                         ".", num_epochs=args.epochs)
    save_net(f"unet_cc_{round(ratio * 100)}.pth", net.denoi)
    net.eval()
    with torch.no_grad():
        x_rec = net(x_test)
        x_lin = net.reconstruct_pinv(net.acquire(x_test))
        for _ in range(20):                             # warm-up, then per-image latency
            net(x_test[:1])
        torch.cuda.synchronize() if device.type == "cuda" else None
        t0 = time.perf_counter()
        for i in range(100):
            net(x_test[i:i + 1])
        torch.cuda.synchronize() if device.type == "cuda" else None
        ms = (time.perf_counter() - t0) * 10
    results[ratio] = (np.mean(batch_psnr_(x_test, x_lin, r=1)), np.mean(batch_psnr_(x_test, x_rec, r=1)),
                      np.mean(batch_ssim(x_test, x_rec)), ms)

print("M/N | PSNR pinv | PSNR pinv+Unet | SSIM pinv+Unet | ms/image")
for ratio, (p_lin, p_rec, s_rec, ms) in results.items():
    print(f"{ratio:.2f} | {p_lin:.2f} | {p_rec:.2f} | {s_rec:.3f} | {ms:.2f}")
