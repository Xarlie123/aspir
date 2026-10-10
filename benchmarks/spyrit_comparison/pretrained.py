#!/usr/bin/env python
"""Part (c): SPyRiT's published DC-Net weights, unchanged, for context (SPyRiT venv).

Weights: `tuto6_dc-net_unet_stl10_N0_100_N_64_M_1024_...` (Creatis warehouse, id 67221559f03a54733161e960),
a U-Net denoiser trained on STL-10 for DC-Net at 64x64, M = 1024 (variance order from the ImageNet covariance
Cov_64x64.pt), Poisson noise alpha = 100. It is used exactly as tutorial/tuto_05_dcnet.py does:
UnsplitRescale(alpha) or UnsplitRescaleEstim, DCNet(meas_op, prep, Cov, denoiser) with the Rerange wrapper
for SPyRiT 2.4 weights.

(i)  UC2 CelebA test images, at the condition the weights were trained for (M = 1024, variance order,
     Poisson alpha = 100): pinv, DC-Net completion (no denoiser) and DC-Net + U-Net.
(ii) SPIHIM star (ten-repeat mean of the split measurements, spihim_split.py), M = 1024 rows picked from
     the full acquisition in SPyRiT's variance order, gain estimated from the data
     (UnsplitRescaleEstim), as SPyRiT does for experimental measurements.

Different ordering, training data and noise from Use Case 2: context only, not parity.
"""

from collections import OrderedDict

import numpy as np
import torch
from spyrit_common import HERE, H, sylvester

DATA = HERE / "data"
WEIGHTS = next((HERE / "models" / "pretrained").glob("tuto6_dc-net_unet_stl10_*M_1024*.pth"))
COV = HERE / "stat" / "Cov_64x64.pt"
M, ALPHA = 1024, 100.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def operators(noise=True):
    from spyrit.core.meas import HadamSplit2d
    from spyrit.core.noise import Poisson
    from spyrit.core.torch import Cov2Var

    cov = torch.load(COV, weights_only=True)
    kwargs = {"noise_model": Poisson(ALPHA)} if noise else {}
    meas = HadamSplit2d(H, M=M, order=Cov2Var(cov), device=DEVICE, **kwargs)
    return meas, cov


def denoiser():
    from spyrit.core.nnet import Unet
    from spyrit.core.prep import Rerange
    from spyrit.core.train import load_net

    rerange = Rerange((0, 1), (-1, 1))
    d = torch.nn.Sequential(OrderedDict({"rerange": rerange, "denoi": Unet(),
                                         "rerange_inv": rerange.inverse()}))
    load_net(WEIGHTS, d, DEVICE, False)
    return d.eval()


def celeba():
    from spyrit.core.prep import UnsplitRescale
    from spyrit.core.recon import DCNet, PinvNet

    torch.manual_seed(0)
    with np.load(DATA / "uc2_split.npz") as z:
        x = torch.tensor(z["test"], dtype=torch.float32)[:, None].to(DEVICE)
    meas, cov = operators(noise=True)
    prep = UnsplitRescale(ALPHA)
    y = meas(x)                                            # one Poisson draw, shared by all methods
    out = {}
    with torch.no_grad():
        out["pinv"] = PinvNet(meas, prep, device=DEVICE).reconstruct(y)
        out["dc_completion"] = DCNet(meas, prep, cov / 4, device=DEVICE).reconstruct(y)
        out["dcnet_unet"] = DCNet(meas, prep, cov, denoiser(), device=DEVICE).reconstruct(y)
    np.savez_compressed(DATA / "pretrained_celeba.npz",
                        **{k: v.squeeze(1).cpu().numpy() for k, v in out.items()})
    print("celeba done")


def walsh_to_sylvester():
    """Sylvester row index of each SPyRiT flat index i * 64 + j."""
    from spyrit.core.torch import walsh_matrix

    W, S = walsh_matrix(H).numpy(), sylvester(H)
    q = np.array([int(np.flatnonzero(np.all(W[i] == S, axis=1))[0]) for i in range(H)])   # Walsh i -> Sylvester a
    i, j = np.divmod(np.arange(H * H), H)
    return q[i] * H + q[j]


def spihim():
    from spyrit.core.prep import UnsplitRescaleEstim
    from spyrit.core.recon import DCNet, PinvNet

    d = np.load(DATA / "spihim_star_split.npz")
    meas, cov = operators(noise=False)
    k = walsh_to_sylvester()[meas.indices.cpu().numpy()[:M]]        # Sylvester rows SPyRiT measures, in order
    y = np.empty(2 * M)
    y[0::2], y[1::2] = d["pos"][k], d["neg"][k]
    y = torch.tensor(y, dtype=torch.float32, device=DEVICE)[None, None]
    prep = UnsplitRescaleEstim(meas)
    out = {}
    with torch.no_grad():
        out["pinv"] = PinvNet(meas, prep, device=DEVICE).reconstruct(y)
        out["dc_completion"] = DCNet(meas, prep, cov / 4, device=DEVICE).reconstruct(y)
        out["dcnet_unet"] = DCNet(meas, prep, cov, denoiser(), device=DEVICE).reconstruct(y)
    np.savez_compressed(DATA / "pretrained_spihim.npz",
                        **{k_: v.squeeze().cpu().numpy() for k_, v in out.items()},
                        sylvester_rows=k)
    print("spihim done; estimated gain", float(prep.alpha.flatten()[0]) if hasattr(prep, "alpha") else None)


if __name__ == "__main__":
    celeba()
    spihim()
