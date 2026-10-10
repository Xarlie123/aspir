#!/usr/bin/env python
"""Score part (c) — SPyRiT's pretrained DC-Net, for context (ASPIR venv).

    ../../.venv/bin/python context.py

(i)  CelebA, UC2 test images: metrics from compare.score(); ASPIR / SPyRiT parity nets at M/N = 0.24
     (M = 983, Cake-Cutting, noiseless) are listed next to it for orientation only.
(ii) SPIHIM star: the protocol of scripts/experimental_spihim_figure.py (affine fit to the ten-repeat-mean
     reference, PSNR / SSIM, spoke-contrast resolution limit). ASPIR's linear reconstruction of the very
     rows SPyRiT measured (M = 1024, variance order) is computed here; ASPIR's U-Nets at M/N = 0.24 come
     from experiments/experimental_spihim/metrics.csv.
"""

import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import experimental_spihim_figure as spihim_fig  # noqa: E402
from compare import score  # noqa: E402

DATA = HERE / "data"


def row(name, m, cond, s, extra=""):
    return {"object": name, "method": m, "condition": cond,
            "psnr": float(np.mean(s["psnr"])), "ssim": float(np.mean(s["ssim"])),
            "lpips": float(np.mean(s["lpips"])) if "lpips" in s else "", "note": extra}


def main():
    rows = []
    with np.load(DATA / "uc2_split.npz") as z:
        gt = z["test"]
    pc = np.load(DATA / "pretrained_celeba.npz")
    cond = "M=1024 variance order, Poisson alpha=100"
    for k, label in (("pinv", "SPyRiT pinv"), ("dc_completion", "SPyRiT DC-Net completion (no denoiser)"),
                     ("dcnet_unet", "SPyRiT DC-Net + pretrained U-Net (STL-10)")):
        rows.append(row("CelebA UC2 test (300)", label, cond, score(gt, pc[k])))
    if (DATA / "spyrit_learned.npz").exists():
        al, sl = np.load(DATA / "aspir_learned.npz"), np.load(DATA / "spyrit_learned.npz")
        c2 = "M=983 Cake-Cutting, noiseless (parity setting)"
        rows.append(row("CelebA UC2 test (300)", "ASPIR UC2 U-Net (CelebA)", c2, score(gt, al["rec_0.24"])))
        rows.append(row("CelebA UC2 test (300)", "SPyRiT PinvNet + Unet (CelebA, parity)", c2,
                        score(gt, sl["rec_0.24"])))

    # (ii) SPIHIM star.
    sp = np.load(DATA / "spihim_star_split.npz")
    ref = sp["reference"]
    star = spihim_fig.Star(ref)
    ps = np.load(DATA / "pretrained_spihim.npz")
    from scipy.linalg import hadamard

    from simulation_engine import experimental as ex
    from simulation_engine._1_dataset_gen.measured import MeasuredData

    k = ps["sylvester_rows"]
    Hs = hadamard(4096).astype(np.float64)
    data = MeasuredData(y=(sp["pos"] - sp["neg"])[k], patterns=Hs[k].reshape(-1, 64, 64).astype(np.int8),
                        img_size=64)
    aspir_lin = ex.reconstruct_measured(ex.scale_to_unit_peak(data), "pseudoinverse", "cpu")[0]

    def star_row(label, img, cond_, note=""):
        p, s = spihim_fig.psnr_ssim(img, ref)
        r, f = star.limit(spihim_fig.fit_to(img, ref))
        return {"object": "SPIHIM star (10-repeat mean)", "method": label, "condition": cond_,
                "psnr": p, "ssim": s, "lpips": "", "note": note + f" res. limit {f:.3f} cyc/px (r={r:.1f} px)"}

    c3 = "M=1024 variance order (SPyRiT rows), measured"
    rows.append(star_row("ASPIR linear (pseudoinverse) on the same rows", aspir_lin, c3))
    for key, label in (("pinv", "SPyRiT pinv"), ("dc_completion", "SPyRiT DC-Net completion"),
                       ("dcnet_unet", "SPyRiT DC-Net + pretrained U-Net (STL-10)")):
        rows.append(star_row(label, ps[key], c3))
    met = REPO / "experiments/experimental_spihim/metrics.csv"
    if met.exists():
        with open(met) as fh:
            ref_rows = list(csv.DictReader(fh))
        for r_ in ref_rows:
            if r_["object"] == "star" and r_["source"] == "experiment" and r_["ratio"] == "0.24" \
                    and r_["method"] in ("linear", "unet_b", "stl_b"):
                label = {"linear": "ASPIR linear", "unet_b": "ASPIR U-Net CelebA (b)",
                         "stl_b": "ASPIR U-Net STL-10 (b)"}[r_["method"]]
                rows.append({"object": "SPIHIM star (10-repeat mean)", "method": label,
                             "condition": "M=983 Cake-Cutting, measured (R1.3b)",
                             "psnr": float(r_["psnr"]), "ssim": float(r_["ssim"]), "lpips": "",
                             "note": f"res. limit {float(r_['res_cyc_per_px']):.3f} cyc/px "
                                     f"(r={float(r_['res_radius_px']):.1f} px)"})

    cols = list(rows[0])
    with open(HERE / "context_pretrained.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    md = ["Context only: ordering, training data and noise differ between rows marked with different "
          "conditions.", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |"
           for r in rows]
    (HERE / "context_pretrained.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [("Reference", ref), ("ASPIR linear\n(same 1024 rows)", aspir_lin), ("SPyRiT pinv", ps["pinv"]),
              ("SPyRiT DC-Net\n+ U-Net (STL-10)", ps["dcnet_unet"])]
    fig, axes = plt.subplots(1, len(panels), figsize=(2.0 * len(panels), 2.4))
    for ax, (t, im) in zip(axes, panels, strict=True):
        ax.imshow(np.clip(spihim_fig.fit_to(im, ref), 0, 1), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(t, fontsize=7)
        ax.axis("off")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"context_spihim.{ext}", dpi=200)


if __name__ == "__main__":
    main()
