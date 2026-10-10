#!/usr/bin/env python
"""Score both tools with one function and write the parity tables and figure (ASPIR venv, which has lpips).

    ../../.venv/bin/python compare.py

Inputs (written by aspir_side.py / spyrit_side.py): data/{aspir,spyrit}_{linear,learned}.npz and their
timing JSON files. Every output is scored against the same 300 UC2 test images by `score()`:

* images clipped to [0, 1] (both tools model [0, 1] intensities);
* PSNR with data_range = 1, SSIM (scikit-image defaults, data_range = 1), LPIPS (AlexNet, as ASPIR's
  analyzer), per image, then mean and standard deviation over the 300 images.
"""

import json
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RATIOS = (0.04, 0.08, 0.16, 0.24, 0.32)
N = 4096

_LPIPS = None


def _lpips(ref, est):
    global _LPIPS
    import lpips

    if _LPIPS is None:
        _LPIPS = lpips.LPIPS(net="alex", verbose=False).eval()

    def t(x):
        x = torch.tensor(x, dtype=torch.float32)[:, None].repeat(1, 3, 1, 1)
        return 2 * x - 1                                   # LPIPS expects [-1, 1]
    with torch.no_grad():
        return _LPIPS(t(ref), t(est)).flatten().numpy()


def score(ref, est):
    """Per-image PSNR / SSIM / LPIPS of `est` against `ref`, both (n, 64, 64)."""
    est = np.clip(np.asarray(est, dtype=np.float64), 0.0, 1.0)
    ref = np.asarray(ref, dtype=np.float64)
    psnr = np.array([peak_signal_noise_ratio(r, e, data_range=1.0) for r, e in zip(ref, est, strict=True)])
    ssim = np.array([structural_similarity(r, e, data_range=1.0) for r, e in zip(ref, est, strict=True)])
    return {"psnr": psnr, "ssim": ssim, "lpips": _lpips(ref, est)}


def summary(m):
    return {f"{k}_mean": float(v.mean()) for k, v in m.items()} | \
           {f"{k}_std": float(v.std(ddof=1)) for k, v in m.items()}


def write_table(stem, rows, cols):
    import csv
    with open(DATA.parent / f"{stem}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(f"{r[c]:.4g}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
    (DATA.parent / f"{stem}.md").write_text("\n".join(lines) + "\n")


styles_learned = ("ASPIR (UC2 U-Net)", "SPyRiT (PinvNet + Unet)", "ASPIR retrained, best-validation epoch")


def main():
    with np.load(DATA / "uc2_split.npz") as z:
        gt = z["test"]
    al, sl = np.load(DATA / "aspir_linear.npz"), np.load(DATA / "spyrit_linear.npz")
    at = json.loads((DATA / "aspir_linear_timing.json").read_text())
    st = json.loads((DATA / "spyrit_linear_timing.json").read_text())

    lin_rows, curves = [], {}
    for r in RATIOS:
        m = int(np.floor(r * N + 1e-9))
        a_pinv, a_nat, s_pinv = al[f"pinv_{r}"], al[f"native_{r}"], sl[f"rec_{r}"]
        diff = np.abs(s_pinv - a_pinv)
        sa, ss = summary(score(gt, a_pinv)), summary(score(gt, s_pinv))
        lin_rows.append({
            "M/N": r, "M": m,
            "max_abs_diff_pinv": float(diff.max()),
            "psnr_between_tools_db": float(10 * np.log10(1 / np.mean(diff ** 2))),
            "max_abs_diff_native_x_M_over_N": float(np.abs(s_pinv - a_nat * m / N).max()),
            "aspir_psnr": sa["psnr_mean"], "spyrit_psnr": ss["psnr_mean"],
            "aspir_ssim": sa["ssim_mean"], "spyrit_ssim": ss["ssim_mean"],
            "aspir_ms_per_image": at[f"pinv_{r}"], "spyrit_ms_per_image": st[str(r)],
        })
        curves.setdefault("ASPIR linear", []).append((r, sa["psnr_mean"], sa["psnr_std"]))
        curves.setdefault("SPyRiT pinv", []).append((r, ss["psnr_mean"], ss["psnr_std"]))
    write_table("parity_linear", lin_rows, list(lin_rows[0]))

    if (DATA / "spyrit_learned.npz").exists():
        alr, slr = np.load(DATA / "aspir_learned.npz"), np.load(DATA / "spyrit_learned.npz")
        alt = json.loads((DATA / "aspir_learned_timing.json").read_text())["timing_ms"]
        slt = json.loads((DATA / "spyrit_learned_timing.json").read_text())
        train = json.loads((DATA / "spyrit_train_log.json").read_text())
        retrained = DATA / "aspir_retrained.npz"
        rt = np.load(retrained) if retrained.exists() else None
        rt_log = json.loads((DATA / "aspir_retrain_log.json").read_text()) if rt is not None else {}
        bv_path = DATA / "aspir_retrained_bestval.npz"
        bv = np.load(bv_path) if bv_path.exists() else None
        bv_log = json.loads((DATA / "aspir_retrain_bestval_log.json").read_text()) if bv is not None else {}
        rows = []
        for r in RATIOS:
            entries = [("ASPIR (UC2 U-Net)", alr[f"rec_{r}"], at[f"native_{r}"] + alt[f"unet_{r}"], None),
                       ("SPyRiT (PinvNet + Unet)", slr[f"rec_{r}"], slt[str(r)], train[str(r)]["train_minutes"])]
            if rt is not None:
                entries.append(("ASPIR retrained today, run 1, last epoch", rt[f"rec_{r}"], "",
                                rt_log[str(r)]["train_minutes"]))
            if bv is not None:
                entries.append(("ASPIR retrained today, run 2, last epoch", bv[f"last_{r}"], "", ""))
                entries.append((f"ASPIR retrained today, run 2, best-validation epoch "
                                f"({bv_log[str(r)]['best_epoch']})", bv[f"best_{r}"], "", ""))
            for tool, est, ms, tmin in entries:
                s = summary(score(gt, est))
                rows.append({"M/N": r, "tool": tool,
                             "psnr": s["psnr_mean"], "psnr_std": s["psnr_std"],
                             "ssim": s["ssim_mean"], "ssim_std": s["ssim_std"],
                             "lpips": s["lpips_mean"], "lpips_std": s["lpips_std"],
                             "ms_per_image": ms, "train_minutes": "" if tmin is None else tmin})
                key = ("ASPIR retrained, best-validation epoch" if tool.startswith(
                    "ASPIR retrained today, run 2, best-validation") else tool)
                if key in styles_learned:
                    curves.setdefault(key, []).append((r, s["psnr_mean"], s["psnr_std"]))
        write_table("parity_learned", rows, list(rows[0]))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    styles = {"ASPIR linear": ("C0", "-", "o"), "SPyRiT pinv": ("C1", "--", "s"),
              "ASPIR (UC2 U-Net)": ("C0", "-", "^"), "SPyRiT (PinvNet + Unet)": ("C1", "--", "v"),
              "ASPIR retrained, best-validation epoch": ("C2", ":", "D")}
    for name, pts in curves.items():
        x, y, e = map(np.array, zip(*pts, strict=True))
        c, ls, mk = styles[name]
        ax.errorbar(x, y, yerr=e, color=c, ls=ls, marker=mk, ms=4, capsize=2, label=name)
    ax.set_xlabel("M/N")
    ax.set_ylabel("PSNR (dB), mean ± std over 300 images")
    ax.set_title("CelebA 64x64, Cake-Cutting Hadamard, noiseless", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"parity_psnr.{ext}", dpi=200)
    print((HERE / "parity_linear.md").read_text())
    if (HERE / "parity_learned.md").exists():
        print((HERE / "parity_learned.md").read_text())


if __name__ == "__main__":
    main()
