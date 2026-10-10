#!/usr/bin/env python
"""Task C: walk from ASPIR's R0 scoring to the shared function, one factor at a time (ASPIR venv).

    ../../.venv/bin/python walk.py

Linear stage, starting from what R0 scored (the network input):
  L0  R0: network input = global min-max of the linear output over the whole dataset, then INT8;
      reference = stored original; data range = each reference image's max - min; no clipping
  L1  without the quantisation of the network input (UC2 INT8, UC1 INT4). UC1's linear outputs are
      recomputed from the stored INT4 originals, so its L1-L4 rows are approximate (see walk.md)
  L2  on the image's scale instead of the global min-max (Hadamard: pseudoinverse = native x M/N;
      Sweep has no image-scale output, so the best global gain and offset against the ground truth is used,
      an oracle upper bound)
  L3  clipped to [0, 1]
  L4  data range 1  (= shared function for Hadamard; checked against benchmarks/spyrit_comparison)
U-Net stage:
  U0  R0: network output, data range per image, no clipping
  U1  clipped to [0, 1]
  U2  data range 1  (= shared function)
Also each factor applied alone to R0, and the mean of -20 log10(range) that the per-image data range adds.
Writes walk.md.
"""

import numpy as np
from common import (
    HERE,
    hadamard_linear,
    md_table,
    metrics,
    network_input_map,
    oracle_affine,
    shared,
    stored,
    sweep_linear,
)

# run, test, linear method, quantisation levels of the dataset (UC2 INT8, UC1 INT4)
CASES = [
    ("UC2_R0", "Celeb - Cake Cutting 4%", "hadamard", 255),
    ("UC2_R0", "Celeb - Cake Cutting 32%", "hadamard", 255),
    ("UC2_RERUN", "Celeb - Cake Cutting 4%", "hadamard", 255),
    ("UC2_RERUN", "Celeb - Cake Cutting 32%", "hadamard", 255),
    ("UC1_SWEEP", "Beam prof - stride 2", "sweep", 15),
    ("UC1_CCH", "Beam prof - CCH 8.3%", "hadamard", 15),
]


def walk(run, test, kind, levels):
    a = stored(run, test)
    X, masks = a["originals"], a["masks"]
    N, M = X[0].size, len(masks)
    L = hadamard_linear(X, masks) if kind == "hadamard" else sweep_linear(X, masks)
    ga, gb, net_in_float, dev = network_input_map(L, a["reconstructions"], levels)
    if kind == "hadamard":
        image_scale, scale_note = L * M / N, f"pseudoinverse (native x M/N, M/N = {M}/{N})"
    else:
        oa, ob, image_scale = oracle_affine(L, X)
        scale_note = f"oracle global affine (gain {oa:.3g}, offset {ob:.3g})"
    steps = [
        ("L0", "R0 (network input, range per image, no clip)", a["reconstructions"], "per_image", False),
        ("L1", f"- {'INT8' if levels == 255 else 'INT4'} quantisation of the network input", net_in_float,
         "per_image", False),
        ("L2", f"image scale: {scale_note}", image_scale, "per_image", False),
        ("L3", "+ clip to [0, 1]", image_scale, "per_image", True),
        ("L4", "+ data range 1", image_scale, 1.0, True),
        ("U0", "R0 (network output, range per image, no clip)", a["denoised"], "per_image", False),
        ("U1", "+ clip to [0, 1]", a["denoised"], "per_image", True),
        ("U2", "+ data range 1", a["denoised"], 1.0, True),
    ]
    rows, prev = [], {}
    for sid, label, est, dr, clip in steps:
        m = metrics(X, est, data_range=dr, clip=clip)
        stage = sid[0]
        delta = m["psnr"] - prev[stage] if stage in prev else float("nan")
        prev[stage] = m["psnr"]
        rows.append({"step": sid, "change": label, "PSNR": m["psnr"], "Δ PSNR": delta, "SSIM": m["ssim"],
                     "LPIPS": m["lpips"]})
    alone = {
        "quantisation removed": metrics(X, net_in_float, with_lpips=False)["psnr"],
        "image scale": metrics(X, image_scale, with_lpips=False)["psnr"],
        "clip": metrics(X, a["reconstructions"], clip=True, with_lpips=False)["psnr"],
        "data range 1": metrics(X, a["reconstructions"], data_range=1.0, with_lpips=False)["psnr"],
    }
    r0_lin = rows[0]["PSNR"]
    alone = {k: v - r0_lin for k, v in alone.items()}
    ranges = X.reshape(len(X), -1).max(1) - X.reshape(len(X), -1).min(1)
    info = {"a": ga, "b": gb, "dev_levels": dev, "range_term_db": float(np.mean(-20 * np.log10(ranges))),
            "range_min": float(ranges.min()), "range_mean": float(ranges.mean())}
    if kind == "hadamard":
        info["shared_linear"] = shared(X, L * M / N)["psnr"]
    info["shared_unet"] = shared(X, a["denoised"])["psnr"]
    return rows, alone, info


def main():
    md = ["# Task C — from ASPIR's R0 scoring to the shared function, one factor at a time", "",
          "Δ PSNR is the change from the previous step of the same stage. Gain = U-Net minus linear at the",
          "same step (R0: U0 - L0; shared: U2 - L4).", ""]
    fmt = {"PSNR": "{:.2f}", "Δ PSNR": "{:+.2f}", "SSIM": "{:.4f}", "LPIPS": "{:.4f}"}
    for run, test, kind, levels in CASES:
        rows, alone, info = walk(run, test, kind, levels)
        lin = {r["step"]: r for r in rows}
        md += [f"## {run} — {test}", "", md_table(rows, ["step", "change", "PSNR", "Δ PSNR", "SSIM", "LPIPS"], fmt),
               "",
               f"- Network input recovered as q({info['a']:.4g} · L + {info['b']:.4g}); largest deviation from "
               f"the stored input: {info['dev_levels']:.0f} quantisation step(s) "
               f"({'INT8' if levels == 255 else 'INT4, linear output recomputed from the stored INT4 originals'}).",
               f"- Gain: R0 {lin['U0']['PSNR'] - lin['L0']['PSNR']:+.2f} dB; shared function "
               f"{lin['U2']['PSNR'] - lin['L4']['PSNR']:+.2f} dB.",
               "- Each factor alone on the R0 linear number: " +
               ", ".join(f"{k} {v:+.2f} dB" for k, v in alone.items()) + ".",
               f"- Per-image data range: mean range {info['range_mean']:.3f} (min {info['range_min']:.3f}); "
               f"mean of -20 log10(range) = {info['range_term_db']:+.2f} dB.",
               f"- Check, shared function computed directly: U-Net {info['shared_unet']:.2f} dB"
               + (f", linear (pseudoinverse) {info['shared_linear']:.2f} dB." if "shared_linear" in info else "."),
               ""]
        print(f"{run} {test} done", flush=True)
    (HERE / "walk.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
