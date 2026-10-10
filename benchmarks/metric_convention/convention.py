#!/usr/bin/env python
"""Task D.5: the quality numbers under the proposed convention, from stored outputs (ASPIR venv).

    ../../.venv/bin/python convention.py

Convention: the linear stage is scored on its own output brought to the image's scale; every estimate is
clipped to [0, 1]; PSNR and SSIM with data range 1; LPIPS as ASPIR (AlexNet); mean of per-image values
(= benchmarks/spyrit_comparison/compare.py:score).

* UC2 (both stored runs): linear = pseudoinverse of the stored rows (native Hadamard Linear x M/N);
  U-Net = the stored network outputs.
* UC1 Table 3: CCH linear = pseudoinverse (as UC2). Sweep Linear has no image-scale output, so two
  stand-ins are given: the best global gain and offset against the ground truth (an oracle upper bound for
  any rescaling of Sweep Linear) and ASPIR's pseudoinverse on the sweep masks (a different algorithm). UC1's
  linear outputs are recomputed from the stored INT4 originals (the float images are not stored), so the
  UC1 linear values are approximate; U-Net values are exact.
Writes convention.md and convention.csv.
"""

import csv
import logging

import numpy as np
from common import HERE, hadamard_linear, md_table, oracle_affine, report, shared, stored, sweep_linear

LOG = logging.getLogger("convention")


def aspir_pinv(X, masks):
    """ASPIR's ApplicatorPseudoinverse on the given masks (public API)."""
    from scales import ArrayDataset

    from simulation_engine._2_mask_gen.mask_measured import MaskMeasured
    from simulation_engine.simulation import Simulation

    sim = Simulation(LOG)
    sim.set_dataset(ArrayDataset(X))
    sim.mask = MaskMeasured(masks.astype(np.int8), logger=LOG)
    sim.set_applicator(reconstruction_method="pseudoinverse")
    return np.asarray(sim.applicator.process_dataset(), dtype=np.float64).reshape(X.shape)


def row(run, t, kind):
    a = stored(run, t["name"])
    X, masks = a["originals"], a["masks"]
    N, M = X[0].size, len(masks)
    out = {"run": run, "test": t["name"], "M": M,
           "R0 linear": t["psnr_recons"], "R0 U-Net": t["psnr_denoised"],
           "R0 gain": t["psnr_denoised"] - t["psnr_recons"]}
    if kind == "hadamard":
        lin, label = hadamard_linear(X, masks) * M / N, "pseudoinverse"
    else:
        _, _, lin = oracle_affine(sweep_linear(X, masks), X)
        label = "Sweep Linear, oracle affine"
        p = shared(X, aspir_pinv(X, masks))
        out.update({"alt linear (Sweep pinv) PSNR": p["psnr"], "alt SSIM": p["ssim"], "alt LPIPS": p["lpips"]})
    li, un = shared(X, lin), shared(X, a["denoised"])
    out.update({"linear method": label, "linear PSNR": li["psnr"], "linear SSIM": li["ssim"],
                "linear LPIPS": li["lpips"], "U-Net PSNR": un["psnr"], "U-Net SSIM": un["ssim"],
                "U-Net LPIPS": un["lpips"], "gain": un["psnr"] - li["psnr"]})
    print(run, t["name"], f"{out['linear PSNR']:.2f} {out['U-Net PSNR']:.2f}", flush=True)
    return out


def main():
    rows = []
    for run in ("UC2_R0", "UC2_RERUN"):
        rows += [row(run, t, "hadamard") for t in report(run)["results"]]
    rows += [row("UC1_SWEEP", t, "sweep") for t in report("UC1_SWEEP")["results"]]
    rows += [row("UC1_CCH", t, "hadamard") for t in report("UC1_CCH")["results"]]

    cols = ["run", "test", "M", "linear method", "linear PSNR", "linear SSIM", "linear LPIPS", "U-Net PSNR",
            "U-Net SSIM", "U-Net LPIPS", "gain", "R0 linear", "R0 U-Net", "R0 gain"]
    fmt = dict.fromkeys(("linear SSIM", "linear LPIPS", "U-Net SSIM", "U-Net LPIPS", "alt SSIM", "alt LPIPS"), "{:.4f}")
    with open(HERE / "convention.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)
    md = ["# Task D.5 — quality under the proposed convention (from stored outputs)", "",
          "Convention: linear output on the image's scale; clip to [0, 1]; data range 1; LPIPS AlexNet;",
          "mean of per-image values. R0 columns: the stored report fields (manuscript convention).", ""]
    for group, title in (("UC2", "UC2 (300 test images)"), ("UC1", "UC1 Table 3 (200 test images)")):
        sub = [r for r in rows if r["run"].startswith(group)]
        md += [f"## {title}", "", md_table(sub, cols, fmt), ""]
    sweep = [r for r in rows if r["run"] == "UC1_SWEEP"]
    md += ["### UC1 Sweep: pseudoinverse on the sweep masks (alternative image-scale linear)", "",
           md_table(sweep, ["test", "M", "alt linear (Sweep pinv) PSNR", "alt SSIM", "alt LPIPS", "U-Net PSNR"],
                    fmt), ""]
    (HERE / "convention.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
