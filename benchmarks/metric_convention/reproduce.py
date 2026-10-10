#!/usr/bin/env python
"""Task B: reproduce the manuscript's quality numbers from stored files (ASPIR venv).

    ../../.venv/bin/python reproduce.py

For every test of UC2 (both stored runs) and UC1 (eight rows), three sources side by side:
* the value printed in the manuscript (UC2: text of Section 3.2; UC1: Table 3 CSV);
* the field stored in the batch report (psnr_recons / psnr_denoised / ssim_* / lpips_*);
* the same metric recomputed now from the stored test_images.npz with ASPIR's own NoiseAnalyzer.
Writes reproduce.md.
"""

import csv

from common import HERE, MANUSCRIPT, MANUSCRIPT_UC2, RUNS, aspir_r0, md_table, report, stored


def rows_for(run):
    out = []
    for t in report(run)["results"]:
        a = stored(run, t["name"])
        rec = aspir_r0(a["originals"], a["reconstructions"])
        den = aspir_r0(a["originals"], a["denoised"])
        out.append({
            "run": run, "test": t["name"], "M": len(a["masks"]),
            "rep_psnr_lin": t["psnr_recons"], "rep_ssim_lin": t["ssim_recons"], "rep_lpips_lin": t["lpips_recons"],
            "rep_psnr_unet": t["psnr_denoised"], "rep_ssim_unet": t["ssim_denoised"],
            "rep_lpips_unet": t["lpips_denoised"], "rep_gain": t["psnr_denoised"] - t["psnr_recons"],
            "re_psnr_lin": rec["psnr"], "re_psnr_unet": den["psnr"],
            "max_dev_recomputed": max(abs(rec["psnr"] - t["psnr_recons"]), abs(den["psnr"] - t["psnr_denoised"])),
        })
    return out


def main():
    md = ["# Task B — reproduction of the manuscript numbers", ""]
    cols = ["test", "M", "rep_psnr_lin", "rep_ssim_lin", "rep_lpips_lin", "rep_psnr_unet", "rep_ssim_unet",
            "rep_lpips_unet", "rep_gain", "max_dev_recomputed"]
    fmt = {"rep_ssim_lin": "{:.4f}", "rep_lpips_lin": "{:.4f}", "rep_ssim_unet": "{:.4f}",
           "rep_lpips_unet": "{:.4f}", "max_dev_recomputed": "{:.1e}"}
    uc2 = {}
    for run in ("UC2_R0", "UC2_RERUN"):
        rows = rows_for(run)
        uc2[run] = {r["test"]: r for r in rows}
        md += [f"## {run}: `{RUNS[run].relative_to(RUNS[run].parents[2])}/results.batch_analysis_report`", "",
               md_table(rows, cols, fmt), ""]

    r0 = uc2["UC2_R0"]
    t4, t32 = r0["Celeb - Cake Cutting 4%"], r0["Celeb - Cake Cutting 32%"]
    check = [
        ("linear 4 %", MANUSCRIPT_UC2["linear_4"], t4["rep_psnr_lin"]),
        ("linear 32 %", MANUSCRIPT_UC2["linear_32"], t32["rep_psnr_lin"]),
        ("U-Net 4 %", MANUSCRIPT_UC2["unet_4"], t4["rep_psnr_unet"]),
        ("U-Net 32 %", MANUSCRIPT_UC2["unet_32"], t32["rep_psnr_unet"]),
        ("gain 4 %", MANUSCRIPT_UC2["gain_4"], t4["rep_gain"]),
        ("gain 32 %", MANUSCRIPT_UC2["gain_32"], t32["rep_gain"]),
    ]
    md += ["## UC2 manuscript text vs UC2_R0", "",
           md_table([{"quantity": q, "manuscript": m, "stored (UC2_R0)": s, "rounded": f"{s:.1f}",
                      "match": "yes" if f"{s:.1f}" == f"{m:.1f}" else "NO"} for q, m, s in check],
                    ["quantity", "manuscript", "stored (UC2_R0)", "rounded", "match"]), ""]

    with open(MANUSCRIPT / "figures/table_3_quality_metrics_UC1.csv", encoding="utf-8") as fh:
        table3 = {r["Test"]: r for r in csv.DictReader(fh)}
    uc1 = rows_for("UC1_SWEEP") + rows_for("UC1_CCH")
    t3rows = []
    for r in uc1:
        m = table3[r["test"]]
        pairs = [("PSNR Recon (dB)", "rep_psnr_lin"), ("SSIM Recon", "rep_ssim_lin"),
                 ("PSNR Denoised (dB)", "rep_psnr_unet"), ("SSIM Denoised", "rep_ssim_unet"),
                 ("LPIPS Denoised", "rep_lpips_unet"), ("ΔPSNR (dB)", "rep_gain")]
        worst = max(abs(float(m[k]) - r[v]) for k, v in pairs)
        t3rows.append({**r, "max |Table 3 - stored|": worst})
    md += ["## UC1 Table 3 vs stored reports", "",
           md_table(t3rows, [*cols, "max |Table 3 - stored|"], {**fmt, "max |Table 3 - stored|": "{:.1e}"}), ""]
    (HERE / "reproduce.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
