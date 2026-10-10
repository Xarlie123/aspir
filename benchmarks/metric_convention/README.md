# PSNR convention diagnostic (R2.1b)

Why ASPIR's Use Case 1 / 2 quality numbers differ from the SPyRiT comparison's for the same outputs, which
factor causes each gap, and what the numbers are under one scale-correct convention.

This folder only reads stored files. It changes no metric code, no analyser, no applicator, and no stored
experiment output. All numbers come from `experiments/batch_tests/` (paths below) and are recomputed by
the scripts.

**Status.** The recommended convention (D) is applied in ASPIR as *metric convention 2*, together with the
Hadamard Linear scale fix and a stable Cake-Cutting order; see `docs/changelog.md`. The scripts here still
reproduce convention 1, the one behind the R0 numbers.

## Reproduce (ASPIR venv, from this folder)

```bash
PY=../../.venv/bin/python
$PY reproduce.py        # B: manuscript numbers from the stored reports and arrays -> reproduce.md
$PY walk.py             # C: one factor at a time, R0 -> shared function           -> walk.md
$PY scales.py           # A.2: output scale of every applicator                     -> scales.md
$PY convention.py       # D.5: UC1 / UC2 under the proposed convention              -> convention.md / .csv
$PY earlier_reports.py  # E: R1.1 FSPI vs Cake-Cutting under both conventions       -> earlier_reports.md
$PY side_checks.py      # F: parameter counts, 403db38 exposure                     -> side_checks.md
```

## Stored runs

| Key | Folder | Notes |
|---|---|---|
| `UC2_R0` | `batch_test_celeb_hadamard_results_1/` | 2026-04-26. **The manuscript's UC2** (Fig. 6a, Section 3.2 text) |
| `UC2_RERUN` | `uc2_batch_test_celeb_3000_hadamard_results/` | 2026-05-02: second training of the same recipe on identical originals, network inputs and masks. Used as "UC2" by the SPyRiT comparison, R1.3b and R1.3c |
| `UC1_SWEEP` | `uc1_batch_test_strides_paper_2000_results/` | UC1 Sweep rows of Table 3 |
| `UC1_CCH` | `uc1_batch_test_cake_cutting_hadamard_paper_2000_results/` | UC1 CCH rows of Table 3 |

## A. ASPIR's metric path (R0)

All three comparison levels go through one function:
`simulation_engine/_5_analyzer/analyzer_noise.py:NoiseAnalyzer._compute_metrics` (lines 43–58).

| Item | What ASPIR does | Where |
|---|---|---|
| Reference | the clean image **after** the dataset's quantisation (UC2 INT8 = lossless for 8-bit CelebA; UC1 INT4 = 16 levels), global min / max of the clean array | `postprocessor_nn.py:_apply_data_format_to_arrays` 520–567; `test_dataset` 813 |
| "Reconstructed" | **the network input**, not the applicator output: linear output min-max normalised with the min / max over the whole dataset (train + val + test), then quantised like the reference | `postprocessor_nn.py:__init__` 352–354, 362; `test_dataset` 813–837 |
| "Denoised" | network output, unclipped (UC2 range −0.11 to 1.41) | `test_dataset` 813–837 |
| PSNR | data range = `reference.max() - reference.min()` of each image; mean of per-image PSNR | `analyzer_noise.py` 47, 52, 138–144 |
| SSIM | scikit-image defaults (7×7 uniform window, no Gaussian weights), same per-image data range | `analyzer_noise.py` 53–57 |
| LPIPS | AlexNet v0.1, grey repeated to 3 channels, x → 2x − 1, no clipping | `analyzer_noise.py` 40, 60–98 |
| Report fields | `psnr_recons` = network input vs reference; `psnr_denoised` = output vs reference | `runner.py:_analyze_results` 1017–1053 (parallel path 475) |

**Same path in:**
- Single Test *Noise Analysis* and *Reports* (`ui_noise_analysis_handler.py:90`, `ui_reports_handler.py:250`);
- `Simulation.set_analyzer` (`simulation.py:239`);
- training curves (`postprocessor_nn.py:_compute_val_metrics` 758–759).

Re-measure copies the quality fields without recomputing them. Batch Reports' *PSNR vs Sampling Ratio*
chart, the source of Fig. 6a, plots those fields with `np.std` of the per-image values
(`quality_view.py:_psnr_vs_sampling_rows` 952, `_draw_psnr_vs_sampling_chart` 997).

**One exception:** the figure-export popup computes its own PSNR with data range 1
(`visual_comparison.py:_compute_psnr` 370–389). Numbers printed on exported figures therefore differ from
the reports (≈ 0.9 dB on UC2).

**Output scale of each applicator** (`scales.md`, 16 UC2 images, M/N ≈ 0.16):

| Method | Formula in code | DC gain | Image scale? |
|---|---|---|---|
| Hadamard Linear | Hᵀy / M (`torch_backend/solvers.py:hadamard_linear` 27–29; legacy `applicator_hadamard.py` 147–149) | 6.25 = N/M | **no**: pseudoinverse × N/M (docstring derives Hᵀ/N with N = pixels, code divides by the row count) |
| Pseudoinverse | least squares | 1.000 | yes |
| FISTA / TV | regularised least squares | 1.000 | yes |
| Fourier inverse | unit scale by construction (`applicator_fourier.py` 43, 117–124) | 1.000 | yes |
| Sweep Linear | Sᵀ(y − ȳ)/M (`applicator_sweep.py` 44, 98) | 3.59 | no (correlation estimator) |
| Ghost Imaging (scatter) | Sᵀ(y − ȳ)/M (`applicator_scatter.py` 44, 87) | 0.00 | no (correlation estimator) |

## B. Reproduction (`reproduce.md`)

**UC1, Table 3:** all eight rows are reproduced from the stored report fields to ≤ 5·10⁻⁵ (the CSV
rounding).

**UC2:** the manuscript numbers are reproduced from `UC2_R0`:

| Manuscript | Stored (UC2_R0) |
|---|---|
| linear 15.3 / 18.8 dB | 15.33 / **18.86** |
| U-Net 21.3 / 28.2 dB | 21.27 / 28.21 |
| gain +6.0 / +9.4 dB | **+5.94 / +9.34** |

- The text's 18.8 should round to 18.9.
- The gains were taken from the rounded values.
- Fig. 6a's curves are `UC2_R0`'s fields.
- `UC2_RERUN` gives U-Net 21.67 / 28.53 dB with the same linear values.

Recomputing every field from the stored `test_images.npz` with ASPIR's own `NoiseAnalyzer` gives the stored
value exactly.

## C. One factor at a time (`walk.md`)

UC2, PSNR in dB. Linear stage from L0 (R0) to L4 (shared function); U-Net stage from U0 to U2.

| Step | Change | R0 4 % | R0 32 % | Rerun 4 % | Rerun 32 % |
|---|---|---|---|---|---|
| L0 | R0: network input, range per image | 15.33 | 18.86 | 15.33 | 18.86 |
| L1 | − INT8 of the network input | +0.00 | +0.00 | +0.00 | +0.00 |
| L2 | image scale (pseudoinverse) instead of the dataset min-max | **+3.91** | **+8.69** | +3.91 | +8.69 |
| L3 | + clip to [0, 1] | +0.04 | +0.04 | +0.04 | +0.04 |
| L4 | + data range 1 → **shared** | +0.93 → 20.21 | +0.93 → 28.52 | 20.21 | 28.52 |
| U0 | R0: network output, range per image | 21.27 | 28.21 | 21.67 | 28.53 |
| U1 | + clip | +0.00 | +0.09 | +0.00 | +0.02 |
| U2 | + data range 1 → **shared** | +0.93 → 22.20 | +0.93 → 29.22 | 22.60 | 29.47 |

- **The U-Net offset (≈ 1.3 dB at 4 %)** is two separate things:
  - 0.93 dB from the per-image data range: exactly the mean of −20 log₁₀(range); the mean range is 0.903;
  - 0.40 dB from the training run: the manuscript used `UC2_R0`, the SPyRiT comparison `UC2_RERUN`.
- **The linear gap** (4.9 / 9.7 dB) is:
  - 3.91 / 8.69 dB from scoring the network input instead of the linear reconstruction on the image's scale;
  - 0.93 dB from the data range;
  - 0.04 dB from clipping.
- The rerun's last rows equal the SPyRiT comparison (20.21 / 28.52 linear, 22.60 / 29.47 U-Net).

**UC1** (approximate linear rows: recomputed from the stored INT4 originals):
- Sweep stride 2: R0 linear 9.45 dB, rising 11.4 dB to 21.1 dB with the best global gain and offset.
- CCH 8.3 %: R0 linear 13.14 dB, rising 10.0 dB to 23.45 dB with the pseudoinverse.
- UC1 images span exactly [0, 1], so the data range contributes nothing there.

## D. Verdicts

| Factor | Verdict |
|---|---|
| Linear stage scored on the network input (dataset min-max + quantisation) | **Bug in what is reported.** The code is consistent, but the field is presented (GUI "Reconstructed", manuscript "raw linear reconstruction") as the reconstruction's quality. It is the quality of an affine-distorted copy, and it drives the R0 "U-Net gain" |
| Hadamard Linear = pseudoinverse × N/M | **Bug.** It contradicts its docstring's derivation and differs from every other image-scale applicator. It is invisible in batch metrics, because the min-max removes any constant gain, but it shows wherever the raw output is used |
| Sweep / Ghost Imaging correlation output | **Convention** (standard estimators), but not on any image scale: their linear PSNR needs an explicit calibration |
| Per-image data range | **Convention**: documented in code, identical for every stage and method. Non-standard (−0.93 dB on UC2), and inconsistent with the figure-export PSNR |
| No clipping | **Convention**, identical everywhere (≤ 0.09 dB here) |
| Quantised reference | Convention, applied identically. UC1 scores against a 16-level reference |
| Mean of per-image values | Same in both conventions (not a factor) |

**Is the R0 "U-Net gain" an artefact?** Yes, mostly:

| Ratio | R0 gain | Scale-correct gain |
|---|---|---|
| UC2, 4 % → 32 % | +5.9 → +9.3 dB | +1.99, +1.89, +0.58, +0.85, +0.70 dB (shrinks with the ratio) |
| UC1 | +12.5 to +16.2 dB | +3.0 to +4.6 dB |

**Same treatment of all algorithms?**
- In the batch report, yes in form: every linear output goes through the same dataset min-max, which also
  hides Hadamard's N/M.
- But the min-max is set by each method's extreme pixels over the dataset, so methods with ringing or
  outliers are compressed more.
- Outside the batch metrics, Hadamard Linear is off by N/M while pseudoinverse, FISTA, TV and Fourier are not.

**Recommended v1.3.0 convention.** I agree with the proposal, with two amendments:
1. Score the linear stage on its own output on the image's scale:
   - fix Hadamard Linear to Hᵀy/N;
   - correlation estimators (Sweep, Ghost Imaging) are scored after a global gain and offset fitted on the
     training split, and labelled "calibrated";
   - keep the network-input metric as a separate field, named as such.
2. Clip every estimate to [0, 1]; data range 1; mean of per-image values; LPIPS on the clipped estimate.

Code changes, not applied (≈ 150–200 lines plus ≈ 80 lines of tests):

| Change | Where | Size |
|---|---|---|
| Data range 1 + clip | `analyzer_noise.py`, `postprocessor_nn.py:_compute_val_metrics`, `visual_comparison.py` | ~15 lines |
| Divide Hadamard Linear by N | `solvers.py:hadamard_linear`, legacy `applicator_hadamard.py`, docstrings | ~5 lines |
| Keep the raw linear output of the test split; train-split calibration for correlation methods | `PostprocessorNN.__init__` / `test_dataset` | ~40 lines |
| New fields `psnr_linear` / `ssim_linear` / `lpips_linear`; `metric_convention: 2` in the metadata | `runner.py`, report writer | ~30 lines |
| Use the new fields and mark mixed-convention reports | Batch Reports chart and tables, Single Test widgets | ~50 lines |
| Tests | Hadamard Linear equals the pseudoinverse at partial sampling; the convention on known arrays | ~80 lines |

What it would break or change:
- every stored report becomes non-comparable with new ones (hence the version field);
- Table 3, Fig. 6a and the UC1 / UC2 text;
- training-curve values;
- R1.1's crossover statement (E);
- the `norm_lo` / `norm_span` exported for edge deployment of Hadamard models scale by M/N. They stay
  consistent within each export; only old exports mixed with new code would mismatch;
- the GPU parity tests that pin the Hadamard formula need updating.

## E. Earlier revision results (`earlier_reports.md`)

| Result | Metric used | Under the proposed convention |
|---|---|---|
| R1.1 FSPI vs CC (50 CelebA) | per-image min-max of **estimate and reference**, data range 1 (a third convention) | **Changes.** On the 300 UC2 test images, FSPI − CC is +0.49 / +0.70 / +0.31 / −0.16 / −0.66 dB under R1.1's convention and −0.01 / +0.04 / −0.50 / −1.09 / −1.71 dB under the shared one: a tie at 4–8 %, Cake-Cutting ahead from 16 %. FSPI keeps a higher SSIM at 4–8 % |
| R1.1 binary vs grey (0.5 dB) | same per-image min-max | relative cost; likely similar but not recomputed |
| R1.2 CPU vs GPU parity | PSNR **between** two outputs of the same method | unaffected (no reference) |
| R1.3a | no quality numbers | — |
| R1.3b SPIHIM | least-squares gain + offset of each output onto the reference, data range 1 | self-consistent and scale-free, but not comparable with either convention. Its "UC2" nets are `UC2_RERUN`. The STL-10 "input → output" numbers use the R0 convention |
| R2.1 SPyRiT | shared function | already the proposed convention; its "ASPIR, UC2 checkpoints" row is `UC2_RERUN`. With `UC2_R0`: 22.20 / 23.93 / 25.27 / 27.55 / 29.22 dB |

## F. Side checks (`side_checks.md`)

**F1. Trainable parameters:**

| Net | Configuration | Trainable parameters | Bottleneck channels |
|---|---|---|---|
| UC2 checkpoints, both runs | `UNetTranspose` [8, 16, 32, 64] | **486,409**, not ~130 K | 128 |
| current `UNet`, same configuration | [8, 16, 32, 64] | 540,809 | 128 |
| UC1 checkpoints | [8, 16] | **29,481**, not 7.3 K | 32 |

- Per-block breakdown for UC2: bottleneck 221,696; up-convolutions 147,360; encoder 73,704; transposed
  convolutions 43,640.
- The nearest configuration to 130 K is a three-level U-Net [8, 16, 32]: 121 K (`UNetTranspose`), 134 K
  (`UNet`).
- 7.3 K matches a two-level U-Net with **4** initial channels, [4, 8]: 7,477 (`UNetTranspose`), not 8.

**F2. Exposure to `403db38`:**
- All stored UC1 and UC2 `.batch_config` files are GUI-saved, with `reconstruction_method: conventional` and
  20 warm-ups; every report records the native method. Not exposed.
- But the `UC2_R0` batch run itself recorded **5 warm-ups**. Before `b4768ac` (2026-05-02) the GUI default
  was 5. Its re-measures on the Jetson recorded 20.
- Table 4's latencies and energies (e.g. GPU 8.01 ms, CPU 76.5 ms at 4 %) match none of the stored UC2
  reports, Jetson re-measures included, so their warm-up count cannot be confirmed from this repository.
