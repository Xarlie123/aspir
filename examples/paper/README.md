# Paper batch configurations (Use Cases 1 and 2)

The Batch Test configurations behind the quality numbers of Use Cases 1 and 2 in the ASPIR paper
(*Intelligent Computing*, revision R1). They reproduce the original submission's recipe, data and settings,
with best-validation weights. Run them with ASPIR ≥ 1.3.

| File | Use case | Tests | Patterns M |
|---|---|---|---|
| `uc1_sweep.batch_config` | UC1, IR beams 32 × 32 | Sweep, bar width 5, 4 angles, strides 2 / 4 / 6 / 8 | 85 / 43 / 29 / 23 |
| `uc1_cake_cutting.batch_config` | UC1, IR beams 32 × 32 | Cake-Cutting Hadamard, 2.2 / 2.8 / 4.2 / 8.3 % | 22 / 28 / 43 / 84 |
| `uc2_cake_cutting.batch_config` | UC2, CelebA 64 × 64 | Cake-Cutting Hadamard, 4 / 8 / 16 / 24 / 32 % | 163 / 327 / 655 / 983 / 1310 |
| `uc2_fspi.batch_config` | UC2, CelebA 64 × 64 | Fourier SPI, four-step, circular order, grey-level, same ratios | 164 / 328 / 656 / 984 / 1312 |
| `uc2_fspi_binary.batch_config` *(optional)* | UC2, CelebA 64 × 64 | binary FSPI at 16 %: Floyd–Steinberg (U = 1) and Bayer (U = 2) | 656 / 656 |

**Shared settings** (every test):
- reconstruction: the mask's native method;
- network: U-Net, [8, 16] for UC1 and [8, 16, 32, 64] for UC2;
- training: 50 epochs, batch 16, Adam, lr 1e-3, weight decay 1e-4, MSE, split 80 / 10 / 10;
- keep best-validation weights: on;
- reports: training curves, quality, timing (20 warm-up + 800 runs) and energy (60 s idle baseline).

There is one batch per mask family on purpose: Batch Reports' *PSNR vs Sampling Ratio* chart draws one curve
per batch.

## Run

1. **Generate the dataset** in Single Test, Dataset step, with exactly these values. Each config also stores
   them under `dataset_info`.

   | | UC1 | UC2 |
   |---|---|---|
   | Source | IR beams | Celebrities (Kaggle download) |
   | Image size | 32 | 64 |
   | Images | 2000 | 3000 |
   | Seed | 10 | 10 |
   | Data format | INT4 | INT8 |
   | Other | modes 25 % each (Gaussian, Hermite–Gauss, Laguerre–Gauss, doughnut), max mode order 3, speckle 0.3 | — |

2. **Batch Test:** *Load Batch Config*, choose a file here, set the export level to *All data*, then *Run All
   Tests*. Tests train in parallel (4 threads). Timing and energy are measured afterwards, one test at a time.
3. **Check the data:**

   ```bash
   .venv/bin/python examples/paper/check_data.py experiments/batch_tests/<batch results folder>
   ```

   It compares the SHA-256 of the exported test images with the original submission's. A mismatch means a
   different dataset or split.

## What differs from the original submission (R0)

Settings are unchanged; ASPIR 1.3 changes how some of them are computed:

| | R0 | Now |
|---|---|---|
| Weights kept | last epoch | best-validation epoch |
| Quality metrics | data range per image, no clipping; "reconstructed" scored the network input | convention 2: clip to [0, 1], data range 1; the linear level scores the reconstruction on the image's scale (`docs/user_guide/analysis.md`) |
| Network-input normalisation | min / max over the whole dataset | min / max over the training split; integer formats quantise on the fixed [0, 1] grid |
| Hadamard Linear | Hᵀy / M | Hᵀy / N (pseudoinverse); the network input is unaffected |
| Cake-Cutting ties | default sort | stable sort (one row differs at 4, 8 and 24 % for UC2) |
| U-Net decoder | transposed convolution: 486,409 (UC2) / 29,481 (UC1) parameters | upsample + convolution: 540,809 / 32,681 parameters |

**Timing and energy:** desktop numbers validate the backends only. The paper's latency and energy come from
re-measuring the exported models on the Jetson Orin NX.
