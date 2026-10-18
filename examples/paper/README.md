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
| `uc2_fspi_binary_sweep.batch_config` | UC2, CelebA 64 × 64 | binary FSPI, Floyd–Steinberg (U = 1), same ratios as `uc2_fspi` | 164 / 328 / 656 / 984 / 1312 |
| `uc2_fspi_extended.batch_config` | UC2, CelebA 64 × 64 | grey-level FSPI at 48 / 64 %: the frames of Cake-Cutting at 24 / 32 % | 1968 / 2620 |
| `uc2_fspi_binary_extended.batch_config` | UC2, CelebA 64 × 64 | binary FSPI, Floyd–Steinberg (U = 1), at 48 / 64 % | 1968 / 2620 |

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

## Reproduce Table 4 on the Jetson (latency and energy)

Use the networks trained on the desktop. Copy `paper_r1_uc2_cake_cutting` (and `paper_r1_uc2_fspi`) from
`experiments/batch_tests/` to the same place on the board; nothing is retrained there.

**Board state.** JetPack 6.2, power mode 40 W (`nvpmodel -q` shows `40W` / `4`).

```bash
sudo jetson_clocks --store
sudo systemctl isolate multi-user.target   # stop the desktop session; work over SSH
sudo jetson_clocks                         # pin CPU 1497.6 MHz, GPU 1173 MHz, EMC maximum
# ... measure ...
sudo jetson_clocks --restore
sudo systemctl isolate graphical.target
```

Each report's `platform_state` shows the pinned clocks and the power mode at the start and the end of the run.

**Start ASPIR on the board without its desktop session:**

```bash
ssh -X user@<board>
cd aspir/src && ../.venv/bin/python main.py
```

The window shows on your computer and the board runs the measurement. Alternatively, keep the board's
desktop session and run ASPIR there. The idle baseline is then noisier: SD of the 1 s means about 100 mW,
against about 40 mW without the desktop session.

**Re-measure** (Batch Reports → Re-measure), on the batch report of each folder:
- timing and energy;
- warm-up 20, measurement runs 800, inference batch size 1;
- idle baseline 60 s.

Run one pass per row:

| Row | Use GPU | CPU threads: torch | CPU threads: BLAS |
|---|---|---|---|
| GPU | yes | Default | Default |
| CPU, torch defaults (R0's protocol) | no | Default | Default |
| CPU, fastest | no | 1 | Default |
| CPU, one core | no | 1 | 1 |

On the Jetson wheel the BLAS default is 8 OpenBLAS threads. The default CPU row is flagged as oversubscribed;
see *User Guide / Analysis → CPU Threads*.

**Acquisition time** is modelled, not measured: the DMD frames divided by the sampling rate, 10.752 kHz in
these configurations. Since ASPIR 1.3 a bipolar Cake-Cutting pattern counts two frames (P⁺ and P⁻), so
UC2 Cake-Cutting gives 30.32 / 60.83 / 121.84 / 182.85 / 243.68 ms at 4 / 8 / 16 / 24 / 32 %. The R1 batch
reports, written before this change, store half of that. Grey-level FSPI counts one frame per pattern:
15.25 / 30.51 / 61.01 / 91.52 / 122.02 ms, at the binary rate, which is optimistic for grey levels.
