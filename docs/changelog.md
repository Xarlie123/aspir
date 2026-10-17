# Changelog

All notable changes to ASPIR will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.3.0] - 2026-10-17

### Added
- **Projections per pattern in the modelled acquisition.** The acquisition time counts the frames a DMD
  would project, not the stored patterns.
  - Bipolar ±1 Hadamard patterns (natural, Walsh–Paley, Cake-Cutting, Cal-Sal) default to differential
    acquisition, P⁺ and P⁻: two frames per pattern. `reference` (P⁺ plus one all-on frame, M + 1) and
    `per_pattern` (the old count) remain available.
  - Binary patterns (Scatter, Sweep, binary FSPI) and grey-level FSPI count one frame per pattern. An
    optional *Grey-level rate* gives the DMD rate for grey-level patterns.
  - Set it per test in Batch Test (`projection_scheme`, `grey_level_rate_khz` in `.batch_config`), in
    Re-measure and in Single Test → Timing. A scheme that does not apply falls back to the family default,
    with a log line.
  - Reports add `timing_num_projections`, `timing_projection_scheme` and `timing_pattern_polarity`.
  - The latency chart's *Mask projection* becomes *Acquisition (modelled)*, and Batch Reports warns when
    loaded reports mix schemes.
  - **New reports show twice the previous acquisition time for every Hadamard-family test.** At the same
    sampling ratio, UC2 Cake-Cutting at 16 % goes from 60.92 ms (655 frames) to 121.84 ms (1310 frames).
  - Stored reports are unchanged; without a scheme they read as one frame per pattern. The sampling ratio
    M/N is unchanged.
  - `tests/test_acquisition_model.py` (8/8).
- **CPU threads setting.** Batch Test (per test, saved in `.batch_config`), Re-measure and Single Test →
  Timing gain *CPU threads: torch / BLAS*.
  - *Default* keeps each library's own choice, as before.
  - The counts apply to every CPU pass (inference timing, energy blocks, reconstruction on the CPU) and
    are restored afterwards. The BLAS pool is set at run time through threadpoolctl, a new dependency.
  - Every CPU pass records the pools it actually ran with (`cpu_threads`), and `platform_state` gains
    `cpu_threading`.
  - Passes that oversubscribe the cores are flagged (`cpu_oversubscribed`) and logged as a warning. An
    example is torch's pool on top of the separate pthreads OpenBLAS pool of the Jetson torch wheels:
    66 ms against 11 ms at batch 1 on an Orin NX.
  - *User Guide / Analysis* has the numbers; `tests/test_cpu_threads.py` (7/7).
- **Platform state in every report.** Batch Test and Re-measure record the
  machine's state at the start and at the end of a run under
  `platform_state` in the report metadata: frequency governor and limits of
  every CPU cluster, whether the clocks are pinned, the Jetson power mode
  (`nvpmodel`), GPU clock and fan, the desktop GPU's clocks and
  performance state (`nvidia-smi`), the temperatures, and the torch version
  and thread count. A run can then be checked afterwards, for example that
  `jetson_clocks` held throughout. Read without root; the memory clock is not
  read.
- **Fourier Single-Pixel Imaging (mask family + native reconstruction).**
  ASPIR gains a seventh mask family and a seventh reconstruction algorithm,
  wired through the same `MaskABC` / `ApplicatorABC` pipeline, Single Test
  wizard, Batch Test runner and reports as the existing six.
  - `MaskFourier` (`simulation_engine/_2_mask_gen/mask_fourier.py`) generates
    sinusoidal grayscale patterns `a + b*cos(2*pi*(fx*x + fy*y) + phi)` with
    `a = b = 0.5`, so patterns are non-negative and need no complementary
    `P+`/`P-` projection. Parameters: `sampling_ratio` (M/N, the same meaning
    as everywhere else in ASPIR), `phase_steps` (4 = four-step, default;
    3 = three-step) and `spectrum_ordering` (`circular`, default, or
    `square`). Only the conjugate-symmetric half-plane is acquired, ordered
    low-frequency first, which is what makes FSPI competitive at low
    sampling ratios.
  - `ApplicatorFourier`
    (`simulation_engine/_3_applicator/applicator_fourier.py`) demodulates the
    phase-shifted bucket values into complex spectrum coefficients, fills the
    conjugate half, and inverts with `numpy.fft.ifft2`. The normalisation is
    unit scale and zero offset: a noiseless full-spectrum acquisition returns
    the ground truth (> 140 dB PSNR in the tests). The phase-shift
    differences are themselves the differential measurement, cancelling the
    pattern offset and the detector's DC drift.
  - The mask-agnostic solvers (Pseudoinverse, FISTA, TV-Norm) accept Fourier
    masks unchanged, even though the phase-shifted patterns of one
    coefficient are linearly dependent and the sensing matrix is therefore
    rank-deficient.
  - References: Zhang, Ma & Zhong, *Nat. Commun.* **6**, 6225 (2015) for the
    four-step scheme; Zhang et al., *Sci. Rep.* **7**, 12029 (2017) for the
    three-step variant.
- **Binary FSPI with dithering for DMD projection.** `MaskFourier` gains
  `binarization` (`"none"` default, `"floyd_steinberg"`, `"bayer"`) and
  `dmd_upsampling` (U, default 1). A DMD switches binary frames about two
  orders of magnitude faster than it renders grayscale by pulse-width
  modulation, so dithering trades pattern fidelity for frame rate. With U > 1
  the sinusoid is rendered on a `U*H x U*W` mirror grid before dithering; the
  scene is still modelled at the image size, so each pixel integrates the
  `U x U` mirrors covering it and recovers `U**2 + 1` effective levels.
  `mask.masks` always holds that effective pattern, so every reconstruction
  algorithm works unchanged, and `mask.dmd_pattern(index)` recomputes the
  binary frame on demand rather than keeping a full binary stack in memory.
  On 20 CelebA images at 64x64 and M/N = 0.16 the cost against grayscale
  patterns is about 0.5 dB. Reference: Zhang et al., *Sci. Rep.* **7**, 12029
  (2017).
- **Device-selectable reconstruction backend (CPU or CUDA).** All seven
  reconstruction algorithms now run on a PyTorch backend with one
  implementation per algorithm, so a CPU-versus-GPU comparison is the same code
  on two devices. Select the device with the new *Device* dropdown next to
  *Reconstruction method* (Single Test and Batch Test), the
  `reconstruction_device` field of a batch test (`"auto"` default, `"cpu"`,
  `"cuda"`), or `device=` on `Simulation.set_applicator` and every applicator.
  `"auto"` uses CUDA when available and silently falls back to the CPU
  otherwise. Precision is float32 by default, float64 optional.
  - The sensing matrix is applied through the cheapest exact operator: a fast
    Walsh-Hadamard transform for all Hadamard-family masks (rows recognised
    from the patterns, whatever the ordering), an FFT for grayscale Fourier
    masks, and a dense matrix otherwise, kept on the device when it fits and
    streamed from host memory in row blocks when it does not. Iterative
    solvers therefore scale to 256x256 and beyond on the GPU for the
    matrix-free families.
  - FISTA and TV reproduce the previous pyproximal iterations step for step
    with the same parameters (TV: Chambolle-Pock with the exact data-term
    prox). The TV prox is factorised once instead of solving a dense N x N
    system at every iteration, which alone makes TV about 25x faster on the
    same CPU.
  - The NumPy / pylops / pyproximal implementations remain available as
    `backend="legacy"` for reference; they are not exposed in the GUI.
  - `scripts/bench_reconstruction_scaling.py` benchmarks all seven algorithms
    across sizes, sampling ratios and devices (CSV and Markdown output).
  - `tests/test_gpu_reconstruction.py` checks parity with the legacy path, CUDA
    against CPU, FWHT against the explicit matrix for all four Hadamard
    orderings, and chunked against resident matrix products.
- **Experimental measurements.** Real single-pixel data can be reconstructed
  in ASPIR alongside simulated data (see *User Guide / Experimental Data*).
  - `measured.py`: an `.npz` measurement format (bucket values, patterns,
    acquisition order, optional reference and flat field, JSON metadata) and
    `DatasetFromMeasurements`.
  - `spihim.py`: an adapter for the open SPIHIM collection (OpenSpyrit,
    CC-BY 4.0). It pairs positive / negative patterns, offers panchromatic,
    band or single-bin reduction, drops saturated bins, applies the
    orientation determined from the printed target, and supports an optional
    flat field. `scripts/download_spihim.py` fetches the data with SHA-256
    checks; the data are never committed.
  - `ApplicatorABC.reconstruct(y)` reconstructs measured values with any of
    the seven algorithms on CPU or CUDA; the simulated path runs the same
    solver step. `MaskMeasured` holds read-only measured patterns.
  - `simulation_engine/experimental.py` sub-samples a full Hadamard
    acquisition in ASPIR's natural, Walsh-Paley, Cake-Cutting or Cal-Sal
    order (or the instrument's own). It also scales measurements to a
    unit-peak image so that the FISTA / TV defaults apply, and runs the
    simulated counterpart on the same rows.
  - Single Test gains an *Experimental measurements (file)* dataset source.
    The mask step then shows the measured patterns read-only, Test Masks
    reconstructs from the measured values, and Post-processing and Reports,
    which need ground truth, are closed.
  - `tests/test_experimental_data.py` covers the adapter, exact noiseless
    reconstruction, Cake-Cutting row selection and measured-vs-simulated
    parity for all seven algorithms.
- **Uncertainty of the dynamic energy.** Inference and reconstruction energy
  now report the dynamic energy per image with its uncertainty, next to the
  totals: E_dyn = (P̄ − P_idle)·t̄ over the measurement blocks.
  - The idle-baseline term uses the standard deviation of the idle power,
    not its standard error; the standard-error version is stored too.
  - The block term propagates the block-to-block spread to first order.
  - The combined σ and a 95 % bootstrap interval complete the set
    (`dynamic_energy_sigma_mj`, `dynamic_energy_ci95_mj`, ...;
    `dynamic_reconstruction_*` for the reconstruction stage).
  - Report metadata gains `P_idle_mean_w`, `P_idle_std_w`, `P_idle_sem_w` and
    `n_idle_samples`, per backend and in total, and the per-block readings are
    stored.
  - Each stage subtracts the idle power of the backends that measured it
    only. The reconstruction stage is now measured in 10 blocks.
  - Existing fields are unchanged. The derivation is in *User Guide /
    Analysis*, and `tests/test_energy_uncertainty.py` checks it.
- **Keep the best-validation weights.** A *Keep best-validation weights*
  checkbox in Single Test (post-processing tab) and in each Batch Test
  configuration (`keep_best_val` in `.batch_config`).
  - When checked, training ends by restoring the weights of the epoch with
    the lowest validation loss instead of keeping the last epoch's. It is the
    usual practice, and what SPyRiT's `train_model` does.
  - With a constant learning rate the last epoch can be several dB below the
    best one. In the SPyRiT comparison of the ASPIR paper the gap was 3.3 dB
    at M/N = 0.32.
  - Off by default, so saved configurations and earlier results reproduce
    unchanged. With an empty validation split the last epoch is kept and a
    warning is logged.
  - The weights copied are the model's whole `state_dict`, BatchNorm buffers
    included.
  - Where the kept epoch shows up:
    - in Single Test, as a dashed line on the training curves;
    - in Batch Reports, under *Training Summary*;
    - in the report, as `kept_epoch` and `kept_weights` (`"best_validation"`
      or `"last"`); `test_config.json` records `keep_best_val`.
  - Without the training-curves report, Batch Test adds one validation pass
    per epoch to find the best epoch. `tests/test_best_val_weights.py` checks
    the restore exactly, on both training paths.
- **Automated tests.** Eleven self-contained scripts in `tests/`, one per
  feature, each run directly (`.venv/bin/python tests/<script>.py`) with no test
  framework:
  - `test_acquisition_model.py`, `test_best_val_weights.py`, `test_cpu_threads.py`,
    `test_energy_uncertainty.py`, `test_experimental_data.py` and
    `test_fourier_fspi.py`;
  - `test_gpu_reconstruction.py`, `test_jetson_energy.py`,
    `test_metric_convention.py`, `test_solver_parameters.py` and
    `test_train_normalisation.py`.

  They pass on a desktop (CUDA) and on a Jetson Orin NX, also headless over
  SSH. Where a torch build cannot run `torch.linalg` on CUDA (the Jetson AI Lab
  wheel on JetPack 6.2), the CUDA check of the pseudoinverse and TV reports them
  as not run.

### Changed
- **The paper's analyses live with the paper.** `benchmarks/metric_convention/`,
  `benchmarks/spyrit_comparison/`, `scripts/experimental_spihim_figure.py` and
  `scripts/train_stl10_unets.py` analysed the results of the ASPIR paper and
  have moved out of the repository. It keeps the software, the batch
  configurations that reproduce the paper's experiments (`examples/paper/`),
  the SPIHIM download (`scripts/download_spihim.py`) and the reconstruction
  scaling benchmark (`scripts/bench_reconstruction_scaling.py`).
- **Reconstruction timing is device-synchronised.** `timing_reconstruction_ms`
  now uses the NN stage's protocol — synchronisation before and after each
  call, warm-up, repeats — and batch reports add
  `timing_reconstruction_std_ms`, `timing_reconstruction_device`,
  `timing_reconstruction_backend`, `timing_reconstruction_precision` and
  `timing_reconstruction_operator`. Values are not comparable with reports
  produced by the previous NumPy path.
- **Reconstruction energy is measured.** Batch tests with the *energy* report
  add `energy_reconstruction_mj` (per image) and its CPU/GPU split, attributed
  to the device the reconstruction ran on. Previously only NN inference energy
  was measured.
- **Quality metrics follow convention 2.** Every batch result and the report
  metadata record `metric_convention: 2`; Batch Reports warns when loaded
  reports mix conventions. Quality numbers are not comparable with earlier
  reports; timing and energy are unaffected. The diagnosis of convention 1 is
  part of the ASPIR paper's analysis.
  - Estimates are clipped to [0, 1], and PSNR / SSIM use a data range of 1,
    instead of each reference image's max − min. This is the same in
    analysis, reports, training curves and figure export.
  - The linear level (`psnr_recons`, `ssim_recons`, `lpips_recons`,
    `quality_per_image.*_noisy`) now scores the linear reconstruction on the
    image's scale (`PostprocessorNN.test_linear`).
    - It used to score the network input: the linear output min-max
      normalised over the dataset and quantised. That input is still
      reported, as `psnr_network_input` / `ssim_network_input`.
    - Correlation estimators (Sweep Linear, Ghost Imaging) have no natural
      scale. They are scored after a gain and offset fitted on the training
      split (`linear_calibration`; applicator attribute `IMAGE_SCALE`).
  - Exports add the image-scale linear reconstructions to `test_images.npz`
    (`linear`), and `metric_convention` plus `normalization.linear_divisor`
    to `test_config.json`.
  - Single Test saves them in `test_results.npz`.
- **Network-input normalisation is fitted on the training split.** The linear
  output entering the network is min-max normalised with the training images'
  minimum and maximum. It used to be the whole dataset's, test images
  included.
  - For the INT8 / INT4 formats the network input is quantised on the fixed
    [0, 1] grid, so validation or test values outside the training range
    clip, as a deployed integer input would. Before, the grid also spanned
    the whole array.
  - The train / val / test split is unchanged: it is now drawn first, from
    the same seed.
  - The exported `norm_lo` / `norm_span` are the training split's, and
    `test_config.json` records `normalization.fitted_on: "train split"`.
  - Reference images are quantised as before.
  - `tests/test_train_normalisation.py` (6/6).
- **Cake-Cutting order is stable.** Patterns with the same block count keep
  their Sylvester order, so the rows chosen at a cut-off no longer depend on
  the NumPy build. Masks saved by earlier versions may differ by a few tied
  rows (UC2's saved masks: one row at 4 %, 8 % and 24 %; ≤ 0.05 dB linear PSNR).

### Fixed
- **Re-measure times the FSPI reconstruction.**
  - Batch exports saved the Fourier patterns without each pattern's
    coefficient and phase, so re-measure could not build the Fourier
    reconstruction. It timed only the inference and left
    `timing_reconstruction_ms` at 0.
  - Exports now save that metadata in `masks.npz`, with the phase steps and
    the binarisation. For older exports, re-measure regenerates the metadata,
    and uses it only if the regenerated patterns equal the saved ones.
  - Re-measure also drops the source report's `dynamic_*` fields along with
    its energy fields, so a stage that is not re-measured no longer keeps the
    original machine's values.
  - `tests/test_fourier_fspi.py` (11/11).
- **Jetson energy is sampled at 200 Hz instead of read twice per block.**
  - The jtop backend multiplied the mean of two power readings, one at each
    end of a block, by the block's duration. jtop refreshes about once a
    second, so a GPU block of 80 inferences (about 0.6 s) got no reading of
    its own, and the readings could predate the block.
  - The Jetson backend now reads the module input rail (`VDD_IN`, or
    `POM_5V_IN`) of the INA3221 from sysfs at 200 Hz while a measurement is
    open, and integrates it with the trapezoidal rule (about 130 samples per
    GPU block on an Orin NX). The energy is scaled to the timed loop at the
    mean power measured. jtop is only the fallback.
  - The sub-rails (`VDD_CPU_GPU_CV`, `VDD_SOC`) are never added to the input
    rail. Before, the sysfs fallback summed every rail it found.
  - The sampler is paused outside measurements, so it does not run during the
    timing pass.
  - The idle baseline keeps the sampler running on the Jetson and stores 1 s
    means. Other backends are polled as before.
  - Reports carry the samples of each block (`energy_blocks.n_samples`,
    `energy_reconstruction_blocks.n_samples`) and the rate achieved
    (`energy_sampling_hz`, `energy_reconstruction_sampling_hz`).
  - Jetson energy measured by earlier versions rests on these two-point
    readings and should be re-measured. `tests/test_jetson_energy.py` (8/8).
- **Hadamard Linear returns the image's scale.**
  - It now computes Hᵀy / N (N = pixels), the pseudoinverse of any subset of
    Sylvester rows. Before, it divided by the number of patterns M, which
    scaled the output by N / M, contrary to its own derivation.
  - The network input was unaffected, since it is min-max normalised. The
    exported `normalization` constants (`lo`, `span`) of Hadamard tests are now
    on the new scale, and `test_config.json` says so
    (`linear_divisor: "pixels"`).
- **Omitted keys in a `.batch_config` now take the GUI defaults.** When a test
  entry left out `reconstruction_method` or `timing_warmup_runs`, loading gave
  the pseudoinverse and 5 warm-up runs instead of the native method and 20.
  Configs saved by the GUI always contain both keys, so no saved experiment
  changes.
- **Re-measure loads U-Nets saved before 2026-05-16.** The U-Net decoder
  changed from `ConvTranspose2d` to `Upsample` + `Conv2d` (4eb6f44), after
  which re-measure could not load the U-Net weights of older reports, Use
  Case 2 included. Such checkpoints are now detected and rebuilt with
  `UNetTranspose`, the previous decoder, which reproduces their stored
  outputs.
- **RAPL idle readings across a counter wrap.** The instantaneous-power read
  used by the idle baseline ignored the counter wrap and divided by the
  nominal 0.1 s instead of the measured interval.
- **FISTA and TV parameters now reach the solvers.** The batch runner set
  attributes the solvers never read, so every FISTA run used lambda = 0.001
  and 500 iterations and every TV run lambda = 0.1 and 100 iterations,
  whatever the batch configuration said; Single Test had no controls at all.
  The values are now passed to the solvers in Batch Test, Single Test (new
  *Lambda* / *Iterations* controls in every mask panel, shown for FISTA and
  TV) and re-measure. The defaults are the values that always ran, so default
  runs reproduce earlier results. Batch configs move to format 2.2: older
  configs stored values that were never applied, so they load with the
  historical values and log a warning. Batch reports record the parameters a
  test used (`reconstruction_params`), and re-measure applies them.
- **Re-measure timed TV tests with the wrong algorithm.** Batch reports store
  the display name of the method (`"TV Norm"`), which the pipeline did not
  recognise as `tv_norm`, so re-measure silently rebuilt the native linear
  applicator. Timing and energy of re-measured TV tests produced by earlier
  versions describe the linear method, not TV, and should be re-measured.
- **Re-measure dropped the reconstruction energy.** It now measures it again
  on the device of the pass. When no energy counter is readable for that
  device (for example RAPL before `chmod a+r`), reports say so in
  `energy_reconstruction_error` instead of omitting the field silently.

## [1.2.1] - 2026-09-14

### Added
- **Train-split calibration set for INT8 post-training quantization.**
  Batch Test exports that include datasets now write
  `data/<test>/calibration_images.npz` next to the existing files.
  Post-training quantization (e.g. Xilinx DPU, TensorRT) needs a few
  hundred representative inputs to observe activation ranges, and those
  inputs must come from the train split: calibrating on test frames
  leaks the evaluation distribution into the quantized model and
  inflates the PSNR reported afterwards. The file contains
  `reconstructions` (float32, N x 64 x 64, same normalization as
  `test_images.npz`), `originals` (float32, N x 64 x 64) and `indices`
  (int64, N, train-split frame indices, disjoint from the test split).
  N is capped at 200 frames (`CALIBRATION_LIMIT` in `runner.py`). The
  model is not run over the train split, so the extra export cost is
  small. A failure while collecting the set is logged as a warning and
  does not abort the batch or the other export files.
- **`PostprocessorNN.train_dataset(limit=None)`** returns
  `(orig, recon)` from the train split, in a fixed order that matches
  the new `train_indices` attribute. It does not consume the global RNG,
  so training results are unaffected.

### Changed
- **`test_config.json` carries deployment metadata.** New keys:
  `M_real` and `M_over_N_real` (the effective number of patterns, which
  for some mask geometries differs from the nominal setting),
  `mask_polarity` (`bipolar` or `binary`; a bipolar pattern needs two
  complementary shots on a binary DMD, which doubles the acquisition
  time per pattern) and `normalization` (`lo`, `span`, to reproduce the
  reconstruction scaling outside ASPIR). They come after the existing
  keys, whose order and values are unchanged, but the file's hash
  differs from v1.2.0 exports. `test_images.npz` and `masks.npz` are
  byte-identical to v1.2.0 output for the same configuration and seed.

## [1.2.0] - 2026-05-26

### Changed
- **Relicensed from CC BY 4.0 to Apache License 2.0.** CC BY 4.0 is a
  content licence that Creative Commons itself advises against for
  software (no patent grant, no notion of source vs object form).
  Apache 2.0 is a purpose-built OSI-approved software licence with an
  explicit patent grant, making downstream/industrial reuse
  unambiguous. The `LICENSE` file now carries the full Apache 2.0
  text; new `NOTICE` (attribution) and `THIRD_PARTY_LICENSES.md`
  (dependency licences, flagging the LGPL-3.0 status of PySide6,
  pylops and pyproximal, and the AGPL-3.0 of the optional Jetson
  `jetson-stats` extra) files were added. License metadata updated in
  `pyproject.toml` (now with the `License :: OSI Approved :: Apache
  Software License` trove classifier), `CITATION.cff`, the README, the
  Read the Docs pages and the in-app About dialog. Documentation and
  figures are covered by the same Apache 2.0 licence as the code.

### Added
- **U-Net-Residual (ResUNet) post-processing model.** New `UNetRes` architecture
  in `simulation_engine/_4_postprocessor/models/unet_res.py`, registered under
  the canonical key `u-net-residual` (display name *U-Net-Residual*). Mirrors
  the vanilla U-Net structurally but replaces the channel-axis concat skip
  connections with residual additions (Zhang et al. 2018). Motivated by FINN
  v0.10.1 FPGA deployment, which cannot map channel-axis Concat to HW
  (FINN issue #329); add-based skips are the only skip pattern with native HW
  support there. Available in both Single Test and Batch Test mode, and the
  Architecture Configuration panel exposes the `features` (encoder channel
  widths) parameter. At identical widths the parameter count is ≈10 % lower
  than the vanilla U-Net.
- **Idle-baseline capture for energy measurements.** Both Batch Test
  and Re-measure now sample system idle power for a configurable
  window (default 60 s, range 30–300) before the first test starts.
  The mean is persisted at report-metadata level under
  `idle_baseline` (per-backend on hosts with multiple energy
  backends) and used to derive three new per-test columns:
  `dynamic_power_W`, `dynamic_energy_mj`, and
  `dynamic_efficiency_imgs_per_J`. Negative dynamics are intentionally
  *not* clipped to zero — they're a real diagnostic signal that the
  rail warmed up between baseline and test, surfaced as a WARNING in
  the log. The Energy view gains a "Subtract idle baseline" toggle
  that swaps total power/energy for their dynamic equivalents in all
  charts; the toggle stays disabled until at least one loaded
  experiment carries a baseline. Capture is opt-in via a checkbox in
  both the Batch Test panel and the Re-measure dialog — turning it
  off only blanks the new dynamic columns; every existing
  total-energy/total-power column in the report stays unchanged.
- **Re-measure timing & energy on already-executed batches.** Right-click any
  loaded experiment in Batch Reports → *Re-measure timing & energy…* opens a
  dialog that re-runs inference on the saved model checkpoints to refresh
  timing and energy on the current host (typical use: train on a workstation,
  measure on a Jetson Orin NX). A copy of the report is written with a
  `_reexecuted_<device>_<timestamp>` suffix and auto-loaded next to the
  original — never overwrites the source data.
- **Run both compute paths** toggle in the re-measure dialog. Sequences a
  CPU pass (`use_gpu=False`) and a GPU pass (`use_gpu=True`) with a 30 s
  thermal cooldown between them and tags each output report with `-cpu` /
  `-gpu`. Designed around the Jetson shared-rail constraint where the only
  honest CPU vs GPU energy comparison is two separate runs.
- **PSNR vs Sampling Ratio** chart in Batch Reports → Quality. Line +
  markers, real dB on Y, real `M/N` (%) on X, ±1σ shading from per-image
  data when available — the headline figure for compressive-sensing
  ablations. Defaults to matplotlib's classic `tab:blue` (reconstruction)
  + `tab:orange` (denoised); both colours are user-overridable from the
  chart-settings dialog.
- **Energy per Image vs Sampling Ratio** chart in Batch Reports → Timing.
  Companion to the PSNR chart with the same X axis, two lines per
  experiment (CPU run vs GPU run), automatic switch to a log Y when one
  series is more than 5× the other.
- **Quality Metrics Table** mode with right-click *Export CSV* / *Export
  LaTeX* for the per-test summary.
- **Right-click "Copy table"** on the Timing Summary and Energy Summary
  panels, producing TSV pasteable into spreadsheets.
- **Chart configuration knobs** for bar value font size, secondary X-tier
  label font size, X / Y label padding (range 0–120 pt), and per-series
  colours of the new line charts. Dialog grows automatically when more
  fields are added.
- Jetson installation instructions for the `jetson-stats` (`jtop`) daemon
  and CUPTI privileges; a `[jetson]` extra in `pyproject.toml` pins
  `torch == 2.8` and `numpy < 2` for JetPack 6.x.
- Initial documentation with Read the Docs.

### Changed
- **Migrated from PyQt5 to PySide6 (Qt6).** PyQt5 is GPLv3-only, which
  conflicted with ASPIR's CC BY 4.0 license; PySide6 is LGPLv3 and
  compatible. Also bumps the Qt runtime from 5.15 to 6.6+ (automatic
  HiDPI scaling, scoped enums available, `QAction` and `QShortcut`
  moved from `QtWidgets` to `QtGui`, `exec_()` renamed to `exec()`).
  No public API changes; all GUI flows behave identically. The Jetson
  install recipe drops the `apt install python3-pyqt5` +
  `--system-site-packages` workaround — PySide6 ships official
  aarch64 wheels on PyPI from 6.5+.
- **Author + affiliation metadata** synced with the Intelligent
  Computing paper draft: hyphenated surname *Chabert-Ull* applied
  consistently, GROC and HPC&A research groups added to all
  affiliations in `CITATION.cff`, BibTeX entry in `docs/citation.md`
  expanded to all five authors, About dialog lists every author with
  superscript markers and both institutional blocks. The dialog
  itself was reflowed (logo + title side-by-side, 700 × 700 min
  size) so the new footer fits.
- **README License section** filled in with the canonical CC BY 4.0
  badge + attribution (was previously a one-line stub).
- **Installation docs** call out the `torch>=2.12` / CUDA 13 trap
  introduced when PyPI made cu130 the default wheel, and list
  `libxcb-cursor0` as a hard Linux runtime dep of Qt 6 ≥ 6.5.
- **Batch Reports → Energy** *Backend* selector renamed to *Compute path*.
  Options are now `CPU run + GPU run`, `CPU run only`, `GPU run only`;
  bars are bucketed by the test's `use_gpu` flag rather than by
  `energy_cpu_mj` vs `energy_gpu_mj` — the previous bucketing showed two
  identical-height bars per test on Jetson because the rail is shared.
- **Pipeline Latency Breakdown** chart legend renames *Acquisition* →
  *Mask projection* (and the Timing Summary row to match). The on-disk
  field name `timing_acquisition_ms` is preserved for backward
  compatibility.
- **Quality Metrics Comparison** bar chart switches from per-metric
  global min-max normalisation to absolute references (PSNR / 40 dB,
  SSIM raw, 1 − LPIPS). The lowest-quality bars are no longer pushed
  to height 0 and disappear.
- Energy reports on Jetson now populate both `energy_cpu_mj` and
  `energy_gpu_mj` with the shared-rail total instead of leaving the
  CPU column blank during CPU-only runs.

### Fixed
- File-name lookup mismatch in figure-export popups: tests with `%`
  (or other non-alphanumeric characters) in their name now resolve to
  the correct directory under `data/<test>/` for `test_images.npz` and
  `masks.npz` lookups.
- `EnergyAnalyzer` no longer reports `energy_std_mj = 0` for every test:
  measurement runs are split into 10 sub-blocks so `np.std` has more
  than one sample to work with, and the CPU/GPU component aggregation
  accumulates across all sub-runs instead of only the last.
- Re-measurement on Jetson stays responsive: per-block progress
  surfaces at INFO level, the `EnergyAnalyzer` is shared across all
  tests of a pass (one `jtop()` connection instead of N), and
  reconstruction timing samples are capped at 3 per test to bound the
  CPU NumPy cost at high `M/N`.
- **Docker image GUI startup** on hosts with NVIDIA driver < 535: the
  Dockerfile now pins torch to the CUDA 12.4 wheel index (two-step
  install with `--upgrade-strategy only-if-needed`) instead of letting
  pip default to the CUDA 13 wheels that fail with "driver too old"
  on the 12.4 runtime base. Also adds `libegl1`, `libopengl0` and
  `libdbus-1-3` to the apt deps so `from PySide6.QtWidgets import
  QApplication` no longer aborts with `libEGL.so.1: cannot open
  shared object file`.

## [1.0.0] - 2026-04-26

### Added
- Complete SPI simulation pipeline
- Multiple mask pattern generators (Scatter, Hadamard, Sweep, Fourier)
- Classical reconstruction algorithms (Ghost Imaging, Pseudoinverse, FISTA, TV-Norm)
- Neural network post-processing (U-Net, DnCNN, Autoencoder, etc.)
- Quality metrics analysis (PSNR, SSIM, LPIPS)
- Timing and energy profiling
- Batch experiment mode
- Batch reports with visualization
- Docker support for easy deployment

### Changed
- N/A (initial release)

### Deprecated
- N/A (initial release)

### Removed
- N/A (initial release)

### Fixed
- N/A (initial release)

### Security
- N/A (initial release)
