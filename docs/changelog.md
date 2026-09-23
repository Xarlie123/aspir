# Changelog

All notable changes to ASPIR will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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
- **First automated tests in the repository.** `tests/test_fourier_fspi.py`
  covers full-spectrum fidelity for both phase schemes, the conjugate-symmetry
  fill, the sampling-ratio to pattern-count mapping, the mask-agnostic solvers
  and an `.npz` save/load round trip. It runs standalone
  (`.venv/bin/python tests/test_fourier_fspi.py`) with no new dependency.

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
