# Changelog

All notable changes to ASPIR will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

## [1.0.0] - 2025-XX-XX

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
