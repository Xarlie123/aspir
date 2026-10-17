# Analysis Tools

ASPIR provides comprehensive analysis tools for evaluating reconstruction quality and performance.

## Quality Metrics

### What is compared (metric convention 2, ASPIR 1.3)

Every test image is scored at two levels against its reference (the clean
image, after the dataset's data-format quantisation):

| Level | Estimate | Report fields |
|-------|----------|---------------|
| Linear reconstruction | the reconstruction algorithm's output **on the image's scale** | `psnr_recons`, `ssim_recons`, `lpips_recons` |
| Post-processed | the network output | `psnr_denoised`, `ssim_denoised`, `lpips_denoised` |

- **Clipping and data range.** Estimates are clipped to [0, 1], and PSNR and SSIM use a data range of 1 for
  every image. Values are the mean of the per-image values.
- **Image scale.** Hadamard Linear, Pseudoinverse, FISTA, TV-Norm and Fourier Inverse return images on the
  reference's scale. Ghost Imaging and Sweep Linear are correlation estimators with no natural scale. Their
  output is scored after one gain and offset fitted by least squares on the **training** split, recorded as
  `linear_calibration`.
- **Network input.** The network does not see the linear output directly: it is min-max normalised with
  the minimum and maximum of the **training** images (since 1.3; before, of the whole dataset) and, for
  the INT8 / INT4 formats, quantised on the fixed [0, 1] grid. Its PSNR and SSIM are kept apart as
  `psnr_network_input` and `ssim_network_input`.
- **The convention is recorded** as `metric_convention: 2` in the report metadata and in every result.
  Batch Reports warns when loaded reports mix conventions.

**Convention 1 (before 1.3)** differs in three ways:
- the data range was each reference image's own max − min;
- estimates were not clipped;
- the "reconstructed" level scored the network input instead of the reconstruction.

On a CelebA Cake-Cutting task this put the linear PSNR 4.9–9.7 dB lower and the network gain correspondingly
higher (`benchmarks/metric_convention/`). Quality numbers are only comparable within one convention; timing
and energy are not affected.

### PSNR (Peak Signal-to-Noise Ratio)

Measures pixel-wise reconstruction accuracy.

$$
PSNR = 10 \cdot \log_{10}\left(\frac{MAX^2}{MSE}\right), \qquad MAX = 1
$$

- **Interpretation**: Higher is better
- **Good value**: > 25 dB
- **Excellent value**: > 30 dB

### SSIM (Structural Similarity Index)

Measures structural similarity between images.

- **Range**: [-1, 1] (typically [0, 1])
- **Interpretation**: Higher is better
- **Good value**: > 0.8
- **Excellent value**: > 0.9

### LPIPS (Learned Perceptual Image Patch Similarity)

Deep learning-based perceptual similarity ([reference](https://github.com/richzhang/PerceptualSimilarity)).

- **Range**: [0, 1]
- **Interpretation**: Lower is better
- **Good value**: < 0.2
- **Excellent value**: < 0.1

## Timing Analysis

Measures inference performance on CPU and GPU.

### Pipeline Stages

| Stage | Description |
|-------|-------------|
| Mask projection | Simulated single-pixel acquisition time, derived from the number of patterns and the configured DMD sampling rate (kHz). The Pipeline Latency Breakdown chart in Batch Reports labels this column "Mask projection"; the underlying field name in the report (`timing_acquisition_ms`) is preserved for backward compatibility. |
| Reconstruction | Classical algorithm time (Ghost Imaging, Hadamard linear, FISTA, …). On Jetson with NumPy this scales roughly linearly with pattern count and can dominate wall time at high `M/N`. |
| Inference | Neural network forward pass. Measured separately on CPU and GPU when both devices are available. |

### Configuration

- **Warmup runs**: Iterations before measurement (NVIDIA CUDA initialization)
- **Measurement runs**: Averaged for stable statistics

### Proper GPU Timing

ASPIR uses correct GPU timing methodology:

```python
torch.cuda.synchronize()  # Wait for GPU
t0 = time.perf_counter()
output = model(input)
torch.cuda.synchronize()  # Wait for completion
t1 = time.perf_counter()
```

## Energy Analysis

Measures power consumption during inference. The energy backend is auto-detected based on the available hardware.

### Supported Hardware

| Platform | Backend | Measurements |
|----------|---------|--------------|
| NVIDIA Desktop GPUs | NVML (`pynvml`) | GPU package power, temperature |
| NVIDIA Jetson | INA3221 read from sysfs at 200 Hz, with jtop (`jetson-stats` daemon) as fallback | Module input power on the shared `VDD_IN` / `POM_5V_IN` rail |
| Intel and AMD CPUs | RAPL (`/sys/class/powercap/intel-rapl`, which Linux also uses for AMD Zen) | Package energy (top-level zones only; sub-zones such as `core` lie inside the package and are not added again) |

```{note}
RAPL energy profiling requires read access to the powercap interface: `sudo chmod -R a+r /sys/class/powercap/intel-rapl/`. The permission resets at every reboot.
```

```{note}
**Jetson sampling.** While a measurement is open, a background thread reads
the voltage and current of the module input rail from the INA3221 files in
sysfs at 200 Hz and integrates the power with the trapezoidal rule. Each
block stores its sample count (`energy_blocks.n_samples`) and the report the
rate achieved (`energy_sampling_hz`). The thread is paused outside
measurements, so the timing pass runs without it. jtop refreshes its reading
about once a second, too slowly for blocks shorter than a second; it is used
only when the INA3221 files cannot be read.
```

```{note}
**Jetson shared rail.** Tegra modules expose a single combined power rail
that feeds CPU + GPU + RAM + the rest of the SoC together. The sub-rails
(`VDD_CPU_GPU_CV`, `VDD_SOC` on an Orin NX) lie inside it and are not added.
ASPIR integrates the total and populates both `energy_cpu_mj` and `energy_gpu_mj` with
that total so any downstream view that gates on either field stays
non-empty. The honest comparison on Jetson is therefore *CPU run vs GPU
run* — i.e. two separate measurement passes with `use_gpu=False` and
`use_gpu=True` — rather than a per-rail breakdown of a single pass. The
Energy view's *Compute path* selector and the **Run both compute paths**
toggle in Batch Reports → Re-measure are designed around this.
```

### Metrics

- **Energy per image** (mJ): Total energy consumed
- **Average power** (W): Mean power draw
- **Efficiency** (images/J): Throughput per energy unit
- **Temperature** (°C): GPU temperature (if available)

### Idle Baseline & Dynamic Energy

Both Batch Test and Re-measure can optionally sample **system idle power** for a configurable window (30–300 s, default 60 s) before the first test. Backends are polled once a second; on a Jetson each poll is the mean of the 200 Hz samples of the past second. The mean of that window is persisted at report metadata level under `idle_baseline` and used to derive three additional per-test columns:

- `dynamic_power_W` = `energy_mean_watts` − `baseline_power_W`
- `dynamic_energy_mj` = `energy_mean_mj` − `baseline_power_W` · *t* · 1000  (where *t* is the per-image inference time from the same energy phase, so the subtraction is consistent with the integration window)
- `dynamic_efficiency_imgs_per_J` = 1000 / `dynamic_energy_mj` (or `None` when the dynamic energy is non-positive)

The intent is to remove the constant pedestal (GUI, background services, idle GPU, RAM refresh) from each measurement so the reported number reflects the actual cost of running inference. The Energy view exposes a *Subtract idle baseline* toggle that switches charts to the dynamic equivalents at display time; the summary panel shows the totals and the dynamic rows side by side when both are available.

Negative dynamic values are intentionally not clipped to zero — a negative number means the SoC was warmer during the test than during the baseline window, and that's a real diagnostic signal worth surfacing.

### Uncertainty of the Dynamic Energy

When most of the energy per image is idle draw (about 87 % on the Jetson GPU path), the dynamic energy is the small difference of two large numbers, so its uncertainty matters. ASPIR reports it next to the totals, for inference and for the reconstruction stage.

**Inputs.** The measurement is split into *B* blocks: inference runs 800 iterations as 10 blocks, and the reconstruction stage runs 10 blocks of about 0.5 s each. Block *b* gives a mean power *P_b* and a time per image *t_b*; the per-image values of every block are stored under `energy_blocks` (`energy_reconstruction_blocks`). The idle baseline gives *P*_idle ± σ_idle from *n* readings, taken only over the backends that took the measurement. For example, a reconstruction on the CPU subtracts only the CPU's idle power.

**Formula.** With *P̄* and *t̄* the means over the blocks,

$$E_\text{dyn} = (\bar P - P_\text{idle})\,\bar t .$$

**Uncertainty components.** Two independent components are reported separately and combined in quadrature:

- **Idle baseline** (`dynamic_energy_sigma_idle_mj`): σ_idle · *t̄*, with σ_idle the *standard deviation* of the idle readings. It is not the standard error σ_idle/√*n*, because the concern is the baseline drifting between the idle window and the run, and the standard error hides that. The standard-error version is stored as `dynamic_energy_sigma_idle_sem_mj` for reference.
- **Blocks** (`dynamic_energy_sigma_blocks_mj`): first-order propagation of the sampling error of *P̄* and *t̄*, with sample (co)variances over the blocks:

$$\sigma_\text{blocks}^2 = \frac{1}{B}\left[\bar t^{\,2} s_P^2 + (\bar P - P_\text{idle})^2 s_t^2 + 2\,\bar t\,(\bar P - P_\text{idle})\,s_{Pt}\right].$$

- **Combined** (`dynamic_energy_sigma_mj`): $\sqrt{\sigma_\text{idle}^2\bar t^{\,2} + \sigma_\text{blocks}^2}$.

**95 % interval.** `dynamic_energy_ci95_mj` comes from a bootstrap with 2000 draws and a fixed seed. Each draw resamples the blocks with replacement and draws *P*_idle from N(*P*_idle, σ_idle).

**Reconstruction fields.** The reconstruction stage carries the same fields with `dynamic_reconstruction_` and `energy_reconstruction_` prefixes.

**Platform state.** Timing and energy depend on how the machine is set up, so the report metadata also stores `platform_state`, a snapshot at the start and at the end of the run: frequency governor and limits of each CPU cluster and whether they are pinned, the Jetson power mode (`nvpmodel`), GPU clock and fan, the desktop GPU's clocks (`nvidia-smi`), the temperatures, and the torch version and thread count. On a Jetson, pin the clocks with `sudo jetson_clocks` before measuring; `platform_state` then shows `"pinned": true` for the CPU and the GPU at both ends of the run.

**Report metadata.** Each backend's entry in `idle_baseline` stores `P_idle_mean_w`, `P_idle_std_w` (sample standard deviation), `P_idle_sem_w` and `n_idle_samples`, and so does the batch total. Independent sensors add their standard deviations in quadrature.

**Relation to the historical field.** `dynamic_energy_mj` keeps its historical definition, *Ē* − *P*_idle · (*Ē*/*P̄*). It differs from `dynamic_energy_pt_mj` = (*P̄* − *P*_idle) · *t̄* only by how the blocks are averaged. The uncertainty fields refer to the latter, which is also the formula of the paper's analysis.

### Inference Batch Size

The Re-measure dialog exposes an **Inference batch size** spin box (range 1–128, default 1). It controls the shape of the sample tensor fed into the model on each forward pass: `(B, 1, H, W)` on conv models, `(B, H·W)` otherwise. The reported `*_mean_ms` and `*_mean_mj` columns stay **per-image** regardless of B (the measurement primitives divide by the batch internally), and a `timing_batch_size` / `energy_batch_size` field is added to the report so the reader can tell which point of the curve the numbers came from. Use B = 1 for single-image latency, B ≥ 8 to characterise throughput-oriented deployment.

## PyTorch Profiler

Deep analysis of neural network operations using the [PyTorch Profiler](https://pytorch.org/docs/stable/profiler.html).

### Features

- Layer-by-layer breakdown
- Operation categorization (Conv, BatchNorm, etc.)
- CPU vs GPU time comparison
- Memory usage

### NVIDIA Nsight Systems Integration

For detailed GPU memory transfer and kernel analysis:

1. Click "Generate Nsight Profile Script"
2. Run generated script with `nsys profile`
3. Open `.nsys-rep` file in NVIDIA Nsight Systems GUI

## Exporting Results

All analysis results can be exported as:

- **CSV**: Raw data for further analysis
- **JSON**: Structured metadata
- **PNG/PDF**: Charts and visualizations
