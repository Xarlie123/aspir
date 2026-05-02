# Analysis Tools

ASPIR provides comprehensive analysis tools for evaluating reconstruction quality and performance.

## Quality Metrics

### PSNR (Peak Signal-to-Noise Ratio)

Measures pixel-wise reconstruction accuracy.

$$
PSNR = 10 \cdot \log_{10}\left(\frac{MAX^2}{MSE}\right)
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
| NVIDIA Jetson | jtop (`jetson-stats` daemon), with sysfs INA3221 as fallback | Module total power on the shared `VDD_IN` / `POM_5V_IN` rail |
| Intel CPUs | RAPL (`/sys/class/powercap/intel-rapl`) | Package, core power |

```{note}
Intel RAPL energy profiling requires read access to the powercap interface: `sudo chmod -R a+r /sys/class/powercap/intel-rapl/`
```

```{note}
**Jetson shared rail.** Tegra modules expose a single combined power rail
that feeds CPU + GPU + RAM + the rest of the SoC together. jtop reports
the total; ASPIR populates both `energy_cpu_mj` and `energy_gpu_mj` with
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

Both Batch Test and Re-measure can optionally sample **system idle power** for a configurable window (30–300 s, default 60 s) before the first test. The mean of that window is persisted at report metadata level under `idle_baseline` and used to derive three additional per-test columns:

- `dynamic_power_W` = `energy_mean_watts` − `baseline_power_W`
- `dynamic_energy_mj` = `energy_mean_mj` − `baseline_power_W` · *t* · 1000  (where *t* is the per-image inference time from the same energy phase, so the subtraction is consistent with the integration window)
- `dynamic_efficiency_imgs_per_J` = 1000 / `dynamic_energy_mj` (or `None` when the dynamic energy is non-positive)

The intent is to remove the constant pedestal (GUI, background services, idle GPU, RAM refresh) from each measurement so the reported number reflects the actual cost of running inference. The Energy view exposes a *Subtract idle baseline* toggle that switches charts to the dynamic equivalents at display time; the summary panel shows the totals and the dynamic rows side by side when both are available.

Negative dynamic values are intentionally not clipped to zero — a negative number means the SoC was warmer during the test than during the baseline window, and that's a real diagnostic signal worth surfacing.

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
