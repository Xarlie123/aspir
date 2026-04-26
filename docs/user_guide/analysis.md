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
