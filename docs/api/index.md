# API Reference

Technical reference for developers integrating ASPIR components.

## Module Overview

```
simulation_engine/
├── _1_dataset_gen/      # Dataset generators
├── _2_mask_gen/         # Mask pattern generators
├── _3_applicator/       # Reconstruction algorithms
├── _4_postprocessor/    # Neural network models
├── _5_analyzer/         # Metrics and profiling
└── _6_pipeline/         # Batch execution
```

## Core Classes

### Simulation (Pipeline Orchestrator)

```python
from simulation_engine.simulation import Simulation

sim = Simulation(logger)
sim.set_dataset(dataset)
# reconstruction_method: "native" (default), "pseudoinverse", "fista" or
# "tv_norm". "native" picks the mask-specific algorithm — Ghost Imaging
# (Scatter), Sweep Linear (Sweep) or Hadamard Linear (Hadamard / Cal-Sal).
sim.set_mask(mask, reconstruction_method='fista')
sim.set_postprocessor(
    dataset, mask, applicator,
    model_name='u-net', batch_size=16, lr=1e-3,
    use_gpu=True, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
)
sim.set_analyzer()
```

### Dataset Generators

All datasets inherit from `DatasetABC` and implement `load_data()`.

```python
from simulation_engine._1_dataset_gen.DatasetFromImage import DatasetFromImage
from simulation_engine._1_dataset_gen.DatasetFromFolder import DatasetFromFolder

# Single image
dataset = DatasetFromImage(img_size=64, img_path="image.png", data_format="FP32")
dataset.load_data()

# Multiple images from folder
dataset = DatasetFromFolder(img_size=64, folder_path="images/", data_format="FP32")
dataset.load_data()
```

### Mask Generators

All masks inherit from `MaskABC` and implement `generate_masks()`.

```python
from simulation_engine._2_mask_gen.mask_scatter import MaskScatter
from simulation_engine._2_mask_gen.mask_hadamard import MaskHadamard
from simulation_engine._2_mask_gen.mask_sweep import MaskSweep

# Random scatter
mask = MaskScatter(img_size=64, point_density=0.1, num_patterns=512, seed=42)
mask.generate_masks()

# Hadamard (subset of patterns)
mask = MaskHadamard(img_size=64, min_idx=0, max_idx=1024)
mask.generate_masks()

# Sweep (list of angle/bar_width/stride dicts)
mask = MaskSweep(img_size=64, parametros=[
    {'angle': 0.0, 'bar_width': 2, 'stride': 4},
    {'angle': 90.0, 'bar_width': 2, 'stride': 4},
])
mask.generate_masks()
```

### Applicators (Reconstruction)

All applicators inherit from `ApplicatorABC` and implement `process_dataset()`.
The mask-specific applicators (`ApplicatorScatter`, `ApplicatorSweep`,
`ApplicatorHadamard`) implement their native algorithm; the generic
iterative solvers (`ApplicatorPseudoinverse`, `ApplicatorFISTA`,
`ApplicatorTV`) work with any mask that exposes a sensing matrix in
`mask.masks`.

```python
# Iterative reconstruction (works with Scatter, Sweep or Hadamard masks)
from simulation_engine._3_applicator.applicator_fista import ApplicatorFISTA

applicator = ApplicatorFISTA(dataset, mask, maxit=500, lam=1e-3)
results_df = applicator.process_dataset()

# Or let Simulation pick the applicator by method name:
sim.set_mask(mask, reconstruction_method='tv_norm')
```

### Neural Network Models

```python
from simulation_engine._4_postprocessor.postprocessor_nn import PostprocessorNN

postprocessor = PostprocessorNN(
    model_name='u-net',
    model_overrides={'features': [8, 16, 32, 64]},
    dataset=dataset,
    applicator=applicator,
    batch_size=16,
    lr=1e-3,
    loss_function='mse',
    optimizer_name='adam',
    use_gpu=True,
    train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
)
postprocessor.train(num_epochs=50)
test_originals, test_noisy, test_denoised = postprocessor.test_dataset()
```

### Analyzers

```python
from simulation_engine._5_analyzer.analyzer_noise import NoiseAnalyzer
from simulation_engine._5_analyzer.analyzer_timing import TimingAnalyzer

# Quality metrics (PSNR, SSIM, LPIPS)
noise_analyzer = NoiseAnalyzer(originals, noisy, reconstructions)
noise_analyzer.analyze()
summary = noise_analyzer.get_metrics_summary()

# Timing analysis (defaults are warmup=20 / measurement=800 — same
# numbers used by the Single Test, Batch Test and Re-measure GUIs)
timing = TimingAnalyzer(model, device='cuda', warmup_runs=20)
mean_time, std_time = timing.time_inference(input_tensor, n_runs=800)
```

## Data Structures

### Image Arrays

```python
# Dataset images: (N, H, W) float32 in [0, 1]
dataset.datos.shape  # (100, 64, 64)

# Mask patterns: (M, H, W)
mask.mascaras.shape  # (512, 64, 64)
```
