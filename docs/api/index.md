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

### Simulation

Main orchestrator for the SPI pipeline.

```python
from simulation_engine.simulation import Simulation

sim = Simulation(logger)
sim.set_dataset(dataset)
sim.set_mask(mask, reconstruction_method='fista')
sim.set_postprocessor(model_name='unet', epochs=50)
sim.set_analyzer()

metrics = sim.analyzer.get_metrics()
```

### Dataset Generators

```python
from simulation_engine._1_dataset_gen.dataset_from_image import DatasetFromImage
from simulation_engine._1_dataset_gen.dataset_from_folder import DatasetFromFolder

# Single image
dataset = DatasetFromImage(path="image.png", size=64)

# Multiple images
dataset = DatasetFromFolder(path="images/", size=64, max_images=100)
```

### Mask Generators

```python
from simulation_engine._2_mask_gen.mask_scatter import MaskScatter
from simulation_engine._2_mask_gen.mask_hadamard import MaskHadamard

# Random scatter
mask = MaskScatter(num_patterns=512, dimensions=(64, 64), seed=42)

# Hadamard
mask = MaskHadamard(size=64, variant='scrambled')
```

### Applicators (Reconstruction)

```python
from simulation_engine._3_applicator.applicator_scatter_fista import ApplicatorScatterFISTA

applicator = ApplicatorScatterFISTA(
    mask=mask,
    lambda_reg=0.1,
    max_iterations=100
)
reconstructed = applicator.apply(dataset)
```

### Neural Network Models

```python
from simulation_engine._4_postprocessor.postprocessor_nn import PostprocessorNN

postprocessor = PostprocessorNN(
    model_name='unet',
    input_channels=1,
    device='cuda'
)
postprocessor.train(train_loader, val_loader, epochs=50)
denoised = postprocessor.inference(reconstructed)
```

### Analyzers

```python
from simulation_engine._5_analyzer.analyzer_quality import AnalyzerQuality
from simulation_engine._5_analyzer.analyzer_timing import AnalyzerTiming

# Quality metrics
quality = AnalyzerQuality(original, denoised)
psnr, ssim, lpips = quality.compute_metrics()

# Timing analysis
timing = AnalyzerTiming(model, device='cuda', warmup=5, runs=20)
results = timing.measure(test_images)
```

## Data Structures

### Image Arrays

```python
# Dataset: (N, H, W) float32 in [0, 1]
dataset.images.shape  # (100, 64, 64)

# Masks: (M, H, W) binary
mask.patterns.shape  # (512, 64, 64)

# Measurements: (N, M) float32
measurements.shape  # (100, 512)
```

### Configuration Objects

```python
from dataclasses import dataclass

@dataclass
class TestConfiguration:
    mask_type: str
    num_patterns: int
    reconstruction_method: str
    model_name: str
    epochs: int
    batch_size: int
    learning_rate: float
    # ...
```

## Extending ASPIR

### Adding a New Mask Type

1. Create class in `_2_mask_gen/`:

```python
from .mask import Mask

class MaskCustom(Mask):
    def __init__(self, dimensions, **params):
        super().__init__(dimensions)
        self.params = params

    def generate_patterns(self):
        # Return numpy array (M, H, W)
        pass
```

2. Register in `simulation.py`

### Adding a New Model

1. Create model in `_4_postprocessor/models/`:

```python
import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        # Define layers

    def forward(self, x):
        # Forward pass
        return x
```

2. Register in `PostprocessorNN.MODEL_REGISTRY`
