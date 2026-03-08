# Batch Experiments

Run systematic parameter sweeps and compare multiple configurations.

## Batch Test Mode

### Creating Configurations

1. Switch to **Batch Test** mode
2. Add test configurations using **Add Test**
3. Configure each test:
   - Mask type and parameters
   - Reconstruction method
   - Neural network architecture
   - Training hyperparameters

### Configuration Parameters

Each test configuration includes:

| Category | Parameters |
|----------|------------|
| Mask | Type, patterns, seed |
| Reconstruction | Method (Ghost/Pseudo/FISTA/TV) |
| Model | Architecture, epochs, batch size, LR |
| Dataset split | Train %, Val %, Test % |
| Reports | Quality, timing, energy |

### Running Batch Tests

1. Set **Export Name** for results identification
2. Click **Run Batch**
3. Monitor progress in status panel

Tests run sequentially by default. Enable parallel execution for independent tests.

### Saved Results

Results are saved to `experiments/<export_name>/`:

```
<export_name>/
├── .batch_analysis_report    # JSON with all metrics
└── data/
    └── <test_name>/
        ├── test_images.npz   # Images and reconstructions
        └── masks.npz         # Mask patterns
```

## Batch Reports Mode

Analyze and compare completed batch experiments.

### Loading Results

1. Switch to **Batch Reports** mode
2. Click **Load Experiment**
3. Select experiment folder

Multiple experiments can be loaded for comparison.

### Available Views

#### Summary View

Table with all tests and metrics:
- Quality: PSNR, SSIM, LPIPS
- Timing: Mean, std, min, max
- Configuration: Model, masks, reconstruction

Columns are configurable via dropdown menu.

#### Quality View

Charts comparing quality metrics across tests:
- Bar charts for PSNR/SSIM/LPIPS
- Before/after neural network comparison
- Per-image quality preview

#### Timing View

Pipeline latency breakdown:
- Stacked bars: Acquisition + Reconstruction + Inference
- CPU vs GPU comparison
- Per-test timing statistics

#### Training View

Training curves visualization:
- Loss curves (train/validation)
- Learning rate schedules
- Convergence analysis

#### Details View

Per-test detailed information:
- Full configuration dump
- Individual metrics
- Model architecture summary

### Exporting Reports

Export results to various formats:
- **HTML**: Interactive web report
- **LaTeX**: Publication-ready tables
- **PDF**: Complete report document
- **CSV**: Raw data

## Best Practices

### Experimental Design

1. **Baseline first**: Always include a baseline configuration
2. **One variable**: Change one parameter at a time for clear comparisons
3. **Seeds**: Use fixed seeds for reproducibility
4. **Statistics**: Run multiple seeds and report mean ± std

### Naming Conventions

Use descriptive names:
- `scatter_512_unet` (mask type, patterns, model)
- `hadamard_fista_50ep` (mask, reconstruction, epochs)

### Version Control

Save batch configurations (`.batch_config`) to version control for reproducibility.
