# Batch Experiments

For a step-by-step walkthrough of how to use Batch Test and Batch Reports, see the {doc}`../quickstart` guide.

This section covers file formats, result structure, and best practices for designing experiments.

## File Formats

### Batch Configuration (`.batch_config`)

JSON file that stores the list of test configurations. Created via `File → Save Batch Config` in Batch Test mode.

Contains:
- Experiment name and description
- Execution options (sequential/parallel, number of threads)
- List of test configurations, each with mask, reconstruction, model, and training parameters

### Batch Analysis Report (`.batch_analysis_report`)

JSON file generated after running a batch test. Contains all metrics and results for each test in the batch.

Can be loaded in **Batch Reports** mode for offline analysis and comparison.

## Results Structure

After running a batch test, results are saved to `experiments/<export_name>/`:

```
<export_name>/
├── <export_name>.batch_analysis_report   # JSON with all metrics and configuration
└── data/
    └── <test_name>/
        ├── test_images.npz               # Original, reconstructed, and denoised images
        └── masks.npz                     # Mask patterns used
```

## Best Practices

### Experimental Design

1. **Baseline first**: Always include a baseline configuration (e.g., Ghost Imaging without NN)
2. **One variable at a time**: Change one parameter per test for clear comparisons
3. **Fixed seeds**: Use the same random seed across tests for reproducibility
4. **Multiple seeds**: For statistical significance, run with different seeds and report mean ± std

### Naming Conventions

Use descriptive test names that encode key parameters:
- `scatter_512_unet` (mask type, number of patterns, model)
- `hadamard_fista_50ep` (mask, reconstruction method, epochs)
- `sweep_3bars_dncnn` (mask, configuration, model)

### Version Control

Save `.batch_config` files to version control so experiments can be reproduced exactly.
