# Quick Start

This guide walks you through your first experiments in ASPIR.

## Operating Modes

ASPIR offers two main modes of operation:

- **Single Test**: Run a single experiment with a specific set of parameters. Ideal for exploring the pipeline, fine-tuning settings, and understanding how each stage works.
- **Batch Test**: Define and run multiple experiments with different parameter combinations. Designed for systematic comparisons and parameter sweeps.

## Single Test Pipeline

The Single Test mode follows this pipeline:

```{mermaid}
flowchart LR
    A[Dataset] --> B[Mask Patterns]
    B --> C[Reconstruction]
    C --> D[Neural Network]
    D --> E[Analysis]
```

```{toctree}
:maxdepth: 1
:caption: Single Test Steps

quickstart/step1_dataset
quickstart/step2_masks
quickstart/step3_reconstruction
quickstart/step4_neural_network
quickstart/step5_analysis
quickstart/step6_saving
```

## Batch Test Pipeline

```{toctree}
:maxdepth: 1
:caption: Batch Test

quickstart/batch_test
```

## Next Steps

- Try different mask types (Hadamard for structured sampling)
- Experiment with reconstruction algorithms
- Compare neural network architectures
- See the {doc}`user_guide/index` for detailed documentation of each component
