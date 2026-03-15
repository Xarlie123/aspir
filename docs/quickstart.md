# Quick Start

This guide walks you through your first experiments in ASPIR. The application offers two main modes of operation:

- **Single Test**: Run a single experiment with a specific set of parameters. Ideal for exploring the pipeline, fine-tuning settings, and understanding how each stage works.
- **Batch Test**: Define and run multiple experiments with different parameter combinations. Designed for systematic comparisons and parameter sweeps.
- **Batch Reports**: Explore offline the results of previously executed batch experiments.

## Single Test Pipeline

```{note}
Recommended for testing and exploring the different configuration parameters of the pipeline.
```

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

```{note}
Once you are familiar with the dataset and neural network parameters, the Batch Test mode is useful for comparing different sampling conditions or experimental setups.
```

The Batch Test mode follows this pipeline:

```{raw} html
<div style="max-width: 350px;">
```

```{mermaid}
flowchart LR
    A[Configure Tests] --> B[Run Batch]
```

```{raw} html
</div>
```

```{toctree}
:maxdepth: 1
:caption: Batch Test Steps

quickstart/batch_step1_configure
quickstart/batch_step2_run
quickstart/batch_step3_results
```

## Batch Reports

```{note}
Since training multiple experiments can be time-consuming, Batch Reports mode allows you to explore offline the results of previously executed batch experiments, loading and analyzing data without having to re-run them.
```

```{raw} html
<div style="max-width: 350px;">
```

```{mermaid}
flowchart LR
    A[Load Batch Results] --> B[Compare & Export]
```

```{raw} html
</div>
```

```{toctree}
:maxdepth: 1
:caption: Batch Reports

quickstart/batch_reports
```
