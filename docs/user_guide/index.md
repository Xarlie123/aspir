# User Guide

This guide covers all features of ASPIR in detail.

## Pipeline Overview

ASPIR implements a complete Single Pixel Imaging pipeline:

```{mermaid}
flowchart LR
    D[Dataset Generator] --> M[Mask Generator]
    M --> A[Applicator / Reconstruction]
    A --> N[Neural Network]
    N --> Q[Quality Metrics]
    N --> T[Timing Analysis]
    N --> E[Energy Analysis]
```

## Application Modes

ASPIR has three main modes:

### Single Test Mode

Interactive experimentation with immediate visual feedback. Use this to:
- Explore different configurations
- Visualize intermediate results
- Debug and understand the pipeline

### Batch Test Mode

Automated parameter sweeps. Use this to:
- Compare multiple configurations systematically
- Run experiments overnight
- Generate reproducible results

### Batch Reports Mode

Analysis and visualization of batch results. Use this to:
- Compare quality metrics across experiments
- Generate publication-ready plots
- Export results to various formats

## Chapters

```{toctree}
:maxdepth: 2

datasets
masks
reconstruction
neural_networks
analysis
batch_experiments
```
