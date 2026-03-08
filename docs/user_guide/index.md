# User Guide

This guide covers all features of ASPIR in detail.

## Pipeline Overview

ASPIR implements a complete Single Pixel Imaging pipeline:

```{mermaid}
flowchart TB
    subgraph Input
        D[Dataset Generator]
    end

    subgraph "Mask & Measurement"
        M[Mask Generator]
        A[Applicator]
    end

    subgraph "Reconstruction"
        R[Classical Algorithm]
        N[Neural Network]
    end

    subgraph "Analysis"
        Q[Quality Metrics]
        T[Timing Analysis]
        E[Energy Analysis]
    end

    D --> M
    M --> A
    A --> R
    R --> N
    N --> Q
    N --> T
    N --> E
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
