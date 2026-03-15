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


```{toctree}
:maxdepth: 2

datasets
masks
reconstruction
neural_networks
analysis
batch_experiments
```
