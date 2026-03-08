# ASPIR Documentation

**A Single-Pixel Imaging Research Platform**

ASPIR is a PyQt5 application for Single Pixel Imaging (SPI) simulation and analysis. It implements a complete computational imaging pipeline for infrared beam profiling using mask patterns, classical reconstruction algorithms, and neural network post-processing.

## Features

- **Multiple mask patterns**: Scatter, Hadamard, Sweep, Fourier
- **Classical reconstruction**: Ghost Imaging, Pseudoinverse, FISTA, TV-Norm
- **Neural network denoising**: U-Net, DnCNN, Autoencoder, and more
- **Quality metrics**: PSNR, SSIM, LPIPS
- **Performance analysis**: Timing profiling, energy measurement
- **Batch experiments**: Automated parameter sweeps and comparison

## Quick Links

::::{grid} 2
:gutter: 3

:::{grid-item-card} Installation
:link: installation
:link-type: doc

Get ASPIR running on your system
:::

:::{grid-item-card} Quick Start
:link: quickstart
:link-type: doc

Your first SPI simulation in 5 minutes
:::

:::{grid-item-card} User Guide
:link: user_guide/index
:link-type: doc

Detailed documentation of all features
:::

:::{grid-item-card} API Reference
:link: api/index
:link-type: doc

Technical reference for developers
:::

::::

## Contents

```{toctree}
:maxdepth: 2
:caption: Getting Started

installation
quickstart
```

```{toctree}
:maxdepth: 2
:caption: User Guide

user_guide/index
user_guide/datasets
user_guide/masks
user_guide/reconstruction
user_guide/neural_networks
user_guide/analysis
user_guide/batch_experiments
```

```{toctree}
:maxdepth: 2
:caption: Reference

api/index
citation
changelog
```

## License

This is a research project developed at [Your Institution].

## Indices

- {ref}`genindex`
- {ref}`search`
