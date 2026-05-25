# ASPIR Introduction

**A Single-Pixel Imaging Research Platform**

ASPIR is an open-source project developed in Python designed to bring the world of Single-Pixel Imaging (SPI) and Artificial Intelligence (AI) closer to researchers and students, breaking down the programming barrier. The software implements an end-to-end pipeline for testing denoising algorithms from A to Z.

ASPIR allows users to import or generate datasets, create mask patterns, select reconstruction algorithms, and apply post-processing models based on neural networks. The application is built on Python and the PySide6 library for its graphical user interface.

## Summary

The ASPIR pipeline covers every stage of a Single-Pixel Imaging experiment. Users can **generate or import datasets** from individual images, folders, infrared beam simulations (LightPipes), or well-known benchmarks such as CelebA and SVHN.

Once a dataset is loaded, ASPIR provides several **mask pattern generators** — Scatter, Hadamard (with variants like Cake-Cutting and Walsh-Paley), Sweep, and Fourier — that simulate the spatial light modulator encoding used in real SPI setups.

The acquired measurements are then passed through **classical reconstruction algorithms** including Ghost Imaging, Pseudoinverse, FISTA, and TV-Norm, each offering different trade-offs between speed, noise robustness, and compression tolerance.

To further improve image quality, ASPIR integrates a **neural network post-processing** stage with ten architectures ready to use: U-Net, U-Net-Residual, U-Net with Residual Attention, DnCNN, Autoencoder, Residual CNN, MobileNet Denoising, Dilated CNN, cGAN, and Noise2Void. Models can be trained, evaluated, and exported directly from the GUI.

Finally, a comprehensive **analysis module** computes quality metrics (PSNR, SSIM, LPIPS), timing breakdowns per pipeline stage, and energy consumption profiling on NVIDIA GPUs (desktop and Jetson) and Intel CPUs via RAPL.

For large-scale studies, the **batch experiment** mode allows users to define parameter sweeps, run them sequentially or in parallel, and generate comparative reports in HTML, PDF, LaTeX, or CSV.

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
```

```{toctree}
:maxdepth: 2
:caption: Reference

api/index
acknowledgements
citation
changelog
```

## License & Citation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19864089.svg)](https://doi.org/10.5281/zenodo.19864089)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

The DOI above is the **concept DOI** — it always points at the latest archived version on Zenodo. To cite a specific release, follow the link and pick the version-specific DOI from the right-hand sidebar. See {doc}`citation` for the full BibTeX entries.

This is a research project developed at the Institute of New Imaging Technologies (INIT) and the Department of Computer Engineering and Computer Science, Universitat Jaume I, Spain. Released under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license.

## Indices

- {ref}`genindex`
- {ref}`search`
