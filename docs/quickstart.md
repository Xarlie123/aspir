# Quick Start

This guide walks you through your first Single Pixel Imaging simulation in ASPIR.

## Overview

The SPI pipeline in ASPIR follows these steps:

```{mermaid}
flowchart LR
    A[Dataset] --> B[Mask Patterns]
    B --> C[Reconstruction]
    C --> D[Neural Network]
    D --> E[Analysis]
```

## Step 1: Load a Dataset

1. Launch ASPIR and ensure you're in **Single Test** mode
2. In the **Dataset** section, select **From Image**
3. Click **Browse** and select any grayscale image
4. Set the resolution (e.g., 64×64 for quick testing)
5. Click **Generate Dataset**

The preview panel shows your loaded images.

## Step 2: Generate Mask Patterns

1. Go to the **Masks** section
2. Select mask type: **Scatter** (random sampling) is a good start
3. Configure parameters:
   - **Number of patterns**: 512 (more = better quality, slower)
   - **Seed**: Any number (for reproducibility)
4. Click **Generate Masks**

The preview shows the generated mask patterns.

## Step 3: Reconstruct Images

1. In the **Reconstruction** section, select a method:
   - **Ghost Imaging**: Fast, classical method
   - **Pseudoinverse**: Matrix inversion approach
   - **FISTA**: Sparse reconstruction (slower, better quality)
   - **TV-Norm**: Edge-preserving reconstruction

2. Click **Reconstruct**

The reconstructed images appear in the preview. They will be noisy - this is expected!

## Step 4: Train a Neural Network

1. Go to the **Post-processing** section
2. Select a model: **U-Net** is recommended for beginners
3. Configure training:
   - **Epochs**: 50
   - **Batch size**: 16
   - **Learning rate**: 0.001
4. Click **Train**

Training progress shows in the status bar. The denoised results appear when complete.

## Step 5: Analyze Results

1. Go to **Reports → Quality Metrics**
2. Click **Analyze Quality**

You'll see metrics comparing:
- Original vs Reconstructed (before NN)
- Original vs Denoised (after NN)

| Metric | Description | Good Value |
|--------|-------------|------------|
| PSNR | Peak Signal-to-Noise Ratio | > 25 dB |
| SSIM | Structural Similarity | > 0.8 |
| LPIPS | Perceptual similarity | < 0.2 |

## Next Steps

- Try different mask types (Hadamard for structured sampling)
- Experiment with reconstruction algorithms
- Compare neural network architectures
- Run batch experiments for systematic comparisons

See the {doc}`user_guide/index` for detailed documentation of each component.
