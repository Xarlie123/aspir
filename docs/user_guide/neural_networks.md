# Neural Networks

ASPIR includes several neural network architectures for post-processing reconstructed images.

## Available Models

### U-Net

Encoder-decoder architecture with skip connections.

- **Best for**: General denoising, preserving fine details
- **Parameters**: ~7M (default configuration)
- **Training time**: Medium

### U-Net with Residual Attention

Enhanced U-Net with residual blocks and attention mechanisms.

- **Best for**: Complex noise patterns, high-quality reconstruction
- **Parameters**: ~10M
- **Training time**: Longer

### DnCNN

Deep residual denoising network.

- **Best for**: Gaussian noise removal
- **Parameters**: ~0.5M
- **Training time**: Fast

### Autoencoder

Basic encoder-decoder without skip connections.

- **Best for**: Simple denoising, baseline comparisons
- **Parameters**: ~2M
- **Training time**: Fast

### Residual CNN

Pure residual architecture.

- **Best for**: Preserving original structure
- **Parameters**: ~1M
- **Training time**: Fast

### MobileNet Denoising

Lightweight architecture using depthwise separable convolutions.

- **Best for**: Edge deployment, real-time inference
- **Parameters**: ~0.3M
- **Training time**: Very fast

### Dilated CNN

Uses dilated convolutions for larger receptive field.

- **Best for**: Large-scale noise patterns
- **Parameters**: ~1M
- **Training time**: Fast

### Noise2Void

Self-supervised denoising (no clean targets needed).

- **Best for**: When clean reference images unavailable
- **Parameters**: ~0.5M
- **Training time**: Medium

## Training Configuration

### Hyperparameters

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| Epochs | Training iterations | 20-100 |
| Batch size | Images per update | 8-32 |
| Learning rate | Step size | 1e-4 to 1e-3 |
| Weight decay | L2 regularization | 0 to 1e-4 |
| Dropout | Regularization | 0 to 0.3 |

### Loss Functions

- **MSE**: Mean Squared Error (default)
- **L1**: Mean Absolute Error
- **SSIM**: Structural Similarity Loss
- **Perceptual**: VGG-based perceptual loss

### Optimizers

- **Adam**: Adaptive moment estimation (default)
- **AdamW**: Adam with decoupled weight decay
- **SGD**: Stochastic gradient descent

## Training Tips

1. **Start simple**: Begin with default parameters
2. **Monitor validation loss**: Stop if it increases (overfitting)
3. **Use GPU**: Training is 10-50x faster on GPU
4. **Adequate data**: At least 100 images recommended

## Model Export

Trained models can be exported to:

- **PyTorch checkpoint** (`.pth`): For continued training or inference
- **ONNX** (`.onnx`): For deployment and cross-platform inference
