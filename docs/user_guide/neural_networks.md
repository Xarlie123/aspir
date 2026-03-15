# Neural Networks

ASPIR includes several neural network architectures for post-processing (denoising) reconstructed images.

## Available Models

| Model | Registry Key | Parameters | Training Time | Best For |
|-------|-------------|------------|---------------|----------|
| U-Net | `u-net` | ~7M | Medium | General denoising, fine details |
| U-Net Residual Attention | `u-net-residual-attention` | ~10M | Longer | Complex noise, high quality |
| DnCNN | `dncnn` | ~0.5M | Fast | Gaussian noise removal |
| Autoencoder | `autoencoder` | ~2M | Fast | Simple denoising, baselines |
| Residual CNN | `residual_cnn` | ~1M | Fast | Preserving original structure |
| MobileNet Denoising | `mobilenet_denoising` | ~0.3M | Very fast | Edge deployment, real-time |
| Dilated CNN | `dilatedcnn` | ~1M | Fast | Large-scale noise patterns |
| Noise2Void | `noise2void` | ~0.5M | Medium | No clean references available |
| cGAN Denoising | `cgan denoising` | ~5M | Longer | Adversarial denoising |

### U-Net

Encoder-decoder architecture with skip connections. The skip connections help preserve spatial details that would otherwise be lost during downsampling.

### U-Net with Residual Attention

Enhanced U-Net with residual blocks and attention mechanisms. The attention gates learn to focus on relevant features, improving denoising in complex scenarios.

### DnCNN

Deep residual denoising network. Learns to predict the noise component rather than the clean image directly.

### Autoencoder

Basic encoder-decoder without skip connections. Useful as a baseline for comparing against more advanced architectures.

### Residual CNN

Pure residual architecture that learns to add corrections to the input image rather than generating the output from scratch.

### MobileNet Denoising

Lightweight architecture using depthwise separable convolutions. Significantly fewer parameters, suitable for deployment on resource-constrained devices.

### Dilated CNN

Uses dilated (atrous) convolutions to achieve a larger receptive field without increasing the number of parameters. Effective for spatially correlated noise.

### Noise2Void

Self-supervised denoising that does not require clean reference images for training. Learns to denoise by masking and predicting individual pixels from their surroundings.

### cGAN Denoising

Conditional Generative Adversarial Network for denoising. A generator network produces denoised images while a discriminator network learns to distinguish between real and generated results, pushing the generator towards more realistic outputs.

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
- **SmoothL1**: Smooth L1 loss (Huber-like)
- **Huber**: Less sensitive to outliers than MSE

### Optimizers

- **Adam**: Adaptive moment estimation (default)
- **AdamW**: Adam with decoupled weight decay
- **SGD**: Stochastic gradient descent
- **RMSprop**: Root mean square propagation

## Training Tips

1. **Start simple**: Begin with default parameters and U-Net
2. **Monitor validation loss**: Stop if it increases (overfitting)
3. **Use GPU**: Training is 10-50x faster on GPU
4. **Adequate data**: At least 100 images recommended

## Model Export

Trained models can be exported as:

- **PyTorch checkpoint** (`.pth`): For continued training or inference
- **ONNX** (`.onnx`): For deployment and cross-platform inference
