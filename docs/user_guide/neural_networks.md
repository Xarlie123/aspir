# Neural Networks

ASPIR includes several neural network architectures for post-processing (denoising) reconstructed images.

## Available Models

All model architectures are defined as PyTorch `nn.Module` classes in `simulation_engine/_4_postprocessor/models/`. To modify an architecture, edit the corresponding file directly.

### U-Net
**Source:** `models/unet.py`

Encoder-decoder architecture with skip connections. The skip connections help preserve spatial details that would otherwise be lost during downsampling.

- **Configurable parameters**:
  - **Encoder Channels** (default: `[8, 16, 32, 64]`): Channel widths per encoder level

### U-Net with Residual Attention
**Source:** `models/unet_res_att.py`

Enhanced U-Net with residual blocks and attention mechanisms. The attention gates learn to focus on relevant features, improving denoising in complex scenarios.

- **Configurable parameters**:
  - **Encoder Widths** (default: `[32, 64, 128, 256]`): Channel widths per encoder level
  - **Dropout** (default: 0.1, range: 0.0–0.5): Dropout probability in residual blocks
  - **SE Blocks** (default: on): Use Squeeze-Excitation (channel attention) blocks
  - **Attention Gates** (default: on): Use attention gates on skip connections

### DnCNN
**Source:** `models/dncnn.py`

Deep residual denoising network. Learns to predict the noise component rather than the clean image directly.

- **Configurable parameters**:
  - **Feature Channels** (default: 128, range: 16–256): Number of feature channels in hidden layers
  - **Network Depth** (default: 5, range: 3–30): Total number of convolutional layers

### Autoencoder
**Source:** `models/autoencoder.py`

Basic encoder-decoder without skip connections. Useful as a baseline for comparing against more advanced architectures.

- **Configurable parameters**:
  - **Image Size** (default: 32, range: 8–256): Input image size (image is flattened to size²)

### Residual CNN
**Source:** `models/residual_cnn.py`

Pure residual architecture that learns to add corrections to the input image rather than generating the output from scratch.

- **Configurable parameters**:
  - **Feature Channels** (default: 64, range: 16–256): Feature channels per residual block
  - **Residual Blocks** (default: 8, range: 1–20): Number of residual blocks

### MobileNet Denoising
**Source:** `models/mobilenet_denoising.py`

Lightweight architecture using depthwise separable convolutions. Significantly fewer parameters, suitable for deployment on resource-constrained devices.

- **Configurable parameters**:
  - **Block Channels** (default: `[16, 32, 64, 128]`): Channel widths for depthwise separable blocks

### Dilated CNN
**Source:** `models/dilated_cnn.py`

Uses dilated (atrous) convolutions to achieve a larger receptive field without increasing the number of parameters. Effective for spatially correlated noise.

- **Configurable parameters**:
  - **Feature Channels** (default: 128, range: 16–256): Number of feature channels
  - **Dilation Rates** (default: `[1, 2, 4, 8]`): Dilation rates for each layer

### Noise2Void
**Source:** `models/noise2void.py`

Self-supervised denoising that does not require clean reference images for training. Learns to denoise by masking and predicting individual pixels from their surroundings.

- **Configurable parameters**:
  - **Backbone Channels** (default: `[8, 16, 32, 64]`): UNet backbone channel widths per level

### cGAN Denoising
**Source:** `models/cgan.py`

Conditional Generative Adversarial Network for denoising. A generator network produces denoised images while a discriminator network learns to distinguish between real and generated results, pushing the generator towards more realistic outputs.

- **Configurable parameters**:
  - **Stem Channels** (default: 96, range: 32–256): Multi-scale stem output channels
  - **Denoise Channels** (default: 64, range: 32–256): Feature-domain denoising trunk width
  - **Denoise Depth** (default: 8, range: 2–16): Number of Conv-BN-ReLU layers in denoiser
  - **HL Blocks** (default: 4, range: 1–8): Number of cooperative attention + residual block stacks
  - **Output Activation** (default: sigmoid): Output activation (sigmoid, tanh, or none)

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
