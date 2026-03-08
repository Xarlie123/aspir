# Datasets

ASPIR supports multiple dataset sources for SPI simulation.

## Dataset Types

### From Image

Load a single image and create a dataset from it.

- **Use case**: Quick testing, specific image analysis
- **Parameters**:
  - Image path
  - Target resolution (images are resized)

### From Folder

Load multiple images from a directory.

- **Use case**: Custom datasets, real captured images
- **Parameters**:
  - Folder path
  - Target resolution
  - Maximum number of images

### From IR Beam

Simulate infrared beam profiles using LightPipes optical simulation.

- **Use case**: Realistic IR beam datasets
- **Parameters**:
  - Beam parameters (wavelength, size, etc.)
  - Number of variations

### From Celebrities (CelebA)

Download celebrity faces dataset via Kaggle API.

- **Use case**: Face reconstruction benchmarks
- **Requirements**: Kaggle API configured
- **Parameters**:
  - Number of images
  - Target resolution

### From SVHN

Street View House Numbers dataset.

- **Use case**: Digit recognition benchmarks
- **Parameters**:
  - Split (train/test)
  - Number of images

## Dataset Splits

When training neural networks, datasets are split into:

| Split | Purpose | Default |
|-------|---------|---------|
| Train | Model training | 80% |
| Validation | Hyperparameter tuning | 10% |
| Test | Final evaluation | 10% |

Configure splits in the Post-processing section or Batch Test configuration.

## Data Format

Internally, datasets are stored as NumPy arrays:

- Shape: `(N, H, W)` for N grayscale images
- Data type: `float32`
- Value range: `[0, 1]`

## Exporting Datasets

Datasets can be exported as:

- **NPZ files**: NumPy compressed archives
- **Individual images**: PNG/JPEG files
