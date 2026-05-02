# Datasets

ASPIR supports multiple dataset sources for SPI simulation. Each dataset type is accessible from the left sidebar menu in **Single Test** mode.

## Dataset Types

### From Image

Load a single image and create a dataset from it.

- **Use case**: Quick testing, specific image analysis
- **Parameters**:
  - Image path (browse for a file)
  - Target resolution (images are resized to square dimensions)

### From Folder

Load multiple images from a directory.

- **Use case**: Custom datasets, real captured images
- **Parameters**:
  - Folder path
  - Target resolution
  - Maximum number of images

### From IR Beam

Generate synthetic infrared beam profiles using LightPipes optical simulation. This is particularly useful for testing the SPI pipeline with realistic beam data.

The diffraction physics is parameterised on a **10.6 μm CO₂ laser line** — the canonical thermal-IR working wavelength of the single-pixel imaging hardware ASPIR is designed around. The wavelength is fixed in `DatasetFromIRBeam.__init__` and not surfaced to the GUI; if you need a different IR line you can override it before calling `load_data()`.

- **Use case**: Realistic IR beam datasets for beam profiling research at 10.6 μm
- **Parameters**:
  - **Image dimension** (pixels): Resolution of each beam profile (e.g., 32, 64, 128)
  - **Number of dataset images**: Total images to generate
  - **Seed** (random): For reproducibility
  - **Data Format**: FP32 (32-bit float, full precision), INT8 (8-bit integer), INT4 (4-bit integer)
  - **Beam Mode Distribution**: Percentage mix of the four beam modes (must sum to 100%)
  - **Speckle Noise**: Amount of speckle noise to add (0.0 = none)
  - **Max Mode Order**: Maximum order for Hermite-Gauss and Laguerre-Gauss modes

#### Beam Modes

The IR Beam generator supports four beam mode types, which can be combined in any proportion:

::::{grid} 2
:gutter: 3

:::{grid-item-card} Gaussian
```{image} ../dataset_samples/ir_beam_gaussian.png
:alt: Gaussian beam profile
:width: 80%
```
The most common beam profile. Smooth, symmetric intensity distribution with a single peak at the center.
:::

:::{grid-item-card} Hermite-Gauss
```{image} ../dataset_samples/ir_beam_hermite_gauss.png
:alt: Hermite-Gauss beam profile
:width: 80%
```
Higher-order modes with rectangular symmetry. Produce grid-like patterns that increase in complexity with the mode order.
:::

:::{grid-item-card} Laguerre-Gauss
```{image} ../dataset_samples/ir_beam_laguerre_gauss.png
:alt: Laguerre-Gauss beam profile
:width: 80%
```
Cylindrical symmetry modes. Produce ring-shaped patterns with azimuthal variation.
:::

:::{grid-item-card} Doughnut
```{image} ../dataset_samples/ir_beam_doughnut.png
:alt: Doughnut beam profile
:width: 80%
```
Annular intensity distribution with zero intensity at the center.
:::

::::

### From Celebrities (CelebA)

Download celebrity faces from the [CelebA Small Images](https://www.kaggle.com/datasets/arnrob/celeba-small-images-dataset) Kaggle dataset (cropped/resized CelebA faces, 64x64 RGB converted to grayscale).

- **Use case**: Face reconstruction benchmarks
- **Requirements**: Kaggle API configured (`~/.kaggle/kaggle.json`)
- **Parameters**:
  - Number of images
  - Target resolution

::::{grid} 3
:gutter: 3

:::{grid-item-card}
```{image} ../dataset_samples/celeb_1.png
:alt: CelebA sample 1
:width: 80%
```
:::

:::{grid-item-card}
```{image} ../dataset_samples/celeb_2.png
:alt: CelebA sample 2
:width: 80%
```
:::

:::{grid-item-card}
```{image} ../dataset_samples/celeb_3.png
:alt: CelebA sample 3
:width: 80%
```
:::

::::

### From SVHN

[Street View House Numbers](http://ufldl.stanford.edu/housenumbers/) dataset from Stanford University. Contains cropped digit images from Google Street View.

- **Use case**: Digit recognition benchmarks
- **Parameters**:
  - Split (train/test)
  - Number of images

::::{grid} 3
:gutter: 3

:::{grid-item-card}
```{image} ../dataset_samples/svhn_1.png
:alt: SVHN sample 1
:width: 80%
```
:::

:::{grid-item-card}
```{image} ../dataset_samples/svhn_2.png
:alt: SVHN sample 2
:width: 80%
```
:::

:::{grid-item-card}
```{image} ../dataset_samples/svhn_3.png
:alt: SVHN sample 3
:width: 80%
```
:::

::::

## Data Format

ASPIR supports three data formats that affect precision and memory usage:

| Format | Description | Value Range | Use Case |
|--------|-------------|-------------|----------|
| FP32 | 32-bit float (full precision) | [0.0, 1.0] | Default, highest quality |
| INT8 | 8-bit unsigned integer | [0, 255] | Reduced memory, faster processing |
| INT4 | 4-bit unsigned integer | [0, 15] | Minimum memory, lowest precision |

Internally, all datasets are stored as NumPy arrays with shape `(N, H, W)` for N grayscale images.

## Dataset Splits

When training neural networks, datasets are split into:

| Split | Purpose | Default |
|-------|---------|---------|
| Train | Model training | 80% |
| Validation | Hyperparameter tuning | 10% |
| Test | Final evaluation | 10% |

Configure splits in the Post-processing section or Batch Test configuration.

## Exporting Datasets

Datasets can be exported as:

- **NPZ files**: NumPy compressed archives
- **Individual images**: PNG/JPEG files
