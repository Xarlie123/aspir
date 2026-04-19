# Mask Patterns

Mask patterns determine how the scene is sampled in Single Pixel Imaging. Each pattern modulates the scene in a different way, and the resulting measurements are used to reconstruct the image.

## Mask Types

### Scatter (Random Sampling)

Random point patterns for compressive sensing.

- **Parameters**:
  - **Point density**: Number of active points per pattern
  - **Number of patterns**: Total mask patterns to generate
  - **Random seed**: For reproducibility
  - **Reconstruction method**: Ghost Imaging (native), Pseudoinverse, FISTA, or TV-Norm

- **Pros**: Works with any compression ratio, flexible
- **Cons**: Reconstruction quality depends on the algorithm chosen

### Hadamard

Structured orthogonal patterns based on Hadamard matrices. Requires image size to be a power of 2.

- **Variants**:
  - **Natural**: Standard Hadamard order
  - **Cake-Cutting**: Frequency-ordered patterns, sampling from low to high frequencies
  - **Walsh-Paley**: Sequential ordering based on Walsh functions

- **Parameters**:
  - **Pattern index range** (min, max): Select a subset of the full Hadamard matrix using a dual slider
  - **Reconstruction method**: Hadamard Linear (native), Pseudoinverse, FISTA, or TV-Norm

- **Pros**: Perfect reconstruction when using full sampling (all patterns)
- **Cons**: Image size must be a power of 2 (e.g., 32, 64, 128)

### Sweep

Linear bar scanning patterns. Each row in the configuration table defines a sweep at a specific angle.

- **Parameters** (per row):
  - **Angle**: Sweep direction in degrees (e.g., 0, 45, 90, 135)
  - **Bar width**: Width of the scanning bar in pixels
  - **Stride**: Step size between consecutive bar positions
- **Reconstruction method**: Sweep Linear (native), Pseudoinverse, FISTA, or TV-Norm

- **Pros**: Simple hardware implementation, intuitive parameterization
- **Cons**: Limited compression capability

### Cal-Sal

Structured patterns based on the Cal-Sal transform, a variant of Hadamard matrices with specific ordering properties.

- **Parameters**:
  - **Pattern index range** (min, max): Select a subset of patterns using a dual slider
  - **Reconstruction method**: Hadamard Linear (native), Pseudoinverse, FISTA, or TV-Norm

- **Pros**: Deterministic ordering with good frequency coverage
- **Cons**: Image size must be a power of 2

## Compression Ratio

The compression ratio is defined as:

$$
CR = \frac{N_{pixels}}{N_{patterns}}
$$

For example, a 64×64 image (4096 pixels) with 512 patterns has CR = 8. Higher compression ratios require fewer measurements but produce noisier reconstructions.

## Choosing Patterns

| Scenario                     | Recommended Mask + Reconstruction            |
|------------------------------|----------------------------------------------|
| High quality, full sampling  | Hadamard (Natural) + Hadamard Linear         |
| Compressive sensing          | Scatter + FISTA / TV-Norm                    |
| Simple hardware setup        | Sweep + Sweep Linear                         |
| Ordered frequency sampling   | Hadamard (Cake-Cutting) or Cal-Sal + native  |
| Noisy acquisition            | any mask + TV-Norm                           |

Any combination in the table (or any other mask + method pairing) is valid —
the iterative solvers are mask-agnostic and the native algorithm is always
offered as the first option in each mask's Reconstruction method dropdown.
