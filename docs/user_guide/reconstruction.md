# Reconstruction Algorithms

ASPIR implements several algorithms to reconstruct images from SPI measurements.

## Available Methods

### Ghost Imaging (Conventional)

Classical correlation-based reconstruction.

$$
\hat{x} = \frac{1}{N} \sum_{i=1}^{N} (B_i - \bar{B}) \cdot M_i
$$

Where:
- $B_i$ is the bucket signal for pattern $i$
- $\bar{B}$ is the mean bucket signal
- $M_i$ is the mask pattern

**Characteristics**:
- Very fast computation
- Works with any number of patterns
- Quality improves with more patterns

### Pseudoinverse

Matrix inversion approach using Moore-Penrose pseudoinverse.

$$
\hat{x} = S^+ \cdot y
$$

Where:
- $S$ is the sensing matrix (stacked mask patterns)
- $y$ is the measurement vector
- $S^+$ is the pseudoinverse of $S$

**Characteristics**:
- Exact reconstruction when patterns ≥ pixels
- Sensitive to noise
- Fast computation with precomputed inverse

### FISTA (Fast Iterative Shrinkage-Thresholding)

Sparse reconstruction with L1 regularization.

$$
\min_x \frac{1}{2}\|y - Sx\|_2^2 + \lambda \|x\|_1
$$

**Characteristics**:
- Works with compressed measurements
- Assumes signal sparsity
- Tunable regularization parameter λ

**Parameters**:
- `lambda`: Regularization strength (default: 0.1)
- `iterations`: Maximum iterations (default: 100)

### TV-Norm (Total Variation)

Edge-preserving reconstruction.

$$
\min_x \frac{1}{2}\|y - Sx\|_2^2 + \lambda \cdot TV(x)
$$

**Characteristics**:
- Preserves edges and sharp transitions
- Removes noise while keeping structure
- Good for natural images

**Parameters**:
- `lambda`: Regularization strength
- `iterations`: Maximum iterations

## Comparison

| Method | Speed | Compression | Noise Robustness |
|--------|-------|-------------|------------------|
| Ghost Imaging | ★★★★★ | ★★☆☆☆ | ★★☆☆☆ |
| Pseudoinverse | ★★★★☆ | ★★☆☆☆ | ★★☆☆☆ |
| FISTA | ★★☆☆☆ | ★★★★★ | ★★★★☆ |
| TV-Norm | ★★☆☆☆ | ★★★★☆ | ★★★★★ |

## When to Use Each

- **Ghost Imaging**: Quick previews, real-time applications
- **Pseudoinverse**: Full sampling with low noise
- **FISTA**: Compressive sensing with sparse signals
- **TV-Norm**: Natural images with edges, noisy measurements
