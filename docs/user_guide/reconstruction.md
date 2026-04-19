# Reconstruction Algorithms

ASPIR implements several algorithms to reconstruct images from SPI measurements.
Every mask family exposes a **Reconstruction method** dropdown with the same
four options: its native algorithm plus the three mask-agnostic iterative
solvers. Internally they all implement the same `ApplicatorABC` interface, so
they can be mixed freely with any mask that populates `mask.masks` as the
sensing matrix.

| Mask family | Native algorithm | Iterative solvers (always available) |
|-------------|------------------|--------------------------------------|
| Scatter     | Ghost Imaging     | Pseudoinverse, FISTA, TV-Norm        |
| Sweep       | Sweep Linear      | Pseudoinverse, FISTA, TV-Norm        |
| Hadamard *  | Hadamard Linear   | Pseudoinverse, FISTA, TV-Norm        |
| Cal-Sal     | Hadamard Linear   | Pseudoinverse, FISTA, TV-Norm        |

\* applies to Natural, Cake-Cutting and Walsh-Paley variants.

## Available Methods

### Ghost Imaging

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
- Default for **Scatter** masks
- Quality improves with more patterns

### Sweep Linear

Correlation-based reconstruction specialised for sweep bar patterns. Shares
the formula above but interprets the patterns as translating bars.

**Characteristics**:
- Default for **Sweep** masks
- Very fast; reuses each bar pattern as both sensing and back-projection.

### Hadamard Linear

Dot-product reconstruction using the orthogonality of Hadamard patterns.

$$
\hat{x} = \frac{1}{N} S^\top y
$$

**Characteristics**:
- Default for **Hadamard** and **Cal-Sal** masks
- Exact reconstruction when all Hadamard patterns are used

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
- `lambda`: Regularization strength (default: 0.01)
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
- `lambda`: Regularization strength (default: 0.1)
- `iterations`: Maximum iterations (default: 50)

## Comparison

| Method          | Speed    | Compression | Noise Robustness |
|-----------------|----------|-------------|------------------|
| Ghost Imaging   | ★★★★★    | ★★☆☆☆       | ★★☆☆☆            |
| Sweep Linear    | ★★★★★    | ★★☆☆☆       | ★★☆☆☆            |
| Hadamard Linear | ★★★★★    | ★★★☆☆       | ★★★☆☆            |
| Pseudoinverse   | ★★★★☆    | ★★☆☆☆       | ★★☆☆☆            |
| FISTA           | ★★☆☆☆    | ★★★★★       | ★★★★☆            |
| TV-Norm         | ★★☆☆☆    | ★★★★☆       | ★★★★★            |

## When to Use Each

- **Ghost Imaging / Sweep Linear / Hadamard Linear**: native algorithm for
  each mask — quick previews and the fastest path to a reconstruction.
- **Pseudoinverse**: full sampling with low noise; works on every mask.
- **FISTA**: compressive sensing with sparse signals.
- **TV-Norm**: natural images with edges, noisy measurements.
