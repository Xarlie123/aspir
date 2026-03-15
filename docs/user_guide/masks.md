# Mask Patterns

Mask patterns determine how the scene is sampled in Single Pixel Imaging.

## Mask Types

### Scatter (Random Sampling)

Random point patterns for compressive sensing.

- **Parameters**:
  - Number of patterns
  - Points per pattern
  - Random seed

- **Pros**: Works with any compression ratio
- **Cons**: Reconstruction quality depends on algorithm

### Hadamard

Structured orthogonal patterns based on Hadamard matrices.

- **Variants**:
  - **Standard**: Natural Hadamard order
  - **Scrambled**: Randomized pattern order
  - **Cake-cutting**: Frequency-ordered
  - **Walsh-Paley**: Sequential order

- **Parameters**:
  - Matrix size (must be power of 2)
  - Variant type

- **Pros**: Perfect reconstruction with full sampling
- **Cons**: Requires specific image sizes (power of 2)

### Sweep

Linear scanning patterns.

- **Parameters**:
  - Sweep direction (horizontal/vertical)
  - Step size

- **Pros**: Simple hardware implementation
- **Cons**: Limited compression capability

### Fourier

Frequency-domain sampling patterns.

- **Parameters**:
  - Sampling strategy
  - Frequency range

- **Pros**: Good for frequency-sparse images
- **Cons**: Complex reconstruction

## Compression Ratio

The compression ratio is defined as:

$$
CR = \frac{N_{pixels}}{N_{patterns}}
$$

For example, a 64×64 image (4096 pixels) with 512 patterns has CR = 8.

## Choosing Patterns

| Scenario | Recommended Mask |
|----------|------------------|
| High quality, full sampling | Hadamard |
| Compressive sensing | Scatter + FISTA |
| Fast acquisition | Sweep |
| Frequency analysis | Fourier |
