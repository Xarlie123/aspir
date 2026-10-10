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
| Fourier     | Fourier Inverse   | Pseudoinverse, FISTA, TV-Norm        |

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
\hat{x} = \frac{1}{N} S^\top y, \qquad N = \text{number of pixels}
$$

**Characteristics**:
- Default for **Hadamard** and **Cal-Sal** masks
- Exact reconstruction when all Hadamard patterns are used
- With any subset of $M$ rows it is the pseudoinverse ($H H^\top = N I$), so the
  output is on the image's scale at every sampling ratio. Versions before 1.3
  divided by $M$, which scaled the output by $N / M$; the network input was not
  affected, because it is min-max normalised over the dataset.

### Fourier Inverse

Phase-shifting demodulation followed by an inverse FFT — the reconstruction
side of Fourier Single-Pixel Imaging (FSPI).

Each spectrum coefficient is illuminated with `phase_steps` sinusoidal
patterns, and the bucket values $D_\phi$ are demodulated into one complex
coefficient:

$$
\tilde{F} = (D_0 - D_\pi) + j\,(D_{\pi/2} - D_{3\pi/2})
\qquad \text{(four-step)}
$$

$$
\tilde{F} = (2 D_0 - D_{2\pi/3} - D_{4\pi/3})
           + j\sqrt{3}\,(D_{2\pi/3} - D_{4\pi/3})
\qquad \text{(three-step)}
$$

ASPIR divides these by $2b$ and $3b$ respectively ($b = 0.5$ is the pattern
amplitude), so the recovered value equals `numpy.fft.fft2(image)[u, v]`
exactly. The conjugate-symmetric half of the spectrum is filled by
conjugation, unsampled entries stay at zero, and the image is the real part
of `numpy.fft.ifft2`.

**Characteristics**:
- Default for **Fourier** masks
- Non-iterative and cheap: one FFT per image regardless of the pattern count
- Exact (unit scale, zero offset) when the whole half-plane is acquired
- The phase-shift subtraction is itself the differential measurement, so the
  $a$ offset of the patterns — and with it the detector's DC drift —
  cancels without needing complementary $P^+/P^-$ projections
- Unchanged for binary FSPI: dithered patterns keep the local mean of the
  sinusoid, so the same demodulation applies to DMD-projectable binary
  illumination (see {doc}`masks`)

**References**: Zhang, Ma & Zhong, *Nat. Commun.* **6**, 6225 (2015);
Zhang et al., *Sci. Rep.* **7**, 12029 (2017).

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

**Parameters** (Batch Test form, or the *Lambda* / *Iterations* controls that
appear in each Single Test mask panel when FISTA is selected):
- `lambda`: weight of the l1 term (default: 0.001)
- `iterations`: number of iterations, run in full — there is no early stop and
  no tolerance parameter (default: 500)

### TV-Norm (Total Variation)

Edge-preserving reconstruction.

$$
\min_x \frac{1}{2}\|y - Sx\|_2^2 + \lambda \cdot TV(x)
$$

**Characteristics**:
- Preserves edges and sharp transitions
- Removes noise while keeping structure
- Good for natural images

**Parameters** (as for FISTA):
- `lambda`: weight of the total-variation term (default: 0.1)
- `iterations`: number of iterations, run in full — no early stop, no tolerance
  (default: 100)

The defaults are the values both solvers have always run with. Up to v1.2.1
the values entered in a batch test were stored but never reached the solvers;
batch configs from before format 2.2 therefore load with these defaults, and a
warning says so. Each batch report records the values a test actually used
(`reconstruction_params`).

## Comparison

| Method          | Speed    | Compression | Noise Robustness |
|-----------------|----------|-------------|------------------|
| Ghost Imaging   | ★★★★★    | ★★☆☆☆       | ★★☆☆☆            |
| Sweep Linear    | ★★★★★    | ★★☆☆☆       | ★★☆☆☆            |
| Hadamard Linear | ★★★★★    | ★★★☆☆       | ★★★☆☆            |
| Fourier Inverse | ★★★★★    | ★★★★☆       | ★★★☆☆            |
| Pseudoinverse   | ★★★★☆    | ★★☆☆☆       | ★★☆☆☆            |
| FISTA           | ★★☆☆☆    | ★★★★★       | ★★★★☆            |
| TV-Norm         | ★★☆☆☆    | ★★★★☆       | ★★★★★            |

## When to Use Each

- **Ghost Imaging / Sweep Linear / Hadamard Linear**: native algorithm for
  each mask — quick previews and the fastest path to a reconstruction.
- **Fourier Inverse**: native algorithm for Fourier masks; strongest at low
  sampling ratios, where the low-frequency-first ordering concentrates the
  measurement budget on the coefficients that carry most of the image energy.
- **Pseudoinverse**: full sampling with low noise; works on every mask.
- **FISTA**: compressive sensing with sparse signals.
- **TV-Norm**: natural images with edges, noisy measurements.

## Device selection and scaling

Every reconstruction algorithm runs on a PyTorch backend that targets either
the CPU or a CUDA GPU. There is one implementation per algorithm, so a CPU
versus GPU comparison is the same code on two devices.

### Choosing the device

- **Single Test**: each mask panel has a **Device** dropdown next to
  *Reconstruction method* — *Auto*, *CPU* or *CUDA* (greyed out when no CUDA
  device is present).
- **Batch Test**: the same dropdown, stored per test as
  `reconstruction_device` (`"auto"`, `"cpu"` or `"cuda"`). Batch configs saved
  before this option existed load as `"auto"`.
- **Python**: `Simulation.set_applicator(reconstruction_method, device="auto")`,
  or directly on any applicator, e.g.
  `ApplicatorFISTA(dataset, mask, device="cuda", dtype="float32")`.

*Auto* picks CUDA when available and the CPU otherwise, and logs the choice
once. Requesting *CUDA* on a machine without it falls back to the CPU with a
warning instead of failing, so a batch config written on a GPU workstation
still runs on a CPU-only host. The device actually used is recorded next to
every reconstruction time (`timing_reconstruction_device`).

### How the sensing matrix is applied

The explicit sensing matrix `S` is `M x N`. At 256 x 256 and M/N = 0.32 it is
5.5 GB in float32, more than a 4 GB GPU holds. The backend therefore picks, per
mask set, how to apply `S` and `S^T`; the algorithms never see the difference.

| Mask family | Operator | Device memory |
|-------------|----------|---------------|
| Hadamard (Natural, Walsh-Paley, Cake-Cutting, Cal-Sal) | Fast Walsh-Hadamard transform | O(N), matrix-free |
| Fourier, grayscale | FFT | O(N), matrix-free |
| Scatter, Sweep, Fourier binarised | Dense matrix | resident if it fits in half the free device memory, otherwise streamed from host memory in row blocks |

Hadamard patterns are recognised from the patterns themselves: the transform
of a Sylvester Hadamard row is a single spike, which gives its row index and
sign whatever the family's ordering. A streamed matrix works at any size, only
slower: on a desktop GPU each pass is limited by the PCIe bandwidth, which is
why batching several images into one call helps most there.

### Algorithms

| Algorithm | Formulation on the backend | Operator use |
|-----------|----------------------------|--------------|
| Ghost Imaging, Sweep Linear | `x = S^T (y - mean(y)) / M` | one adjoint |
| Hadamard Linear | `x = S^T y / N` (N = pixels) | one adjoint (one FWHT) |
| Fourier Inverse | phase-shift demodulation, conjugate fill, `ifft2` | FFT |
| Pseudoinverse | cached `S^+` for small dense matrices; otherwise CG on `S S^T z = y`, `x = S^T z` (closed form when `S S^T = N I`) | minimum-norm solution |
| FISTA | proximal gradient with FISTA momentum, step `0.95 / L`, `L` from power iteration on `S^T S` | forward + adjoint per iteration |
| TV-Norm | Chambolle-Pock primal-dual, `tau = 1/sqrt(8)`, `mu = 1/(8 tau)`; exact data-term prox (Cholesky for dense N <= 4096, closed form for Hadamard, warm-started CG otherwise) | forward + adjoint per prox |

FISTA and TV reproduce the iterations of the previous NumPy/pyproximal code
step for step, with the same parameters, and record the objective at every
iteration (`applicator.costf`).

### Precision

The backend works in float32 by default; pass `dtype="float64"` for double
precision. Direct methods agree with float64 to better than 1e-5. Iterative
methods accumulate round-off over hundreds of iterations: in float32 the TV
result differs from its float64 counterpart by about 1.6e-3 (relative) on the
same device, which changes the objective by less than 0.1 % and is invisible in
PSNR. Use float64 when two runs must agree bit for bit — in float64, CUDA and
CPU agree to 1e-11.

### Timing and energy

Reconstruction timing uses the same protocol as NN inference: the device is
synchronised before and after each timed call, two warm-up calls run first
(they also absorb one-off costs such as building the operator or caching the
pseudoinverse), and the reported value is the mean and standard deviation over
the timed images and repeats. Each call includes simulating the single-pixel
measurements.

Batch tests with the *energy* report also measure the reconstruction stage
(`energy_reconstruction_mj`, per image). As for inference, the GPU energy
backend is enabled only when the reconstruction runs on CUDA, so a CPU
reconstruction does not pick up the GPU's idle draw.

### Scaling benchmark

`scripts/bench_reconstruction_scaling.py` times all seven algorithms across
image sizes and sampling ratios on each device and writes a CSV and a Markdown
table:

```bash
# desktop, both devices, plus the legacy NumPy path at 64x64
python scripts/bench_reconstruction_scaling.py --device cpu cuda --legacy
# embedded target
python scripts/bench_reconstruction_scaling.py --device cuda --sizes 64 128 256 --out results/
```

Measured on the development desktop (NVIDIA GeForce GTX 1650, 4 GB, driver
580.178.04; torch 2.6.0 with CUDA 12.4; 15.5 GB host RAM). Each cell is the
per-image latency in ms, **CPU / CUDA (CPU-to-CUDA speed-up)**, float32, M/N =
0.16, including the simulated measurement; up to 50 CelebA images per cell
(64x64 native, resized for the larger sizes — content does not affect timing).
FISTA runs 500 iterations and TV 100, the defaults the platform actually uses.

| Algorithm | Masks | 64x64 | 128x128 | 256x256 | 512x512 |
|---|---|---|---|---|---|
| Ghost Imaging | Scatter (dense) | 0.95 / 0.30 (3.2x) | 12 / 2.12 (5.6x) | 94 / 431 (0.2x) † ‡ | does not fit |
| Sweep Linear | Sweep (dense) | 1.08 / 0.30 (3.6x) | 8.46 / 1.09 (7.8x) | 50 / 7.88 (6.4x) | not run |
| Hadamard Linear | Hadamard (FWHT) | 0.33 / 0.63 (0.5x) | 0.59 / 0.68 (0.9x) | 0.99 / 0.81 (1.2x) | 3.21 / 1.44 (2.2x) |
| Fourier Inverse | Fourier (FFT) | 0.42 / 1.03 (0.4x) | 0.53 / 0.94 (0.6x) | 0.94 / 1.39 (0.7x) | 2.20 / 2.64 (0.8x) |
| Pseudoinverse | Hadamard (FWHT) | 0.52 / 0.54 (1.0x) | 0.56 / 0.69 (0.8x) | 0.99 / 0.84 (1.2x) | 4.18 / 1.44 (2.9x) |
| Pseudoinverse | Scatter (dense) | 1.15 / 0.23 (5.0x) | 13 / 2.04 (6.2x) | 1.3 s / 5.6 s (0.2x) † ‡ | does not fit |
| FISTA | Hadamard (FWHT) | 226 / 344 (0.7x) | 440 / 458 (1.0x) | 795 / 459 (1.7x) | 3.2 s / 470 (6.9x) |
| FISTA | Scatter (dense) | 784 / 118 (6.7x) | 9.4 s / 1.5 s (6.5x) | 73.8 s / 219.1 s (0.3x) † ‡ | does not fit |
| TV-Norm | Hadamard (FWHT) | 87 / 122 (0.7x) | 151 / 158 (1.0x) | 261 / 161 (1.6x) | 896 / 162 (5.5x) |
| TV-Norm | Scatter (dense) | 3.3 s / 352 (9.4x) | 3.0 s / 456 (6.6x) | not run | not run |

† M/N = 0.08: at 0.16 the dense matrix no longer fits in the host RAM free on
the test machine. ‡ The dense matrix exceeds the GPU budget and is streamed
from host memory in row blocks, so each pass is bound by PCIe bandwidth and
the GPU is slower than the CPU. Sweep's geometry fixes M (eight angles, unit
stride), so it runs at its own ratio.

The full results for every size, ratio and device — including peak GPU
memory, batched throughput and the cells that did not fit — are in
`docs/benchmarks/reconstruction_scaling_gtx1650/` (`results.csv`,
`results.md`, `environment.json`).

What the numbers show:

- **Matrix-free families scale; that is where the GPU pays.** With Hadamard
  masks at 512x512, FISTA takes 3.2 s per image on the CPU and 0.47 s on the
  GPU (6.9x), TV 0.90 s and 0.16 s (5.5x). The GPU time stays nearly flat from
  128x128 to 512x512: at these sizes it is bound by kernel launches per
  iteration, not by arithmetic.
- **Batching amortises those launches.** Reconstructing 50 images in one call,
  FISTA with Hadamard masks costs 7.8 ms per image at 64x64 on the GPU
  (single-image latency 344 ms), and TV 2.6 ms (latency 122 ms).
- **Dense matrices gain up to ~9x while they fit on the GPU** (TV with scatter
  masks at 64x64, FISTA at 128x128), and lose once they have to be streamed:
  at 256x256 FISTA with scatter masks takes 74 s per image on the CPU and 219 s
  on the GPU.
- **At 64x64 single-image latency, direct methods are faster on the CPU**
  (0.3-1 ms): the transfer and launch overhead of the GPU exceeds the work.

Against the previous NumPy / pylops / pyproximal implementation on the same
CPU, at 64x64 and M/N = 0.16:

| Algorithm | Masks | Legacy NumPy (ms) | Torch CPU (ms) | Speed-up |
|---|---|---|---|---|
| Ghost Imaging | Scatter (dense) | 8.97 | 0.95 | 9x |
| Hadamard Linear | Hadamard (FWHT) | 8.90 | 0.33 | 27x |
| Fourier Inverse | Fourier (FFT) | 12 | 0.42 | 28x |
| Pseudoinverse | Hadamard (FWHT) | 812 | 0.52 | 1548x |
| Pseudoinverse | Scatter (dense) | 709 | 1.15 | 617x |
| FISTA | Hadamard (FWHT) | 1.9 s | 226 | 9x |
| FISTA | Scatter (dense) | 1.9 s | 784 | 2x |
| TV-Norm | Hadamard (FWHT) | 164.3 s | 87 | 1897x |
| TV-Norm | Scatter (dense) | 163.4 s | 3.3 s | 49x |

The legacy pseudoinverse recomputes `S^+` for every image and the legacy TV
solves a dense 4096 x 4096 system at every iteration; the backend caches the
former and factorises the latter once. Reconstruction times measured with the
previous path (for example `T_recon` in earlier reports) are therefore not
comparable with new ones.

### Legacy reference path

The original NumPy / pylops / pyproximal implementations remain available with
`backend="legacy"` on any applicator. They are not exposed in the GUI; they
exist so the tests can check that the backend reproduces them.
