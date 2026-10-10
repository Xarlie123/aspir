# Task C — from ASPIR's R0 scoring to the shared function, one factor at a time

Δ PSNR is the change from the previous step of the same stage. Gain = U-Net minus linear at the
same step (R0: U0 - L0; shared: U2 - L4).

## UC2_R0 — Celeb - Cake Cutting 4%

| step | change | PSNR | Δ PSNR | SSIM | LPIPS |
|---|---|---|---|---|---|
| L0 | R0 (network input, range per image, no clip) | 15.33 | +nan | 0.4089 | 0.5624 |
| L1 | - INT8 quantisation of the network input | 15.33 | +0.00 | 0.4091 | 0.5626 |
| L2 | image scale: pseudoinverse (native x M/N, M/N = 163/4096) | 19.24 | +3.91 | 0.5046 | 0.5365 |
| L3 | + clip to [0, 1] | 19.28 | +0.04 | 0.5093 | 0.5347 |
| L4 | + data range 1 | 20.21 | +0.93 | 0.5215 | 0.5347 |
| U0 | R0 (network output, range per image, no clip) | 21.27 | +nan | 0.6712 | 0.2596 |
| U1 | + clip to [0, 1] | 21.27 | +0.00 | 0.6713 | 0.2596 |
| U2 | + data range 1 | 22.20 | +0.93 | 0.6818 | 0.2596 |

- Network input recovered as q(0.02042 · L + 0.1973); largest deviation from the stored input: 1 quantisation step(s) (INT8).
- Gain: R0 +5.94 dB; shared function +1.99 dB.
- Each factor alone on the R0 linear number: quantisation removed +0.00 dB, image scale +3.91 dB, clip +0.00 dB, data range 1 +0.93 dB.
- Per-image data range: mean range 0.903 (min 0.490); mean of -20 log10(range) = +0.93 dB.
- Check, shared function computed directly: U-Net 22.20 dB, linear (pseudoinverse) 20.21 dB.

## UC2_R0 — Celeb - Cake Cutting 32%

| step | change | PSNR | Δ PSNR | SSIM | LPIPS |
|---|---|---|---|---|---|
| L0 | R0 (network input, range per image, no clip) | 18.86 | +nan | 0.7713 | 0.1067 |
| L1 | - INT8 quantisation of the network input | 18.86 | +0.00 | 0.7715 | 0.1065 |
| L2 | image scale: pseudoinverse (native x M/N, M/N = 1310/4096) | 27.55 | +8.69 | 0.9034 | 0.0726 |
| L3 | + clip to [0, 1] | 27.59 | +0.04 | 0.9044 | 0.0721 |
| L4 | + data range 1 | 28.52 | +0.93 | 0.9083 | 0.0721 |
| U0 | R0 (network output, range per image, no clip) | 28.21 | +nan | 0.9183 | 0.0663 |
| U1 | + clip to [0, 1] | 28.29 | +0.09 | 0.9188 | 0.0660 |
| U2 | + data range 1 | 29.22 | +0.93 | 0.9221 | 0.0660 |

- Network input recovered as q(0.2026 · L + 0.1672); largest deviation from the stored input: 1 quantisation step(s) (INT8).
- Gain: R0 +9.34 dB; shared function +0.70 dB.
- Each factor alone on the R0 linear number: quantisation removed +0.00 dB, image scale +8.69 dB, clip +0.00 dB, data range 1 +0.93 dB.
- Per-image data range: mean range 0.903 (min 0.490); mean of -20 log10(range) = +0.93 dB.
- Check, shared function computed directly: U-Net 29.22 dB, linear (pseudoinverse) 28.52 dB.

## UC2_RERUN — Celeb - Cake Cutting 4%

| step | change | PSNR | Δ PSNR | SSIM | LPIPS |
|---|---|---|---|---|---|
| L0 | R0 (network input, range per image, no clip) | 15.33 | +nan | 0.4089 | 0.5624 |
| L1 | - INT8 quantisation of the network input | 15.33 | +0.00 | 0.4091 | 0.5626 |
| L2 | image scale: pseudoinverse (native x M/N, M/N = 163/4096) | 19.24 | +3.91 | 0.5046 | 0.5365 |
| L3 | + clip to [0, 1] | 19.28 | +0.04 | 0.5093 | 0.5347 |
| L4 | + data range 1 | 20.21 | +0.93 | 0.5215 | 0.5347 |
| U0 | R0 (network output, range per image, no clip) | 21.67 | +nan | 0.6741 | 0.2600 |
| U1 | + clip to [0, 1] | 21.67 | +0.00 | 0.6742 | 0.2600 |
| U2 | + data range 1 | 22.60 | +0.93 | 0.6848 | 0.2600 |

- Network input recovered as q(0.02042 · L + 0.1973); largest deviation from the stored input: 1 quantisation step(s) (INT8).
- Gain: R0 +6.34 dB; shared function +2.39 dB.
- Each factor alone on the R0 linear number: quantisation removed +0.00 dB, image scale +3.91 dB, clip +0.00 dB, data range 1 +0.93 dB.
- Per-image data range: mean range 0.903 (min 0.490); mean of -20 log10(range) = +0.93 dB.
- Check, shared function computed directly: U-Net 22.60 dB, linear (pseudoinverse) 20.21 dB.

## UC2_RERUN — Celeb - Cake Cutting 32%

| step | change | PSNR | Δ PSNR | SSIM | LPIPS |
|---|---|---|---|---|---|
| L0 | R0 (network input, range per image, no clip) | 18.86 | +nan | 0.7713 | 0.1067 |
| L1 | - INT8 quantisation of the network input | 18.86 | +0.00 | 0.7715 | 0.1065 |
| L2 | image scale: pseudoinverse (native x M/N, M/N = 1310/4096) | 27.55 | +8.69 | 0.9034 | 0.0726 |
| L3 | + clip to [0, 1] | 27.59 | +0.04 | 0.9044 | 0.0721 |
| L4 | + data range 1 | 28.52 | +0.93 | 0.9083 | 0.0721 |
| U0 | R0 (network output, range per image, no clip) | 28.53 | +nan | 0.9251 | 0.0599 |
| U1 | + clip to [0, 1] | 28.55 | +0.02 | 0.9255 | 0.0599 |
| U2 | + data range 1 | 29.47 | +0.93 | 0.9285 | 0.0599 |

- Network input recovered as q(0.2026 · L + 0.1672); largest deviation from the stored input: 1 quantisation step(s) (INT8).
- Gain: R0 +9.67 dB; shared function +0.95 dB.
- Each factor alone on the R0 linear number: quantisation removed +0.00 dB, image scale +8.69 dB, clip +0.00 dB, data range 1 +0.93 dB.
- Per-image data range: mean range 0.903 (min 0.490); mean of -20 log10(range) = +0.93 dB.
- Check, shared function computed directly: U-Net 29.47 dB, linear (pseudoinverse) 28.52 dB.

## UC1_SWEEP — Beam prof - stride 2

| step | change | PSNR | Δ PSNR | SSIM | LPIPS |
|---|---|---|---|---|---|
| L0 | R0 (network input, range per image, no clip) | 9.45 | +nan | 0.1432 | 0.2975 |
| L1 | - INT4 quantisation of the network input | 9.46 | +0.01 | 0.1460 | 0.4074 |
| L2 | image scale: oracle global affine (gain 0.165, offset 0.0122) | 20.89 | +11.43 | 0.3602 | 0.2293 |
| L3 | + clip to [0, 1] | 21.11 | +0.22 | 0.3906 | 0.2315 |
| L4 | + data range 1 | 21.11 | +0.00 | 0.3906 | 0.2315 |
| U0 | R0 (network output, range per image, no clip) | 25.67 | +nan | 0.8205 | 0.1366 |
| U1 | + clip to [0, 1] | 25.68 | +0.00 | 0.8224 | 0.1367 |
| U2 | + data range 1 | 25.68 | +0.00 | 0.8224 | 0.1367 |

- Network input recovered as q(0.186 · L + 0.3265); largest deviation from the stored input: 1 quantisation step(s) (INT4, linear output recomputed from the stored INT4 originals).
- Gain: R0 +16.23 dB; shared function +4.57 dB.
- Each factor alone on the R0 linear number: quantisation removed +0.01 dB, image scale +11.44 dB, clip +0.00 dB, data range 1 +0.00 dB.
- Per-image data range: mean range 1.000 (min 1.000); mean of -20 log10(range) = +0.00 dB.
- Check, shared function computed directly: U-Net 25.68 dB.

## UC1_CCH — Beam prof - CCH 8.3%

| step | change | PSNR | Δ PSNR | SSIM | LPIPS |
|---|---|---|---|---|---|
| L0 | R0 (network input, range per image, no clip) | 13.14 | +nan | 0.2803 | 0.1845 |
| L1 | - INT4 quantisation of the network input | 13.16 | +0.03 | 0.2859 | 0.2164 |
| L2 | image scale: pseudoinverse (native x M/N, M/N = 84/1024) | 23.16 | +9.99 | 0.6261 | 0.1242 |
| L3 | + clip to [0, 1] | 23.45 | +0.30 | 0.6227 | 0.1093 |
| L4 | + data range 1 | 23.45 | +0.00 | 0.6227 | 0.1093 |
| U0 | R0 (network output, range per image, no clip) | 26.45 | +nan | 0.8655 | 0.0930 |
| U1 | + clip to [0, 1] | 26.46 | +0.01 | 0.8655 | 0.0934 |
| U2 | + data range 1 | 26.46 | +0.00 | 0.8655 | 0.0934 |

- Network input recovered as q(0.07912 · L + 0.2094); largest deviation from the stored input: 1 quantisation step(s) (INT4, linear output recomputed from the stored INT4 originals).
- Gain: R0 +13.31 dB; shared function +3.01 dB.
- Each factor alone on the R0 linear number: quantisation removed +0.03 dB, image scale +10.02 dB, clip +0.00 dB, data range 1 +0.00 dB.
- Per-image data range: mean range 1.000 (min 1.000); mean of -20 log10(range) = +0.00 dB.
- Check, shared function computed directly: U-Net 26.46 dB, linear (pseudoinverse) 23.45 dB.

