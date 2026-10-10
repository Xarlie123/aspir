# Task E — R1.1 FSPI vs Cake-Cutting under two conventions (300 UC2 test images)

PSNR in dB. R1.1 convention: estimate and reference min-max normalised per image, data range 1.
Shared: estimate on the image's scale, clipped to [0, 1], data range 1.

| M/N | M FSPI / CC | R1.1 conv. FSPI | R1.1 conv. CC | R1.1 conv. FSPI - CC | shared FSPI | shared CC | shared FSPI - CC | shared SSIM FSPI | shared SSIM CC |
|---|---|---|---|---|---|---|---|---|---|
| 0.04 | 164 / 163 | 17.87 | 17.38 | +0.49 | 20.20 | 20.21 | -0.01 | 0.5400 | 0.5215 |
| 0.08 | 328 / 327 | 19.67 | 18.98 | +0.70 | 22.09 | 22.05 | +0.04 | 0.6718 | 0.6442 |
| 0.16 | 656 / 655 | 21.68 | 21.37 | +0.31 | 24.19 | 24.69 | -0.50 | 0.7927 | 0.7896 |
| 0.24 | 984 / 983 | 22.85 | 23.01 | -0.16 | 25.61 | 26.70 | -1.09 | 0.8498 | 0.8615 |
| 0.32 | 1312 / 1310 | 23.93 | 24.58 | -0.66 | 26.81 | 28.52 | -1.71 | 0.8832 | 0.9083 |

