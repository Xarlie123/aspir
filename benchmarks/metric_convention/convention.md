# Task D.5 — quality under the proposed convention (from stored outputs)

Convention: linear output on the image's scale; clip to [0, 1]; data range 1; LPIPS AlexNet;
mean of per-image values. R0 columns: the stored report fields (manuscript convention).

## UC2 (300 test images)

| run | test | M | linear method | linear PSNR | linear SSIM | linear LPIPS | U-Net PSNR | U-Net SSIM | U-Net LPIPS | gain | R0 linear | R0 U-Net | R0 gain |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| UC2_R0 | Celeb - Cake Cutting 4% | 163 | pseudoinverse | 20.21 | 0.5215 | 0.5347 | 22.20 | 0.6818 | 0.2596 | 1.99 | 15.33 | 21.27 | 5.94 |
| UC2_R0 | Celeb - Cake Cutting 8% | 327 | pseudoinverse | 22.05 | 0.6442 | 0.4537 | 23.93 | 0.7715 | 0.2024 | 1.89 | 16.28 | 22.92 | 6.64 |
| UC2_R0 | Celeb - Cake Cutting 16% | 655 | pseudoinverse | 24.69 | 0.7896 | 0.2703 | 25.27 | 0.8451 | 0.1404 | 0.58 | 17.45 | 24.33 | 6.88 |
| UC2_R0 | Celeb - Cake Cutting 24% | 983 | pseudoinverse | 26.70 | 0.8615 | 0.1491 | 27.55 | 0.8951 | 0.1237 | 0.85 | 18.66 | 26.60 | 7.94 |
| UC2_R0 | Celeb - Cake Cutting 32% | 1310 | pseudoinverse | 28.52 | 0.9083 | 0.0721 | 29.22 | 0.9221 | 0.0660 | 0.70 | 18.86 | 28.21 | 9.34 |
| UC2_RERUN | Celeb - Cake Cutting 4% | 163 | pseudoinverse | 20.21 | 0.5215 | 0.5347 | 22.60 | 0.6848 | 0.2600 | 2.39 | 15.33 | 21.67 | 6.34 |
| UC2_RERUN | Celeb - Cake Cutting 8% | 327 | pseudoinverse | 22.05 | 0.6442 | 0.4537 | 23.59 | 0.7577 | 0.2046 | 1.55 | 16.28 | 22.67 | 6.39 |
| UC2_RERUN | Celeb - Cake Cutting 16% | 655 | pseudoinverse | 24.69 | 0.7896 | 0.2703 | 26.03 | 0.8526 | 0.1394 | 1.34 | 17.45 | 25.10 | 7.65 |
| UC2_RERUN | Celeb - Cake Cutting 24% | 983 | pseudoinverse | 26.70 | 0.8615 | 0.1491 | 27.18 | 0.8945 | 0.1167 | 0.48 | 18.66 | 26.11 | 7.45 |
| UC2_RERUN | Celeb - Cake Cutting 32% | 1310 | pseudoinverse | 28.52 | 0.9083 | 0.0721 | 29.47 | 0.9285 | 0.0599 | 0.95 | 18.86 | 28.53 | 9.67 |

## UC1 Table 3 (200 test images)

| run | test | M | linear method | linear PSNR | linear SSIM | linear LPIPS | U-Net PSNR | U-Net SSIM | U-Net LPIPS | gain | R0 linear | R0 U-Net | R0 gain |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| UC1_SWEEP | Beam prof - stride 2 | 85 | Sweep Linear, oracle affine | 21.11 | 0.3906 | 0.2315 | 25.68 | 0.8224 | 0.1367 | 4.57 | 9.45 | 25.67 | 16.23 |
| UC1_SWEEP | Beam prof - stride 4 | 43 | Sweep Linear, oracle affine | 20.76 | 0.3447 | 0.1262 | 25.07 | 0.7904 | 0.1379 | 4.31 | 10.17 | 25.04 | 14.87 |
| UC1_SWEEP | Beam prof - stride 6 | 29 | Sweep Linear, oracle affine | 20.09 | 0.2763 | 0.1072 | 23.55 | 0.6509 | 0.1535 | 3.46 | 9.02 | 23.54 | 14.53 |
| UC1_SWEEP | Beam prof - stride 8 | 23 | Sweep Linear, oracle affine | 19.31 | 0.2085 | 0.1119 | 22.59 | 0.6288 | 0.1524 | 3.28 | 10.07 | 22.58 | 12.52 |
| UC1_CCH | Beam prof - CCH 2.2% | 22 | pseudoinverse | 20.75 | 0.4101 | 0.1957 | 24.59 | 0.7717 | 0.1421 | 3.83 | 10.77 | 24.57 | 13.80 |
| UC1_CCH | Beam prof - CCH 2.8% | 28 | pseudoinverse | 21.31 | 0.4414 | 0.1786 | 25.48 | 0.8092 | 0.1330 | 4.18 | 11.02 | 25.46 | 14.44 |
| UC1_CCH | Beam prof - CCH 4.2% | 43 | pseudoinverse | 22.22 | 0.5168 | 0.1475 | 25.69 | 0.8288 | 0.1175 | 3.47 | 12.11 | 25.68 | 13.57 |
| UC1_CCH | Beam prof - CCH 8.3% | 84 | pseudoinverse | 23.45 | 0.6227 | 0.1093 | 26.46 | 0.8655 | 0.0934 | 3.01 | 13.14 | 26.45 | 13.31 |

### UC1 Sweep: pseudoinverse on the sweep masks (alternative image-scale linear)

| test | M | alt linear (Sweep pinv) PSNR | alt SSIM | alt LPIPS | U-Net PSNR |
|---|---|---|---|---|---|
| Beam prof - stride 2 | 85 | 23.06 | 0.5558 | 0.1353 | 25.68 |
| Beam prof - stride 4 | 43 | 21.91 | 0.4641 | 0.0881 | 25.07 |
| Beam prof - stride 6 | 29 | 20.45 | 0.3353 | 0.1009 | 23.55 |
| Beam prof - stride 8 | 23 | 19.60 | 0.2848 | 0.1032 | 22.59 |

