Context only: ordering, training data and noise differ between rows marked with different conditions.

| object | method | condition | psnr | ssim | lpips | note |
|---|---|---|---|---|---|---|
| CelebA UC2 test (300) | SPyRiT pinv | M=1024 variance order, Poisson alpha=100 | 25.261 | 0.787 | 0.228 |  |
| CelebA UC2 test (300) | SPyRiT DC-Net completion (no denoiser) | M=1024 variance order, Poisson alpha=100 | 26.526 | 0.826 | 0.168 |  |
| CelebA UC2 test (300) | SPyRiT DC-Net + pretrained U-Net (STL-10) | M=1024 variance order, Poisson alpha=100 | 25.465 | 0.837 | 0.064 |  |
| CelebA UC2 test (300) | ASPIR UC2 U-Net (CelebA) | M=983 Cake-Cutting, noiseless (parity setting) | 27.176 | 0.895 | 0.117 |  |
| CelebA UC2 test (300) | SPyRiT PinvNet + Unet (CelebA, parity) | M=983 Cake-Cutting, noiseless (parity setting) | 29.023 | 0.913 | 0.083 |  |
| SPIHIM star (10-repeat mean) | ASPIR linear (pseudoinverse) on the same rows | M=1024 variance order (SPyRiT rows), measured | 17.012 | 0.570 |  |  res. limit 0.357 cyc/px (r=15.2 px) |
| SPIHIM star (10-repeat mean) | SPyRiT pinv | M=1024 variance order (SPyRiT rows), measured | 17.012 | 0.570 |  |  res. limit 0.357 cyc/px (r=15.2 px) |
| SPIHIM star (10-repeat mean) | SPyRiT DC-Net completion | M=1024 variance order (SPyRiT rows), measured | 17.541 | 0.592 |  |  res. limit 0.317 cyc/px (r=17.1 px) |
| SPIHIM star (10-repeat mean) | SPyRiT DC-Net + pretrained U-Net (STL-10) | M=1024 variance order (SPyRiT rows), measured | 17.870 | 0.615 |  |  res. limit 0.322 cyc/px (r=16.8 px) |
| SPIHIM star (10-repeat mean) | ASPIR linear | M=983 Cake-Cutting, measured (R1.3b) | 16.391 | 0.508 |  | res. limit 0.360 cyc/px (r=15.0 px) |
| SPIHIM star (10-repeat mean) | ASPIR U-Net CelebA (b) | M=983 Cake-Cutting, measured (R1.3b) | 16.639 | 0.511 |  | res. limit 0.313 cyc/px (r=17.3 px) |
| SPIHIM star (10-repeat mean) | ASPIR U-Net STL-10 (b) | M=983 Cake-Cutting, measured (R1.3b) | 16.597 | 0.520 |  | res. limit 0.337 cyc/px (r=16.1 px) |
