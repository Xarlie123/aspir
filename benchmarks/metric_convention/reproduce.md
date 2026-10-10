# Task B — reproduction of the manuscript numbers

## UC2_R0: `experiments/batch_tests/batch_test_celeb_hadamard_results_1/results.batch_analysis_report`

| test | M | rep_psnr_lin | rep_ssim_lin | rep_lpips_lin | rep_psnr_unet | rep_ssim_unet | rep_lpips_unet | rep_gain | max_dev_recomputed |
|---|---|---|---|---|---|---|---|---|---|
| Celeb - Cake Cutting 4% | 163 | 15.33 | 0.4089 | 0.5624 | 21.27 | 0.6712 | 0.2596 | 5.94 | 0.0e+00 |
| Celeb - Cake Cutting 8% | 327 | 16.28 | 0.5160 | 0.4875 | 22.92 | 0.7620 | 0.2033 | 6.64 | 0.0e+00 |
| Celeb - Cake Cutting 16% | 655 | 17.45 | 0.6472 | 0.2994 | 24.33 | 0.8335 | 0.1405 | 6.88 | 0.0e+00 |
| Celeb - Cake Cutting 24% | 983 | 18.66 | 0.7332 | 0.1774 | 26.60 | 0.8896 | 0.1239 | 7.94 | 0.0e+00 |
| Celeb - Cake Cutting 32% | 1310 | 18.86 | 0.7713 | 0.1067 | 28.21 | 0.9183 | 0.0663 | 9.34 | 0.0e+00 |

## UC2_RERUN: `experiments/batch_tests/uc2_batch_test_celeb_3000_hadamard_results/results.batch_analysis_report`

| test | M | rep_psnr_lin | rep_ssim_lin | rep_lpips_lin | rep_psnr_unet | rep_ssim_unet | rep_lpips_unet | rep_gain | max_dev_recomputed |
|---|---|---|---|---|---|---|---|---|---|
| Celeb - Cake Cutting 4% | 163 | 15.33 | 0.4089 | 0.5624 | 21.67 | 0.6741 | 0.2600 | 6.34 | 0.0e+00 |
| Celeb - Cake Cutting 8% | 327 | 16.28 | 0.5160 | 0.4875 | 22.67 | 0.7486 | 0.2046 | 6.39 | 0.0e+00 |
| Celeb - Cake Cutting 16% | 655 | 17.45 | 0.6472 | 0.2994 | 25.10 | 0.8458 | 0.1394 | 7.65 | 0.0e+00 |
| Celeb - Cake Cutting 24% | 983 | 18.66 | 0.7332 | 0.1774 | 26.11 | 0.8850 | 0.1174 | 7.45 | 0.0e+00 |
| Celeb - Cake Cutting 32% | 1310 | 18.86 | 0.7713 | 0.1067 | 28.53 | 0.9251 | 0.0599 | 9.67 | 0.0e+00 |

## UC2 manuscript text vs UC2_R0

| quantity | manuscript | stored (UC2_R0) | rounded | match |
|---|---|---|---|---|
| linear 4 % | 15.30 | 15.33 | 15.3 | yes |
| linear 32 % | 18.80 | 18.86 | 18.9 | NO |
| U-Net 4 % | 21.30 | 21.27 | 21.3 | yes |
| U-Net 32 % | 28.20 | 28.21 | 28.2 | yes |
| gain 4 % | 6.00 | 5.94 | 5.9 | NO |
| gain 32 % | 9.40 | 9.34 | 9.3 | NO |

## UC1 Table 3 vs stored reports

| test | M | rep_psnr_lin | rep_ssim_lin | rep_lpips_lin | rep_psnr_unet | rep_ssim_unet | rep_lpips_unet | rep_gain | max_dev_recomputed | max |Table 3 - stored| |
|---|---|---|---|---|---|---|---|---|---|---|
| Beam prof - stride 2 | 85 | 9.45 | 0.1432 | 0.2975 | 25.67 | 0.8205 | 0.1366 | 16.23 | 0.0e+00 | 4.6e-05 |
| Beam prof - stride 4 | 43 | 10.17 | 0.1373 | 0.2781 | 25.04 | 0.7718 | 0.1401 | 14.87 | 0.0e+00 | 3.6e-05 |
| Beam prof - stride 6 | 29 | 9.02 | 0.1132 | 0.2490 | 23.54 | 0.6521 | 0.1541 | 14.53 | 0.0e+00 | 1.1e-05 |
| Beam prof - stride 8 | 23 | 10.07 | 0.1002 | 0.1967 | 22.58 | 0.6297 | 0.1527 | 12.52 | 0.0e+00 | 2.0e-05 |
| Beam prof - CCH 2.2% | 22 | 10.77 | 0.1712 | 0.2938 | 24.57 | 0.7689 | 0.1450 | 13.80 | 0.0e+00 | 3.8e-05 |
| Beam prof - CCH 2.8% | 28 | 11.02 | 0.1902 | 0.2767 | 25.46 | 0.7954 | 0.1334 | 14.44 | 0.0e+00 | 2.3e-05 |
| Beam prof - CCH 4.2% | 43 | 12.11 | 0.2338 | 0.2219 | 25.68 | 0.8275 | 0.1180 | 13.57 | 0.0e+00 | 3.2e-05 |
| Beam prof - CCH 8.3% | 84 | 13.14 | 0.2803 | 0.1845 | 26.45 | 0.8655 | 0.0930 | 13.31 | 0.0e+00 | 4.9e-05 |

