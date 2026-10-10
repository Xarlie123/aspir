# Task A.2 — output scale of each applicator (16 UC2 test images, M/N ≈ 0.16)

DC gain = mean(output) / mean(image); fit: output ≈ g · image + c (least squares). An output on the
image's scale has DC gain ≈ 1 and c ≈ 0.

| masks | M | method | applicator | DC gain | fit gain g | fit offset c | raw PSNR (shared) |
|---|---|---|---|---|---|---|---|
| Hadamard (UC2 rows) | 655 | native | ApplicatorHadamard | 6.253 | 5.915 | 0.137 | 5.40 |
| Hadamard (UC2 rows) | 655 | pseudoinverse | ApplicatorPseudoinverse | 1.000 | 0.946 | 0.022 | 23.93 |
| Hadamard (UC2 rows) | 655 | fista | ApplicatorFISTA | 1.000 | 0.946 | 0.022 | 23.94 |
| Hadamard (UC2 rows) | 655 | tv_norm | ApplicatorTV | 1.000 | 0.975 | 0.010 | 26.94 |
| Fourier | 656 | native | ApplicatorFourier | 1.000 | 0.940 | 0.024 | 23.52 |
| Sweep | 162 | native | ApplicatorSweep | 3.593 | 4.177 | -0.237 | 7.18 |
| Sweep | 162 | pseudoinverse | ApplicatorPseudoinverse | 0.991 | 0.803 | 0.076 | 18.04 |
| Scatter | 655 | native | ApplicatorScatter | -0.000 | 0.083 | -0.034 | 6.91 |

