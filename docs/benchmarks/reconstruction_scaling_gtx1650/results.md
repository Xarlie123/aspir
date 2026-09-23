Environment: torch 2.6.0+cu124 | CUDA runtime 12.4 | NVIDIA GeForce GTX 1650 (3.6 GB) | driver 580.178.04 | x86_64 host, 15.5 GB RAM

Images: CelebA (64x64 native, resized to each larger size). Precision: torch path float32, legacy float64. Times are per image (ms), synchronised, including the simulated measurement.

### 64x64

| Algorithm | Masks | M/N | Operator | CPU (torch) | CUDA | Speed-up | CUDA batched | CUDA peak MB | CPU legacy |
|---|---|---|---|---|---|---|---|---|---|
| ghost-imaging | scatter | 0.080 | explicit | 0.584 ± 0.061 (n=50) | 0.248 ± 0.2 (n=50) | 2.4x | 0.224 (B=50) | 16 | 4.45 ± 0.071 (n=50) |
| ghost-imaging | scatter | 0.160 | explicit | 0.954 ± 0.041 (n=50) | 0.3 ± 0.2 (n=50) | 3.2x | 0.0279 (B=50) | 21 | 8.97 ± 0.42 (n=50) |
| ghost-imaging | scatter | 0.320 | explicit | 1.88 ± 0.13 (n=50) | 0.469 ± 0.27 (n=50) | 4.0x | 0.0372 (B=50) | 31 | 18.2 ± 0.63 (n=50) |
| sweep-linear | sweep | 0.157 | explicit | 1.08 ± 0.11 (n=50) | 0.298 ± 0.23 (n=50) | 3.6x | 0.0276 (B=50) | 21 | 8.98 ± 0.21 (n=50) |
| hadamard-linear | hadamard | 0.080 | fwht | 0.319 ± 0.049 (n=50) | 0.558 ± 0.23 (n=50) | 0.6x | 0.221 (B=50) | 12 | 4.21 ± 0.12 (n=50) |
| hadamard-linear | hadamard | 0.160 | fwht | 0.331 ± 0.046 (n=50) | 0.631 ± 0.28 (n=50) | 0.5x | 0.0345 (B=50) | 12 | 8.9 ± 0.69 (n=50) |
| hadamard-linear | hadamard | 0.320 | fwht | 0.322 ± 0.048 (n=50) | 0.607 ± 0.26 (n=50) | 0.5x | 0.0329 (B=50) | 13 | 17.3 ± 0.87 (n=50) |
| fourier-inverse | fourier | 0.080 | fft | 0.399 ± 0.053 (n=50) | 1.01 ± 0.53 (n=50) | 0.4x | 0.0744 (B=50) | 12 | 5.72 ± 1 (n=50) |
| fourier-inverse | fourier | 0.160 | fft | 0.421 ± 0.088 (n=50) | 1.03 ± 0.35 (n=50) | 0.4x | 0.0363 (B=50) | 12 | 11.9 ± 2 (n=50) |
| fourier-inverse | fourier | 0.320 | fft | 0.411 ± 0.033 (n=50) | 0.996 ± 0.37 (n=50) | 0.4x | 0.0361 (B=50) | 13 | 19.5 ± 2.8 (n=50) |
| pseudoinverse | hadamard | 0.080 | fwht / closed-form | 0.341 ± 0.18 (n=50) | 0.532 ± 0.23 (n=50) | 0.6x | 0.0483 (B=50) | 12 | 372 ± 1.8e+02 (n=50) |
| pseudoinverse | hadamard | 0.160 | fwht / closed-form | 0.525 ± 0.18 (n=50) | 0.541 ± 0.24 (n=50) | 1.0x | 0.0368 (B=50) | 12 | 812 ± 1.6e+02 (n=50) |
| pseudoinverse | hadamard | 0.320 | fwht / closed-form | 0.319 ± 0.029 (n=50) | 0.512 ± 0.22 (n=50) | 0.6x | 0.0363 (B=50) | 13 | 2.63e+03 ± 3.4e+02 (n=35) |
| pseudoinverse | scatter | 0.080 | explicit / pinv | 0.724 ± 0.048 (n=50) | 0.164 ± 0.011 (n=50) | 4.4x | 0.0482 (B=50) | 35 | 233 ± 86 (n=50) |
| pseudoinverse | scatter | 0.160 | explicit / pinv | 1.15 ± 0.036 (n=50) | 0.228 ± 0.0082 (n=50) | 5.0x | 0.0272 (B=50) | 63 | 709 ± 86 (n=50) |
| pseudoinverse | scatter | 0.320 | explicit / pinv | 1.95 ± 0.049 (n=50) | 0.343 ± 0.011 (n=50) | 5.7x | 0.0367 (B=50) | 125 | 2.26e+03 ± 1.6e+02 (n=40) |
| fista | hadamard | 0.080 | fwht | 231 ± 8.6 (n=50) | 339 ± 1.8 (n=50) | 0.7x | 7.66 (B=50) | 15 | 1.04e+03 ± 1.3e+02 (n=50) |
| fista | hadamard | 0.160 | fwht | 226 ± 5.3 (n=50) | 344 ± 1.8 (n=50) | 0.7x | 7.83 (B=50) | 15 | 1.93e+03 ± 1.8e+02 (n=50) |
| fista | hadamard | 0.320 | fwht | 225 ± 2.1 (n=50) | 328 ± 8.2 (n=50) | 0.7x | 7.49 (B=50) | 16 | 3.04e+03 ± 1.5e+02 (n=30) |
| fista | scatter | 0.080 | explicit | 419 ± 4.7 (n=50) | 118 ± 4.3 (n=50) | 3.5x | 5.19 (B=50) | 20 | 1.02e+03 ± 1.4e+02 (n=50) |
| fista | scatter | 0.160 | explicit | 784 ± 19 (n=50) | 118 ± 1.5 (n=50) | 6.7x | 8.76 (B=50) | 25 | 1.86e+03 ± 1.4e+02 (n=49) |
| fista | scatter | 0.320 | explicit | 1.44e+03 ± 36 (n=50) | 234 ± 15 (n=50) | 6.2x | 15.3 (B=50) | 36 | 2.99e+03 ± 1.5e+02 (n=31) |
| tv | hadamard | 0.080 | fwht / woodbury | 87.6 ± 3.8 (n=50) | 128 ± 4.7 (n=50) | 0.7x | 2.75 (B=50) | 19 | 1.67e+05 (n=1) |
| tv | hadamard | 0.160 | fwht / woodbury | 86.6 ± 8.5 (n=50) | 122 ± 2.3 (n=50) | 0.7x | 2.64 (B=50) | 19 | 1.64e+05 (n=1) |
| tv | hadamard | 0.320 | fwht / woodbury | 85.2 ± 2.5 (n=50) | 126 ± 2.9 (n=50) | 0.7x | 2.63 (B=50) | 19 | 1.66e+05 (n=1) |
| tv | scatter | 0.080 | explicit / cholesky | 3.26e+03 ± 21 (n=27) | 346 ± 0.11 (n=50) | 9.4x | 75.1 (B=50) | 408 | 1.61e+05 (n=1) |
| tv | scatter | 0.160 | explicit / cholesky | 3.32e+03 ± 36 (n=26) | 352 ± 0.2 (n=50) | 9.4x | 75.3 (B=50) | 423 | 1.63e+05 (n=1) |
| tv | scatter | 0.320 | explicit / cholesky | 3.34e+03 ± 47 (n=27) | 365 ± 1 (n=50) | 9.2x | 76.1 (B=50) | 454 | 1.6e+05 (n=1) |

### 128x128

| Algorithm | Masks | M/N | Operator | CPU (torch) | CUDA | Speed-up | CUDA batched | CUDA peak MB | CPU legacy |
|---|---|---|---|---|---|---|---|---|---|
| ghost-imaging | scatter | 0.080 | explicit | 6.03 ± 0.2 (n=50) | 1.14 ± 0.013 (n=50) | 5.3x | 0.162 (B=50) | 100 | — |
| ghost-imaging | scatter | 0.160 | explicit | 12 ± 0.21 (n=50) | 2.12 ± 0.036 (n=50) | 5.6x | 0.251 (B=50) | 182 | — |
| ghost-imaging | scatter | 0.320 | explicit | 23.7 ± 0.48 (n=50) | 4.04 ± 0.031 (n=50) | 5.9x | 0.257 (B=50) | 348 | — |
| sweep-linear | sweep | 0.079 | explicit | 8.46 ± 0.054 (n=50) | 1.09 ± 0.016 (n=50) | 7.8x | 0.148 (B=50) | 99 | — |
| hadamard-linear | hadamard | 0.080 | fwht | 0.507 ± 0.019 (n=50) | 0.631 ± 0.019 (n=50) | 0.8x | 0.0758 (B=50) | 25 | — |
| hadamard-linear | hadamard | 0.160 | fwht | 0.589 ± 0.019 (n=50) | 0.679 ± 0.019 (n=50) | 0.9x | 0.0721 (B=50) | 25 | — |
| hadamard-linear | hadamard | 0.320 | fwht | 0.613 ± 0.019 (n=50) | 0.636 ± 0.022 (n=50) | 1.0x | 0.0713 (B=50) | 27 | — |
| fourier-inverse | fourier | 0.080 | fft | 0.469 ± 0.019 (n=50) | 0.869 ± 0.037 (n=50) | 0.5x | 0.08 (B=50) | 25 | — |
| fourier-inverse | fourier | 0.160 | fft | 0.531 ± 0.096 (n=50) | 0.942 ± 0.16 (n=50) | 0.6x | 0.073 (B=50) | 26 | — |
| fourier-inverse | fourier | 0.320 | fft | 0.651 ± 0.039 (n=50) | 0.897 ± 0.046 (n=50) | 0.7x | 0.072 (B=50) | 28 | — |
| pseudoinverse | hadamard | 0.080 | fwht / closed-form | 0.644 ± 0.03 (n=50) | 0.647 ± 0.036 (n=50) | 1.0x | 0.0794 (B=50) | 25 | — |
| pseudoinverse | hadamard | 0.160 | fwht / closed-form | 0.564 ± 0.04 (n=50) | 0.687 ± 0.044 (n=50) | 0.8x | 0.0803 (B=50) | 25 | — |
| pseudoinverse | hadamard | 0.320 | fwht / closed-form | 0.688 ± 0.043 (n=50) | 0.709 ± 0.041 (n=50) | 1.0x | 0.0775 (B=50) | 27 | — |
| pseudoinverse | scatter | 0.080 | explicit / pinv | 6.9 ± 0.096 (n=50) | 1.14 ± 0.034 (n=50) | 6.1x | 0.163 (B=50) | 433 | — |
| pseudoinverse | scatter | 0.160 | explicit / pinv | 12.6 ± 0.085 (n=50) | 2.04 ± 0.029 (n=50) | 6.2x | 0.287 (B=50) | 882 | — |
| pseudoinverse | scatter | 0.320 | explicit / cgne | 635 ± 29 (n=50) | 103 ± 4.3 (n=50) | 6.2x | 7.6 (B=50) | 350 | — |
| fista | hadamard | 0.080 | fwht | 431 ± 12 (n=50) | 465 ± 7.9 (n=50) | 0.9x | 22.6 (B=50) | 35 | — |
| fista | hadamard | 0.160 | fwht | 440 ± 18 (n=50) | 458 ± 6.6 (n=50) | 1.0x | 22.8 (B=50) | 36 | — |
| fista | hadamard | 0.320 | fwht | 446 ± 19 (n=50) | 461 ± 11 (n=50) | 1.0x | 23.9 (B=50) | 39 | — |
| fista | scatter | 0.080 | explicit | 4.85e+03 ± 30 (n=18) | 743 ± 0.54 (n=50) | 6.5x | 101 (B=50) | 116 | — |
| fista | scatter | 0.160 | explicit | 9.45e+03 ± 31 (n=9) | 1.46e+03 ± 1.4 (n=50) | 6.5x | 192 (B=50) | 199 | — |
| fista | scatter | 0.320 | explicit | 1.84e+04 ± 74 (n=4) | 2.88e+03 ± 2.2 (n=31) | 6.4x | 272 (B=31) | 354 | — |
| tv | hadamard | 0.080 | fwht / woodbury | 150 ± 4.6 (n=50) | 158 ± 5.8 (n=50) | 1.0x | 8.69 (B=50) | 50 | — |
| tv | hadamard | 0.160 | fwht / woodbury | 151 ± 4.7 (n=50) | 158 ± 2.3 (n=50) | 1.0x | 8.66 (B=50) | 50 | — |
| tv | hadamard | 0.320 | fwht / woodbury | 154 ± 4.6 (n=50) | 160 ± 1.4 (n=50) | 1.0x | 8.94 (B=50) | 52 | — |
| tv | scatter | 0.080 | explicit / cg | 2.67e+03 ± 6.4e+02 (n=27) | 408 ± 1.1e+02 (n=50) | 6.5x | 63.5 (B=50) | 135 | — |
| tv | scatter | 0.160 | explicit / cg | 2.99e+03 ± 9e+02 (n=31) | 456 ± 86 (n=50) | 6.6x | 125 (B=50) | 217 | — |
| tv | scatter | 0.320 | explicit / cg | 5.36e+03 ± 65 (n=16) | 855 ± 5.9 (n=50) | 6.3x | 54.5 (B=50) | 382 | — |

### 256x256

| Algorithm | Masks | M/N | Operator | CPU (torch) | CUDA | Speed-up | CUDA batched | CUDA peak MB | CPU legacy |
|---|---|---|---|---|---|---|---|---|---|
| ghost-imaging | scatter | 0.080 | explicit-chunked | 94 ± 2.4 (n=50) | 431 ± 0.28 (n=50) | 0.2x | 9.56 (B=50) | 1128 | — |
| ghost-imaging | scatter | 0.16 | — | n/a (dense matrix 2.7 GB (float32) does not fit host RAM) | n/a (dense matrix 2.7 GB (float32) does not fit host RAM) | — | — | — | — |
| ghost-imaging | scatter | 0.32 | — | n/a (dense matrix 5.5 GB (float32) does not fit host RAM) | n/a (dense matrix 5.5 GB (float32) does not fit host RAM) | — | — | — | — |
| sweep-linear | sweep | 0.039 | explicit | 50.2 ± 1.1 (n=50) | 7.88 ± 0.12 (n=50) | 6.4x | 0.671 (B=50) | 689 | — |
| hadamard-linear | hadamard | 0.080 | fwht | 0.968 ± 0.067 (n=50) | 0.813 ± 0.017 (n=50) | 1.2x | 0.27 (B=50) | 74 | — |
| hadamard-linear | hadamard | 0.160 | fwht | 0.988 ± 0.085 (n=50) | 0.809 ± 0.0068 (n=50) | 1.2x | 0.271 (B=50) | 75 | — |
| hadamard-linear | hadamard | 0.320 | fwht | 0.985 ± 0.025 (n=50) | 0.809 ± 0.02 (n=50) | 1.2x | 0.279 (B=50) | 79 | — |
| fourier-inverse | fourier | 0.080 | fft | 0.805 ± 0.055 (n=50) | 1.27 ± 0.023 (n=50) | 0.6x | 0.239 (B=50) | 76 | — |
| fourier-inverse | fourier | 0.160 | fft | 0.935 ± 0.048 (n=50) | 1.39 ± 0.099 (n=50) | 0.7x | 0.234 (B=50) | 78 | — |
| fourier-inverse | fourier | 0.320 | fft | 0.94 ± 0.033 (n=50) | 1.3 ± 0.032 (n=50) | 0.7x | 0.287 (B=50) | 84 | — |
| pseudoinverse | hadamard | 0.080 | fwht / closed-form | 0.982 ± 0.03 (n=50) | 0.814 ± 0.01 (n=50) | 1.2x | 0.3 (B=50) | 74 | — |
| pseudoinverse | hadamard | 0.160 | fwht / closed-form | 0.99 ± 0.052 (n=50) | 0.841 ± 0.037 (n=50) | 1.2x | 0.29 (B=50) | 75 | — |
| pseudoinverse | hadamard | 0.320 | fwht / closed-form | 1 ± 0.042 (n=50) | 0.82 ± 0.027 (n=50) | 1.2x | 0.295 (B=50) | 79 | — |
| pseudoinverse | scatter | 0.080 | explicit-chunked / cgne | 1.27e+03 ± 53 (n=50) | 5.62e+03 ± 1e+02 (n=16) | 0.2x | 439 (B=16) | 1102 | — |
| pseudoinverse | scatter | 0.16 | — | n/a (dense matrix 2.7 GB (float32) does not fit host RAM) | n/a (dense matrix 2.7 GB (float32) does not fit host RAM) | — | — | — | — |
| pseudoinverse | scatter | 0.32 | — | n/a (dense matrix 5.5 GB (float32) does not fit host RAM) | n/a (dense matrix 5.5 GB (float32) does not fit host RAM) | — | — | — | — |
| fista | hadamard | 0.080 | fwht | 796 ± 20 (n=50) | 457 ± 4.6 (n=50) | 1.7x | 77 (B=50) | 114 | — |
| fista | hadamard | 0.160 | fwht | 795 ± 14 (n=50) | 459 ± 8.6 (n=50) | 1.7x | 78.8 (B=50) | 116 | — |
| fista | hadamard | 0.320 | fwht | 805 ± 30 (n=50) | 467 ± 1.8 (n=50) | 1.7x | 82.1 (B=50) | 125 | — |
| fista | scatter | 0.080 | explicit-chunked | 7.38e+04 ± 6.3e+02 (n=3) | 2.19e+05 (n=1) | 0.3x | — | 1090 | — |
| fista | scatter | 0.16 | — | n/a (dense matrix 2.7 GB (float32) does not fit host RAM) | n/a (dense matrix 2.7 GB (float32) does not fit host RAM) | — | — | — | — |
| fista | scatter | 0.32 | — | n/a (dense matrix 5.5 GB (float32) does not fit host RAM) | n/a (dense matrix 5.5 GB (float32) does not fit host RAM) | — | — | — | — |
| tv | hadamard | 0.080 | fwht / woodbury | 262 ± 6.8 (n=50) | 162 ± 4.2 (n=50) | 1.6x | 30.3 (B=50) | 179 | — |
| tv | hadamard | 0.160 | fwht / woodbury | 261 ± 6.1 (n=50) | 161 ± 2.7 (n=50) | 1.6x | 30.8 (B=50) | 181 | — |
| tv | hadamard | 0.320 | fwht / woodbury | 266 ± 6.7 (n=50) | 158 ± 3.2 (n=50) | 1.7x | 31.5 (B=50) | 185 | — |
| tv | scatter | 0.08 | — | n/a (TV with inner CG on a >=1.4 GB dense matrix: estimated >15 min per image) | n/a (TV with inner CG on a streamed matrix: estimated hours per image) | — | — | — | — |
| tv | scatter | 0.16 | — | n/a (TV with inner CG on a >=1.4 GB dense matrix: estimated >15 min per image) | n/a (TV with inner CG on a streamed matrix: estimated hours per image) | — | — | — | — |
| tv | scatter | 0.32 | — | n/a (TV with inner CG on a >=1.4 GB dense matrix: estimated >15 min per image) | n/a (TV with inner CG on a streamed matrix: estimated hours per image) | — | — | — | — |

### 512x512

| Algorithm | Masks | M/N | Operator | CPU (torch) | CUDA | Speed-up | CUDA batched | CUDA peak MB | CPU legacy |
|---|---|---|---|---|---|---|---|---|---|
| ghost-imaging | scatter | 0.08 | — | n/a (dense matrix 22.0 GB (float32) does not fit host RAM) | n/a (dense matrix 22.0 GB (float32) does not fit host RAM) | — | — | — | — |
| ghost-imaging | scatter | 0.16 | — | n/a (dense matrix 44.0 GB (float32) does not fit host RAM) | n/a (dense matrix 44.0 GB (float32) does not fit host RAM) | — | — | — | — |
| ghost-imaging | scatter | 0.32 | — | n/a (dense matrix 88.0 GB (float32) does not fit host RAM) | n/a (dense matrix 88.0 GB (float32) does not fit host RAM) | — | — | — | — |
| sweep-linear | sweep | 0.08 | — | n/a (MaskSweep materialises float64 patterns: ~12 GB at 512x512) | n/a (MaskSweep materialises float64 patterns: ~12 GB at 512x512) | — | — | — | — |
| hadamard-linear | hadamard | 0.080 | fwht | 3.08 ± 0.77 (n=50) | 1.48 ± 0.014 (n=50) | 2.1x | 1.81 (B=50) | 266 | — |
| hadamard-linear | hadamard | 0.160 | fwht | 3.21 ± 1.1 (n=50) | 1.44 ± 0.015 (n=50) | 2.2x | 1.8 (B=50) | 275 | — |
| hadamard-linear | hadamard | 0.320 | fwht | 2.44 ± 0.079 (n=50) | 1.44 ± 0.024 (n=50) | 1.7x | 1.82 (B=50) | 291 | — |
| fourier-inverse | fourier | 0.080 | fft | 2.2 ± 0.17 (n=50) | 2.69 ± 0.15 (n=50) | 0.8x | 1.63 (B=50) | 270 | — |
| fourier-inverse | fourier | 0.160 | fft | 2.2 ± 0.068 (n=50) | 2.64 ± 0.076 (n=50) | 0.8x | 1.64 (B=50) | 280 | — |
| fourier-inverse | fourier | 0.320 | fft | 2.53 ± 0.069 (n=50) | 2.6 ± 0.052 (n=50) | 1.0x | 1.67 (B=50) | 302 | — |
| pseudoinverse | hadamard | 0.080 | fwht / closed-form | 4.15 ± 0.062 (n=50) | 1.45 ± 0.026 (n=50) | 2.9x | 1.77 (B=50) | 266 | — |
| pseudoinverse | hadamard | 0.160 | fwht / closed-form | 4.18 ± 0.093 (n=50) | 1.44 ± 0.023 (n=50) | 2.9x | 1.79 (B=50) | 275 | — |
| pseudoinverse | hadamard | 0.320 | fwht / closed-form | 4.11 ± 0.059 (n=50) | 1.45 ± 0.028 (n=50) | 2.8x | 1.82 (B=50) | 291 | — |
| pseudoinverse | scatter | 0.08 | — | n/a (dense matrix 22.0 GB (float32) does not fit host RAM) | n/a (dense matrix 22.0 GB (float32) does not fit host RAM) | — | — | — | — |
| pseudoinverse | scatter | 0.16 | — | n/a (dense matrix 44.0 GB (float32) does not fit host RAM) | n/a (dense matrix 44.0 GB (float32) does not fit host RAM) | — | — | — | — |
| pseudoinverse | scatter | 0.32 | — | n/a (dense matrix 88.0 GB (float32) does not fit host RAM) | n/a (dense matrix 88.0 GB (float32) does not fit host RAM) | — | — | — | — |
| fista | hadamard | 0.080 | fwht | 3.14e+03 ± 1.7e+02 (n=30) | 471 ± 9 (n=50) | 6.7x | 316 (B=50) | 425 | — |
| fista | hadamard | 0.160 | fwht | 3.25e+03 ± 1.1e+02 (n=27) | 470 ± 2.6 (n=50) | 6.9x | 323 (B=50) | 441 | — |
| fista | hadamard | 0.320 | fwht | 3.3e+03 ± 2.1e+02 (n=27) | 471 ± 6.2 (n=50) | 7.0x | 342 (B=50) | 473 | — |
| fista | scatter | 0.08 | — | n/a (dense matrix 22.0 GB (float32) does not fit host RAM) | n/a (dense matrix 22.0 GB (float32) does not fit host RAM) | — | — | — | — |
| fista | scatter | 0.16 | — | n/a (dense matrix 44.0 GB (float32) does not fit host RAM) | n/a (dense matrix 44.0 GB (float32) does not fit host RAM) | — | — | — | — |
| fista | scatter | 0.32 | — | n/a (dense matrix 88.0 GB (float32) does not fit host RAM) | n/a (dense matrix 88.0 GB (float32) does not fit host RAM) | — | — | — | — |
| tv | hadamard | 0.080 | fwht / woodbury | 894 ± 50 (n=50) | 161 ± 1.6 (n=50) | 5.6x | 124 (B=50) | 666 | — |
| tv | hadamard | 0.160 | fwht / woodbury | 896 ± 55 (n=50) | 162 ± 1.2 (n=50) | 5.5x | 126 (B=50) | 675 | — |
| tv | hadamard | 0.320 | fwht / woodbury | 919 ± 60 (n=50) | 157 ± 5.9 (n=50) | 5.9x | 129 (B=50) | 691 | — |
| tv | scatter | 0.08 | — | n/a (TV with inner CG on a >=1.4 GB dense matrix: estimated >15 min per image) | n/a (TV with inner CG on a streamed matrix: estimated hours per image) | — | — | — | — |
| tv | scatter | 0.16 | — | n/a (TV with inner CG on a >=1.4 GB dense matrix: estimated >15 min per image) | n/a (TV with inner CG on a streamed matrix: estimated hours per image) | — | — | — | — |
| tv | scatter | 0.32 | — | n/a (TV with inner CG on a >=1.4 GB dense matrix: estimated >15 min per image) | n/a (TV with inner CG on a streamed matrix: estimated hours per image) | — | — | — | — |
