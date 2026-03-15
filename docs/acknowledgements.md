# Acknowledgements

ASPIR is built on top of several open-source libraries. We gratefully acknowledge the developers and researchers behind these projects.

## Core Framework

- **[PyTorch](https://pytorch.org/)** — Deep learning framework used for all neural network models, training, and inference.
- **[PyQt5](https://www.riverbankcomputing.com/software/pyqt/)** — Python bindings for Qt, used for the graphical user interface.
- **[NumPy](https://numpy.org/)** — Fundamental package for numerical computing in Python.
- **[Matplotlib](https://matplotlib.org/)** — Plotting library used for all charts and visualizations in the application.

## Optical Simulation

- **[LightPipes](https://opticspy.github.io/lightpipes/)** — Optical simulation library used to generate synthetic infrared beam profiles (Gaussian, Hermite-Gauss, Laguerre-Gauss, Doughnut modes) in the IR Beam dataset generator.

## Image Quality Metrics

- **[LPIPS](https://github.com/richzhang/PerceptualSimilarity)** — Learned Perceptual Image Patch Similarity. Deep learning-based perceptual similarity metric using a pre-trained AlexNet backbone.
  > R. Zhang, P. Isola, A. A. Efros, E. Shechtman, O. Wang. *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric*. CVPR 2018.

- **[scikit-image](https://scikit-image.org/)** — Image processing library used for computing PSNR (`peak_signal_noise_ratio`) and SSIM (`structural_similarity`) quality metrics.

## Scientific Computing

- **[SciPy](https://scipy.org/)** — Used for Hadamard matrix generation (`scipy.linalg.hadamard`) and MATLAB file loading (`scipy.io.loadmat`) for the SVHN dataset.
- **[Pandas](https://pandas.pydata.org/)** — Data manipulation and analysis, used for structuring reconstruction results and dataset metadata.

## Image Processing

- **[OpenCV](https://opencv.org/)** — Image reading, writing, resizing, and color space conversion in dataset generators.
- **[Pillow](https://python-pillow.org/)** — Image manipulation used in mask visualization and UI components.

## GPU Monitoring and Profiling

- **[pynvml](https://github.com/gpuopenanalytics/pynvml)** — Python bindings for the NVIDIA Management Library (NVML). Used for GPU power and energy measurement on NVIDIA desktop GPUs.
- **[psutil](https://github.com/giampaolo/psutil)** — Cross-platform system monitoring. Used for CPU and RAM resource monitoring during batch test execution.
- **[PyTorch Profiler](https://pytorch.org/docs/stable/profiler.html)** — Layer-by-layer performance profiling of neural network operations.

## Datasets

- **[CelebA Small Images](https://www.kaggle.com/datasets/arnrob/celeba-small-images-dataset)** — Celebrity faces dataset hosted on Kaggle, used for face reconstruction benchmarks.
- **[SVHN](http://ufldl.stanford.edu/housenumbers/)** — Street View House Numbers dataset from Stanford University, used for digit recognition benchmarks.

## Documentation

- **[Sphinx](https://www.sphinx-doc.org/)** with **[MyST Parser](https://myst-parser.readthedocs.io/)** — Documentation build system with Markdown support.
- **[Read the Docs](https://readthedocs.org/)** — Documentation hosting platform.
