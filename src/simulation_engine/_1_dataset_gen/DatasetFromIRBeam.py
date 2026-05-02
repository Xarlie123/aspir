
import numpy as np
import matplotlib.pyplot as plt
import LightPipes as lp
import pandas as pd
from tqdm import tqdm
from time import perf_counter
from simulation_engine._1_dataset_gen.dataset import DatasetABC


class BeamMode:
    """Constants for beam mode types."""
    GAUSSIAN = "gaussian"
    HERMITE_GAUSS = "hermite_gauss"
    LAGUERRE_GAUSS = "laguerre_gauss"
    DOUGHNUT = "doughnut"


class DatasetFromIRBeam(DatasetABC):
    """
    Creates a dataset generating IR beam profile images using LightPipes.

    Supports multiple beam modes:
    - Gaussian: Standard Gaussian beam (TEM00)
    - Hermite-Gauss: TEMnm modes with Hermite polynomials
    - Laguerre-Gauss: TEMpm modes with Laguerre polynomials
    - Doughnut: Laguerre-Gauss modes with orbital angular momentum

    Can add speckle noise to simulate real IR sensor behavior.
    """

    def __init__(self, name, img_size, num_images, seed=None, logger=None, data_format=None,
                 mode_distribution=None, speckle_noise=0.0, max_mode_order=3):
        """
        Initialize the IR beam dataset generator.

        Parameters:
            name: Dataset name
            img_size: Image dimension (square images)
            num_images: Number of images to generate
            seed: Random seed for reproducibility
            logger: Logger instance
            data_format: Data format for quantization (FP32, INT8, INT4)
            mode_distribution: Dict with mode percentages, e.g.:
                {"gaussian": 25, "hermite_gauss": 25, "laguerre_gauss": 25, "doughnut": 25}
                Percentages should sum to 100.
            speckle_noise: Speckle noise level (0.0 to 1.0). 0 = no noise.
            max_mode_order: Maximum order for HG/LG modes (n, m values)
        """
        super().__init__(name, img_size, logger, data_format=data_format)
        self.num_images = num_images
        self.seed = seed
        self.size = 15 * lp.mm
        # 10.6 μm — CO₂ laser line, the canonical "thermal IR" working
        # wavelength for the single-pixel imaging setup ASPIR targets.
        # The previous 410 nm default was visible (violet), which made
        # any "IR beam" dataset a misnomer.
        self.wavelength = 10.6 * lp.um
        self.N = img_size
        self.R = 3 * lp.mm
        self.w0 = 3 * lp.mm  # Default beam waist

        # Mode distribution (default: 100% Gaussian for backward compatibility)
        if mode_distribution is None:
            mode_distribution = {BeamMode.GAUSSIAN: 100}
        self.mode_distribution = self._normalize_distribution(mode_distribution)

        # Noise parameters
        self.speckle_noise = max(0.0, min(1.0, speckle_noise))

        # Mode order range
        self.max_mode_order = max(1, max_mode_order)

        self.logger.debug(
            "Initializing IRBeam: %s, size %d, seed %s, format %s, modes %s, noise %.2f",
            name, img_size, seed, self.data_format, self.mode_distribution, self.speckle_noise
        )

    def _normalize_distribution(self, distribution):
        """Normalize mode distribution to ensure percentages sum to 100."""
        total = sum(distribution.values())
        if total == 0:
            return {BeamMode.GAUSSIAN: 100}
        return {k: (v / total) * 100 for k, v in distribution.items()}

    def get_mode_counts(self):
        """Calculate number of images for each mode based on distribution."""
        counts = {}
        remaining = self.num_images

        # Sort modes by percentage (descending) to assign largest first
        sorted_modes = sorted(self.mode_distribution.items(), key=lambda x: -x[1])

        for i, (mode, percentage) in enumerate(sorted_modes):
            if i == len(sorted_modes) - 1:
                # Last mode gets remaining images
                counts[mode] = remaining
            else:
                count = int(round(self.num_images * percentage / 100))
                count = min(count, remaining)
                counts[mode] = count
                remaining -= count

        return counts

    def load_data(self, progress_callback=None):
        self.logger.info("Generating %d IR beam profiles...", self.num_images)
        start = perf_counter()
        mats, params = self.generate_intensity_profiles(self.num_images, progress_callback)
        self.data = list(mats)
        self.df = pd.concat([
            self.to_dataframe(),
            pd.DataFrame(params, columns=["mode_type", "mode_n", "mode_m", "center_x", "center_y"])
        ], axis=1)
        # Apply data format conversion
        self.apply_data_format()
        elapsed = perf_counter() - start
        self.logger.info("Generated %d profiles in %.3f s (format: %s)", self.num_images, elapsed, self.data_format)

    def to_dataframe(self):
        self.logger.debug("Converting %d matrices to DataFrame", len(self.data))
        data_flat = [img.flatten() for img in self.data]
        return pd.DataFrame(data_flat)

    def _generate_gaussian(self, F, center_x=0, center_y=0):
        """Generate standard Gaussian beam (TEM00)."""
        # Apply random position offset
        sigma_x = np.random.uniform(1, 3) * lp.mm
        sigma_y = np.random.uniform(1, 3) * lp.mm
        angle = np.random.uniform(-np.pi/2, np.pi/2)

        x, y = np.meshgrid(
            np.linspace(-self.size/2, self.size/2, self.N),
            np.linspace(-self.size/2, self.size/2, self.N)
        )
        x_rot = (x - center_x) * np.cos(angle) - (y - center_y) * np.sin(angle) + center_x
        y_rot = (x - center_x) * np.sin(angle) + (y - center_y) * np.cos(angle) + center_y
        gaussian = np.exp(-((x_rot - center_x)**2 / sigma_x**2 + (y_rot - center_y)**2 / sigma_y**2))
        gaussian = np.clip(gaussian, 0, 1)

        F_mod = lp.SubIntensity(gaussian, F)
        return lp.Intensity(0, F_mod)

    def _generate_hermite_gauss(self, n, m):
        """
        Generate Hermite-Gauss TEMnm mode.

        Parameters:
            n: Horizontal mode order (0, 1, 2, ...)
            m: Vertical mode order (0, 1, 2, ...)
        """
        F = lp.Begin(self.size, self.wavelength, self.N)
        # GaussHermite(n, m, A, w0, Field) - amplitude A=1
        F = lp.GaussHermite(n, m, 1.0, self.w0, F)
        return lp.Intensity(0, F)

    def _generate_laguerre_gauss(self, p, m):
        """
        Generate Laguerre-Gauss TEMpm mode.

        Parameters:
            p: Radial mode order (0, 1, 2, ...)
            m: Azimuthal mode order (0, 1, 2, ...)
        """
        F = lp.Begin(self.size, self.wavelength, self.N)
        # GaussLaguerre(p, m, A, w0, Field) - amplitude A=1
        F = lp.GaussLaguerre(p, m, 1.0, self.w0, F)
        return lp.Intensity(0, F)

    def _generate_doughnut(self, m):
        """
        Generate Doughnut mode (Laguerre-Gauss with orbital angular momentum).

        Parameters:
            m: Azimuthal mode order (1, 2, 3, ...) - determines number of rings
        """
        F = lp.Begin(self.size, self.wavelength, self.N)
        # GaussBeam with doughnut=True, n=0 (radial), m=azimuthal order
        F = lp.GaussBeam(F, self.w0, doughnut=True, n=0, m=m)
        return lp.Intensity(0, F)

    def _add_speckle_noise(self, intensity):
        """
        Add speckle noise to simulate IR sensor behavior.

        Speckle noise follows a multiplicative model: I_noisy = I * (1 + noise)
        where noise is drawn from a Rayleigh-like distribution.
        """
        if self.speckle_noise <= 0:
            return intensity

        # Generate speckle pattern (multiplicative noise)
        noise = np.random.exponential(scale=self.speckle_noise, size=intensity.shape)
        noisy = intensity * (1 + noise - self.speckle_noise)  # Center around original

        # Normalize to [0, 1]
        noisy = np.clip(noisy, 0, None)
        if noisy.max() > 0:
            noisy = noisy / noisy.max()

        return noisy

    def _apply_random_transform(self, intensity):
        """Apply random position offset and scaling to the intensity pattern."""
        # Random offset (shift the pattern)
        offset_x = np.random.randint(-self.N // 8, self.N // 8 + 1)
        offset_y = np.random.randint(-self.N // 8, self.N // 8 + 1)

        # Roll the array to apply offset
        intensity = np.roll(intensity, offset_x, axis=1)
        intensity = np.roll(intensity, offset_y, axis=0)

        # Random intensity scaling
        scale = np.random.uniform(0.7, 1.0)
        intensity = intensity * scale

        return intensity

    def generate_intensity_profiles(self, num_images, progress_callback=None):
        if self.seed is not None:
            np.random.seed(self.seed)

        intensity_matrices = np.zeros((num_images, self.N, self.N))
        # Parameters: mode_type, mode_n, mode_m, center_x, center_y
        parameters = []

        # Get count for each mode
        mode_counts = self.get_mode_counts()
        self.logger.info("Mode distribution: %s", mode_counts)

        # Create list of (mode_type, count) pairs and shuffle the order of generation
        mode_list = []
        for mode_type, count in mode_counts.items():
            mode_list.extend([mode_type] * count)
        np.random.shuffle(mode_list)

        # Initialize base field for Gaussian mode
        F_base = lp.Begin(self.size, self.wavelength, self.N)
        F_base = lp.CircAperture(self.R, 0, 0, F_base)
        F_base = lp.GaussBeam(F_base, w0=0.5 * lp.mm)

        for i in tqdm(range(num_images), desc="Generating IR profiles"):
            mode_type = mode_list[i]
            mode_n, mode_m = 0, 0
            center_x, center_y = 0, 0

            if mode_type == BeamMode.GAUSSIAN:
                center_x = np.random.uniform(-3, 3) * lp.mm
                center_y = np.random.uniform(-3, 3) * lp.mm
                intensity = self._generate_gaussian(F_base, center_x, center_y)
                center_x /= lp.mm
                center_y /= lp.mm

            elif mode_type == BeamMode.HERMITE_GAUSS:
                mode_n = np.random.randint(0, self.max_mode_order + 1)
                mode_m = np.random.randint(0, self.max_mode_order + 1)
                intensity = self._generate_hermite_gauss(mode_n, mode_m)
                intensity = self._apply_random_transform(intensity)

            elif mode_type == BeamMode.LAGUERRE_GAUSS:
                mode_n = np.random.randint(0, self.max_mode_order + 1)  # radial p
                mode_m = np.random.randint(0, self.max_mode_order + 1)  # azimuthal m
                intensity = self._generate_laguerre_gauss(mode_n, mode_m)
                intensity = self._apply_random_transform(intensity)

            elif mode_type == BeamMode.DOUGHNUT:
                mode_m = np.random.randint(1, self.max_mode_order + 1)  # m >= 1 for doughnut
                intensity = self._generate_doughnut(mode_m)
                intensity = self._apply_random_transform(intensity)

            else:
                # Fallback to Gaussian
                intensity = self._generate_gaussian(F_base, 0, 0)

            # Add speckle noise if enabled
            intensity = self._add_speckle_noise(intensity)

            # Normalize intensity to [0, 1]
            if intensity.max() > 0:
                intensity = intensity / intensity.max()

            intensity_matrices[i] = intensity
            parameters.append([mode_type, mode_n, mode_m, center_x, center_y])

            if progress_callback:
                progress_callback(i + 1, num_images)

        return intensity_matrices, parameters

    def visualize_profiles(self, num_images=6):
        self.logger.info("Visualizing first %d IR profiles", num_images)
        mats, params = self.generate_intensity_profiles(num_images)
        plt.figure(figsize=(15, 10))
        for i in range(min(6, num_images)):
            plt.subplot(2, 3, i + 1)
            plt.imshow(mats[i], cmap='hot', extent=[-self.size/2/lp.mm, self.size/2/lp.mm,
                                                     -self.size/2/lp.mm, self.size/2/lp.mm])
            p = params[i]
            mode_type = p[0]
            n, m = p[1], p[2]
            if mode_type == BeamMode.GAUSSIAN:
                title = f"Gaussian\ncx={p[3]:.1f}mm, cy={p[4]:.1f}mm"
            elif mode_type == BeamMode.HERMITE_GAUSS:
                title = f"HG({n},{m})"
            elif mode_type == BeamMode.LAGUERRE_GAUSS:
                title = f"LG({n},{m})"
            elif mode_type == BeamMode.DOUGHNUT:
                title = f"Doughnut m={m}"
            else:
                title = f"Unknown"
            plt.title(title)
            plt.colorbar(label='Intensity')
        plt.tight_layout()
        plt.show()

    def get_mode_distribution_summary(self):
        """Return a summary of the mode distribution for display."""
        mode_counts = self.get_mode_counts()
        summary = []
        for mode, count in mode_counts.items():
            pct = (count / self.num_images) * 100 if self.num_images > 0 else 0
            summary.append({
                "mode": mode,
                "count": count,
                "percentage": pct
            })
        return summary
