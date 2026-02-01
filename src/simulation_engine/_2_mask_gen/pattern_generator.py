
# File: Simulacion/mascara_gen/pattern_generator.py
import logging
import numpy as np
import matplotlib.pyplot as plt

class PatternGenerator:
    """Combined generator of different patterns for analysis and visualization."""

    def __init__(self, img_size=64, number_steps=8, point_density=0.1, num_patterns=6, logger=None):
        if logger is None:
            self.logger = logging.getLogger(self.__class__.__name__)
        else:
            self.logger = logger.getChild(self.__class__.__name__)
        self.logger.debug("Initializing PatternGenerator img_size=%d, steps=%d, density=%.2f, patterns=%d",
                          img_size, number_steps, point_density, num_patterns)
        self.img_size = img_size
        self.number_steps = number_steps
        self.point_density = point_density
        self.num_patterns = num_patterns

    def generate_combined_sweep_patterns(self):
        self.logger.info("Generating combined Sweep patterns")
        if self.img_size % self.number_steps != 0:
            msg = "number_steps must be exact divisor of img_size"
            self.logger.error(msg)
            raise ValueError(msg)
        stripe = self.img_size // self.number_steps
        total = self.number_steps * 4
        patterns = np.zeros((total, self.img_size, self.img_size))
        for i in range(self.number_steps):
            patterns[i] = 0
            patterns[i, i*stripe:(i+1)*stripe, :] = 1
            patterns[i+self.number_steps] = 0
            patterns[i+self.number_steps, :, i*stripe:(i+1)*stripe] = 1
            # diagonals omitted for brevity
        return patterns

    def generate_scatter_patterns(self):
        self.logger.info("Generating Scatter patterns from PatternGenerator")
        if not (0 <= self.point_density <= 1):
            msg = "point_density out of range [0,1]"
            self.logger.error(msg)
            raise ValueError(msg)
        total_pixels = self.img_size * self.img_size
        num_points = int(total_pixels * self.point_density)
        masks = np.zeros((self.num_patterns, self.img_size, self.img_size))
        for i in range(self.num_patterns):
            x = np.random.randint(0, self.img_size, num_points)
            y = np.random.randint(0, self.img_size, num_points)
            masks[i, x, y] = 1
        return masks

    def plot_sweep_patterns(self, dataset_name):
        self.logger.info("Showing Sweep Patterns plot for %s", dataset_name)
        data, params = self.load_npz_to_dataframe(dataset_name + ".npz")
        first = data[0]
        fig, axes = plt.subplots(self.number_steps, 2, figsize=(8, self.number_steps*2))
        sweep = self.generate_combined_sweep_patterns()
        vmin, vmax = sweep.min(), sweep.max()
        for i in range(self.number_steps):
            axes[i,0].imshow(sweep[i], cmap='gray', vmin=vmin, vmax=vmax)
            axes[i,0].axis('off')
            masked = first * sweep[i]
            axes[i,1].imshow(masked, cmap='gray', vmin=first.min(), vmax=first.max())
            axes[i,1].axis('off')
        plt.tight_layout()
        plt.show()

    def plot_scatter_patterns(self, dataset_name):
        self.logger.info("Showing Scatter Patterns plot for %s", dataset_name)
        data, _ = self.load_npz_to_dataframe(dataset_name + ".npz")
        first = data[0]
        masks = self.generate_scatter_patterns()
        fig, axes = plt.subplots(self.num_patterns, 2, figsize=(8, self.num_patterns*2), constrained_layout=True)
        vmin, vmax = masks.min(), masks.max()
        for i in range(self.num_patterns):
            axes[i,0].imshow(masks[i], cmap='gray', vmin=vmin, vmax=vmax)
            axes[i,0].axis('off')
            axes[i,1].imshow(first*masks[i], cmap='gray', vmin=first.min(), vmax=first.max())
            axes[i,1].axis('off')
        plt.show()

    def load_npz_to_dataframe(self, dataset_name):
        self.logger.debug("Loading npz %s", dataset_name)
        data = np.load(dataset_name, allow_pickle=True)
        return data['intensity_matrices'], data['parameters']