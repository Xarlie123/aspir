
import logging
import numpy as np
from simulation_engine._2_mask_gen.mask import MaskABC

class MaskScatter(MaskABC):
    """Clase para generar patrones de dispersión aleatoria."""

    def __init__(self, img_size=64, point_density=0.1, num_patterns=6, seed=10, logger=None):
        super().__init__(img_size, logger)
        self.logger.debug("Initializing MaskScatter with img_size=%d, point_density=%.2f, num_patterns=%d, seed=%d",
                          img_size, point_density, num_patterns, seed)
        self.point_density = point_density
        self.num_patterns = num_patterns
        self.validate_size()
        np.random.seed(seed)
        self.logger.info("Scatter configuration -> density: %.2f%%, patterns: %d", point_density*100, num_patterns)

    def generate_masks(self, progress_callback=None):
        self.logger.info("Generating Scatter masks")
        if not (0 <= self.point_density <= 100):
            msg = f"point_density out of range: {self.point_density}"
            self.logger.error(msg)
            raise ValueError("point_density must be in range [0, 100]")

        total_pixels = self.img_size * self.img_size
        num_points = int(total_pixels * self.point_density / 100.0)
        H = np.zeros((self.num_patterns, self.img_size, self.img_size))
        for i in range(self.num_patterns):
            if self.point_density == 100:
                H[i, :, :] = 1
            else:
                all_positions = np.arange(total_pixels)
                np.random.shuffle(all_positions)
                selected = all_positions[:num_points]
                x, y = np.divmod(selected, self.img_size)
                H[i, x, y] = 1
            if progress_callback:
                progress_callback(i + 1, self.num_patterns)
        self.masks = H
        self.num_patterns = H.shape[0]
        self.logger.info("Generated %d Scatter masks", self.num_patterns)
