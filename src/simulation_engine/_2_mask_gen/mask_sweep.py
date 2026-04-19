
import logging
import numpy as np
from simulation_engine._2_mask_gen.mask import MaskABC

class MaskSweep(MaskABC):
    """Class for generating sweep patterns with configurable lines."""

    def __init__(self, img_size=64, parametros=None, logger=None):
        super().__init__(img_size, logger)
        self.logger.debug("Initializing MaskSweep with img_size=%d, parametros=%s", img_size, parametros)
        self.validate_size()
        if not isinstance(parametros, (list, tuple)) or not parametros:
            msg = "Parameters list is required for MaskSweep"
            self.logger.error(msg)
            raise ValueError(msg)
        self.parametros = parametros
        self.logger.info("Sweep configuration -> parameters: %s", parametros)

    def generate_masks(self, progress_callback=None):
        """
        Generate sweep masks using pixel-wise distance calculation.

        Each mask represents a single bar of width bar_width pixels,
        with bars separated by stride pixels in the perpendicular direction.

        This approach calculates which pixels belong to each bar using
        geometric distance, avoiding artifacts from polygon clipping.
        """
        self.logger.info("Generating Sweep masks")
        masks = []
        total = 0

        # First pass: count total masks for progress reporting
        for p in self.parametros:
            bar_width = p['bar_width']
            stride = p['stride']

            # Calculate max offset to cover entire image from corner to corner
            # The perpendicular distance from center to the farthest corner
            max_offset = int(np.ceil(self.img_size * np.sqrt(2) / 2)) + bar_width // 2
            num_masks = int(np.ceil(2 * max_offset / stride))
            total += num_masks
            self.logger.debug(
                "Angle=%.1f°: max_offset=%d, stride=%d -> ~%d masks",
                p['angle'], max_offset, stride, num_masks
            )

        current = 0
        center = np.array([self.img_size / 2, self.img_size / 2], dtype=np.float64)

        # Precompute pixel coordinate grids (y, x) for distance calculations
        y_coords, x_coords = np.mgrid[0:self.img_size, 0:self.img_size]
        # Shift to center-relative coordinates
        x_rel = x_coords - center[0]
        y_rel = y_coords - center[1]

        # Minimum pixel threshold (very small, just to filter empty masks)
        min_pixels = 1

        # Second pass: generate masks
        for p in self.parametros:
            angle_deg = p['angle']
            angle_rad = np.deg2rad(angle_deg)
            bar_width = p['bar_width']
            stride = p['stride']
            half_width = bar_width / 2.0

            # Perpendicular direction (normal to the bar)
            # For angle θ, bar runs along (cos θ, -sin θ)
            # Perpendicular is (-sin θ, -cos θ) or (sin θ, cos θ)
            perp_x = np.sin(angle_rad)
            perp_y = np.cos(angle_rad)

            # Calculate max offset for this angle
            max_offset = int(np.ceil(self.img_size * np.sqrt(2) / 2)) + bar_width // 2

            # Generate bars at each perpendicular offset
            for offset in np.arange(-max_offset, max_offset + stride, stride):
                # For each pixel, calculate its signed distance to the bar's center line
                # The bar's center line is at perpendicular distance 'offset' from image center
                # A pixel at (x, y) has perpendicular distance: x*perp_x + y*perp_y
                # The pixel is inside the bar if |distance - offset| <= half_width

                pixel_perp_dist = x_rel * perp_x + y_rel * perp_y
                distance_to_bar_center = np.abs(pixel_perp_dist - offset)

                # Pixel is inside bar if distance to bar center line <= half_width
                mask = distance_to_bar_center <= half_width

                # Only add non-empty masks
                pixel_count = mask.sum()
                if pixel_count >= min_pixels:
                    masks.append(mask.astype(np.uint8) * 255)
                    self.logger.debug(
                        "Mask generated at angle=%.1f°, offset=%.1f, pixels=%d",
                        angle_deg, offset, pixel_count
                    )

                current += 1
                if progress_callback and current <= total:
                    progress_callback(current, total)

        self.masks = np.array(masks)
        self.num_patterns = len(masks)
        self.logger.info("Generated %d Sweep masks", self.num_patterns)