"""Patterns that were projected on real hardware, loaded from a measurement file."""

import numpy as np

from simulation_engine._2_mask_gen.mask import MaskABC


class MaskMeasured(MaskABC):
    """A fixed, read-only pattern set that came with measured data.

    Nothing is generated: ``masks`` holds the patterns in the order of the
    measured bucket values. Bipolar +-1 Hadamard patterns are recognised by the
    reconstruction backend, which then applies them with the fast
    Walsh-Hadamard transform whatever their ordering.
    """

    def __init__(self, patterns, logger=None):
        patterns = np.asarray(patterns)
        if patterns.ndim != 3 or patterns.shape[1] != patterns.shape[2]:
            raise ValueError(f"patterns must be (M, N, N), got {patterns.shape}")
        super().__init__(int(patterns.shape[1]), logger)
        self.masks = patterns.astype(np.float32)
        self.num_patterns = int(patterns.shape[0])
        self.read_only = True
        # +-1 patterns have a native linear reconstruction (Hadamard Linear).
        self.is_bipolar = bool(np.all(np.abs(self.masks) == 1))

    def generate_masks(self, progress_callback=None):
        """The patterns are fixed by the measurement; there is nothing to generate."""
        if progress_callback:
            progress_callback(1, 1)
