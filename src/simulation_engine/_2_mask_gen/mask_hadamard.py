# File: Simulacion/mascara_gen/mascara_hadamard.py

import logging
import numpy as np
from simulation_engine._2_mask_gen.mask import MaskABC

class MaskHadamard(MaskABC):
    """Clase para generar patrones de Hadamard dentro de un rango de índices especificado."""

    def __init__(self, img_size=64, min_idx=0, max_idx=None, logger=None):
        super().__init__(img_size, logger)
        self.logger.debug("Initializing MaskHadamard with img_size=%d, min_idx=%d, max_idx=%s",
                          img_size, min_idx, max_idx)
        self.validate_size()
        self.min_idx = min_idx
        self.max_idx = max_idx if max_idx is not None else img_size * img_size
        self.logger.info("Hadamard configuration -> img_size: %d, min_idx: %d, max_idx: %d",
                         img_size, self.min_idx, self.max_idx)

    def walsh_hadamard_transform(self, n):
        self.logger.debug("Computing Walsh-Hadamard transform for n=%d", n)
        w = np.array([[1, 1], [1, -1]], dtype=np.float64)
        for _ in range(n - 1):
            w = np.block([[w, w], [w, -w]])
        return w

    def generate_masks(self, progress_callback=None):
        self.logger.info("Generating Hadamard masks")
        if not ((self.img_size & (self.img_size - 1)) == 0 and self.img_size > 0):
            msg = f"img_size is not a power of 2: {self.img_size}"
            self.logger.error(msg)
            raise ValueError("Image size must be a power of 2 for the Hadamard matrix.")

        n = int(np.log2(self.img_size))
        H = self.walsh_hadamard_transform(n)
        total = self.img_size * self.img_size
        patterns = []
        for count, (i, j) in enumerate(((i, j) for i in range(self.img_size) for j in range(self.img_size)), 1):
            patterns.append(np.outer(H[i], H[j]))
            if progress_callback:
                progress_callback(count, total)
        arr = np.array(patterns)
        self.mascaras = arr[self.min_idx:self.max_idx]
        self.num_patterns = self.mascaras.shape[0]
        self.logger.info("Generated %d Hadamard masks (indices %d:%d)",
                         self.num_patterns, self.min_idx, self.max_idx)