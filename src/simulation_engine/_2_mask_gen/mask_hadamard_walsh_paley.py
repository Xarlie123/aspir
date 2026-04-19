import logging
import numpy as np
from simulation_engine._2_mask_gen.mask import MaskABC

class MaskHadamardWalshPaley(MaskABC):
    """Clase para generar patrones Walsh-Paley dentro de un rango de índices especificado."""

    def __init__(self, img_size=64, min_idx=0, max_idx=None, logger=None):
        super().__init__(img_size, logger)
        self.logger.debug("Initializing MaskHadamardWalshPaley with img_size=%d, min_idx=%d, max_idx=%s",
                          img_size, min_idx, max_idx)
        self.validate_size()
        self.min_idx = min_idx
        total = img_size * img_size
        self.max_idx = max_idx if max_idx is not None else total
        self.logger.info("Walsh-Paley configuration -> img_size: %d, min_idx: %d, max_idx: %d",
                         img_size, self.min_idx, self.max_idx)

    def walsh_paley_transform(self, n):
        self.logger.debug("Computing Walsh-Paley transform for n=%d", n)
        Pk = np.array([[1]], dtype=np.float64)
        B1 = np.array([1, 1], dtype=np.float64)
        B2 = np.array([1, -1], dtype=np.float64)
        for _ in range(n):
            pk1 = np.kron(Pk, B1)
            pk2 = np.kron(Pk, B2)
            Pk = np.concatenate((pk1, pk2), axis=0)
        return Pk

    def generate_masks(self, progress_callback=None):
        self.logger.info("Generating Walsh-Paley masks")
        if not ((self.img_size & (self.img_size - 1)) == 0 and self.img_size > 0):
            msg = f"img_size is not a power of 2: {self.img_size}"
            self.logger.error(msg)
            raise ValueError("Image size must be a power of 2 for the Walsh-Paley matrix.")

        n = int(np.log2(self.img_size))
        W = self.walsh_paley_transform(n)
        total = self.img_size * self.img_size
        patterns = []
        for count, (i, j) in enumerate(((i, j) for i in range(self.img_size) for j in range(self.img_size)), 1):
            patterns.append(np.outer(W[i], W[j]))
            if progress_callback:
                progress_callback(count, total)
        arr = np.array(patterns)
        self.masks = arr[self.min_idx:self.max_idx]
        self.num_patterns = self.masks.shape[0]
        self.logger.info("Generated %d Walsh-Paley masks (indices %d:%d)",
                         self.num_patterns, self.min_idx, self.max_idx)