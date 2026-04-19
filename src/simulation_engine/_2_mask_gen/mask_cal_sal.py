
import numpy as np
import logging
from simulation_engine._2_mask_gen.mask import MaskABC

class MaskCalSal(MaskABC):
    """Generates máscaras Cal-Sal dentro de un rango de índices."""

    def __init__(self, img_size=64, min_idx=0, max_idx=None, logger=None):
        super().__init__(img_size, logger)
        self.validate_size()
        self.min_idx = min_idx
        self.max_idx = max_idx if max_idx is not None else img_size * img_size
        self.logger.debug(
            "Configured Cal-Sal: img_size=%d, min_idx=%d, max_idx=%d",
            img_size, min_idx, self.max_idx
        )

    def cal_sal_transform(self, n):
        self.logger.debug("Computing Cal-Sal transform for n=%d", n)
        size = 2 ** n
        w = np.zeros((size, size), dtype=np.int64)
        R = np.zeros((n, n), dtype=np.int64)

        if n == 1:
            R = np.array([[1]], dtype=np.int64)
        else:
            for k in range(1, n):
                R[k - 1, n - k] = 1
                R[n - k, k] = 1
            R[n - 1, 0] = 1

        for j in range(size):
            bj = np.array(list(np.binary_repr(j, width=n)), dtype=int)[::-1]
            exponent = np.dot(bj, R.dot(bj))
            w[j, j] = (-1) ** exponent
            for k in range(j + 1, size):
                bk = np.array(list(np.binary_repr(k, width=n)), dtype=int)[::-1]
                exponent = np.dot(bj, R.dot(bk))
                entry = (-1) ** exponent
                w[j, k] = entry
                w[k, j] = entry
        self.logger.debug("Cal-Sal transform computed, size %dx%d", size, size)
        return w

    def generate_masks(self, progress_callback=None):
        self.logger.info("Generating Cal-Sal masks")
        # Validate power of 2
        if not ((self.img_size & (self.img_size - 1)) == 0 and self.img_size > 0):
            self.logger.error("img_size is not a power of 2: %d", self.img_size)
            raise ValueError("Image size must be a power of 2 for the Cal-Sal transform.")

        n = int(np.log2(self.img_size))
        W = self.cal_sal_transform(n)
        total = self.img_size * self.img_size
        patterns = []
        count = 0

        for i in range(self.img_size):
            for j in range(self.img_size):
                patterns.append(np.outer(W[i], W[j]))
                count += 1
                if progress_callback:
                    progress_callback(count, total)
        # recorte
        arr = np.array(patterns)
        self.masks = arr[self.min_idx:self.max_idx]
        self.num_patterns = self.masks.shape[0]
        self.logger.info(
            "Generated %d Cal-Sal masks (indices %d:%d)",
            self.num_patterns, self.min_idx, self.max_idx
        )
