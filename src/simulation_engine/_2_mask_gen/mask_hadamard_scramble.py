# File: Simulacion/mascara_gen/mascara_hadamard_scramble.py

import logging
import numpy as np
from scipy.linalg import hadamard
from simulation_engine._2_mask_gen.mask import MaskABC

class MaskHadamardScramble(MaskABC):
    """Generates patrones SBHE (scramble block Hadamard)."""

    def __init__(self, img_size=64, min_idx=0, max_idx=None,
                 seed=42, B=32, M=None, logger=None):
        super().__init__(img_size, logger)
        self.logger.debug("Initializing Scramble Hadamard with img_size=%d, B=%d, M=%s, seed=%d",
                          img_size, B, M, seed)
        self.validate_size()
        self.seed, self.B, self.M = seed, B, M
        self.min_idx = min_idx
        self.max_idx = max_idx if max_idx is not None else (self.M if self.M else img_size*img_size)

    def generate_masks(self, progress_callback=None):
        self.logger.info("Generating Hadamard Scramble masks")
        N = self.img_size**2
        Mtot = self.M if self.M else N
        if not ((self.B & (self.B-1))==0 and self.B>0):
            self.logger.error("B is not a power of 2: %d", self.B)
            raise ValueError("B must be a power of 2.")
        np.random.seed(self.seed)
        rowperm = np.random.permutation(N)[:Mtot]
        colperm = np.random.permutation(N)
        WB = hadamard(self.B).astype(np.float64)
        masks=[]
        for i in range(Mtot):
            r = rowperm[i]
            z = (r//self.B)*self.B
            part1 = np.zeros(z)
            part2 = WB[r%self.B]
            part3 = np.zeros(N-self.B-z)
            phi = np.concatenate([part1,part2,part3])[colperm]
            mask2D = phi.reshape((self.img_size,self.img_size))
            masks.append(mask2D)
            if progress_callback: progress_callback(i+1, Mtot)
        arr=np.array(masks)[self.min_idx:self.max_idx]
        self.mascaras=arr
        self.num_patterns=arr.shape[0]
        self.logger.info("Generated %d Scramble masks", self.num_patterns)
