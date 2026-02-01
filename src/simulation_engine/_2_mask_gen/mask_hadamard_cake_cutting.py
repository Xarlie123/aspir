# File: Simulacion/mascara_gen/mascara_hadamard_cake_cutting.py

import logging
import numpy as np
import torch
from simulation_engine._2_mask_gen.mask import MaskABC

class MaskHadamardCakeCutting(MaskABC):
    """Clase para generar patrones Hadamard usando el método 'cake cutting'."""

    def __init__(self,
                 img_size=64,
                 min_idx=0,
                 max_idx=None,
                 cutting_type='blocks',
                 use_cuda=False,
                 logger=None):
        # Solo pasamos img_size al ABC
        super().__init__(img_size)
        # Configuramos el logger
        if logger is None:
            self.logger = logging.getLogger(self.__class__.__name__)
        else:
            self.logger = logger.getChild(self.__class__.__name__)
        self.logger.debug("Initializing CakeCutting: img_size=%d, min_idx=%d, max_idx=%s, type=%s, use_cuda=%s",
                          img_size, min_idx, max_idx, cutting_type, use_cuda)

        self.validate_size()
        self.min_idx = min_idx
        self.max_idx = max_idx if max_idx is not None else img_size * img_size
        self.cutting_type = cutting_type
        # Solo activamos CUDA si realmente está disponible
        self.use_cuda = use_cuda and torch.cuda.is_available()
        if use_cuda and not self.use_cuda:
            self.logger.warning("GPU not available, using CPU instead.")

    def walsh_hadamard_transform(self, n):
        self.logger.debug("Generating base Walsh-Hadamard matrix (GPU=%s)", self.use_cuda)
        if self.use_cuda:
            w = torch.tensor([[1, 1], [1, -1]], dtype=torch.float64, device='cuda')
            for _ in range(n - 1):
                w = torch.cat([torch.cat([w, w],1), torch.cat([w, -w],1)],0)
            return w
        else:
            w = np.array([[1, 1], [1, -1]], dtype=np.float64)
            for _ in range(n - 1):
                w = np.block([[w, w], [w, -w]])
            return w

    def cake_cutting(self, H_big, dim, progress_callback=None,
                     phase_offset=0, phase_total=None):
        self.logger.debug("Starting cake cutting (phase 2)")
        length = 2 ** (2 * dim)
        if phase_total is None:
            phase_total = length
        blocks = np.zeros(length, dtype=np.int64)

        for ii in range(length):
            # reconstruimos el patrón 2D
            p = H_big[ii].reshape((2**dim, 2**dim))
            # ahora sí existe
            b0, b1, lbs = self.count_regions(p)
            blocks[ii] = (b0 + b1) if self.cutting_type == 'blocks' else lbs
            if progress_callback:
                progress_callback(phase_offset + ii + 1, phase_offset + phase_total)

        I = np.argsort(blocks) if self.cutting_type == 'blocks' else np.argsort(blocks)[::-1]
        H_new = H_big[I]
        return H_new, I, blocks

    def generate_masks(self, progress_callback=None):
        self.logger.info("Generating CakeCutting Hadamard masks")
        self.validate_size()

        n = int(np.log2(self.img_size))
        total = self.img_size ** 2
        phase1 = total
        phase2 = total
        overall = phase1 + phase2 + (self.max_idx - self.min_idx)

        # Fase 1: construimos el gran H
        H_small = self.walsh_hadamard_transform(n)
        if self.use_cuda:
            H_big = torch.kron(H_small, H_small).cpu().numpy()
        else:
            H_big = np.kron(H_small, H_small)
        if progress_callback:
            progress_callback(phase1, overall)

        # Fase 2: cake cutting
        H_cake, _, _ = self.cake_cutting(H_big, n,
                                         progress_callback=progress_callback,
                                         phase_offset=phase1,
                                         phase_total=phase2)

        # Fase 3: seleccionamos el rango
        patterns = []
        cnt = phase1 + phase2
        for i in range(self.min_idx, self.max_idx):
            cnt += 1
            if progress_callback:
                progress_callback(cnt, overall)
            patterns.append(H_cake[i].reshape((self.img_size, self.img_size)))
        if progress_callback:
            progress_callback(overall, overall)

        self.mascaras = np.array(patterns)
        self.num_patterns = self.mascaras.shape[0]
        self.logger.info("Generated %d CakeCutting masks", self.num_patterns)

    def count_regions(self, B):
        """
        Cuenta regiones conectadas (4-conectividad) de 0s y 1s
        y retorna (pieces_back, pieces_white, largest_block_size).
        """
        n = B.shape[0]
        visited = np.zeros((n, n), dtype=bool)
        pieces_back = pieces_white = largest_block_size = 0

        for i in range(n):
            for j in range(n):
                if not visited[i, j]:
                    seed = B[i, j]
                    size_block = self._fill_region(B, visited, i, j, seed)
                    if seed == 1:
                        pieces_white += 1
                    else:
                        pieces_back += 1
                    largest_block_size = max(largest_block_size, size_block)
        return pieces_back, pieces_white, largest_block_size

    def _fill_region(self, B, visited, i, j, seed):
        """
        Marca la región conectada iniciando en (i,j) y devuelve su tamaño.
        """
        stack = [(i, j)]
        visited[i, j] = True
        count_pixels = 0
        n = B.shape[0]
        while stack:
            cy, cx = stack.pop()
            count_pixels += 1
            for ny, nx in [(cy-1, cx), (cy+1, cx), (cy, cx-1), (cy, cx+1)]:
                if 0 <= ny < n and 0 <= nx < n and not visited[ny, nx] and B[ny, nx] == seed:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        return count_pixels
