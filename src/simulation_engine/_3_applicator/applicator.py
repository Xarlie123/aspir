from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

# Reconstruction backends. "torch" is the default: one PyTorch implementation per
# algorithm, run on the CPU or on CUDA. "legacy" keeps the original NumPy /
# pylops / pyproximal code as a reference path for parity tests; it is not
# exposed in the GUI.
BACKEND_CHOICES = ("torch", "legacy")


class ApplicatorABC(ABC):
    """Abstract base class for applying masks to images and datasets.

    Backend options (keyword-only in every subclass):

    ``device``   ``"auto"`` (CUDA if available, else CPU), ``"cpu"`` or ``"cuda"``.
    ``dtype``    working precision of the torch path, ``"float32"`` (default) or
                 ``"float64"``.
    ``backend``  ``"torch"`` (default) or ``"legacy"``.
    ``operator`` how the sensing matrix is applied on the torch path:
                 ``"auto"`` (fast Walsh-Hadamard for Hadamard-family masks, FFT
                 for grayscale Fourier masks, dense otherwise), ``"explicit"``,
                 ``"fwht"`` or ``"fft"``.
    ``batch_size`` images reconstructed per call on the torch path.
    ``max_device_bytes`` cap on the dense matrix kept resident on the device;
                 above it the matrix is streamed in row blocks.
    """

    def __init__(self, dataset, mask, device="auto", dtype="float32", backend="torch",
                 operator="auto", batch_size=32, max_device_bytes=None):
        self.name = dataset.name
        self.img_size = dataset.img_size
        self.dataset = dataset  # Dataset object
        self.mask = mask  # Mask object
        self.data = pd.DataFrame()  # DataFrame to store processed images
        self.dataset_type = self.__class__.__name__
        self.num_images = len(dataset.data)

        backend = str(backend).strip().lower()
        if backend not in BACKEND_CHOICES:
            raise ValueError(f"backend must be one of {BACKEND_CHOICES}, got {backend!r}")
        self.backend = backend
        self.device_request = device
        self.dtype_request = dtype
        self.operator_mode = operator
        self.batch_size = max(1, int(batch_size))
        self.max_device_bytes = max_device_bytes
        self._torch_device = None
        self._torch_dtype = None
        self._op_cache = {}
        self._solver_cache = {}
        self.last_operator_kind = None

    @abstractmethod
    def apply_mask_range(self, idx_mask_min, idx_mask_max, idx_image):
        """Abstract method to apply a range of masks to an image and accumulate values."""
        pass

    @abstractmethod
    def process_image(self, idx):
        """Abstract method to process a single image by applying masks."""
        pass

    @abstractmethod
    def process_dataset(self, idx_mask_min=0, idx_mask_max=None):
        """Abstract method to process entire dataset by applying masks."""
        pass

    # ------------------------------------------------------------------
    # Device / precision
    # ------------------------------------------------------------------
    @property
    def torch_device(self):
        """Resolved ``torch.device`` (resolved lazily, on first use)."""
        if self._torch_device is None:
            from simulation_engine._3_applicator.torch_backend import resolve_device
            self._torch_device = resolve_device(self.device_request)
        return self._torch_device

    @property
    def torch_dtype(self):
        if self._torch_dtype is None:
            from simulation_engine._3_applicator.torch_backend import resolve_dtype
            self._torch_dtype = resolve_dtype(self.dtype_request)
        return self._torch_dtype

    @property
    def reconstruction_device(self) -> str:
        """Device the reconstruction actually runs on: ``"cuda"`` or ``"cpu"``."""
        return "cpu" if self.backend == "legacy" else self.torch_device.type

    @property
    def reconstruction_precision(self) -> str:
        return "float64" if self.backend == "legacy" else str(self.torch_dtype).replace("torch.", "")

    def release_device_memory(self):
        """Drop cached operators and solver state; both are rebuilt on demand.

        Call it once a dataset has been reconstructed and the device is needed
        for something else (NN training): on a 4 GB GPU a resident sensing
        matrix plus a cached pseudoinverse can take hundreds of MB.
        """
        self._op_cache.clear()
        self._solver_cache.clear()
        if self._torch_device is not None and self._torch_device.type == "cuda":
            import torch
            torch.cuda.empty_cache()

    def synchronize(self):
        """Wait for queued device work (for timing); a no-op on the CPU."""
        if self.backend == "torch" and self.torch_device.type == "cuda":
            import torch
            torch.cuda.synchronize(self.torch_device)

    # ------------------------------------------------------------------
    # Torch path shared by every algorithm
    # ------------------------------------------------------------------
    def _num_patterns(self):
        masks = getattr(self.mask, "masks", None)
        if masks is not None and len(masks) > 0:
            return len(masks)
        for attr in ("hadamard_indices", "phases"):
            meta = getattr(self.mask, attr, None)
            if meta is not None:
                return len(meta)
        return 0

    def _ensure_patterns(self):
        """Generate the patterns if the mask has neither patterns nor metadata."""
        if self._num_patterns() == 0 and hasattr(self.mask, "generate_masks"):
            self.mask.generate_masks()
        if self._num_patterns() == 0:
            raise ValueError(
                "The mask has no available patterns in 'masks'. "
                "Make sure to generate them before using the applicator."
            )

    def _operator(self, idx_min, idx_max):
        key = (idx_min, idx_max)
        op = self._op_cache.get(key)
        if op is None:
            from simulation_engine._3_applicator.torch_backend import build_operator
            op = build_operator(self.mask, idx_min, idx_max, self.torch_device, self.torch_dtype,
                                mode=self.operator_mode, max_device_bytes=self.max_device_bytes)
            self._op_cache[key] = op
        self.last_operator_kind = op.kind
        return op

    def _images(self, indices):
        import torch
        np_dtype = np.float64 if self.torch_dtype == torch.float64 else np.float32
        batch = np.stack([np.asarray(self.dataset.data[i], dtype=np_dtype).reshape(-1)
                          for i in indices])
        return torch.from_numpy(batch).to(self.torch_device)

    def _torch_solve(self, op, Y, key):
        """Algorithm-specific reconstruction of measurements ``Y`` (B, M) -> (B, N)."""
        raise NotImplementedError

    def _mask_range(self, idx_mask_min, idx_mask_max):
        n_total = self._num_patterns()
        idx_min = max(0, int(idx_mask_min))
        idx_max = n_total if idx_mask_max is None else min(int(idx_mask_max), n_total)
        return idx_min, idx_max

    def _solve_measurements(self, op, Y, key, n):
        """Run the algorithm on bucket values ``Y`` (n, M); returns (n, H, W) float64."""
        R = self._torch_solve(op, Y, key)
        return R.detach().to("cpu").double().numpy().reshape(n, self.img_size, self.img_size)

    def _torch_reconstruct(self, idx_mask_min, idx_mask_max, indices):
        """Simulate the detector and reconstruct ``indices``; returns (B, H, W) float64."""
        self._ensure_patterns()
        idx_min, idx_max = self._mask_range(idx_mask_min, idx_mask_max)
        if idx_max <= idx_min:
            return np.zeros((len(indices), self.img_size, self.img_size), dtype=np.float64)
        op = self._operator(idx_min, idx_max)
        X = self._images(indices)
        Y = op.forward(X)                     # simulated single-pixel detector values
        return self._solve_measurements(op, Y, (idx_min, idx_max), len(indices))

    def reconstruct(self, y, idx_mask_min=0, idx_mask_max=None):
        """Reconstruct *measured* bucket values, without simulating the acquisition.

        ``y`` holds one value per pattern of ``self.mask``, in the mask's order:
        ``(M,)`` for one frame or ``(F, M)`` for several. Rows
        ``[idx_mask_min, idx_mask_max)`` are used, exactly as for simulated
        data; the simulated path runs the same solver on the values it
        computes itself. Returns ``(H, W)`` for 1-D ``y``, else ``(F, H, W)``.

        Runs on the torch backend (either device); the legacy reference path
        only exists for simulated data.
        """
        if self.backend == "legacy":
            raise NotImplementedError(
                "Measured data are reconstructed on the torch backend; backend='legacy' "
                "only reproduces the original simulated pipeline.")
        import torch

        self._ensure_patterns()
        y = np.asarray(y, dtype=np.float64)
        single = y.ndim == 1
        Y2 = np.atleast_2d(y)
        if Y2.shape[1] != self._num_patterns():
            raise ValueError(f"y has {Y2.shape[1]} values per frame but the mask has "
                             f"{self._num_patterns()} patterns")
        idx_min, idx_max = self._mask_range(idx_mask_min, idx_mask_max)
        if idx_max <= idx_min:
            out = np.zeros((Y2.shape[0], self.img_size, self.img_size), dtype=np.float64)
        else:
            op = self._operator(idx_min, idx_max)
            Yt = torch.as_tensor(Y2[:, idx_min:idx_max], dtype=self.torch_dtype,
                                 device=self.torch_device)
            out = self._solve_measurements(op, Yt, (idx_min, idx_max), Y2.shape[0])
        self.reconstructed_image = out[-1]
        return out[0] if single else out

    def _torch_process_dataset(self, idx_mask_min=0, idx_mask_max=None):
        """Batched torch reconstruction of the whole dataset (one row per image)."""
        rows = []
        n = len(self.dataset.data)
        for start in range(0, n, self.batch_size):
            idx = list(range(start, min(n, start + self.batch_size)))
            rec = self._torch_reconstruct(idx_mask_min, idx_mask_max, idx)
            rows.extend(r.reshape(-1) for r in rec)
        df = pd.DataFrame(np.asarray(rows)) if rows else pd.DataFrame()
        self.reconstructed_dataset = df
        return df
