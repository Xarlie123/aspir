
import logging
import time
import numpy as np

try:
    import torch
except ImportError:
    torch = None

class TimingAnalyzer:
    """
    Benchmarks inference time of a PyTorch model, returning mean ± std.
    """
    def __init__(self, model, device='cpu', warmup_runs=20):
        if torch is None:
            raise ImportError("PyTorch is required for TimingAnalyzer")

        # Set up logger for timing analysis
        self.logger = logging.getLogger("ASPIR.TimingAnalyzer")
        self.logger.setLevel(logging.DEBUG)
        self.logger.debug(
            "Initializing TimingAnalyzer: device=%s, warmup_runs=%d",
            device, warmup_runs
        )

        self.model = model.to(device)
        self.device = device
        self.warmup_runs = warmup_runs

    def time_inference(self, input_tensor, n_runs=20):
        """
        Warm up, then run `n_runs` inferences measuring time.
        Returns (mean_seconds, std_seconds).
        """
        self.logger.info("Starting warm-up (%d runs)", self.warmup_runs)
        for _ in range(self.warmup_runs):
            _ = self.model(input_tensor.to(self.device))
            if self.device == 'cuda':
                torch.cuda.synchronize()

        self.logger.info("Warm-up complete; starting timed runs (%d runs)", n_runs)
        times = []
        for i in range(n_runs):
            if self.device == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            _ = self.model(input_tensor.to(self.device))

            if self.device == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            interval = t1 - t0
            times.append(interval)
            self.logger.debug("Run %d: %.6fs", i + 1, interval)

        arr = np.array(times, dtype=float)
        mean, std = float(arr.mean()), float(arr.std())
        self.logger.info(
            "Inference timing complete: mean=%.6fs, std=%.6fs", mean, std
        )
        return mean, std


def time_reconstruction(applicator, image_indices, warmup_runs=2, repeats=3):
    """Per-image reconstruction latency with the same protocol as NN inference.

    ``synchronize()`` before and after every timed call (so GPU kernels are
    counted when they finish, not when they are queued), ``warmup_runs`` untimed
    calls first — they also absorb one-off costs such as building the operator,
    the cached pseudoinverse or the Lipschitz constant — then ``repeats``
    passes over ``image_indices``. Each call is ``applicator.process_image``, so
    it includes simulating the single-pixel measurements, exactly as before.

    The applicator is used by duck typing (``process_image``, and optionally
    ``synchronize``, ``reconstruction_device``, ``backend``,
    ``reconstruction_precision``, ``last_operator_kind``), which keeps this
    stage independent of the reconstruction stage.
    """
    sync = getattr(applicator, "synchronize", None) or (lambda: None)
    indices = list(image_indices)
    if not indices:
        raise ValueError("time_reconstruction needs at least one image index")

    n_warm = min(int(warmup_runs), len(indices))
    for idx in indices[:n_warm]:
        applicator.process_image(idx)
        sync()

    times_ms = []
    for _ in range(max(1, int(repeats))):
        for idx in indices:
            sync()
            t0 = time.perf_counter()
            applicator.process_image(idx)
            sync()
            times_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.asarray(times_ms)
    backend = getattr(applicator, "backend", "legacy")
    return {
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std()),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "samples": int(arr.size),
        "warmup": n_warm,
        "repeats": max(1, int(repeats)),
        "device": getattr(applicator, "reconstruction_device", "cpu"),
        "backend": backend,
        "precision": getattr(applicator, "reconstruction_precision", "float64"),
        "operator": getattr(applicator, "last_operator_kind", None)
        or ("explicit" if backend == "legacy" else None),
    }
