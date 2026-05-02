
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
