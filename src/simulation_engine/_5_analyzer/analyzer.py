
import logging
from .analyzer_noise import NoiseAnalyzer
from .analyzer_timing import TimingAnalyzer
from .analyzer_energy import EnergyAnalyzer

class Analyzer:
    """
    Facade combining noise-metrics, timing, and energy analyzers.
    Supports different data formats for embedded system testing.
    """
    def __init__(self,
                 originals,
                 noisy,
                 reconstructions,
                 model=None,
                 device='cpu',
                 warmup_runs=20,
                 data_format=None,
                 enable_energy_analysis=False,
                 pmlib_server_ip="127.0.0.1",
                 pmlib_server_port=6526):
        # Set up a logger for this facade
        self.logger = logging.getLogger("ASPIR.Analyzer")
        self.logger.setLevel(logging.DEBUG)
        self.logger.debug(
            "Initializing Analyzer: model=%s, device=%s, warmup_runs=%d, data_format=%s",
            type(model).__name__ if model is not None else None,
            device, warmup_runs, data_format
        )

        # Store data format
        self.data_format = data_format

        # PSNR/SSIM analyzer with data format
        self.noise = NoiseAnalyzer(originals, noisy, reconstructions, data_format=data_format)
        self.logger.debug("NoiseAnalyzer instantiated with format: %s", data_format)

        # Timing analyzer (only if model provided)
        self.timing = None
        if model is not None:
            self.timing = TimingAnalyzer(model, device=device, warmup_runs=warmup_runs)
            self.logger.debug("TimingAnalyzer instantiated")

        # Energy analyzer (optional)
        self.energy = None
        if enable_energy_analysis and model is not None:
            self.energy = EnergyAnalyzer(
                model=model,
                device=device,
                warmup_runs=warmup_runs,
                pmlib_server_ip=pmlib_server_ip,
                pmlib_server_port=pmlib_server_port,
                logger=self.logger
            )
            self.logger.debug("EnergyAnalyzer instantiated")

        # Arrays to hold per-image timing stats
        self.per_image_mean:    list[float] = []
        self.per_image_std:     list[float] = []
        self.mean_overall:      float      = 0.0
        self.std_overall:       float      = 0.0

    def analyze_noise(self):
        """
        Delegate to NoiseAnalyzer.analyze()
        """
        self.logger.info("Starting noise analysis via NoiseAnalyzer")
        results = self.noise.analyze()
        self.logger.info(
            "Noise analysis complete: %d frames -> psnr_mean=%.4f, ssim_mean=%.4f",
            len(self.noise.psnr_rec),
            self.noise.psnr_rec_mean,
            self.noise.ssim_rec_mean
        )
        return results

    def time_inference(self, input_tensors, n_runs=20):
        """
        If input_tensors is a single tensor, returns (mean, std) and stores in
        mean_overall/std_overall. If it's a sequence of tensors, runs timing
        on each, stores per_image_mean/std, and computes overall mean/std.
        """
        if self.timing is None:
            msg = "No model provided for timing."
            self.logger.error(msg)
            raise RuntimeError(msg)

        # helper to time one tensor
        def _time_one(tensor):
            self.logger.debug("Timing single tensor for %d runs", n_runs)
            return self.timing.time_inference(tensor, n_runs=n_runs)

        # Case 1: list/tuple -> per-image
        if isinstance(input_tensors, (list, tuple)):
            means, stds = [], []
            for i, t in enumerate(input_tensors):
                m, s = _time_one(t)
                self.logger.debug("Image %d: mean=%.6fs, std=%.6fs", i, m, s)
                means.append(m)
                stds.append(s)
            self.per_image_mean = means
            self.per_image_std  = stds
            # overall stats
            import numpy as _np
            self.mean_overall = float(_np.mean(means))
            self.std_overall  = float(_np.std(means))
            self.logger.info(
                "Per-image timing complete: overall mean=%.6fs, std=%.6fs",
                self.mean_overall, self.std_overall
            )
            return means, stds

        # Case 2: single tensor
        mean, std = _time_one(input_tensors)
        self.mean_overall = mean
        self.std_overall  = std
        self.logger.info(
            "Inference timing complete: mean=%.6fs, std=%.6fs", mean, std
        )
        return mean, std

    def analyze_energy(self, input_tensors, n_runs=10):
        """
        Analyze energy consumption during inference.

        Args:
            input_tensors: Single tensor or list of tensors
            n_runs: Number of measurement runs per image

        Returns:
            EnergyAnalysisResult with detailed energy measurements
        """
        if self.energy is None:
            msg = "Energy analyzer not initialized. Set enable_energy_analysis=True."
            self.logger.error(msg)
            raise RuntimeError(msg)

        self.logger.info("Starting energy analysis")
        result = self.energy.analyze_inference(input_tensors, n_runs=n_runs)
        self.logger.info(
            "Energy analysis complete: mean=%.3f mJ, power=%.2f W",
            result.mean_energy_mj, result.mean_power_watts
        )
        return result

    def shutdown(self):
        """Clean up resources (especially energy analyzer)."""
        if self.energy is not None:
            self.energy.shutdown()
            self.logger.debug("Energy analyzer shutdown")