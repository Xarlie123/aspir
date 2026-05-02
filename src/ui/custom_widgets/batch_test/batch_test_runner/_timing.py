"""Inference timing measurement for batch tests."""
from __future__ import annotations

import time
from typing import Any, Optional

from simulation_engine._4_postprocessor.postprocessor_nn import PostprocessorNN
from ui.custom_widgets.batch_test.test_config_model import TestConfiguration


def measure_timing(
    postprocessor: PostprocessorNN,
    config: TestConfiguration,
    dataset,
    logger,
    applicator=None,
    warmup_runs: int = 5,
    measurement_runs: int = 20,
    sampling_rate_khz: float = 10.752,
    max_recon_samples: Optional[int] = None,
    inference_batch_size: int = 1,
) -> dict[str, Any]:
    """
    Measure inference timing with configurable parameters.

    Always measures CPU timing. If use_gpu is enabled and CUDA is available,
    also measures GPU timing (like Single Test behavior).
    Also measures reconstruction time if applicator is provided.
    """
    import numpy as np
    import torch

    model = postprocessor.model
    img_size = postprocessor.img_size
    is_conv = postprocessor.is_conv

    model.eval()

    # Calculate acquisition time based on sampling rate and number of patterns
    num_patterns = 1  # Default
    mask = None

    # Try to get mask from applicator
    if applicator is not None and hasattr(applicator, 'mask'):
        mask = applicator.mask
    elif hasattr(postprocessor, 'applicator') and postprocessor.applicator is not None:
        if hasattr(postprocessor.applicator, 'mask'):
            mask = postprocessor.applicator.mask

    if mask is not None:
        if hasattr(mask, 'num_patterns'):
            num_patterns = mask.num_patterns
        elif hasattr(mask, 'patterns') and mask.patterns is not None:
            num_patterns = len(mask.patterns)

    t_acquisition_ms = num_patterns / sampling_rate_khz if sampling_rate_khz > 0 else 0

    # Measure reconstruction time if applicator is available
    # Match Single Test behavior: measure all test images, with warmup
    t_reconstruction_ms = 0.0
    if applicator is not None:
        logger.debug("Measuring reconstruction timing...")
        try:
            dataset_size = len(getattr(dataset, 'data', []))
            # Use same number of samples as test set (from config split)
            test_ratio = config.test_split / 100.0
            n_recon_samples = max(1, min(dataset_size, int(dataset_size * test_ratio)))
            # Optional cap from the caller — reconstruction with NumPy
            # on Jetson CPU costs ~1 s per image at 1000+ patterns,
            # which dominates the entire re-measurement run; for the
            # re-measure worker we only need a 2-3 sample estimate.
            if max_recon_samples is not None and max_recon_samples > 0:
                n_recon_samples = min(n_recon_samples, int(max_recon_samples))

            # Warmup runs to ensure fair comparison (like inference warmup)
            warmup_recon = min(2, n_recon_samples)
            for idx in range(warmup_recon):
                _ = applicator.process_image(idx)

            # Measure reconstruction time for test images
            recon_times = []
            for idx in range(n_recon_samples):
                t0 = time.perf_counter()
                _ = applicator.process_image(idx)
                t1 = time.perf_counter()
                recon_times.append((t1 - t0) * 1000.0)  # ms

            if recon_times:
                t_reconstruction_ms = float(np.mean(recon_times))
                logger.debug("Reconstruction time: %.2f ms (mean of %d samples, after %d warmup)",
                             t_reconstruction_ms, len(recon_times), warmup_recon)
        except Exception as e:
            logger.warning("Reconstruction timing failed: %s", e)

    # Helper function to measure timing on a specific device.
    # ``inference_batch_size`` controls how many images the model
    # processes per forward pass; per-call timings are divided by B
    # at the end of this helper so the returned ``_mean_ms`` columns
    # stay per-image regardless of the chosen batch.
    B = max(1, int(inference_batch_size))
    def measure_on_device(device: torch.device) -> list[float]:
        model.to(device)
        if is_conv:
            sample = torch.randn(B, 1, img_size, img_size, device=device)
        else:
            sample = torch.randn(B, img_size * img_size, device=device)

        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(sample)
                if device.type == "cuda":
                    torch.cuda.synchronize()

        # Measure
        times = []
        with torch.no_grad():
            for _ in range(measurement_runs):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                _ = model(sample)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                end = time.perf_counter()
                times.append((end - start) * 1000)  # Convert to ms

        return times

    # Always measure CPU timing
    logger.debug("Measuring CPU inference timing...")
    cpu_device = torch.device('cpu')
    cpu_times = measure_on_device(cpu_device)
    # Per-call → per-image (each call processed B images, so dividing
    # by B yields the mean latency per image at this batch size).
    cpu_times_np = np.array(cpu_times) / B

    results = {
        "timing_cpu_mean_ms": float(cpu_times_np.mean()),
        "timing_cpu_std_ms": float(cpu_times_np.std()),
        "timing_cpu_min_ms": float(cpu_times_np.min()),
        "timing_cpu_max_ms": float(cpu_times_np.max()),
        "timing_warmup_runs": warmup_runs,
        "timing_measurement_runs": measurement_runs,
        "timing_sampling_rate_khz": sampling_rate_khz,
        "timing_acquisition_ms": float(t_acquisition_ms),
        "timing_reconstruction_ms": float(t_reconstruction_ms),
        "timing_num_patterns": num_patterns,
        "timing_batch_size": B,
        "use_gpu": config.use_gpu,
    }

    # Also measure GPU timing if requested and available
    if config.use_gpu and torch.cuda.is_available():
        logger.debug("Measuring GPU inference timing...")
        try:
            gpu_device = torch.device('cuda')
            gpu_times = measure_on_device(gpu_device)
            gpu_times_np = np.array(gpu_times) / B  # per-image

            results["timing_gpu_mean_ms"] = float(gpu_times_np.mean())
            results["timing_gpu_std_ms"] = float(gpu_times_np.std())
            results["timing_gpu_min_ms"] = float(gpu_times_np.min())
            results["timing_gpu_max_ms"] = float(gpu_times_np.max())

            # Move model back to original device
            model.to(postprocessor.device)
        except Exception as e:
            logger.warning("GPU timing measurement failed: %s", e)
    else:
        # Ensure model is on CPU if GPU was not used
        model.to(cpu_device)

    # For backwards compatibility, also set timing_mean_ms to the primary device
    if config.use_gpu and "timing_gpu_mean_ms" in results:
        results["timing_mean_ms"] = results["timing_gpu_mean_ms"]
    else:
        results["timing_mean_ms"] = results["timing_cpu_mean_ms"]

    return results
