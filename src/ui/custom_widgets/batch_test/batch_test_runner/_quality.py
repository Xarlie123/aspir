"""Quality metrics of one batch test, shared by the sequential and parallel paths.

Metric convention: ``analyzer_noise.METRIC_CONVENTION`` (2 since v1.3). The
"recons" fields score the linear reconstruction on the image's scale
(``PostprocessorNN.test_linear``); the network input, which earlier versions
scored under that name, is reported separately as ``*_network_input``.
"""

from __future__ import annotations

from typing import Any

from simulation_engine._5_analyzer.analyzer import Analyzer
from simulation_engine._5_analyzer.analyzer_noise import METRIC_CONVENTION, mean_psnr_ssim


def quality_results(postprocessor, reports, logger) -> tuple[dict[str, Any], tuple]:
    """Score the linear stage and the network output against the test references.

    Returns the result fields and the test arrays ``(orig, network_input,
    denoised)`` for export.
    """
    orig, network_input, denoised = postprocessor.test_dataset()
    linear = postprocessor.test_linear()
    analyzer = Analyzer(orig, linear, denoised)
    analyzer.analyze_noise()
    noise = analyzer.noise

    results: dict[str, Any] = {
        "metric_convention": METRIC_CONVENTION,
        "linear_calibration": getattr(postprocessor, "linear_calibration", None),
    }
    if "quality" not in reports:
        return results, (orig, network_input, denoised)

    # recons = linear reconstruction (image scale); denoised = network output
    results["quality_metrics"] = {"psnr": noise.psnr_rec_mean, "ssim": noise.ssim_rec_mean}
    results["psnr_recons"] = noise.psnr_noise_mean
    results["psnr_denoised"] = noise.psnr_rec_mean
    results["ssim_recons"] = noise.ssim_noise_mean
    results["ssim_denoised"] = noise.ssim_rec_mean
    try:
        results["lpips_recons"] = noise.lpips_noise_mean
        results["lpips_denoised"] = noise.lpips_rec_mean
        results["quality_metrics"]["lpips"] = noise.lpips_rec_mean
    except Exception as e:
        logger.warning("LPIPS computation failed: %s", e)
        results["lpips_error"] = str(e)

    # Per-image metrics for detailed charts
    results["quality_per_image"] = {
        "psnr_noisy": list(noise.psnr_noise),
        "psnr_denoised": list(noise.psnr_rec),
        "ssim_noisy": list(noise.ssim_noise),
        "ssim_denoised": list(noise.ssim_rec),
        "lpips_noisy": list(noise.lpips_noise),
        "lpips_denoised": list(noise.lpips_rec),
    }

    # What the network actually sees (dataset min-max + quantisation), same convention
    results["psnr_network_input"], results["ssim_network_input"] = mean_psnr_ssim(orig, network_input)
    return results, (orig, network_input, denoised)
