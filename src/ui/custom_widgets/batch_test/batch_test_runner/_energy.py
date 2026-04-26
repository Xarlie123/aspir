"""Inference energy measurement for batch tests."""
from __future__ import annotations

from typing import Any

from simulation_engine._4_postprocessor.postprocessor_nn import PostprocessorNN
from simulation_engine._5_analyzer.analyzer_energy import EnergyAnalyzer
from ui.custom_widgets.batch_test.test_config_model import TestConfiguration


def measure_energy(
    postprocessor: PostprocessorNN,
    config: TestConfiguration,
    logger,
    warmup_runs: int = 5,
    measurement_runs: int = 10,
) -> dict[str, Any]:
    """Measure inference energy with configurable parameters."""
    import torch

    model = postprocessor.model
    device = postprocessor.device
    img_size = postprocessor.img_size
    is_conv = postprocessor.is_conv

    model.eval()

    # Create sample input tensor for measurement
    if is_conv:
        sample = torch.randn(1, 1, img_size, img_size, device=device)
    else:
        sample = torch.randn(1, img_size * img_size, device=device)

    # Create energy analyzer
    analyzer = EnergyAnalyzer(
        model=model,
        device=str(device),
        warmup_runs=warmup_runs,
        measurement_runs=measurement_runs,
        enable_gpu_energy=config.use_gpu,
        enable_cpu_energy=True,
        logger=logger
    )

    try:
        if not analyzer.initialize():
            logger.warning("Energy analyzer could not be initialized - no backends available")
            return {
                "energy_error": "No energy measurement backends available"
            }

        logger.info("Measuring energy with backends: %s", analyzer.available_backends)

        # Split the measurement into ``n_blocks`` sub-runs so the
        # analyzer collects multiple energy readings instead of a
        # single integrated value — otherwise ``np.std`` of one point
        # is trivially 0 and the report shows no spread. Each sub-run
        # still benchmarks the same tensor; the warmup only happens
        # once thanks to ``return_per_image=False``.
        #
        # Block count is capped so we don't end up with 1-iteration
        # blocks at low measurement_runs (each block needs enough
        # iterations to be measurable on jtop / RAPL latency).
        n_blocks = max(1, min(10, measurement_runs))
        runs_per_block = max(1, measurement_runs // n_blocks)
        logger.debug(
            "Energy measurement split: %d blocks x %d runs (=%d total)",
            n_blocks, runs_per_block, n_blocks * runs_per_block,
        )

        # Run energy analysis on the sample tensor
        result = analyzer.analyze_inference(
            [sample] * n_blocks,
            n_runs=runs_per_block,
            warmup_runs=warmup_runs,
            return_per_image=False,
        )

        # Build per-backend energy data
        energy_data = {
            # Total/combined values (for backward compatibility)
            "energy_mean_mj": result.mean_energy_mj,
            "energy_std_mj": result.std_energy_joules * 1000,
            "energy_mean_watts": result.mean_power_watts,
            "energy_std_watts": result.std_power_watts,
            "energy_device_name": result.device_name,
            "energy_warmup_runs": warmup_runs,
            "energy_measurement_runs": measurement_runs,
            "efficiency_images_per_joule": result.efficiency_images_per_joule,
            # Per-backend breakdown
            "energy_backends": analyzer.available_backends,
        }

        # Add GPU-specific data if available
        if result.gpu_energy_joules is not None and result.gpu_energy_joules > 0:
            energy_data["energy_gpu_mj"] = result.gpu_energy_joules * 1000
            # Estimate GPU power from energy and time
            if result.mean_time_ms > 0:
                energy_data["energy_gpu_watts"] = (result.gpu_energy_joules * 1000) / result.mean_time_ms

        # Add CPU-specific data if available
        if result.cpu_energy_joules is not None and result.cpu_energy_joules > 0:
            energy_data["energy_cpu_mj"] = result.cpu_energy_joules * 1000
            # Estimate CPU power from energy and time
            if result.mean_time_ms > 0:
                energy_data["energy_cpu_watts"] = (result.cpu_energy_joules * 1000) / result.mean_time_ms

        return energy_data

    except Exception as e:
        logger.error("Energy measurement failed: %s", e)
        return {
            "energy_error": str(e)
        }
    finally:
        analyzer.shutdown()
