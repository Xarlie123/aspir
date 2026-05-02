"""Inference energy measurement for batch tests."""
from __future__ import annotations

from typing import Any, Optional

from simulation_engine._4_postprocessor.postprocessor_nn import PostprocessorNN
from simulation_engine._5_analyzer.analyzer_energy import EnergyAnalyzer
from ui.custom_widgets.batch_test.test_config_model import TestConfiguration


def measure_energy(
    postprocessor: PostprocessorNN,
    config: TestConfiguration,
    logger,
    warmup_runs: int = 5,
    measurement_runs: int = 10,
    analyzer: Optional[EnergyAnalyzer] = None,
    inference_batch_size: int = 1,
) -> dict[str, Any]:
    """Measure inference energy with configurable parameters.

    When ``analyzer`` is provided, the function reuses it and only
    swaps the model for the current ``postprocessor``; the caller is
    responsible for the analyzer lifecycle (initialize / shutdown).
    This avoids spinning up a new ``jtop`` connection per test in
    re-measurement runs, which on Jetson can flake out after several
    init/shutdown cycles.

    ``inference_batch_size`` controls the shape of the sample tensor
    fed into the model on each iteration ((B, 1, H, W) on conv
    models, (B, H*W) otherwise). The energy / power values reported
    here remain *per image* — the analyzer's per-iteration outputs
    are divided by B at the end of this function — but the
    ``energy_batch_size`` field is added to the returned dict so the
    reader can tell what batch was actually measured."""
    import torch

    model = postprocessor.model
    device = postprocessor.device
    img_size = postprocessor.img_size
    is_conv = postprocessor.is_conv

    model.eval()

    # Create sample input tensor for measurement
    B = max(1, int(inference_batch_size))
    if is_conv:
        sample = torch.randn(B, 1, img_size, img_size, device=device)
    else:
        sample = torch.randn(B, img_size * img_size, device=device)

    owns_analyzer = analyzer is None
    if analyzer is None:
        analyzer = EnergyAnalyzer(
            model=model,
            device=str(device),
            warmup_runs=warmup_runs,
            measurement_runs=measurement_runs,
            enable_gpu_energy=config.use_gpu,
            enable_cpu_energy=True,
            logger=logger
        )
    else:
        # Reuse: re-aim the analyzer at this test's model + device
        # without restarting the jtop connection.
        analyzer.model = model
        analyzer.device = str(device)
        analyzer.warmup_runs = warmup_runs
        analyzer.measurement_runs = measurement_runs

    try:
        if owns_analyzer and not analyzer.initialize():
            logger.warning("Energy analyzer could not be initialized - no backends available")
            return {
                "energy_error": "No energy measurement backends available"
            }
        if not owns_analyzer and not analyzer.is_initialized:
            # Caller forgot to initialize — fail loud instead of
            # silently producing a no-backend result.
            raise RuntimeError(
                "Reused EnergyAnalyzer was not initialized by the caller"
            )

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

        # Per-iteration values from the analyzer cover the full
        # batch (B images per forward pass). Divide the energy by B
        # to land on a per-image figure; power doesn't scale with
        # B (it's an instantaneous-rate metric averaged across the
        # measurement window), so it stays untouched. Efficiency
        # (images/J) accordingly scales by B.
        e_mean_mj_per_image  = result.mean_energy_mj / B
        e_std_mj_per_image   = (result.std_energy_joules * 1000) / B
        eff_per_image = (
            result.efficiency_images_per_joule * B
            if result.efficiency_images_per_joule else 0
        )

        energy_data = {
            # Per-image values (preserve historical column semantics)
            "energy_mean_mj": e_mean_mj_per_image,
            "energy_std_mj": e_std_mj_per_image,
            "energy_mean_watts": result.mean_power_watts,
            "energy_std_watts": result.std_power_watts,
            "energy_device_name": result.device_name,
            "energy_warmup_runs": warmup_runs,
            "energy_measurement_runs": measurement_runs,
            "energy_batch_size": B,
            "efficiency_images_per_joule": eff_per_image,
            # Per-backend breakdown
            "energy_backends": analyzer.available_backends,
        }

        # Add GPU-specific data if available (per-image: the
        # analyzer's component fields are also per-iteration so they
        # carry the same /B division as the totals above).
        if result.gpu_energy_joules is not None and result.gpu_energy_joules > 0:
            energy_data["energy_gpu_mj"] = (result.gpu_energy_joules * 1000) / B
            # Estimate GPU power from energy and time
            if result.mean_time_ms > 0:
                energy_data["energy_gpu_watts"] = (result.gpu_energy_joules * 1000) / result.mean_time_ms

        # Add CPU-specific data if available
        if result.cpu_energy_joules is not None and result.cpu_energy_joules > 0:
            energy_data["energy_cpu_mj"] = (result.cpu_energy_joules * 1000) / B
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
        if owns_analyzer:
            analyzer.shutdown()
