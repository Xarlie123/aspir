"""Inference and reconstruction energy measurement for batch tests."""
from __future__ import annotations

from typing import Any, Optional

from simulation_engine._4_postprocessor.postprocessor_nn import PostprocessorNN
from simulation_engine._5_analyzer.analyzer_energy import EnergyAnalyzer, sampling_rate
from ui.custom_widgets.batch_test.test_config_model import TestConfiguration


def measure_energy(
    postprocessor: PostprocessorNN,
    config: TestConfiguration,
    logger,
    warmup_runs: int = 20,
    measurement_runs: int = 800,
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
            # Per-block readings, per image: the input of the uncertainty
            # propagation (baseline_capture.dynamic_energy_uncertainty).
            "energy_blocks": {
                "energy_mj": [1000.0 * e / B for e in result.energy_per_image_joules],
                "power_W": [float(w) for w in result.power_per_image_watts],
                "time_ms": [float(t) / B for t in result.time_per_image_ms],
            },
        }
        if result.samples_per_block:
            # Power samples behind each block, and the rate achieved
            energy_data["energy_blocks"]["n_samples"] = list(result.samples_per_block)
            energy_data["energy_sampling_hz"] = result.sampling_hz

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


def measure_reconstruction_energy(
    applicator,
    config: TestConfiguration,
    logger,
    image_indices,
    analyzer: EnergyAnalyzer | None = None,
    t_reconstruction_ms: float | None = None,
    target_seconds: float = 0.5,
    max_passes: int = 200,
    n_blocks: int = 10,
) -> dict[str, Any]:
    """Measure the energy of the reconstruction stage, per image.

    The stage is attributed to the device it runs on the same way inference
    is: the GPU energy backend is enabled only when the reconstruction runs on
    CUDA, so a CPU reconstruction does not pick up the GPU's idle draw. The
    monitor synchronises CUDA around the measured window, and each pass also
    calls ``applicator.synchronize()``, so queued kernels are inside it.

    ``image_indices`` are reconstructed once per pass; the number of passes is
    chosen so the window lasts about ``target_seconds`` (energy counters are
    too coarse for millisecond windows), using ``t_reconstruction_ms`` from the
    timing phase when available.

    The measurement is ``n_blocks`` consecutive blocks of that length (5 s in
    all by default), as inference is split into blocks, so the block-to-block spread enters the uncertainty of
    the dynamic energy. The totals are the sums over the blocks.
    """
    import math

    device = getattr(applicator, "reconstruction_device", "cpu")
    indices = list(image_indices)
    if not indices:
        return {}
    sync = getattr(applicator, "synchronize", None) or (lambda: None)

    def one_pass():
        for idx in indices:
            applicator.process_image(idx)
        sync()

    per_pass_ms = float(t_reconstruction_ms or 0.0) * len(indices)
    passes = 1
    if per_pass_ms > 0:
        passes = int(min(max_passes, max(1, math.ceil(target_seconds * 1000.0 / per_pass_ms))))

    owns_analyzer = analyzer is None
    previous_device = None
    if owns_analyzer:
        analyzer = EnergyAnalyzer(
            model=None,
            device=device,
            enable_gpu_energy=(device == "cuda"),
            enable_cpu_energy=True,
            logger=logger,
        )
    else:
        previous_device = analyzer.device
        analyzer.device = device

    n_blocks = max(1, int(n_blocks))
    results = []
    try:
        for b in range(n_blocks):
            results.append(analyzer.measure_reconstruction_energy(
                one_pass, n_iterations=passes, warmup_iterations=1 if b == 0 else 0,
                cuda_sync=True))
    finally:
        if owns_analyzer:
            analyzer.shutdown()
        else:
            analyzer.device = previous_device

    images_per_block = passes * len(indices)
    n_images = images_per_block * n_blocks
    readings = [r for res in results for r in res.readings]
    total_j = sum(res.total_energy_joules for res in results)
    total_s = sum(res.execution_time_ms for res in results) / 1000.0
    gpu_j = sum(r.gpu_energy_joules or 0.0 for r in readings)
    cpu_j = sum(r.cpu_energy_joules or 0.0 for r in readings)
    out = {
        "energy_reconstruction_mj": 1000.0 * total_j / n_images,
        "energy_reconstruction_watts": float(total_j / total_s) if total_s > 0 else 0.0,
        "energy_reconstruction_device": device,
        "energy_reconstruction_images": int(n_images),
        "energy_reconstruction_backends": [r.device_type.value for r in results[0].readings],
        # Device names, as the idle baseline keys them.
        "energy_reconstruction_backend_names": [r.device_name for r in results[0].readings],
        "energy_reconstruction_blocks": {
            "energy_mj": [1000.0 * res.total_energy_joules / images_per_block for res in results],
            "power_W": [float(res.avg_power_watts) for res in results],
            "time_ms": [res.execution_time_ms / images_per_block for res in results],
        },
    }
    counts = [sum(r.n_samples or 0 for r in res.readings) for res in results]
    if any(counts):
        out["energy_reconstruction_blocks"]["n_samples"] = counts
        out["energy_reconstruction_sampling_hz"] = sampling_rate(readings)
    if gpu_j > 0:
        out["energy_reconstruction_gpu_mj"] = 1000.0 * gpu_j / n_images
    if cpu_j > 0:
        out["energy_reconstruction_cpu_mj"] = 1000.0 * cpu_j / n_images
    logger.debug("Reconstruction energy: %.3f mJ/image on %s over %d images",
                 out["energy_reconstruction_mj"], device, n_images)
    return out
