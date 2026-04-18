"""PyTorch profiler helpers for batch tests."""
from __future__ import annotations

from typing import Any, Optional

from simulation_engine._4_postprocessor.postprocessor_nn import PostprocessorNN
from ui.custom_widgets.batch_test.test_config_model import TestConfiguration

# PyTorch profiler imports
try:
    from torch.profiler import ProfilerActivity, profile, record_function
    PROFILER_AVAILABLE = True
except ImportError:
    PROFILER_AVAILABLE = False


def profile_model(
    postprocessor: PostprocessorNN,
    config: TestConfiguration,
    logger,
    num_images: int = 10,
    warmup_runs: int = 3,
) -> Optional[dict[str, Any]]:
    """
    Profile model with PyTorch profiler and return serializable results.

    When GPU is enabled, profiles both CPU and GPU separately.
    When only CPU is used, profiles CPU only.

    Args:
        postprocessor: The trained postprocessor
        config: Test configuration
        logger: Logger instance
        num_images: Number of images to profile
        warmup_runs: Warmup iterations before profiling

    Returns:
        Dictionary with profiler results for JSON serialization:
        - If GPU enabled: {"cpu": cpu_results, "gpu": gpu_results}
        - If CPU only: {"cpu": cpu_results}
        Returns None if profiler unavailable.
    """
    if not PROFILER_AVAILABLE:
        logger.warning("PyTorch profiler not available")
        return None

    import torch

    model = postprocessor.model
    device = postprocessor.device
    img_size = postprocessor.img_size
    is_conv = postprocessor.is_conv
    has_gpu = device.type == "cuda"

    model.eval()

    # Create sample inputs on target device
    if is_conv:
        sample = torch.randn(num_images, 1, img_size, img_size, device=device)
    else:
        sample = torch.randn(num_images, img_size * img_size, device=device)

    # Warmup
    logger.debug("Profiler: Running %d warmup iterations", warmup_runs)
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(sample[:1])
            if has_gpu:
                torch.cuda.synchronize()

    results = {}

    # Profile CPU (always)
    logger.info("Running PyTorch profiler (CPU)...")
    try:
        cpu_results = _run_profiler_pass(
            model, sample, num_images, [ProfilerActivity.CPU], "cpu", has_gpu, logger
        )
        if cpu_results:
            results["cpu"] = cpu_results
    except Exception as e:
        logger.error("CPU profiling failed: %s", e)

    # Profile GPU (if available)
    if has_gpu:
        logger.info("Running PyTorch profiler (GPU)...")
        try:
            gpu_results = _run_profiler_pass(
                model, sample, num_images,
                [ProfilerActivity.CPU, ProfilerActivity.CUDA], "cuda", has_gpu, logger
            )
            if gpu_results:
                results["gpu"] = gpu_results
        except Exception as e:
            logger.error("GPU profiling failed: %s", e)

    if results:
        logger.info("Profiling complete (CPU: %s, GPU: %s)",
                    "yes" if "cpu" in results else "no",
                    "yes" if "gpu" in results else "no")
        return results
    return None


def _run_profiler_pass(
    model,
    sample,
    num_images: int,
    activities: list,
    device_str: str,
    sync_cuda: bool,
    logger,
) -> Optional[dict[str, Any]]:
    """Run a single profiler pass with specified activities."""
    import torch

    try:
        with profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
            with_flops=True
        ) as prof:
            with torch.no_grad():
                for i in range(num_images):
                    with record_function(f"inference_image_{i}"):
                        _ = model(sample[i:i+1])
                        if sync_cuda:
                            torch.cuda.synchronize()

        # Extract results (JSON-serializable)
        return _extract_profiler_results(prof, device_str, num_images)

    except Exception as e:
        logger.error("Profiler pass failed for %s: %s", device_str, e)
        return None


def _extract_profiler_results(prof, device: str, num_images: int) -> dict[str, Any]:
    """Extract JSON-serializable profiler results."""
    is_cuda = "cuda" in device

    # Get key averages
    if is_cuda:
        top_ops = prof.key_averages()
        top_ops = sorted(top_ops, key=lambda x: x.device_time_total, reverse=True)[:20]
    else:
        top_ops = prof.key_averages()
        top_ops = sorted(top_ops, key=lambda x: x.cpu_time_total, reverse=True)[:20]

    # Extract top operations (bottlenecks)
    bottlenecks = []
    for op in top_ops:
        device_time = op.device_time_total if is_cuda else 0
        bottlenecks.append({
            'name': op.key,
            'cpu_time_ms': float(op.cpu_time_total / 1000),
            'cuda_time_ms': float(device_time / 1000),
            'calls': int(op.count),
            'cpu_time_per_call_ms': float(op.cpu_time_total / op.count / 1000) if op.count > 0 else 0,
            'cuda_time_per_call_ms': float(device_time / op.count / 1000) if op.count > 0 and is_cuda else 0,
        })

    # Calculate totals
    total_cpu_time = sum(op.cpu_time_total for op in prof.key_averages()) / 1000
    total_device_time = sum(op.device_time_total for op in prof.key_averages()) / 1000 if is_cuda else 0

    # Group by layer type for pie chart
    layer_breakdown = _categorize_profiler_ops(prof, is_cuda)

    # Generate summary text
    summary_lines = [
        f"Device: {'CUDA' if is_cuda else 'CPU'}",
        f"Images profiled: {num_images}",
        f"Total CPU time: {total_cpu_time:.2f} ms",
    ]
    if is_cuda:
        summary_lines.append(f"Total CUDA time: {total_device_time:.2f} ms")
    summary_lines.append(f"Avg time per image: {(total_device_time if is_cuda else total_cpu_time) / num_images:.2f} ms")

    return {
        'device': 'cuda' if is_cuda else 'cpu',
        'num_images': num_images,
        'total_cpu_time_ms': float(total_cpu_time),
        'total_cuda_time_ms': float(total_device_time),
        'avg_time_per_image_ms': float((total_device_time if is_cuda else total_cpu_time) / num_images),
        'bottlenecks': bottlenecks,
        'layer_breakdown': layer_breakdown,
        'summary': '\n'.join(summary_lines),
    }


def _categorize_profiler_ops(prof, is_cuda: bool) -> list[dict[str, Any]]:
    """Categorize profiler operations by type for pie chart."""
    layer_types = {}

    for op in prof.key_averages():
        name = op.key
        name_lower = name.lower()

        # Skip profiler markers
        if (name_lower.startswith('inference_image_') or
            name_lower.startswith('profiler') or
            name_lower.startswith('cudalaunch') or
            name_lower.startswith('enumerate')):
            continue

        # Categorize
        category = _categorize_operation(name_lower)

        if category not in layer_types:
            layer_types[category] = {'category': category, 'total_time_ms': 0.0}

        # Use self time to avoid double-counting
        if is_cuda:
            time_ms = op.self_device_time_total / 1000
        else:
            time_ms = op.self_cpu_time_total / 1000

        layer_types[category]['total_time_ms'] += time_ms

    # Convert to list and sort
    result = [{'category': k, 'total_time_ms': float(v['total_time_ms'])}
              for k, v in layer_types.items() if v['total_time_ms'] > 0]
    return sorted(result, key=lambda x: x['total_time_ms'], reverse=True)


def _categorize_operation(name_lower: str) -> str:
    """Categorize a PyTorch operation by its name."""
    # Convolution
    if any(p in name_lower for p in ['conv', 'winograd', 'scudnn', 'cudnn_conv', 'implicit_convolve']):
        return 'Convolution'

    # BatchNorm
    if any(p in name_lower for p in ['batch_norm', 'batchnorm', '_bn', 'cudnn_batch_norm']):
        return 'BatchNorm'

    # Activations
    if any(p in name_lower for p in ['relu', 'leaky_relu', 'prelu', 'elu', 'sigmoid', 'tanh', 'softmax', 'gelu', 'silu']):
        return 'Activation'

    # Pooling
    if 'pool' in name_lower:
        return 'Pooling'

    # Linear/Dense layers
    if any(p in name_lower for p in ['linear', 'matmul', 'gemm', 'addmm', 'cublas', 'mm']):
        return 'Linear'

    # Skip connections
    if 'add_' in name_lower or name_lower.endswith('::add') or 'aten::add' in name_lower:
        return 'Add (Skip)'

    # Concatenation
    if 'cat' in name_lower or 'concat' in name_lower:
        return 'Concatenation'

    # Upsampling
    if any(p in name_lower for p in ['upsample', 'interpolate', 'nearest', 'bilinear']):
        return 'Upsample'

    # Memory operations
    if any(p in name_lower for p in ['copy', 'contiguous', 'clone', 'memcpy', 'memset', 'to']):
        return 'Memory'

    # Reshape operations
    if any(p in name_lower for p in ['view', 'reshape', 'flatten', 'squeeze', 'permute', 'transpose']):
        return 'Reshape'

    return 'Other'
