"""
PyTorch Profiler utilities for detecting performance bottlenecks.

Usage:
    from simulation_engine._5_analyzer.profiler import PipelineProfiler

    profiler = PipelineProfiler(simulation, logger=logger)
    results = profiler.profile_inference(num_images=10)
    profiler.export_to_tensorboard("./profiler_logs")
"""
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from contextlib import contextmanager

import numpy as np
import torch
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler


class PipelineProfiler:
    """
    Profiler for the SPIm pipeline to detect performance bottlenecks.

    Profiles:
    - Reconstruction (CPU)
    - DNN inference (CPU and/or GPU)
    - Memory transfers
    - Individual layer operations
    """

    def __init__(self, simulation, logger=None):
        """
        Initialize the profiler.

        Args:
            simulation: The Simulation object containing applicator and postprocessor
            logger: Optional logger instance
        """
        self.simulation = simulation

        if logger:
            self.logger = logger.getChild("PipelineProfiler")
        else:
            self.logger = logging.getLogger("SPIm.PipelineProfiler")

        self._last_results = None
        self._trace_dir = None

    def profile_inference(
        self,
        num_images: int = 10,
        warmup_runs: int = 3,
        device: str = "auto",
        profile_memory: bool = True,
        record_shapes: bool = True
    ) -> Dict[str, Any]:
        """
        Profile the DNN inference to find bottlenecks.

        Args:
            num_images: Number of images to profile
            warmup_runs: Warmup iterations before profiling
            device: Device to profile ('cpu', 'cuda', or 'auto')
            profile_memory: Whether to profile memory usage
            record_shapes: Whether to record tensor shapes

        Returns:
            Dictionary with profiling results and statistics
        """
        post = getattr(self.simulation, 'postprocessor', None)
        if post is None or not getattr(post, 'trained', False):
            raise RuntimeError("No trained postprocessor available")

        # Determine device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        torch_device = torch.device(device)
        post.model.to(torch_device)
        post.model.eval()

        self.logger.info(f"Profiling inference on {device} with {num_images} images")

        # Get test data
        val_loader = post.loaders.get("val")
        if val_loader is None:
            raise RuntimeError("No validation data loader available")

        # Collect sample inputs
        sample_inputs = []
        for noisy, _ in val_loader:
            sample_inputs.append(noisy)
            if len(sample_inputs) * noisy.size(0) >= num_images:
                break

        if not sample_inputs:
            raise RuntimeError("No validation data available")

        # Concatenate and limit to num_images
        all_inputs = torch.cat(sample_inputs, dim=0)[:num_images]
        all_inputs = all_inputs.to(torch_device)

        self.logger.debug(f"Prepared {all_inputs.size(0)} input images for profiling")

        # Warmup runs
        self.logger.debug(f"Running {warmup_runs} warmup iterations")
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = post.model(all_inputs)
                if torch_device.type == 'cuda':
                    torch.cuda.synchronize()

        # Configure profiler activities
        activities = [ProfilerActivity.CPU]
        if torch_device.type == 'cuda':
            activities.append(ProfilerActivity.CUDA)

        # Profile with detailed tracing
        self.logger.info("Starting profiling...")

        with profile(
            activities=activities,
            record_shapes=record_shapes,
            profile_memory=profile_memory,
            with_stack=True,
            with_flops=True
        ) as prof:
            with torch.no_grad():
                # Profile individual images for fine-grained analysis
                for i in range(all_inputs.size(0)):
                    with record_function(f"inference_image_{i}"):
                        single_input = all_inputs[i:i+1]
                        _ = post.model(single_input)
                        if torch_device.type == 'cuda':
                            torch.cuda.synchronize()

        # Extract results
        results = self._extract_profiler_results(prof, device, num_images)
        self._last_results = results
        self._last_prof = prof

        self.logger.info("Profiling complete")
        return results

    def profile_full_pipeline(
        self,
        num_images: int = 5,
        device: str = "auto"
    ) -> Dict[str, Any]:
        """
        Profile the full pipeline: reconstruction + inference.

        Args:
            num_images: Number of images to profile
            device: Device for inference ('cpu', 'cuda', or 'auto')

        Returns:
            Dictionary with profiling results
        """
        aplic = getattr(self.simulation, 'applicator', None)
        post = getattr(self.simulation, 'postprocessor', None)

        if aplic is None:
            raise RuntimeError("No applicator available")
        if post is None or not getattr(post, 'trained', False):
            raise RuntimeError("No trained postprocessor available")

        # Determine device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        torch_device = torch.device(device)
        post.model.to(torch_device)
        post.model.eval()

        self.logger.info(f"Profiling full pipeline on {device}")

        # Configure profiler activities
        activities = [ProfilerActivity.CPU]
        if torch_device.type == 'cuda':
            activities.append(ProfilerActivity.CUDA)

        dataset_size = len(getattr(self.simulation.dataset, 'data', []))
        num_images = min(num_images, dataset_size)

        with profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            for idx in range(num_images):
                # Profile reconstruction
                with record_function("reconstruction"):
                    recon_img = aplic.process_image(idx)

                # Profile inference
                with record_function("dnn_inference"):
                    arr = np.array(recon_img, dtype=np.float32)
                    if getattr(post, 'is_conv', False):
                        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
                    else:
                        tensor = torch.from_numpy(arr.flatten()).unsqueeze(0)

                    tensor = tensor.to(torch_device)

                    with torch.no_grad():
                        _ = post.model(tensor)

                    if torch_device.type == 'cuda':
                        torch.cuda.synchronize()

        results = self._extract_pipeline_results(prof, device, num_images)
        self._last_results = results
        self._last_prof = prof

        return results

    def _extract_profiler_results(
        self,
        prof,
        device: str,
        num_images: int
    ) -> Dict[str, Any]:
        """Extract and format profiler results."""

        # Get key averages sorted by device time if GPU, else CPU time
        # Note: PyTorch 2.x uses 'device_time_total' instead of 'cuda_time_total'
        if device == "cuda":
            key_averages = prof.key_averages().table(
                sort_by="self_device_time_total",
                row_limit=20
            )
            top_ops = prof.key_averages()
            top_ops = sorted(top_ops, key=lambda x: x.device_time_total, reverse=True)[:15]
        else:
            key_averages = prof.key_averages().table(
                sort_by="cpu_time_total",
                row_limit=20
            )
            top_ops = prof.key_averages()
            top_ops = sorted(top_ops, key=lambda x: x.cpu_time_total, reverse=True)[:15]

        # Extract top operations
        bottlenecks = []
        for op in top_ops:
            device_time = op.device_time_total if device == "cuda" else 0
            bottlenecks.append({
                'name': op.key,
                'cpu_time_ms': op.cpu_time_total / 1000,  # Convert to ms
                'cuda_time_ms': device_time / 1000,
                'calls': op.count,
                'cpu_time_per_call_ms': (op.cpu_time_total / op.count / 1000) if op.count > 0 else 0,
                'cuda_time_per_call_ms': (device_time / op.count / 1000) if op.count > 0 and device == "cuda" else 0,
            })

        # Calculate totals
        total_cpu_time = sum(op.cpu_time_total for op in prof.key_averages()) / 1000
        total_device_time = sum(op.device_time_total for op in prof.key_averages()) / 1000 if device == "cuda" else 0

        return {
            'device': device,
            'num_images': num_images,
            'total_cpu_time_ms': total_cpu_time,
            'total_cuda_time_ms': total_device_time,
            'avg_time_per_image_ms': (total_device_time if device == "cuda" else total_cpu_time) / num_images,
            'bottlenecks': bottlenecks,
            'key_averages_table': key_averages,
        }

    def _extract_pipeline_results(
        self,
        prof,
        device: str,
        num_images: int
    ) -> Dict[str, Any]:
        """Extract results for full pipeline profiling."""

        # Find reconstruction and inference times
        recon_time = 0
        inference_time = 0

        for event in prof.key_averages():
            if event.key == "reconstruction":
                recon_time = event.cpu_time_total / 1000  # ms
            elif event.key == "dnn_inference":
                if device == "cuda":
                    inference_time = event.device_time_total / 1000
                else:
                    inference_time = event.cpu_time_total / 1000

        # Get all operations
        if device == "cuda":
            top_ops = sorted(prof.key_averages(), key=lambda x: x.device_time_total, reverse=True)[:20]
        else:
            top_ops = sorted(prof.key_averages(), key=lambda x: x.cpu_time_total, reverse=True)[:20]

        bottlenecks = []
        for op in top_ops:
            device_time = op.device_time_total if device == "cuda" else 0
            bottlenecks.append({
                'name': op.key,
                'cpu_time_ms': op.cpu_time_total / 1000,
                'cuda_time_ms': device_time / 1000,
                'calls': op.count,
                'cpu_time_per_call_ms': (op.cpu_time_total / op.count / 1000) if op.count > 0 else 0,
                'cuda_time_per_call_ms': (device_time / op.count / 1000) if op.count > 0 else 0,
            })

        return {
            'device': device,
            'num_images': num_images,
            'reconstruction_time_ms': recon_time,
            'inference_time_ms': inference_time,
            'total_time_ms': recon_time + inference_time,
            'avg_recon_per_image_ms': recon_time / num_images if num_images > 0 else 0,
            'avg_inference_per_image_ms': inference_time / num_images if num_images > 0 else 0,
            'bottlenecks': bottlenecks,
            'key_averages_table': prof.key_averages().table(
                sort_by="self_device_time_total" if device == "cuda" else "cpu_time_total",
                row_limit=20
            ),
        }

    def export_to_tensorboard(self, log_dir: str = "./profiler_logs"):
        """
        Export profiling results to TensorBoard format.

        Args:
            log_dir: Directory to save TensorBoard logs

        After exporting, run:
            tensorboard --logdir=./profiler_logs
        """
        if not hasattr(self, '_last_prof') or self._last_prof is None:
            raise RuntimeError("No profiling data available. Run profile_inference() first.")

        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        # Export Chrome trace
        trace_file = log_path / f"trace_{int(time.time())}.json"
        self._last_prof.export_chrome_trace(str(trace_file))

        self._trace_dir = str(log_path)
        self.logger.info(f"Exported trace to {trace_file}")
        self.logger.info(f"View with: tensorboard --logdir={log_dir}")

        return str(trace_file)

    def get_summary(self) -> str:
        """Get a human-readable summary of the profiling results."""
        if self._last_results is None:
            return "No profiling results available. Run profile_inference() first."

        r = self._last_results

        # Calculate kernel time from layer breakdown (self times only)
        layer_breakdown = self.get_layer_breakdown()
        kernel_time_ms = sum(layer['total_time_ms'] for layer in layer_breakdown)

        lines = [
            "=" * 60,
            "PROFILING SUMMARY",
            "=" * 60,
            f"Device: {r['device'].upper()}",
            f"Images profiled: {r['num_images']}",
            "",
        ]

        if 'reconstruction_time_ms' in r:
            # Full pipeline mode
            inference_time_ms = r['inference_time_ms']
            overhead_ms = inference_time_ms - kernel_time_ms if inference_time_ms > 0 else 0
            overhead_pct = (overhead_ms / inference_time_ms * 100) if inference_time_ms > 0 else 0

            lines.extend([
                "PIPELINE BREAKDOWN:",
                f"  Reconstruction:    {r['reconstruction_time_ms']:.2f} ms ({r['avg_recon_per_image_ms']:.2f} ms/image)",
                f"  DNN Inference:     {r['inference_time_ms']:.2f} ms ({r['avg_inference_per_image_ms']:.2f} ms/image)",
                f"    ├─ GPU kernels:  {kernel_time_ms:.2f} ms ({kernel_time_ms/inference_time_ms*100:.1f}%)" if inference_time_ms > 0 else f"    ├─ GPU kernels:  {kernel_time_ms:.2f} ms",
                f"    └─ Overhead:     {overhead_ms:.2f} ms ({overhead_pct:.1f}%)",
                f"  Total:             {r['total_time_ms']:.2f} ms",
                "",
            ])
        else:
            # Calculate inference time from markers
            inference_time_ms = 0
            for op in r['bottlenecks']:
                if op['name'].startswith('inference_image_'):
                    inference_time_ms += op['cuda_time_ms'] if r['device'] == 'cuda' else op['cpu_time_ms']

            overhead_ms = inference_time_ms - kernel_time_ms if inference_time_ms > 0 else 0
            overhead_pct = (overhead_ms / inference_time_ms * 100) if inference_time_ms > 0 else 0

            lines.extend([
                "TIME BREAKDOWN:",
                f"  Inference time:    {inference_time_ms:.2f} ms ({inference_time_ms/r['num_images']:.2f} ms/image)",
                f"  ├─ GPU kernels:    {kernel_time_ms:.2f} ms ({kernel_time_ms/inference_time_ms*100:.1f}%)" if inference_time_ms > 0 else f"  ├─ GPU kernels:    {kernel_time_ms:.2f} ms",
                f"  └─ Overhead:       {overhead_ms:.2f} ms ({overhead_pct:.1f}%) - kernel launch, sync, framework",
                "",
            ])

        lines.extend([
            "TOP BOTTLENECKS:",
            "-" * 60,
        ])

        for i, op in enumerate(r['bottlenecks'][:10], 1):
            time_ms = op['cuda_time_ms'] if r['device'] == 'cuda' and op['cuda_time_ms'] > 0 else op['cpu_time_ms']
            lines.append(f"  {i:2d}. {op['name'][:40]:<40} {time_ms:>8.2f} ms ({op['calls']} calls)")

        lines.extend([
            "",
            "=" * 60,
        ])

        return "\n".join(lines)

    def print_summary(self):
        """Print the profiling summary to console."""
        print(self.get_summary())

    def get_layer_breakdown(self) -> List[Dict[str, Any]]:
        """
        Get detailed breakdown by layer type.

        Returns:
            List of layer statistics grouped by operation type
        """
        if self._last_results is None or not hasattr(self, '_last_prof'):
            return []

        device = self._last_results['device']

        # Use ALL operations from profiler, not just top bottlenecks
        all_ops = self._last_prof.key_averages()

        # Group by operation type
        layer_types = {}
        for op in all_ops:
            name = op.key
            name_lower = name.lower()

            # Skip profiler markers (they wrap other operations, would double-count)
            if (name_lower.startswith('inference_image_') or
                name_lower == 'dnn_inference' or
                name_lower == 'reconstruction' or
                name_lower.startswith('profiler') or
                name_lower.startswith('cudalaunch') or
                name_lower.startswith('enumerate')):
                continue

            # Categorize operation
            category = self._categorize_operation(name_lower)

            if category not in layer_types:
                layer_types[category] = {
                    'category': category,
                    'total_time_ms': 0,
                    'operations': []
                }

            # Use SELF time (excludes children) to avoid double-counting
            if device == 'cuda':
                time_ms = op.self_device_time_total / 1000
            else:
                time_ms = op.self_cpu_time_total / 1000

            layer_types[category]['total_time_ms'] += time_ms
            layer_types[category]['operations'].append({
                'name': name,
                'time_ms': time_ms,
                'calls': op.count
            })

        # Log operations in 'Other' category for debugging
        if 'Other' in layer_types and layer_types['Other']['operations']:
            other_ops = layer_types['Other']['operations']
            self.logger.info(f"Operations in 'Other' category ({len(other_ops)} ops, {layer_types['Other']['total_time_ms']:.2f} ms):")
            for op in sorted(other_ops, key=lambda x: x['time_ms'], reverse=True)[:10]:
                self.logger.info(f"  - {op['name']}: {op['time_ms']:.3f} ms ({op['calls']} calls)")

        # Sort by total time
        return sorted(layer_types.values(), key=lambda x: x['total_time_ms'], reverse=True)

    def _categorize_operation(self, name_lower: str) -> str:
        """Categorize a PyTorch operation by its name."""
        # Convolution - include CUDA kernel patterns
        if any(p in name_lower for p in ['conv', 'winograd', 'scudnn', 'cudnn_conv', 'implicit_convolve']):
            return 'Convolution'

        # BatchNorm
        if any(p in name_lower for p in ['batch_norm', 'batchnorm', '_bn', 'cudnn_batch_norm']):
            return 'BatchNorm'

        # Activations
        if any(p in name_lower for p in ['relu', 'leaky_relu', 'prelu', 'elu']):
            return 'Activation (ReLU)'
        if any(p in name_lower for p in ['sigmoid', 'tanh', 'softmax', 'gelu', 'silu', 'hardswish']):
            return 'Activation (Other)'

        # Pooling
        if 'pool' in name_lower:
            return 'Pooling'

        # Linear/Dense layers - include GEMM CUDA kernels
        if any(p in name_lower for p in ['linear', 'matmul', 'gemm', 'addmm', 'cublas']):
            return 'Linear'
        if name_lower.split('::')[-1] == 'mm':
            return 'Linear'

        # Skip connections
        if 'add_' in name_lower or name_lower.endswith('::add') or 'aten::add' in name_lower:
            return 'Add (Skip Connection)'

        # Concatenation
        if 'cat' in name_lower or 'concat' in name_lower:
            return 'Concatenation'

        # Upsampling
        if any(p in name_lower for p in ['upsample', 'interpolate', 'nearest', 'bilinear']):
            return 'Upsample'

        # Memory operations
        if any(p in name_lower for p in ['copy', 'contiguous', 'clone', 'memcpy', 'memset']):
            return 'Memory Transfer'
        if 'to' == name_lower.split('::')[-1] or '_to_' in name_lower:
            return 'Memory Transfer'

        # Reshape operations
        if any(p in name_lower for p in ['view', 'reshape', 'flatten', 'squeeze', 'permute', 'transpose']):
            return 'Reshape'

        # Dropout
        if 'dropout' in name_lower:
            return 'Dropout'

        # Tensor allocation
        if any(p in name_lower for p in ['empty', 'zero', 'fill', 'ones', 'alloc']):
            return 'Tensor Allocation'

        # Element-wise operations
        if any(p in name_lower for p in ['mul', 'div', 'sub', 'clamp', 'threshold']):
            return 'Element-wise Ops'

        # CUDNN operations that weren't caught above
        if 'cudnn' in name_lower:
            return 'CUDNN Ops'

        # Log and return Other
        self.logger.debug(f"Unknown operation categorized as 'Other': {name_lower}")
        return 'Other'
