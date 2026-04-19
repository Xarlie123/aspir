"""Export helpers for batch test results, models and datasets."""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from ui.custom_widgets.batch_test.test_config_model import BatchTestConfig, ExportLevel
from ui.utils.file_formats import BATCH_TESTS_DIR, BatchAnalysisReport, FileExtensions


def get_unique_output_dir(base_name: str) -> Path:
    """
    Get a unique output directory name, appending _1, _2, etc. if needed.

    Args:
        base_name: The desired folder name

    Returns:
        Path to a unique directory that doesn't exist yet
    """
    base_dir = BATCH_TESTS_DIR / base_name

    if not base_dir.exists():
        return base_dir

    # Directory exists, find a unique suffix
    counter = 1
    while True:
        new_dir = BATCH_TESTS_DIR / f"{base_name}_{counter}"
        if not new_dir.exists():
            return new_dir
        counter += 1


def export_results_json(
    all_results: list[dict[str, Any]],
    dataset,
    batch_name: str,
    batch_config: BatchTestConfig,
    export_level: ExportLevel,
    logger,
    output_dir: Path,
) -> str:
    """Export all results to JSON file with .batch_analysis_report extension."""
    filename = f"results{FileExtensions.BATCH_ANALYSIS_REPORT}"
    filepath = output_dir / filename

    if not all_results:
        logger.warning("No results to export")
        return ""

    # Build metadata
    metadata = {
        "created_at": datetime.now().isoformat(),
        "version": "2.0",
        "batch_name": batch_name or batch_config.name,  # Use export name from UI
        "batch_description": batch_config.description,
        "export_level": export_level.name,
        "dataset_info": {
            "name": getattr(dataset, 'name', 'unknown'),
            "img_size": getattr(dataset, 'img_size', 0),
            "num_images": len(dataset.data) if hasattr(dataset, 'data') else 0,
        },
    }

    # Use the BatchAnalysisReport handler
    BatchAnalysisReport.save(
        results=all_results,
        metadata=metadata,
        path=filepath
    )

    logger.info("Exported %d results to: %s", len(all_results), filepath)
    return str(filepath)


def export_models(
    trained_models: dict[int, Any],
    logger,
    output_dir: str,
):
    """Export trained models (.pt and ONNX)."""
    import torch

    models_dir = os.path.join(output_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    for model_data in trained_models.values():
        postprocessor = model_data["postprocessor"]
        config = model_data["config"]

        # Safe name for files
        safe_name = config.name.replace(" ", "_").replace("/", "-")

        # Export PyTorch model
        pt_path = os.path.join(models_dir, f"{safe_name}.pt")
        try:
            torch.save(postprocessor.model.state_dict(), pt_path)
            logger.info("Exported model: %s", pt_path)
        except Exception as e:
            logger.error("Failed to export model %s: %s", safe_name, e)

        # Export ONNX model
        onnx_path = os.path.join(models_dir, f"{safe_name}.onnx")
        try:
            export_onnx(postprocessor, onnx_path)
            logger.info("Exported ONNX: %s", onnx_path)
        except Exception as e:
            logger.warning("Failed to export ONNX %s: %s", safe_name, e)


def export_onnx(postprocessor, onnx_path: str):
    """Export model to ONNX format."""
    import torch

    model = postprocessor.model
    device = postprocessor.device
    img_size = postprocessor.img_size
    is_conv = postprocessor.is_conv

    model.eval()

    # Create sample input
    if is_conv:
        sample = torch.randn(1, 1, img_size, img_size, device=device)
    else:
        sample = torch.randn(1, img_size * img_size, device=device)

    # Export to ONNX
    torch.onnx.export(
        model,
        sample,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )


def export_datasets(
    test_data: dict[int, dict],
    logger,
    output_dir: str,
):
    """Export all test data including masks and inference results (test images).

    Note: We don't export the original training dataset as it's not needed for reports.
    We only export the test images (ground truth, noisy/reconstructed, denoised) per test.
    """
    import numpy as np

    data_dir = os.path.join(output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    # Export per-test data
    for entry in test_data.values():
        config = entry["config"]
        safe_name = config.name.replace(" ", "_").replace("/", "-")
        test_dir = os.path.join(data_dir, safe_name)
        os.makedirs(test_dir, exist_ok=True)

        try:
            # Export mask patterns
            mask = entry["mask"]
            if hasattr(mask, "masks") and mask.masks is not None:
                np.savez_compressed(
                    os.path.join(test_dir, "masks.npz"),
                    masks=mask.masks
                )
                logger.debug("Exported masks for %s", safe_name)

            # Export test images: originals, reconstructions (after mask), denoised (after DNN)
            np.savez_compressed(
                os.path.join(test_dir, "test_images.npz"),
                originals=entry["originals"],
                reconstructions=entry["reconstructions"],
                denoised=entry["denoised"]
            )
            logger.debug("Exported test images for %s", safe_name)

            # Export test configuration as JSON for reference
            config_path = os.path.join(test_dir, "test_config.json")
            config_dict = {
                "name": config.name,
                "mask_type": config.mask_type,
                "reconstruction_method": config.reconstruction_method,
                "model_name": config.model_name,
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
            }
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)

            logger.info("Exported data for test: %s", safe_name)

        except Exception as e:
            logger.error("Failed to export data for %s: %s", safe_name, e)
