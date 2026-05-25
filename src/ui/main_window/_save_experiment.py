"""Save a complete single-test experiment to a ``.single_test_experiment`` folder."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PySide6.QtWidgets import QFileDialog, QMessageBox

from ui.utils.file_formats import (
    SINGLE_TESTS_DIR,
    FileExtensions,
    SingleTestExperiment,
)


def save_experiment(window):
    """
    Save a complete experiment to a .single_test_experiment directory.

    Structure:
    <name>.single_test_experiment/
    ├── config.json              # Widget configurations
    ├── datasets/
    │   └── dataset.npz          # Training/test/validation data
    ├── masks/
    │   └── masks.npz            # Generated mask patterns
    ├── model/
    │   ├── model.pt             # PyTorch weights
    │   ├── model.onnx           # ONNX export (if available)
    │   └── manifest.json        # Model metadata
    └── results/
        ├── test_results.npz     # Validation triplets
        └── metrics.json         # Training curves
    """
    # Get experiment name from user
    default_name = f"experiment_{time.strftime('%Y%m%d_%H%M%S')}"
    name, ok = QFileDialog.getSaveFileName(
        window,
        "Save Experiment",
        str(SINGLE_TESTS_DIR / f"{default_name}{FileExtensions.SINGLE_TEST_EXPERIMENT}"),
        f"Single Test Experiment (*{FileExtensions.SINGLE_TEST_EXPERIMENT});;All Files (*.*)"
    )

    if not name or not ok:
        return

    # Ensure correct extension
    exp_path = Path(name)
    if exp_path.suffix != FileExtensions.SINGLE_TEST_EXPERIMENT:
        exp_path = exp_path.with_suffix(FileExtensions.SINGLE_TEST_EXPERIMENT)

    window.logger.info("Saving experiment to: %s", exp_path)

    # Create directory structure
    dirs = SingleTestExperiment.create_structure(exp_path)

    ok_config = ok_dataset = ok_masks = False
    ok_model = ok_onnx = ok_test_results = ok_metrics = False

    # ---- Save Config (JSON) ----
    try:
        config_data = window.config_yaml_handler._collect_config_data()
        SingleTestExperiment.save_config(exp_path, config_data)
        ok_config = True
        window.logger.info("Config saved to %s", dirs["root"] / "config.json")
    except Exception as e:
        window.logger.exception("Error saving config: %s", e)

    # ---- Save Dataset ----
    ds = getattr(window.simulation, "dataset", None)
    try:
        if ds is not None and getattr(ds, "data", None) is not None and len(ds.data) > 0:
            ds_path = dirs["datasets"] / "dataset.npz"
            old_path = getattr(ds, "dataset_path", None)
            ds.dataset_path = str(ds_path)
            ds.save_dataset()
            if old_path is not None:
                ds.dataset_path = old_path
            ok_dataset = True
            window.logger.info("Dataset saved to %s", ds_path)
        else:
            window.logger.warning("No dataset in memory to save.")
    except Exception as e:
        window.logger.exception("Error saving dataset: %s", e)

    # ---- Save Masks ----
    mk = getattr(window.simulation, "mask", None)
    try:
        if mk is not None and getattr(mk, "masks", None) is not None:
            mk_path = dirs["masks"] / "masks.npz"
            if hasattr(mk, "save_masks"):
                ok_masks = bool(mk.save_masks(path=str(mk_path), compress=True))
                if ok_masks:
                    window.logger.info("Masks saved to %s", mk_path)
            else:
                window.logger.error("Mask object does not implement save_masks(path=...).")
        else:
            window.logger.warning("No masks in memory to save.")
    except Exception as e:
        window.logger.exception("Error saving masks: %s", e)

    # ---- Save Model + Results + Metrics ----
    pp = getattr(window.simulation, "postprocessor", None)
    if pp is not None and getattr(pp, "model", None) is not None:
        # Save PyTorch weights
        model_path = dirs["model"] / "model.pt"
        try:
            pp.save_model(str(model_path))
            ok_model = True
            window.logger.info("Model saved to %s", model_path)
        except Exception as e:
            window.logger.exception("Could not save model: %s", e)

        # Export to ONNX
        onnx_path = dirs["model"] / "model.onnx"
        try:
            import torch
            model = pp.model
            device = pp.device
            img_size = pp.img_size
            is_conv = pp.is_conv
            model.eval()

            if is_conv:
                sample = torch.randn(1, 1, img_size, img_size, device=device)
            else:
                sample = torch.randn(1, img_size * img_size, device=device)

            torch.onnx.export(
                model, sample, str(onnx_path),
                export_params=True, opset_version=17,
                do_constant_folding=True,
                input_names=['input'], output_names=['output'],
                dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
            )
            ok_onnx = True
            window.logger.info("ONNX model saved to %s", onnx_path)
        except Exception as e:
            window.logger.warning("Could not export ONNX: %s", e)

        # Save model manifest
        manifest_path = dirs["model"] / "manifest.json"
        try:
            model_name = ""
            if hasattr(window.ui_postprocessor_handler, "get_current_model"):
                from simulation_engine._4_postprocessor.postprocessor_nn import display_to_key
                model_name = display_to_key(window.ui_postprocessor_handler.get_current_model())
            elif hasattr(pp, "model_name"):
                model_name = str(pp.model_name).lower()

            model_manifest = {
                "model_name": model_name,
                "img_size": getattr(window.simulation.dataset, "img_size", None),
                "is_conv": getattr(pp, "is_conv", None),
                "n_params": int(pp.n_params) if hasattr(pp, "n_params") else None,
                "has_onnx": ok_onnx,
            }
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(model_manifest, f, indent=2, ensure_ascii=False)
        except Exception as e:
            window.logger.warning("Could not write model manifest: %s", e)

        # Save validation preview triplets
        results_path = dirs["results"] / "test_results.npz"
        try:
            vr = getattr(window.simulation, "validation_results", None)
            if vr and all(k in vr for k in ("original", "recons", "denoised")):
                np.savez(
                    str(results_path),
                    original=np.asarray(vr["original"], dtype=np.float32),
                    recons=np.asarray(vr["recons"], dtype=np.float32),
                    denoised=np.asarray(vr["denoised"], dtype=np.float32),
                )
                ok_test_results = True
                window.logger.info("Test results saved to %s", results_path)
        except Exception as e:
            window.logger.warning("Could not save test_results: %s", e)

        # Save training curves
        metrics_path = dirs["results"] / "metrics.json"
        try:
            viz = getattr(window.ui_postprocessor_handler, "visual_pp", None)
            if viz is not None and hasattr(viz, "val_losses") and hasattr(viz, "test_losses"):
                metrics = {
                    "val_losses": list(getattr(viz, "val_losses", [])),
                    "test_losses": list(getattr(viz, "test_losses", [])),
                }
                with open(metrics_path, "w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=2, ensure_ascii=False)
                ok_metrics = True
                window.logger.info("Training metrics saved to %s", metrics_path)
        except Exception as e:
            window.logger.warning("Could not save metrics: %s", e)
    else:
        window.logger.info("No trained/loaded postprocessor to save model/results.")

    # ---- Summary dialog ----
    msg = [
        f"Experiment saved to: {exp_path.name}",
        "",
        f"Config: {'OK' if ok_config else 'NO'}",
        f"Dataset: {'OK' if ok_dataset else 'NO'}",
        f"Masks: {'OK' if ok_masks else 'NO'}",
        f"Model (.pt): {'OK' if ok_model else 'NO'}",
        f"Model (.onnx): {'OK' if ok_onnx else 'NO'}",
        f"Test results: {'OK' if ok_test_results else 'NO'}",
        f"Metrics: {'OK' if ok_metrics else 'NO'}",
    ]
    QMessageBox.information(window, "Save Experiment", "\n".join(msg))
