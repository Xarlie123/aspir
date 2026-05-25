"""Load a single-test experiment from disk, restoring dataset/mask/model/preview."""
from __future__ import annotations

import importlib
import json
import os

import numpy as np
from PySide6.QtWidgets import QFileDialog, QMessageBox


def load_experiment(window):
    """
    Load a complete experiment from a folder.

    Ensures dataset/mask/postprocessor objects exist, loads artifacts,
    synthesizes preview if needed, and refreshes UI.
    """
    in_dir = QFileDialog.getExistingDirectory(
        window, "Select experiment folder to load", ""
    )
    if not in_dir:
        return

    window.logger.info("Loading experiment from folder: %s", in_dir)

    yaml_path           = os.path.join(in_dir, "config.yaml")
    ds_path             = os.path.join(in_dir, "dataset.npz")
    mk_path             = os.path.join(in_dir, "masks.npz")
    manifest_path       = os.path.join(in_dir, "manifest.json")
    model_path          = os.path.join(in_dir, "model.pth")
    model_manifest_path = os.path.join(in_dir, "model_manifest.json")
    test_results_path   = os.path.join(in_dir, "test_results.npz")
    metrics_path        = os.path.join(in_dir, "metrics.json")

    # ---- Read manifest (optional) ----
    manifest = {}
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f) or {}
            window.logger.debug("Manifest: %s", manifest)
        except Exception as e:
            window.logger.warning("Invalid manifest: %s", e)

    # ---- Helper to instantiate dataset/mask if missing ----
    def _instantiate_from_manifest(kind: str, fallback_img_size: int | None):
        """
        kind: 'dataset' or 'mask'. Creates instance using module/class if present
        or falls back to lightweight holders compatible with our load methods.
        """
        cls_name = manifest.get(f"{kind}_cls")
        mod_name = manifest.get(f"{kind}_module")
        img_size = manifest.get(f"{kind}_img_size", fallback_img_size)

        if cls_name and mod_name:
            try:
                mod = importlib.import_module(mod_name)
                cls = getattr(mod, cls_name)
                if kind == "dataset":
                    try:
                        return cls(getattr(window, "name", "LoadedDataset"), img_size, logger=window.logger)
                    except Exception:
                        return cls(img_size, logger=window.logger)
                else:
                    return cls(img_size=img_size, logger=window.logger)
            except Exception as e:
                window.logger.warning("Could not instantiate %s %s.%s: %s", kind, mod_name, cls_name, e)

        # Fallback holders
        if kind == "dataset":
            class _DatasetNPZHolder:
                """Minimal dataset holder for NPZ loading."""
                def __init__(self, img_size, logger):
                    self.name = "LoadedDataset"
                    self.img_size = int(img_size) if img_size is not None else None
                    self.data = []
                    self.dataset_type = "NPZHolder"
                    self.dataset_path = ""
                    self.logger = logger

                def save_dataset(self):
                    if self.dataset_path and self.data:
                        images_array = np.array(self.data)
                        np.savez(self.dataset_path, images=images_array)
                        self.logger.info("Dataset holder saved to %s", self.dataset_path)

                def load_data(self, progress_callback=None):
                    if not self.dataset_path or not os.path.exists(self.dataset_path):
                        self.logger.error("Dataset file not found: %s", self.dataset_path)
                        return False
                    with np.load(self.dataset_path, allow_pickle=False) as npz:
                        if "images" not in npz.files:
                            self.logger.error("Key 'images' not found in NPZ: %s", self.dataset_path)
                            return False
                        images = npz["images"]
                    if images.ndim < 3:
                        self.logger.error("Invalid images shape: %s", images.shape)
                        return False
                    if self.img_size is None:
                        self.img_size = int(images.shape[1])
                    self.data = list(images)
                    if progress_callback:
                        try:
                            progress_callback(1, 1)
                        except Exception:
                            pass
                    return True

            if img_size is None:
                try:
                    with np.load(ds_path, allow_pickle=False) as _npz:
                        im = _npz["images"]
                        img_size = int(im.shape[1])
                except Exception:
                    img_size = 0
            return _DatasetNPZHolder(img_size, logger=window.logger)

        else:
            from simulation_engine._2_mask_gen.mask import MaskABC
            class _MaskHolder(MaskABC):
                def generate_masks(self, progress_callback=None):
                    raise RuntimeError("Holder cannot generate masks; use load_masks().")
            if img_size is None:
                try:
                    with np.load(mk_path, allow_pickle=False) as _npz:
                        m = _npz["masks"]
                        img_size = int(m.shape[1])
                except Exception:
                    img_size = 0
            return _MaskHolder(img_size=img_size, logger=window.logger)

    # ---- Load YAML (UI state) ----
    ok_yaml = True
    try:
        if os.path.exists(yaml_path):
            window.config_yaml_handler.load_from_yaml(yaml_path)
        else:
            ok_yaml = False
            window.logger.warning("config.yaml not found in selected folder.")
    except Exception as e:
        ok_yaml = False
        window.logger.exception("Error loading YAML: %s", e)

    # ---- Ensure dataset exists, then load NPZ ----
    ok_dataset = True
    ds = getattr(window.simulation, "dataset", None)
    if ds is None and os.path.exists(ds_path):
        ds = _instantiate_from_manifest("dataset", fallback_img_size=None)
        if hasattr(window.simulation, "set_dataset"):
            try:
                window.simulation.set_dataset(ds)
            except Exception:
                window.simulation.dataset = ds
        else:
            window.simulation.dataset = ds

    try:
        if ds is not None and os.path.exists(ds_path):
            old_path = getattr(ds, "dataset_path", None)
            ds.dataset_path = ds_path
            loaded = bool(ds.load_data(progress_callback=None))
            if old_path is not None:
                ds.dataset_path = old_path
            ok_dataset = loaded
            if loaded:
                size = getattr(ds, "img_size", None) or 0
                try:
                    window.ui_dataset_handler.dataset_updated.emit(size)
                except Exception:
                    pass
                window.logger.info("Dataset loaded from %s", ds_path)
            else:
                window.logger.error("Failed to load dataset from %s", ds_path)
        else:
            ok_dataset = False
            window.logger.warning("No dataset.npz or could not create dataset.")
    except Exception as e:
        ok_dataset = False
        window.logger.exception("Error loading dataset: %s", e)

    # ---- Ensure mask exists, then load NPZ ----
    ok_masks = True
    mk = getattr(window.simulation, "mask", None)
    if mk is None and os.path.exists(mk_path):
        mk = _instantiate_from_manifest("mask", fallback_img_size=getattr(ds, "img_size", None))
        if hasattr(window.simulation, "set_mask"):
            try:
                window.simulation.set_mask(mk)
            except Exception:
                window.simulation.mask = mk
        else:
            window.simulation.mask = mk

    try:
        if mk is not None and os.path.exists(mk_path):
            if hasattr(mk, "load_masks"):
                loaded = bool(mk.load_masks(path=mk_path, progress_callback=None, mmap_mode=None))
                ok_masks = loaded
                if loaded:
                    try:
                        window.ui_mask_handler.mask_created.emit(
                            window.simulation.dataset,
                            window.simulation.mask,
                            getattr(window.simulation, "applicator", None)
                        )
                    except Exception:
                        pass
                    window.logger.info("Masks loaded from %s", mk_path)
                else:
                    window.logger.error("Failed to load masks from %s", mk_path)
            else:
                ok_masks = False
                window.logger.error("Mask object does not expose load_masks(path=...).")
        else:
            ok_masks = False
            window.logger.warning("No masks.npz or could not create mask.")
    except Exception as e:
        ok_masks = False
        window.logger.exception("Error loading masks: %s", e)

    # ---- Load model + preview + metrics (optional) ----
    try:
        from simulation_engine._4_postprocessor.postprocessor_nn import PostprocessorNN

        model_manifest = {}
        if os.path.exists(model_manifest_path):
            try:
                with open(model_manifest_path, "r", encoding="utf-8") as f:
                    model_manifest = json.load(f) or {}
                window.logger.debug("Model manifest: %s", model_manifest)
            except Exception as e:
                window.logger.warning("Could not read model_manifest.json: %s", e)

        # Create engine if model exists
        if os.path.exists(model_path):
            pp = getattr(window.simulation, "postprocessor", None)
            if pp is None:
                model_name = (model_manifest.get("model_name") or "").lower()
                overrides  = model_manifest.get("overrides") or {}
                if "img_size" not in overrides or overrides["img_size"] is None:
                    overrides["img_size"] = getattr(window.simulation.dataset, "img_size", None)
                try:
                    window.simulation.set_postprocessor(
                        window.simulation.dataset,
                        window.simulation.mask,
                        getattr(window.simulation, "applicator", None),
                        postprocesador_cls=PostprocessorNN,
                        model_name=model_name,
                        model_overrides=overrides,
                        batch_size=16,
                        lr=1e-3,
                        weight_decay=1e-5
                    )
                except Exception as e:
                    window.logger.exception("Could not create PostprocessorNN: %s", e)
                    window.simulation.postprocessor = PostprocessorNN(
                        model_name=model_name,
                        model_overrides=overrides,
                        dataset=window.simulation.dataset,
                        applicator=getattr(window.simulation, "applicator", None),
                        batch_size=16,
                        lr=1e-3,
                        weight_decay=1e-5,
                        logger=window.logger
                    )

            # Load weights and mark as trained
            try:
                window.simulation.postprocessor.load_model(model_path)
                window.simulation.postprocessor.trained = True
                # Friendly type for other tabs
                window.simulation.postprocessor.postproc_type = model_manifest.get("model_name", "NN")
                window.logger.info("Model loaded from %s", model_path)
            except Exception as e:
                window.logger.exception("Error loading model: %s", e)

        # Load saved preview if present
        if os.path.exists(test_results_path):
            try:
                with np.load(test_results_path, allow_pickle=False) as npz:
                    orig = npz["original"]
                    rec  = npz["recons"]
                    den  = npz["denoised"]
                window.simulation.validation_results = {
                    "original": list(orig),
                    "recons":   list(rec),
                    "denoised": list(den)
                }
                viz = getattr(window.ui_postprocessor_handler, "visual_pp", None)
                if viz is not None:
                    viz.set_images(window.simulation.validation_results["original"],
                                   window.simulation.validation_results["recons"],
                                   window.simulation.validation_results["denoised"])
                    model_name_for_info = model_manifest.get("model_name", "")
                    if not model_name_for_info and hasattr(window.ui_postprocessor_handler, "get_current_model"):
                        model_name_for_info = window.ui_postprocessor_handler.get_current_model()
                    viz.update_info(
                        num_images=len(window.simulation.validation_results["denoised"]),
                        img_size=getattr(window.simulation.dataset, "img_size", 0),
                        dataset_type=getattr(window.simulation.dataset, "dataset_type", ""),
                        mask_type=type(getattr(window.simulation, "mask", object())).__name__,
                        postprocessor_type=model_name_for_info,
                        n_params=getattr(window.simulation.postprocessor, "n_params", None)
                    )
                    viz.image_slider_value.setValue(0)
                window.logger.info("Test results loaded from %s", test_results_path)
            except Exception as e:
                window.logger.warning("Could not load test_results: %s", e)

        # Load metrics if present
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, "r", encoding="utf-8") as f:
                    metrics = json.load(f) or {}
                viz = getattr(window.ui_postprocessor_handler, "visual_pp", None)
                if viz is not None:
                    viz.val_losses = metrics.get("val_losses", [])
                    viz.test_losses = metrics.get("test_losses", [])
                    viz.plot_losses()
                window.logger.info("Metrics loaded from %s", metrics_path)
            except Exception as e:
                window.logger.warning("Could not load metrics: %s", e)

        # ---- If no preview saved, synthesize from the loaded model ----
        try:
            if not getattr(window.simulation, "validation_results", None):
                pp = getattr(window.simulation, "postprocessor", None)
                if pp is not None and getattr(pp, "trained", False):
                    orig, recons, denoised = pp.test_dataset()
                    window.simulation.validation_results = {
                        "original": orig,
                        "recons":   recons,
                        "denoised": denoised
                    }
                    viz = getattr(window.ui_postprocessor_handler, "visual_pp", None)
                    if viz is not None:
                        viz.set_images(orig, recons, denoised)
                        model_name_for_info = ""
                        if os.path.exists(model_manifest_path):
                            try:
                                with open(model_manifest_path, "r", encoding="utf-8") as f:
                                    _mm = json.load(f) or {}
                                model_name_for_info = _mm.get("model_name", "")
                            except Exception:
                                pass
                        if not model_name_for_info and hasattr(window.ui_postprocessor_handler, "get_current_model"):
                            model_name_for_info = window.ui_postprocessor_handler.get_current_model()
                        viz.update_info(
                            num_images=len(denoised),
                            img_size=getattr(window.simulation.dataset, "img_size", 0),
                            dataset_type=getattr(window.simulation.dataset, "dataset_type", ""),
                            mask_type=type(getattr(window.simulation, "mask", object())).__name__,
                            postprocessor_type=model_name_for_info,
                            n_params=getattr(pp, "n_params", None)
                        )
                        viz.image_slider_value.setValue(0)
                    pp.postproc_type = model_name_for_info or "NN"
                    window.logger.info("Preview synthesized from loaded model (no test_results.npz found)")
        except Exception as e:
            window.logger.warning("Could not synthesize preview from model: %s", e)

    except Exception as e:
        window.logger.exception("General error loading model/results: %s", e)

    # ---- Final safety: ask handler to refresh preview if possible ----
    try:
        window.ui_postprocessor_handler.refresh_preview_from_state()
    except Exception:
        pass

    # ---- Summary dialog ----
    msg = [
        f"YAML: {'OK' if ok_yaml else 'NO'}",
        f"Dataset: {'OK' if ok_dataset else 'NO'}",
        f"Masks: {'OK' if ok_masks else 'NO'}",
    ]
    QMessageBox.information(window, "Load experiment", "\n".join(msg))
