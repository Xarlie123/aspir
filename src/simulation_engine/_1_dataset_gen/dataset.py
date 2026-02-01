# File: Simulacion/dataset_gen/dataset.py

import os
import cv2
import numpy as np
import logging
from abc import ABC, abstractmethod

# Default data format
DEFAULT_DATA_FORMAT = "FP32"

class DatasetABC(ABC):
    """Clase base abstracta para la generación de datasets."""

    def __init__(self, name, img_size, logger=None, data_format=None, speckle_noise=0.0):
        # Logger
        if logger is None:
            self.logger = logging.getLogger(self.__class__.__name__)
        else:
            self.logger = logger.getChild(self.__class__.__name__)
        self.logger.debug(f"Initializing dataset '{name}' de tipo {self.__class__.__name__}")

        # Atributos básicos
        self.name = name
        self.img_size = img_size
        self.data = []  # Lista para almacenar imágenes
        self.dataset_path = f"{self.name}.npz"
        self.dataset_type = self.__class__.__name__
        self.num_images = None

        # Data format for quantization testing (FP32, INT8, INT4)
        self.data_format = data_format if data_format is not None else DEFAULT_DATA_FORMAT
        self.logger.info("Data format set to: %s", self.data_format)

        # Speckle noise level (0.0 = no noise, 1.0 = maximum noise)
        self.speckle_noise = max(0.0, min(1.0, speckle_noise))
        if self.speckle_noise > 0:
            self.logger.info("Speckle noise set to: %.2f", self.speckle_noise)

    @abstractmethod
    def load_data(self, progress_callback=None):
        """
        English comment:
        Load dataset images from `self.dataset_path` (NPZ file) into memory,
        validate their shapes, optionally resize them to (img_size, img_size),
        and populate `self.data` and `self.num_images`.

        Parameters
        ----------
        progress_callback : callable or None
            Optional callback taking (current_index, total) to report progress.

        Returns
        -------
        bool
            True if load succeeded, False otherwise.
        """
        try:
            if not os.path.exists(self.dataset_path):
                self.logger.error("Dataset file not found: %s", self.dataset_path)
                return False

            # English comment: load NPZ safely and check required key
            with np.load(self.dataset_path, allow_pickle=False) as npz:
                self.logger.debug("NPZ keys: %s", npz.files)
                images = npz.get("images", None)
                if images is None:
                    self.logger.error("Key 'images' missing in NPZ")
                    return False
                self.logger.debug("images shape: %s dtype: %s", images.shape, images.dtype)

            # English comment: normalize dtype (uint8 or float32 are typical)
            if images.dtype not in (np.uint8, np.float32, np.float64):
                self.logger.warning("Unexpected dtype %s; converting to float32", images.dtype)
                images = images.astype(np.float32, copy=False)

            # English comment: ensure list of images for downstream flexibility
            # If array is 3D (N,H,W) or 4D (N,H,W,C), split into a list
            if images.ndim < 3:
                self.logger.error("Images array must have at least 3 dims (N,H,W). Got %s", images.shape)
                return False

            total = images.shape[0]
            loaded = []
            resize_needed = False

            # English comment: quick check if resizing is required
            # Case 3D (N,H,W) or 4D (N,H,W,C) -> check uniform H,W equal to img_size
            h, w = images.shape[1], images.shape[2]
            if h != self.img_size or w != self.img_size:
                resize_needed = True

            # English comment: iterate to optionally resize and to support progress callback
            for i in range(total):
                img = images[i]

                # English comment: if grayscale (H,W), expand channel dim to keep consistent handling later if needed
                if img.ndim == 2:
                    pass  # keep as (H,W); cv2.resize works with 2D as well
                elif img.ndim == 3:
                    pass  # (H,W,C) is fine
                else:
                    self.logger.error("Unsupported image ndim=%d at index %d", img.ndim, i)
                    return False

                if resize_needed:
                    # English comment: choose interpolation based on dtype (area is robust for downscale)
                    interp = cv2.INTER_AREA
                    img = cv2.resize(img, (self.img_size, self.img_size), interpolation=interp)

                loaded.append(img)

                if progress_callback is not None:
                    # English comment: report progress as 1-based index for human-friendly display
                    try:
                        progress_callback(i + 1, total)
                    except Exception as cb_err:
                        # English comment: don't break loading if callback fails
                        self.logger.warning("Progress callback failed at %d/%d: %s", i + 1, total, cb_err)

            # English comment: final consistency check (all same size)
            if not self.check_image_sizes(loaded):
                self.logger.warning("Not all images share the same size after load; forcing resize.")
                loaded = self.resize_images(loaded)

            self.data = loaded
            self.num_images = len(self.data)
            self.logger.info("Loaded %d images from %s (img_size=%d)", self.num_images, self.dataset_path,
                             self.img_size)
            return True

        except Exception as e:
            self.logger.exception("Failed to load dataset from %s: %s", self.dataset_path, e)
            return False

    def resize_images(self, images):
        """Redimensiona todas las imágenes al tamaño especificado."""
        self.logger.debug("Redimensionando %d imágenes a %dx%d", len(images), self.img_size, self.img_size)
        return [cv2.resize(img, (self.img_size, self.img_size)) for img in images]

    def check_image_sizes(self, images):
        """Verifica si todas las imágenes tienen el mismo tamaño."""
        sizes = {img.shape[:2] for img in images}
        ok = len(sizes) == 1
        self.logger.debug("Verificación de tamaños uniforme: %s", ok)
        return ok

    def save_dataset(self):
        """Save the dataset to an NPZ file at dataset_path."""
        images_array = np.array(self.data)
        np.savez(self.dataset_path, images=images_array)
        self.logger.info("Dataset saved to %s", self.dataset_path)

    @abstractmethod
    def to_dataframe(self):
        """Abstract method para convertir el dataset en un DataFrame de Pandas."""
        pass

    def apply_data_format(self):
        """
        Apply the specified data format conversion to all images in self.data.
        Called after loading/generating data to ensure proper quantization.
        """
        if not self.data:
            self.logger.warning("No data to convert")
            return

        self.logger.info("Applying data format %s to %d images", self.data_format, len(self.data))

        converted = []
        for img in self.data:
            converted.append(self._convert_image_format(img))
        self.data = converted

        self.logger.debug("Data format conversion complete")

    def _convert_image_format(self, img: np.ndarray) -> np.ndarray:
        """
        Convert a single image to the specified data format.
        All formats output values in [0, 1] range for consistency.

        Supported formats:
            - FP32: Full 32-bit float precision (for computers)
            - INT8: 8-bit integer quantization with 256 levels (for embedded systems)
            - INT4: 4-bit integer quantization with 16 levels (for FPGA)

        Args:
            img: Input numpy array (image)

        Returns:
            Converted numpy array (always float32 with quantization applied)
        """
        # First normalize to float32 [0, 1] range
        if img.dtype == np.uint8:
            img_f32 = img.astype(np.float32) / 255.0
        else:
            img_f32 = img.astype(np.float32)

        if self.data_format == "FP32":
            return img_f32

        elif self.data_format == "INT8":
            # Quantize to 256 levels (8-bit)
            num_levels = 256
            min_val = img_f32.min()
            max_val = img_f32.max()
            if max_val - min_val > 0:
                normalized = (img_f32 - min_val) / (max_val - min_val)
                quantized = np.round(normalized * (num_levels - 1)) / (num_levels - 1)
                result = quantized * (max_val - min_val) + min_val
            else:
                result = img_f32
            return result.astype(np.float32)

        elif self.data_format == "INT4":
            # Quantize to 16 levels (4-bit) - typical for FPGA deployment
            num_levels = 16
            min_val = img_f32.min()
            max_val = img_f32.max()
            if max_val - min_val > 0:
                normalized = (img_f32 - min_val) / (max_val - min_val)
                quantized = np.round(normalized * (num_levels - 1)) / (num_levels - 1)
                result = quantized * (max_val - min_val) + min_val
            else:
                result = img_f32
            return result.astype(np.float32)

        else:
            self.logger.warning("Unknown data format: %s, using FP32", self.data_format)
            return img_f32

    def get_numpy_dtype(self) -> np.dtype:
        """Get the numpy dtype for the current data format."""
        # All formats are stored as float32 after quantization simulation
        dtype_map = {
            "FP32": np.float32,
            "INT8": np.float32,  # Stored as float32 after quantization
            "INT4": np.float32,  # Stored as float32 after quantization
        }
        return dtype_map.get(self.data_format, np.float32)

    def apply_speckle_noise(self):
        """
        Apply speckle noise to all images in self.data.
        Speckle noise simulates the multiplicative noise pattern
        typical of coherent imaging systems (radar, ultrasound, IR sensors).

        Should be called after loading/generating data.
        """
        if self.speckle_noise <= 0 or not self.data:
            return

        self.logger.info("Applying speckle noise (level=%.2f) to %d images",
                        self.speckle_noise, len(self.data))

        noisy_data = []
        for img in self.data:
            noisy_img = self._add_speckle_to_image(img, self.speckle_noise)
            noisy_data.append(noisy_img)

        self.data = noisy_data
        self.logger.debug("Speckle noise applied to all images")

    def _add_speckle_to_image(self, img: np.ndarray, noise_level: float) -> np.ndarray:
        """
        Add speckle noise to a single image.

        Speckle noise is multiplicative: noisy = img * (1 + noise_level * N)
        where N is standard normal noise.

        Args:
            img: Input image (numpy array)
            noise_level: Noise intensity (0.0 to 1.0)

        Returns:
            Noisy image (same dtype as input, clipped to valid range)
        """
        # Generate multiplicative noise
        noise = np.random.randn(*img.shape).astype(np.float32)
        noisy = img.astype(np.float32) * (1 + noise_level * noise)

        # Clip to valid range
        if img.dtype == np.uint8:
            noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        else:
            # For float images, assume [0, 1] range
            noisy = np.clip(noisy, 0, 1).astype(np.float32)

        return noisy
