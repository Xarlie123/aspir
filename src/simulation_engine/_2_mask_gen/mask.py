import logging
import numpy as np
from abc import ABC, abstractmethod

class MaskABC(ABC):
    """Abstract base class for mask generation."""

    def __init__(self, img_size, logger=None):
        if logger is None:
            self.logger = logging.getLogger(self.__class__.__name__)
        else:
            self.logger = logger.getChild(self.__class__.__name__)
        self.logger.debug("Initializing %s with img_size=%d", self.__class__.__name__, img_size)

        self.img_size = img_size
        self.num_patterns = None
        self.masks = None
        self.mask_type = self.__class__.__name__

    @abstractmethod
    def generate_masks(self, progress_callback=None):
        """Abstract method to generate masks."""
        pass

    def validate_size(self):
        """Verify that the image size is a positive value."""
        self.logger.debug("Validating size: %d", self.img_size)
        if self.img_size <= 0:
            self.logger.error("Invalid image size: %d", self.img_size)
            raise ValueError("Image size must be greater than 0.")
        self.logger.debug("Size validated successfully")

    def save_masks(self, path: str | None = None, compress: bool = True) -> bool:
        """
        Save current masks to an NPZ file together with minimal metadata
        required to verify compatibility on load.
        """
        try:
            if self.masks is None:
                self.logger.error("No masks to save: 'self.masks' is None.")
                return False

            if path is None:
                path = f"masks_{self.mask_type}_{self.img_size}.npz"

            if self.masks.ndim != 3 or self.masks.shape[1] != self.img_size or self.masks.shape[2] != self.img_size:
                self.logger.error(
                    "Masks must have shape (N, %d, %d). Got %s",
                    self.img_size, self.img_size, self.masks.shape
                )
                return False

            self.num_patterns = int(self.masks.shape[0])

            saver = np.savez_compressed if compress else np.savez
            saver(
                path,
                masks=self.masks,
                img_size=np.int64(self.img_size),
                num_patterns=np.int64(self.num_patterns),
                mask_type=np.array(self.mask_type),
            )
            self.logger.info("Saved %d masks to %s (img_size=%d, type=%s)",
                             self.num_patterns, path, self.img_size, self.mask_type)
            return True
        except Exception as e:
            self.logger.exception("Failed to save masks: %s", e)
            return False

    def load_masks(self, path: str | None = None, progress_callback=None, mmap_mode: str | None = None) -> bool:
        """
        Load masks from an NPZ file into `self.masks` and update metadata.
        This method does not resize masks; it enforces shape compatibility with `self.img_size`.
        """
        try:
            if path is None:
                path = f"masks_{self.mask_type}_{self.img_size}.npz"

            if progress_callback:
                try:
                    progress_callback(0, 1)
                except Exception as cb_err:
                    self.logger.warning("Progress callback failed at start: %s", cb_err)

            with np.load(path, allow_pickle=False, mmap_mode=mmap_mode) as npz:
                if "masks" not in npz.files:
                    self.logger.error("Key 'masks' not found in NPZ: %s", path)
                    return False

                masks = npz["masks"]

                if masks.ndim != 3:
                    self.logger.error("Masks array must be 3D (N,H,W). Got shape %s", masks.shape)
                    return False

                n, h, w = masks.shape
                if (h != self.img_size) or (w != self.img_size):
                    self.logger.error(
                        "Mask size mismatch: expected (%d,%d), got (%d,%d).",
                        self.img_size, self.img_size, h, w
                    )
                    return False

                self.masks = masks
                self.num_patterns = int(n)

                if "img_size" in npz.files and int(npz["img_size"]) != self.img_size:
                    self.logger.warning("Loaded img_size (%s) differs from current (%d). Using current.",
                                        str(npz["img_size"]), self.img_size)
                if "mask_type" in npz.files:
                    saved_type = str(npz["mask_type"])
                    if saved_type != self.mask_type:
                        self.logger.info("Loaded masks saved as type '%s' (current instance type: '%s').",
                                         saved_type, self.mask_type)

            if progress_callback:
                try:
                    progress_callback(1, 1)
                except Exception as cb_err:
                    self.logger.warning("Progress callback failed at end: %s", cb_err)

            self.logger.info("Loaded %d masks from %s", self.num_patterns, path)
            return True

        except FileNotFoundError:
            self.logger.error("Masks file not found: %s", path)
            return False
        except Exception as e:
            self.logger.exception("Failed to load masks from %s: %s", path, e)
            return False
