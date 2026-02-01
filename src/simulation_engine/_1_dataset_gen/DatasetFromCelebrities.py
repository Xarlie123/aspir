# File: Simulacion/dataset_gen/DatasetFromCelebrities.py
# Always comment Python code in English

import os
import logging
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import cv2
import glob
import subprocess
import shutil
import stat


from simulation_engine._1_dataset_gen.dataset import DatasetABC


class DatasetFromCelebrities(DatasetABC):
    """
    Drop-in replacement for the previous Celebrities reader.

    Instead of reading Celebrities TFRecords with TensorFlow, this class downloads and
    loads images from the Kaggle dataset:
        arnrob/celeba-small-images-dataset
    (cropped/resized CelebA faces, typically 64x64 RGB)

    Output:
      - self.data:   (N, H, W) uint8 grayscale for your visualizer
      - self.data_f32: (N, H, W) float32 [0,1] (optional convenience)
      - self.labels: List[str] (we store the filename for debugging)
    """

    # Kaggle dataset identifier
    KAGGLE_DATASET = "arnrob/celeba-small-images-dataset"

    def __init__(
        self,
        name: str,
        img_size: int,
        num_images: int,
        seed: Optional[int] = None,
        use_split: str = "train",           # kept for API compatibility (ignored here)
        download_dir: str = "../datasets/tmp_downloaded_datasets/celebrities",    # download to repo root datasets folder
        logger: Optional[logging.Logger] = None,
        max_shards: Optional[int] = None,   # kept for API compatibility (ignored here)
        custom_urls: Optional[List[str]] = None,  # kept for API compatibility (ignored here)
        expand_views: bool = True,          # kept for API compatibility (ignored here)
        kaggle_subdir: Optional[str] = None, # if dataset unzips into a known subfolder
        data_format: Optional[str] = None,
        speckle_noise: float = 0.0
    ):
        super().__init__(name, img_size, logger, data_format=data_format, speckle_noise=speckle_noise)
        self.num_images = int(num_images)
        self.seed = seed
        self.use_split = use_split
        self.download_dir = download_dir
        self.labels: Optional[List[str]] = None
        self.kaggle_subdir = kaggle_subdir  # e.g. "celeba_small" if needed

        self.logger.debug(
            "Initializing DatasetFromCelebrities(Kaggle): name=%s, img_size=%d, num_images=%d, seed=%s, dir=%s, format=%s, speckle=%.2f",
            name, img_size, num_images, seed, download_dir, self.data_format, self.speckle_noise
        )

    # ------------------------------- Kaggle download helpers -------------------------------

    def _ensure_kaggle_download(self, progress_callback=None) -> str:
        """
        Ensure the Kaggle dataset is present locally.
        Prefers the Kaggle CLI (more robust) and falls back to the Python API with force=True.
        Returns the directory path that contains the images.

        Args:
            progress_callback: Optional callable(current, total) for progress reporting during download
        """
        # Target root
        root = os.path.join(self.download_dir, "kaggle_celeba_small")
        os.makedirs(root, exist_ok=True)

        # If images already exist, skip downloading
        if self._count_images(root) > 0:
            self.logger.info("Images already present in: %s (Kaggle download skipped)", root)
            return root

        # Fix loose permissions on kaggle.json if present (silence warnings)
        try:
            cfg = os.path.join(os.path.expanduser("~"), ".kaggle", "kaggle.json")
            if os.path.isfile(cfg):
                os.chmod(cfg, stat.S_IRUSR | stat.S_IWUSR)  # 0o600
        except Exception:
            # Non-fatal; continue
            pass

        # 1) Prefer the CLI if available
        cli = shutil.which("kaggle")
        if cli is not None:
            self.logger.info(
                "Downloading Kaggle dataset '%s' with CLI -> %s (this may take a while)",
                self.KAGGLE_DATASET, root
            )
            if progress_callback:
                progress_callback(0, 100)  # Start progress

            cmd = [
                cli, "datasets", "download",
                "-d", self.KAGGLE_DATASET,
                "-p", root,
                "--force",  # overwrite / skip date parsing
                "--unzip",  # unzip automatically
                "--quiet"
            ]
            try:
                # Note: if you want to see CLI output, remove stdout/stderr redirection
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                if progress_callback:
                    progress_callback(100, 100)  # Complete progress
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to run Kaggle CLI: {e}") from e

            if self._count_images(root) == 0:
                raise RuntimeError(
                    f"No images found after downloading {self.KAGGLE_DATASET} with CLI. "
                    f"Check the contents in: {root}"
                )
            return root

        # 2) Fallback: Kaggle Python API with force=True (avoids Last-Modified parsing bug)
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except Exception as e:
            raise ModuleNotFoundError(
                "The 'kaggle' package is not available and the 'kaggle' CLI was not found. "
                "Install 'kaggle' or add the CLI to PATH."
            ) from e

        api = KaggleApi()
        try:
            api.authenticate()
        except Exception as e:
            raise RuntimeError(
                "Could not authenticate with Kaggle. Ensure you have ~/.kaggle/kaggle.json "
                "with 600 permissions or set KAGGLE_USERNAME/KAGGLE_KEY environment variables."
            ) from e

        self.logger.info(
            "Downloading Kaggle dataset '%s' with API -> %s (this may take a while)",
            self.KAGGLE_DATASET, root
        )

        # Create a wrapper for progress reporting if callback is provided
        if progress_callback:
            progress_callback(0, 100)

        # The 'force=True' flag skips date comparison and avoids the Python 3.12 strptime issue
        api.dataset_download_files(
            self.KAGGLE_DATASET,
            path=root,
            unzip=True,
            force=True,  # <-- key fix
            quiet=True
        )

        if progress_callback:
            progress_callback(100, 100)  # Signal completion

        if self._count_images(root) == 0:
            raise RuntimeError(
                f"No images found after downloading {self.KAGGLE_DATASET} with API. "
                f"Check the uncompressed structure in: {root}"
            )

        return root

    @staticmethod
    def _is_image(path: str) -> bool:
        ext = os.path.splitext(path)[1].lower()
        return ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

    def _gather_image_paths(self, root: str) -> List[str]:
        """
        Recursively gather image files in 'root' (or subdir if provided).
        """
        base = os.path.join(root, self.kaggle_subdir) if self.kaggle_subdir else root
        # Use glob to match common image extensions
        patterns = ["**/*.jpg", "**/*.jpeg", "**/*.png", "**/*.bmp", "**/*.webp", "**/*.tif", "**/*.tiff"]
        files: List[str] = []
        for p in patterns:
            files.extend(glob.glob(os.path.join(base, p), recursive=True))
        # Filter (safety) and sort for deterministic ordering before sampling
        files = [f for f in files if os.path.isfile(f) and self._is_image(f)]
        files.sort()
        return files

    def _count_images(self, root: str) -> int:
        return len(self._gather_image_paths(root))

    # ------------------------------- Public API -------------------------------

    def load_data(self, progress_callback=None):
        """
        Download (if needed) and load grayscale 2D images resized to img_size.

        During download phase (0-50% progress), calls progress_callback(current, 100).
        During loading phase (50-100% progress), calls progress_callback(50 + current//2, 100).

        Exports:
          - self.data: (N, H, W) uint8 [0..255]
          - self.data_f32: (N, H, W) float32 [0..1]
          - self.labels: filenames (str)
        """
        from time import perf_counter

        start = perf_counter()

        # Create a wrapper callback that adjusts progress range for download phase
        def download_progress(current, total):
            # Download phase: 0-50% of overall progress
            overall_progress = int((current / total) * 50)
            if progress_callback:
                progress_callback(overall_progress, 100)

        # Ensure images are available locally (with progress reporting)
        images_root = self._ensure_kaggle_download(progress_callback=download_progress)
        all_paths = self._gather_image_paths(images_root)
        total = len(all_paths)
        if total == 0:
            raise RuntimeError(f"No images found in {images_root} after download.")

        # Deterministic sampling
        rng = np.random.default_rng(self.seed)
        if self.num_images is None or self.num_images > total:
            self.logger.warning("num_images (%s) > total images (%d). Using N=%d", self.num_images, total, total)
            self.num_images = total
        idx = rng.choice(total, size=self.num_images, replace=False)
        sel_paths = [all_paths[i] for i in idx]

        mats_u8 = np.empty((self.num_images, self.img_size, self.img_size), dtype=np.uint8)
        labels: List[str] = []

        # Load, convert to grayscale 2D, resize
        for i, path in enumerate(sel_paths):
            # Read with OpenCV (BGR)
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                self.logger.warning("Could not read image: %s (skipping)", path)
                # Fill with zeros to keep shape consistent
                mats_u8[i] = 0
                labels.append(os.path.basename(path))
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if gray.shape[0] != self.img_size or gray.shape[1] != self.img_size:
                gray = cv2.resize(gray, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
            mats_u8[i] = gray
            labels.append(os.path.basename(path))

            # Loading phase: 50-100% of overall progress
            if progress_callback and ((i + 1) % 50 == 0 or (i + 1) == self.num_images):
                loading_progress = 50 + int((i + 1) / self.num_images * 50)
                progress_callback(loading_progress, 100)

        self.data = list(mats_u8)  # Convert to list for consistency with base class
        self.data_f32 = mats_u8.astype(np.float32) / 255.0
        self.labels = labels

        # Apply data format conversion
        self.apply_data_format()

        # Apply speckle noise if configured
        self.apply_speckle_noise()

        elapsed = perf_counter() - start
        self.logger.info("Kaggle dataset (CelebA-small) ready: %d images in %.3f s (format: %s, speckle: %.2f)",
                        self.num_images, elapsed, self.data_format, self.speckle_noise)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Flatten grayscale images to a DataFrame and append 'label' (filename).
        """
        if getattr(self, "data", None) is None or len(self.data) == 0:
            return pd.DataFrame()

        src = getattr(self, "data_f32", self.data.astype(np.float32) / 255.0)
        num = src.shape[0]
        flat = src.reshape(num, -1)
        df = pd.DataFrame(flat)
        if self.labels is not None:
            df["label"] = self.labels
        return df
