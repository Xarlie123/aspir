# File: Simulacion/dataset_gen/DatasetFromSVHN.py
import os
import logging
import numpy as np
import pandas as pd
import cv2
from urllib.request import urlretrieve
from scipy.io import loadmat
from time import perf_counter

from simulation_engine._1_dataset_gen.dataset import DatasetABC

class DatasetFromSVHN(DatasetABC):
    """
    Loads images from the SVHN (Street View House Numbers) dataset.
    Downloads .mat files if they don't exist, allows sampling and resizing.
    """

    SVHN_TRAIN_URL = "http://ufldl.stanford.edu/housenumbers/train_32x32.mat"
    SVHN_TEST_URL  = "http://ufldl.stanford.edu/housenumbers/test_32x32.mat"

    def __init__(self, name, img_size, num_images, seed=None, use_split="train", download_dir="../datasets/tmp_downloaded_datasets/svhn", logger=None, data_format=None, speckle_noise=0.0):
        super().__init__(name, img_size, logger, data_format=data_format, speckle_noise=speckle_noise)
        self.num_images = num_images
        self.seed = seed
        self.use_split = use_split  # "train", "test" or "both"
        self.download_dir = download_dir
        self.labels = None
        self.logger.debug(
            "Initializing DatasetFromSVHN: name=%s, img_size=%d, num_images=%d, seed=%s, split=%s, format=%s, speckle=%.2f",
            name, img_size, num_images, seed, use_split, self.data_format, self.speckle_noise
        )

    def _ensure_download(self):
        os.makedirs(self.download_dir, exist_ok=True)
        paths = []

        def _download_if_needed(url, fname):
            dst = os.path.join(self.download_dir, fname)
            if not os.path.exists(dst):
                self.logger.info("Downloading %s -> %s", url, dst)
                urlretrieve(url, dst)
            else:
                self.logger.debug("Already exists: %s", dst)
            return dst

        if self.use_split in ("train", "both"):
            paths.append(_download_if_needed(self.SVHN_TRAIN_URL, "train_32x32.mat"))
        if self.use_split in ("test", "both"):
            paths.append(_download_if_needed(self.SVHN_TEST_URL, "test_32x32.mat"))
        return paths

    def _load_mat_images(self, mat_path):
        self.logger.debug("Loading MAT: %s", mat_path)
        mat = loadmat(mat_path)
        X = mat["X"]  # (32, 32, 3, N)
        y = mat["y"].reshape(-1)  # (N,)
        # Label '10' represents digit 0 in SVHN
        y = np.where(y == 10, 0, y)
        # Reorder to (N, H, W, C)
        X = np.transpose(X, (3, 0, 1, 2))
        return X, y

    def _concat_splits(self, paths):
        imgs_list = []
        labels_list = []
        for p in paths:
            X, y = self._load_mat_images(p)
            imgs_list.append(X)
            labels_list.append(y)
        X_all = np.concatenate(imgs_list, axis=0) if len(imgs_list) > 1 else imgs_list[0]
        y_all = np.concatenate(labels_list, axis=0) if len(labels_list) > 1 else labels_list[0]
        return X_all, y_all

    def load_data(self, progress_callback=None):
        """
        Load SVHN, sample deterministically, convert to grayscale 2D and resize.
        Export:
          - self.data: (N, H, W) uint8 in [0,255]  -> UI/visualizer friendly (expects 2D)
          - self.data_f32: (N, H, W) float32 [0,1] -> optional for modeling/DF
        """
        start = perf_counter()

        # Ensure files exist and load all selected splits
        paths = self._ensure_download()
        X_all, y_all = self._concat_splits(paths)  # X_all: (N, 32, 32, 3) RGB uint8
        N = X_all.shape[0]
        self.logger.info("SVHN loaded with %d images", N)

        # Deterministic sampling
        rng = np.random.default_rng(self.seed)
        if self.num_images is None or self.num_images > N:
            self.logger.warning("num_images (%s) > total SVHN (%d). Using N=%d", self.num_images, N, N)
            self.num_images = N
        idx = rng.choice(N, size=self.num_images, replace=False)
        X_sel = X_all[idx]
        y_sel = y_all[idx]

        # Allocate target arrays
        mats_u8 = np.empty((self.num_images, self.img_size, self.img_size), dtype=np.uint8)

        # Convert each RGB image to grayscale 2D uint8 and resize
        for i, img in enumerate(X_sel):
            # RGB uint8 -> grayscale uint8 (2D)
            gray_u8 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            # Resize if needed
            if gray_u8.shape[0] != self.img_size or gray_u8.shape[1] != self.img_size:
                gray_u8 = cv2.resize(gray_u8, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)

            mats_u8[i] = gray_u8

            # Progress callback (every 50 or last)
            if progress_callback and ((i + 1) % 50 == 0 or (i + 1) == self.num_images):
                progress_callback(i + 1, self.num_images)

        # Public dataset used by the UI/visualizer: strictly 2D uint8
        self.data = list(mats_u8)  # Convert to list for consistency with base class
        self.labels = y_sel.astype(int)

        # Optional float copy for analytics/DF (keeps IR-like range)
        self.data_f32 = mats_u8.astype(np.float32) / 255.0

        # Apply data format conversion
        self.apply_data_format()

        # Apply speckle noise if configured
        self.apply_speckle_noise()

        elapsed = perf_counter() - start
        self.logger.info("Dataset SVHN (grayscale 2D) ready: %d images in %.3f s (format: %s, speckle: %.2f)",
                        self.num_images, elapsed, self.data_format, self.speckle_noise)

    def to_dataframe(self):
        """
        Build a DataFrame from grayscale images.
        Uses float32 [0,1] if available (self.data_f32), otherwise converts from uint8.
        """
        if getattr(self, "data", None) is None or len(self.data) == 0:
            return pd.DataFrame()

        # Prefer normalized float data when available
        src = getattr(self, "data_f32", self.data.astype(np.float32) / 255.0)

        num = src.shape[0]
        flat = src.reshape(num, -1)
        df = pd.DataFrame(flat)
        df["label"] = self.labels
        return df
