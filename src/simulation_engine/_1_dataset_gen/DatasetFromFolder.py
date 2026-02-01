
# File: simulation_engine/_1_dataset_gen/DatasetFromFolder.py

import os
import cv2
import pandas as pd
from simulation_engine._1_dataset_gen.dataset import DatasetABC

class DatasetFromFolder(DatasetABC):
    """Creates a dataset from images in a folder."""

    def __init__(self, img_size, folder_path, logger=None, data_format=None, speckle_noise=0.0):
        dataset_name = os.path.basename(folder_path)
        super().__init__(dataset_name, img_size, logger, data_format=data_format, speckle_noise=speckle_noise)
        self.folder_path = folder_path
        self.logger.debug("DatasetFromFolder for path %s, format %s, speckle %.2f",
                        folder_path, self.data_format, self.speckle_noise)

    def load_data(self, progress_callback=None):
        self.logger.info("Loading images from folder: %s", self.folder_path)
        images = []
        files = sorted([f for f in os.listdir(self.folder_path)
                        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))])
        total = len(files)
        if total == 0:
            self.logger.error("No images found in %s", self.folder_path)
            raise ValueError("No images found in the folder.")
        for i, fname in enumerate(files):
            path = os.path.join(self.folder_path, fname)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
            if progress_callback:
                progress_callback(i+1, total)
        self.num_images = len(images)
        if not images:
            self.logger.error("Could not load any valid images")
            raise ValueError("No images found in the folder.")
        if not self.check_image_sizes(images):
            self.logger.error("Inconsistent image sizes in folder")
            raise ValueError("Images in the folder have different sizes.")
        self.data = self.resize_images(images)
        self.df = self.to_dataframe()
        # Apply data format conversion
        self.apply_data_format()
        # Apply speckle noise if configured
        self.apply_speckle_noise()
        self.logger.info("Loaded %d images (format: %s, speckle: %.2f)",
                        self.num_images, self.data_format, self.speckle_noise)

    def to_dataframe(self):
        self.logger.debug("Converting %d images to DataFrame", len(self.data))
        data_flat = [img.flatten() for img in self.data]
        return pd.DataFrame(data_flat)
