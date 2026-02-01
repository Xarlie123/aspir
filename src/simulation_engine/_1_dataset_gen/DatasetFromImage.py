# File: simulation_engine/_1_dataset_gen/DatasetFromImage.py

import os
import cv2
import pandas as pd
from simulation_engine._1_dataset_gen.dataset import DatasetABC

class DatasetFromImage(DatasetABC):
    """Creates a dataset from a single image."""

    def __init__(self, img_size, img_path, logger=None, data_format=None, speckle_noise=0.0):
        name = os.path.splitext(os.path.basename(img_path))[0]
        super().__init__(name, img_size, logger, data_format=data_format, speckle_noise=speckle_noise)
        self.img_path = img_path
        self.logger.debug("DatasetFromImage for file %s, format %s, speckle %.2f",
                        img_path, self.data_format, self.speckle_noise)

    def load_data(self, progress_callback=None):
        self.logger.info("Loading single image: %s", self.img_path)
        img = cv2.imread(self.img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            self.logger.error("Could not load image %s", self.img_path)
            raise ValueError("Unable to load the image")
        img_resized = cv2.resize(img, (self.img_size, self.img_size))
        self.data = [img_resized]
        self.df = self.to_dataframe()
        self.num_images = 1
        # Apply data format conversion
        self.apply_data_format()
        # Apply speckle noise if configured
        self.apply_speckle_noise()
        if progress_callback:
            progress_callback(1, 1)
        self.logger.info("Image loaded and resized, dataset ready (format: %s, speckle: %.2f)",
                        self.data_format, self.speckle_noise)

    def to_dataframe(self):
        self.logger.debug("Converting image to DataFrame")
        data_flat = [img.flatten() for img in self.data]
        return pd.DataFrame(data_flat)
