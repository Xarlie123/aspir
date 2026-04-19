from simulation_engine._3_applicator.applicator import ApplicatorABC
import numpy as np
import pandas as pd

class ApplicatorScatter(ApplicatorABC):
    """Applicator for scatter masks using ghost imaging reconstruction formula."""

    # Reconstruction method identifier for reports
    RECONSTRUCTION_METHOD = "Ghost Imaging"

    def __init__(self, dataset, mask):
        """
        Initialize the applicator with a dataset and masks.

        Parameters:
            dataset: Dataset object
            mask: Mask object
        """
        super().__init__(dataset, mask)
        self.dataset = dataset
        self.mask = mask
        self.reconstructed_image = None  # Stores the last accumulated image
        self.reconstructed_dataset = None  # Stores the complete dataset reconstruction

    def apply_mask_range(self, idx_mask_min, idx_mask_max, idx_image):
        """
        Apply a range of masks to an image using the ghost imaging formula
        and return the accumulated result.

        Parameters:
            idx_mask_min: Minimum mask index to use.
            idx_mask_max: Maximum mask index (exclusive) to use.
            idx_image: Index of the image in the dataset.

        Returns:
            Reconstructed image.
        """
        # Convert image to float64 for computation to ensure precision with quantized formats
        image = np.asarray(self.dataset.data[idx_image], dtype=np.float64)
        accumulated_image = np.zeros(image.shape, dtype=np.float64)

        # Calculate detector measurements for each mask
        measurements = []
        for i in range(idx_mask_min, min(idx_mask_max, len(self.mask.masks))):
            mask = self.mask.masks[i].astype(np.float64)
            masked_image = image * mask
            measurement = masked_image.sum()
            measurements.append(measurement)

        N = len(measurements)
        if N == 0:
            self.reconstructed_image = accumulated_image
            return accumulated_image

        # Calculate average measurement (DC offset removal)
        measurement_avg = np.mean(measurements)

        # Accumulate weighted masks by measurement difference
        for i, measurement in enumerate(measurements, start=idx_mask_min):
            mask = self.mask.masks[i]
            accumulated_image += (measurement - measurement_avg) * mask

        # Normalize by number of measurements
        result = accumulated_image / N

        # Store and return result
        self.reconstructed_image = result
        return result

    def process_image(self, idx):
        """
        Process a single image from the dataset by applying all masks
        and return the reconstruction.

        Parameters:
            idx: Index of the image in the dataset.

        Returns:
            Reconstructed image.
        """
        image_rec = self.apply_mask_range(0, len(self.mask.masks), idx)
        self.reconstructed_image = image_rec
        return image_rec

    def process_dataset(self, idx_mask_min=0, idx_mask_max=None):
        """
        Process all images in the dataset by applying a range of masks.
        Each reconstructed image is flattened and stored as a row in a DataFrame.

        Parameters:
            idx_mask_min: Minimum mask index to use (default is 0).
            idx_mask_max: Maximum mask index (exclusive). If None, uses all masks.

        Returns:
            DataFrame where each row is a flattened reconstructed image.
        """
        if idx_mask_max is None:
            idx_mask_max = len(self.mask.masks)

        accumulated_images = []
        for idx in range(len(self.dataset.data)):
            image_rec = self.apply_mask_range(idx_mask_min, idx_mask_max, idx)
            accumulated_images.append(image_rec.flatten())
        df_accumulated = pd.DataFrame(accumulated_images)
        self.reconstructed_dataset = df_accumulated
        return df_accumulated
