from simulation_engine._3_applicator.applicator import ApplicatorABC
import numpy as np
import pandas as pd

class ApplicatorSweep(ApplicatorABC):
    """Applicator class that applies sweep masks and accumulates values using ghost imaging reconstruction."""

    # Reconstruction method identifier for reports
    RECONSTRUCTION_METHOD = "Sweep Linear"

    def __init__(self, dataset, mask):
        """
        Initialize the applicator with a dataset and set of masks.

        Parameters:
            dataset: Dataset object providing 2D images.
            mask: Mask object containing sweep patterns.
        """
        super().__init__(dataset, mask)
        self.dataset = dataset
        self.mask = mask
        self.reconstructed_image = None
        self.reconstructed_dataset = None

    def apply_mask_range(self, idx_mask_min, idx_mask_max, idx_image):
        """
        Apply a range of sweep masks to an image and reconstruct using
        ghost imaging correlation formula to eliminate artifacts.

        Uses the standard computational imaging reconstruction:
        x̂ = (1/N) * Σ(B_i - B_avg) * M_i

        Where:
            B_i = measurement value for mask i (sum of pixels under mask)
            B_avg = average of all measurements (removes DC offset)
            M_i = mask i (spatial correlation)
            N = number of measurements (normalization factor)

        Parameters:
            idx_mask_min: Minimum mask index to apply.
            idx_mask_max: Maximum (exclusive) mask index to apply.
            idx_image: Image index in dataset.

        Returns:
            Reconstructed image with artifacts eliminated.
        """
        # Convert image to float64 for computation to ensure precision with quantized formats
        image = np.asarray(self.dataset.data[idx_image], dtype=np.float64)
        accumulated_image = np.zeros(image.shape, dtype=np.float64)

        # Step 1: Calculate measurement values for each mask
        measurements = []
        for i in range(idx_mask_min, min(idx_mask_max, len(self.mask.masks))):
            mask = self.mask.masks[i].astype(np.float64)
            masked_image = image * mask
            sum_masked_pixels = masked_image.sum()
            measurements.append(sum_masked_pixels)

        N = len(measurements)
        if N == 0:
            self.reconstructed_image = accumulated_image
            return accumulated_image

        # Step 2: Calculate average measurement value (removes DC offset)
        measurement_avg = np.mean(measurements)

        # Step 3: Accumulate correlation between (measurement - average) and mask
        # This implements the standard ghost imaging reconstruction formula
        for i, measurement in enumerate(measurements, start=idx_mask_min):
            mask = self.mask.masks[i].astype(np.float64)
            accumulated_image += (measurement - measurement_avg) * mask

        # Step 4: Normalize by number of measurements
        # This ensures output is in reasonable value range and eliminates artifacts
        result = accumulated_image / N

        # Store the reconstructed image
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
            pandas.DataFrame: Each row is a flattened reconstructed image.
        """
        if idx_mask_max is None:
            idx_mask_max = len(self.mask.masks)

        accumulated_images = []
        for idx in range(len(self.dataset.data)):
            accumulated_image = self.apply_mask_range(idx_mask_min, idx_mask_max, idx)
            accumulated_images.append(accumulated_image.flatten())
        df = pd.DataFrame(accumulated_images)
        self.reconstructed_dataset = df
        return df
