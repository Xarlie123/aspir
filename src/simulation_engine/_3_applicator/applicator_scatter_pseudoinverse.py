from simulation_engine._3_applicator.applicator import ApplicatorABC
import numpy as np
import pandas as pd


class ApplicatorScatterPseudoinverse(ApplicatorABC):
    """
    Applicator class that reconstructs images using the pseudoinverse method.

    Solves the linear least squares problem:
    minimize ||y - S @ x||_2^2

    Where:
        S: measurement matrix (masks as rows)
        y: measurements (scalar detector values)
        x: image to reconstruct (flattened)

    Solution: x = S^+ @ y (using Moore-Penrose pseudoinverse)
    """

    # Reconstruction method identifier for reports
    RECONSTRUCTION_METHOD = "Pseudoinverse"

    def __init__(self, dataset, mask):
        """
        Initialize the applicator with a dataset and masks.

        Parameters:
            dataset: Object containing the images (has attribute 'data').
            mask: Object containing the masks (has attribute 'mascaras').
        """
        super().__init__(dataset, mask)
        self.dataset = dataset
        self.mask = mask
        self.reconstructed_image = None
        self.reconstructed_dataset = None

    def apply_mask_range(self, idx_mask_min, idx_mask_max, idx_image):
        """
        Apply a range of masks to a given image, simulate measurements,
        and reconstruct using the pseudoinverse method.

        Solves: minimize ||y - S @ x||_2^2
        Solution: x = S^+ @ y (Moore-Penrose pseudoinverse)

        Parameters:
            idx_mask_min: Minimum mask index to use.
            idx_mask_max: Maximum mask index (exclusive) to use.
            idx_image: Index of the image in the dataset.

        Returns:
            Reconstructed image as a numpy array.
        """
        # Retrieve the image from the dataset and convert to float64 for computation
        # to ensure precision with quantized formats
        image = np.asarray(self.dataset.data[idx_image], dtype=np.float64)

        # Extract the selected masks
        masks = self.mask.mascaras[idx_mask_min:idx_mask_max]

        # If no masks selected, return zero image
        if len(masks) == 0:
            reconstructed = np.zeros_like(image, dtype=np.float64)
            self.reconstructed_image = reconstructed
            return reconstructed

        # Build measurement matrix S and measurement vector y
        # S shape: (M, N) where M = number of masks, N = number of pixels
        # y shape: (M,) - detector measurements for each mask
        S = np.array([m.flatten().astype(np.float64) for m in masks], dtype=np.float64)
        y = np.array([(image * m.astype(np.float64)).sum() for m in masks], dtype=np.float64)

        # Ensure S is 2D
        if S.ndim != 2:
            reconstructed = np.zeros_like(image, dtype=np.float64)
            self.reconstructed_image = reconstructed
            return reconstructed

        # Compute Moore-Penrose pseudoinverse and reconstruct
        # x = S^+ @ y
        Spinv = np.linalg.pinv(S)
        img_vec = Spinv @ y
        reconstructed = img_vec.reshape(image.shape)

        self.reconstructed_image = reconstructed
        return reconstructed

    def process_image(self, idx):
        """
        Process a single image from the dataset using all available masks.

        Parameters:
            idx: Index of the image in the dataset.

        Returns:
            Reconstructed image.
        """
        return self.apply_mask_range(0, len(self.mask.mascaras), idx)

    def process_dataset(self, idx_mask_min=0, idx_mask_max=None):
        """
        Process all images in the dataset using a given range of masks.
        Each reconstructed image is flattened and stored as a row in a DataFrame.

        Parameters:
            idx_mask_min: Minimum mask index to use (default is 0).
            idx_mask_max: Maximum mask index (exclusive). If None, uses all masks.

        Returns:
            pandas.DataFrame: Each row is a flattened reconstructed image.
        """
        if idx_mask_max is None:
            idx_mask_max = len(self.mask.mascaras)

        reconstructed_images = []
        for idx in range(len(self.dataset.data)):
            rec = self.apply_mask_range(idx_mask_min, idx_mask_max, idx)
            reconstructed_images.append(rec.flatten())

        df = pd.DataFrame(reconstructed_images)
        self.reconstructed_dataset = df
        return df
