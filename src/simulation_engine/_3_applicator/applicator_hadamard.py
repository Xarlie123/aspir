from simulation_engine._3_applicator.applicator import ApplicatorABC
import numpy as np
import pandas as pd


class ApplicatorHadamard(ApplicatorABC):
    """
    Hadamard applicator using true linear Hadamard reconstruction.

    Uses the Hadamard inverse transform for reconstruction:
    x̂ = (1/N) * Σ yᵢ * Hᵢ = H^T · y / N

    Where:
        yᵢ = <x, Hᵢ> = measurement value (inner product with bipolar pattern)
        Hᵢ = Hadamard pattern with values {+1, -1}
        N = number of measurements (normalization factor)

    This exploits the orthogonality property of Hadamard matrices:
    H · H^T = N · I, therefore H^{-1} = H^T / N

    Note: Masks must contain bipolar values {+1, -1}, not binary {0, 1}.
    """

    # Reconstruction method identifier for reports
    RECONSTRUCTION_METHOD = "Hadamard Linear"

    def __init__(self, dataset, mask):
        """
        Initialize Hadamard applicator.

        Parameters:
            dataset: Dataset object providing 2D grayscale images (H, W).
            mask: Mask object with `mascaras` attribute containing Hadamard patterns
                  with bipolar values {+1, -1}.
        """
        super().__init__(dataset, mask)
        self.dataset = dataset
        self.mask = mask
        self.reconstructed_image = None
        self.reconstructed_dataset = None

    def _ensure_masks_available(self):
        """
        Ensure self.mask.mascaras exists and is non-empty.
        Try to call a generator method if available; otherwise raise a clear error.
        """
        masks = getattr(self.mask, "mascaras", None)
        if masks is None or len(masks) == 0:
            # Try common generator method names if they exist (best-effort).
            for name in ("generar_mascaras", "generate_masks", "generar", "generate", "build"):
                if hasattr(self.mask, name):
                    fn = getattr(self.mask, name)
                    try:
                        fn()  # Try without args; many implementations fill self.mask.mascaras
                    except TypeError:
                        # If the generator needs a size/dim, infer from dataset images
                        if len(getattr(self.dataset, "data", [])) == 0:
                            raise ValueError(
                                "No masks and cannot infer size from dataset. "
                                "Load dataset before generating masks."
                            )
                        size = int(self.dataset.data[0].shape[0])
                        try:
                            fn(size)  # attempt with size
                        except Exception:
                            pass
                    break  # we tried one generator; re-check below

            masks = getattr(self.mask, "mascaras", None)
            if masks is None or len(masks) == 0:
                raise ValueError(
                    "The mask has no available patterns in 'mascaras'. "
                    "Make sure to generate them before using the applicator."
                )
        return masks

    def apply_mask_range(self, idx_mask_min, idx_mask_max, idx_image):
        """
        Apply a range of Hadamard masks to an image and reconstruct using
        true linear Hadamard reconstruction (inverse Hadamard transform).

        Uses the Hadamard inverse formula exploiting orthogonality:
        x̂ = (1/N) * Σ yᵢ * Hᵢ = H^T · y / N

        Where:
            yᵢ = <x, Hᵢ> = inner product of image with bipolar pattern {+1, -1}
            Hᵢ = Hadamard pattern (not binarized, keeps {+1, -1} values)
            N = number of measurements

        Parameters:
            idx_mask_min: Minimum mask index to apply.
            idx_mask_max: Maximum (exclusive) mask index to apply.
            idx_image: Image index in dataset.

        Returns:
            Reconstructed image using linear Hadamard reconstruction.
        """
        # Convert image to float64 for computation to ensure precision
        image = np.asarray(self.dataset.data[idx_image], dtype=np.float64)
        accumulated_image = np.zeros(image.shape, dtype=np.float64)

        masks = self._ensure_masks_available()
        idx_max = min(idx_mask_max, len(masks))

        # Step 1: Calculate measurements and accumulate reconstruction
        # Using bipolar Hadamard patterns {+1, -1} directly (NOT binarized)
        measurements = []
        for i in range(idx_mask_min, idx_max):
            # Get Hadamard pattern as float64, preserving {+1, -1} values
            pattern = np.asarray(masks[i], dtype=np.float64)

            # Measurement: inner product y_i = <x, H_i> = sum(x * H_i)
            measurement = np.sum(image * pattern)
            measurements.append(measurement)

            # Accumulate: x̂ += y_i * H_i
            accumulated_image += measurement * pattern

        N = len(measurements)
        if N == 0:
            self.reconstructed_image = accumulated_image
            return accumulated_image

        # Step 2: Normalize by number of measurements
        # This implements H^{-1} = H^T / N due to orthogonality
        result = accumulated_image / N

        # Store the reconstructed image
        self.reconstructed_image = result
        return result

    def process_image(self, idx):
        """
        Process a single image using the full mask set.

        Parameters:
            idx: Index of the image in the dataset.

        Returns:
            Reconstructed image.
        """
        masks = self._ensure_masks_available()
        image_rec = self.apply_mask_range(0, len(masks), idx)
        self.reconstructed_image = image_rec
        return image_rec

    def process_dataset(self, idx_mask_min=0, idx_mask_max=None):
        """
        Process the whole dataset and return a DataFrame of flattened reconstructions.

        Parameters:
            idx_mask_min: Minimum mask index to use (default is 0).
            idx_mask_max: Maximum mask index (exclusive). If None, uses all masks.

        Returns:
            pandas.DataFrame: Each row is a flattened reconstructed image.
        """
        masks = self._ensure_masks_available()
        if idx_mask_max is None:
            idx_mask_max = len(masks)

        accumulated_images = []
        for idx in range(len(self.dataset.data)):
            image_rec = self.apply_mask_range(idx_mask_min, idx_mask_max, idx)
            accumulated_images.append(image_rec.flatten())

        df = pd.DataFrame(accumulated_images)
        self.reconstructed_dataset = df
        return df
