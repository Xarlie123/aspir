from abc import ABC, abstractmethod
import pandas as pd

class ApplicatorABC(ABC):
    """Abstract base class for applying masks to images and datasets."""

    def __init__(self, dataset, mask):
        self.name = dataset.name
        self.img_size = dataset.img_size
        self.dataset = dataset  # Dataset object
        self.mask = mask  # Mask object
        self.data = pd.DataFrame()  # DataFrame to store processed images
        self.dataset_type = self.__class__.__name__
        self.num_images = len(dataset.data)

    @abstractmethod
    def apply_mask_range(self, idx_mask_min, idx_mask_max, idx_image):
        """Abstract method to apply a range of masks to an image and accumulate values."""
        pass

    @abstractmethod
    def process_image(self, idx):
        """Abstract method to process a single image by applying masks."""
        pass

    @abstractmethod
    def process_dataset(self, idx_mask_min=0, idx_mask_max=None):
        """Abstract method to process entire dataset by applying masks."""
        pass