from simulation_engine._3_applicator.applicator import ApplicatorABC
import numpy as np
import pandas as pd
import pylops
import pyproximal


class ApplicatorScatterTV(ApplicatorABC):
    """
    Applicator class that reconstructs images using TV-norm regularized inverse problem.

    Solves the inverse problem:
    minimize ||y - S @ x||_2^2 + λ TV(x)

    Where:
        S: measurement matrix (masks as rows)
        y: measurements (scalar detector values)
        x: image to reconstruct
        λ: regularization parameter for total variation (self.lam)

    Uses primal-dual optimization to handle the non-smooth TV term.
    """

    # Reconstruction method identifier for reports
    RECONSTRUCTION_METHOD = "TV Norm"

    def __init__(self, dataset, mask, niter=100, lam=1e-1):
        """
        Initializes the applicator with a dataset, masks, and TV parameters.

        Parameters:
            dataset: Object containing the images.
            mask: Object containing the masks.
            niter: Number of iterations for the TV solver (default is 100).
            lam: Regularization strength for TV (default is 0.1).
        """
        super().__init__(dataset, mask)
        self.dataset = dataset
        self.mask = mask
        self.niter = niter
        self.lam = lam  # TV regularization strength
        self.reconstructed_image = None
        self.reconstructed_dataset = None

    def apply_mask_range(self, idx_mask_min, idx_mask_max, idx_image):
        """
        Apply a range of masks to a given image, simulate measurements,
        and reconstruct using TV-regularized inverse problem.

        Solves: minimize ||y - S @ x||_2^2 + λ TV(x)

        Parameters:
            idx_mask_min: Minimum mask index to use.
            idx_mask_max: Maximum mask index (exclusive) to use.
            idx_image: Index of the image in the dataset.

        Returns:
            Reconstructed image after TV regularization.
        """
        # Retrieve the image from the dataset and convert to float64 for computation
        # to ensure precision with quantized formats
        image = np.asarray(self.dataset.data[idx_image], dtype=np.float64)

        # Select masks and compute measurements
        masks = self.mask.mascaras[idx_mask_min:idx_mask_max]
        N = len(masks)

        # If no masks, return zero image
        if N == 0:
            zero_img = np.zeros_like(image, dtype=np.float64)
            self.reconstructed_image = zero_img
            return zero_img

        # Build measurement matrix S and measurement vector y
        # S shape: (M, N) where M = number of masks, N = number of pixels
        # y shape: (M,) - detector measurements
        S = np.array([m.flatten().astype(np.float64) for m in masks], dtype=np.float64)
        y = np.array([(image * m.astype(np.float64)).sum() for m in masks], dtype=np.float64)

        M, N_pix = S.shape

        # Wrap measurement matrix with pylops
        Sop = pylops.MatrixMult(S)

        # Build gradient operator for TV (2D spatial derivatives)
        Gop = pylops.Gradient(dims=image.shape, sampling=1.0, edge=False,
                              kind='forward', dtype='float64')

        # Define l2 data fidelity: f(x) = 1/2 ||S @ x - y||_2^2
        l2 = pyproximal.proximal.L2(Op=Sop, b=y)

        # Define l21 regularizer for isotropic TV: g(x) = λ TV(x)
        # Scale L21 by multiplying with regularization parameter
        l21 = self.lam * pyproximal.proximal.L21(ndim=2)

        # Estimate Lipschitz constant of Gop (gradient operator)
        L_tv = 8.0  # Conservative estimate for 2D gradient operator

        # Primal-dual step sizes
        tau_tv = 1.0 / np.sqrt(L_tv)
        mu_tv = 1.0 / (tau_tv * L_tv)

        # Run primal-dual algorithm
        # minimize f(x) + g(G @ x) = ||y - S @ x||_2^2 + λ TV(x)
        pd_solver = pyproximal.optimization.primaldual.PrimalDual(
            l2, l21, Gop,
            tau=tau_tv,
            mu=mu_tv,
            theta=1.0,
            x0=np.zeros(N_pix, dtype=np.float64),
            niter=self.niter,
            show=False
        )

        # Extract solution
        if hasattr(pd_solver, 'x'):
            x_tv = pd_solver.x
        else:
            x_tv = pd_solver

        # Reshape to image
        reconstructed = x_tv.reshape(image.shape)
        self.reconstructed_image = reconstructed
        return reconstructed

    def process_image(self, idx):
        """
        Process a single image using all available masks with TV regularization.

        Parameters:
            idx: Index of the image in the dataset.

        Returns:
            Reconstructed image.
        """
        return self.apply_mask_range(0, len(self.mask.mascaras), idx)

    def process_dataset(self, idx_mask_min=0, idx_mask_max=None):
        """
        Process all images in the dataset with TV regularization.
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
