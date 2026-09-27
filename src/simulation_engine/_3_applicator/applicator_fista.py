from simulation_engine._3_applicator.applicator import ApplicatorABC
import numpy as np
import pandas as pd
import pylops
import pyproximal

# Defaults every entry point shares (Batch Test, Single Test, re-measure).
# They are the values FISTA has always run with: until R1.3a the batch
# runner set attributes the class never read, so these were in effect.
DEFAULT_LAMBDA = 1e-3
DEFAULT_ITERATIONS = 500


class ApplicatorFISTA(ApplicatorABC):
    """
    Applicator class that reconstructs images using FISTA (l2/l1 minimization).

    Solves the inverse problem:
    minimize ||y - S @ x||_2^2 + λ ||x||_1

    Where:
        S: measurement matrix (masks as rows)
        y: measurements (scalar detector values)
        x: image to reconstruct (flattened)
        λ: regularization parameter (self.lam) for l1 sparsity
    """

    # Reconstruction method identifier for reports
    RECONSTRUCTION_METHOD = "FISTA"

    def __init__(self, dataset, mask, maxit=DEFAULT_ITERATIONS, lam=DEFAULT_LAMBDA,
                 **backend_options):
        super().__init__(dataset, mask, **backend_options)
        self.maxit = maxit
        self.lam = lam  # l1 regularization strength
        self.reconstructed_image = None
        self.reconstructed_dataset = None
        self.costf = []  # cost curve for monitoring convergence
        # The legacy callback evaluates the objective every iteration; the torch
        # path does the same by default so both do equal work.
        self.record_cost = True
        self.cost_trace = None

    def apply_mask_range(self, idx_mask_min, idx_mask_max, idx_image):
        """Reconstruct one image from masks ``[idx_mask_min, idx_mask_max)``.

        Runs the torch backend on ``self.device_request`` unless
        ``backend="legacy"``, which keeps the original implementation below.
        """
        if self.backend == "legacy":
            return self._legacy_apply_mask_range(idx_mask_min, idx_mask_max, idx_image)
        result = self._torch_reconstruct(idx_mask_min, idx_mask_max, [idx_image])[0]
        self.reconstructed_image = result
        return result

    def process_dataset(self, idx_mask_min=0, idx_mask_max=None):
        """Reconstruct every image; one flattened reconstruction per DataFrame row."""
        if self.backend == "legacy":
            return self._legacy_process_dataset(idx_mask_min, idx_mask_max)
        return self._torch_process_dataset(idx_mask_min, idx_mask_max)

    def _torch_solve(self, op, Y, key):
        """FISTA, same iteration and parameters (``lam``, ``maxit``) as the legacy path."""
        from simulation_engine._3_applicator.torch_backend.solvers import fista, power_iteration
        ck = ("L", *key)
        if ck not in self._solver_cache:
            self._solver_cache[ck] = power_iteration(op)
        X, cost = fista(op, Y, lam=self.lam, maxit=self.maxit,
                        L=self._solver_cache[ck], record_cost=self.record_cost)
        self.costf.clear()
        if cost.numel():
            self.costf.extend(cost[-1].detach().cpu().tolist())
        self.cost_trace = cost.detach().cpu().numpy()
        self.last_solver_info = {"solver": "fista", "lipschitz": self._solver_cache[ck]}
        return X

    def _legacy_apply_mask_range(self, idx_mask_min, idx_mask_max, idx_image):
        """
        Apply a range of masks to one image, simulate measurements,
        and reconstruct via FISTA l1-regularized least squares.

        Parameters:
            idx_mask_min: Minimum mask index to use.
            idx_mask_max: Maximum mask index (exclusive) to use.
            idx_image: Index of the image in the dataset.

        Returns:
            Reconstructed image.
        """
        # Retrieve the target image and convert to float64 for computation
        # to ensure precision with quantized formats
        image = np.asarray(self.dataset.data[idx_image], dtype=np.float64)
        masks = self.mask.masks[idx_mask_min:idx_mask_max]

        # If no masks, return zero image
        if len(masks) == 0:
            zero_img = np.zeros_like(image, dtype=np.float64)
            self.reconstructed_image = zero_img
            return zero_img

        # Build measurement matrix S and measurement vector y
        # S shape: (M, N) where M = number of masks, N = number of pixels
        # y shape: (M,) - each entry is the detector measurement for one mask
        S = np.array([m.flatten().astype(np.float64) for m in masks], dtype=np.float64)
        y = np.array([(image * m.astype(np.float64)).sum() for m in masks], dtype=np.float64)

        M, N = S.shape

        # Wrap measurement matrix with pylops
        Sop = pylops.MatrixMult(S)

        # Define the data fidelity term: f(x) = 1/2 ||S @ x - y||_2^2
        l2 = pyproximal.proximal.L2(Op=Sop, b=y)

        # Define l1 regularizer: g(x) = λ ||x||_1
        # Scale L1 by multiplying with regularization parameter
        l1 = self.lam * pyproximal.proximal.L1()

        # Compute step size tau = 0.95 / L where L is the Lipschitz constant
        # L ≈ largest eigenvalue of S^T @ S
        L_val = np.abs((Sop.H * Sop).eigs(1)[0])
        tau = 0.95 / L_val

        # Clear cost history
        self.costf.clear()

        def _callback(x):
            """Record cost function for monitoring convergence."""
            res = Sop * x - y
            fx = 0.5 * np.dot(res, res)
            gx = self.lam * np.linalg.norm(x, 1)
            self.costf.append(fx + gx)

        # Initial guess: zero vector
        x0 = np.zeros(N, dtype=np.float64)

        # Run FISTA (Fast Iterative Shrinkage-Thresholding Algorithm)
        opt = pyproximal.optimization.primal.ProximalGradient(
            l2, l1,
            tau=tau,
            x0=x0,
            niter=self.maxit,
            acceleration='fista',
            show=False,
            callback=_callback
        )

        # Extract solution from optimizer
        if hasattr(opt, 'run'):
            x_fista = opt.run()
        elif hasattr(opt, 'solve'):
            opt.solve()
            x_fista = opt.x
        else:
            x_fista = opt

        # Reshape back to image
        reconstructed = x_fista.reshape(image.shape)
        self.reconstructed_image = reconstructed
        return reconstructed

    def process_image(self, idx):
        """Process one image using all masks."""
        return self.apply_mask_range(0, len(self.mask.masks), idx)

    def _legacy_process_dataset(self, idx_mask_min=0, idx_mask_max=None):
        """
        Process entire dataset: each row in the returned DataFrame
        is the flattened reconstructed image.
        """
        if idx_mask_max is None:
            idx_mask_max = len(self.mask.masks)

        recs = [
            self.apply_mask_range(idx_mask_min, idx_mask_max, i).flatten()
            for i in range(len(self.dataset.data))
        ]
        df = pd.DataFrame(recs)
        self.reconstructed_dataset = df
        return df
