import logging
import numpy as np
import torch

import lpips  # [cite: 1]
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# Data format constants (same as postprocessor)
DATA_FORMAT_FP32 = "FP32"
DATA_FORMAT_INT8 = "INT8"
DATA_FORMAT_INT4 = "INT4"


class NoiseAnalyzer:
    """
    Computes per-image and average PSNR/SSIM/LPIPS for noisy vs reconstructions.
    Supports different data formats for embedded system testing.
    """

    def __init__(self, originals, noisy, reconstructions, data_format=None):
        # Set up a logger for noise analysis
        self.logger = logging.getLogger("ASPIR.NoiseAnalyzer")
        self.logger.setLevel(logging.DEBUG)
        self.logger.debug(
            "Initializing NoiseAnalyzer: n_originals=%d, n_noisy=%d, n_recon=%d",
            len(originals), len(noisy), len(reconstructions)
        )

        self.originals = originals
        self.noisy = noisy
        self.reconstructions = reconstructions

        # Store data format for metrics reporting
        self.data_format = data_format if data_format is not None else DATA_FORMAT_FP32
        self.logger.info("Data format for analysis: %s", self.data_format)

        # Initialize LPIPS model
        # 'alex' is generally faster and closer to human perception for this task
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_fn_lpips = lpips.LPIPS(net='alex').to(self.device)
        self.loss_fn_lpips.eval()

    def _compute_metrics(self, reference, target):
        """
        Compute PSNR and SSIM between reference and target.
        """
        data_range = reference.max() - reference.min()
        # Fallback if constant image (data_range=0)
        if data_range == 0:
            data_range = 1.0

        psnr = peak_signal_noise_ratio(reference, target, data_range=data_range)
        ssim = structural_similarity(
            reference, target,
            multichannel=(reference.ndim == 3),
            data_range=data_range
        )
        return psnr, ssim

    def _compute_lpips_batch(self, refs, preds, batch_size=32):
        """
        Computes LPIPS in batches to avoid OOM.
        Assumes inputs are lists/arrays of shape (H, W) in range [0, 1].
        """
        vals = []
        n_samples = len(refs)

        # Convert entire list to tensor first (cpu)
        t_refs_all = torch.from_numpy(np.array(refs)).float()
        t_preds_all = torch.from_numpy(np.array(preds)).float()

        # Process in batches
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                # Slice batch
                batch_ref = t_refs_all[i: i + batch_size].to(self.device)
                batch_pred = t_preds_all[i: i + batch_size].to(self.device)

                # Preprocessing for LPIPS:
                # 1. Add channel dim if grayscale: (B, H, W) -> (B, 1, H, W)
                if batch_ref.ndim == 3:
                    batch_ref = batch_ref.unsqueeze(1)
                    batch_pred = batch_pred.unsqueeze(1)

                # 2. LPIPS expects 3 channels (RGB). Repeat channel.
                if batch_ref.shape[1] == 1:
                    batch_ref = batch_ref.repeat(1, 3, 1, 1)
                    batch_pred = batch_pred.repeat(1, 3, 1, 1)

                # 3. Scale from [0, 1] to [-1, 1]
                batch_ref = batch_ref * 2.0 - 1.0
                batch_pred = batch_pred * 2.0 - 1.0

                # Compute distance
                d = self.loss_fn_lpips(batch_ref, batch_pred)
                vals.extend(d.view(-1).cpu().numpy().tolist())

        return vals

    def analyze(self):
        """
        Compute PSNR/SSIM/LPIPS for every image in `noisy` and `reconstructions`
        against `originals`. Returns six lists and stores their means.
        """
        self.logger.info("Starting detailed PSNR/SSIM/LPIPS computation")
        n = len(self.originals)

        # 1. Compute PSNR/SSIM loop (CPU)
        self.psnr_noise = []
        self.ssim_noise = []
        self.psnr_rec = []
        self.ssim_rec = []

        for i in range(n):
            orig = self.originals[i]
            noisy = self.noisy[i]
            rec = self.reconstructions[i]

            pn, sn = self._compute_metrics(orig, noisy)
            pr, sr = self._compute_metrics(orig, rec)

            self.psnr_noise.append(pn)
            self.ssim_noise.append(sn)
            self.psnr_rec.append(pr)
            self.ssim_rec.append(sr)

            if i % 50 == 0:  # Reduce log spam
                self.logger.debug("Metrics frame %d calculated", i)

        # 2. Compute LPIPS (Batch / GPU)
        self.logger.info("Calculating LPIPS (Noisy)...")
        self.lpips_noise = self._compute_lpips_batch(self.originals, self.noisy)

        self.logger.info("Calculating LPIPS (Reconstructed)...")
        self.lpips_rec = self._compute_lpips_batch(self.originals, self.reconstructions)

        # 3. Compute means
        self.psnr_noise_mean = float(np.mean(self.psnr_noise))
        self.ssim_noise_mean = float(np.mean(self.ssim_noise))
        self.lpips_noise_mean = float(np.mean(self.lpips_noise))

        self.psnr_rec_mean = float(np.mean(self.psnr_rec))
        self.ssim_rec_mean = float(np.mean(self.ssim_rec))
        self.lpips_rec_mean = float(np.mean(self.lpips_rec))

        self.logger.info(
            "Completed Metrics (Format: %s):\n"
            "  Noise -> PSNR: %.2f, SSIM: %.4f, LPIPS: %.4f\n"
            "  Recon -> PSNR: %.2f, SSIM: %.4f, LPIPS: %.4f",
            self.data_format,
            self.psnr_noise_mean, self.ssim_noise_mean, self.lpips_noise_mean,
            self.psnr_rec_mean, self.ssim_rec_mean, self.lpips_rec_mean
        )

        return (self.psnr_noise, self.ssim_noise, self.lpips_noise,
                self.psnr_rec, self.ssim_rec, self.lpips_rec)

    def get_metrics_summary(self) -> dict:
        """
        Returns a dictionary with all computed metrics and data format info.
        Useful for exporting results for comparison across different formats.
        """
        return {
            "data_format": self.data_format,
            "noise_metrics": {
                "psnr_mean": getattr(self, 'psnr_noise_mean', None),
                "ssim_mean": getattr(self, 'ssim_noise_mean', None),
                "lpips_mean": getattr(self, 'lpips_noise_mean', None),
                "psnr_values": getattr(self, 'psnr_noise', []),
                "ssim_values": getattr(self, 'ssim_noise', []),
                "lpips_values": getattr(self, 'lpips_noise', []),
            },
            "reconstruction_metrics": {
                "psnr_mean": getattr(self, 'psnr_rec_mean', None),
                "ssim_mean": getattr(self, 'ssim_rec_mean', None),
                "lpips_mean": getattr(self, 'lpips_rec_mean', None),
                "psnr_values": getattr(self, 'psnr_rec', []),
                "ssim_values": getattr(self, 'ssim_rec', []),
                "lpips_values": getattr(self, 'lpips_rec', []),
            }
        }