# File: ui/custom_widgets/dataset_control/generate_dataset_internet/download_worker.py
"""
Worker for downloading datasets from the internet in a background thread.
"""

import os
import sys
import logging
import shutil
import subprocess
import stat
from PyQt5.QtCore import QObject, pyqtSignal
from urllib.request import urlretrieve


class DownloadWorker(QObject):
    """Worker to download datasets in a separate thread."""
    progress = pyqtSignal(int)      # Progress percentage (0-100)
    finished = pyqtSignal()         # Download completed successfully
    error = pyqtSignal(Exception)   # Error occurred
    status = pyqtSignal(str)        # Status message

    # Dataset configurations
    SVHN_URL = "http://ufldl.stanford.edu/housenumbers/train_32x32.mat"
    SVHN_DIR = "../datasets/tmp_downloaded_datasets/svhn"
    SVHN_FILE = "train_32x32.mat"
    SVHN_NATIVE_SIZE = 32

    CELEBRITIES_KAGGLE = "arnrob/celeba-small-images-dataset"
    CELEBRITIES_DIR = "../datasets/tmp_downloaded_datasets/celebrities/kaggle_celeba_small"
    CELEBRITIES_NATIVE_SIZE = 64

    def __init__(self, dataset_type: str, kaggle_username: str = None,
                 kaggle_key: str = None, logger=None):
        """
        Initialize download worker.

        Args:
            dataset_type: "svhn" or "celebrities"
            kaggle_username: Kaggle username (for celebrities)
            kaggle_key: Kaggle API key (for celebrities)
            logger: Optional logger
        """
        super().__init__()
        self.dataset_type = dataset_type.lower()
        self.kaggle_username = kaggle_username
        self.kaggle_key = kaggle_key

        if logger is None:
            self.logger = logging.getLogger(self.__class__.__name__)
        else:
            self.logger = logger.getChild(self.__class__.__name__)

    def run(self):
        """Execute the download."""
        try:
            if self.dataset_type == "svhn":
                self._download_svhn()
            elif self.dataset_type == "celebrities":
                self._download_celebrities()
            else:
                raise ValueError(f"Unknown dataset type: {self.dataset_type}")
            self.finished.emit()
        except Exception as e:
            self.logger.error("Download error: %s", e, exc_info=True)
            self.error.emit(e)

    def _download_svhn(self):
        """Download SVHN dataset."""
        os.makedirs(self.SVHN_DIR, exist_ok=True)
        dst = os.path.join(self.SVHN_DIR, self.SVHN_FILE)

        if os.path.exists(dst):
            self.logger.info("SVHN already exists: %s", dst)
            self.status.emit("SVHN already downloaded")
            self.progress.emit(100)
            return

        self.status.emit("Downloading SVHN (train_32x32.mat)...")
        self.logger.info("Downloading SVHN from %s", self.SVHN_URL)

        def progress_hook(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, int(block_num * block_size * 100 / total_size))
                self.progress.emit(percent)

        urlretrieve(self.SVHN_URL, dst, reporthook=progress_hook)
        self.status.emit("SVHN download complete")
        self.logger.info("SVHN downloaded to: %s", dst)

    def _download_celebrities(self):
        """Download Celebrities dataset from Kaggle."""
        root = self.CELEBRITIES_DIR
        os.makedirs(root, exist_ok=True)

        # Check if already downloaded
        if self._count_images(root) > 0:
            self.logger.info("Celebrities already downloaded in: %s", root)
            self.status.emit("Celebrities already downloaded")
            self.progress.emit(100)
            return

        # Set Kaggle credentials if provided
        if self.kaggle_username and self.kaggle_key:
            os.environ['KAGGLE_USERNAME'] = self.kaggle_username
            os.environ['KAGGLE_KEY'] = self.kaggle_key
            self.logger.debug("Kaggle credentials set from UI")

        # Fix kaggle.json permissions
        try:
            cfg = os.path.join(os.path.expanduser("~"), ".kaggle", "kaggle.json")
            if os.path.isfile(cfg):
                os.chmod(cfg, stat.S_IRUSR | stat.S_IWUSR)
        except Exception:
            pass

        # Check if kaggle module is available for CLI usage
        try:
            import kaggle
            has_kaggle = True
            self.logger.debug("Kaggle module available, using CLI method")
        except ImportError:
            has_kaggle = False
            self.logger.debug("Kaggle module not available")

        if has_kaggle:
            self._download_via_cli(root)
        else:
            raise ModuleNotFoundError(
                "Kaggle package not installed. Install with: pip install kaggle"
            )

    def _download_via_cli(self, root: str):
        """Download using Kaggle CLI module."""
        self.status.emit("Downloading Celebrities via Kaggle CLI...")
        self.logger.info("Using Kaggle CLI via: %s -m kaggle", sys.executable)
        self.progress.emit(10)

        # Use python -m kaggle.cli to avoid shebang issues with venv scripts
        cmd = [
            sys.executable, "-m", "kaggle.cli", "datasets", "download",
            "-d", self.CELEBRITIES_KAGGLE,
            "-p", root,
            "--force",
            "--unzip",
            "--quiet"
        ]

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            self.progress.emit(100)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Kaggle CLI failed: {e}") from e

        if self._count_images(root) == 0:
            raise RuntimeError(f"No images found after download in: {root}")

        self.status.emit("Celebrities download complete")
        self.logger.info("Celebrities downloaded via CLI to: %s", root)

    def _download_via_api(self, root: str):
        """Download using Kaggle Python API."""
        self.status.emit("Downloading Celebrities via Kaggle API...")
        self.logger.info("Using Kaggle Python API")
        self.progress.emit(10)

        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except ImportError as e:
            raise ModuleNotFoundError(
                "Kaggle package not available and CLI not found. "
                "Install 'kaggle' package or add CLI to PATH."
            ) from e

        # Workaround for Python 3.12 strptime bug with GMT timezone
        # The Kaggle API fails to parse 'Last-Modified' header dates
        self._patch_kaggle_date_parsing()

        api = KaggleApi()
        try:
            api.authenticate()
        except Exception as e:
            raise RuntimeError(
                "Could not authenticate with Kaggle. "
                "Ensure ~/.kaggle/kaggle.json exists with 600 permissions, "
                "or set KAGGLE_USERNAME/KAGGLE_KEY environment variables, "
                "or provide credentials in the UI."
            ) from e

        api.dataset_download_files(
            self.CELEBRITIES_KAGGLE,
            path=root,
            unzip=True,
            force=True,
            quiet=True
        )
        self.progress.emit(100)

        if self._count_images(root) == 0:
            raise RuntimeError(f"No images found after download in: {root}")

        self.status.emit("Celebrities download complete")
        self.logger.info("Celebrities downloaded via API to: %s", root)

    @staticmethod
    def _patch_kaggle_date_parsing():
        """
        Monkey-patch Kaggle API to fix Python 3.12 strptime bug.
        The issue: 'GMT' timezone is not recognized by strptime in Python 3.12.
        """
        try:
            import kaggle.api.kaggle_api_extended as kaggle_api
            from datetime import datetime
            import email.utils

            # Store original method
            original_download_file = kaggle_api.KaggleApi.download_file

            def patched_download_file(self, response, outfile, http_client, quiet=True, resume=False):
                """Patched download_file that handles GMT timezone parsing."""
                # Try to parse the date using email.utils which handles GMT correctly
                try:
                    return original_download_file(self, response, outfile, http_client, quiet, resume)
                except ValueError as e:
                    if "does not match format" in str(e) and "GMT" in str(e):
                        # The date parsing failed, but we can continue anyway
                        # since we're using force=True which should skip date comparison
                        # Just proceed with the download without date check
                        import kaggle
                        outfile_name = outfile
                        if not quiet:
                            print(f"Downloading {outfile_name}...")

                        # Direct download without date comparison
                        with open(outfile, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                        return
                    raise

            # Apply patch
            kaggle_api.KaggleApi.download_file = patched_download_file
        except Exception as e:
            # If patching fails, log but continue - the download might still work
            pass

    @staticmethod
    def _count_images(root: str) -> int:
        """Count image files in directory recursively."""
        count = 0
        for dirpath, _, filenames in os.walk(root):
            for f in filenames:
                ext = os.path.splitext(f)[1].lower()
                if ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"):
                    count += 1
        return count

    @classmethod
    def is_svhn_downloaded(cls) -> bool:
        """Check if SVHN is already downloaded."""
        dst = os.path.join(cls.SVHN_DIR, cls.SVHN_FILE)
        return os.path.exists(dst)

    @classmethod
    def is_celebrities_downloaded(cls) -> bool:
        """Check if Celebrities dataset is already downloaded."""
        return cls._count_images(cls.CELEBRITIES_DIR) > 0

    @classmethod
    def get_native_size(cls, dataset_type: str) -> int:
        """Get native image size for a dataset type."""
        if dataset_type.lower() == "svhn":
            return cls.SVHN_NATIVE_SIZE
        elif dataset_type.lower() == "celebrities":
            return cls.CELEBRITIES_NATIVE_SIZE
        return 64  # default
