# Language: python
# File: ui/custom_widgets/dataset_control/generate_dataset_internet/generate_dataset_from_internet_widget.py

import os
import logging
from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal, QThread
from PyQt5.QtWidgets import QMessageBox, QGroupBox, QFormLayout, QLineEdit, QProgressBar

from ui.custom_widgets.dataset_control.generate_dataset_internet.ui_generate_dataset_internet_widget import Ui_Generates_dataset_internet
from ui.custom_widgets.dataset_control.generate_dataset_internet.download_worker import DownloadWorker
from ui.custom_widgets.common.data_format_selector import DataFormatSelector
from ui.custom_widgets.common.speckle_noise_widget import SpeckleNoiseWidget
from ui.custom_widgets.common.button_styles import (
    BUTTON_STYLE_GREEN, BUTTON_STYLE_BLUE, apply_button_style
)

# Datasets
from simulation_engine._1_dataset_gen.DatasetFromSVHN import DatasetFromSVHN
from simulation_engine._1_dataset_gen.DatasetFromCelebrities import DatasetFromCelebrities


class GeneratesDatasetInternetWidget(QtWidgets.QWidget, Ui_Generates_dataset_internet):
    """
    Widget that configures and emits a dataset object from the Internet:
    - SVHN (digits) or Celebrities (celebrities faces), selected via combo box.

    Features:
    - Download button to download datasets before creating
    - Status indicator showing if dataset is downloaded
    - Dynamic image dimension options based on native dataset resolution
    - Kaggle credentials configuration for Celebrities dataset

    Emits `datasetReady(dataset_instance)` so the handler can run it in a worker.
    """
    datasetReady = pyqtSignal(object)

    # Dataset configurations
    DATASET_INFO = {
        0: {"name": "svhn", "native_size": 32, "label": "SVHN"},
        1: {"name": "celebrities", "native_size": 64, "label": "Celebrities"}
    }

    def __init__(self, parent=None, logger=None):
        super().__init__(parent)
        self.setupUi(self)

        # Logger
        if logger is None:
            self.logger = logging.getLogger(self.__class__.__name__)
        else:
            self.logger = logger.getChild(self.__class__.__name__)
        self.logger.debug("Initializing GeneratesDatasetInternetWidget (SVHN/Celebrities)")

        # Worker thread reference
        self._download_thread = None
        self._download_worker = None

        # === Add download section ===
        self._setup_download_section()

        # === Add Kaggle credentials section ===
        self._setup_kaggle_credentials()

        # === Add data format selector (to main_layout, not formLayout, for proper label alignment) ===
        self.data_format_selector = DataFormatSelector(logger=self.logger)
        self.main_layout.insertWidget(self.main_layout.count() - 1, self.data_format_selector)

        # === Add speckle noise widget (default 0 = disabled) ===
        self.speckle_noise_widget = SpeckleNoiseWidget(logger=self.logger)
        self.main_layout.insertWidget(self.main_layout.count() - 1, self.speckle_noise_widget)

        # Connect dataset selection change
        self.select_dataset_comboBox.currentIndexChanged.connect(self._on_dataset_changed)

        # Connect and style the button
        self.generate_dataset_button.clicked.connect(self._on_generate_dataset)
        apply_button_style(self.generate_dataset_button, BUTTON_STYLE_GREEN)

        # Initialize UI state
        self._on_dataset_changed(self.select_dataset_comboBox.currentIndex())

    def _setup_download_section(self):
        """Setup download button, progress bar, and status indicator."""
        # Create horizontal layout for download controls
        download_layout = QtWidgets.QHBoxLayout()
        download_layout.setContentsMargins(0, 5, 0, 5)

        # Download button
        self.download_button = QtWidgets.QPushButton("Download Dataset")
        self.download_button.clicked.connect(self._on_download_clicked)
        apply_button_style(self.download_button, BUTTON_STYLE_BLUE)
        download_layout.addWidget(self.download_button)

        # Status indicator
        self.status_label = QtWidgets.QLabel()
        self.status_label.setMinimumWidth(150)
        download_layout.addWidget(self.status_label, 1)

        # Progress bar (hidden by default)
        self.download_progress = QProgressBar()
        self.download_progress.setMinimum(0)
        self.download_progress.setMaximum(100)
        self.download_progress.setVisible(False)
        self.download_progress.setMaximumHeight(20)

        # Insert download controls after dataset selection (row 2 in form)
        # Insert as a widget in the main layout after form layout
        self.main_layout.insertLayout(1, download_layout)
        self.main_layout.insertWidget(2, self.download_progress)

    def _setup_kaggle_credentials(self):
        """Setup Kaggle credentials section (visible only for Celebrities)."""
        self.kaggle_group = QGroupBox("Kaggle Credentials (optional)")
        kaggle_layout = QFormLayout(self.kaggle_group)

        self.kaggle_username_input = QLineEdit()
        self.kaggle_username_input.setPlaceholderText("Leave empty to use ~/.kaggle/kaggle.json")
        kaggle_layout.addRow("Username:", self.kaggle_username_input)

        self.kaggle_key_input = QLineEdit()
        self.kaggle_key_input.setPlaceholderText("Kaggle API Key")
        self.kaggle_key_input.setEchoMode(QLineEdit.Password)
        kaggle_layout.addRow("API Key:", self.kaggle_key_input)

        # Help label
        help_label = QtWidgets.QLabel(
            '<a href="https://www.kaggle.com/settings/account">Get API key from Kaggle</a>'
        )
        help_label.setOpenExternalLinks(True)
        help_label.setStyleSheet("color: #666; font-size: 11px;")
        kaggle_layout.addRow("", help_label)

        # Insert after download section
        self.main_layout.insertWidget(3, self.kaggle_group)
        self.kaggle_group.setVisible(False)  # Hidden by default

    def _on_dataset_changed(self, index: int):
        """Handle dataset selection change."""
        self.logger.debug("Dataset changed to index: %d", index)

        # Update image dimension options
        self._update_image_dimensions(index)

        # Update download status
        self._update_download_status(index)

        # Show/hide Kaggle credentials
        self.kaggle_group.setVisible(index == 1)  # Show for Celebrities

    def _update_image_dimensions(self, dataset_index: int):
        """Update image dimension combobox based on dataset native size."""
        info = self.DATASET_INFO.get(dataset_index, {"native_size": 64})
        native_size = info["native_size"]

        # Generate valid sizes (powers of 2, up to native size)
        valid_sizes = []
        size = 4
        while size <= native_size:
            valid_sizes.append(str(size))
            size *= 2

        # Remember current selection if possible
        current_text = self.image_dimension_value.currentText()

        # Update combobox
        self.image_dimension_value.clear()
        self.image_dimension_value.addItems(valid_sizes)

        # Try to restore selection, otherwise select max
        if current_text in valid_sizes:
            self.image_dimension_value.setCurrentText(current_text)
        else:
            self.image_dimension_value.setCurrentIndex(len(valid_sizes) - 1)

        self.logger.debug("Image dimensions updated: %s (native: %d)", valid_sizes, native_size)

    def _update_download_status(self, dataset_index: int = None):
        """Update the download status indicator."""
        if dataset_index is None:
            dataset_index = self.select_dataset_comboBox.currentIndex()

        info = self.DATASET_INFO.get(dataset_index, {"name": "unknown", "label": "Unknown"})

        if info["name"] == "svhn":
            is_downloaded = DownloadWorker.is_svhn_downloaded()
        elif info["name"] == "celebrities":
            is_downloaded = DownloadWorker.is_celebrities_downloaded()
        else:
            is_downloaded = False

        if is_downloaded:
            self.status_label.setText(f"<b style='color: #228B22;'>Downloaded</b>")
            self.status_label.setToolTip(f"{info['label']} dataset is ready to use")
            self.download_button.setText("Re-download")
        else:
            self.status_label.setText(f"<b style='color: #B22222;'>Not Downloaded</b>")
            self.status_label.setToolTip(f"{info['label']} dataset needs to be downloaded first")
            self.download_button.setText("Download Dataset")

        # Enable/disable create button based on download status
        self.generate_dataset_button.setEnabled(is_downloaded)

    def _on_download_clicked(self):
        """Handle download button click."""
        if self._download_thread is not None and self._download_thread.isRunning():
            QMessageBox.warning(self, "Download in progress",
                                "A download is already in progress. Please wait.")
            return

        dataset_index = self.select_dataset_comboBox.currentIndex()
        info = self.DATASET_INFO.get(dataset_index, {"name": "unknown"})

        # Get Kaggle credentials if Celebrities
        kaggle_username = None
        kaggle_key = None
        if info["name"] == "celebrities":
            kaggle_username = self.kaggle_username_input.text().strip() or None
            kaggle_key = self.kaggle_key_input.text().strip() or None

        # Create worker
        self._download_worker = DownloadWorker(
            dataset_type=info["name"],
            kaggle_username=kaggle_username,
            kaggle_key=kaggle_key,
            logger=self.logger
        )

        # Create thread
        self._download_thread = QThread()
        self._download_worker.moveToThread(self._download_thread)

        # Connect signals
        self._download_worker.progress.connect(self._on_download_progress)
        self._download_worker.status.connect(self._on_download_status)
        self._download_worker.finished.connect(self._on_download_finished)
        self._download_worker.error.connect(self._on_download_error)

        # Cleanup
        self._download_worker.finished.connect(self._download_thread.quit)
        self._download_worker.error.connect(self._download_thread.quit)
        self._download_thread.finished.connect(self._download_worker.deleteLater)
        self._download_thread.finished.connect(self._download_thread.deleteLater)

        # Start
        self._download_thread.started.connect(self._download_worker.run)

        # UI updates
        self.download_button.setEnabled(False)
        self.download_progress.setVisible(True)
        self.download_progress.setValue(0)

        self._download_thread.start()
        self.logger.info("Started download for %s", info["name"])

    def _on_download_progress(self, percent: int):
        """Handle download progress update."""
        self.download_progress.setValue(percent)

    def _on_download_status(self, message: str):
        """Handle download status message."""
        self.status_label.setText(f"<i>{message}</i>")
        self.logger.debug("Download status: %s", message)

    def _on_download_finished(self):
        """Handle download completion."""
        self.download_button.setEnabled(True)
        self.download_progress.setVisible(False)
        self._update_download_status()
        self.logger.info("Download completed successfully")

    def _on_download_error(self, error: Exception):
        """Handle download error."""
        self.download_button.setEnabled(True)
        self.download_progress.setVisible(False)
        self._update_download_status()
        QMessageBox.critical(self, "Download Error", str(error))
        self.logger.error("Download error: %s", error)

    def _on_generate_dataset(self):
        """
        Read UI parameters, instantiate the selected dataset (SVHN or Celebrities),
        and emit `datasetReady`.
        """
        # --- Read image size ---
        try:
            img_size = int(self.image_dimension_value.currentText())
            if img_size <= 0:
                raise ValueError("img_size must be positive")
            self.logger.info("Image size selected: %d", img_size)
        except Exception as e:
            self.logger.error("Error reading image dimension: %s", e, exc_info=True)
            QMessageBox.warning(self, "Error", "Please select a valid image size.")
            return

        # --- Read dataset length ---
        try:
            num_images = int(self.dataset_size_value.value())
            if num_images <= 0:
                raise ValueError("num_images must be positive")
        except Exception as e:
            self.logger.error("Error reading number of images: %s", e, exc_info=True)
            QMessageBox.warning(self, "Error", "Please select a valid number of images.")
            return

        # --- Read seed ---
        try:
            seed = int(self.random_seed_value.value())
            self.logger.info("Random seed selected: %d", seed)
        except Exception as e:
            self.logger.error("Error reading random seed: %s", e, exc_info=True)
            QMessageBox.warning(self, "Error", "Please select a valid random seed.")
            return

        # --- Determine dataset type ---
        ds_index = self.select_dataset_comboBox.currentIndex()
        ds_label = self.select_dataset_comboBox.currentText()
        self.logger.debug("Dataset selected: index=%d, label='%s'", ds_index, ds_label)

        # --- Get selected data format ---
        data_format = self.data_format_selector.get_format()
        self.logger.info("Data format selected: %s", data_format)

        # --- Get speckle noise level ---
        speckle_noise = self.speckle_noise_widget.get_value()

        # --- Create dataset instance ---
        try:
            if ds_index == 0:
                # ----- SVHN -----
                self.logger.info(
                    "Creating DatasetFromSVHN -> img_size=%d, num_images=%d, seed=%d, format=%s, speckle=%.2f",
                    img_size, num_images, seed, data_format, speckle_noise
                )
                ds = DatasetFromSVHN(
                    name="SVHN",
                    img_size=img_size,
                    num_images=num_images,
                    seed=seed,
                    use_split="train",
                    logger=self.logger,
                    data_format=data_format,
                    speckle_noise=speckle_noise
                )
            else:
                # ----- Celebrities -----
                self.logger.info(
                    "Creating DatasetFromCelebrities -> img_size=%d, num_images=%d, seed=%d, format=%s, speckle=%.2f",
                    img_size, num_images, seed, data_format, speckle_noise
                )
                ds = DatasetFromCelebrities(
                    name="Celebrities",
                    img_size=img_size,
                    num_images=num_images,
                    seed=seed,
                    use_split="train",
                    download_dir="../datasets/tmp_downloaded_datasets/celebrities",
                    logger=self.logger,
                    max_shards=1,
                    expand_views=True,
                    data_format=data_format,
                    speckle_noise=speckle_noise
                )
        except Exception as e:
            self.logger.exception("Error creating dataset: %s", e)
            QMessageBox.critical(self, "Error", f"Could not create dataset:\n{e}")
            return

        # Emit so the main handler can run the worker
        self.logger.info(
            "Emitting datasetReady with %s (%s)",
            getattr(ds, "name", "<unnamed>"), ds.__class__.__name__
        )
        self.datasetReady.emit(ds)
