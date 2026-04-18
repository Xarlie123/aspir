"""UI builder for :class:`QualityPreviewPopup` — attaches widgets to the popup."""
from __future__ import annotations

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)


def build_ui(popup):
    """Build the Quality Preview popup layout and attach widgets to ``popup``."""
    popup.setWindowTitle("Quality Metrics Preview")
    popup.setMinimumSize(950, 700)
    popup.resize(1050, 750)

    main_layout = QVBoxLayout(popup)
    main_layout.setContentsMargins(10, 10, 10, 10)
    main_layout.setSpacing(8)

    # Test selection row
    selection_layout = QHBoxLayout()

    test_label = QLabel("Select Test:")
    test_label.setStyleSheet("font-weight: bold;")
    selection_layout.addWidget(test_label)

    popup.test_combo = QComboBox()
    popup.test_combo.setMinimumWidth(300)
    popup.test_combo.currentIndexChanged.connect(popup._on_test_changed)
    selection_layout.addWidget(popup.test_combo)

    selection_layout.addStretch()

    # Data status label
    popup.data_status_label = QLabel("")
    popup.data_status_label.setStyleSheet("color: #666; font-style: italic;")
    selection_layout.addWidget(popup.data_status_label)

    main_layout.addLayout(selection_layout)

    # Images section with slider right below
    images_group = QGroupBox("Image Comparison")
    images_main_layout = QVBoxLayout(images_group)
    images_main_layout.setSpacing(8)

    # Images row
    images_layout = QHBoxLayout()
    images_layout.setSpacing(15)

    label_font = QFont()
    label_font.setPointSize(10)

    # Ground-Truth image
    orig_container = QVBoxLayout()
    orig_label = QLabel("Ground-Truth")
    orig_label.setAlignment(Qt.AlignCenter)
    orig_label.setFont(label_font)
    orig_container.addWidget(orig_label)
    popup.orig_image = QLabel()
    popup.orig_image.setAlignment(Qt.AlignCenter)
    popup.orig_image.setFixedSize(popup.IMAGE_DISPLAY_SIZE, popup.IMAGE_DISPLAY_SIZE)
    popup.orig_image.setStyleSheet("background-color: #2a2a2a; border: 1px solid #555;")
    orig_container.addWidget(popup.orig_image)
    orig_container.addStretch()
    images_layout.addLayout(orig_container)

    # Noisy image
    noisy_container = QVBoxLayout()
    noisy_label = QLabel("Noisy (Reconstructed)")
    noisy_label.setAlignment(Qt.AlignCenter)
    noisy_label.setFont(label_font)
    noisy_container.addWidget(noisy_label)
    popup.noisy_image = QLabel()
    popup.noisy_image.setAlignment(Qt.AlignCenter)
    popup.noisy_image.setFixedSize(popup.IMAGE_DISPLAY_SIZE, popup.IMAGE_DISPLAY_SIZE)
    popup.noisy_image.setStyleSheet("background-color: #2a2a2a; border: 1px solid #555;")
    noisy_container.addWidget(popup.noisy_image)
    noisy_container.addStretch()
    images_layout.addLayout(noisy_container)

    # Denoised image
    recon_container = QVBoxLayout()
    recon_label = QLabel("Denoised")
    recon_label.setAlignment(Qt.AlignCenter)
    recon_label.setFont(label_font)
    recon_container.addWidget(recon_label)
    popup.recon_image = QLabel()
    popup.recon_image.setAlignment(Qt.AlignCenter)
    popup.recon_image.setFixedSize(popup.IMAGE_DISPLAY_SIZE, popup.IMAGE_DISPLAY_SIZE)
    popup.recon_image.setStyleSheet("background-color: #2a2a2a; border: 1px solid #555;")
    recon_container.addWidget(popup.recon_image)
    recon_container.addStretch()
    images_layout.addLayout(recon_container)

    images_main_layout.addLayout(images_layout, 1)

    # Image slider (right below images)
    slider_layout = QHBoxLayout()
    slider_layout.setSpacing(10)

    popup.slider_label = QLabel("Image:")
    slider_layout.addWidget(popup.slider_label)

    popup.image_slider = QSlider(Qt.Horizontal)
    popup.image_slider.setMinimum(0)
    popup.image_slider.setMaximum(0)
    popup.image_slider.setValue(0)
    popup.image_slider.valueChanged.connect(popup._on_slider_changed)
    slider_layout.addWidget(popup.image_slider, 1)

    popup.index_label = QLabel("0  (0 images)")
    popup.index_label.setMinimumWidth(100)
    slider_layout.addWidget(popup.index_label)

    # View Mask Application button
    popup.mask_btn = QPushButton("View Mask Application")
    popup.mask_btn.setEnabled(False)
    popup.mask_btn.setToolTip("View how masks are applied to create the noisy image")
    popup.mask_btn.clicked.connect(popup._on_view_mask_application)
    popup.mask_btn.setStyleSheet("""
        QPushButton {
            background-color: #5C6BC0;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 6px 12px;
            font-weight: bold;
        }
        QPushButton:hover:enabled {
            background-color: #3F51B5;
        }
        QPushButton:disabled {
            background-color: #ccc;
            color: #888;
        }
    """)
    slider_layout.addWidget(popup.mask_btn)

    images_main_layout.addLayout(slider_layout)

    main_layout.addWidget(images_group)

    # Metrics section
    metrics_group = QGroupBox("Quality Metrics for Current Image")
    metrics_main_layout = QHBoxLayout(metrics_group)
    metrics_main_layout.setSpacing(15)

    # Left: Metrics table
    table_widget = QWidget()
    table_layout = QGridLayout(table_widget)
    table_layout.setSpacing(8)

    header_font = QFont()
    header_font.setBold(True)

    # Headers
    table_layout.addWidget(QLabel(""), 0, 0)
    noisy_header = QLabel("Noisy")
    noisy_header.setFont(header_font)
    noisy_header.setAlignment(Qt.AlignCenter)
    table_layout.addWidget(noisy_header, 0, 1)
    recon_header = QLabel("Denoised")
    recon_header.setFont(header_font)
    recon_header.setAlignment(Qt.AlignCenter)
    table_layout.addWidget(recon_header, 0, 2)
    change_header = QLabel("Change")
    change_header.setFont(header_font)
    change_header.setAlignment(Qt.AlignCenter)
    table_layout.addWidget(change_header, 0, 3)

    # PSNR row
    popup.psnr_label = QLabel("PSNR (dB) \u2191:")
    popup.psnr_label.setFont(header_font)
    popup.psnr_label.setToolTip("Higher is better")
    popup.psnr_noisy_display = QLabel("-")
    popup.psnr_noisy_display.setAlignment(Qt.AlignCenter)
    popup.psnr_noisy_display.setStyleSheet("font-size: 13px;")
    popup.psnr_recon_display = QLabel("-")
    popup.psnr_recon_display.setAlignment(Qt.AlignCenter)
    popup.psnr_recon_display.setStyleSheet("font-size: 13px;")
    popup.psnr_change_display = QLabel("-")
    popup.psnr_change_display.setAlignment(Qt.AlignCenter)
    popup.psnr_change_display.setStyleSheet("font-size: 13px; font-weight: bold;")
    table_layout.addWidget(popup.psnr_label, 1, 0)
    table_layout.addWidget(popup.psnr_noisy_display, 1, 1)
    table_layout.addWidget(popup.psnr_recon_display, 1, 2)
    table_layout.addWidget(popup.psnr_change_display, 1, 3)

    # SSIM row
    popup.ssim_label = QLabel("SSIM \u2191:")
    popup.ssim_label.setFont(header_font)
    popup.ssim_label.setToolTip("Higher is better")
    popup.ssim_noisy_display = QLabel("-")
    popup.ssim_noisy_display.setAlignment(Qt.AlignCenter)
    popup.ssim_noisy_display.setStyleSheet("font-size: 13px;")
    popup.ssim_recon_display = QLabel("-")
    popup.ssim_recon_display.setAlignment(Qt.AlignCenter)
    popup.ssim_recon_display.setStyleSheet("font-size: 13px;")
    popup.ssim_change_display = QLabel("-")
    popup.ssim_change_display.setAlignment(Qt.AlignCenter)
    popup.ssim_change_display.setStyleSheet("font-size: 13px; font-weight: bold;")
    table_layout.addWidget(popup.ssim_label, 2, 0)
    table_layout.addWidget(popup.ssim_noisy_display, 2, 1)
    table_layout.addWidget(popup.ssim_recon_display, 2, 2)
    table_layout.addWidget(popup.ssim_change_display, 2, 3)

    # LPIPS row
    popup.lpips_label = QLabel("LPIPS \u2193:")
    popup.lpips_label.setFont(header_font)
    popup.lpips_label.setToolTip("Lower is better")
    popup.lpips_noisy_display = QLabel("-")
    popup.lpips_noisy_display.setAlignment(Qt.AlignCenter)
    popup.lpips_noisy_display.setStyleSheet("font-size: 13px;")
    popup.lpips_recon_display = QLabel("-")
    popup.lpips_recon_display.setAlignment(Qt.AlignCenter)
    popup.lpips_recon_display.setStyleSheet("font-size: 13px;")
    popup.lpips_change_display = QLabel("-")
    popup.lpips_change_display.setAlignment(Qt.AlignCenter)
    popup.lpips_change_display.setStyleSheet("font-size: 13px; font-weight: bold;")
    table_layout.addWidget(popup.lpips_label, 3, 0)
    table_layout.addWidget(popup.lpips_noisy_display, 3, 1)
    table_layout.addWidget(popup.lpips_recon_display, 3, 2)
    table_layout.addWidget(popup.lpips_change_display, 3, 3)

    metrics_main_layout.addWidget(table_widget)

    # Right: Bar chart
    popup.bar_figure = Figure(figsize=(4, 2.5), dpi=100)
    popup.bar_canvas = FigureCanvas(popup.bar_figure)
    popup.bar_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    popup.bar_canvas.setMinimumSize(250, 150)
    metrics_main_layout.addWidget(popup.bar_canvas)

    main_layout.addWidget(metrics_group)

    # Info label
    popup.info_label = QLabel("Select a test to preview quality metrics")
    popup.info_label.setStyleSheet("color: #666; font-size: 11px;")
    popup.info_label.setAlignment(Qt.AlignCenter)
    main_layout.addWidget(popup.info_label)
