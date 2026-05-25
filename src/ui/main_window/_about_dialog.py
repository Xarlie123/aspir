"""About dialog — rich HTML description with logo, shown from the Help menu."""
from __future__ import annotations

import os

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)


def show_about_dialog(parent, assets_dir: str):
    """Show the About dialog with application information and logo."""
    dialog = QDialog(parent)
    dialog.setWindowTitle("About ASPIR")
    dialog.setMinimumWidth(500)

    layout = QVBoxLayout(dialog)
    layout.setSpacing(15)
    layout.setContentsMargins(20, 20, 20, 20)

    # Logo at top center
    logo_path = os.path.join(assets_dir, 'logo_banner.png')
    if os.path.exists(logo_path):
        logo_label = QLabel()
        logo_pixmap = QPixmap(logo_path)
        scaled_logo = logo_pixmap.scaledToHeight(80, Qt.SmoothTransformation)
        logo_label.setPixmap(scaled_logo)
        logo_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(logo_label)

    # About text
    about_text = """<h2 style="text-align: center;">ASPIR</h2>
<p style="text-align: center;"><b>A Single-Pixel Imaging Research Platform</b></p>
<p style="text-align: center;">Version 1.0.1</p>
<hr>
<p>ASPIR is an open-source platform developed in Python designed to bring the world of
Single-Pixel Imaging (SPI) and Artificial Intelligence (AI) closer to researchers and
students, breaking down the programming barrier. The software implements an end-to-end
pipeline for testing denoising algorithms.</p>
<p><b>Features:</b></p>
<ul>
<li>Dataset generation: IR Beam (LightPipes), images, folders, CelebA, SVHN</li>
<li>Mask patterns: Scatter, Hadamard (4 variants), Sweep, Cal-Sal</li>
<li>Reconstruction: Ghost Imaging, Pseudoinverse, FISTA, TV-Norm</li>
<li>Neural network denoising: 10 architectures (U-Net, DnCNN, cGAN, etc.)</li>
<li>Analysis: quality metrics (PSNR, SSIM, LPIPS), timing, energy profiling</li>
<li>Batch experiments with comparative reports</li>
</ul>
<hr>
<p><b>Author:</b> Carlos Chabert Ull &mdash; <a href="mailto:cchabert@uji.es">cchabert@uji.es</a></p>
<p><b>Repository:</b> <a href="https://github.com/Xarlie123/aspir">github.com/Xarlie123/aspir</a></p>
<p><b>Documentation:</b> <a href="https://aspir.readthedocs.io">aspir.readthedocs.io</a></p>
<p><b>License:</b> <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a></p>
<p style="text-align: center;">Institute of New Imaging Technologies (INIT), Universitat Jaume I, Spain</p>
"""
    text_label = QLabel(about_text)
    text_label.setTextFormat(Qt.RichText)
    text_label.setWordWrap(True)
    text_label.setOpenExternalLinks(True)
    layout.addWidget(text_label)

    # Close button
    button_layout = QHBoxLayout()
    button_layout.addStretch()
    close_btn = QPushButton("Close")
    close_btn.clicked.connect(dialog.accept)
    close_btn.setFixedWidth(100)
    button_layout.addWidget(close_btn)
    button_layout.addStretch()
    layout.addLayout(button_layout)

    dialog.exec()
