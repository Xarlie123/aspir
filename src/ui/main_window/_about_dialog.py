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
    dialog.setMinimumWidth(700)
    dialog.setMinimumHeight(700)

    layout = QVBoxLayout(dialog)
    layout.setSpacing(15)
    layout.setContentsMargins(20, 20, 20, 20)

    # Header row: logo left, title/subtitle/version right
    header_layout = QHBoxLayout()
    header_layout.setSpacing(20)
    header_layout.setContentsMargins(0, 0, 0, 0)

    logo_path = os.path.join(assets_dir, 'logo_banner.png')
    if os.path.exists(logo_path):
        logo_label = QLabel()
        logo_pixmap = QPixmap(logo_path)
        scaled_logo = logo_pixmap.scaledToHeight(90, Qt.SmoothTransformation)
        logo_label.setPixmap(scaled_logo)
        logo_label.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        header_layout.addWidget(logo_label)

    title_label = QLabel(
        '<h1 style="margin:0; padding:0;">ASPIR</h1>'
        '<p style="margin:4px 0 0 0; padding:0;"><b>A Single-Pixel Imaging Research Platform</b></p>'
        '<p style="margin:2px 0 0 0; padding:0; color:#666;">Version 1.2.0</p>'
    )
    title_label.setTextFormat(Qt.RichText)
    title_label.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
    header_layout.addWidget(title_label)
    header_layout.addStretch()

    layout.addLayout(header_layout)

    # Body text
    about_text = """<hr>
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
<p><b>Authors:</b> Carlos Chabert-Ull&sup1; (corresponding, <a href="mailto:cchabert@uji.es">cchabert@uji.es</a>),
Heberley Tob&oacute;n-Maya&sup1;, Samuel I. Zapata-Valencia&sup1;, Enrique Tajahuerce&sup1;, Germ&aacute;n Le&oacute;n&sup2;</p>
<p><b>Repository:</b> <a href="https://github.com/Xarlie123/aspir">github.com/Xarlie123/aspir</a></p>
<p><b>Documentation:</b> <a href="https://aspir.readthedocs.io">aspir.readthedocs.io</a></p>
<p><b>License:</b> <a href="https://www.apache.org/licenses/LICENSE-2.0">Apache License 2.0</a></p>
<p style="text-align: center;">
&sup1; GROC research group &mdash; Institute of New Imaging Technologies (INIT)<br>
&sup2; HPC&amp;A research group &mdash; Department of Computer Engineering and Computer Science<br>
Universitat Jaume I, Castell&oacute; de la Plana, Spain
</p>
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
