"""UI builder helpers for :class:`ArchitecturePreviewPopup`.

These functions take the popup instance and set up widgets on it, attaching
layouts and controls as attributes so the dialog's own methods can reference
them. Kept separate from ``popup.py`` purely to keep the file readable —
there is no behavioural split.
"""
from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ui.custom_widgets.postprocessor_control.architecture_config.architecture_preview_popup._zoomable_image_view import (
    ZoomableImageView,
)


def populate_model_info_panel(popup):
    """Populate (or refresh) the model information panel of the popup.

    Clears any previous labels, recomputes parameter counts and appends rows
    for total/trainable/non-trainable parameters, input size, model class
    and optional architecture configuration.
    """
    # Clear existing labels
    for label in popup._info_value_labels.values():
        label.deleteLater()
    popup._info_value_labels.clear()

    # Remove all items from layout
    while popup._info_layout.count():
        item = popup._info_layout.takeAt(0)
        if item.widget():
            item.widget().deleteLater()

    # Calculate parameters
    total_params = sum(p.numel() for p in popup.model.parameters())
    trainable_params = sum(p.numel() for p in popup.model.parameters()
                          if p.requires_grad)
    non_trainable = total_params - trainable_params

    # Format parameter counts
    def format_params(n: int) -> str:
        if n >= 1_000_000:
            return f"{n:,} ({n / 1_000_000:.2f}M)"
        elif n >= 1_000:
            return f"{n:,} ({n / 1_000:.1f}K)"
        return f"{n:,}"

    # Create info labels
    font_bold = QFont()
    font_bold.setBold(True)

    info_items = [
        ("Total Parameters:", format_params(total_params)),
        ("Trainable:", format_params(trainable_params)),
        ("Non-trainable:", format_params(non_trainable)),
        ("Input Size:", f"{popup.input_size} x {popup.input_size} x 1"),
        ("Model Class:", type(popup.model).__name__),
    ]

    for i, (label_text, value_text) in enumerate(info_items):
        row = i // 3
        col = (i % 3) * 2

        label = QLabel(label_text)
        label.setFont(font_bold)
        value = QLabel(value_text)
        popup._info_value_labels[label_text] = value

        popup._info_layout.addWidget(label, row, col)
        popup._info_layout.addWidget(value, row, col + 1)

    # Add configuration info if available
    if popup.config:
        config_str = ", ".join(f"{k}={v}" for k, v in popup.config.items())
        config_label = QLabel("Configuration:")
        config_label.setFont(font_bold)
        config_value = QLabel(config_str)
        config_value.setWordWrap(True)
        popup._info_value_labels["Configuration:"] = config_value

        row = len(info_items) // 3 + 1
        popup._info_layout.addWidget(config_label, row, 0)
        popup._info_layout.addWidget(config_value, row, 1, 1, 5)


def build_ui(popup):
    """Build the full dialog layout and attach every widget to ``popup``.

    Equivalent to the original ``ArchitecturePreviewPopup._setup_ui`` method;
    extracted only to keep ``popup.py`` readable.
    """
    main_layout = QVBoxLayout(popup)
    main_layout.setContentsMargins(15, 15, 15, 15)
    main_layout.setSpacing(12)

    # Test selector (only in Batch Reports mode)
    if popup._tests and len(popup._tests) > 1:
        test_selector_layout = QHBoxLayout()
        test_selector_layout.setSpacing(10)

        test_label = QLabel("Test:")
        test_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        test_selector_layout.addWidget(test_label)

        popup.test_combo = QComboBox()
        popup.test_combo.setMinimumWidth(300)
        popup.test_combo.setStyleSheet("""
            QComboBox {
                padding: 5px 10px;
                font-size: 12px;
                border: 1px solid #ccc;
                border-radius: 4px;
            }
        """)
        for test in popup._tests:
            t_name = test.get("name", "Unknown")
            m_name = test.get("model_name", test.get("config", {}).get("model_name", "Unknown"))
            popup.test_combo.addItem(f"{t_name} ({m_name})")
        popup.test_combo.currentIndexChanged.connect(popup._on_test_changed)
        test_selector_layout.addWidget(popup.test_combo)

        test_selector_layout.addStretch()
        main_layout.addLayout(test_selector_layout)
    else:
        popup.test_combo = None

    # Title
    if popup._test_name:
        popup.title_label = QLabel(f"<h2>{popup._test_name} - {popup.model_name} Architecture</h2>")
    else:
        popup.title_label = QLabel(f"<h2>{popup.model_name} Architecture</h2>")
    popup.title_label.setAlignment(Qt.AlignCenter)
    main_layout.addWidget(popup.title_label)

    # Model information panel
    info_group = QGroupBox("Model Information")
    popup._info_layout = QGridLayout(info_group)
    popup._info_layout.setSpacing(15)

    # Store references to value labels for dynamic updates
    popup._info_value_labels = {}
    populate_model_info_panel(popup)

    main_layout.addWidget(info_group)

    # Status bar with progress and zoom info
    status_layout = QHBoxLayout()

    popup.progress_label = QLabel("Generating visualization...")
    popup.progress_label.setStyleSheet("color: #666;")
    status_layout.addWidget(popup.progress_label)

    status_layout.addStretch()

    # Image visualization controls (only if post-processing results available)
    if popup._has_postprocessing:
        # Checkbox to enable/disable image display
        popup.show_images_checkbox = QCheckBox("Show I/O images")
        popup.show_images_checkbox.setToolTip("Display input and output images in the visualization")
        popup.show_images_checkbox.stateChanged.connect(popup._on_show_images_changed)
        status_layout.addWidget(popup.show_images_checkbox)

        # Calculate number of images (use minimum across all available sets)
        popup._num_images = min(
            len(popup.ground_truth_images) if popup.ground_truth_images is not None else 0,
            len(popup.noisy_images) if popup.noisy_images is not None else 0,
            len(popup.denoised_images) if popup.denoised_images is not None else 0
        )

        # Image index selector with total count
        popup.image_idx_label = QLabel("Image:")
        popup.image_idx_label.setStyleSheet("margin-left: 8px;")
        popup.image_idx_label.setEnabled(False)
        status_layout.addWidget(popup.image_idx_label)

        popup.image_idx_spinbox = QSpinBox()
        popup.image_idx_spinbox.setRange(0, max(0, popup._num_images - 1))
        popup.image_idx_spinbox.setValue(0)
        popup.image_idx_spinbox.setMaximumWidth(70)
        popup.image_idx_spinbox.setEnabled(False)
        popup.image_idx_spinbox.valueChanged.connect(popup._on_image_idx_changed)
        status_layout.addWidget(popup.image_idx_spinbox)

        # Max index label (show maximum selectable index, matching spinbox range)
        popup._max_image_idx = max(0, popup._num_images - 1)
        popup.image_total_label = QLabel(f"/ {popup._max_image_idx}")
        popup.image_total_label.setStyleSheet("color: #666;")
        popup.image_total_label.setEnabled(False)
        status_layout.addWidget(popup.image_total_label)

        # Colormap selector
        popup.colormap_label = QLabel("Colormap:")
        popup.colormap_label.setStyleSheet("margin-left: 8px;")
        popup.colormap_label.setEnabled(False)
        status_layout.addWidget(popup.colormap_label)

        popup.colormap_combo = QComboBox()
        popup.colormap_combo.addItems([
            popup.COLORMAP_GRAY,
            popup.COLORMAP_VIRIDIS,
            popup.COLORMAP_JET,
            popup.COLORMAP_HOT,
            popup.COLORMAP_INFERNO,
            popup.COLORMAP_PLASMA
        ])
        popup.colormap_combo.setCurrentText(popup.COLORMAP_HOT)
        popup.colormap_combo.setMaximumWidth(85)
        popup.colormap_combo.setEnabled(False)
        popup.colormap_combo.currentTextChanged.connect(popup._on_colormap_changed)
        status_layout.addWidget(popup.colormap_combo)

        # Input image type selector
        popup.input_type_label = QLabel("Input:")
        popup.input_type_label.setStyleSheet("margin-left: 8px;")
        popup.input_type_label.setEnabled(False)
        status_layout.addWidget(popup.input_type_label)

        popup.input_type_combo = QComboBox()
        popup.input_type_combo.addItems([
            popup.IMAGE_TYPE_GROUND_TRUTH,
            popup.IMAGE_TYPE_NOISY,
            popup.IMAGE_TYPE_DENOISED
        ])
        popup.input_type_combo.setCurrentText(popup.IMAGE_TYPE_NOISY)
        popup.input_type_combo.setMaximumWidth(100)
        popup.input_type_combo.setEnabled(False)
        popup.input_type_combo.currentTextChanged.connect(popup._on_input_type_changed)
        status_layout.addWidget(popup.input_type_combo)

        # Output image type selector
        popup.output_type_label = QLabel("Output:")
        popup.output_type_label.setStyleSheet("margin-left: 8px;")
        popup.output_type_label.setEnabled(False)
        status_layout.addWidget(popup.output_type_label)

        popup.output_type_combo = QComboBox()
        popup.output_type_combo.addItems([
            popup.IMAGE_TYPE_GROUND_TRUTH,
            popup.IMAGE_TYPE_NOISY,
            popup.IMAGE_TYPE_DENOISED
        ])
        popup.output_type_combo.setCurrentText(popup.IMAGE_TYPE_DENOISED)
        popup.output_type_combo.setMaximumWidth(100)
        popup.output_type_combo.setEnabled(False)
        popup.output_type_combo.currentTextChanged.connect(popup._on_output_type_changed)
        status_layout.addWidget(popup.output_type_combo)

        status_layout.addSpacing(10)
    else:
        # No post-processing results - show info note
        popup.show_images_checkbox = None
        popup.image_idx_spinbox = None
        popup.image_idx_label = None
        popup.image_total_label = None
        popup.colormap_combo = None
        popup.colormap_label = None
        popup.input_type_combo = None
        popup.input_type_label = None
        popup.output_type_combo = None
        popup.output_type_label = None
        popup._num_images = 0

        # Info label
        info_label = QLabel("ℹ️ Train model and apply post-processing to enable I/O image preview")
        info_label.setStyleSheet("color: #888; font-style: italic; margin-left: 10px;")
        info_label.setToolTip(
            "To visualize input/output images:\n"
            "1. Load a dataset\n"
            "2. Generate masks and apply reconstruction\n"
            "3. Train the neural network\n"
            "4. Apply post-processing to generate denoised images"
        )
        status_layout.addWidget(info_label)

    status_layout.addStretch()

    # Zoom controls
    popup.zoom_label = QLabel("Zoom: --")
    popup.zoom_label.setStyleSheet("color: #666; margin-right: 10px;")
    status_layout.addWidget(popup.zoom_label)

    popup.fit_btn = QPushButton("Fit")
    popup.fit_btn.setMaximumWidth(50)
    popup.fit_btn.setToolTip("Fit image to window (also double-click)")
    popup.fit_btn.clicked.connect(popup._on_fit_clicked)
    popup.fit_btn.setEnabled(False)
    status_layout.addWidget(popup.fit_btn)

    popup.zoom_in_btn = QPushButton("+")
    popup.zoom_in_btn.setMaximumWidth(30)
    popup.zoom_in_btn.setToolTip("Zoom in")
    popup.zoom_in_btn.clicked.connect(popup._on_zoom_in)
    popup.zoom_in_btn.setEnabled(False)
    status_layout.addWidget(popup.zoom_in_btn)

    popup.zoom_out_btn = QPushButton("-")
    popup.zoom_out_btn.setMaximumWidth(30)
    popup.zoom_out_btn.setToolTip("Zoom out")
    popup.zoom_out_btn.clicked.connect(popup._on_zoom_out)
    popup.zoom_out_btn.setEnabled(False)
    status_layout.addWidget(popup.zoom_out_btn)

    status_layout.addSpacing(20)

    popup.progress_bar = QProgressBar()
    popup.progress_bar.setMaximumWidth(200)
    popup.progress_bar.setRange(0, 0)  # Indeterminate
    status_layout.addWidget(popup.progress_bar)

    main_layout.addLayout(status_layout)

    # Visualization area with zoomable view
    viz_group = QGroupBox("Architecture Diagram (PlotNeuralNet) - Scroll to zoom, drag to pan")
    viz_layout = QVBoxLayout(viz_group)

    # Zoomable image view
    popup.image_view = ZoomableImageView()
    popup.image_view.zoom_changed.connect(popup._on_zoom_changed)
    popup.image_view.setMinimumHeight(400)

    # Loading widget (shown initially)
    popup.loading_widget = QWidget()
    loading_layout = QVBoxLayout(popup.loading_widget)
    loading_label = QLabel("Compiling LaTeX... This may take a few seconds.")
    loading_label.setAlignment(Qt.AlignCenter)
    loading_label.setStyleSheet("color: #666; font-size: 14px;")
    loading_layout.addWidget(loading_label)

    # Stack loading widget on top initially
    viz_layout.addWidget(popup.loading_widget)
    viz_layout.addWidget(popup.image_view)
    popup.image_view.hide()

    main_layout.addWidget(viz_group, 1)  # Stretch factor 1

    # Buttons
    buttons_layout = QHBoxLayout()
    buttons_layout.addStretch()

    # Save buttons
    popup.save_png_btn = QPushButton("Save as PNG")
    popup.save_png_btn.clicked.connect(lambda: popup._on_save("png"))
    popup.save_png_btn.setEnabled(False)
    buttons_layout.addWidget(popup.save_png_btn)

    popup.save_pdf_btn = QPushButton("Save as PDF")
    popup.save_pdf_btn.clicked.connect(lambda: popup._on_save("pdf"))
    popup.save_pdf_btn.setEnabled(False)
    buttons_layout.addWidget(popup.save_pdf_btn)

    popup.save_tex_btn = QPushButton("Save as TEX")
    popup.save_tex_btn.clicked.connect(lambda: popup._on_save("tex"))
    popup.save_tex_btn.setToolTip("Save LaTeX/TikZ source code for manual editing")
    popup.save_tex_btn.setEnabled(False)
    buttons_layout.addWidget(popup.save_tex_btn)

    buttons_layout.addSpacing(20)

    popup.close_button = QPushButton("Close")
    popup.close_button.clicked.connect(popup.close)
    buttons_layout.addWidget(popup.close_button)

    main_layout.addLayout(buttons_layout)
