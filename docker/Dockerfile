# syntax=docker/dockerfile:1.7
# =============================================================================
# ASPIR: A Single-Pixel Imaging Research Platform
# Docker image with GPU support, LaTeX, and profiling tools
# =============================================================================

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

LABEL maintainer="ASPIR Development Team"
LABEL description="PyQt5 application for single-pixel imaging with PyTorch GPU support"

# -----------------------------------------------------------------------------
# Environment variables
# -----------------------------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    # Qt/X11 settings for GUI
    QT_QPA_PLATFORM=xcb \
    QT_X11_NO_MITSHM=1 \
    MPLBACKEND=Qt5Agg \
    # Ensure UTF-8 locale
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# -----------------------------------------------------------------------------
# System dependencies + NVIDIA Nsight Systems
# -----------------------------------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      # Python
      python3 python3-pip python3-venv \
      # Build tools
      build-essential ca-certificates wget gnupg2 \
      # X11 and Qt dependencies
      xauth dbus-x11 \
      libx11-6 libx11-xcb1 libxext6 libxrender1 libxau6 libxdmcp6 \
      libxkbcommon0 libxkbcommon-x11-0 \
      libxcb1 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 \
      libxcb-render0 libxcb-render-util0 libxcb-shm0 libxcb-xfixes0 \
      libxcb-xinerama0 libxcb-xkb1 libxcb-shape0 libxcb-sync1 \
      libfontconfig1 libfreetype6 libgl1 libgl1-mesa-dri libglib2.0-0 \
      # LaTeX for DNN architecture preview (PlotNeuralNet)
      texlive-latex-base texlive-latex-extra texlive-fonts-recommended \
      # PDF utilities
      poppler-utils \
      # Emoji font for UI icons
      fonts-noto-color-emoji \
    && \
    # Create Python symlinks
    ln -sf /usr/bin/python3 /usr/local/bin/python && \
    ln -sf /usr/bin/pip3 /usr/local/bin/pip && \
    # Cleanup apt cache
    rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Install NVIDIA Nsight Systems (nsys) for GPU profiling
# -----------------------------------------------------------------------------
RUN wget -qO - https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/nvidia.pub | \
      gpg --dearmor -o /usr/share/keyrings/nvidia-devtools.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/nvidia-devtools.gpg] https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/ /" \
      > /etc/apt/sources.list.d/nvidia-devtools.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends nsight-systems-cli && \
    rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Python dependencies
# -----------------------------------------------------------------------------
WORKDIR /app

# Install Python dependencies from pyproject.toml (single source of truth).
# The application source tree itself is mounted at runtime via
# `-v "$(pwd)":/app`, so we only need the dependency manifest during the
# image build. We create stub package directories so setuptools' editable
# install succeeds — pip writes a .pth file pointing to /app/src/, and the
# real src/ will be there at runtime thanks to the volume mount.
COPY pyproject.toml README.md ./
RUN mkdir -p src/simulation_engine src/ui && \
    touch src/main.py src/simulation_engine/__init__.py src/ui/__init__.py && \
    python -m pip install --upgrade pip && \
    pip install -e . && \
    rm -rf src

# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
# Repository is mounted at runtime via: -v "$(pwd)":/app
# Application runs from /app/src
WORKDIR /app/src
ENTRYPOINT ["python"]
CMD ["main.py"]
