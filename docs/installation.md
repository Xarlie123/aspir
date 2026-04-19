# Installation

ASPIR can be installed locally or run via Docker. Docker is recommended for most users as it includes all dependencies pre-configured.

## Requirements

- Python 3.10+
- NVIDIA GPU with NVIDIA CUDA 12.4 (for GPU acceleration)
- 8GB+ RAM recommended

## Option 1: Docker (Recommended)

Docker provides a pre-configured environment with all dependencies and optional tools (pdflatex, Nsight Systems, Kaggle CLI).

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows/Mac) or Docker Engine (Linux)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for GPU support

### Build the image

```bash
git clone https://github.com/Xarlie123/aspir.git
cd aspir
docker build -f docker/Dockerfile -t aspir .
```

### Run on Linux

```bash
xhost +local:
docker run --rm -it \
  --gpus all \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v /sys/class/powercap:/sys/class/powercap:ro \
  -v "$(pwd)":/app \
  aspir
```

### Run on Windows

1. Install [VcXsrv](https://sourceforge.net/projects/vcxsrv/) (free X server)

2. Launch VcXsrv with these settings:
   - Display settings: **Multiple windows**, Display number: **0**
   - Client startup: **Start no client**
   - Extra settings: Check **Disable access control**

3. Run the container:

```powershell
$env:DISPLAY="host.docker.internal:0.0"
docker run --rm -it `
  --gpus all `
  -e DISPLAY=$env:DISPLAY `
  -v "${PWD}:/app" `
  aspir
```

## Option 2: Local Installation

For development or if you prefer not to use Docker.

Dependencies are declared once in `pyproject.toml` (PEP 621). Optional-dependency
groups (`cpu`, `jetson`, `dev`, `docs`) let you pick the right set for your
platform without maintaining separate lock files.

### Linux (CUDA)

```bash
# Clone repository
git clone https://github.com/Xarlie123/aspir.git
cd aspir

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install the project and its dependencies (editable)
pip install -e .

# Run (working directory MUST be src/)
cd src
python main.py
```

The legacy `pip install -r requirements_linux.txt` still works — the file is a
thin wrapper that resolves to `-e .`.

### Windows (CUDA)

```powershell
git clone https://github.com/Xarlie123/aspir.git
cd aspir

python -m venv .venv
.venv\Scripts\activate

pip install -e .

cd src
python main.py
```

### Windows (CPU-only)

For machines without an NVIDIA GPU:

```powershell
pip install -e . --extra-index-url https://download.pytorch.org/whl/cpu
```

`requirements_windows_cpu.txt` is a convenience wrapper that encodes this.

### NVIDIA Jetson (JetPack 6.2 / CUDA 12.6)

PyTorch for aarch64 is not on PyPI; install it first from the Jetson AI Lab
index, then the rest of the project:

```bash
# 1. Torch for Jetson (pick the wheel that matches your JetPack + Python)
pip install --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126/+simple/ torch

# 2. The rest of the project plus the Jetson extras
pip install -e .[jetson]
```

TensorRT comes pre-installed with JetPack — do not try to `pip install` it
on the device.

### Developer extras

```bash
pip install -e .[dev]          # ruff + import-linter
pip install -e .[docs]         # Sphinx stack
pip install -e .[dev,docs]     # both
```

## Optional Tools

These are pre-installed in Docker. For local installation, configure paths via **Settings → External Applications**:

| Tool | Purpose | Installation |
|------|---------|--------------|
| pdflatex | DNN architecture diagrams | [TeX Live](https://www.tug.org/texlive/) or [MiKTeX](https://miktex.org/) |
| nsys | NVIDIA profiling | [Nsight Systems](https://developer.nvidia.com/nsight-systems) |
| kaggle | Dataset downloads | `pip install kaggle` |

## Verifying Installation

After launching ASPIR, you should see the main window with three mode tabs:

1. **Single Test** - Interactive experimentation
2. **Batch Test** - Automated parameter sweeps
3. **Batch Reports** - Results analysis

Try loading a sample image from **Dataset → From Image** to verify everything works.

## Troubleshooting

### GPU not detected

Ensure NVIDIA CUDA is properly installed:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### X11 forwarding issues (Docker)

On Linux, ensure `xhost +local:` was executed before running the container.

### Import errors

Make sure you're running from the `src/` directory:

```bash
cd src
python main.py  # Correct
```

Not from the repository root:

```bash
python src/main.py  # Wrong - imports will fail
```
