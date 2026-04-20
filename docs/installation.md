# Installation

ASPIR can be installed locally or run via Docker. Docker is recommended for most users as it includes all dependencies pre-configured.

## Requirements

- Python 3.10+ (tested against 3.10, 3.11 and 3.12)
- NVIDIA GPU with CUDA 12.4 runtime (optional — the project works on CPU
  too; see the CPU-only instructions below)
- 8 GB+ RAM recommended
- Poppler (only needed by the "Preview Architecture" diagram renderer):
  - Linux: `sudo apt install poppler-utils`
  - Windows: download a Poppler build and add its `bin/` to `PATH`
  - macOS: `brew install poppler`

Dependencies are declared in `pyproject.toml` (PEP 621) as the single
source of truth. Previous `requirements_*.txt` files have been
removed — a `pip install -e .` picks everything up.

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

> **RAPL CPU-energy note (Linux only)**: Intel RAPL registers are root
> by default. If you want the Energy analysis to read CPU power, run
> `sudo chmod -R a+r /sys/class/powercap/intel-rapl/` once per boot
> before launching Docker.

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

# Run (either approach works after the editable install)
cd src && python main.py       # historical workflow
# or
python -m main                 # from anywhere in the repo
```

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

### CPU-only (Linux or Windows)

For machines without an NVIDIA GPU. Ask pip to look at PyTorch's
CPU wheel index before PyPI so the CPU build of torch is selected:

```bash
pip install -e . --extra-index-url https://download.pytorch.org/whl/cpu
```

The `[cpu]` extra exists only as a marker — the CPU wheel selection is
driven by `--extra-index-url`.

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

These are pre-installed in Docker. For local installation, configure
paths via **Settings → External Applications…** in the menu bar.

| Tool      | Purpose                          | Installation                                                              |
|-----------|----------------------------------|---------------------------------------------------------------------------|
| pdflatex  | DNN architecture diagrams         | [TeX Live](https://www.tug.org/texlive/) or [MiKTeX](https://miktex.org/) |
| poppler   | Convert pdflatex output to PNG   | `apt install poppler-utils` / Poppler-windows / `brew install poppler`    |
| nsys      | NVIDIA profiling                  | [Nsight Systems](https://developer.nvidia.com/nsight-systems)             |
| kaggle    | Dataset downloads                 | included in the base install (`kaggle` is a core dependency)              |

## Verifying Installation

After launching ASPIR, the main window opens with a mode selector at
the top offering three modes:

1. **Single Test** — interactive step-by-step experimentation (Dataset
   → Masks → Test → DNN → Reports wizard).
2. **Batch Test** — queue several test configurations and run them
   sequentially or in parallel.
3. **Batch Reports** — load one or more `.batch_analysis_report` files
   and compare results (quality, timing, energy, training curves,
   Samples Grid / Visual Comparison exports).

Quick smoke-tests:

- **Dataset**: stay in *Single Test*, pick **Load Single Image** on
  the left menu of the *Dataset* step and point it at any PNG/JPG to
  load a one-image dataset.
- **GPU**: open **Settings → Log Settings…** and watch the first log
  lines: the message `CUDA available: <device-name>` confirms PyTorch
  sees your GPU.
- **External tools**: open **Settings → External Applications…** to
  see which optional tools (pdflatex, poppler, nsys, kaggle) were
  auto-detected on your `PATH`.

## Troubleshooting

### GPU not detected

Ensure NVIDIA CUDA is properly installed:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### X11 forwarding issues (Docker)

On Linux, ensure `xhost +local:` was executed before running the container.

### Import errors

`python src/main.py` does **not** work — the codebase uses absolute
imports like `from simulation_engine._1_dataset_gen…`, which only
resolve when `src/` is on `sys.path`. Two ways to achieve that:

```bash
cd src && python main.py      # historical workflow (no install needed)
# OR after `pip install -e .`
python -m main                # works from any directory
```

### Architecture preview never finishes

The "Preview Architecture" popup pipes LaTeX through `pdflatex` and
then `pdf2image` (Poppler) to produce a PNG. If either is missing a
warning appears in the log. Install the missing tool with the commands
from the **Optional Tools** table above, or, in Docker, everything is
already wired.
