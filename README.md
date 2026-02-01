# ASPIR: A Single-Pixel Imaging Research Platform

PyQt5 application for **Single Pixel Imaging (SPI)** simulation and analysis. Implements a complete computational imaging pipeline for infrared beam profiling using mask patterns, classical reconstruction algorithms, and neural network post-processing.

## Quick Start

### Local Development

```bash
# 1. Clone and enter
git clone https://github.com/Xarlie123/aspir.git
cd aspir

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements_laptop.txt

# 4. Run (working directory MUST be src/)
cd src
python main.py
```

### Docker

```bash
# Build (only needed once, or after changing dependencies)
docker build -f docker/Dockerfile -t aspir .
```

**Linux:**
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

**Windows (PowerShell + VcXsrv/X410):**
```powershell
$env:DISPLAY="host.docker.internal:0.0"
docker run --rm -it `
  --gpus all `
  -e DISPLAY=$env:DISPLAY `
  -v "${PWD}:/app" `
  aspir
```

> **Notes:**
> - Code changes are reflected immediately without rebuilding (mounted at `/app`)
> - **Linux only:** RAPL CPU energy profiling requires: `sudo chmod -R a+r /sys/class/powercap/intel-rapl/`
> - **Windows:** Requires Docker Desktop with WSL2 backend and an X server (VcXsrv, X410, etc.)

## Directory Structure

```
aspir/
├── src/                          # Source code (working directory)
│   ├── main.py                   # Application entry point
│   ├── simulation_engine/        # Core simulation pipeline
│   │   ├── _1_dataset_gen/       # Dataset generators
│   │   ├── _2_mask_gen/          # Mask pattern generators
│   │   ├── _3_applicator/        # Reconstruction algorithms
│   │   ├── _4_postprocessor/     # Neural network models
│   │   ├── _5_analyzer/          # Metrics & profiling
│   │   └── _6_pipeline/          # Batch execution
│   └── ui/                       # PyQt5 GUI
│       ├── custom_widgets/       # Reusable UI components
│       ├── modes/                # Single Test, Batch Test, Batch Reports
│       └── utils/                # Config, logging, file handling
├── datasets/                     # Downloaded and generated datasets
├── settings/                     # Application settings (log config, etc.)
├── docker/                       # Docker configuration
├── requirements_laptop.txt       # Python dependencies (local dev)
└── requirements_docker.txt       # Python dependencies (Docker)
```

## Requirements

- Python 3.10+
- NVIDIA GPU + CUDA 12.4 (for GPU acceleration)
- PyQt5, PyTorch, NumPy, SciPy, pylops, pyproximal

## External Applications (Optional)

Configure via **Settings → External Applications**:
- **pdflatex**: DNN architecture preview diagrams
- **nsys**: NVIDIA Nsight Systems profiling
- **kaggle**: Dataset downloads from Kaggle

## License

Research project 
