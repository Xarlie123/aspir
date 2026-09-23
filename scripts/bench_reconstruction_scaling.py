#!/usr/bin/env python
"""Reconstruction scaling benchmark: the seven algorithms, CPU vs CUDA.

Times every reconstruction algorithm at several image sizes and sampling
ratios on the requested devices, with the same GPU-synchronised protocol the
platform uses for the NN stage, and writes a CSV plus a Markdown table.

Examples
--------
Desktop, both devices, plus the legacy NumPy path at 64x64 for comparison:

    python scripts/bench_reconstruction_scaling.py --device cpu cuda --legacy

Jetson:

    python scripts/bench_reconstruction_scaling.py --device cuda --sizes 64 128 256 --out results/

Protocol (per cell = size x ratio x algorithm x mask family x device)
---------------------------------------------------------------------
* One untimed call first: it absorbs one-off work (operator construction,
  cached pseudoinverse, Lipschitz constant, TV factorisation) and is reported
  separately as ``setup_ms``. The legacy path caches nothing, so it gets none.
* Per-image latency: ``synchronize(); t0; reconstruct one image; synchronize()``
  over up to ``--images`` images, within ``--budget`` seconds per cell (at least
  3 images unless one image alone exceeds the budget). Each call includes
  simulating the single-pixel measurements, as in the platform.
* Batched throughput (torch only): all measured images in one call, divided by
  the batch size. For a streamed matrix this is where batching pays: one pass
  over the matrix serves the whole batch.
* Peak device memory: ``torch.cuda.max_memory_allocated`` since the cell began.

Mask families: scatter-like random binary patterns (exact ``int(0.10 N)`` ones
per pattern, as ``MaskScatter``, generated in float32 so large sizes fit) for
the explicit path; Sylvester Hadamard rows 0..M-1 given as metadata for the
matrix-free FWHT path; grayscale four-step Fourier metadata for the FFT path;
and the real ``MaskSweep`` (eight angles, unit stride, unit bar) whose geometry
fixes M, so it runs at one ratio per size.
"""

import argparse
import csv
import gc
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import logging  # noqa: E402

import torch  # noqa: E402

logging.basicConfig(level=logging.WARNING)
LOG = logging.getLogger("bench")

from simulation_engine._2_mask_gen.mask_fourier import PHASE_SETS, MaskFourier  # noqa: E402
from simulation_engine._2_mask_gen.mask_sweep import MaskSweep  # noqa: E402
from simulation_engine._3_applicator.applicator_fista import ApplicatorFISTA  # noqa: E402
from simulation_engine._3_applicator.applicator_fourier import ApplicatorFourier  # noqa: E402
from simulation_engine._3_applicator.applicator_hadamard import ApplicatorHadamard  # noqa: E402
from simulation_engine._3_applicator.applicator_pseudoinverse import (  # noqa: E402
    ApplicatorPseudoinverse,
)
from simulation_engine._3_applicator.applicator_scatter import ApplicatorScatter  # noqa: E402
from simulation_engine._3_applicator.applicator_sweep import ApplicatorSweep  # noqa: E402
from simulation_engine._3_applicator.applicator_tv_norm import ApplicatorTV  # noqa: E402
from simulation_engine._3_applicator.torch_backend.device import available_bytes  # noqa: E402

CONFIGS = [
    # algorithm,         applicator,              mask family
    ("ghost-imaging",   ApplicatorScatter,       "scatter"),
    ("sweep-linear",    ApplicatorSweep,         "sweep"),
    ("hadamard-linear", ApplicatorHadamard,      "hadamard"),
    ("fourier-inverse", ApplicatorFourier,       "fourier"),
    ("pseudoinverse",   ApplicatorPseudoinverse, "hadamard"),
    ("pseudoinverse",   ApplicatorPseudoinverse, "scatter"),
    ("fista",           ApplicatorFISTA,         "hadamard"),
    ("fista",           ApplicatorFISTA,         "scatter"),
    ("tv",              ApplicatorTV,            "hadamard"),
    ("tv",              ApplicatorTV,            "scatter"),
]


class NotRunError(Exception):
    """A cell that cannot run on this machine; the message says why."""


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
def environment():
    env = {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "host": platform.machine(),
        "cpu": platform.processor() or platform.machine(),
        "ram_gb": round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024 ** 3, 1),
    }
    if env["cuda_available"]:
        env["gpu"] = torch.cuda.get_device_name(0)
        env["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1)
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
        env["driver"] = out.stdout.strip().splitlines()[0] if out.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired, IndexError):
        env["driver"] = None
    tegra = Path("/etc/nv_tegra_release")
    if env["driver"] is None and tegra.exists():
        env["driver"] = "L4T " + tegra.read_text().split(",")[0].replace("# ", "").strip()
    return env


def header_line(env):
    gpu = f"{env.get('gpu')} ({env.get('gpu_mem_gb')} GB)" if env["cuda_available"] else "no CUDA"
    return (f"torch {env['torch']} | CUDA runtime {env['cuda_runtime']} | {gpu} | "
            f"driver {env['driver'] or 'n/a'} | {env['host']} host, {env['ram_gb']} GB RAM")


# ---------------------------------------------------------------------------
# Data and masks
# ---------------------------------------------------------------------------
def load_images(size, n, source):
    """``(dataset, description)``: CelebA if present locally, else synthetic."""
    celeba = REPO / "datasets" / "tmp_downloaded_datasets" / "celebrities"
    if source in ("auto", "celeba") and (celeba / "kaggle_celeba_small").exists():
        from simulation_engine._1_dataset_gen.DatasetFromCelebrities import DatasetFromCelebrities
        ds = DatasetFromCelebrities(name="bench", img_size=size, num_images=n, seed=42,
                                    download_dir=str(celeba), logger=LOG)
        ds.load_data()
        imgs = [np.asarray(a, dtype=np.float32) for a in ds.data[:n]]
        desc = "CelebA (64x64 native, resized to each larger size)"
    else:
        if source == "celeba":
            raise SystemExit("CelebA requested but not found under datasets/")
        from scipy.ndimage import gaussian_filter
        rng = np.random.default_rng(42)
        imgs = [gaussian_filter(rng.random((size, size)), size / 32).astype(np.float32)
                for _ in range(n)]
        desc = "synthetic smooth images"
    imgs = [(a - a.min()) / (a.max() - a.min() + 1e-12) for a in imgs]
    return SimpleNamespace(name="bench", img_size=size, data=imgs), desc


def _fits_host(nbytes, fraction=0.6):
    return nbytes <= fraction * available_bytes(torch.device("cpu"))


def scatter_mask(size, m, seed=1):
    n = size * size
    nbytes = m * n * 4
    if not _fits_host(nbytes):
        raise NotRunError(f"dense matrix {nbytes / 1e9:.1f} GB (float32) does not fit host RAM")
    k = int(n * 10.0 / 100.0)
    rng = np.random.default_rng(seed)
    masks = np.zeros((m, n), dtype=np.float32)
    for s in range(0, m, 256):
        rows = min(256, m - s)
        pos = np.argpartition(rng.random((rows, n), dtype=np.float32), k, axis=1)[:, :k]
        np.put_along_axis(masks[s:s + rows], pos, 1.0, axis=1)
    return SimpleNamespace(img_size=size, masks=masks.reshape(m, size, size))


def hadamard_mask(size, m, explicit=False):
    mask = SimpleNamespace(img_size=size, hadamard_indices=np.arange(m, dtype=np.int64),
                           hadamard_signs=np.ones(m), masks=None)
    if explicit:   # the legacy path needs the patterns themselves
        import scipy.linalg
        mask.masks = scipy.linalg.hadamard(size * size)[:m].astype(np.float32).reshape(m, size, size)
        mask.hadamard_indices = None
    return mask


def fourier_mask(size, ratio, explicit=False):
    mf = MaskFourier(img_size=size, sampling_ratio=ratio, phase_steps=4)
    if explicit:
        mf.generate_masks()
        return mf
    n_coeff, _ = mf._resolve_counts()
    reps = mf._order_representatives(mf._half_plane_representatives())[:n_coeff]
    steps = PHASE_SETS[4]
    return SimpleNamespace(
        img_size=size, masks=None, phase_steps=4, binarization="none",
        freq_indices=np.repeat(reps, len(steps), axis=0).astype(np.int32),
        phases=np.tile(np.asarray(steps, dtype=np.float64), n_coeff),
        coeff_ids=np.repeat(np.arange(n_coeff, dtype=np.int32), len(steps)),
    )


def sweep_mask(size):
    params = [{"angle": a, "bar_width": 1, "stride": 1}
              for a in (0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5)]
    ms = MaskSweep(img_size=size, parametros=params, logger=LOG)
    ms.generate_masks()
    masks = np.asarray(ms.masks, dtype=np.float32)
    del ms
    return SimpleNamespace(img_size=size, masks=masks)


def build_mask(family, size, ratio, legacy):
    m = round(ratio * size * size)
    if family == "scatter":
        return scatter_mask(size, m)
    if family == "hadamard":
        return hadamard_mask(size, m, explicit=legacy)
    if family == "fourier":
        return fourier_mask(size, ratio, explicit=legacy)
    if family == "sweep":
        return sweep_mask(size)
    raise ValueError(family)


def n_patterns(mask):
    if getattr(mask, "masks", None) is not None:
        return len(mask.masks)
    if getattr(mask, "hadamard_indices", None) is not None:
        return len(mask.hadamard_indices)
    return len(mask.phases)


# ---------------------------------------------------------------------------
# One cell
# ---------------------------------------------------------------------------
def _sync(dev):
    if dev == "cuda":
        torch.cuda.synchronize()


def run_cell(size, ratio, algorithm, cls, family, device, dataset, args):
    legacy = device == "legacy"
    dev = "cpu" if legacy else device
    row = {"size": size, "ratio_requested": ratio, "algorithm": algorithm, "masks": family,
           "device": device, "backend": "legacy" if legacy else "torch",
           "precision": "float64" if legacy else args.dtype}

    # Skip what cannot finish: TV's data-term prox falls back to inner CG on a
    # dense matrix, and when that matrix has to be streamed every CG step is a
    # full pass over PCIe - hours per image.
    m_req = round(ratio * size * size)
    if (algorithm == "tv" and family == "scatter" and dev == "cuda"
            and m_req * size * size * 4 > 0.5 * available_bytes(torch.device("cuda"))):
        raise NotRunError("TV with inner CG on a streamed matrix: estimated hours per image")

    if family == "sweep" and size > 256:
        raise NotRunError("MaskSweep materialises float64 patterns: ~12 GB at 512x512")
    if algorithm == "tv" and family == "scatter" and size >= 256:
        raise NotRunError("TV with inner CG on a >=1.4 GB dense matrix: estimated >15 min per image")

    mask = build_mask(family, size, ratio, legacy)
    m = n_patterns(mask)
    row.update({"M": m, "ratio_actual": round(m / float(size * size), 4)})

    kwargs = {"backend": "legacy"} if legacy else {"device": dev, "dtype": args.dtype}
    app = cls(dataset, mask, **kwargs)
    if dev == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    def one(idx):
        _sync(dev)
        t0 = time.perf_counter()
        app.apply_mask_range(0, m, idx)
        _sync(dev)
        return (time.perf_counter() - t0) * 1000.0

    if not legacy:
        row["setup_ms"] = round(one(0), 3)

    n_img = len(dataset.data)
    lat = [one(0)]
    if lat[0] / 1000.0 > args.budget:
        target = 1
    else:
        target = max(min(3, n_img), min(n_img, int(args.budget * 1000.0 // max(lat[0], 1e-6))))
    for idx in range(1, target):
        lat.append(one(idx))
    arr = np.asarray(lat)
    row.update({"n_images": len(lat), "latency_mean_ms": round(float(arr.mean()), 4),
                "latency_std_ms": round(float(arr.std()), 4) if len(lat) > 1 else None})

    if not legacy:
        row["operator"] = app.last_operator_kind
        info = getattr(app, "last_solver_info", None)
        if info:
            row["solver"] = info.get("prox_f") or info.get("solver")
        # Batched throughput over the same images, when affordable.
        if arr.mean() * len(lat) / 1000.0 <= 2 * args.budget and len(lat) > 1:
            _sync(dev)
            t0 = time.perf_counter()
            app._torch_reconstruct(0, m, list(range(len(lat))))
            _sync(dev)
            row["batch_ms_per_image"] = round((time.perf_counter() - t0) * 1000.0 / len(lat), 4)
            row["batch_size"] = len(lat)
    else:
        row["operator"] = "explicit (NumPy)"

    if dev == "cuda":
        row["peak_device_mb"] = round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1)
    return row


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def fmt_ms(row):
    if row is None:
        return "—"
    if row.get("status"):
        return f"n/a ({row['status']})"
    s = f"{row['latency_mean_ms']:.3g}"
    if row.get("latency_std_ms") is not None:
        s += f" ± {row['latency_std_ms']:.2g}"
    return s + f" (n={row['n_images']})"


def markdown(rows, env, data_desc):
    lines = [f"Environment: {header_line(env)}", "",
             f"Images: {data_desc}. Precision: torch path {rows[0]['precision'] if rows else ''}, "
             "legacy float64. Times are per image (ms), synchronised, including the simulated "
             "measurement.", ""]
    keyed = {}
    for r in rows:
        keyed[(r["size"], r["algorithm"], r["masks"], r["ratio_requested"], r["device"])] = r
    sizes = sorted({r["size"] for r in rows})
    for size in sizes:
        lines += [f"### {size}x{size}", "",
                  "| Algorithm | Masks | M/N | Operator | CPU (torch) | CUDA | Speed-up | "
                  "CUDA batched | CUDA peak MB | CPU legacy |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        combos = sorted({(r["algorithm"], r["masks"], r["ratio_requested"]) for r in rows if r["size"] == size},
                        key=lambda k: ([c[0] for c in CONFIGS].index(k[0]), k[1], k[2]))
        for alg, fam, ratio in combos:
            cpu = keyed.get((size, alg, fam, ratio, "cpu"))
            gpu = keyed.get((size, alg, fam, ratio, "cuda"))
            leg = keyed.get((size, alg, fam, ratio, "legacy"))
            ref = gpu or cpu or leg
            speed = "—"
            if cpu and gpu and not cpu.get("status") and not gpu.get("status"):
                speed = f"{cpu['latency_mean_ms'] / gpu['latency_mean_ms']:.1f}x"
            ratio_txt = f"{ref.get('ratio_actual', ratio):.3f}" if ref and ref.get("ratio_actual") else f"{ratio}"
            op = (gpu or cpu or {}).get("operator", "—")
            if (gpu or cpu or {}).get("solver") and alg in ("pseudoinverse", "tv"):
                op += f" / {(gpu or cpu)['solver']}"
            batched = f"{gpu['batch_ms_per_image']:.3g} (B={gpu['batch_size']})" \
                if gpu and gpu.get("batch_ms_per_image") else "—"
            peak = f"{gpu['peak_device_mb']:.0f}" if gpu and gpu.get("peak_device_mb") else "—"
            lines.append(f"| {alg} | {fam} | {ratio_txt} | {op} | {fmt_ms(cpu)} | {fmt_ms(gpu)} | "
                         f"{speed} | {batched} | {peak} | {fmt_ms(leg) if leg else '—'} |")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--device", nargs="+", default=None, choices=["cpu", "cuda"],
                    help="devices to benchmark (default: cpu, plus cuda if available)")
    ap.add_argument("--sizes", nargs="+", type=int, default=[64, 128, 256])
    ap.add_argument("--ratios", nargs="+", type=float, default=[0.08, 0.16, 0.32])
    ap.add_argument("--algorithms", nargs="+", default=None,
                    help="subset of: " + " ".join(sorted({c[0] for c in CONFIGS})))
    ap.add_argument("--images", type=int, default=50)
    ap.add_argument("--dataset", choices=["auto", "celeba", "synthetic"], default="auto")
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--budget", type=float, default=90.0, help="seconds per cell (soft)")
    ap.add_argument("--legacy", action="store_true",
                    help="also time the legacy NumPy/pylops path (CPU) at --legacy-sizes")
    ap.add_argument("--legacy-sizes", nargs="+", type=int, default=[64])
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    devices = args.device or (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
    if "cuda" in devices and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is not available")
    env = environment()
    print(header_line(env), flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = out / f"reconstruction_scaling_{stamp}.csv"
    rows, data_desc = [], ""
    fields = ["size", "ratio_requested", "M", "ratio_actual", "algorithm", "masks", "device",
              "backend", "precision", "operator", "solver", "n_images", "latency_mean_ms",
              "latency_std_ms", "setup_ms", "batch_ms_per_image", "batch_size",
              "peak_device_mb", "status"]

    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for size in args.sizes:
            dataset, data_desc = load_images(size, args.images, args.dataset)
            devs = list(devices) + (["legacy"] if args.legacy and size in args.legacy_sizes else [])
            for algorithm, cls, family in CONFIGS:
                if args.algorithms and algorithm not in args.algorithms:
                    continue
                ratios = args.ratios[:1] if family == "sweep" else args.ratios
                for ratio in ratios:
                    for dev in devs:
                        t0 = time.time()
                        try:
                            row = run_cell(size, ratio, algorithm, cls, family, dev, dataset, args)
                        except NotRunError as e:
                            row = {"size": size, "ratio_requested": ratio, "algorithm": algorithm,
                                   "masks": family, "device": dev, "status": str(e)}
                        except (torch.cuda.OutOfMemoryError, MemoryError) as e:
                            row = {"size": size, "ratio_requested": ratio, "algorithm": algorithm,
                                   "masks": family, "device": dev, "status": f"out of memory: {e}"[:120]}
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        rows.append(row)
                        writer.writerow(row)
                        fh.flush()
                        lat = row.get("latency_mean_ms")
                        print(f"{size:4d} {algorithm:16s} {family:8s} r={ratio:<5} {dev:6s} "
                              + (f"{lat:10.3f} ms (n={row['n_images']}) op={row.get('operator')} "
                                 f"peak={row.get('peak_device_mb', '-')} MB" if lat is not None
                                 else f"NOT RUN: {row.get('status')}")
                              + f"  [{time.time() - t0:.0f}s]", flush=True)
            del dataset
            gc.collect()

    md = markdown(rows, env, data_desc)
    (out / f"reconstruction_scaling_{stamp}.md").write_text(md)
    (out / f"reconstruction_scaling_{stamp}_env.json").write_text(json.dumps(env, indent=2))
    print(f"\nWrote {csv_path} and the matching .md / _env.json")


if __name__ == "__main__":
    main()
