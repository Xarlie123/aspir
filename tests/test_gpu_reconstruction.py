"""Checks for the device-selectable PyTorch reconstruction backend.

Self-contained script in the style of tests/test_fourier_fspi.py — the
repository has no test framework:

    .venv/bin/python tests/test_gpu_reconstruction.py

Checks that need CUDA are skipped on CPU-only machines. The Cake-Cutting
ordering at 128x128 is skipped unless ASPIR_SLOW_TESTS=1: its mask generator
(not the backend) needs ~2 GB and several minutes at that size. The legacy TV
parity check alone takes a few minutes: the legacy path solves a dense
4096 x 4096 system at every one of its 100 iterations.
"""

import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch
from scipy.ndimage import gaussian_filter

from simulation_engine._2_mask_gen.mask_cal_sal import MaskCalSal
from simulation_engine._2_mask_gen.mask_fourier import MaskFourier
from simulation_engine._2_mask_gen.mask_hadamard import MaskHadamard
from simulation_engine._2_mask_gen.mask_hadamard_cake_cutting import MaskHadamardCakeCutting
from simulation_engine._2_mask_gen.mask_hadamard_walsh_paley import MaskHadamardWalshPaley
from simulation_engine._2_mask_gen.mask_scatter import MaskScatter
from simulation_engine._2_mask_gen.mask_sweep import MaskSweep
from simulation_engine._3_applicator.applicator_fista import ApplicatorFISTA
from simulation_engine._3_applicator.applicator_fourier import ApplicatorFourier
from simulation_engine._3_applicator.applicator_hadamard import ApplicatorHadamard
from simulation_engine._3_applicator.applicator_pseudoinverse import ApplicatorPseudoinverse
from simulation_engine._3_applicator.applicator_scatter import ApplicatorScatter
from simulation_engine._3_applicator.applicator_sweep import ApplicatorSweep
from simulation_engine._3_applicator.applicator_tv_norm import ApplicatorTV
from simulation_engine._3_applicator.torch_backend import device as device_mod
from simulation_engine._3_applicator.torch_backend.operators import (
    ExplicitOperator,
    build_operator,
)
from simulation_engine._3_applicator.torch_backend.solvers import gradient_2d
from simulation_engine._5_analyzer.analyzer_timing import time_reconstruction

logging.basicConfig(level=logging.ERROR)
LOG = logging.getLogger("test")

CUDA = torch.cuda.is_available()
SLOW = os.environ.get("ASPIR_SLOW_TESTS") == "1"
IMG = 64
RATIO = 0.16
M = round(RATIO * IMG * IMG)          # 655 patterns


class _Dataset:
    def __init__(self, images):
        self.name = "gpu_test"
        self.img_size = images[0].shape[0]
        self.data = images


class SkippedError(Exception):
    pass


def _image(size=IMG, seed=0):
    """Smooth test image in [0, 1] — natural-image-like, so the metrics mean something."""
    img = gaussian_filter(np.random.default_rng(seed).random((size, size)), 2.0)
    return (img - img.min()) / (img.max() - img.min())


def _psnr(reference, test):
    mse = float(np.mean((reference - test) ** 2))
    peak = float(reference.max() - reference.min())
    return float("inf") if mse == 0 else 10.0 * np.log10(peak ** 2 / mse)


def _rel(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / np.linalg.norm(np.asarray(b)))


# --- mask builders ----------------------------------------------------------
def _scatter(size=IMG, m=M):
    mask = MaskScatter(img_size=size, point_density=10.0, num_patterns=m, seed=1, logger=LOG)
    mask.generate_masks()
    return mask


def _sweep(size=IMG):
    # The sweep geometry fixes M; eight angles at unit stride is the densest
    # sweep the family offers and lands close to the target ratio.
    params = [{"angle": a, "bar_width": 1, "stride": 1}
              for a in (0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5)]
    mask = MaskSweep(img_size=size, parametros=params, logger=LOG)
    mask.generate_masks()
    return mask


def _hadamard(size=IMG, m=M, cls=MaskHadamard):
    mask = cls(size, 0, m, logger=LOG)
    mask.generate_masks()
    return mask


def _fourier(size=IMG):
    mask = MaskFourier(img_size=size, sampling_ratio=RATIO)
    mask.generate_masks()
    return mask


ALGORITHMS = [
    # name,            applicator,              mask builder
    ("ghost-imaging",   ApplicatorScatter,       _scatter),
    ("sweep-linear",    ApplicatorSweep,         _sweep),
    ("hadamard-linear", ApplicatorHadamard,      _hadamard),
    ("fourier-inverse", ApplicatorFourier,       _fourier),
    ("pseudoinverse",   ApplicatorPseudoinverse, _scatter),
    ("fista",           ApplicatorFISTA,         _scatter),
    ("tv",              ApplicatorTV,            _scatter),
]


def _objective(name, mask, image, x, lam):
    """The objective each iterative method minimises, evaluated in float64."""
    S = np.asarray(mask.masks).reshape(len(mask.masks), -1).astype(np.float64)
    y = S @ image.ravel()
    x = np.asarray(x, dtype=np.float64).ravel()
    data = 0.5 * float(np.sum((S @ x - y) ** 2))
    if name == "fista":
        return data + lam * float(np.abs(x).sum())
    g = gradient_2d(torch.as_tensor(x.reshape(1, IMG, IMG))).numpy()
    return data + lam * float(np.sqrt((g ** 2).sum(axis=1)).sum())


# ---------------------------------------------------------------------------
def test_torch_cpu_matches_legacy():
    """Each algorithm: torch on the CPU (float32) vs the legacy NumPy path."""
    image = _image()
    dataset = _Dataset([image])
    for name, cls, build in ALGORITHMS:
        mask = build()
        legacy = cls(dataset, mask, backend="legacy").process_image(0)
        app = cls(dataset, mask, device="cpu")
        torch_rec = app.process_image(0)
        psnr = _psnr(legacy, torch_rec)
        ratio = len(mask.masks) / float(IMG * IMG)
        if name in ("fista", "tv"):
            o_leg = _objective(name, mask, image, legacy, app.lam)
            o_tor = _objective(name, mask, image, torch_rec, app.lam)
            rel = abs(o_tor - o_leg) / abs(o_leg)
            assert rel < 0.01, f"{name}: objective differs by {rel * 100:.3f} %"
            assert psnr > 30.0, f"{name}: PSNR between results {psnr:.1f} dB"
            print(f"  {name:16s} M/N={ratio:.3f} op={app.last_operator_kind:9s} "
                  f"PSNR={psnr:6.1f} dB  objective legacy={o_leg:.6g} torch={o_tor:.6g} "
                  f"({rel * 100:.4f} %)")
        else:
            assert psnr > 40.0, f"{name}: PSNR between results {psnr:.1f} dB"
            print(f"  {name:16s} M/N={ratio:.3f} op={app.last_operator_kind:9s} "
                  f"PSNR={psnr:6.1f} dB")


def test_cuda_matches_cpu():
    """Same torch code on two devices.

    Direct methods: relative error < 1e-4 in float32. Iterative methods: after
    hundreds of iterations float32 alone already differs from float64 by ~1e-3
    on a single device (TV's per-pixel projection is non-smooth and amplifies
    round-off), so no device pair can be closer than that. For them the check
    is split: float64 must agree to 1e-8 (same code, same result), and in
    float32 the device may add no error beyond float32's own.
    """
    if not CUDA:
        raise SkippedError("no CUDA device")
    image = _image(seed=1)
    dataset = _Dataset([image])
    for name, cls, build in ALGORITHMS:
        mask = build()
        gpu_app = cls(dataset, mask, device="cuda")
        gpu = gpu_app.process_image(0)
        assert gpu_app.reconstruction_device == "cuda"
        cpu = cls(dataset, mask, device="cpu").process_image(0)
        rel = _rel(gpu, cpu)
        if name in ("fista", "tv"):
            c64 = cls(dataset, mask, device="cpu", dtype="float64").process_image(0)
            g64 = cls(dataset, mask, device="cuda", dtype="float64").process_image(0)
            e64 = _rel(g64, c64)
            intrinsic = _rel(cpu, c64)
            assert e64 < 1e-8, f"{name}: float64 CUDA vs CPU {e64:.2e}"
            assert rel < max(1e-4, 2.0 * intrinsic), (
                f"{name}: float32 CUDA vs CPU {rel:.2e} exceeds float32's own {intrinsic:.2e}")
            print(f"  {name:16s} float32 {rel:.2e} (float32 vs float64 on CPU: {intrinsic:.2e}); "
                  f"float64 {e64:.2e}")
        else:
            assert rel < 1e-4, f"{name}: CUDA vs CPU relative error {rel:.2e}"
            print(f"  {name:16s} relative error {rel:.2e}")


def test_fwht_path_equals_explicit_hadamard_path():
    """Matrix-free FWHT == explicit matrix, for all four Hadamard orderings."""
    dev = torch.device("cuda" if CUDA else "cpu")
    orderings = [
        ("natural", lambda n: _hadamard(n, n * n // 4, MaskHadamard)),
        ("walsh-paley", lambda n: _hadamard(n, n * n // 4, MaskHadamardWalshPaley)),
        ("cake-cutting", lambda n: _hadamard(n, n * n // 4, MaskHadamardCakeCutting)),
        ("cal-sal", lambda n: MaskCalSal(img_size=n, logger=LOG)),
    ]
    for size in (64, 128):
        image = _image(size, seed=2)
        dataset = _Dataset([image])
        for label, build in orderings:
            if label == "cake-cutting" and size == 128 and not SLOW:
                print(f"  {size}x{size} {label:12s} skipped (set ASPIR_SLOW_TESTS=1)")
                continue
            mask = build(size)
            if mask.masks is None:
                mask.generate_masks()
            m = min(len(mask.masks), size * size // 4)
            op_f = build_operator(mask, 0, m, dev, torch.float32, mode="fwht")
            op_e = build_operator(mask, 0, m, dev, torch.float32, mode="explicit")
            X = torch.as_tensor(image.reshape(1, -1), dtype=torch.float32, device=dev)
            Y = op_e.forward(X)
            e_fwd = _rel(op_f.forward(X).cpu(), Y.cpu())
            e_adj = _rel(op_f.adjoint(Y).cpu(), op_e.adjoint(Y).cpu())
            rec_f = ApplicatorHadamard(dataset, mask, device=dev.type, operator="fwht").apply_mask_range(0, m, 0)
            rec_e = ApplicatorHadamard(dataset, mask, device=dev.type, operator="explicit").apply_mask_range(0, m, 0)
            e_rec = _rel(rec_f, rec_e)
            assert op_f.kind == "fwht" and op_e.kind.startswith("explicit")
            for what, err in (("forward", e_fwd), ("adjoint", e_adj), ("reconstruction", e_rec)):
                assert err < 1e-5, f"{size} {label} {what}: {err:.2e}"
            print(f"  {size}x{size} {label:12s} fwd={e_fwd:.1e} adj={e_adj:.1e} recon={e_rec:.1e}")
            del mask, op_e


def test_chunked_matmul_equals_unchunked():
    """Streaming the matrix in row blocks gives the same products."""
    g = torch.Generator().manual_seed(0)
    S = torch.rand((700, 4096), generator=g)
    X = torch.rand((3, 4096), generator=g)
    Y = torch.rand((3, 700), generator=g)
    devices = [torch.device("cpu")] + ([torch.device("cuda")] if CUDA else [])
    for dev in devices:
        whole = ExplicitOperator(S, dev, torch.float32)
        chunk = ExplicitOperator(S, dev, torch.float32, max_device_bytes=64 * 4096 * 4)
        assert whole.kind == "explicit" and chunk.kind == "explicit-chunked", (whole.kind, chunk.kind)
        Xd, Yd = X.to(dev), Y.to(dev)
        g1, r1 = whole.normal(Xd, Yd)
        g2, r2 = chunk.normal(Xd, Yd)
        errs = {
            "forward": _rel(chunk.forward(Xd).cpu(), whole.forward(Xd).cpu()),
            "adjoint": _rel(chunk.adjoint(Yd).cpu(), whole.adjoint(Yd).cpu()),
            "normal": _rel(g2.cpu(), g1.cpu()),
            "residual": _rel(r2.cpu(), r1.cpu()),
        }
        for what, err in errs.items():
            assert err < 1e-5, f"{dev} {what}: {err:.2e}"
        print(f"  {dev.type:4s} {chunk.rows_per_chunk} rows/block: "
              + " ".join(f"{k}={v:.1e}" for k, v in errs.items()))

    # End to end: a capped applicator streams and reconstructs the same image.
    image = _image(seed=3)
    dataset = _Dataset([image])
    mask = _scatter()
    dev = "cuda" if CUDA else "cpu"
    free = ApplicatorFISTA(dataset, mask, device=dev)
    capped = ApplicatorFISTA(dataset, mask, device=dev, max_device_bytes=128 * IMG * IMG * 4)
    a, b = free.process_image(0), capped.process_image(0)
    assert capped.last_operator_kind == "explicit-chunked", capped.last_operator_kind
    err = _rel(b, a)
    assert err < 1e-4, f"FISTA chunked vs resident: {err:.2e}"
    print(f"  FISTA through a capped applicator ({dev}): relative error {err:.1e}")


def test_out_of_memory_falls_back_to_chunking():
    """An allocation that fails despite the estimate degrades to streaming."""
    if not CUDA:
        raise SkippedError("no CUDA device")
    S = torch.rand((500, 4096))
    real_to = torch.Tensor.to

    def failing_to(self, *args, **kwargs):
        target = args[0] if args else kwargs.get("device")
        if self.numel() == S.numel() and str(target).startswith("cuda"):
            raise torch.cuda.OutOfMemoryError("simulated out of memory")
        return real_to(self, *args, **kwargs)

    torch.Tensor.to = failing_to
    try:
        op = ExplicitOperator(S, torch.device("cuda"), torch.float32,
                              max_device_bytes=10 * 1024 ** 3)
    finally:
        torch.Tensor.to = real_to
    assert op.kind == "explicit-chunked", op.kind
    X = torch.rand((2, 4096))
    err = _rel(op.forward(X.cuda()).cpu(), X @ S.T)
    assert err < 1e-5, err
    print(f"  simulated OOM -> {op.kind}, forward relative error {err:.1e}")


def test_device_resolution():
    """'auto' picks CUDA when present; CPU-only machines resolve silently to CPU."""
    auto = device_mod.resolve_device("auto")
    assert auto.type == ("cuda" if CUDA else "cpu"), auto
    real = torch.cuda.is_available
    torch.cuda.is_available = lambda: False
    try:
        assert device_mod.resolve_device("auto").type == "cpu"
        assert device_mod.resolve_device("cuda").type == "cpu"   # falls back, no error
    finally:
        torch.cuda.is_available = real
    try:
        device_mod.resolve_device("tpu")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown device name must be rejected")
    print(f"  auto -> {auto.type}; without CUDA both 'auto' and 'cuda' -> cpu")


def test_timing_records_device():
    """The synchronised timing protocol reports where the reconstruction ran."""
    image = _image(seed=4)
    dataset = _Dataset([image, image])
    mask = _hadamard()
    for dev in ["cpu"] + (["cuda"] if CUDA else []):
        app = ApplicatorHadamard(dataset, mask, device=dev)
        t = time_reconstruction(app, [0, 1], warmup_runs=1, repeats=2)
        assert t["device"] == dev and t["samples"] == 4 and t["backend"] == "torch"
        assert t["operator"] == "fwht" and t["precision"] == "float32"
        print(f"  {dev:4s}: {t['mean_ms']:.3f} +- {t['std_ms']:.3f} ms, operator={t['operator']}")
    legacy = time_reconstruction(ApplicatorHadamard(dataset, mask, backend="legacy"), [0], 1, 1)
    assert legacy["device"] == "cpu" and legacy["backend"] == "legacy"


def test_batch_config_round_trip():
    """reconstruction_device survives to_dict / from_dict and defaults to 'auto'."""
    from ui.custom_widgets.batch_test.test_config_model import TestConfiguration
    assert TestConfiguration().reconstruction_device == "auto"
    cfg = TestConfiguration(reconstruction_device="cuda")
    back = TestConfiguration.from_dict(cfg.to_dict())
    assert back.reconstruction_device == "cuda"
    legacy = cfg.to_dict()
    legacy.pop("reconstruction_device")
    assert TestConfiguration.from_dict(legacy).reconstruction_device == "auto"
    print("  round trip preserved 'cuda'; configs without the key load as 'auto'")


if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    failures = skipped = 0
    for fn in tests:
        print(f"\n{fn.__name__}")
        try:
            fn()
            print("  PASS")
        except SkippedError as exc:
            skipped += 1
            print(f"  SKIP: {exc}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL: {exc}")
    ran = len(tests) - skipped
    print(f"\n{ran - failures}/{ran} passed, {skipped} skipped (CUDA={'yes' if CUDA else 'no'})")
    sys.exit(1 if failures else 0)
