"""CPU thread pools of the measuring process: what they are, and setting them for a measurement.

PyTorch parallelises CPU operators over its intra-op pool (``torch.set_num_threads``). The BLAS library
it calls may keep a second pool of its own:

* x86 wheels link MKL, which follows torch's setting: one pool, never oversubscribed by itself.
* The Jetson wheels (torch 2.8 from Jetson AI Lab) call the system OpenBLAS built with pthreads. Its pool
  defaults to one thread per core, and ``torch.set_num_threads`` does not reach it. With torch's default
  8 threads on top, an Orin NX runs a 64 x 64 U-Net at batch 1 in 66 ms; with 8 x 4 threads in 33 ms, with
  8 x 2 in 24 ms, with OpenBLAS 8 x torch 1 in 11 ms and with 1 x 1 in 16.5 ms (R1.3e, clocks pinned).

``cpu_threads(torch_threads, blas_threads)`` sets either pool for the duration of a block (``None`` keeps
the library default) and restores both afterwards. BLAS pools are found and set through threadpoolctl,
which calls ``openblas_set_num_threads`` (MKL's, BLIS's...) in the libraries the process has loaded, at
run time; the BLAS setting applies to every BLAS library loaded, including numpy's own copy.
``cpu_threading_state`` reports what is in effect: the values read back after setting them, never the
requested ones.

A pass is *oversubscribed* when torch's intra-op pool and a separate BLAS pool that torch uses both run
more than one thread, or when either asks for more threads than the process may use. numpy's and scipy's
vendored OpenBLAS copies are recorded but not counted: torch does not call them.
"""
from __future__ import annotations

import logging
import os
import re
from contextlib import contextmanager
from typing import Any, Iterator, Optional

try:
    import threadpoolctl
except ImportError:  # recorded as unavailable; a BLAS setting then fails loudly
    threadpoolctl = None

_ENV_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
# Wheels that vendor their own OpenBLAS; torch does not call these copies.
_VENDORED = ("numpy.libs", "scipy.libs", "scipy_openblas")


def usable_cores() -> int:
    """Cores this process may run on (its affinity), or the machine's count."""
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def torch_blas() -> Optional[str]:
    """The BLAS torch was built against (``BLAS_INFO`` of its build config): "mkl", "open", ..."""
    try:
        import torch
        m = re.search(r"BLAS_INFO=(\w+)", torch.__config__.show())
    except Exception:
        return None
    return m.group(1) if m else None


def blas_pools() -> list[dict[str, Any]]:
    """Every BLAS and OpenMP pool threadpoolctl can see in this process."""
    if threadpoolctl is None:
        return []
    pools = []
    for p in threadpoolctl.threadpool_info():
        path = p.get("filepath") or ""
        pools.append({
            "user_api": p.get("user_api"),
            "library": p.get("internal_api"),
            "version": p.get("version"),
            "threads": p.get("num_threads"),
            "threading_layer": p.get("threading_layer"),
            "path": path,
            "used_by_torch": p.get("user_api") == "blas" and not any(v in path for v in _VENDORED),
        })
    return pools


def oversubscription(torch_threads: int, pools: list[dict[str, Any]], cores: int,
                     blas: Optional[str]) -> Optional[str]:
    """Why a CPU pass with these pools is oversubscribed, or ``None`` if it is not (see module docstring)."""
    if torch_threads > cores:
        return f"torch uses {torch_threads} threads on {cores} usable cores"
    for p in pools:
        if not p["used_by_torch"] or not p["threads"]:
            continue
        if p["threads"] > cores:
            return f"{p['library']} uses {p['threads']} threads on {cores} usable cores"
        separate = blas == "open" and p["library"] == "openblas" and p["threading_layer"] == "pthreads"
        if separate and torch_threads > 1 and p["threads"] > 1:
            return (f"torch's {torch_threads} intra-op threads and the {p['threads']}-thread pthreads pool of "
                    f"{p['library']} {p['version'] or ''} run on top of each other".replace("  ", " "))
    return None


def cpu_threading_state() -> dict[str, Any]:
    """What is in effect now: torch's pools, the BLAS pools, the variables at start-up, the cores."""
    try:
        import torch
        intra, interop = torch.get_num_threads(), torch.get_num_interop_threads()
    except Exception:
        intra = interop = None
    pools, cores, blas = blas_pools(), usable_cores(), torch_blas()
    reason = oversubscription(intra or 1, pools, cores, blas)
    return {
        "torch_intra_op_threads": intra,
        "torch_inter_op_threads": interop,
        "torch_blas": blas,
        "blas_pools": pools,
        "threadpoolctl": getattr(threadpoolctl, "__version__", None),
        "environment": {k: os.environ.get(k) for k in _ENV_VARS},
        "usable_cores": cores,
        "oversubscribed": reason is not None,
        "oversubscribed_reason": reason,
    }


@contextmanager
def cpu_threads_for(config, on_cpu: bool = True,
                    logger: Optional[logging.Logger] = None) -> Iterator[Optional[dict[str, Any]]]:
    """``cpu_threads`` with a configuration's ``cpu_torch_threads`` / ``cpu_blas_threads``, for a CPU pass.

    For a pass that does not run on the CPU it changes nothing and yields ``None``.
    """
    if not on_cpu:
        yield None
        return
    with cpu_threads(getattr(config, "cpu_torch_threads", None), getattr(config, "cpu_blas_threads", None),
                     logger=logger) as state:
        yield state


def threads_fields(state: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Report fields for a CPU pass's thread state (empty when the pass did not run on the CPU)."""
    if state is None:
        return {}
    return {"cpu_threads": state, "cpu_oversubscribed": state["oversubscribed"]}


@contextmanager
def cpu_threads(torch_threads: Optional[int] = None, blas_threads: Optional[int] = None,
                logger: Optional[logging.Logger] = None) -> Iterator[dict[str, Any]]:
    """Run a block with these CPU thread counts (``None`` = library default) and restore them after.

    Yields the state in effect inside the block (:func:`cpu_threading_state`, plus ``requested``). A BLAS
    count that cannot be applied (threadpoolctl missing, or no BLAS pool found) raises, so that a report
    never claims a setting that did not happen. An oversubscribed state is logged as a warning.
    """
    import torch

    log = logger or logging.getLogger(__name__)
    previous = torch.get_num_threads()
    limiter = None
    try:
        if torch_threads:
            torch.set_num_threads(int(torch_threads))
        if blas_threads:
            if threadpoolctl is None:
                raise RuntimeError("a BLAS thread count was requested but threadpoolctl is not installed")
            if not any(p["user_api"] == "blas" for p in blas_pools()):
                raise RuntimeError("a BLAS thread count was requested but no BLAS pool is visible")
            limiter = threadpoolctl.threadpool_limits(limits=int(blas_threads), user_api="blas")
        state = cpu_threading_state()
        state["requested"] = {"torch_threads": torch_threads, "blas_threads": blas_threads}
        if state["oversubscribed"]:
            log.warning("CPU pass oversubscribed: %s. Set the CPU threads (torch and/or BLAS) to 1 "
                        "for one of the two pools", state["oversubscribed_reason"])
        yield state
    finally:
        if limiter is not None:
            limiter.restore_original_limits()
        torch.set_num_threads(previous)
