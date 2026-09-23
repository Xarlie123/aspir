"""PyTorch reconstruction backend shared by every applicator.

One implementation per algorithm, written against a linear sensing operator,
so a CPU-vs-GPU comparison is the same code on two devices. Which operator is
used (explicit resident, explicit chunked, fast Walsh-Hadamard, FFT) is chosen
per mask set and is transparent to the solvers.
"""

from simulation_engine._3_applicator.torch_backend.device import (
    DEVICE_CHOICES,
    DTYPE_CHOICES,
    describe_device,
    resolve_device,
    resolve_dtype,
)
from simulation_engine._3_applicator.torch_backend.operators import (
    ExplicitOperator,
    FourierOperator,
    HadamardOperator,
    SensingOperator,
    build_operator,
)

__all__ = [
    "DEVICE_CHOICES",
    "DTYPE_CHOICES",
    "ExplicitOperator",
    "FourierOperator",
    "HadamardOperator",
    "SensingOperator",
    "build_operator",
    "describe_device",
    "resolve_device",
    "resolve_dtype",
]
