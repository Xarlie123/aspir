"""Backward-compat shim. Prefer ``applicator_tv_norm``."""
from simulation_engine._3_applicator.applicator_tv_norm import (
    ApplicatorTV as ApplicatorScatterTV,
)

__all__ = ["ApplicatorScatterTV"]
