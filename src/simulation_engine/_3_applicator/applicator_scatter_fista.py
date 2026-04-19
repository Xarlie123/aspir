"""Backward-compat shim. Prefer ``applicator_fista``."""
from simulation_engine._3_applicator.applicator_fista import (
    ApplicatorFISTA as ApplicatorScatterFISTA,
)

__all__ = ["ApplicatorScatterFISTA"]
