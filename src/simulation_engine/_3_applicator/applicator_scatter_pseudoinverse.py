"""Backward-compat shim. Prefer ``applicator_pseudoinverse``."""
from simulation_engine._3_applicator.applicator_pseudoinverse import (
    ApplicatorPseudoinverse as ApplicatorScatterPseudoinverse,
)

__all__ = ["ApplicatorScatterPseudoinverse"]
