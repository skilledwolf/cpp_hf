"""The engine: jitted solvers, Fock builds and exact Fock JVPs."""

from __future__ import annotations

from .fock import build_fock, free_energy, hf_energy, occupation_entropy
from .operators import fock_operators
from .scf import roothaan
from .solver import minimize
from .superlattice import make_superlattice_fock_fn, superlattice_build_fock
from .utils import resample_kgrid


def solve(kernel, P0, n_electrons, *, config=None):
    """Roothaan iteration or direct minimization, as ``config.method`` says."""
    from ..config import SolverConfig

    config = SolverConfig() if config is None else config
    run = roothaan if config.roothaan else minimize
    return run(kernel, P0, n_electrons, config=config)

__all__ = [
    "build_fock",
    "fock_operators",
    "free_energy",
    "hf_energy",
    "make_superlattice_fock_fn",
    "occupation_entropy",
    "resample_kgrid",
    "solve",
    "superlattice_build_fock",
]
