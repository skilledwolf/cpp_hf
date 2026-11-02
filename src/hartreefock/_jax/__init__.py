"""The engine: jitted solvers, Fock builds and exact Fock JVPs."""

from __future__ import annotations

from .fock import build_fock, free_energy, hf_energy, occupation_entropy
from .operators import fock_operators
from .scf import solve_scf
from .solver import solve
from .superlattice import make_superlattice_fock_fn, superlattice_build_fock
from .utils import resample_kgrid

__all__ = [
    "build_fock",
    "fock_operators",
    "free_energy",
    "hf_energy",
    "make_superlattice_fock_fn",
    "occupation_entropy",
    "resample_kgrid",
    "solve",
    "solve_scf",
    "superlattice_build_fock",
]
