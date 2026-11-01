"""Native backend: the C++ extension (FFTW exchange, Eigen eigensolvers).

Works in double precision throughout and needs no JAX.
"""

from __future__ import annotations

from .fock import build_fock, superlattice_exchange
from .scf import solve_scf
from .solver import solve
from .utils import find_chemical_potential, resample_kgrid

__all__ = [
    "build_fock",
    "find_chemical_potential",
    "resample_kgrid",
    "solve",
    "solve_scf",
    "superlattice_exchange",
]
