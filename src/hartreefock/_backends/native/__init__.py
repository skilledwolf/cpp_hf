"""Native backend: the C++ extension (FFTW exchange, Eigen eigensolvers).

Works in double precision throughout and needs no JAX.
"""

from __future__ import annotations

from .fock import build_fock, fock_operators, superlattice_exchange
from .scf import solve_scf
from .solver import solve
from .utils import density_matrix_from_fock, find_chemical_potential, resample_kgrid

__all__ = [
    "build_fock",
    "density_matrix_from_fock",
    "find_chemical_potential",
    "fock_operators",
    "resample_kgrid",
    "solve",
    "solve_scf",
    "superlattice_exchange",
]
