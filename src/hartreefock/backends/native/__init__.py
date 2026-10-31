"""Native backend: the C++ extension (FFTW exchange, Eigen eigensolvers).

Works in double precision throughout and needs no JAX.
"""

from __future__ import annotations

from .fock import build_fock
from .reference_scf import solve_scf
from .solver import solve
from .superlattice import ExtendedGridLayout, build_extended_layout, superlattice_fock
from .utils import find_chemical_potential, resample_kgrid

__all__ = [
    "ExtendedGridLayout",
    "build_extended_layout",
    "build_fock",
    "find_chemical_potential",
    "resample_kgrid",
    "solve",
    "solve_scf",
    "superlattice_fock",
]
