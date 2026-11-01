"""JAX backend: jitted solvers, plus the JAX-only RDIIS/Anderson SCF schemes.
Stability analysis and checked minimization (``stability.py``, ``minima.py``)
are exposed as :func:`hartreefock.stability_analysis` and
:func:`hartreefock.solve_minimum`.

Enable ``jax_enable_x64`` for double precision (required by the Newton
optimizer and deflation).
"""

from __future__ import annotations

from .fock import build_fock, free_energy, hf_energy, occupation_entropy
from .scf import solve_scf
from .solver import solve
from .superlattice import make_superlattice_fock_fn, superlattice_build_fock
from .utils import resample_kgrid

__all__ = [
    "build_fock",
    "free_energy",
    "hf_energy",
    "make_superlattice_fock_fn",
    "occupation_entropy",
    "resample_kgrid",
    "solve",
    "solve_scf",
    "superlattice_build_fock",
]
