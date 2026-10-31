"""JAX backend: jitted solvers, plus the JAX-only RDIIS/Anderson SCF schemes,
density stability analysis (:func:`stability_analysis`) and checked local
minimization (:func:`solve_minimum`).

Enable ``jax_enable_x64`` for double precision (required by the Newton
optimizer and deflation).
"""

from __future__ import annotations

from .fock import build_fock, free_energy, hf_energy, occupation_entropy
from .reference_scf import solve_scf
from .solver import solve
from .superlattice import (
    ExtendedGridLayout,
    build_extended_layout,
    make_superlattice_build_fock_fn,
    make_superlattice_fock_fn,
)
from .utils import resample_kgrid


def __getattr__(name):
    # SciPy-backed orchestration is imported on first use.
    if name in ("StabilityResult", "stability_analysis"):
        from . import stability

        return getattr(stability, name)
    if name in ("MinimumResult", "solve_minimum"):
        from . import minima

        return getattr(minima, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ExtendedGridLayout",
    "MinimumResult",
    "StabilityResult",
    "build_extended_layout",
    "build_fock",
    "free_energy",
    "hf_energy",
    "make_superlattice_build_fock_fn",
    "make_superlattice_fock_fn",
    "occupation_entropy",
    "resample_kgrid",
    "solve",
    "solve_minimum",
    "solve_scf",
    "stability_analysis",
]
