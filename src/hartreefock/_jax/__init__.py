"""The engine: jitted solvers, Fock builds and exact Fock JVPs."""

from __future__ import annotations

from .fock import build_fock, free_energy, hf_energy, occupation_entropy
from .operators import fock_operators
from .scf import roothaan
from .solver import minimize
from .superlattice import make_superlattice_fock_fn, superlattice_build_fock
from .utils import resample_kgrid


def solve(kernel, P0, n_electrons, *, config=None):
    """Roothaan iteration or direct minimization, as ``config.method`` says.

    The energy is the standard HF energy, also for an embedded reference.
    """
    from ..config import SolverConfig

    config = SolverConfig() if config is None else config
    P0 = kernel.noninteracting_density(n_electrons) if P0 is None else P0
    run = roothaan if config.roothaan else minimize
    result = run(kernel, P0, n_electrons, config=config)
    offset = kernel.embedded_energy_offset
    history = {key: value - offset if key in ("E", "free_energy") else value
               for key, value in result.history.items()}
    return result._replace(energy=result.energy - offset, history=history)

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
