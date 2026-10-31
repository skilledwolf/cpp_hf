"""Hartree-Fock solvers on 2D k-meshes with native (C++) and JAX backends.

Build a :class:`HartreeFockKernel` (or, for a moiré plane-wave basis, a
:class:`SuperlatticeHartreeFockKernel`) once, then solve it with either backend::

    kernel = hartreefock.HartreeFockKernel(weights, h, Vq, T=0.01)
    result = hartreefock.solve(kernel, P0, n_electrons, backend="native")

``backend="native"`` (default) runs the C++ extension in double precision;
``backend="jax"`` runs the jitted JAX solvers and adds the RDIIS/Anderson SCF
schemes, stability analysis and checked minimization
(:mod:`hartreefock.backends.jax`).
"""

from __future__ import annotations

from importlib import metadata

from .backends import available_backends, get_backend
from .config import (
    ContinuationResult,
    DeflatedResult,
    SCFConfig,
    SCFResult,
    SolverConfig,
    SolveResult,
)
from .continuation import solve_continuation
from .deflation import solve_deflated
from .kernel import HartreeFockKernel
from .superlattice import SuperlatticeHartreeFockKernel

try:  # pragma: no cover - metadata only
    __version__ = metadata.version("hartreefock")
except metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"


def solve(kernel, P0, n_electrons, *, config: SolverConfig | None = None, backend: str = "native"):
    """Direct minimization (Riemannian CG or trust-region Newton)."""
    return get_backend(backend).solve(kernel, P0, n_electrons, config=config)


def solve_scf(kernel, P0, n_electrons, *, config: SCFConfig | None = None, backend: str = "native"):
    """Roothaan SCF iteration with linear, DIIS, ODA (or, on JAX, RDIIS/Anderson) acceleration."""
    return get_backend(backend).solve_scf(kernel, P0, n_electrons, config=config)


__all__ = [
    "ContinuationResult",
    "DeflatedResult",
    "HartreeFockKernel",
    "SCFConfig",
    "SCFResult",
    "SolveResult",
    "SolverConfig",
    "SuperlatticeHartreeFockKernel",
    "available_backends",
    "get_backend",
    "solve",
    "solve_continuation",
    "solve_deflated",
    "solve_scf",
]
