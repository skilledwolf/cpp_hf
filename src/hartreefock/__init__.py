"""Hartree-Fock solvers on 2D k-meshes with native (C++) and JAX backends.

Build a :class:`HartreeFockKernel` (or, for a moiré plane-wave basis, a
:class:`SuperlatticeHartreeFockKernel`) once, then solve it with either backend::

    kernel = hartreefock.HartreeFockKernel(weights, h, Vq, T=0.01)
    result = hartreefock.solve(kernel, P0, n_electrons, backend="native")

``backend="native"`` (default) runs the C++ extension in double precision;
``backend="jax"`` runs the jitted JAX solvers and adds the RDIIS/Anderson SCF
schemes, stability analysis and checked minimization
(:mod:`hartreefock._backends.jax`).
"""

from __future__ import annotations

from importlib import metadata

from ._backends import available_backends, get_backend
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


def __getattr__(name):
    # JAX-only tools, imported on first use.
    if name in ("StabilityResult", "stability_analysis"):
        from ._backends.jax import stability

        return getattr(stability, name)
    if name in ("MinimumResult", "solve_minimum"):
        from ._backends.jax import minima

        return getattr(minima, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
    "MinimumResult",
    "SCFConfig",
    "SCFResult",
    "StabilityResult",
    "SolveResult",
    "SolverConfig",
    "SuperlatticeHartreeFockKernel",
    "available_backends",
    "get_backend",
    "solve",
    "solve_continuation",
    "solve_deflated",
    "solve_minimum",
    "solve_scf",
    "stability_analysis",
]
