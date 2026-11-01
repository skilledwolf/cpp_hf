"""Hartree-Fock solvers on 2D k-meshes with native (C++) and JAX backends.

Build a :class:`HartreeFockKernel` (or, for a moiré plane-wave basis, a
:class:`SuperlatticeHartreeFockKernel`) once, then solve it with either backend::

    kernel = hartreefock.HartreeFockKernel(weights, h, Vq, T=0.01)
    result = hartreefock.solve(kernel, P0, n_electrons, backend="native")

``backend="native"`` (default) runs the C++ extension in double precision;
``backend="jax"`` runs the jitted JAX solvers (differentiable, GPU-capable).
:func:`stability_analysis` and :func:`solve_minimum` test and enforce local
stability on either backend.
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
from .stability import MinimumResult, StabilityResult, solve_minimum, stability_analysis
from .superlattice import SuperlatticeHartreeFockKernel

try:  # pragma: no cover - metadata only
    __version__ = metadata.version("hartreefock")
except metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"


def solve(kernel, P0, n_electrons, *, config: SolverConfig | None = None, backend: str = "native"):
    """Minimize the free energy directly, with Riemannian CG or a Newton trust region.

    Parameters
    ----------
    kernel : HartreeFockKernel or SuperlatticeHartreeFockKernel
        The problem.
    P0 : (nk1, nk2, nb, nb) array
        Starting density. Its occupations seed the iteration; the particle
        number is imposed through the chemical potential.
    n_electrons : float
        Particle number, counted with the kernel weights, ``Σ_k w_k Tr P_k``.
    config : SolverConfig, optional
        Optimizer and stopping rule; ``None`` uses the defaults (CG).
    backend : {"native", "jax"}
        Which implementation runs the solve.

    Returns
    -------
    SolveResult
        The density, Fock matrix, energy, chemical potential, iteration count,
        convergence flag and per-iteration history.
    """
    return get_backend(backend).solve(kernel, P0, n_electrons, config=config)


def solve_scf(kernel, P0, n_electrons, *, config: SCFConfig | None = None, backend: str = "native"):
    """Solve by Roothaan iteration: diagonalize the Fock matrix, refill, repeat.

    Parameters
    ----------
    kernel : HartreeFockKernel or SuperlatticeHartreeFockKernel
        The problem.
    P0 : (nk1, nk2, nb, nb) array
        Starting density. Every acceleration except ``"oda"`` refills it at
        ``n_electrons`` on the first step; ``"oda"`` starts from it as given,
        so it must already hold ``n_electrons``.
    n_electrons : float
        Particle number, counted with the kernel weights, ``Σ_k w_k Tr P_k``.
    config : SCFConfig, optional
        Acceleration and tolerances; ``None`` uses plain linear mixing.
        ``SCFConfig(acceleration="rdiis")`` is the recommended choice at finite
        temperature.
    backend : {"native", "jax"}
        Which implementation runs the solve.

    Returns
    -------
    SCFResult
        The density, Fock matrix, energy, chemical potential, iteration count,
        convergence flag, a message and the per-iteration history.
    """
    return get_backend(backend).solve_scf(kernel, P0, n_electrons, config=config)


__all__ = [
    "ContinuationResult",
    "DeflatedResult",
    "HartreeFockKernel",
    "MinimumResult",
    "SCFConfig",
    "SCFResult",
    "SolveResult",
    "SolverConfig",
    "StabilityResult",
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
