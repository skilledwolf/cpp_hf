"""Hartree-Fock solvers on 2D k-meshes.

Build a :class:`HartreeFockKernel` (or, for a moiré plane-wave basis, a
:class:`SuperlatticeHartreeFockKernel`) once, then solve it::

    kernel = hartreefock.HartreeFockKernel(weights, h, Vq, T=0.01)
    result = hartreefock.solve(kernel, P0, n_electrons)

The solvers are jitted JAX (differentiable, GPU-capable).  On CPU the batched
eigensolvers, complex products and the superlattice exchange run as C++
kernels through ``jax.ffi``.  Importing hartreefock turns on
``jax_enable_x64``, so a solve runs in the dtype of the kernel's arrays:
complex128 from NumPy defaults, complex64 when the kernel is built from
single-precision arrays.
:func:`stability_analysis` and :func:`solve_minimum` test and enforce local
stability.
"""

from __future__ import annotations

from importlib import metadata

import jax

jax.config.update("jax_enable_x64", True)

from . import _jax
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


def fock_operators(kernel):
    """The Fock map of ``kernel`` and its derivative, on host arrays.

    Parameters
    ----------
    kernel : HartreeFockKernel or SuperlatticeHartreeFockKernel
        The problem.

    Returns
    -------
    types.SimpleNamespace
        ``evaluate(P) -> (F, E)``: the Fock matrix and HF energy of density
        ``P``; ``response(P, dP) -> dF``: the Fock response to ``dP``; and
        ``dtype``, the complex dtype both work in.
    """
    return _jax.fock_operators(kernel)


def solve(kernel, P0, n_electrons, *, config: SolverConfig | None = None):
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

    Returns
    -------
    SolveResult
        The density, Fock matrix, energy, chemical potential, iteration count,
        convergence flag and per-iteration history.
    """
    return _jax.solve(kernel, P0, n_electrons, config=config)


def solve_scf(kernel, P0, n_electrons, *, config: SCFConfig | None = None):
    """Solve by Roothaan iteration: diagonalize the Fock matrix, refill, repeat.

    Parameters
    ----------
    kernel : HartreeFockKernel or SuperlatticeHartreeFockKernel
        The problem.
    P0 : (nk1, nk2, nb, nb) array
        Starting density; its particle number is corrected to
        ``n_electrons`` before the first step.
    n_electrons : float
        Particle number, counted with the kernel weights, ``Σ_k w_k Tr P_k``.
    config : SCFConfig, optional
        Acceleration and tolerances; ``None`` uses ``SCFConfig()``, RDIIS.

    Returns
    -------
    SCFResult
        The density, Fock matrix, energy, chemical potential, iteration count,
        convergence flag, a message and the per-iteration history.
    """
    return _jax.solve_scf(kernel, P0, n_electrons, config=config)


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
    "fock_operators",
    "solve",
    "solve_continuation",
    "solve_deflated",
    "solve_minimum",
    "solve_scf",
    "stability_analysis",
]
