"""Hartree-Fock solvers on 2D k-meshes.

Build a :class:`HartreeFockKernel` on a :class:`Mesh` once, then solve it::

    mesh = hartreefock.Mesh.square(kmax=1.2, nk=48)
    kernel = hartreefock.HartreeFockKernel(mesh, h, V, T=0.01)
    result = hartreefock.solve(kernel, P0, n_electrons)

``V`` is a function of ``|q|`` (or its table on ``mesh.transfer_grid``); a
moiré plane-wave basis adds ``plane_waves=``.

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
    SolverConfig,
    SolveResult,
)
from .continuation import solve_continuation
from .deflation import solve_deflated
from .kernel import HartreeFockKernel
from .mesh import Mesh
from .stability import MinimumResult, StabilityResult, solve_minimum, stability_analysis

try:  # pragma: no cover - metadata only
    __version__ = metadata.version("hartreefock")
except metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"


def fock_operators(kernel):
    """The Fock map of ``kernel`` and its derivative, on host arrays.

    Parameters
    ----------
    kernel : HartreeFockKernel
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
    """Solve the Hartree-Fock problem at fixed particle number.

    ``config.method`` picks a Roothaan iteration (``"rdiis"``, the default,
    ``"anderson"``, ``"oda"``, ``"linear"``: diagonalize the Fock matrix,
    refill, repeat) or a direct minimization of the free energy (``"cg"``,
    ``"newton"``).

    Parameters
    ----------
    kernel : HartreeFockKernel
        The problem.
    P0 : (nk1, nk2, nb, nb) array or None
        Starting density; ``None`` is ``kernel.noninteracting_density(n)``.
        Roothaan corrects its particle number before the first step; direct
        minimization takes its occupations as the seed and imposes the
        particle number through the chemical potential.
    n_electrons : float
        Particle number, counted with the kernel weights, ``Σ_k w_k Tr P_k``.
    config : SolverConfig, optional
        Method and stopping rule; ``None`` is RDIIS with the defaults.

    Returns
    -------
    SolveResult
        The density, Fock matrix, energy, chemical potential, iteration count,
        convergence flag and per-iteration history (and, from direct
        minimization, the orbitals and occupations).
    """
    return _jax.solve(kernel, P0, n_electrons, config=config)


__all__ = [
    "ContinuationResult",
    "DeflatedResult",
    "HartreeFockKernel",
    "Mesh",
    "MinimumResult",
    "SolveResult",
    "SolverConfig",
    "StabilityResult",
    "fock_operators",
    "solve",
    "solve_continuation",
    "solve_deflated",
    "solve_minimum",
    "stability_analysis",
]
