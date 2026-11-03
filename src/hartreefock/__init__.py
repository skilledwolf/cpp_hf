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
single-precision arrays.  A complex128 kernel built with x64 turned back
off raises instead of running in single precision.
:func:`stability_analysis` and :func:`solve_minimum` test and enforce local
stability.
"""

from __future__ import annotations

from importlib import metadata

import jax

jax.config.update("jax_enable_x64", True)

from . import _jax
from ._jax.utils import chemical_potential
from .config import (
    DeflatedResult,
    SolverConfig,
    SolveResult,
)
from .continuation import resample_to
from .deflation import solve_deflated
from .kernel import HartreeFockKernel
from .mesh import Mesh
from .stability import MinimumResult, StabilityResult, solve_minimum, stability_analysis

try:  # pragma: no cover - metadata only
    __version__ = metadata.version("hartreefock")
except metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"


def solve(kernel, P0, n_electrons, *, config: SolverConfig | None = None, coarse=None):
    """Solve the Hartree-Fock problem at fixed particle number.

    ``config.method`` picks a Roothaan iteration (``"rdiis"``, the default,
    or ``"linear"``: diagonalize the Fock matrix, refill, repeat) or a direct minimization of the free energy (``"cg"``,
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
    coarse : HartreeFockKernel, optional
        The same problem on a coarser mesh.  It is solved first, from ``P0``
        and with the same config and ``n_electrons``, and its density,
        resampled to the mesh of ``kernel``, is the start of the solve.

    Returns
    -------
    SolveResult
        The density, Fock matrix, energy, chemical potential, iteration count,
        convergence flag and per-iteration history (and, from direct
        minimization, the orbitals and occupations); with ``coarse``, its
        result in ``coarse``.
    """
    if coarse is None:
        return _jax.solve(kernel, P0, n_electrons, config=config)
    first = _jax.solve(coarse, P0, n_electrons, config=config)
    P0 = resample_to(coarse, kernel, first.density)
    return _jax.solve(kernel, P0, n_electrons, config=config)._replace(coarse=first)


__all__ = [
    "DeflatedResult",
    "HartreeFockKernel",
    "Mesh",
    "MinimumResult",
    "SolveResult",
    "SolverConfig",
    "StabilityResult",
    "chemical_potential",
    "solve",
    "solve_deflated",
    "solve_minimum",
    "stability_analysis",
]
