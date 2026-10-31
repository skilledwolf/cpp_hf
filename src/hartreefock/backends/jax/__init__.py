"""jax_hf — Hartree-Fock solvers on 2D k-meshes.

Two solvers are available, sharing a single :class:`HartreeFockKernel`:

* :func:`solve` (alias :func:`solve_direct_minimization`) — preconditioned
  Riemannian CG on Stiefel × capped simplex with Cayley retraction and
  true-free-energy backtracking. The legacy default is preserved.
* :func:`solve_scf` — self-consistent field iteration, including safeguarded
  full-density residual DIIS and Anderson acceleration.
* :func:`solve_minimum` — RDIIS followed by local density stability checks
  and bounded saddle recovery. Recommended for checked local minimization.

Both return rich NamedTuples (:class:`SolveResult`, :class:`SCFResult`) with
the converged density, Fock matrix, energy, and full iteration history.

Lower-level building blocks are exposed for users who want to evaluate the
HF objective at arbitrary densities without running a solver:

* :func:`build_fock` — construct the Fock matrix at a given density
* :func:`hf_energy` — total HF energy at (P, Σ, H, h)
* :func:`free_energy` — free energy Ω = E − T·S
"""

from __future__ import annotations

from importlib import metadata

try:  # pragma: no cover - metadata only
    __version__ = metadata.version("hartreefock")
except metadata.PackageNotFoundError:  # pragma: no cover - package not installed
    __version__ = "0.0.0"

from .problem import HartreeFockKernel  # noqa: F401
from .solver import (  # noqa: F401
    SolverConfig,
    SolveResult,
    solve,
    solve_direct_minimization,
)
from .reference_scf import (  # noqa: F401
    SCFConfig,
    SCFResult,
    solve_scf,
)
from .fock import (  # noqa: F401
    build_fock,
    hf_energy,
    free_energy,
    occupation_entropy,
)
from .utils import resample_kgrid  # noqa: F401
from .continuation import (  # noqa: F401
    ContinuationResult,
    solve_continuation,
)
from .deflation import (  # noqa: F401
    DeflatedResult,
    solve_deflated,
)
from .superlattice import (  # noqa: F401
    ExtendedGridLayout,
    build_extended_layout,
    make_superlattice_fock_fn,
    make_superlattice_build_fock_fn,
)


def __getattr__(name):
    # Stability/minimum orchestration imports SciPy only when requested, and
    # does not allocate JAX arrays or initialize devices during package import.
    if name in ("StabilityResult", "stability_analysis"):
        from . import stability
        return getattr(stability, name)
    if name in ("MinimumResult", "solve_minimum"):
        from . import minima
        return getattr(minima, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Kernel (problem definition + precomputed arrays)
    "HartreeFockKernel",
    # Direct-minimization solver (primary)
    "SolverConfig",
    "SolveResult",
    "solve",
    "solve_direct_minimization",
    # Reference SCF solver (fallback / baseline)
    "SCFConfig",
    "SCFResult",
    "solve_scf",
    "StabilityResult",
    "stability_analysis",
    "MinimumResult",
    "solve_minimum",
    # HF objective building blocks
    "build_fock",
    "hf_energy",
    "free_energy",
    "occupation_entropy",
    # Continuation / seeding utilities
    "resample_kgrid",
    "ContinuationResult",
    "solve_continuation",
    # Deflation (distinct HF solutions, Newton path)
    "DeflatedResult",
    "solve_deflated",
    # Superlattice (g_basis / SuperlatticeModel) Fock
    "ExtendedGridLayout",
    "build_extended_layout",
    "make_superlattice_fock_fn",
    "make_superlattice_build_fock_fn",
]
