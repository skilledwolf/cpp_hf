"""cpp_hf — Hartree-Fock solvers on 2D k-meshes (C++ reimplementation of jax_hf).

Two solvers are available, sharing a single :class:`HartreeFockKernel`:

* :func:`solve` (alias :func:`solve_direct_minimization`) — preconditioned
  Riemannian CG (legacy default), coupled trust-region Newton, or an automatic
  full-residual DIIS / Newton hybrid.
* :func:`solve_scf` — self-consistent field iteration, including safeguarded
  full-density-residual DIIS (``acceleration="rdiis"``), Anderson, and the
  legacy linear, commutator-DIIS and optimal-damping methods.

The common kernel/config/solve/result API closely follows ``jax_hf`` so that
ordinary built-in-kernel workflows are source-compatible. JAX-specific
tracing, differentiation, and the ``fock_build_fn=`` callback are intentionally
outside this backend's contract. The hot kernels (FFT-based exchange and
Hermitian eigendecomposition) run in the required FFTW + Eigen extension
``cpp_hf._native``; JAX is not used or required.
"""

from __future__ import annotations

from importlib import metadata

try:  # pragma: no cover - metadata only
    __version__ = metadata.version("cpp_hf")
    _meta = metadata.metadata("cpp_hf")
    __author__ = _meta.get("Author", "unknown")
except metadata.PackageNotFoundError:  # pragma: no cover - editable install
    __version__ = "0.0.0"
    __author__ = "unknown"

from .problem import HartreeFockKernel
from .solver import (
    SolverConfig,
    SolveResult,
    solve,
    solve_direct_minimization,
)
from .reference_scf import (
    SCFConfig,
    SCFResult,
    solve_scf,
)
from .fock import (
    build_fock,
    hf_energy,
    free_energy,
    occupation_entropy,
)
from .utils import resample_kgrid
from .continuation import (
    ContinuationResult,
    solve_continuation,
)
from .deflation import (
    DeflatedResult,
    solve_deflated,
)
from .superlattice import (
    ExtendedGridLayout,
    build_extended_layout,
    superlattice_fock,
)

__all__ = [
    "HartreeFockKernel",
    "SolverConfig",
    "SolveResult",
    "solve",
    "solve_direct_minimization",
    "SCFConfig",
    "SCFResult",
    "solve_scf",
    "build_fock",
    "hf_energy",
    "free_energy",
    "occupation_entropy",
    "resample_kgrid",
    "ContinuationResult",
    "solve_continuation",
    "DeflatedResult",
    "solve_deflated",
    "ExtendedGridLayout",
    "build_extended_layout",
    "superlattice_fock",
    "stability_analysis",
    "StabilityResult",
    "solve_minimum",
    "MinimumResult",
]


def __getattr__(name):
    if name in ("solve_minimum", "MinimumResult"):
        from .minima import solve_minimum, MinimumResult
        return {"solve_minimum": solve_minimum, "MinimumResult": MinimumResult}[name]
    if name in ("stability_analysis", "StabilityResult"):
        from .stability import stability_analysis, StabilityResult
        return {"stability_analysis": stability_analysis, "StabilityResult": StabilityResult}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
