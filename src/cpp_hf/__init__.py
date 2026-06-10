"""cpp_hf — Hartree-Fock solvers on 2D k-meshes (C++ reimplementation of jax_hf).

Two solvers are available, sharing a single :class:`HartreeFockKernel`:

* :func:`solve` (alias :func:`solve_direct_minimization`) — preconditioned
  Riemannian CG on Stiefel × capped simplex.  One Fock build per iteration,
  eigen-free inner loop, Cayley retraction.  **Default and recommended.**
* :func:`solve_scf` — reference self-consistent field iteration with linear
  mixing.  Useful as a baseline or fallback.

The package mirrors the public API of ``jax_hf`` exactly so that the same
tests and regression examples apply.  The hot kernels (FFT-based exchange,
Hermitian eigendecomposition) are routed through compiled libraries
(``numpy`` / ``scipy`` LAPACK + pocketfft, plus an optional FFTW + Eigen
extension named ``cpp_hf._native``); JAX is not used or required.
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
]
