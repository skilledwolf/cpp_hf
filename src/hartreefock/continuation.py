"""Coarse-to-fine k-grid continuation.

Solve on a coarse kernel, resample the converged density onto the fine grid
and solve again.  The driver is physics-agnostic: it does not reinterpret
``n_electrons`` between grids or remap reference densities; callers build
both kernels themselves.
"""

from __future__ import annotations

import numpy as np

from . import _jax
from .config import ContinuationResult, SolverConfig
from .kernel import HartreeFockKernel
from .superlattice import upsample_density


def _validate_kernels(coarse: HartreeFockKernel, fine: HartreeFockKernel) -> None:
    if coarse.h.shape[-2:] != fine.h.shape[-2:]:
        raise ValueError(
            "coarse_kernel and fine_kernel must share orbital dimensions "
            f"(..., nb, nb); got {tuple(coarse.h.shape[-2:])} vs {tuple(fine.h.shape[-2:])}."
        )
    for flag in ("include_hartree", "include_exchange"):
        if getattr(coarse, flag) != getattr(fine, flag):
            raise ValueError(f"coarse_kernel and fine_kernel must agree on {flag}.")


def solve_continuation(
    coarse_kernel: HartreeFockKernel,
    fine_kernel: HartreeFockKernel,
    P0_coarse,
    n_electrons_coarse: float,
    n_electrons_fine: float,
    *,
    coarse_config: SolverConfig | None = None,
    fine_config: SolverConfig | None = None,
    retain_P0_fine: bool = True,
) -> ContinuationResult:
    """Solve on a coarse mesh, resample the result to a fine mesh, and solve there.

    Each stage is a :func:`~hartreefock.solve` with its own config, so the
    stages can use different methods. The driver does not reinterpret the particle number
    or the reference densities between meshes; build both kernels consistently.

    Parameters
    ----------
    coarse_kernel, fine_kernel : HartreeFockKernel
        The same problem on two meshes.
    P0_coarse : (nk1, nk2, nb, nb) array
        Starting density on the coarse mesh.
    n_electrons_coarse, n_electrons_fine : float
        Particle numbers on each mesh, counted with its weights.
    coarse_config, fine_config : SolverConfig, optional
        Method and options of each stage; ``None`` is RDIIS with the defaults.
    retain_P0_fine : bool
        Keep the resampled seed in the result.

    Returns
    -------
    ContinuationResult
        Both stage results and the resampled seed.
    """
    _validate_kernels(coarse_kernel, fine_kernel)
    P0_coarse = np.asarray(P0_coarse)
    if P0_coarse.shape != coarse_kernel.h.shape:
        raise ValueError(
            f"P0_coarse shape {tuple(P0_coarse.shape)} does not match "
            f"coarse_kernel hamiltonian shape {tuple(coarse_kernel.h.shape)}."
        )

    coarse = _jax.solve(coarse_kernel, P0_coarse, float(n_electrons_coarse), config=coarse_config)
    density = coarse.density
    if density is None:
        raise ValueError("the coarse stage must return its density (return_density=True).")
    if fine_kernel.superlattice:
        P0_fine = upsample_density(density, coarse_kernel.mesh, fine_kernel.mesh)
    else:
        P0_fine = np.asarray(_jax.resample_kgrid(density, int(fine_kernel.h.shape[0])))
        P0_fine = 0.5 * (P0_fine + P0_fine.conj().swapaxes(-1, -2))
    fine = _jax.solve(fine_kernel, P0_fine, float(n_electrons_fine), config=fine_config)
    return ContinuationResult(coarse=coarse, fine=fine, P0_fine=P0_fine if retain_P0_fine else None)
