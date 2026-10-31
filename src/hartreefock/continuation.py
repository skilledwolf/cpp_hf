"""Coarse-to-fine k-grid continuation.

Solve on a coarse kernel, resample the converged density onto the fine grid
and solve again.  The driver is physics-agnostic: it does not reinterpret
``n_electrons`` between grids or remap reference densities; callers build
both kernels themselves.
"""

from __future__ import annotations

import numpy as np

from .backends import get_backend
from .config import ContinuationResult, SCFConfig, SolverConfig
from .kernel import HartreeFockKernel


def _run_stage(backend, kernel, P0, n_electrons, config, stage):
    if isinstance(config, SCFConfig):
        return backend.solve_scf(kernel, P0, float(n_electrons), config=config)
    if config is None or isinstance(config, SolverConfig):
        return backend.solve(kernel, P0, float(n_electrons), config=config)
    raise TypeError(
        f"{stage}_config must be SolverConfig, SCFConfig, or None; got {type(config).__name__}."
    )


def _validate_kernels(coarse: HartreeFockKernel, fine: HartreeFockKernel) -> None:
    if coarse.h.shape[-2:] != fine.h.shape[-2:]:
        raise ValueError(
            "coarse_kernel and fine_kernel must share orbital dimensions "
            f"(..., nb, nb); got {tuple(coarse.h.shape[-2:])} vs {tuple(fine.h.shape[-2:])}."
        )
    for flag in ("include_hartree", "include_exchange"):
        if getattr(coarse, flag) != getattr(fine, flag):
            raise ValueError(f"coarse_kernel and fine_kernel must agree on {flag}.")
    for name, k in (("coarse_kernel", coarse), ("fine_kernel", fine)):
        if k.h.shape[0] != k.h.shape[1]:
            raise ValueError(f"{name} must have a square k-mesh, got shape {tuple(k.h.shape[:2])}.")


def solve_continuation(
    coarse_kernel: HartreeFockKernel,
    fine_kernel: HartreeFockKernel,
    P0_coarse,
    n_electrons_coarse: float,
    n_electrons_fine: float,
    *,
    coarse_config: SolverConfig | SCFConfig | None = None,
    fine_config: SolverConfig | SCFConfig | None = None,
    backend: str = "native",
    retain_P0_fine: bool = True,
) -> ContinuationResult:
    """Run a coarse → fine continuation.

    Each stage runs ``solve_scf`` when its config is an :class:`SCFConfig`
    and direct minimization otherwise, so stages can mix (robust SCF on the
    coarse grid, fast minimization on the fine one).  ``P0_fine`` in the
    result is the resampled seed of the fine solve (None when
    ``retain_P0_fine=False``).
    """
    _validate_kernels(coarse_kernel, fine_kernel)
    engine = get_backend(backend)
    P0_coarse = np.asarray(P0_coarse)
    if P0_coarse.shape != coarse_kernel.h.shape:
        raise ValueError(
            f"P0_coarse shape {tuple(P0_coarse.shape)} does not match "
            f"coarse_kernel hamiltonian shape {tuple(coarse_kernel.h.shape)}."
        )

    coarse = _run_stage(engine, coarse_kernel, P0_coarse, n_electrons_coarse, coarse_config, "coarse")
    density = coarse.density
    if density is None:
        raise ValueError("the coarse stage must return its density (return_density=True).")
    P0_fine = engine.resample_kgrid(density, int(fine_kernel.h.shape[0]))
    P0_fine = 0.5 * (P0_fine + P0_fine.conj().swapaxes(-1, -2))
    fine = _run_stage(engine, fine_kernel, P0_fine, n_electrons_fine, fine_config, "fine")
    return ContinuationResult(coarse=coarse, fine=fine, P0_fine=P0_fine if retain_P0_fine else None)
