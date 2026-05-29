"""Coarse-to-fine multigrid continuation (numpy port of ``jax_hf.continuation``).

Runs ``solve`` (or ``solve_scf``) on a coarse kernel, resamples the converged
density onto a fine grid, and runs the solver again on a fine kernel.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Union

import numpy as np

from .problem import HartreeFockKernel
from .reference_scf import SCFConfig, SCFResult, solve_scf
from .solver import SolverConfig, SolveResult, solve
from .utils import resample_kgrid

StageResult = Union[SolveResult, SCFResult]


class ContinuationResult(NamedTuple):
    coarse: StageResult
    fine: StageResult
    P0_fine: np.ndarray | None


def _resample_density(density_c: np.ndarray, nk_fine: int) -> np.ndarray:
    P = resample_kgrid(density_c, nk_fine, method="linear")
    return 0.5 * (P + np.conj(np.swapaxes(P, -1, -2)))


def _run_stage(
    kernel: HartreeFockKernel,
    P0: np.ndarray,
    n_electrons: float,
    config: Any,
    *,
    stage_name: str,
) -> StageResult:
    if isinstance(config, SCFConfig):
        return solve_scf(kernel, P0, float(n_electrons), config=config)
    if isinstance(config, SolverConfig) or config is None:
        return solve(kernel, P0, float(n_electrons), config=config)
    raise TypeError(
        f"{stage_name}_config must be SolverConfig, SCFConfig, or None; "
        f"got {type(config).__name__}."
    )


def _density_of(result: StageResult) -> np.ndarray:
    if isinstance(result, SCFResult):
        if result.density_matrix is None:
            raise ValueError("coarse SCF result did not retain density_matrix.")
        return result.density_matrix
    if result.density is None:
        raise ValueError("coarse direct-minimization result did not retain density.")
    return result.density


def solve_continuation(
    coarse_kernel: HartreeFockKernel,
    fine_kernel: HartreeFockKernel,
    P0_coarse: np.ndarray,
    n_electrons_coarse: float,
    n_electrons_fine: float,
    *,
    coarse_config: SolverConfig | SCFConfig | None = None,
    fine_config: SolverConfig | SCFConfig | None = None,
    retain_P0_fine: bool = True,
) -> ContinuationResult:
    """Run a coarse → fine multigrid HF continuation.

    Numpy port of :func:`jax_hf.solve_continuation`.
    """
    _validate_kernels(coarse_kernel, fine_kernel)

    P0_coarse_a = np.asarray(P0_coarse, dtype=coarse_kernel.h.dtype)
    if P0_coarse_a.shape != coarse_kernel.h.shape:
        raise ValueError(
            f"P0_coarse shape {tuple(P0_coarse_a.shape)} does not match "
            f"coarse_kernel hamiltonian shape {tuple(coarse_kernel.h.shape)}."
        )

    coarse_result = _run_stage(
        coarse_kernel, P0_coarse_a, n_electrons_coarse, coarse_config,
        stage_name="coarse",
    )

    nk_fine = int(fine_kernel.h.shape[0])
    P0_fine = _resample_density(_density_of(coarse_result), nk_fine)
    P0_fine = P0_fine.astype(fine_kernel.h.dtype, copy=False)

    fine_result = _run_stage(
        fine_kernel, P0_fine, n_electrons_fine, fine_config,
        stage_name="fine",
    )

    return ContinuationResult(
        coarse=coarse_result,
        fine=fine_result,
        P0_fine=P0_fine if retain_P0_fine else None,
    )


def _validate_kernels(coarse: HartreeFockKernel, fine: HartreeFockKernel) -> None:
    if coarse.h.shape[-2:] != fine.h.shape[-2:]:
        raise ValueError(
            "coarse_kernel and fine_kernel must share orbital dimensions "
            f"(..., nb, nb); got {tuple(coarse.h.shape[-2:])} vs "
            f"{tuple(fine.h.shape[-2:])}."
        )
    if coarse.include_hartree != fine.include_hartree:
        raise ValueError(
            "coarse_kernel and fine_kernel must agree on include_hartree."
        )
    if coarse.include_exchange != fine.include_exchange:
        raise ValueError(
            "coarse_kernel and fine_kernel must agree on include_exchange."
        )
    if coarse.h.shape[0] != coarse.h.shape[1]:
        raise ValueError(
            f"coarse_kernel must have a square k-mesh, got shape {tuple(coarse.h.shape[:2])}."
        )
    if fine.h.shape[0] != fine.h.shape[1]:
        raise ValueError(
            f"fine_kernel must have a square k-mesh, got shape {tuple(fine.h.shape[:2])}."
        )


__all__ = ["ContinuationResult", "solve_continuation"]
