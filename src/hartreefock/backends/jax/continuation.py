"""Coarse-to-fine multigrid continuation helpers.

A thin driver that runs ``solve`` (or ``solve_scf``) on a coarse kernel, then
resamples the converged density onto a fine grid and runs the solver again on
a fine kernel. The two solver calls are separate JITs — cache warmth within
each call is preserved across repeated continuations with the same shapes.

This driver is intentionally *algorithm-agnostic*: it does not know about
filling conventions, reference-density interpolation, or self-energy seeds.
Callers that need physics-aware seeding (e.g. contimod) should build the two
kernels themselves and hand them to this function; the only thing we do is
resample the coarse-solve density onto the fine grid to warm-start the fine
solve.

Both kernels must share orbital dimensions ``(..., nb, nb)`` and the same
``include_hartree`` / ``include_exchange`` flags. The first two axes (the
``(nk, nk)`` k-mesh) may differ.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Union

import jax.numpy as jnp

from .problem import HartreeFockKernel
from .reference_scf import SCFConfig, SCFResult, solve_scf
from .solver import SolverConfig, SolveResult, solve
from .utils import resample_kgrid


StageResult = Union[SolveResult, SCFResult]


class ContinuationResult(NamedTuple):
    """Paired (coarse, fine) solver outputs plus the fine-grid seed.

    ``P0_fine`` is the resampled coarse density used to seed the fine solve,
    exposed so callers can inspect / cache the multigrid hand-off without
    re-deriving it.
    """

    coarse: StageResult
    fine: StageResult
    P0_fine: jnp.ndarray


def _resample_density(density_c: jnp.ndarray, nk_fine: int) -> jnp.ndarray:
    """Resample a coarse density matrix onto the fine k-grid, preserving hermiticity."""
    P = resample_kgrid(density_c, nk_fine, method="linear")
    return 0.5 * (P + jnp.conj(jnp.swapaxes(P, -1, -2)))


def _run_stage(
    kernel: HartreeFockKernel,
    P0: jnp.ndarray,
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


def _density_of(result: StageResult) -> jnp.ndarray:
    if isinstance(result, SCFResult):
        return result.density_matrix
    return result.density


def solve_continuation(
    coarse_kernel: HartreeFockKernel,
    fine_kernel: HartreeFockKernel,
    P0_coarse: jnp.ndarray,
    n_electrons_coarse: float,
    n_electrons_fine: float,
    *,
    coarse_config: SolverConfig | SCFConfig | None = None,
    fine_config: SolverConfig | SCFConfig | None = None,
) -> ContinuationResult:
    """Run a coarse → fine multigrid SCF / direct-minimization continuation.

    Parameters
    ----------
    coarse_kernel, fine_kernel
        :class:`HartreeFockKernel` instances. They must have the same orbital
        dimensions ``(..., nb, nb)`` and the same ``include_hartree`` /
        ``include_exchange`` flags. The first two axes (k-mesh) may differ.
    P0_coarse
        Initial density on the coarse grid, shape
        ``(nk_c, nk_c, nb, nb)``. Pass zeros for an unseeded coarse solve.
    n_electrons_coarse, n_electrons_fine
        Target electron counts on each grid (weights × occupations sum).
        Passed separately because callers sometimes need different values
        when a reference density shifts filling across grids.
    coarse_config, fine_config
        Per-stage solver configs. Pass :class:`SCFConfig` to use
        :func:`solve_scf` at that stage, or :class:`SolverConfig` (or
        ``None``) to use :func:`solve` (direct minimization). The two stages
        can mix and match — e.g. SCF for the robust coarse solve and direct
        minimization for the fast fine solve.

    Returns
    -------
    ContinuationResult
        ``(coarse, fine, P0_fine)`` — the two stage results plus the
        resampled coarse density that seeded the fine solve.

    Notes
    -----
    This is a thin orchestration layer. It does not reinterpret
    ``n_electrons`` between grids, remap a reference density, or translate
    self-energy seeds into density matrices — those decisions are
    physics-aware and belong in the caller. See the ``contimod`` meanfield
    API for a worked example.
    """
    _validate_kernels(coarse_kernel, fine_kernel)

    P0_coarse_j = jnp.asarray(P0_coarse, dtype=coarse_kernel.h.dtype)
    if P0_coarse_j.shape != coarse_kernel.h.shape:
        raise ValueError(
            f"P0_coarse shape {tuple(P0_coarse_j.shape)} does not match "
            f"coarse_kernel hamiltonian shape {tuple(coarse_kernel.h.shape)}."
        )

    coarse_result = _run_stage(
        coarse_kernel, P0_coarse_j, n_electrons_coarse, coarse_config,
        stage_name="coarse",
    )

    nk_fine = int(fine_kernel.h.shape[0])
    P0_fine = _resample_density(_density_of(coarse_result), nk_fine)
    P0_fine = P0_fine.astype(fine_kernel.h.dtype, copy=False)

    fine_result = _run_stage(
        fine_kernel, P0_fine, n_electrons_fine, fine_config,
        stage_name="fine",
    )

    return ContinuationResult(coarse=coarse_result, fine=fine_result, P0_fine=P0_fine)


def _validate_kernels(coarse: HartreeFockKernel, fine: HartreeFockKernel) -> None:
    # Orbital dims must match
    if coarse.h.shape[-2:] != fine.h.shape[-2:]:
        raise ValueError(
            "coarse_kernel and fine_kernel must share orbital dimensions "
            f"(..., nb, nb); got {tuple(coarse.h.shape[-2:])} vs "
            f"{tuple(fine.h.shape[-2:])}."
        )
    # Channel flags must match — mixing them would be a user error
    if coarse.include_hartree != fine.include_hartree:
        raise ValueError(
            "coarse_kernel and fine_kernel must agree on include_hartree."
        )
    if coarse.include_exchange != fine.include_exchange:
        raise ValueError(
            "coarse_kernel and fine_kernel must agree on include_exchange."
        )
    # k-mesh must be square (2D) on both grids — matches the solver contract
    if coarse.h.shape[0] != coarse.h.shape[1]:
        raise ValueError(
            f"coarse_kernel must have a square k-mesh, got shape {tuple(coarse.h.shape[:2])}."
        )
    if fine.h.shape[0] != fine.h.shape[1]:
        raise ValueError(
            f"fine_kernel must have a square k-mesh, got shape {tuple(fine.h.shape[:2])}."
        )


__all__ = ["ContinuationResult", "solve_continuation"]
