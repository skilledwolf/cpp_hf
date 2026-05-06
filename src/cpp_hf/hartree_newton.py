"""Newton-on-charge Hartree-only solver (top-level cpp_hf entry point).

Exploits the rank-N structure of the q=0 Hartree self-consistency: the
outer fixed-point lives entirely in the orbital-charge subspace, while the
inner problem at fixed orbital potential is one non-interacting eigh.
Newton with the exact Jacobian (I + Π·HH) converges quadratically; the
linear system is small (nb × nb) and PSD-well-conditioned regardless of
HH's spectral norm.

Hartree-only — set ``include_hartree=True, include_exchange=False`` on the
kernel.  For HF problems use :func:`solve_direct_minimization` or
:func:`solve_scf` instead.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, NamedTuple

import numpy as np

from ._compat import _native, native_required


@dataclass(frozen=True)
class HartreeNewtonConfig:
    max_iter: int = 30
    tol_E: float = 1e-7
    tol_sigma: float = 1e-7
    mu_maxiter: int = 25
    level_shift: float = 0.0
    block_sizes: tuple[int, ...] | None = None
    backtrack_max: int = 6
    backtrack_shrink: float = 0.5
    fix_pi_at_start: bool = False
    project_fn: Callable[[Any], Any] | None = None

    def __post_init__(self) -> None:
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive")


class HartreeNewtonResult(NamedTuple):
    density_matrix: Any
    fock_matrix: Any
    energy: Any
    chemical_potential: Any
    iterations: int
    converged: bool
    history: dict[str, Any]


def _kernel_args_for_native(kernel) -> dict:
    return dict(
        h=np.ascontiguousarray(kernel.h, dtype=np.complex128),
        VR=np.ascontiguousarray(kernel._VR_shifted, dtype=np.complex128),
        refP=np.ascontiguousarray(kernel.refP, dtype=np.complex128),
        w2d=np.ascontiguousarray(kernel.w2d, dtype=np.float64),
        HH=np.ascontiguousarray(kernel.HH, dtype=np.float64),
        contact_g=np.ascontiguousarray(kernel.contact_g, dtype=np.float64),
        contact_Oi=np.ascontiguousarray(kernel.contact_Oi, dtype=np.complex128),
        contact_Oj=np.ascontiguousarray(kernel.contact_Oj, dtype=np.complex128),
        weight_sum=float(kernel.weight_sum),
        T=float(kernel.T),
        include_hartree=bool(kernel.include_hartree),
        include_exchange=bool(kernel.include_exchange),
        exchange_hcp=bool(kernel.exchange_hermitian_channel_packing),
    )


def solve_hartree_newton(
    kernel,
    P0,
    n_electrons: float,
    *,
    config: HartreeNewtonConfig | None = None,
) -> HartreeNewtonResult:
    """Hartree-only Newton-on-charge SCF.  Quadratic convergence."""
    native_required()
    if config is None:
        config = HartreeNewtonConfig()
    P0 = np.ascontiguousarray(P0, dtype=np.complex128)
    block_sizes = list(int(s) for s in (config.block_sizes or ()))
    density, fock, energy, mu, iterations, converged, hE, hSigma = (
        _native.solve_hartree_newton(
            _kernel_args_for_native(kernel),
            P0, float(n_electrons),
            int(config.max_iter), float(config.tol_E), float(config.tol_sigma),
            int(config.mu_maxiter), float(config.level_shift),
            block_sizes,
            int(config.backtrack_max), float(config.backtrack_shrink),
            bool(config.fix_pi_at_start),
            config.project_fn,
        )
    )
    return HartreeNewtonResult(
        density_matrix=density,
        fock_matrix=fock,
        energy=energy,
        chemical_potential=mu,
        iterations=int(iterations),
        converged=bool(converged),
        history={"E": hE, "sigma_residual": hSigma},
    )
