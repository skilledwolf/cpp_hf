"""Direct minimization solver wrapper.

A thin Python layer over ``cpp_hf._native.solve_dm``: prepares the kernel as
a dict of contiguous ``complex128`` / ``float64`` arrays, validates inputs,
and unpacks the C++ tuple back into a :class:`SolveResult`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np

from ._compat import _native, native_required
from .utils import validate_electron_count


@dataclass(frozen=True)
class SolverConfig:
    max_iter: int = 200
    tol_E: float = 1e-7
    tol_grad: float = 0.0
    denom_scale: float = 1e-3
    max_step: float = 0.6
    cg_restart: int = 10
    bt_shrink: float = 0.5
    bt_max: int = 8
    mu_maxiter: int = 25
    block_sizes: tuple[int, ...] | None = None
    project_fn: Any = None


class SolveResult(NamedTuple):
    Q: np.ndarray
    p: np.ndarray
    mu: np.ndarray
    density: np.ndarray
    fock: np.ndarray
    energy: np.ndarray
    n_iter: np.ndarray
    converged: np.ndarray
    history: dict


def _kernel_args_for_native(kernel) -> dict:
    """Pack a HartreeFockKernel into the dict shape that the C++ binding expects."""
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


def solve_direct_minimization(
    kernel,
    P0,
    n_electrons: float,
    *,
    config: SolverConfig | None = None,
) -> SolveResult:
    """Direct-minimization Hartree-Fock solver.

    Preconditioned Riemannian CG on Stiefel × capped simplex with one Fock
    build per iteration.  Algorithm matches :func:`jax_hf.solve` so the same
    tests apply.  Runs entirely in C++ via the ``cpp_hf._native`` extension.
    """
    native_required()
    if config is None:
        config = SolverConfig()
    validate_electron_count(
        kernel.w2d, kernel.h.shape[-1], n_electrons, context="n_electrons",
    )
    P0 = np.ascontiguousarray(P0, dtype=np.complex128)

    block_sizes = list(int(s) for s in (config.block_sizes or ()))
    Q, p, density, fock, mu, energy, n_iter, converged, hE, hG = _native.solve_dm(
        _kernel_args_for_native(kernel),
        P0, float(n_electrons),
        int(config.max_iter), float(config.tol_E), float(config.tol_grad),
        float(config.max_step), float(config.bt_shrink), float(config.denom_scale),
        int(config.bt_max), int(config.cg_restart), int(config.mu_maxiter),
        block_sizes,
        config.project_fn,
    )
    return SolveResult(
        Q=Q,
        p=p,
        mu=np.asarray(mu),
        density=density,
        fock=fock,
        energy=np.asarray(energy),
        n_iter=np.asarray(int(n_iter), dtype=np.int32),
        converged=np.asarray(bool(converged)),
        history={"E": hE, "grad_norm": hG},
    )


solve = solve_direct_minimization


def _cayley_retract(d: np.ndarray, tau: float) -> np.ndarray:
    """Reference LU Cayley retraction ``U = (I − τd/2)(I + τd/2)^-1``.

    Provided as a small numpy reference for the algorithm-correctness tests
    (``TestSpectralCayley``); production code uses the spectral form in
    ``cpp_hf._native``.
    """
    n = d.shape[-1]
    eye = np.eye(n, dtype=d.dtype)
    A = 0.5 * float(tau) * d
    return np.linalg.solve(eye + A, eye - A)
