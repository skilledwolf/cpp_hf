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
    """Configuration for the direct-minimization solver.

    ``block_sizes`` enables a checked block-diagonal acceleration for problems
    whose Fock matrices remain block diagonal in the original orbital basis.
    The solver falls back to the full dense path if a generated Fock matrix has
    off-block entries above numerical tolerance.
    """

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
    return_Q: bool = True
    return_density: bool = True
    return_fock: bool = True


class SolveResult(NamedTuple):
    Q: np.ndarray | None
    p: np.ndarray
    mu: np.ndarray
    density: np.ndarray | None
    fock: np.ndarray | None
    energy: np.ndarray
    n_iter: np.ndarray
    converged: np.ndarray
    history: dict


def _kernel_args_for_native(kernel) -> dict:
    """Pack a HartreeFockKernel into the dict shape that the C++ binding expects.

    Also packs the optional superlattice layout fields when the kernel exposes
    them (see :class:`cpp_hf.superlattice.SuperlatticeHartreeFockKernel`).
    The native solver then dispatches the streaming Fock + full Hartree paths
    inside ``build_fock_compact``.
    """
    refP = kernel._refP if kernel._refP is not None else kernel._empty_refP
    args = dict(
        h=np.ascontiguousarray(kernel.h, dtype=np.complex128),
        VR=np.ascontiguousarray(kernel._VR_shifted, dtype=np.complex128),
        refP=np.ascontiguousarray(refP, dtype=np.complex128),
        has_refP=bool(kernel.has_reference_density),
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
    if getattr(kernel, "superlattice_fock_active", False) \
            or getattr(kernel, "superlattice_hartree_active", False):
        layout = kernel.layout
        args.update(dict(
            superlattice_fock_active=bool(kernel.superlattice_fock_active),
            superlattice_hartree_active=bool(kernel.superlattice_hartree_active),
            hartree_degeneracy=float(kernel.hartree_degeneracy),
            n_G=int(kernel.n_G),
            dim_orb=int(kernel.dim_orb),
            n_delta=int(layout.n_delta),
            N_ext_x=int(layout.N_ext_x),
            N_ext_y=int(layout.N_ext_y),
            V_lag_fft=np.ascontiguousarray(layout.V_lag_fft, dtype=np.complex128),
            g_a_off=np.ascontiguousarray(layout.g_a_off, dtype=np.int64),
            pair_i=np.ascontiguousarray(layout.delta_pair_i, dtype=np.int64),
            pair_j=np.ascontiguousarray(layout.delta_pair_j, dtype=np.int64),
            pair_start=np.ascontiguousarray(layout.delta_pair_start, dtype=np.int64),
            pair_to_delta=np.ascontiguousarray(layout.pair_to_delta, dtype=np.int64),
            HH_GG=np.ascontiguousarray(kernel.HH_GG, dtype=np.float64),
        ))
        HH_orb = getattr(kernel, "HH_GG_orbital", None)
        if HH_orb is not None:
            args["HH_GG_orbital"] = np.ascontiguousarray(
                np.asarray(HH_orb), dtype=np.float64,
            )
        V_orb = getattr(kernel, "V_lag_fft_orbital", None)
        if V_orb is not None:
            args["V_lag_fft_orbital"] = np.ascontiguousarray(
                np.asarray(V_orb), dtype=np.complex128,
            )
    return args


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
        bool(config.return_Q), bool(config.return_density), bool(config.return_fock),
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
