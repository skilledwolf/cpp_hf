"""Reference SCF solver wrapper.

A thin Python layer over ``cpp_hf._native.solve_scf``: prepares the kernel
as a dict of contiguous ``complex128`` / ``float64`` arrays, validates
inputs, and unpacks the C++ tuple back into a :class:`SCFResult`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ._compat import _native, native_required


@dataclass(frozen=True)
class SCFConfig:
    max_iter: int = 200
    density_tol: float = 1e-7
    comm_tol: float = 1e-6
    mixing: float = 0.5
    level_shift: float = 0.0
    project_fn: Callable[[Any], Any] | None = None
    # Block-diagonal acceleration for the Fock eigh (4× speedup when usable).
    block_sizes: tuple[int, ...] | None = None
    # Acceleration scheme: "linear" (α-mixing), "diis" (Pulay commutator-DIIS
    # extrapolation of Fock), or "oda" (analytic optimal damping).
    acceleration: str = "linear"
    diis_size: int = 6
    diis_start: int = 2
    # Damping for DIIS extrapolation: F_used = damp·F_extrap + (1-damp)·F_curr.
    # 1.0 = pure DIIS (default); ~0.7 suppresses late-iter oscillation.
    diis_damping: float = 1.0
    # Trust-region clip on the per-iter density step (Frobenius norm).
    # 0.0 = disabled.  Useful for ill-conditioned cases where DIIS extrapolates
    # to unphysical states (ungated 1/q + high doping).
    trust_radius: float = 0.0
    return_density: bool = True
    return_fock: bool = True

    def __post_init__(self) -> None:
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive")


@dataclass(frozen=True)
class SCFResult:
    density_matrix: Any | None
    fock_matrix: Any | None
    energy: Any
    chemical_potential: Any
    iterations: int
    converged: bool
    message: str
    history: dict[str, Any] = field(default_factory=dict)


def _kernel_args_for_native(kernel) -> dict:
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
    # Superlattice path: kernel exposes the layout + CSR + HH_GG.  The
    # native dispatcher uses these instead of the FFT exchange (VR is
    # ignored) and instead of the diagonal Hartree (HH is ignored).
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


def solve_scf(
    kernel,
    P0,
    n_electrons: float,
    *,
    config: SCFConfig | None = None,
) -> SCFResult:
    """Reference SCF solver with linear mixing.  Mirrors jax_hf.solve_scf."""
    native_required()
    if config is None:
        config = SCFConfig()

    P0 = np.ascontiguousarray(P0, dtype=np.complex128)
    density, fock, energy, mu, iterations, converged, hE, hD, hC = _native.solve_scf(
        _kernel_args_for_native(kernel),
        P0, float(n_electrons),
        int(config.max_iter),
        float(config.density_tol), float(config.comm_tol),
        float(config.mixing), float(config.level_shift),
        config.project_fn,
        list(int(s) for s in (config.block_sizes or ())),
        str(config.acceleration),
        int(config.diis_size), int(config.diis_start),
        float(config.diis_damping),
        float(config.trust_radius),
        bool(config.return_density), bool(config.return_fock),
    )
    n_iter = int(iterations)
    is_conv = bool(converged)
    if is_conv:
        message = f"converged in {n_iter} iterations"
    else:
        dr = float(hD[n_iter - 1]) if n_iter > 0 else float("nan")
        cr = float(hC[n_iter - 1]) if n_iter > 0 else float("nan")
        message = (
            f"stopped after {n_iter} iterations "
            f"(density_res={dr:.3e}, comm_res={cr:.3e})"
        )
    return SCFResult(
        density_matrix=density,
        fock_matrix=fock,
        energy=energy,
        chemical_potential=mu,
        iterations=n_iter,
        converged=is_conv,
        message=message,
        history={
            "E": hE,
            "density_residual": hD,
            "commutator_residual": hC,
        },
    )
