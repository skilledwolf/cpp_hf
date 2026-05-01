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

    def __post_init__(self) -> None:
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive")


@dataclass(frozen=True)
class SCFResult:
    density_matrix: Any
    fock_matrix: Any
    energy: Any
    chemical_potential: Any
    iterations: int
    converged: bool
    message: str
    history: dict[str, Any] = field(default_factory=dict)


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
