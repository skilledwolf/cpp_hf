"""SCF iteration through the C++ ``solve_scf`` loop."""

from __future__ import annotations

import numpy as np

from ...config import SCFConfig, SCFResult
from ...kernel import validate_temperature
from ._compat import _native, native_required
from ._lower import kernel_args
from .utils import validate_electron_count


def solve_scf(
    kernel,
    P0,
    n_electrons: float,
    *,
    config: SCFConfig | None = None,
) -> SCFResult:
    """Roothaan SCF; see :class:`hartreefock.SCFConfig` for the schemes."""
    if config is None:
        config = SCFConfig()
    P0 = np.asarray(P0)
    if P0.shape != kernel.h.shape:
        raise ValueError(f"P0 must have shape {tuple(kernel.h.shape)}, got {tuple(P0.shape)}.")
    if config.acceleration in ("anderson", "rdiis") and config.level_shift != 0.0:
        raise ValueError("Native full-residual acceleration requires level_shift=0")
    validate_electron_count(kernel.w2d, kernel.h.shape[-1], n_electrons)
    validate_temperature(kernel.T, context="kernel.T")
    native_required()

    P0 = np.ascontiguousarray(P0, dtype=np.complex128)
    diagnostics: dict[str, int] = {}
    args = kernel_args(kernel)
    args["diagnostics"] = diagnostics
    args["preconditioner"] = config.preconditioner
    args["safeguard"] = bool(config.safeguard)
    density, fock, energy, mu, iterations, converged, hE, hF, hD, hC = _native.solve_scf(
        args,
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
        density=density,
        fock=fock,
        energy=energy,
        mu=mu,
        n_iter=n_iter,
        converged=is_conv,
        message=message,
        history={
            "E": hE,
            "free_energy": hF,
            "density_residual": hD,
            "commutator_residual": hC,
            "fock_builds": diagnostics.get("fock_builds"),
        },
    )
