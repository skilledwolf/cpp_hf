"""Direct minimization solver wrapper.

A thin Python layer over the C++ ``solve_dm``: prepares the kernel as
a dict of contiguous ``complex128`` / ``float64`` arrays, validates inputs,
and unpacks the C++ tuple back into a :class:`SolveResult`.
"""

from __future__ import annotations

import numpy as np

from ...config import SolverConfig, SolveResult
from ...kernel import validate_temperature
from ._compat import _native, native_required
from ._lower import kernel_args
from .utils import validate_electron_count

_OPTIMIZER_CODES = {"cg": 0, "newton": 1, "auto": 2}


def _optimizer_code(name: str) -> int:
    try:
        return _OPTIMIZER_CODES[str(name).lower()]
    except KeyError:
        raise ValueError(
            f"Unknown optimizer {name!r}; expected one of "
            f"{sorted(_OPTIMIZER_CODES)}."
        ) from None


def solve(
    kernel,
    P0,
    n_electrons: float,
    *,
    config: SolverConfig | None = None,
) -> SolveResult:
    """Direct-minimization Hartree-Fock solver.

    ``optimizer``: preconditioned Riemannian CG on Stiefel × capped simplex
    (``"cg"``), coupled trust-region Newton (``"newton"``), or safeguarded
    full-residual DIIS with Newton recovery (``"auto"``).  Runs in C++.
    """
    if config is None:
        config = SolverConfig()
    P0 = np.asarray(P0)
    if P0.shape != kernel.h.shape:
        raise ValueError(f"P0 must have shape {tuple(kernel.h.shape)}, got {tuple(P0.shape)}.")
    validate_electron_count(kernel.w2d, kernel.h.shape[-1], n_electrons, context="n_electrons")
    validate_temperature(kernel.T, context="kernel.T")
    native_required()
    P0 = np.ascontiguousarray(P0, dtype=np.complex128)

    block_sizes = list(int(s) for s in (config.block_sizes or ()))
    optimizer_code = _optimizer_code(config.optimizer)

    # Deflation targets -> (n_solutions, nk1, nk2, nb, nb) complex128 (empty array
    # when no deflation).  The native binding infers n_deflation from shape[0].
    nk1, nk2, nb = kernel.h.shape[0], kernel.h.shape[1], kernel.h.shape[-1]
    if config.deflation_targets is None:
        defl_arr = np.empty((0, nk1, nk2, nb, nb), dtype=np.complex128)
    else:
        defl_arr = np.ascontiguousarray(config.deflation_targets, dtype=np.complex128)
        if defl_arr.ndim == 4:
            defl_arr = defl_arr[None, ...]
        if defl_arr.ndim != 5 or defl_arr.shape[1:] != (nk1, nk2, nb, nb):
            raise ValueError(
                "deflation_targets must have shape (n_solutions, nk1, nk2, nb, nb) "
                f"matching the kernel; got {tuple(defl_arr.shape)}."
            )
    defl_arr = np.ascontiguousarray(defl_arr, dtype=np.complex128)
    if defl_arr.shape[0] > 0 and float(config.deflation_sigma) > 0.0 \
            and str(config.optimizer).lower() != "newton":
        raise ValueError(
            "deflation is only supported with optimizer='newton' "
            f"(got optimizer={config.optimizer!r})."
        )

    diagnostics: dict[str, int] = {}
    args = kernel_args(kernel)
    args["diagnostics"] = diagnostics
    Q, p, density, fock, mu, energy, n_iter, converged, hE, hF, hG = _native.solve_dm(
        args,
        P0, float(n_electrons),
        int(config.max_iter), float(config.tol_E), float(config.tol_grad),
        float(config.max_step), float(config.bt_shrink), float(config.denom_scale),
        int(config.bt_max), int(config.cg_restart),
        int(config.plateau_window), int(config.mu_maxiter),
        int(optimizer_code),
        float(config.tr_delta0), int(config.tr_cg_max),
        defl_arr, float(config.deflation_sigma), float(config.deflation_length),
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
        history={
            "E": hE,
            "free_energy": hF,
            "grad_norm": hG,
            "fock_builds": diagnostics.get("fock_builds"),
        },
    )



def _cayley_retract(d: np.ndarray, tau: float) -> np.ndarray:
    """Reference LU Cayley retraction ``U = (I − τd/2)(I + τd/2)^-1``.

    Provided as a small numpy reference for the algorithm-correctness tests
    (``TestSpectralCayley``); production code uses the spectral form in
    the C++ extension.
    """
    n = d.shape[-1]
    eye = np.eye(n, dtype=d.dtype)
    A = 0.5 * float(tau) * d
    return np.linalg.solve(eye + A, eye - A)
