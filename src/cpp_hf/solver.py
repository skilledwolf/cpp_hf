"""Direct minimization solver wrapper.

A thin Python layer over ``cpp_hf._native.solve_dm``: prepares the kernel as
a dict of contiguous ``complex128`` / ``float64`` arrays, validates inputs,
and unpacks the C++ tuple back into a :class:`SolveResult`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral
from typing import Any, NamedTuple

import numpy as np

from ._compat import _native, native_required
from .utils import validate_electron_count, validate_temperature


@dataclass(frozen=True)
class SolverConfig:
    """Configuration for the direct-minimization solver.

    ``block_sizes`` enables a checked block-diagonal acceleration for problems
    whose Fock matrices remain block diagonal in the original orbital basis.
    The solver falls back to the full dense path if a generated Fock matrix has
    off-block entries above numerical tolerance.

    For CG and Newton, ``project_fn`` must be an orthogonal, trace-preserving
    symmetry projection onto a matrix algebra, so matrix functions preserve
    the constrained space. The complete Fock gradient is
    projected, including the bare Hamiltonian. Use ``optimizer="auto"`` for
    other linear orthogonal projections that preserve admissible densities.

    Stopping criteria: when ``tol_grad > 0`` it is the sole and *sufficient*
    stop for the optimizers — the run ends (``converged=True``) as soon as
    the larger of the orbital-gradient norm and the full finite-temperature
    Fermi--Dirac density fixed-point residual (the quantity logged in
    ``history["grad_norm"]``) drops to ``tol_grad``.  With ``tol_grad == 0``
    (default) the CG path stops on the windowed-free-energy criterion
    (``tol_E`` / ``plateau_window``) and the Newton/auto paths fall back to an
    internal 1e-6 stationarity tolerance.
    """

    max_iter: int = 200
    tol_E: float = 1e-7
    tol_grad: float = 0.0
    denom_scale: float = 1e-3
    max_step: float = 0.6
    cg_restart: int = 10
    bt_shrink: float = 0.5
    bt_max: int = 8
    mu_maxiter: int = 50
    # Free-energy convergence judged over a sliding window: stop when F improves by
    # < tol_E over the last `plateau_window` iters (robust to per-step CG noise
    # that makes a single-iteration |dF| test stop early).  0 = per-iteration.
    plateau_window: int = 5
    # "cg": preconditioned Riemannian CG (legacy default).
    # "newton": coupled orbital/occupation trust-region Newton, using one
    # interaction build per Hessian-vector product.
    # "auto": safeguarded full-residual DIIS, reserving 20% of the iteration
    # budget for Newton recovery when needed. With a symmetry callback,
    # auto keeps the entire solve in the projected density representation.
    optimizer: str = "cg"
    tr_delta0: float = 0.5      # Newton: initial trust radius
    tr_cg_max: int = 20         # Newton: max Steihaug inner CG iterations / outer step
    # Deflation (optimizer="newton" only): add a repulsive Gaussian bias around
    # each density in `deflation_targets` so the solve is pushed out of those
    # basins and converges to a DISTINCT HF solution.  `deflation_targets` has
    # shape (n_solutions, nk1, nk2, nb, nb) — or None/empty for no deflation;
    # `deflation_sigma` is the bias height (0 = off) and `deflation_length` its
    # width in weighted-Frobenius density distance.  Usually driven by
    # `solve_deflated` rather than set by hand.
    deflation_targets: Any = None
    deflation_sigma: float = 0.0
    deflation_length: float = 1.0
    block_sizes: tuple[int, ...] | None = None
    project_fn: Any = None
    return_Q: bool = True
    return_density: bool = True
    return_fock: bool = True

    def __post_init__(self) -> None:
        finite_fields = (
            "tol_E", "tol_grad", "denom_scale", "max_step", "bt_shrink",
            "tr_delta0", "deflation_sigma", "deflation_length",
        )
        for name in finite_fields:
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        integer_fields = (
            "max_iter", "cg_restart", "bt_max", "mu_maxiter",
            "plateau_window", "tr_cg_max",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be an integer")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if self.tol_E < 0.0:
            raise ValueError("tol_E must be non-negative")
        if self.tol_grad < 0.0:
            raise ValueError("tol_grad must be non-negative")
        if self.denom_scale < 0.0:
            raise ValueError("denom_scale must be non-negative")
        if self.max_step <= 0.0:
            raise ValueError("max_step must be positive")
        if self.cg_restart <= 0:
            raise ValueError("cg_restart must be positive")
        if not 0.0 < self.bt_shrink < 1.0:
            raise ValueError("bt_shrink must be strictly between 0 and 1")
        if self.bt_max <= 0:
            raise ValueError("bt_max must be positive")
        if self.mu_maxiter <= 0:
            raise ValueError("mu_maxiter must be positive")
        if self.plateau_window < 0:
            raise ValueError("plateau_window must be non-negative")
        if str(self.optimizer).lower() not in ("cg", "newton", "auto"):
            raise ValueError(
                f"optimizer must be cg, newton, or auto, got {self.optimizer!r}"
            )
        if self.tr_delta0 <= 0.0:
            raise ValueError("tr_delta0 must be positive")
        if self.tr_cg_max <= 0:
            raise ValueError("tr_cg_max must be positive")
        if self.deflation_sigma < 0.0:
            raise ValueError("deflation_sigma must be non-negative")
        if self.deflation_length <= 0.0:
            raise ValueError("deflation_length must be positive")
        if self.block_sizes is not None:
            if any(
                isinstance(size, bool)
                or not isinstance(size, Integral)
                or size <= 0
                for size in self.block_sizes
            ):
                raise ValueError("block_sizes entries must be positive integers")
        if self.project_fn is not None and not callable(self.project_fn):
            raise ValueError("project_fn must be callable or None")


class SolveResult(NamedTuple):
    """Result of :func:`solve_direct_minimization`.

    ``history`` holds optimizer-internal per-iteration logs: ``"E"`` is the
    internal HF energy E[P], ``"free_energy"`` is the thermodynamic objective
    E-T*S (including the optional Newton deflation bias), and
    ``"grad_norm"`` is the larger of the weighted orbital-gradient norm and
    the full finite-temperature density fixed-point residual.  ``energy`` is
    recomputed after finalisation and is always the internal HF energy.
    """

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


_OPTIMIZER_CODES = {"cg": 0, "newton": 1, "auto": 2}


def _optimizer_code(name: str) -> int:
    try:
        return _OPTIMIZER_CODES[str(name).lower()]
    except KeyError:
        raise ValueError(
            f"Unknown optimizer {name!r}; expected one of "
            f"{sorted(_OPTIMIZER_CODES)}."
        ) from None


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
    if config is None:
        config = SolverConfig()
    P0_arr = np.asarray(P0)
    if P0_arr.shape != kernel.h.shape:
        raise ValueError(
            f"P0 must have shape {tuple(kernel.h.shape)}, "
            f"got {tuple(P0_arr.shape)}."
        )
    validate_electron_count(
        kernel.w2d, kernel.h.shape[-1], n_electrons, context="n_electrons",
    )
    validate_temperature(kernel.T, context="kernel.T")
    native_required()
    P0 = np.ascontiguousarray(P0_arr, dtype=np.complex128)

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
    native_args = _kernel_args_for_native(kernel)
    native_args["diagnostics"] = diagnostics
    Q, p, density, fock, mu, energy, n_iter, converged, hE, hF, hG = _native.solve_dm(
        native_args,
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
        history={"E": hE, "free_energy": hF, "grad_norm": hG, "fock_builds": diagnostics.get("fock_builds")},
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
