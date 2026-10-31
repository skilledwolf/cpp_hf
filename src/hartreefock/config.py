"""Solver configurations and result types shared by every backend."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from numbers import Integral
from typing import Any, Callable, NamedTuple

_OPTIMIZERS = ("cg", "newton")
_ACCELERATIONS = ("linear", "diis", "oda", "rdiis", "anderson")


def _check_finite(cfg, names):
    for name in names:
        if not math.isfinite(float(getattr(cfg, name))):
            raise ValueError(f"{name} must be finite")


def _check_int(cfg, names):
    for name in names:
        value = getattr(cfg, name)
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"{name} must be an integer")


def _check_blocks(block_sizes):
    if block_sizes is not None and any(
        isinstance(s, bool) or not isinstance(s, Integral) or s <= 0 for s in block_sizes
    ):
        raise ValueError("block_sizes entries must be positive integers")


@dataclass(frozen=True, kw_only=True)
class SolverConfig:
    """Direct minimization (Riemannian CG or trust-region Newton).

    Stopping: with ``tol_grad > 0`` the stationarity norm logged in
    ``history["grad_norm"]`` is the sole and sufficient stop and ``tol_E`` is
    inactive. With ``tol_grad == 0`` CG stops when the free energy improves
    by less than ``tol_E`` over ``plateau_window`` iterations (0 = per
    iteration), and Newton uses an internal 1e-6 stationarity tolerance.

    ``optimizer="newton"`` is a Steihaug truncated-CG trust region on the
    joint (Q, p) response Hessian; it ignores the CG-only knobs ``max_step``,
    ``cg_restart``, ``bt_shrink`` and ``bt_max`` and needs float64.

    Deflation (Newton only) adds a Gaussian bias of height
    ``deflation_sigma`` and width ``deflation_length`` around each density
    in ``deflation_targets`` (shape ``(n, nk1, nk2, nb, nb)``); it is
    normally driven by :func:`hartreefock.solve_deflated`.

    ``block_sizes`` enables a checked block-diagonal eigensolver path.
    ``project_fn`` must be a linear, orthogonal, trace-preserving projection
    onto a matrix algebra (see :mod:`hartreefock.symmetry`).
    ``return_Q``, ``return_density`` and ``return_fock`` set to False drop the
    large arrays from the result (they come back as None).
    """

    max_iter: int = 200
    tol_E: float = 1e-7
    tol_grad: float = 0.0
    plateau_window: int = 5
    denom_scale: float = 1e-3
    max_step: float = 0.6
    cg_restart: int = 10
    bt_shrink: float = 0.5
    bt_max: int = 8
    mu_maxiter: int = 50
    optimizer: str = "cg"
    tr_delta0: float = 0.5
    tr_cg_max: int = 20
    deflation_targets: Any = None
    deflation_sigma: float = 0.0
    deflation_length: float = 1.0
    block_sizes: tuple[int, ...] | None = None
    project_fn: Callable[[Any], Any] | None = None
    return_Q: bool = True
    return_density: bool = True
    return_fock: bool = True

    def __post_init__(self) -> None:
        _check_finite(
            self,
            (
                "tol_E",
                "tol_grad",
                "denom_scale",
                "max_step",
                "bt_shrink",
                "tr_delta0",
                "deflation_sigma",
                "deflation_length",
            ),
        )
        _check_int(
            self, ("max_iter", "cg_restart", "bt_max", "mu_maxiter", "plateau_window", "tr_cg_max")
        )
        positive = ("max_iter", "max_step", "cg_restart", "bt_max", "mu_maxiter")
        positive += ("tr_delta0", "tr_cg_max", "deflation_length")
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("tol_E", "tol_grad", "denom_scale", "plateau_window", "deflation_sigma"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if not 0.0 < self.bt_shrink < 1.0:
            raise ValueError("bt_shrink must be strictly between 0 and 1")
        if str(self.optimizer).lower() not in _OPTIMIZERS:
            raise ValueError(f"optimizer must be 'cg' or 'newton', got {self.optimizer!r}")
        _check_blocks(self.block_sizes)
        if self.project_fn is not None and not callable(self.project_fn):
            raise ValueError("project_fn must be callable or None")


@dataclass(frozen=True, kw_only=True)
class SCFConfig:
    """Roothaan SCF iteration.

    ``acceleration``: ``"linear"`` mixing, ``"diis"`` (Pulay commutator DIIS
    on the Fock), ``"oda"`` (optimal damping: exact line search at T = 0,
    free-energy backtracking at T > 0), or the full-residual,
    free-energy-safeguarded ``"rdiis"`` (Fock) and ``"anderson"`` (density)
    schemes. ``diis_damping`` blends the extrapolated and current Fock;
    ``trust_radius`` (weighted Frobenius norm, 0 = off) clips each density
    step. ``safeguard`` lets RDIIS/Anderson accept free-energy descent within
    floating-point objective noise. ``return_density`` / ``return_fock`` set
    to False drop those arrays from the result.
    """

    max_iter: int = 200
    density_tol: float = 1e-7
    comm_tol: float = 1e-6
    mixing: float = 0.5
    level_shift: float = 0.0
    acceleration: str = "linear"
    diis_size: int = 6
    diis_start: int = 2
    diis_damping: float = 1.0
    trust_radius: float = 0.0
    safeguard: bool = True
    block_sizes: tuple[int, ...] | None = None
    project_fn: Callable[[Any], Any] | None = None
    return_density: bool = True
    return_fock: bool = True

    def __post_init__(self) -> None:
        _check_finite(
            self, ("density_tol", "comm_tol", "mixing", "level_shift", "diis_damping", "trust_radius")
        )
        _check_int(self, ("max_iter", "diis_size", "diis_start"))
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if self.acceleration not in _ACCELERATIONS:
            raise ValueError(
                f"acceleration must be one of {_ACCELERATIONS}, got {self.acceleration!r}"
            )
        if self.diis_size <= 0:
            raise ValueError("diis_size must be positive")
        for name in ("density_tol", "comm_tol", "diis_start", "trust_radius"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        for name in ("mixing", "diis_damping"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if not isinstance(self.safeguard, bool):
            raise ValueError("safeguard must be a bool")
        _check_blocks(self.block_sizes)
        if self.project_fn is not None and not callable(self.project_fn):
            raise ValueError("project_fn must be callable or None")


class SolveResult(NamedTuple):
    """Direct-minimization result.

    ``history["E"]`` is the HF energy (without -T*S) at each iteration's
    pre-step density and ``history["grad_norm"]`` the stationarity norm the
    ``tol_grad`` stop tests. ``energy`` is recomputed after finalisation, so
    compare ``energy`` (not history tails) across runs and optimizers.
    """

    Q: Any
    p: Any
    mu: Any
    density: Any
    fock: Any
    energy: Any
    n_iter: Any
    converged: Any
    history: dict


@dataclass(frozen=True)
class SCFResult:
    """SCF result; ``history`` holds per-iteration residuals and energies."""

    density: Any
    fock: Any
    energy: Any
    mu: Any
    n_iter: int
    converged: bool
    message: str
    history: dict[str, Any] = field(default_factory=dict)


def drop_unrequested(result, config):
    """Replace result arrays the config did not ask for with None."""
    names = ("Q", "density", "fock")
    drop = {n: None for n in names if hasattr(result, n) and not getattr(config, f"return_{n}", True)}
    if not drop:
        return result
    if isinstance(result, tuple):
        return result._replace(**drop)
    return replace(result, **drop)


class ContinuationResult(NamedTuple):
    """Coarse and fine stage results plus the resampled fine-grid seed."""

    coarse: SolveResult | SCFResult
    fine: SolveResult | SCFResult
    P0_fine: Any


class DeflatedResult(NamedTuple):
    """Distinct HF solutions found by :func:`hartreefock.solve_deflated`."""

    solutions: list
    energies: Any
    best: Any
    n_found: int
