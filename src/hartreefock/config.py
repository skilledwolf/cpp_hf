"""Solver configurations and result types."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from numbers import Integral
from typing import Any, NamedTuple

_OPTIMIZERS = ("cg", "newton")
_ACCELERATIONS = ("rdiis", "anderson", "oda", "linear")


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
    """Options of :func:`hartreefock.solve`: the optimizer and its stopping rule.

    With ``tol_grad > 0`` the stationarity norm logged in
    ``history["grad_norm"]`` is the stopping test and ``tol_E`` is inactive.
    With ``tol_grad == 0`` CG stops when the free energy improves by less than
    ``tol_E`` over ``plateau_window`` iterations, and Newton uses an internal
    1e-6 stationarity tolerance.

    Attributes
    ----------
    optimizer : {"cg", "newton"}
        ``"cg"``: Riemannian conjugate gradients. ``"newton"``: a Steihaug
        truncated-CG trust region on the exact (Q, p) Hessian; needs float64
        and ignores ``max_step``, ``cg_restart``, ``bt_shrink`` and ``bt_max``.
    max_iter : int
        Iteration limit.
    tol_E, tol_grad, plateau_window : float, float, int
        The stopping rule above.
    denom_scale, max_step, cg_restart, bt_shrink, bt_max : float, float, int, float, int
        CG step control: occupation-denominator regularization, the largest
        step, the restart period, and the backtracking factor and count.
    mu_maxiter : int
        Iterations of the chemical-potential bisection.
    tr_delta0, tr_cg_max : float, int
        Newton's initial trust radius and inner CG iterations.
    deflation_targets, deflation_sigma, deflation_length : array, float, float
        A Gaussian bias of height ``deflation_sigma`` and width
        ``deflation_length`` around each density in ``deflation_targets``
        (shape ``(n, nk1, nk2, nb, nb)``), Newton only; normally set by
        :func:`hartreefock.solve_deflated`.
    block_sizes : tuple of int, optional
        Block sizes of a block-diagonal Fock matrix, for a faster eigensolver.
    project_fn : callable, optional
        A linear, orthogonal, trace-preserving projection onto a matrix algebra,
        applied to every density and Fock matrix (see :mod:`hartreefock.symmetry`).
    return_Q, return_density, return_fock : bool
        ``False`` drops that array from the result (it comes back as ``None``).
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
            raise ValueError(f"optimizer must be one of {_OPTIMIZERS}, got {self.optimizer!r}")
        _check_blocks(self.block_sizes)
        if self.project_fn is not None and not callable(self.project_fn):
            raise ValueError("project_fn must be callable or None")


@dataclass(frozen=True, kw_only=True)
class SCFConfig:
    """Options of :func:`hartreefock.solve_scf`: the acceleration and its tolerances.

    The iteration stops when the density change is below ``density_tol`` and
    the commutator ``[F, P]`` below ``comm_tol``.

    Attributes
    ----------
    acceleration : {"rdiis", "anderson", "oda", "linear"}
        ``"rdiis"`` (the default, Pulay extrapolation of the Fock) and
        ``"anderson"`` (of the density) fit the full residual ``f(F[P]) - P``
        and guard the free energy; ``"oda"``, optimal damping (exact line
        search at T = 0, free-energy backtracking at T > 0); ``"linear"``
        mixing.
    max_iter : int
        Iteration limit.
    density_tol, comm_tol : float
        The stopping tolerances above, in the weighted Frobenius norm.
    mixing : float
        Fraction of the new density taken per step with ``"linear"``.
    level_shift : float
        Shift of the virtual levels, which damps charge sloshing.
    diis_size, diis_start, diis_damping : int, int, float
        RDIIS/Anderson history length, the iteration they start at, and the
        blend of the extrapolated and current Fock matrices.
    trust_radius : float
        Largest density step in the weighted Frobenius norm; 0 turns it off.
    safeguard : bool
        Let RDIIS and Anderson accept a free-energy rise within floating-point
        noise of the objective.
    project_fn : callable, optional
        Symmetry projection, as in :class:`SolverConfig`.
    return_density, return_fock : bool
        ``False`` drops that array from the result.
    """

    max_iter: int = 200
    density_tol: float = 1e-7
    comm_tol: float = 1e-6
    mixing: float = 0.5
    level_shift: float = 0.0
    acceleration: str = "rdiis"
    diis_size: int = 6
    diis_start: int = 2
    diis_damping: float = 1.0
    trust_radius: float = 0.0
    safeguard: bool = True
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
        if self.project_fn is not None and not callable(self.project_fn):
            raise ValueError("project_fn must be callable or None")


class SolveResult(NamedTuple):
    """Result of :func:`hartreefock.solve`.

    Compare ``energy``, which is recomputed after the last step, across runs
    and optimizers, not the tail of ``history["E"]``.

    Attributes
    ----------
    Q : (nk1, nk2, nb, nb) array or None
        Orbitals of the final density.
    p : (nk1, nk2, nb) array
        Their occupations.
    mu : float
        Chemical potential.
    density : (nk1, nk2, nb, nb) array or None
        The density matrix.
    fock : (nk1, nk2, nb, nb) array or None
        The Fock matrix at the density.
    energy : float
        Hartree-Fock energy, summed with the weights (without ``-TS``).
    n_iter : int
        Iterations taken.
    converged : bool
        Whether the stopping rule was met.
    history : dict
        Per iteration: ``"E"``, the energy at the pre-step density, and
        ``"grad_norm"``, the stationarity norm that ``tol_grad`` tests.
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
    """Result of :func:`hartreefock.solve_scf`.

    Attributes
    ----------
    density : (nk1, nk2, nb, nb) array or None
        The density matrix.
    fock : (nk1, nk2, nb, nb) array or None
        The Fock matrix at the density.
    energy : float
        Hartree-Fock energy per unit of the weights,
        ``Tr(hP) + ½ Tr(V_HF[P - P_ref] (P - P_ref))``.
    mu : float
        Chemical potential.
    n_iter : int
        Iterations taken.
    converged : bool
        Whether both tolerances were met.
    message : str
        Why the iteration stopped, with the final residuals.
    history : dict
        Per iteration: ``"E"``, ``"free_energy"``, ``"density_residual"``,
        ``"commutator_residual"`` and ``"fock_builds"``.
    """

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
    """Result of :func:`hartreefock.solve_continuation`.

    Attributes
    ----------
    coarse : SolveResult or SCFResult
        The coarse-mesh solve.
    fine : SolveResult or SCFResult
        The fine-mesh solve, the answer.
    P0_fine : (nk1, nk2, nb, nb) array or None
        The coarse density resampled to the fine mesh, the seed of ``fine``.
    """

    coarse: SolveResult | SCFResult
    fine: SolveResult | SCFResult
    P0_fine: Any


class DeflatedResult(NamedTuple):
    """Result of :func:`hartreefock.solve_deflated`.

    Attributes
    ----------
    solutions : list of SolveResult
        The distinct solutions found, lowest energy first.
    energies : (n_found,) array
        Their energies.
    best : SolveResult
        The lowest, ``solutions[0]``.
    n_found : int
        How many were found.
    """

    solutions: list
    energies: Any
    best: Any
    n_found: int
