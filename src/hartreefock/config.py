"""Solver configurations and result types."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral
from typing import Any, NamedTuple

ROOTHAAN_METHODS = ("rdiis", "anderson", "oda", "linear")
METHODS = (*ROOTHAAN_METHODS, "cg", "newton")


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
    """Options of :func:`hartreefock.solve`: the method and its stopping rule.

    ``method`` picks one of two families; each reads only its own options.

    * Roothaan iteration (``"rdiis"``, ``"anderson"``, ``"oda"``,
      ``"linear"``): diagonalize the Fock matrix, refill, repeat; stops when
      the density change is below ``density_tol`` and the commutator
      ``[F, P]`` below ``comm_tol``.
    * Direct minimization of the free energy (``"cg"``, ``"newton"``): with
      ``tol_grad > 0`` the stationarity norm in ``history["grad_norm"]`` is
      the stopping test and ``tol_E`` is inactive; with ``tol_grad == 0`` CG
      stops when the free energy improves by less than ``tol_E`` over
      ``plateau_window`` iterations, and Newton uses an internal 1e-6
      stationarity tolerance.

    Attributes
    ----------
    method : {"rdiis", "anderson", "oda", "linear", "cg", "newton"}
        ``"rdiis"`` (the default, Pulay extrapolation of the Fock) and
        ``"anderson"`` (of the density) fit the full residual ``f(F[P]) - P``
        and guard the free energy; ``"oda"``, optimal damping (exact line
        search at T = 0, free-energy backtracking at T > 0); ``"linear"``
        mixing. ``"cg"``: Riemannian conjugate gradients. ``"newton"``: a
        Steihaug truncated-CG trust region on the exact (Q, p) Hessian; needs
        float64.
    max_iter : int
        Iteration limit.
    density_tol, comm_tol : float
        Roothaan stopping tolerances, in the weighted Frobenius norm.
    mixing : float
        Fraction of the new density taken per step with ``"linear"``, and
        Anderson's base update.
    level_shift : float
        Shift of the virtual levels, which damps charge sloshing.
    diis_size, diis_start, diis_damping : int, int, float
        RDIIS/Anderson history length, the iteration they start at, and the
        blend of the extrapolated and current Fock matrices.
    trust_radius : float
        Largest Roothaan density step in the weighted Frobenius norm; 0 turns
        it off.
    safeguard : bool
        Let RDIIS and Anderson accept a free-energy rise within floating-point
        noise of the objective.
    tol_E, tol_grad, plateau_window : float, float, int
        The direct-minimization stopping rule above.
    denom_scale, max_step, cg_restart, bt_shrink, bt_max : float, float, int, float, int
        CG step control: occupation-denominator regularization, the largest
        step, the restart period, and the backtracking factor and count.
    mu_maxiter : int
        Iterations of the chemical-potential bisection (direct minimization).
    tr_delta0, tr_cg_max : float, int
        Newton's initial trust radius and inner CG iterations.
    deflation_targets, deflation_sigma, deflation_length : array, float, float
        A Gaussian bias of height ``deflation_sigma`` and width
        ``deflation_length`` around each density in ``deflation_targets``
        (shape ``(n, nk1, nk2, nb, nb)``), Newton only; normally set by
        :func:`hartreefock.solve_deflated`.
    block_sizes : tuple of int, optional
        Block sizes of a block-diagonal Fock matrix, for a faster eigensolver
        (direct minimization).
    project_fn : callable, optional
        A linear, orthogonal, trace-preserving projection onto a matrix algebra,
        applied to every density and Fock matrix (see :mod:`hartreefock.symmetry`).
    return_Q, return_density, return_fock : bool
        ``False`` drops that array from the result (it comes back as ``None``).
    """

    method: str = "rdiis"
    max_iter: int = 200
    density_tol: float = 1e-7
    comm_tol: float = 1e-6
    mixing: float = 0.5
    level_shift: float = 0.0
    diis_size: int = 6
    diis_start: int = 2
    diis_damping: float = 1.0
    trust_radius: float = 0.0
    safeguard: bool = True
    tol_E: float = 1e-7
    tol_grad: float = 0.0
    plateau_window: int = 5
    denom_scale: float = 1e-3
    max_step: float = 0.6
    cg_restart: int = 10
    bt_shrink: float = 0.5
    bt_max: int = 8
    mu_maxiter: int = 50
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
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {self.method!r}")
        _check_finite(
            self,
            (
                "density_tol", "comm_tol", "mixing", "level_shift", "diis_damping",
                "trust_radius", "tol_E", "tol_grad", "denom_scale", "max_step",
                "bt_shrink", "tr_delta0", "deflation_sigma", "deflation_length",
            ),
        )
        _check_int(
            self,
            (
                "max_iter", "diis_size", "diis_start", "cg_restart", "bt_max",
                "mu_maxiter", "plateau_window", "tr_cg_max",
            ),
        )
        positive = ("max_iter", "diis_size", "max_step", "cg_restart", "bt_max", "mu_maxiter")
        positive += ("tr_delta0", "tr_cg_max", "deflation_length")
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        non_negative = ("density_tol", "comm_tol", "diis_start", "trust_radius", "tol_E")
        non_negative += ("tol_grad", "denom_scale", "plateau_window", "deflation_sigma")
        for name in non_negative:
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        for name in ("mixing", "diis_damping"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if not 0.0 < self.bt_shrink < 1.0:
            raise ValueError("bt_shrink must be strictly between 0 and 1")
        if not isinstance(self.safeguard, bool):
            raise ValueError("safeguard must be a bool")
        _check_blocks(self.block_sizes)
        if self.project_fn is not None and not callable(self.project_fn):
            raise ValueError("project_fn must be callable or None")

    @property
    def roothaan(self) -> bool:
        """Whether ``method`` is a Roothaan iteration (else direct minimization)."""
        return self.method in ROOTHAAN_METHODS


class SolveResult(NamedTuple):
    """Result of :func:`hartreefock.solve`.

    Compare ``energy``, which is recomputed after the last step, across runs
    and methods, not the tail of ``history["E"]``.

    Attributes
    ----------
    density : (nk1, nk2, nb, nb) array or None
        The density matrix.
    fock : (nk1, nk2, nb, nb) array or None
        The Fock matrix at the density.
    energy : float
        Hartree-Fock energy summed with the weights (without ``-TS``),
        ``Tr(hP) + ½ Tr(V_HF[P - P_ref] (P - P_ref))``; with an embedded
        reference the standard ``Tr(hP) + ½ Tr(V_HF[P] P)``.
    mu : float
        Chemical potential.
    n_iter : int
        Iterations taken.
    converged : bool
        Whether the stopping rule was met.
    history : dict
        Per iteration ``"E"`` and ``"free_energy"``; Roothaan methods add
        ``"density_residual"`` and ``"commutator_residual"``, direct
        minimization ``"grad_norm"``, the stationarity norm ``tol_grad`` tests.
    Q : (nk1, nk2, nb, nb) array or None
        Orbitals of the final density (direct minimization only).
    p : (nk1, nk2, nb) array or None
        Their occupations (direct minimization only).
    coarse : SolveResult or None
        The coarse-mesh solve that seeded this one (``hf.solve(coarse=)``).
    """

    density: Any
    fock: Any
    energy: Any
    mu: Any
    n_iter: Any
    converged: Any
    history: dict
    Q: Any = None
    p: Any = None
    coarse: Any = None


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
