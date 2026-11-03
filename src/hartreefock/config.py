"""Solver configurations and result types."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, fields
from numbers import Integral
from typing import Any, ClassVar, NamedTuple

_POSITIVE = {"max_iter", "diis_size", "max_step", "cg_restart", "bt_max", "mu_maxiter",
             "tr_delta0", "tr_cg_max", "deflation_length"}
_UNIT = {"mixing", "diis_damping"}


@dataclass(frozen=True, kw_only=True)
class _Config:
    """Options every method reads: ``max_iter``; ``project_fn``, a linear,
    orthogonal, trace-preserving projection onto a matrix algebra applied to
    every density and Fock matrix; and ``return_*=False``, which drops that
    array from the result."""

    method: ClassVar[str]
    roothaan: ClassVar[bool]

    max_iter: int = 200
    project_fn: Callable[[Any], Any] | None = None
    return_Q: bool = True
    return_density: bool = True
    return_fock: bool = True

    def __post_init__(self) -> None:
        for f in fields(self):
            name, value = f.name, getattr(self, f.name)
            if f.type == "int" and (isinstance(value, bool) or not isinstance(value, Integral)):
                raise ValueError(f"{name} must be an integer")
            if f.type == "bool" and not isinstance(value, bool):
                raise ValueError(f"{name} must be a bool")
            if f.type in ("int", "float"):
                if not math.isfinite(float(value)):
                    raise ValueError(f"{name} must be finite")
                if name in _POSITIVE and value <= 0:
                    raise ValueError(f"{name} must be positive")
                if value < 0:
                    raise ValueError(f"{name} must be non-negative")
                if name in _UNIT and value > 1.0:
                    raise ValueError(f"{name} must be between 0 and 1")
        if self.project_fn is not None and not callable(self.project_fn):
            raise ValueError("project_fn must be callable or None")


@dataclass(frozen=True, kw_only=True)
class _Roothaan(_Config):
    """Roothaan iteration: diagonalize the Fock matrix, refill, repeat."""

    roothaan: ClassVar[bool] = True

    density_tol: float = 1e-7
    comm_tol: float = 1e-6
    level_shift: float = 0.0
    trust_radius: float = 0.0


@dataclass(frozen=True, kw_only=True)
class RDIIS(_Roothaan):
    """Roothaan iteration with Pulay extrapolation of the Fock matrix, fitted
    to the full residual ``f(F[P]) - P`` and guarding the free energy; the
    default method.

    Attributes
    ----------
    max_iter : int
        Iteration limit.
    density_tol, comm_tol : float
        Stop when the density change and the commutator ``[F, P]`` are below
        these, in the weighted Frobenius norm.
    level_shift : float
        Shift of the virtual levels, which damps charge sloshing.
    trust_radius : float
        Largest density step in the weighted Frobenius norm; 0 turns it off.
    diis_size, diis_start, diis_damping : int, int, float
        History length, the iteration it starts at, and the blend of the
        extrapolated and current Fock matrices.
    safeguard : bool
        Accept a free-energy rise within floating-point noise.
    project_fn, return_Q, return_density, return_fock
        As for every method: a symmetry projection (see
        :mod:`hartreefock.symmetry`) and which arrays the result keeps.
    """

    method: ClassVar[str] = "rdiis"

    diis_size: int = 6
    diis_start: int = 2
    diis_damping: float = 1.0
    safeguard: bool = True


@dataclass(frozen=True, kw_only=True)
class Linear(_Roothaan):
    """Roothaan iteration with linear mixing.

    Attributes
    ----------
    mixing : float
        Fraction of the new density taken per step.
    max_iter, density_tol, comm_tol, level_shift, trust_radius
        As for :class:`RDIIS`.
    project_fn, return_Q, return_density, return_fock
        As for every method.
    """

    method: ClassVar[str] = "linear"

    mixing: float = 0.5


@dataclass(frozen=True, kw_only=True)
class _Direct(_Config):
    """Direct minimization of the free energy."""

    roothaan: ClassVar[bool] = False

    tol_grad: float = 0.0
    denom_scale: float = 1e-3
    mu_maxiter: int = 50


@dataclass(frozen=True, kw_only=True)
class CG(_Direct):
    """Direct minimization of the free energy by Riemannian conjugate
    gradients with free-energy backtracking.

    Attributes
    ----------
    max_iter : int
        Iteration limit.
    tol_grad : float
        With ``tol_grad > 0`` the stationarity norm in ``history["grad_norm"]``
        is the stopping test.
    tol_E, plateau_window : float, int
        With ``tol_grad == 0``, stop when the free energy improves by less
        than ``tol_E`` over ``plateau_window`` iterations.
    denom_scale : float
        Regularization of the occupation denominators.
    mu_maxiter : int
        Iteration cap of the chemical-potential solve.
    max_step, cg_restart, bt_shrink, bt_max : float, int, float, int
        The largest step, the restart period, and the backtracking factor
        and count.
    project_fn, return_Q, return_density, return_fock
        As for every method.
    """

    method: ClassVar[str] = "cg"

    tol_E: float = 1e-7
    plateau_window: int = 5
    max_step: float = 0.6
    cg_restart: int = 10
    bt_shrink: float = 0.5
    bt_max: int = 8

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0.0 < self.bt_shrink < 1.0:
            raise ValueError("bt_shrink must be strictly between 0 and 1")


@dataclass(frozen=True, kw_only=True)
class Newton(_Direct):
    """Direct minimization by a Steihaug truncated-CG trust region on the
    exact (Q, p) Hessian; needs float64.

    Attributes
    ----------
    max_iter : int
        Iteration limit.
    tol_grad : float
        Stationarity tolerance; 0 means 1e-6.
    denom_scale, mu_maxiter : float, int
        As for :class:`CG`.
    tr_delta0, tr_cg_max : float, int
        Initial trust radius and inner CG iterations.
    deflation_targets, deflation_sigma, deflation_length : array, float, float
        A Gaussian bias of height ``deflation_sigma`` and width
        ``deflation_length`` around each density in ``deflation_targets``
        (shape ``(n, nk1, nk2, nb, nb)``); normally set by
        :func:`hartreefock.solve_deflated`.
    project_fn, return_Q, return_density, return_fock
        As for every method.
    """

    method: ClassVar[str] = "newton"

    tr_delta0: float = 0.5
    tr_cg_max: int = 20
    deflation_targets: Any = None
    deflation_sigma: float = 0.0
    deflation_length: float = 1.0


METHODS = {cls.method: cls for cls in (RDIIS, Linear, CG, Newton)}
"""The config class of each method name, for ``METHODS[name](**options)``."""


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
