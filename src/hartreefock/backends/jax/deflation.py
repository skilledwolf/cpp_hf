"""Deflated basin search for Hartree-Fock — find distinct self-consistent solutions.

HF is non-convex: SCF / CG / Newton all converge to whichever self-consistent
solution sits in the basin of the initial guess.  This module layers *deflation*
on the trust-region Newton solver (``optimizer="newton"``): once a solution P* is
found, a repulsive Gaussian bias around P* is added to the free energy (via
:attr:`SolverConfig.deflation_targets`), pushing a re-solve out of that basin so
it converges to a *different* solution.  Each biased solve is re-polished
unbiased, so every returned density is a true (unbiased) HF stationary point.
Results are sorted by physical energy — the lowest is the best ground-state
candidate.

The biased free energy minimised by the Newton solver is

    Omega_eff(P) = Omega(P) + sigma * sum_i exp(-d_i^2 / (2 L^2)),
    d_i^2 = sum_k w_k ||P_k - P*_{i,k}||_F^2   (weighted Frobenius; gauge-invariant)

with ``sigma = deflation_sigma`` (height) and ``L = deflation_length`` (width).

Deflation needs float64 like the rest of the Newton path (enable x64).
"""

from __future__ import annotations

from dataclasses import replace
from typing import NamedTuple

import numpy as np

from .solver import SolveResult, SolverConfig, solve


class DeflatedResult(NamedTuple):
    solutions: list          # list[SolveResult], distinct minima, ascending energy
    energies: np.ndarray     # physical energies of `solutions`, ascending
    best: SolveResult        # lowest-energy solution (ground-state candidate)
    n_found: int             # == len(solutions)


def _wnorm2(M: np.ndarray, w2d: np.ndarray) -> float:
    """sum_k w2d_k ||M_k||_F^2  (raw weighted Frobenius^2, matching the solver metric)."""
    per_k = np.sum(np.abs(M) ** 2, axis=(-2, -1))  # (nk1, nk2)
    return float(np.sum(np.asarray(w2d) * per_k))


def _wdist(A: np.ndarray, B: np.ndarray, w2d: np.ndarray) -> float:
    """sqrt(sum_k w2d_k ||A_k - B_k||_F^2) — same units as ``deflation_length``."""
    return float(np.sqrt(max(_wnorm2(np.asarray(A) - np.asarray(B), w2d), 0.0)))


def _hermitian_perturbation(shape, scale: float, rng) -> np.ndarray:
    A = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex128)
    return scale * 0.5 * (A + np.conj(np.swapaxes(A, -1, -2)))


def solve_deflated(
    kernel,
    P0,
    n_electrons: float,
    *,
    base_config: SolverConfig | None = None,
    n_solutions: int = 4,
    sigma: float | None = None,
    length: float | None = None,
    distinct_tol: float | None = None,
    max_retries: int = 4,
    sigma_growth: float = 3.0,
    perturb: float = 1e-2,
    repolish: bool = True,
    seed: int = 0,
) -> DeflatedResult:
    """Find distinct HF solutions by deflated trust-region Newton.

    Runs an initial unbiased Newton solve from ``P0``, then repeatedly re-solves
    with a repulsive bias around every solution found so far (escalating the bias
    height on failure) and re-polishes each result unbiased, collecting the
    distinct minima.

    Parameters
    ----------
    base_config:
        Template :class:`SolverConfig`; ``optimizer`` is forced to ``"newton"``
        and the deflation knobs are managed internally.  ``None`` uses defaults.
    n_solutions:
        Stop once this many distinct solutions are found (or no new basin is
        reachable within ``max_retries``).
    sigma, length:
        Bias height / width.  Auto-scaled from the first solution when ``None``
        (``length = 0.5 * ||P*_0||_w``, ``sigma = max(|E_0|, 1)``).
    distinct_tol:
        Two densities closer than this (weighted distance) are the same basin.
        Defaults to ``0.25 * length``.
    max_retries, sigma_growth:
        Per new solution, retry the biased solve up to ``max_retries`` times,
        multiplying ``sigma`` by ``sigma_growth`` each failure.
    perturb:
        Std. of the symmetry-breaking Hermitian perturbation added to the seed
        (grows with the attempt index); essential for escaping a found minimum,
        whose bias gradient vanishes exactly at its centre.
    repolish:
        Re-solve each biased result unbiased so it is a true HF stationary point.

    Returns
    -------
    DeflatedResult
        Distinct solutions sorted by ascending physical energy.
    """
    w2d = np.asarray(kernel.w2d)
    rng = np.random.default_rng(seed)

    base = base_config if base_config is not None else SolverConfig()
    # Deflation is a Newton-path feature; force the optimizer and clear any
    # stray deflation settings on the base (unbiased) config.
    newton_cfg = replace(
        base, optimizer="newton", deflation_targets=None, deflation_sigma=0.0,
    )

    P0 = np.ascontiguousarray(np.asarray(P0), dtype=np.complex128)

    # 1. initial unbiased solve
    r0 = solve(kernel, P0, float(n_electrons), config=newton_cfg)
    solutions = [r0]

    # 2. auto-scale bias width/height from the first solution
    if length is None:
        d0 = float(np.sqrt(max(_wnorm2(np.asarray(r0.density), w2d), 0.0)))
        length = 0.5 * d0 if d0 > 0.0 else 1.0
    if sigma is None:
        sigma = max(abs(float(r0.energy)), 1.0)
    if distinct_tol is None:
        distinct_tol = 0.25 * length

    # 3. deflation loop: bias against all found, re-solve, re-polish, keep distinct
    while len(solutions) < n_solutions:
        found_new = False
        cur_sigma = float(sigma)
        for attempt in range(max_retries):
            targets = np.stack([np.asarray(s.density) for s in solutions])
            biased_cfg = replace(
                newton_cfg,
                deflation_targets=targets,
                deflation_sigma=cur_sigma,
                deflation_length=float(length),
            )
            # Seed from P0 (the principled deflation seed) plus a symmetry-breaking
            # perturbation that grows with the attempt index.
            seed_P = P0 + _hermitian_perturbation(P0.shape, perturb * (attempt + 1), rng)
            seed_P = np.ascontiguousarray(seed_P, dtype=np.complex128)

            rb = solve(kernel, seed_P, float(n_electrons), config=biased_cfg)
            cand = (
                solve(
                    kernel,
                    np.ascontiguousarray(np.asarray(rb.density), dtype=np.complex128),
                    float(n_electrons),
                    config=newton_cfg,
                )
                if repolish
                else rb
            )

            if bool(cand.converged) and all(
                _wdist(cand.density, s.density, w2d) > distinct_tol for s in solutions
            ):
                solutions.append(cand)
                found_new = True
                break
            cur_sigma *= sigma_growth  # escalate the bias and retry
        if not found_new:
            break  # no new basin reachable within the budget

    order = np.argsort([float(s.energy) for s in solutions])
    solutions = [solutions[i] for i in order]
    energies = np.array([float(s.energy) for s in solutions])
    return DeflatedResult(
        solutions=solutions,
        energies=energies,
        best=solutions[0],
        n_found=len(solutions),
    )


__all__ = ["DeflatedResult", "solve_deflated"]
