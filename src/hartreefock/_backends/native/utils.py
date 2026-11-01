"""Input validation and k-grid resampling for the native backend."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import expit

from ...kernel import validate_temperature
from ._compat import _native, native_required


def validate_electron_count(
    weights: Any,
    nbands: int,
    n_electrons: Any,
    *,
    context: str = "n_electrons",
) -> None:
    total_weight = float(np.real(np.asarray(weights)).sum())
    target = float(np.asarray(n_electrons))
    max_electrons = total_weight * int(nbands)
    tol = max(1e-8, 1e-6 * max(1.0, abs(max_electrons)))

    if not np.isfinite(target):
        raise ValueError(f"{context} must be finite, got {target!r}.")
    if target < -tol or target > max_electrons + tol:
        raise ValueError(
            f"{context}={target} is outside the physically reachable range "
            f"[0, {max_electrons}] for {int(nbands)} bands and weights.sum()={total_weight}."
        )


def resample_kgrid(values: np.ndarray, nk: int, *, method: str = "linear") -> np.ndarray:
    """Resample a centered, periodic (nk,nk,...) k-grid array to a new nk."""
    nk = int(nk)
    x = np.asarray(values)
    if x.ndim < 2:
        raise ValueError(
            "resample_kgrid expects an array with at least 2 dimensions (nk,nk,...)."
        )
    if x.shape[0] == nk and x.shape[1] == nk:
        return x

    method = str(method).lower()
    if method != "linear":
        raise ValueError(
            f"Unsupported resample method {method!r}. Only 'linear' is supported."
        )

    native_required()
    x_c = np.ascontiguousarray(x, dtype=np.complex128)
    return _native.resample_kgrid_2d(x_c, nk)


def find_chemical_potential(bands, weights, n_electrons: float, T: float) -> float:
    """Chemical potential that puts ``n_electrons`` into ``bands`` (bisection in C++)."""
    validate_electron_count(weights, np.shape(bands)[-1], n_electrons)
    validate_temperature(T)
    target = float(np.asarray(n_electrons))
    capacity = float(np.sum(np.asarray(weights))) * int(np.shape(bands)[-1])
    if target <= 0.0:
        return -np.inf
    if target >= capacity:
        return np.inf
    native_required()
    bands = np.ascontiguousarray(bands, dtype=np.float64)
    weights = np.ascontiguousarray(weights, dtype=np.float64)
    return float(_native.find_mu_bisection(bands, weights, float(n_electrons), float(T), 0))


def density_matrix_from_fock(F, weights, n_electrons: float, T: float):
    """Fixed-particle Fermi density ``(P, mu)`` of the Hermitian part of ``F``."""
    F = np.asarray(F)
    F = 0.5 * (F + F.conj().swapaxes(-2, -1))
    eps, U = np.linalg.eigh(F)
    mu = find_chemical_potential(eps, weights, n_electrons, T)
    occ = expit((mu - eps) / float(T))
    P = (U * occ[..., None, :]) @ U.conj().swapaxes(-2, -1)
    return 0.5 * (P + P.conj().swapaxes(-2, -1)), mu
