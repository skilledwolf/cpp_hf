"""Low-level physics utilities (numpy port of ``jax_hf.utils``)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._compat import (
    _native,
    batched_eigh,
    expit,
    fftn2d,
    hermitize,
    ifftn2d,
    ifftshift2d,
    native_required,
)
from .linalg import eigh, normalize_block_specs


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


def fermidirac(x: np.ndarray, T: float) -> np.ndarray:
    return expit(-np.asarray(x) / (float(T) + 1e-12))


def electron_density(P: np.ndarray) -> np.ndarray:
    return np.real(np.trace(P, axis1=-2, axis2=-1))


def density_spectrum(bands: np.ndarray, mu: float, T: float) -> np.ndarray:
    return fermidirac(bands - mu, T).sum(axis=-1)


def selfenergy_fft(
    VR: np.ndarray,
    P: np.ndarray,
    *,
    block_specs: Any | None = None,
    check_offdiag: bool | None = None,
    offdiag_atol: float = 1e-12,
    offdiag_rtol: float = 0.0,
    _apply_ifftshift: bool = True,
    hermitian_channel_packing: bool = False,
) -> np.ndarray:
    """Exchange self-energy Σ(k) with optional block-diagonal acceleration.

    Numpy port of ``jax_hf.utils.selfenergy_fft``.  Always: ``Σ(k) = -FFT⁻¹[FFT(P) · VR]``.
    """
    VR = np.asarray(VR)
    P = np.asarray(P)

    if hermitian_channel_packing and VR.shape[-2:] != (1, 1):
        raise ValueError(
            "hermitian_channel_packing requires a scalar interaction kernel "
            "with shape (..., 1, 1)."
        )

    if block_specs is None:
        native_required()
        VR_c = np.ascontiguousarray(VR, dtype=np.complex128)
        P_c = np.ascontiguousarray(P, dtype=np.complex128)
        result = _native.selfenergy_fft_full(
            VR_c, P_c,
            apply_ifftshift=False,  # ifftshift handled below
            hermitian_channel_packing=bool(hermitian_channel_packing),
        )
    else:
        specs = normalize_block_specs(block_specs)
        if not specs:
            result = _selfenergy_fft_full(
                VR, P, hermitian_channel_packing=hermitian_channel_packing
            )
        else:
            check = bool(check_offdiag) if check_offdiag is not None else True
            result = _selfenergy_fft_block_specs(
                VR,
                P,
                specs,
                check_offdiag=check,
                offdiag_atol=float(offdiag_atol),
                offdiag_rtol=float(offdiag_rtol),
                hermitian_channel_packing=hermitian_channel_packing,
            )

    if _apply_ifftshift:
        return ifftshift2d(result)
    return result


def _selfenergy_fft_scalar_hermitian_channels(
    VR: np.ndarray, P: np.ndarray
) -> np.ndarray:
    n = int(P.shape[-1])
    tri_i, tri_j = np.triu_indices(n)
    offdiag = tri_i != tri_j
    flat_idx = (tri_i * n + tri_j).astype(np.intp)
    P_flat = P.reshape(P.shape[:-2] + (n * n,))
    packed = P_flat[..., flat_idx]
    packed_fft = fftn2d(packed)
    VR_scalar = VR[..., 0, 0][..., None]
    sigma_packed = -ifftn2d(packed_fft * VR_scalar)

    out_flat = np.zeros_like(P_flat)
    out_flat[..., flat_idx] = sigma_packed

    lower_idx = (tri_j[offdiag] * n + tri_i[offdiag]).astype(np.intp)
    out_flat[..., lower_idx] = np.conj(sigma_packed[..., offdiag])
    return out_flat.reshape(P.shape)


def _selfenergy_fft_full(
    VR: np.ndarray,
    P: np.ndarray,
    *,
    hermitian_channel_packing: bool = False,
) -> np.ndarray:
    if hermitian_channel_packing:
        return _selfenergy_fft_scalar_hermitian_channels(VR, P)
    P_fft = fftn2d(P)
    return -ifftn2d(P_fft * VR)


def _slice_interaction(VR: np.ndarray, idx: slice) -> np.ndarray:
    if VR.shape[-2] == 1 and VR.shape[-1] == 1:
        return VR
    return VR[..., idx, idx]


def _take_interaction(VR: np.ndarray, idx: np.ndarray) -> np.ndarray:
    if VR.shape[-2] == 1 and VR.shape[-1] == 1:
        return VR
    return VR[..., idx[:, None], idx[None, :]]


def _selfenergy_fft_block_sizes(
    VR: np.ndarray,
    P: np.ndarray,
    block_sizes: tuple[int, ...],
    *,
    hermitian_channel_packing: bool,
) -> np.ndarray:
    n = int(P.shape[-1])
    if sum(int(s) for s in block_sizes) != n:
        raise ValueError(f"block_sizes must sum to {n}.")

    out = np.zeros_like(P)
    start = 0
    for size in block_sizes:
        stop = start + int(size)
        s = slice(start, stop)
        sigma_block = _selfenergy_fft_full(
            _slice_interaction(VR, s),
            P[..., s, s],
            hermitian_channel_packing=hermitian_channel_packing,
        )
        out[..., s, s] = sigma_block
        start = stop
    return out


def _selfenergy_fft_block_indices(
    VR: np.ndarray,
    P: np.ndarray,
    block_indices: tuple[tuple[int, ...], ...],
    *,
    hermitian_channel_packing: bool,
) -> np.ndarray:
    out = np.zeros_like(P)
    for idx in block_indices:
        idx_arr = np.asarray(idx, dtype=np.intp)
        sub = P[..., idx_arr[:, None], idx_arr[None, :]]
        sigma_block = _selfenergy_fft_full(
            _take_interaction(VR, idx_arr),
            sub,
            hermitian_channel_packing=hermitian_channel_packing,
        )
        out[..., idx_arr[:, None], idx_arr[None, :]] = sigma_block
    return out


def _selfenergy_fft_block_specs(
    VR: np.ndarray,
    P: np.ndarray,
    block_specs: tuple,
    *,
    check_offdiag: bool,
    offdiag_atol: float,
    offdiag_rtol: float,
    hermitian_channel_packing: bool,
) -> np.ndarray:
    n = int(P.shape[-1])
    if check_offdiag:
        abs_P = np.abs(P)
        scale = float(np.max(abs_P))
        tol = float(offdiag_atol) + float(offdiag_rtol) * scale

    for kind, data in block_specs:
        kind = str(kind).strip().lower()
        if kind == "sizes":
            sizes = tuple(int(x) for x in data)
            mask = np.ones((n, n), dtype=bool)
            start = 0
            for size in sizes:
                stop = start + size
                mask[start:stop, start:stop] = False
                start = stop
            ok = (not check_offdiag) or (float(np.max(np.abs(P) * mask)) <= tol)
            if ok:
                return _selfenergy_fft_block_sizes(
                    VR, P, sizes,
                    hermitian_channel_packing=hermitian_channel_packing,
                )
        elif kind == "indices":
            blocks = tuple(tuple(int(i) for i in b) for b in data)
            mask = np.ones((n, n), dtype=bool)
            for idx in blocks:
                idx_arr = np.asarray(idx, dtype=int)
                mask[np.ix_(idx_arr, idx_arr)] = False
            ok = (not check_offdiag) or (float(np.max(np.abs(P) * mask)) <= tol)
            if ok:
                return _selfenergy_fft_block_indices(
                    VR, P, blocks,
                    hermitian_channel_packing=hermitian_channel_packing,
                )
        else:
            raise ValueError("block_specs kind must be 'sizes' or 'indices'.")

    return _selfenergy_fft_full(
        VR, P, hermitian_channel_packing=hermitian_channel_packing
    )


def find_chemical_potential(
    bands: np.ndarray,
    weights: np.ndarray,
    n_electrons: float,
    T: float,
    *,
    maxiter: int | None = None,
    method: str = "bisection",
) -> np.ndarray:
    """Find μ such that ∑_k w_k Σ_j f(ε_kj − μ) = n_electrons.

    Numpy port of ``jax_hf.utils.find_chemical_potential``.  ``bisection`` is
    the safe default; ``newton`` is faster but more fragile in cold-T regimes.
    """
    validate_electron_count(weights, bands.shape[-1], n_electrons, context="n_electrons")
    method = str(method).lower()
    if method == "bisection":
        return _find_mu_bisection(bands, weights, n_electrons, T, maxiter=maxiter)
    if method == "newton":
        return _find_mu_newton(bands, weights, n_electrons, T, maxiter=maxiter)
    raise ValueError(f"method must be 'bisection' or 'newton', got {method!r}")


def _find_mu_bisection(
    bands: np.ndarray,
    weights: np.ndarray,
    n_electrons: float,
    T: float,
    *,
    maxiter: int | None = None,
) -> np.ndarray:
    bands = np.asarray(bands)
    weights = np.asarray(weights)
    real_dtype = bands.real.dtype if np.iscomplexobj(bands) else bands.dtype
    if maxiter is None:
        maxiter = 30 if real_dtype == np.float64 else 54
    Tj = max(float(T), 1e-12)

    bands_min = float(np.min(bands))
    bands_max = float(np.max(bands))
    span = bands_max - bands_min + 10.0 * max(Tj, 1e-6)
    lo = bands_min - span
    hi = bands_max + span

    w_b = weights[..., None]
    for _ in range(int(maxiter)):
        mid = 0.5 * (lo + hi)
        occ = fermidirac(bands - mid, T)
        count = float(np.sum(w_b * occ))
        too_high = count > n_electrons
        if too_high:
            hi = mid
        else:
            lo = mid
    return np.asarray(0.5 * (lo + hi), dtype=real_dtype)


def _find_mu_newton(
    bands: np.ndarray,
    weights: np.ndarray,
    n_electrons: float,
    T: float,
    *,
    maxiter: int | None = None,
) -> np.ndarray:
    bands = np.asarray(bands)
    weights = np.asarray(weights)
    real_dtype = bands.real.dtype if np.iscomplexobj(bands) else bands.dtype
    if maxiter is None:
        maxiter = 15 if real_dtype == np.float64 else 25
    Tj = max(float(T), 1e-12)
    n_target = float(n_electrons)

    bands_min = float(np.min(bands))
    bands_max = float(np.max(bands))
    span = bands_max - bands_min + 10.0 * max(Tj, 1e-6)
    lo = bands_min - span
    hi = bands_max + span
    mu = 0.5 * (lo + hi)
    w_b = weights[..., None]

    def count_and_slope(mu_val):
        x = (mu_val - bands) / Tj
        p = expit(x)
        N = float(np.sum(w_b * p))
        Z = float(np.sum(w_b * p * (1.0 - p) / Tj))
        return N, Z

    for _ in range(int(maxiter)):
        N, Z = count_and_slope(mu)
        g = N - n_target
        if g < 0:
            lo = mu
        if g > 0:
            hi = mu
        Z_safe = max(Z, 1e-18)
        mu_new = mu - g / Z_safe
        mu_bis = 0.5 * (lo + hi)
        if mu_new <= lo or mu_new >= hi or not np.isfinite(mu_new):
            mu_new = mu_bis
        mu_new = max(min(mu_new, hi), lo)
        mu = mu_new

    N_fin, _ = count_and_slope(mu)
    if abs(N_fin - n_target) > 1e-12:
        mu = 0.5 * (lo + hi)
    return np.asarray(mu, dtype=real_dtype)


def density_matrix_from_fock(
    F: np.ndarray,
    weights: np.ndarray,
    n_electrons: float,
    T: float,
    *,
    eigh_block_specs: object | None = None,
    eigh_check_offdiag: bool | None = None,
    eigh_offdiag_atol: float = 1e-12,
    eigh_offdiag_rtol: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a physical density matrix P(k) from a Hermitian Fock matrix F(k)."""
    F = hermitize(np.asarray(F))
    validate_electron_count(weights, F.shape[-1], n_electrons, context="n_electrons")
    eps, U = eigh(
        F,
        block_specs=eigh_block_specs,
        check_offdiag=eigh_check_offdiag,
        offdiag_atol=eigh_offdiag_atol,
        offdiag_rtol=eigh_offdiag_rtol,
    )
    mu = find_chemical_potential(
        eps, np.asarray(weights), n_electrons=float(n_electrons), T=float(T)
    )
    occ = fermidirac(eps - mu, float(T))
    P = np.einsum("...in,...n,...jn->...ij", U, occ, np.conj(U))
    return hermitize(P), mu


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
