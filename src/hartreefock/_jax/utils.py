"""Low-level physics utilities."""

from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.numpy.fft import fftn, ifftn, ifftshift
from jax.scipy.special import expit

from .linalg import adjoint, bmm, eigh


def device_args(kernel) -> dict:
    """``kernel._as_args()`` with arrays on the JAX device in canonical dtypes."""
    return {
        k: jnp.asarray(v) if isinstance(v, np.ndarray) else v
        for k, v in kernel._as_args().items()
    }


def _has_tracer(value: Any) -> bool:
    return isinstance(value, jax.core.Tracer)


def validate_electron_count(
    weights: Any,
    nbands: int,
    n_electrons: Any,
    *,
    context: str = "n_electrons",
) -> None:
    """Validate a requested particle count when concrete values are available."""
    if _has_tracer(weights) or _has_tracer(n_electrons):
        return

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


def validate_temperature(T: Any, *, context: str = "T") -> Any:
    """``T`` as a float, rejecting non-finite or negative values (a traced ``T`` passes as is)."""
    if _has_tracer(T):
        return T
    temperature = float(np.asarray(T))
    if not np.isfinite(temperature):
        raise ValueError(f"{context} must be finite, got {temperature!r}.")
    if temperature < 0.0:
        raise ValueError(f"{context} must be non-negative, got {temperature!r}.")
    return temperature


def hermitize(X):
    """Hermitize the last two axes, ``(X + X†) / 2``; NumPy stays NumPy, JAX stays JAX."""
    return 0.5 * (X + X.conj().swapaxes(-1, -2))


def fermidirac(x: jax.Array, T: float) -> jax.Array:
    """Finite-T Fermi–Dirac occupation: 1/(1 + exp(x/T))."""
    validate_temperature(T)
    return expit(-x / jnp.maximum(T, 1e-12))


def electron_density(P: jax.Array) -> jax.Array:
    """Real trace of the one-particle density matrix (per k-point)."""
    return jnp.real(jnp.trace(P, axis1=-2, axis2=-1))


def density_spectrum(bands: jax.Array, mu: float, T: float) -> jax.Array:
    """Sum of Fermi occupations for each k over bands."""
    return fermidirac(bands - mu, T).sum(axis=-1)


def selfenergy_fft(
    VR: jax.Array,
    P: jax.Array,
    *,
    sectors: tuple[tuple[int, ...], ...] | None = None,
    _apply_ifftshift: bool = True,
    hermitian_channel_packing: bool = False,
) -> jax.Array:
    """Exchange self-energy ``Σ(k) = -FFT⁻¹[FFT(P) · VR]``, aligned to the meshgrid.

    With ``sectors`` (index tuples the density does not couple) only the
    channels inside each sector are transformed.

    When ``_apply_ifftshift=False`` the final ifftshift is skipped.  Use this
    together with a phase-shifted ``VR`` to avoid the data permutation in hot
    loops (the shift is absorbed into VR at construction time).

    When ``hermitian_channel_packing=True``, the function assumes:
      1) ``VR`` is scalar in orbital space with shape ``(..., 1, 1)``, and
      2) ``P`` is Hermitian in its last two axes.

    In that case only the diagonal plus upper-triangular orbital channels are
    transformed, and the lower half is reconstructed by conjugation.  This is
    exact for the real-scalar Coulomb kernels used by ``HartreeFockKernel`` and
    roughly halves the FFT batch count.
    """
    VR = jnp.asarray(VR)
    P = jnp.asarray(P)

    if hermitian_channel_packing and VR.shape[-2:] != (1, 1):
        raise ValueError(
            "hermitian_channel_packing requires a scalar interaction kernel "
            "with shape (..., 1, 1)."
        )

    if sectors is None:
        result = _selfenergy_fft_full(VR, P, hermitian_channel_packing=hermitian_channel_packing)
    else:
        result = jnp.zeros_like(P)
        for idx in sectors:
            i = jnp.asarray(idx, dtype=jnp.int32)
            V_i = VR if VR.shape[-2:] == (1, 1) else VR[..., i[:, None], i[None, :]]
            sigma = _selfenergy_fft_full(
                V_i, P[..., i[:, None], i[None, :]], hermitian_channel_packing=hermitian_channel_packing
            )
            result = result.at[..., i[:, None], i[None, :]].set(sigma)

    if _apply_ifftshift:
        return ifftshift(result, axes=(0, 1))
    return result


def _selfenergy_fft_scalar_hermitian_channels(VR: jax.Array, P: jax.Array) -> jax.Array:
    n = int(P.shape[-1])
    tri_i_np, tri_j_np = np.triu_indices(n)
    offdiag_np = tri_i_np != tri_j_np
    flat_idx = jnp.asarray(tri_i_np * n + tri_j_np, dtype=jnp.int32)
    P_flat = P.reshape(P.shape[:-2] + (n * n,))
    packed = jnp.take(P_flat, flat_idx, axis=-1)
    packed_fft = fftn(packed, axes=(0, 1))
    VR_scalar = VR[..., 0, 0][..., None]
    sigma_packed = -ifftn(packed_fft * VR_scalar, axes=(0, 1))

    out_flat = jnp.zeros_like(P_flat)
    out_flat = out_flat.at[..., flat_idx].set(sigma_packed)

    lower_flat_idx = jnp.asarray(
        tri_j_np[offdiag_np] * n + tri_i_np[offdiag_np],
        dtype=jnp.int32,
    )
    out_flat = out_flat.at[..., lower_flat_idx].set(
        jnp.conj(sigma_packed[..., offdiag_np])
    )
    return out_flat.reshape(P.shape)


def _selfenergy_fft_full(
    VR: jax.Array,
    P: jax.Array,
    *,
    hermitian_channel_packing: bool = False,
) -> jax.Array:
    if hermitian_channel_packing:
        return _selfenergy_fft_scalar_hermitian_channels(VR, P)
    P_fft = fftn(P, axes=(0, 1))
    return -ifftn(P_fft * VR, axes=(0, 1))


def _chemical_potential_bounds(bands, capacity, target, temperature):
    """Bracket μ using a flat spectrum at each band edge, preserving dtype."""
    dtype = bands.real.dtype
    capacity = jnp.asarray(capacity, dtype=dtype)
    target = jnp.asarray(target, dtype=dtype)
    temperature = jnp.asarray(temperature, dtype=dtype)
    # Callers mask exact empty/full fillings to infinity after solving. Keep
    # the unused branch finite while tracing those endpoints through JIT.
    interior = (target > 0.0) & (target < capacity)
    safe_target = jnp.where(interior, target, 0.5 * capacity)
    shift = temperature * (jnp.log(safe_target) - jnp.log(capacity - safe_target))
    lo, hi = jnp.min(bands) + shift, jnp.max(bands) + shift
    # Round outward locally without nextafter, whose JAX derivative is absent.
    epsilon = jnp.asarray(jnp.finfo(dtype).eps, dtype=dtype)
    return (lo - epsilon * jnp.maximum(jnp.abs(lo), temperature),
            hi + epsilon * jnp.maximum(jnp.abs(hi), temperature))


def chemical_potential(
    bands: Any,
    weights: Any,
    n_electrons: Any,
    T: Any,
    *,
    mu0: Any = None,
    maxiter: int = 100,
) -> jax.Array:
    """The μ with ``Σ_k w_k Σ_j f(ε_kj − μ) = n_electrons`` at temperature ``T``.

    Safeguarded Newton iteration on a shrinking bracket, from ``mu0`` (the
    bracket midpoint by default, or when ``mu0`` is NaN).  ``maxiter`` is an upper bound: the loop
    stops as soon as the particle count matches to a few ULPs of the occupied
    or empty fraction, or ``mu`` stops moving at the working precision, which
    from a warm start takes a handful of steps.  Exactly empty and full
    fillings give ``-inf`` and ``+inf``.  Runs on NumPy and JAX input, and
    under ``jit``.
    """
    eps = jnp.asarray(bands)
    validate_electron_count(weights, eps.shape[-1], n_electrons)
    validate_temperature(T)
    real_dtype = eps.real.dtype
    mu0 = jnp.asarray(jnp.nan if mu0 is None else mu0, dtype=real_dtype)
    return _chemical_potential(eps.real, jnp.asarray(weights, dtype=real_dtype),
                               jnp.asarray(n_electrons, dtype=real_dtype),
                               jnp.asarray(T, dtype=real_dtype), mu0, maxiter)


@partial(jax.jit, static_argnums=5)
def _chemical_potential(eps, w_norm, n_target_norm, T, mu0, maxiter):
    real_dtype = eps.dtype
    Tj = jnp.maximum(jnp.asarray(T, dtype=real_dtype), jnp.asarray(1e-12, dtype=real_dtype))

    w_b = w_norm[..., None]
    capacity = jnp.sum(w_norm) * jnp.asarray(eps.shape[-1], dtype=real_dtype)
    lo, hi = _chemical_potential_bounds(eps, capacity, n_target_norm, Tj)
    mu = jnp.where(jnp.isnan(mu0), 0.5 * (lo + hi), jnp.clip(mu0, lo, hi))
    eps_m = jnp.asarray(jnp.finfo(real_dtype).eps, dtype=real_dtype)
    # Resolve the count to the precision of the *occupied* (or empty) fraction,
    # not of the total capacity: at low filling the callers' occupations carry
    # the target count, and a capacity-scaled tolerance would leave a relative
    # particle error of order capacity/n_target.
    occ_scale = jnp.minimum(n_target_norm, capacity - n_target_norm)
    count_tol = 4.0 * eps_m * occ_scale

    def cond(state):
        i, _mu, _lo, _hi, g_prev, step_prev = state
        return jnp.logical_and(
            i < maxiter,
            jnp.logical_and(jnp.abs(g_prev) > count_tol, step_prev > 0.0),
        )

    def body(state):
        i, mu, lo, hi, _g_prev, _step_prev = state
        x = (mu - eps) / Tj
        p = jax.nn.sigmoid(x)
        N = jnp.sum(w_b * p)
        Z = jnp.sum(w_b * p * (1.0 - p) / Tj)
        g = N - n_target_norm

        lo = jnp.where(g < 0, mu, lo)
        hi = jnp.where(g > 0, mu, hi)

        mu_bis = 0.5 * (lo + hi)
        Z_safe = jnp.where(Z > 0.0, Z, jnp.ones_like(Z))
        mu_new = jnp.where(Z > 0.0, mu - g / Z_safe, mu_bis)
        # A Newton update that rounds back onto ``mu`` (which has just become
        # a bracket end) is a converged iterate, not an out-of-bracket step;
        # bisecting from it would throw the solution away.
        stalled = mu_new == mu
        out_of = jnp.logical_and(
            jnp.logical_or(mu_new <= lo, mu_new >= hi),
            jnp.logical_not(stalled),
        )
        mu_new = jnp.where(out_of, mu_bis, mu_new)
        mu_new = jnp.clip(mu_new, lo, hi)
        mu_new = jnp.where(jnp.isfinite(mu_new), mu_new, mu_bis)
        # Keep an iterate whose count already matches in place.
        mu_new = jnp.where(jnp.abs(g) <= count_tol, mu, mu_new)
        return (i + 1, mu_new, lo, hi, g, jnp.abs(mu_new - mu))

    inf = jnp.asarray(jnp.inf, dtype=real_dtype)
    _, mu_fin, lo_fin, hi_fin, _, _ = lax.while_loop(
        cond, body, (jnp.int32(0), mu, lo, hi, inf, inf),
    )
    mu_bis = 0.5 * (lo_fin + hi_fin)

    def count(mu_value):
        return jnp.sum(w_b * jax.nn.sigmoid((mu_value - eps) / Tj))

    mu_fin = jnp.where(
        jnp.abs(count(mu_bis) - n_target_norm)
        < jnp.abs(count(mu_fin) - n_target_norm),
        mu_bis,
        mu_fin,
    )
    neg_inf = jnp.asarray(-jnp.inf, dtype=real_dtype)
    pos_inf = jnp.asarray(jnp.inf, dtype=real_dtype)
    return jnp.where(
        n_target_norm <= 0.0,
        neg_inf,
        jnp.where(n_target_norm >= capacity, pos_inf, mu_fin),
    )


def density_matrix_from_fock(
    F: jax.Array,
    weights: jax.Array,
    n_electrons: float,
    T: float,
    *,
    sectors: tuple[tuple[int, ...], ...] | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Build a physical density matrix P(k) from a (Hermitian) Fock matrix F(k)."""
    F = hermitize(jnp.asarray(F))
    validate_temperature(T)
    validate_electron_count(
        weights,
        F.shape[-1],
        n_electrons,
        context="n_electrons",
    )
    eps, U = eigh(F, sectors)
    mu = chemical_potential(eps, weights, n_electrons, T)
    occ = fermidirac(eps - mu, float(T))
    P = bmm(U * occ[..., None, :].astype(U.dtype), adjoint(U))
    return hermitize(P), mu


def _resample_axis_periodic_linear(x: jax.Array, n_new: int, *, axis: int) -> jax.Array:
    x = jnp.asarray(x)
    if x.ndim == 0:
        raise ValueError("resample_kgrid expects an array with at least 1 dimension.")
    n_old = int(x.shape[axis])
    n_new = int(n_new)
    if n_new <= 0:
        raise ValueError("n_new must be a positive integer.")
    if n_old == n_new:
        return x

    real_dtype = x.real.dtype
    j = jnp.arange(n_new, dtype=real_dtype)
    # Coordinate convention: arrays are stored on a centered periodic grid (k=0 at index n//2).
    u = (j - jnp.asarray(n_new // 2, dtype=real_dtype)) * (jnp.asarray(n_old, dtype=real_dtype) / jnp.asarray(n_new, dtype=real_dtype))
    u = u + jnp.asarray(n_old // 2, dtype=real_dtype)

    i0 = jnp.floor(u).astype(jnp.int32)

    i0 = jnp.mod(i0, n_old)
    i1 = jnp.mod(i0 + 1, n_old)

    x0 = jnp.take(x, i0, axis=axis)
    x1 = jnp.take(x, i1, axis=axis)

    # reshape frac to broadcast along all non-resampled axes
    shape = [1] * x.ndim
    shape[axis] = n_new
    frac = (u - jnp.floor(u)).reshape(shape).astype(real_dtype)
    if jnp.iscomplexobj(x0):
        frac = frac.astype(x0.dtype)

    return (1.0 - frac) * x0 + frac * x1


def resample_kgrid(values: jax.Array, nk: int, *, method: str = "linear") -> jax.Array:
    """Resample a centered, periodic (nk,nk,...) k-grid array to a new nk.

    Notes
    -----
    - Assumes the first two axes are the 2D uniform k-grid, stored in *centered*
      meshgrid order (k=0 at index nk//2), as in the package regression tests.
    - This is intended for coarse-to-fine continuation and seeding, not for
      high-accuracy quadrature of sharply peaked functions.
    """
    nk = int(nk)
    x = jnp.asarray(values)
    if x.ndim < 2:
        raise ValueError("resample_kgrid expects an array with at least 2 dimensions (nk,nk,...).")
    if x.shape[0] == nk and x.shape[1] == nk:
        return x

    method = str(method).lower()
    if method != "linear":
        raise ValueError(f"Unsupported resample method {method!r}. Only 'linear' is supported.")

    y = _resample_axis_periodic_linear(x, nk, axis=0)
    y = _resample_axis_periodic_linear(y, nk, axis=1)
    return y
