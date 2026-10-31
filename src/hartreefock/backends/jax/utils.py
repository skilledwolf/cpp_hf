"""Low-level physics utilities for jax_hf (self-contained, no external deps)."""

from __future__ import annotations

from typing import Any

import numpy as np

import jax
import jax.numpy as jnp
from jax.numpy.fft import fftn, ifftn, ifftshift
from jax.scipy.special import expit

from .linalg import eigh, normalize_block_specs


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


def validate_temperature(T: Any, *, context: str = "T") -> None:
    """Reject non-finite or negative temperatures when values are concrete."""
    if _has_tracer(T):
        return
    temperature = float(np.asarray(T))
    if not np.isfinite(temperature):
        raise ValueError(f"{context} must be finite, got {temperature!r}.")
    if temperature < 0.0:
        raise ValueError(f"{context} must be non-negative, got {temperature!r}.")


def hermitize(X: jax.Array) -> jax.Array:
    """Hermitize the last two axes: 0.5*(X + X†)."""
    return 0.5 * (X + jnp.conj(jnp.swapaxes(X, -1, -2)))


def fermidirac(x: jax.Array, T: float) -> jax.Array:
    """Finite-T Fermi–Dirac occupation: 1/(1 + exp(x/T))."""
    validate_temperature(T)
    return expit(-x / (T + 1e-12))


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
    block_specs: Any | None = None,
    check_offdiag: bool | None = None,
    offdiag_atol: float = 1e-12,
    offdiag_rtol: float = 0.0,
    _apply_ifftshift: bool = True,
    hermitian_channel_packing: bool = False,
) -> jax.Array:
    """Exchange self-energy Σ(k) with optional block-diagonal acceleration.

    Computes Σ(k) = -FFT⁻¹[FFT(P) · VR], aligned to meshgrid (ifftshift).

    When `block_specs` is provided, contimod-style specs are supported and the
    function will:
      1) check the off-block magnitude of `P` against `(atol + rtol*max|P|)`,
      2) run a reduced set of FFTs for the first compatible block partition,
      3) fall back to the full computation when no spec is compatible.

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

    if block_specs is None:
        result = _selfenergy_fft_full(
            VR,
            P,
            hermitian_channel_packing=hermitian_channel_packing,
        )
    else:
        specs = normalize_block_specs(block_specs)
        if not specs:
            result = _selfenergy_fft_full(
                VR,
                P,
                hermitian_channel_packing=hermitian_channel_packing,
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


def _mask_from_block_sizes(block_sizes: tuple[int, ...], n: int) -> jax.Array:
    mask = np.ones((int(n), int(n)), dtype=bool)
    start = 0
    for size in block_sizes:
        stop = start + int(size)
        mask[start:stop, start:stop] = False
        start = stop
    return jnp.asarray(mask)


def _mask_from_block_indices(block_indices: tuple[tuple[int, ...], ...], n: int) -> jax.Array:
    mask = np.ones((int(n), int(n)), dtype=bool)
    for idx in block_indices:
        idx_arr = np.asarray(idx, dtype=int)
        mask[np.ix_(idx_arr, idx_arr)] = False
    return jnp.asarray(mask)


def _slice_interaction(VR: jax.Array, idx: slice | jax.Array) -> jax.Array:
    if VR.shape[-2] == 1 and VR.shape[-1] == 1:
        return VR
    return VR[..., idx, idx]


def _take_interaction(VR: jax.Array, idx: jax.Array) -> jax.Array:
    if VR.shape[-2] == 1 and VR.shape[-1] == 1:
        return VR
    return jnp.take(jnp.take(VR, idx, axis=-2), idx, axis=-1)


def _selfenergy_fft_block_sizes(
    VR: jax.Array,
    P: jax.Array,
    block_sizes: tuple[int, ...],
    *,
    hermitian_channel_packing: bool = False,
) -> jax.Array:
    n = int(P.shape[-1])
    if sum(int(s) for s in block_sizes) != n:
        raise ValueError(f"block_sizes must sum to {n}.")

    out = jnp.zeros_like(P)
    start = 0
    for size in block_sizes:
        stop = start + int(size)
        s = slice(start, stop)
        sigma_block = _selfenergy_fft_full(
            _slice_interaction(VR, s),
            P[..., s, s],
            hermitian_channel_packing=hermitian_channel_packing,
        )
        out = out.at[..., s, s].set(sigma_block)
        start = stop
    return out


def _selfenergy_fft_block_indices(
    VR: jax.Array,
    P: jax.Array,
    block_indices: tuple[tuple[int, ...], ...],
    *,
    hermitian_channel_packing: bool = False,
) -> jax.Array:
    out = jnp.zeros_like(P)
    for idx in block_indices:
        idx_j = jnp.asarray(idx, dtype=jnp.int32)
        sub = jnp.take(jnp.take(P, idx_j, axis=-2), idx_j, axis=-1)
        sigma_block = _selfenergy_fft_full(
            _take_interaction(VR, idx_j),
            sub,
            hermitian_channel_packing=hermitian_channel_packing,
        )
        out = out.at[..., idx_j[:, None], idx_j[None, :]].set(sigma_block)
    return out


def _selfenergy_fft_block_specs(
    VR: jax.Array,
    P: jax.Array,
    block_specs: tuple[tuple[str, tuple[Any, ...]], ...],
    *,
    check_offdiag: bool,
    offdiag_atol: float,
    offdiag_rtol: float,
    hermitian_channel_packing: bool = False,
) -> jax.Array:
    n = int(P.shape[-1])
    if check_offdiag:
        abs_P = jnp.abs(P)
        scale = jnp.max(abs_P)
        tol = jnp.asarray(offdiag_atol, dtype=abs_P.dtype) + jnp.asarray(offdiag_rtol, dtype=abs_P.dtype) * scale

    def do_full(_):
        return _selfenergy_fft_full(
            VR,
            P,
            hermitian_channel_packing=hermitian_channel_packing,
        )

    fn = do_full

    for kind, data in reversed(block_specs):
        kind = str(kind).strip().lower()
        if kind == "sizes":
            sizes = tuple(int(x) for x in data)
            mask = _mask_from_block_sizes(sizes, n)
            ok = (jnp.max(abs_P * mask) <= tol) if check_offdiag else jnp.array(True)

            def do_block(_, *, _sizes=sizes):
                return _selfenergy_fft_block_sizes(
                    VR,
                    P,
                    _sizes,
                    hermitian_channel_packing=hermitian_channel_packing,
                )

        elif kind == "indices":
            blocks = tuple(tuple(int(i) for i in b) for b in data)
            mask = _mask_from_block_indices(blocks, n)
            ok = (jnp.max(abs_P * mask) <= tol) if check_offdiag else jnp.array(True)

            def do_block(_, *, _blocks=blocks):
                return _selfenergy_fft_block_indices(
                    VR,
                    P,
                    _blocks,
                    hermitian_channel_packing=hermitian_channel_packing,
                )

        else:
            raise ValueError("block_specs kind must be 'sizes' or 'indices'.")

        prev_fn = fn
        fn = (lambda _prev, _ok, _do: (lambda _: jax.lax.cond(_ok, _do, _prev, operand=None)))(prev_fn, ok, do_block)

    return fn(None)


def find_chemical_potential(
    bands: jax.Array,
    weights: jax.Array,
    n_electrons: float,
    T: float,
    *,
    maxiter: int | None = None,
    method: str = "bisection",
) -> jax.Array:
    """Find μ such that ∑_k w_k Σ_j f(ε_kj − μ) = n_electrons.

    Parameters
    ----------
    method : ``"bisection"`` (default) or ``"newton"``.
        ``"bisection"`` is a pure bracket bisection — unconditionally stable and
        deterministic, the safe choice inside DIIS-accelerated SCF loops.
        ``"newton"`` is a Newton-Raphson solver with bracket safeguards, adapted
        from the variational solver.  Converges in ~10 iterations instead of
        ~30–54, but produces slightly different mu values from bisection
        (~1e-8 difference) which can destabilize DIIS.

    When *maxiter* is ``None``, the iteration count is chosen automatically:
    bisection uses 30 (f32) / 54 (f64); Newton uses 15 (f32) / 25 (f64).
    """
    validate_electron_count(
        weights,
        bands.shape[-1],
        n_electrons,
        context="n_electrons",
    )
    validate_temperature(T)
    method = str(method).lower()
    if method == "bisection":
        mu = _find_mu_bisection(bands, weights, n_electrons, T, maxiter=maxiter)
    elif method == "newton":
        mu = _find_mu_newton(bands, weights, n_electrons, T, maxiter=maxiter)
    else:
        raise ValueError(f"method must be 'bisection' or 'newton', got {method!r}")

    # Exact empty/full fillings require an infinite chemical potential at
    # finite temperature.  Returning a merely "wide" finite bracket endpoint
    # leaves exponentially small tails and can make SCF report convergence at
    # the wrong particle count.
    real_dtype = bands.real.dtype
    target = jnp.asarray(n_electrons, dtype=real_dtype)
    capacity = (
        jnp.sum(jnp.asarray(weights, dtype=real_dtype))
        * jnp.asarray(bands.shape[-1], dtype=real_dtype)
    )
    neg_inf = jnp.asarray(-jnp.inf, dtype=real_dtype)
    pos_inf = jnp.asarray(jnp.inf, dtype=real_dtype)
    return jnp.where(target <= 0.0, neg_inf, jnp.where(target >= capacity, pos_inf, mu))


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


def _find_mu_bisection(
    bands: jax.Array,
    weights: jax.Array,
    n_electrons: float,
    T: float,
    *,
    maxiter: int | None = None,
) -> jax.Array:
    """Pure bracket bisection — unconditionally stable."""
    if maxiter is None:
        maxiter = 30 if bands.dtype in (jnp.float32, jnp.complex64) else 54
    real_dtype = bands.real.dtype
    capacity = jnp.sum(jnp.asarray(weights, dtype=real_dtype)) * bands.shape[-1]
    temperature = jnp.asarray(T, dtype=real_dtype) + jnp.asarray(1e-12, dtype=real_dtype)
    lo, hi = _chemical_potential_bounds(bands, capacity, n_electrons, temperature)

    def body(state, _):
        lo_curr, hi_curr = state
        mid = 0.5 * (lo_curr + hi_curr)
        occ = fermidirac(bands - mid, T)
        count = jnp.sum(jnp.asarray(weights)[..., None] * occ)
        too_high = count > n_electrons
        lo_next = jnp.where(too_high, lo_curr, mid)
        hi_next = jnp.where(too_high, mid, hi_curr)
        return (lo_next, hi_next), None

    (lo_fin, hi_fin), _ = jax.lax.scan(body, (lo, hi), xs=None, length=int(maxiter))
    return 0.5 * (lo_fin + hi_fin)


def _find_mu_newton(
    bands: jax.Array,
    weights: jax.Array,
    n_electrons: float,
    T: float,
    *,
    maxiter: int | None = None,
) -> jax.Array:
    """Newton-Raphson with bracket safeguards — faster convergence (~10 iters)."""
    if maxiter is None:
        maxiter = 15 if bands.dtype in (jnp.float32, jnp.complex64) else 25
    real_dtype = bands.real.dtype
    Tj = jnp.asarray(T, dtype=real_dtype)
    Tj = Tj + jnp.asarray(1e-12, dtype=real_dtype)
    n_target = jnp.asarray(n_electrons, dtype=real_dtype)

    capacity = jnp.sum(jnp.asarray(weights, dtype=real_dtype)) * bands.shape[-1]
    lo, hi = _chemical_potential_bounds(bands, capacity, n_target, Tj)
    mu = 0.5 * (lo + hi)

    w_b = jnp.asarray(weights)[..., None]

    def count_and_slope(mu_val):
        x = (mu_val - bands) / Tj
        p = jax.nn.sigmoid(x)
        N = jnp.sum(w_b * p)
        dp = (p * (1.0 - p)) / Tj
        Z = jnp.sum(w_b * dp)
        return N, Z

    def body(state, _):
        mu, lo, hi = state
        N, Z = count_and_slope(mu)
        g = N - n_target

        lo = jnp.where(g < 0, mu, lo)
        hi = jnp.where(g > 0, mu, hi)

        mu_bis = 0.5 * (lo + hi)
        has_slope = Z > 0.0
        mu_new = mu - g / jnp.where(has_slope, Z, jnp.ones_like(Z))
        mu_new = jnp.where(has_slope, mu_new, mu_bis)
        # An update that rounds onto μ must not restart at a distant midpoint.
        stalled = mu_new == mu
        out_of = ((mu_new <= lo) | (mu_new >= hi)) & ~stalled
        mu_new = jnp.where(out_of | ~jnp.isfinite(mu_new), mu_bis, mu_new)
        mu_new = jnp.clip(mu_new, lo, hi)
        mu_new = jnp.where(g == 0.0, mu, mu_new)
        return (mu_new, lo, hi), None

    (mu_fin, lo_fin, hi_fin), _ = jax.lax.scan(body, (mu, lo, hi), xs=None, length=int(maxiter))

    # Return whichever of the safeguarded Newton iterate and the remaining
    # bracket midpoint has the smaller particle-count residual.  At very low
    # temperature an integer filling has a broad, numerically flat gap.  The
    # old absolute 1e-12 cutoff discarded an already-correct Newton iterate in
    # that gap and returned the midpoint of a still-asymmetric bracket, which
    # could occupy an entire extra band.
    N_fin, _ = count_and_slope(mu_fin)
    mu_bis = 0.5 * (lo_fin + hi_fin)
    N_bis, _ = count_and_slope(mu_bis)
    return jnp.where(
        jnp.abs(N_bis - n_target) < jnp.abs(N_fin - n_target),
        mu_bis,
        mu_fin,
    )


def density_matrix_from_fock(
    F: jax.Array,
    weights: jax.Array,
    n_electrons: float,
    T: float,
    *,
    eigh_block_specs: object | None = None,
    eigh_check_offdiag: bool | None = None,
    eigh_offdiag_atol: float = 1e-12,
    eigh_offdiag_rtol: float = 0.0,
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
    eps, U = eigh(
        F,
        block_specs=eigh_block_specs,
        check_offdiag=eigh_check_offdiag,
        offdiag_atol=eigh_offdiag_atol,
        offdiag_rtol=eigh_offdiag_rtol,
    )
    mu = find_chemical_potential(eps, jnp.asarray(weights), n_electrons=float(n_electrons), T=float(T))
    occ = fermidirac(eps - mu, float(T))
    P = jnp.einsum("...in,...n,...jn->...ij", U, occ, jnp.conj(U))
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
