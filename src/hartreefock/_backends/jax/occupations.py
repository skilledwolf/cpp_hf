"""Occupations on the capped simplex: logits, active bounds, μ solve and the
particle-conserving projections shared by the CG and Newton solvers.

Occupations that have saturated to 0 or 1 at the working precision are treated
as active bound constraints: their logit is clipped, so their entropy gradient
is not a gradient, and letting it into the particle-conservation projection
turns the occupation step into an ascent direction (see
:func:`_occupation_gradient`).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from .utils import (
    _chemical_potential_bounds,
)


def _logit_clip_eps(dtype) -> jax.Array:
    # Use the tightest symmetric bound representable by the working dtype.
    # A fixed 1e-6 bound badly distorts finite-T gradients in float64 when the
    # physical thermal tail is smaller than 1e-6.
    return jnp.maximum(
        jnp.asarray(1e-14, dtype=dtype),
        jnp.asarray(jnp.finfo(dtype).eps, dtype=dtype),
    )


def _logit(p: jax.Array) -> jax.Array:
    eps = _logit_clip_eps(p.dtype)
    p_safe = jnp.clip(p, eps, 1.0 - eps)
    return jnp.log(p_safe) - jnp.log1p(-p_safe)


def _active_occupations(p: jax.Array) -> jax.Array:
    """Bands whose occupation is resolvable away from the 0/1 bounds.

    Occupations inside the logit clipping window are saturated: their entropy
    gradient is numerically undefined and they cannot move at the working
    precision, so the optimizers treat them as active bound constraints.
    """
    eps = _logit_clip_eps(p.dtype)
    return jnp.logical_and(p > eps, p < 1.0 - eps)


# ---------------------------------------------------------------------------
# Chemical potential solver (Newton-bracket)
# ---------------------------------------------------------------------------

def _solve_mu(
    eps: jax.Array,
    w_norm: jax.Array,
    n_target_norm: jax.Array,
    mu0: jax.Array,
    T: jax.Array,
    maxiter: int = 25,
) -> jax.Array:
    """Find mu so sum_k w_k sum_i sigmoid((mu - eps)/T) = n_target_norm.

    Safeguarded Newton iteration on a shrinking bracket.  ``maxiter`` is an
    upper bound: the loop stops as soon as the particle count matches to a few
    ULPs of the occupied or empty fraction, or stops moving ``mu`` at the working
    precision, which from a warm start takes a handful of steps rather than
    the full budget.
    """
    real_dtype = eps.dtype
    Tj = jnp.maximum(jnp.asarray(T, dtype=real_dtype), jnp.asarray(1e-12, dtype=real_dtype))

    w_b = w_norm[..., None]
    capacity = jnp.sum(w_norm) * jnp.asarray(eps.shape[-1], dtype=real_dtype)
    lo, hi = _chemical_potential_bounds(eps, capacity, n_target_norm, Tj)
    mu = jnp.clip(jnp.asarray(mu0, dtype=real_dtype), lo, hi)
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


def _project_occ(dp: jax.Array, w2d: jax.Array) -> jax.Array:
    """Project an occupation variation onto the particle-conserving subspace.

    Removes the component that changes the total electron count, i.e. enforces
    ``sum_k w_k sum_b dp[k, b] = 0`` by subtracting a uniform shift.
    """
    nb = dp.shape[-1]
    num = jnp.sum(w2d[..., None] * dp)
    den = jnp.sum(w2d) * nb
    return dp - num / den


def _project_occ_masked(
    dp: jax.Array, w2d: jax.Array, active: jax.Array,
) -> jax.Array:
    """Particle-conserving projection restricted to the ``active`` bands.

    Uniform shift over the active bands only; saturated bands are held fixed
    (zero variation).  Equals :func:`_project_occ` when every band is active,
    which is why that unmasked form is kept: it is the reference case this
    projection is defined against, and the tests check the two agree.
    """
    real_dtype = dp.dtype
    tiny = jnp.asarray(1e-30, dtype=real_dtype)
    wa = w2d[..., None] * active.astype(real_dtype)
    shift = jnp.sum(wa * dp) / jnp.maximum(jnp.sum(wa), tiny)
    return jnp.where(active, dp - shift, jnp.zeros_like(dp))


def _project_occ_curvature(
    v: jax.Array, w2d: jax.Array, curv: jax.Array,
) -> jax.Array:
    """Curvature-preconditioned particle-conserving projection.

    Returns ``curv * (v - lam)`` with ``lam`` chosen so that
    ``sum_k w_k sum_i curv_i (v_i - lam) = 0``: the projection of ``v`` onto
    the fixed-particle tangent space under the metric ``1/curv``.  Bands with
    ``curv = 0`` (saturated occupations) receive no variation and do not
    influence ``lam``, so an unresolvable entropy gradient on a saturated band
    cannot leak into the partially occupied ones.
    """
    real_dtype = v.dtype
    tiny = jnp.asarray(1e-30, dtype=real_dtype)
    wc = w2d[..., None] * curv
    lam = jnp.sum(wc * v) / jnp.maximum(jnp.sum(wc), tiny)
    return curv * (v - lam)


def _occupation_gradient(
    eps: jax.Array, p: jax.Array, T: jax.Array, w2d: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Projected occupation gradient of ``Omega = E - T S`` and its preconditioned form.

    The raw gradient with respect to ``p_i`` is ``eps_i + T logit(p_i)``.  At a
    Fermi-Dirac state it equals the chemical potential for every band, but a
    band whose occupation has saturated to 0 or 1 at the working precision has
    a clipped logit, so its raw gradient is off by up to ``|eps_i - mu|``.  On
    low-temperature problems most bands are saturated; if that error enters the
    uniform particle-conservation shift it is spread onto the partially filled
    bands and turns the occupation step into an ascent direction.  Saturated
    bands are therefore masked out of the Euclidean projection, get zero
    curvature, and the particle constraint is enforced over the active bands.

    Returns ``(g, h, curv, active)``: ``g`` is the masked particle-conserving
    gradient, ``h = curv * (g_raw - lam)`` its Fermi-Dirac-curvature
    preconditioned form with ``sum_k w_k sum_i h_i = 0`` (a descent direction:
    ``<g_raw, h> = sum w curv (g_raw - lam)^2 >= 0``), ``curv = p(1-p)/T`` on
    active bands and 0 elsewhere, and ``active`` the band mask.
    """
    real_dtype = eps.dtype
    active = _active_occupations(p)
    g_raw = (eps + T * _logit(p)).astype(real_dtype)
    curv = jnp.where(active, p * (1.0 - p) / T, jnp.zeros_like(p)).astype(real_dtype)
    g = _project_occ_masked(g_raw, w2d, active)
    h = _project_occ_curvature(g_raw, w2d, curv)
    return g, h, curv, active


def _project_capped_occupations(
    p: jax.Array,
    w2d: jax.Array,
    n_target: jax.Array,
    *,
    maxiter: int = 50,
) -> jax.Array:
    """Euclidean projection onto ``0 <= p <= 1`` at fixed weighted count.

    The weighted metric used by the optimizers has the same k-point weights as
    the count constraint, so the Lagrange multiplier is a uniform shift:
    ``p_projected = clip(p + shift, 0, 1)``.  Bisection finds that shift while
    remaining fully JIT compatible.
    """
    real_dtype = p.dtype
    lo = -jnp.max(p) - jnp.asarray(1.0, dtype=real_dtype)
    hi = jnp.asarray(1.0, dtype=real_dtype) - jnp.min(p) + jnp.asarray(1.0, dtype=real_dtype)

    def body(state, _):
        lo_i, hi_i = state
        mid = 0.5 * (lo_i + hi_i)
        count = jnp.sum(w2d[..., None] * jnp.clip(p + mid, 0.0, 1.0))
        lo_i = jnp.where(count < n_target, mid, lo_i)
        hi_i = jnp.where(count > n_target, mid, hi_i)
        return (lo_i, hi_i), None

    (lo, hi), _ = lax.scan(body, (lo, hi), xs=None, length=maxiter)
    projected = jnp.clip(p + 0.5 * (lo + hi), 0.0, 1.0)
    capacity = jnp.sum(w2d) * jnp.asarray(p.shape[-1], dtype=real_dtype)
    return jnp.where(
        n_target <= 0.0,
        jnp.zeros_like(projected),
        jnp.where(n_target >= capacity, jnp.ones_like(projected), projected),
    )
