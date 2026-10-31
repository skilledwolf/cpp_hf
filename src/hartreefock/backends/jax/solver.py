"""Direct minimization solver: preconditioned Riemannian CG on Stiefel x Simplex.

Minimizes the Hartree-Fock free energy Omega(Q, p) = E[P] - T*S(p) over
unitary orbitals Q (Stiefel manifold) and occupations p (capped simplex).

Algorithm:
  1. Build F[P] (an exchange FFT — the dominant cost)
  2. Analytical Riemannian gradient from Fock matrix (no autodiff)
  3. Energy-gap preconditioning (diagonal Hessian)
  4. Polak-Ribiere CG direction with periodic restart
  5. Backtracking line search on the true nonlinear free energy
  6. Cayley retraction for orbitals, simplex projection for occupations

Occupations that have saturated to 0 or 1 at the working precision are treated
as active bound constraints: their logit is clipped, so their entropy gradient
is not a gradient, and letting it into the particle-conservation projection
turns the occupation step into an ascent direction (see
:func:`_occupation_gradient`).
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
from jax import lax

from ...config import SolverConfig, SolveResult, drop_unrequested
from .fock import build_fock, hf_energy, free_energy
from .utils import (
    _chemical_potential_bounds, device_args, validate_electron_count, validate_temperature,
)


# ---------------------------------------------------------------------------
# Small linear algebra helpers
# ---------------------------------------------------------------------------

def _herm(X: jax.Array) -> jax.Array:
    return 0.5 * (X + jnp.conj(jnp.swapaxes(X, -1, -2)))


def _skew_hermitian(A: jax.Array) -> jax.Array:
    return 0.5 * (A - jnp.conj(jnp.swapaxes(A, -1, -2)))


def _density_from_Qp(Q: jax.Array, p: jax.Array) -> jax.Array:
    """P = Q diag(p) Q†."""
    return jnp.einsum("...in,...n,...jn->...ij", Q, p, jnp.conj(Q))


def _project_density_eigensystem(Q, p, project_fn):
    """Project a density and keep its orbital/occupation representation consistent.

    Symmetry projection can change a trial's spectrum, especially when a
    degenerate density has an arbitrary orbital basis. Only re-diagonalize
    when that change exceeds roundoff; entropy must use the projected spectrum.
    """
    raw = _density_from_Qp(Q, p)
    if project_fn is None:
        return _herm(raw), Q, p, jnp.bool_(False)
    P = _herm(jnp.asarray(project_fn(raw), dtype=Q.dtype))
    roundoff = (64.0 * jnp.finfo(p.dtype).eps) * jnp.max(jnp.abs(raw))
    changed = jnp.max(jnp.abs(P - raw)) > roundoff

    def repair(_):
        p_new, Q_new = jnp.linalg.eigh(P)
        return Q_new.astype(Q.dtype), p_new.astype(p.dtype)

    Q, p = lax.cond(changed, repair, lambda _: (Q, p), operand=None)
    return P, Q, p, changed


def _fock_in_orbital_basis(Q: jax.Array, F: jax.Array) -> jax.Array:
    """Ft = Q† F Q."""
    return jnp.einsum("...in,...ij,...jm->...nm", jnp.conj(Q), F, Q)


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


# ---------------------------------------------------------------------------
# QR retraction
# ---------------------------------------------------------------------------

def _qr_retract(G: jax.Array, tau: jax.Array) -> jax.Array:
    """U = QR(I - tau*G) with phase normalization."""
    tiny = jnp.asarray(1e-30, dtype=jnp.real(G).dtype)
    tau_bc = jnp.asarray(tau, dtype=jnp.real(G).dtype)[..., None, None]
    n = G.shape[-1]
    U_trial = jnp.eye(n, dtype=G.dtype) - tau_bc * G
    U, R = jnp.linalg.qr(U_trial)
    phases = jnp.diagonal(R, axis1=-2, axis2=-1)
    phase_norm = jnp.where(
        jnp.abs(phases) > tiny, phases / jnp.abs(phases), jnp.ones_like(phases),
    )
    return U * phase_norm[..., None, :]


def _cayley_retract(d: jax.Array, tau: jax.Array) -> jax.Array:
    """Cayley retraction: U = (I - tau*d/2) (I + tau*d/2)^-1.

    For skew-Hermitian d, (I+A) and (I-A) with A = tau*d/2 commute, so
    U = solve(I + A, I - A) = (I + A)^-1 (I - A) is unitary.

    Agrees with exp(-tau*d) to first order in tau.  Cheaper than QR
    retraction (one LU solve vs one QR, ~2x fewer flops on nb*nb blocks).

    Valid as long as ||tau*d|| < 2 (eigenvalues of (I + tau*d/2) stay away
    from zero).  With max_step <= 1 and our typical tau <= 1, this is safe.
    """
    tau_bc = jnp.asarray(tau, dtype=jnp.real(d).dtype)[..., None, None]
    n = d.shape[-1]
    eye = jnp.eye(n, dtype=d.dtype)
    A = 0.5 * tau_bc * d
    return jnp.linalg.solve(eye + A, eye - A)


# ---------------------------------------------------------------------------
# Spectral Cayley: amortise eigh of (i*d) across line search backtracking.
#
# For skew-Hermitian d, i*d is Hermitian.  Diagonalise once:
#     i*d = V diag(lam) V†      (lam real, V unitary)
# then for any tau:
#     U(tau) = (I - tau*d/2)(I + tau*d/2)^-1
#            = V · diag((1 + i*tau*lam/2)/(1 - i*tau*lam/2)) · V†
# The factor c(tau, lam) = (1 + iτλ/2)/(1 − iτλ/2) has |c|=1 ⇒ U is exactly
# unitary independently of how many times we evaluate it.
#
# In the line search we only need diag(U(tau)† Ft U(tau)) (the trial orbital
# energies).  Pre-rotating Ft once into the eigenbasis of i*d gives
#     Ft_eig = V† Ft V
# and per-tau the diagonal is computed with one Hadamard scaling and one
# matmul, with no LU solve and no full Ft_trial:
#     M(tau) = c̄(tau) Ft_eig c(tau)              (Hadamard, O(nb²))
#     diag(U†FtU) = (V @ M(tau) ⊙ conj(V)).sum(-1)  (one matmul, O(nb³))
# ---------------------------------------------------------------------------

def _cayley_spectral_setup(d: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Eigendecompose i*d for a skew-Hermitian d.  Returns (V, lam)."""
    iA = (1j * d).astype(d.dtype)
    iA = 0.5 * (iA + jnp.conj(jnp.swapaxes(iA, -1, -2)))  # exact Hermitisation
    lam, V = jnp.linalg.eigh(iA)
    return V, lam.astype(jnp.real(d).dtype)


def _cayley_factor(lam: jax.Array, tau: jax.Array, complex_dtype) -> jax.Array:
    """c(τ, λ) = (1 + iτλ/2) / (1 - iτλ/2) — |c|=1 by construction."""
    half = jnp.asarray(0.5, dtype=lam.dtype)
    arg = jnp.asarray(tau, dtype=lam.dtype) * lam * half
    iexp = (1j * arg).astype(complex_dtype)
    one = jnp.asarray(1.0, dtype=complex_dtype)
    return (one + iexp) / (one - iexp)


def _cayley_unitary_from_spectrum(V: jax.Array, lam: jax.Array,
                                  tau: jax.Array) -> jax.Array:
    """U(τ) = V · diag(c(τ, λ)) · V†."""
    c = _cayley_factor(lam, tau, V.dtype)
    return (V * c[..., None, :]) @ jnp.conj(jnp.swapaxes(V, -1, -2))


def _diag_UFU_from_spectrum(V: jax.Array, Ft_eig: jax.Array,
                            lam: jax.Array, tau: jax.Array) -> jax.Array:
    """Compute diag(U(τ)† Ft U(τ)) using precomputed Ft_eig = V† Ft V.

    Cost: one Hadamard (nb²) plus one matmul (nb³) — versus the LU+two-matmul
    path in :func:`_cayley_retract` followed by a full U†FtU triple product.
    """
    c = _cayley_factor(lam, tau, V.dtype)
    cbar = jnp.conj(c)
    M = (cbar[..., :, None]) * Ft_eig * (c[..., None, :])
    A = V @ M
    diag = jnp.real(jnp.sum(A * jnp.conj(V), axis=-1))
    return diag



# ---------------------------------------------------------------------------
# Weighted inner products
# ---------------------------------------------------------------------------

def _ip_matrix(X: jax.Array, Y: jax.Array, w_norm: jax.Array) -> jax.Array:
    """Weighted Frobenius inner product: sum_k w_k Re(Tr(X†Y))."""
    per_k = jnp.sum(jnp.real(jnp.conj(X) * Y), axis=(-2, -1))
    return jnp.sum(w_norm * per_k)


def _ip_vec(x: jax.Array, y: jax.Array, w_norm: jax.Array) -> jax.Array:
    """Weighted vector inner product: sum_k w_k sum_i x_i y_i."""
    return jnp.sum(w_norm[..., None] * x * y)


def _norm_matrix(X: jax.Array, w_norm: jax.Array) -> jax.Array:
    return jnp.sqrt(jnp.maximum(0.0, _ip_matrix(X, X, w_norm)))


def _norm_vec(x: jax.Array, w_norm: jax.Array) -> jax.Array:
    return jnp.sqrt(jnp.maximum(0.0, _ip_vec(x, x, w_norm)))


def _joint_predicted_reduction(
    G: jax.Array,
    g_p: jax.Array,
    vX: jax.Array,
    vp: jax.Array,
    HvX: jax.Array,
    Hvp: jax.Array,
    w2d: jax.Array,
) -> jax.Array:
    """Quadratic-model reduction for the actual joint Newton trial step."""
    linear = _ip_matrix(G, vX, w2d) + _ip_vec(g_p, vp, w2d)
    quadratic = _ip_matrix(vX, HvX, w2d) + _ip_vec(vp, Hvp, w2d)
    return -(linear + 0.5 * quadratic)


def _joint_inner_product(aX, ap, bX, bp, w_norm):
    """One normalized quadrature metric for both Newton tangent sectors."""
    return _ip_matrix(aX, bX, w_norm) + _ip_vec(ap, bp, w_norm)


def _joint_hvp(Q, p, Ft, X, dp, T, w2d, response_fn, project_fn=None):
    """Covariant Hessian action on the fixed-particle unitary/occupation space.

    ``response_fn`` is the derivative of the effective Fock at the current
    density, including any deflation force. Symmetry projections must be
    linear, orthogonal, trace preserving and map onto a matrix algebra.
    """
    project = project_fn if project_fn is not None else (lambda A: A)
    eye = jnp.eye(p.shape[-1], dtype=Q.dtype)
    offdiag = 1.0 - jnp.eye(p.shape[-1], dtype=p.dtype)
    diff_p = p[..., None, :] - p[..., :, None]
    M = X * diff_p + dp[..., :, None] * eye
    dP = _herm(project(Q @ M @ jnp.conj(jnp.swapaxes(Q, -1, -2))))
    St = _fock_in_orbital_basis(Q, _herm(project(response_fn(dP))))
    A = Ft @ X - X @ Ft + St
    G = diff_p * Ft
    C = (dp[..., None, :] - dp[..., :, None]) * Ft + diff_p * A
    # Parallel transport from Q exp(tX) back to Q contributes 1/2 [X,G].
    # Without this connection the operator is not self-adjoint off stationarity.
    C = C + 0.5 * (X @ G - G @ X)
    HX = _skew_hermitian(C) * offdiag
    clip = _logit_clip_eps(p.dtype)
    pc = jnp.clip(p, clip, 1.0 - clip)
    Hp = jnp.real(jnp.diagonal(A, axis1=-2, axis2=-1)) + T / (pc * (1.0 - pc)) * dp
    return HX, _project_occ_masked(Hp, w2d, _active_occupations(p))


def _deflation_bias(P, targets, w2d, sigma, length):
    """Gaussian bias, its density gradient, and its diagonal response part."""
    diff = P[None] - targets
    d2 = jnp.sum(w2d[None, ..., None, None] * jnp.abs(diff) ** 2, axis=(1, 2, 3, 4))
    invL2 = 1.0 / jnp.asarray(length, dtype=P.real.dtype) ** 2
    phi = sigma * jnp.exp(-0.5 * d2 * invL2)
    coeff = -phi * invL2
    force = jnp.sum(coeff[:, None, None, None, None] * diff, axis=0)
    return jnp.sum(phi), force, jnp.sum(coeff)


def _deflation_response(P, dP, targets, w2d, sigma, length):
    """Full Gaussian force derivative, including its nonlocal outer product."""
    diff = P[None] - targets
    wb = w2d[None, ..., None, None]
    d2 = jnp.sum(wb * jnp.abs(diff) ** 2, axis=(1, 2, 3, 4))
    directional = jnp.sum(wb * jnp.real(jnp.conj(diff) * dP[None]), axis=(1, 2, 3, 4))
    invL2 = 1.0 / jnp.asarray(length, dtype=P.real.dtype) ** 2
    phi = sigma * jnp.exp(-0.5 * d2 * invL2)
    diagonal = -jnp.sum(phi) * invL2 * dP
    outer = jnp.sum(
        (phi * invL2 ** 2 * directional)[:, None, None, None, None] * diff,
        axis=0,
    )
    return diagonal + outer


def _scale_joint_step_to_bounds(p, vX, vp, HvX, Hvp):
    """Apply one fraction-to-boundary factor to the entire Newton model step."""
    nonzero = jnp.where(vp != 0.0, vp, jnp.ones_like(vp))
    limits = jnp.where(vp < 0.0, -0.99 * p / nonzero, 0.99 * (1.0 - p) / nonzero)
    limits = jnp.where(vp != 0.0, limits, jnp.ones_like(limits))
    fraction = jnp.minimum(1.0, jnp.min(limits))
    return fraction * vX, fraction * vp, fraction * HvX, fraction * Hvp


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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _exchange_block_specs(block_sizes):
    if block_sizes is None:
        return None
    return (("sizes", tuple(int(s) for s in block_sizes)),)


def _solve_impl(
    P0: jax.Array,
    n_electrons: float,
    tol_E: float,
    tol_grad: float,
    max_step: float,
    bt_shrink: float,
    denom_scale: float,
    *,
    h: jax.Array,
    weights_b: jax.Array,
    weight_sum: jax.Array,
    VR: jax.Array,
    T: float,
    refP: jax.Array,
    HH: jax.Array,
    include_hartree: bool,
    include_exchange: bool,
    exchange_hermitian_channel_packing: bool,
    contact_g: jax.Array,
    contact_Oi: jax.Array,
    contact_Oj: jax.Array,
    max_iter: int,
    bt_max: int,
    cg_restart: int,
    mu_maxiter: int,
    plateau_window: int,
    block_sizes: tuple | None,
    project_fn,
    fock_build_fn=None,
) -> SolveResult:
    """Direct minimization via preconditioned Riemannian CG.

    Jointly optimizes orbitals (Q on Stiefel manifold via Cayley retraction)
    and occupations (p on capped simplex). True-objective backtracking may
    require multiple Fock builds per outer iteration.

    Implementation detail — users should call :func:`solve`
    (or the :func:`solve` alias) instead.  JIT caching is handled by that wrapper.

    Only structurally-static arguments (``max_iter``, ``bt_max``,
    ``cg_restart``, ``mu_maxiter``, ``block_sizes``, ``project_fn``, plus the
    Hartree/exchange/hcp flags from the kernel) are marked as static in the
    JIT decorator.  Float tolerances and step sizes stay dynamic so that
    tuning them does not trigger recompilation.
    """
    target_dtype = h.dtype
    real_dtype = jnp.zeros((), dtype=target_dtype).real.dtype
    tiny = jnp.asarray(1e-30, dtype=real_dtype)

    _project = project_fn if project_fn is not None else (lambda A: A)
    _fock_fn = fock_build_fn if fock_build_fn is not None else build_fock
    block_specs = _exchange_block_specs(block_sizes)

    w2d = jnp.asarray(weights_b[..., 0, 0], dtype=real_dtype)
    weight_sum_r = jnp.asarray(weight_sum, dtype=real_dtype)
    w_norm = w2d / jnp.maximum(weight_sum_r, tiny)

    n_target = jnp.asarray(n_electrons, dtype=real_dtype)
    n_target_norm = n_target / jnp.maximum(weight_sum_r, tiny)

    T_r = jnp.maximum(jnp.asarray(T, dtype=real_dtype), jnp.asarray(1e-12, dtype=real_dtype))
    tol_E_r = jnp.asarray(tol_E, dtype=real_dtype)
    tol_grad_r = jnp.asarray(tol_grad, dtype=real_dtype)
    max_step_r = jnp.asarray(max_step, dtype=real_dtype)
    bt_shrink_r = jnp.asarray(bt_shrink, dtype=real_dtype)
    denom_scale_r = jnp.asarray(denom_scale, dtype=real_dtype)

    nb = h.shape[-1]
    offdiag = (1.0 - jnp.eye(nb, dtype=real_dtype))[None, None, ...]

    # ---- Initialize (Q, p) from Fock at P0 ----
    # Build initial Fock matrix and diagonalize to get a good starting point.
    # This handles P0=0 gracefully (gives non-interacting ground state).
    P0_h = _herm(_project(jnp.asarray(P0, dtype=target_dtype)))
    Sigma0, H0, F0 = _fock_fn(
        P0_h, h=h, VR=VR, refP=refP, HH=HH, w2d=w2d,
        include_exchange=include_exchange, include_hartree=include_hartree,
        exchange_hermitian_channel_packing=exchange_hermitian_channel_packing,
        contact_g=contact_g, contact_Oi=contact_Oi, contact_Oj=contact_Oj,
        exchange_block_specs=block_specs,
        project_fn=project_fn,
    )
    eps0, Q0 = jnp.linalg.eigh(F0)
    eps0 = eps0.astype(real_dtype)
    Q0 = Q0.astype(target_dtype)
    mu0 = _solve_mu(eps0, w_norm, n_target_norm,
                    jnp.asarray(0.0, dtype=real_dtype), T_r, maxiter=mu_maxiter)
    p0 = jax.nn.sigmoid((mu0 - eps0) / T_r).astype(real_dtype)

    # ---- History buffers ----
    hist_E = jnp.zeros(max_iter, dtype=real_dtype)
    hist_free = jnp.zeros(max_iter, dtype=real_dtype)
    hist_grad = jnp.zeros(max_iter, dtype=real_dtype)

    # ---- Carry for lax.while_loop ----
    # (iter, Q, p, mu, d_Q, d_p, G_Q_prev, g_p_prev, stationarity_norm,
    #  free_energy_prev, free_energy_improvement, hist_E, hist_free_energy,
    #  hist_grad)
    zeros_Q = jnp.zeros_like(Q0)
    zeros_p = jnp.zeros_like(p0)

    carry0 = (
        jnp.int32(0),       # k
        Q0,                  # Q
        p0,                  # p
        mu0,                 # mu
        zeros_Q,             # d_Q (CG search direction, orbital)
        zeros_p,             # d_p (CG search direction, occupation)
        zeros_Q,             # G_Q_prev (previous orbital gradient)
        zeros_p,             # g_p_prev (previous occupation gradient)
        jnp.asarray(jnp.inf, dtype=real_dtype),  # grad_norm
        jnp.asarray(jnp.inf, dtype=real_dtype),  # previous free energy
        jnp.asarray(jnp.inf, dtype=real_dtype),  # previous free-energy improvement
        hist_E,
        hist_free,
        hist_grad,
    )

    def cond(carry):
        # Stopping rule (mirrors cpp_hf solve_dm):
        #
        # tol_grad > 0 — the joint orbital/occupation stationarity criterion is
        # the sole stop: continue only while grad_norm > tol_grad; the energy criterion
        # is inactive.  grad_norm in the carry was measured at the previous
        # body's pre-step point and is the last recorded history entry, so the
        # loop stops with the histories ending at the qualifying (E, grad)
        # pair; the returned state is one accepted descent step past it.
        #
        # tol_grad == 0 — windowed free-energy test: stop only when the objective
        # improved by less than tol_E over the last `plateau_window` recorded
        # iters and the joint stationarity residual is small.  An objective
        # increase always continues: it must never be interpreted as a plateau.
        k, _, _, _, _, _, _, _, grad_norm, _, dE, _, hist_free, _ = carry
        objective_scale = jnp.maximum(
            jnp.abs(hist_free[jnp.maximum(k - 1, 0)]), 1.0,
        )
        energy_noise = (
            jnp.asarray(100.0 * jnp.finfo(real_dtype).eps, dtype=real_dtype)
            * objective_scale
        )
        if plateau_window > 0:
            e_now = hist_free[jnp.maximum(k - 1, 0)]
            e_past = hist_free[jnp.maximum(k - 1 - plateau_window, 0)]
            improvement = e_past - e_now
            energy_not_converged = jnp.where(
                k > plateau_window,
                jnp.logical_or(
                    improvement < -energy_noise, improvement > tol_E_r,
                ),
                jnp.logical_or(dE < -energy_noise, dE > tol_E_r),
            )
        else:
            energy_not_converged = jnp.logical_or(
                dE < -energy_noise, dE > tol_E_r,
            )
        stationarity_tol = jnp.maximum(
            jnp.sqrt(jnp.maximum(tol_E_r, 0.0)),
            jnp.asarray(jnp.sqrt(jnp.finfo(real_dtype).eps), dtype=real_dtype),
        )
        not_converged = jnp.where(
            tol_grad_r > 0.0,
            grad_norm > tol_grad_r,
            jnp.logical_or(energy_not_converged, grad_norm > stationarity_tol),
        )
        return jnp.logical_and(k < max_iter, not_converged)

    def body(carry):
        (k, Q, p, mu, d_Q_prev, d_p_prev,
         G_Q_prev, g_p_prev, _grad_norm_prev, E_prev, _dE_prev,
         hE, hO, hG) = carry

        # ========== 1. Build Fock matrix (THE expensive step) ==========
        # The constrained gradient uses the projected full Fock, even when
        # the bare Hamiltonian or interaction does not share the symmetry.
        P_cur, Q, p, chart_changed = _project_density_eigensystem(Q, p, project_fn)
        Sigma, H_h, F = _fock_fn(
            P_cur, h=h, VR=VR, refP=refP, HH=HH, w2d=w2d,
            include_exchange=include_exchange, include_hartree=include_hartree,
            exchange_hermitian_channel_packing=exchange_hermitian_channel_packing,
            contact_g=contact_g, contact_Oi=contact_Oi, contact_Oj=contact_Oj,
            exchange_block_specs=block_specs,
            project_fn=project_fn,
        )
        E = hf_energy(
            P_cur, h=h, Sigma=Sigma, H=H_h, weights_b=weights_b, refP=refP,
        )
        Omega = free_energy(E, p, w2d, T_r)
        Ft = _fock_in_orbital_basis(Q, F)
        eps = jnp.real(jnp.diagonal(Ft, axis1=-2, axis2=-1)).astype(real_dtype)

        # ========== 2. Gradient ==========
        # Orbital gradient: G_ij = skew_herm((p_j - p_i) * Ft_ij)
        diff_p = p[..., None, :] - p[..., :, None]
        G_Q = _skew_hermitian(diff_p * Ft) * offdiag

        # Occupation gradient on the fixed-particle tangent space, with
        # saturated bands masked out (see _occupation_gradient).
        g_p, h_p, _curv_p, _active_p = _occupation_gradient(eps, p, T_r, w_norm)

        # A Fermi-Dirac fixed-point residual remains well-defined at saturated
        # occupations, where the logit gradient is numerically singular.  Both
        # orbital and occupation stationarity are required for convergence.
        eps_fd, Q_fd = jnp.linalg.eigh(F)
        mu_fd = _solve_mu(
            eps_fd.astype(real_dtype), w_norm, n_target_norm, mu, T_r,
            maxiter=mu_maxiter,
        )
        p_fd = jax.nn.sigmoid((mu_fd - eps_fd) / T_r).astype(real_dtype)
        P_fd = _herm(_project(_density_from_Qp(Q_fd.astype(target_dtype), p_fd)))
        orbital_norm = _norm_matrix(G_Q, w_norm)
        fixed_point_norm = _norm_matrix(P_cur - P_fd, w_norm)
        grad_norm = jnp.maximum(orbital_norm, fixed_point_norm)

        # ========== 3. Precondition ==========
        # Orbital: scale by inverse energy gaps
        gap = eps[..., :, None] - eps[..., None, :]
        eps_scale = jnp.sqrt(jnp.mean(eps ** 2) + tiny)
        lam = jnp.maximum(T_r, denom_scale_r * eps_scale)
        denom = jnp.sqrt(gap ** 2 + lam ** 2)
        H_Q = G_Q / denom

        # ========== 4. CG direction (Polak-Ribiere) ==========
        # On first iteration (k=0), just use steepest descent
        # For CG: transport previous gradient/direction to current tangent space
        # Since Q changed by Q_new = Q_old @ U, the transport is U† * prev * U.
        # We approximate transport as identity (reset via periodic restart).

        # Preconditioned Polak-Ribiere+: beta = <g, M^-1 g - M^-1 g_prev> / <g_prev, g_prev>
        # Here H_Q = M^-1 G_Q is the preconditioned orbital gradient, h_p = M^-1 g_p likewise.
        # The denominator uses the unpreconditioned squared norm of the previous
        # gradient — always well-scaled and avoids the division-then-multiplication
        # round-trip an earlier draft had.
        pr_num = (_ip_matrix(G_Q, H_Q, w_norm) - _ip_matrix(G_Q_prev, H_Q, w_norm)
                  + _ip_vec(g_p, h_p, w_norm) - _ip_vec(g_p_prev, h_p, w_norm))
        pr_den = _ip_matrix(G_Q_prev, G_Q_prev, w_norm) + _ip_vec(g_p_prev, g_p_prev, w_norm)
        beta = jnp.where(pr_den > tiny, jnp.maximum(0.0, pr_num / pr_den), 0.0)
        beta = jnp.minimum(beta, 5.0)  # cap to prevent runaway
        # Reset to steepest descent on first step and every cg_restart steps
        beta = jnp.where(
            jnp.logical_or(chart_changed, jnp.logical_or(k == 0, k % cg_restart == 0)),
            0.0, beta,
        )

        # Search direction: positive preconditioned gradient.
        # The QR retraction QR(I - tau*d) steps in the -d direction,
        # so d = +H_Q gives descent along -H_Q.
        d_Q = H_Q + beta * d_Q_prev
        d_p = h_p + beta * d_p_prev

        # ========== 5. Backtracking on the true free energy ==========
        # A frozen-F surrogate is only first-order accurate and can accept a
        # genuine free-energy increase, producing stable two-cycles that a
        # signed plateau test then mistakes for convergence.  Evaluate the
        # actual nonlinear HF free energy at every candidate.
        #
        # Spectral Cayley path: eigendecompose (i * d_Q) once and pre-rotate Ft
        # into the d_Q eigenbasis.  Each line-search trial then costs one
        # Hadamard scaling + one matmul (versus an LU solve + two matmuls in
        # the LU-based Cayley). Cache the accepted trial's (Q,p), including
        # any eigensystem repair after symmetry projection.
        V_d, lam_d = _cayley_spectral_setup(d_Q)
        Ft_eig = jnp.conj(jnp.swapaxes(V_d, -2, -1)) @ Ft @ V_d

        d_Q_norm = _norm_matrix(d_Q, w_norm)
        tau0 = jnp.minimum(1.0, max_step_r / jnp.maximum(d_Q_norm, tiny))

        def trial_free_energy(tau_val):
            U_trial = _cayley_unitary_from_spectrum(V_d, lam_d, tau_val)
            Q_trial = Q @ U_trial
            p_trial = _project_capped_occupations(
                p - tau_val * d_p, w_norm, n_target_norm,
            )
            P_trial, Q_trial, p_trial, recharted = _project_density_eigensystem(
                Q_trial, p_trial, project_fn,
            )
            Sigma_trial, H_trial, _ = _fock_fn(
                P_trial, h=h, VR=VR, refP=refP, HH=HH, w2d=w2d,
                include_exchange=include_exchange, include_hartree=include_hartree,
                exchange_hermitian_channel_packing=exchange_hermitian_channel_packing,
                contact_g=contact_g, contact_Oi=contact_Oi, contact_Oj=contact_Oj,
                exchange_block_specs=block_specs, project_fn=None,
            )
            E_trial = hf_energy(
                P_trial, h=h, Sigma=Sigma_trial, H=H_trial,
                weights_b=weights_b, refP=refP,
            )
            return free_energy(E_trial, p_trial, w2d, T_r), Q_trial, p_trial, recharted

        def bt_cond(state):
            i, _tau, accepted, *_ = state
            return jnp.logical_and(i < bt_max, jnp.logical_not(accepted))

        def bt_body(state):
            i, tau, _accepted, *_ = state
            Omega_trial, Q_trial, p_trial, recharted = trial_free_energy(tau)
            accepted = Omega_trial < Omega
            tau_next = jnp.where(accepted, tau, tau * bt_shrink_r)
            return i + 1, tau_next, accepted, Omega_trial, Q_trial, p_trial, recharted

        _, tau_final, bt_accepted, Omega_bt, Q_bt, p_bt, recharted = lax.while_loop(
            bt_cond, bt_body,
            (jnp.int32(0), tau0, jnp.bool_(False), Omega, Q, p, jnp.bool_(False)),
        )
        recharted = jnp.logical_and(bt_accepted, recharted)
        tau_final = jnp.where(bt_accepted, tau_final, jnp.asarray(0.0, real_dtype))
        if project_fn is None:
            # Retain the original unprojected retraction order: reusing trial
            # arrays changes rounding enough to stall float32 convergence.
            # The compiled line search need not carry unused trial orbitals.
            U = _cayley_unitary_from_spectrum(V_d, lam_d, tau_final)
            Q_new = Q @ U
            p_new = _project_capped_occupations(
                p - tau_final * d_p, w_norm, n_target_norm,
            )
        else:
            # Projection may replace the eigensystem. Reuse the accepted one
            # so its entropy and orbital chart remain consistent.
            Q_new = jnp.where(bt_accepted, Q_bt, Q)
            p_new = jnp.where(bt_accepted, p_bt, p)
        # The spectral shortcut applies until projection changes the chart.
        eps_new = lax.cond(
            recharted,
            lambda _: jnp.real(jnp.diagonal(
                _fock_in_orbital_basis(Q_new, F), axis1=-2, axis2=-1,
            )).astype(real_dtype),
            lambda _: _diag_UFU_from_spectrum(
                V_d, Ft_eig, lam_d, tau_final,
            ).astype(real_dtype),
            operand=None,
        )
        mu_new = _solve_mu(eps_new, w_norm, n_target_norm, mu, T_r, maxiter=mu_maxiter)

        # When near-equal occupations hide a large full FD residual, the local
        # (Q,p) direction can make negligible progress.  Compare it with a
        # damped density fixed-point direction, retaining only a true
        # free-energy descent and a consistent eigendecomposition.
        # ``Omega_bt`` is the objective already evaluated for the accepted
        # candidate. Reusing it avoids an otherwise redundant Fock build.
        Omega_direct = jnp.where(bt_accepted, Omega_bt, Omega)

        def fb_cond(state):
            i, _lam, accepted_fb, *_ = state
            return jnp.logical_and(i < bt_max, jnp.logical_not(accepted_fb))

        def fb_body(state):
            i, lam_fb, _accepted_fb, *_rest = state
            P_mix = _herm(
                _project(P_cur + lam_fb.astype(target_dtype) * (P_fd - P_cur))
            )
            p_mix, Q_mix = jnp.linalg.eigh(P_mix)
            p_mix = _project_capped_occupations(
                p_mix.astype(real_dtype), w_norm, n_target_norm,
            )
            Q_mix = Q_mix.astype(target_dtype)
            P_mix, Q_mix, p_mix, repaired = _project_density_eigensystem(Q_mix, p_mix, project_fn)
            Sigma_mix, H_mix, _ = _fock_fn(
                P_mix, h=h, VR=VR, refP=refP, HH=HH, w2d=w2d,
                include_exchange=include_exchange, include_hartree=include_hartree,
                exchange_hermitian_channel_packing=exchange_hermitian_channel_packing,
                contact_g=contact_g, contact_Oi=contact_Oi, contact_Oj=contact_Oj,
                exchange_block_specs=block_specs, project_fn=None,
            )
            E_mix = hf_energy(
                P_mix, h=h, Sigma=Sigma_mix, H=H_mix,
                weights_b=weights_b, refP=refP,
            )
            O_mix = free_energy(E_mix, p_mix, w2d, T_r)
            accepted_fb = O_mix < Omega
            lam_next = jnp.where(accepted_fb, lam_fb, 0.5 * lam_fb)
            return i + 1, lam_next, accepted_fb, Q_mix, p_mix, O_mix, repaired

        fb0 = (
            jnp.int32(0), jnp.asarray(1.0, real_dtype), jnp.bool_(False),
            Q, p, Omega, jnp.bool_(False),
        )
        _fbi, _fbl, fb_accepted, Q_fb, p_fb, Omega_fb, repaired_fb = lax.while_loop(
            fb_cond, fb_body, fb0,
        )
        use_fb = jnp.logical_and(fb_accepted, Omega_fb < Omega_direct)
        Q_new = jnp.where(use_fb, Q_fb, Q_new)
        p_new = jnp.where(use_fb, p_fb, p_new)
        Ft_fb = _fock_in_orbital_basis(Q_fb, F)
        eps_fb = jnp.real(
            jnp.diagonal(Ft_fb, axis1=-2, axis2=-1)
        ).astype(real_dtype)
        mu_fb = _solve_mu(
            eps_fb, w_norm, n_target_norm, mu, T_r, maxiter=mu_maxiter,
        )
        mu_new = jnp.where(use_fb, mu_fb, mu_new)

        # ========== 7. Record history + energy change ==========
        hE = hE.at[k].set(E)
        hO = hO.at[k].set(Omega)
        hG = hG.at[k].set(grad_norm)
        dE = E_prev - Omega

        # A material projection repair replaces the orbital chart; its CG
        # history cannot be reused. Keep ordinary fallback behavior unchanged.
        restart = jnp.where(use_fb, repaired_fb, recharted)
        d_Q = jnp.where(restart, zeros_Q, d_Q)
        d_p = jnp.where(restart, zeros_p, d_p)
        G_Q = jnp.where(restart, zeros_Q, G_Q)
        g_p = jnp.where(restart, zeros_p, g_p)
        return (k + 1, Q_new, p_new, mu_new, d_Q, d_p, G_Q, g_p, grad_norm,
                Omega, dE, hE, hO, hG)

    # ---- Run ----
    (k_fin, Q_fin, p_fin, mu_fin, _, _, _, _, grad_fin, E_fin_loop, dE_fin,
     hist_E, hist_free, hist_grad) = lax.while_loop(cond, body, carry0)

    # ---- Finalize without changing the optimized state -------------------
    # A previous finalize diagonalized F[P] and re-occupied it, which is an
    # additional, unchecked SCF step rather than a gauge cleanup.  It could
    # materially change both density and energy after convergence.
    P_fin, Q_fin, p_fin, _ = _project_density_eigensystem(Q_fin, p_fin, project_fn)
    Sigma_fin, H_fin, F_fin = _fock_fn(
        P_fin, h=h, VR=VR, refP=refP, HH=HH, w2d=w2d,
        include_exchange=include_exchange, include_hartree=include_hartree,
        exchange_hermitian_channel_packing=exchange_hermitian_channel_packing,
        contact_g=contact_g, contact_Oi=contact_Oi, contact_Oj=contact_Oj,
        exchange_block_specs=block_specs,
        project_fn=project_fn,
    )
    E_fin = hf_energy(
        P_fin, h=h, Sigma=Sigma_fin, H=H_fin, weights_b=weights_b, refP=refP,
    )

    # Recheck the full self-consistency residual at the exact returned state.
    eps_check, Q_check = jnp.linalg.eigh(F_fin)
    mu_check = _solve_mu(
        eps_check.astype(real_dtype), w_norm, n_target_norm, mu_fin, T_r,
        maxiter=mu_maxiter,
    )
    p_check = jax.nn.sigmoid((mu_check - eps_check) / T_r).astype(real_dtype)
    P_check = _herm(_project(_density_from_Qp(Q_check.astype(target_dtype), p_check)))
    final_res = _norm_matrix(P_fin - P_check, w_norm)

    # A single canonical fixed-point polish is safe only when it is explicitly
    # rechecked and improves an already-small residual.  This recovers a clean
    # float32 output without reviving the old unconditional finalize step that
    # could materially change a nonstationary solution.
    if project_fn is None:
        Sigma_polish, H_polish, F_polish = _fock_fn(
            P_check, h=h, VR=VR, refP=refP, HH=HH, w2d=w2d,
            include_exchange=include_exchange, include_hartree=include_hartree,
            exchange_hermitian_channel_packing=exchange_hermitian_channel_packing,
            contact_g=contact_g, contact_Oi=contact_Oi, contact_Oj=contact_Oj,
            exchange_block_specs=block_specs, project_fn=None,
        )
        eps_next, Q_next = jnp.linalg.eigh(F_polish)
        mu_next = _solve_mu(
            eps_next.astype(real_dtype), w_norm, n_target_norm, mu_check, T_r,
            maxiter=mu_maxiter,
        )
        p_next = jax.nn.sigmoid((mu_next - eps_next) / T_r).astype(real_dtype)
        P_next = _herm(_density_from_Qp(Q_next.astype(target_dtype), p_next))
        polish_res = _norm_matrix(P_check - P_next, w_norm)
        polish_limit = jnp.maximum(
            jnp.where(
                tol_grad_r > 0.0,
                tol_grad_r,
                jnp.sqrt(jnp.maximum(tol_E_r, 0.0)),
            ),
            jnp.asarray(jnp.sqrt(jnp.finfo(real_dtype).eps), dtype=real_dtype),
        )
        use_polish = jnp.logical_and(
            final_res <= polish_limit, polish_res < final_res,
        )
        Q_fin = jnp.where(use_polish, Q_check.astype(target_dtype), Q_fin)
        p_fin = jnp.where(use_polish, p_check, p_fin)
        P_fin = jnp.where(use_polish, P_check, P_fin)
        Sigma_fin = jnp.where(use_polish, Sigma_polish, Sigma_fin)
        H_fin = jnp.where(use_polish, H_polish, H_fin)
        F_fin = jnp.where(use_polish, F_polish, F_fin)
        final_res = jnp.where(use_polish, polish_res, final_res)
        E_polish = hf_energy(
            P_check, h=h, Sigma=Sigma_polish, H=H_polish,
            weights_b=weights_b, refP=refP,
        )
        E_fin = jnp.where(use_polish, E_polish, E_fin)

    Ft_return = _fock_in_orbital_basis(Q_fin, F_fin)
    eps_return = jnp.real(
        jnp.diagonal(Ft_return, axis1=-2, axis2=-1)
    ).astype(real_dtype)
    mu_fin = _solve_mu(
        eps_return, w_norm, n_target_norm, mu_fin, T_r, maxiter=mu_maxiter,
    )

    # Convergence flag.  Gradient-primary when tol_grad > 0: grad_fin is the
    # last measured gradient — the entry the loop stopped on, or the final
    # recorded one when max_iter ran out (which cpp_hf's mid-iteration break
    # would have accepted; the plain k_fin < max_iter test would miss it).
    # tol_grad == 0 keeps the legacy flag: loop exited early on the energy
    # criterion.
    stationarity_tol = jnp.maximum(
        jnp.where(tol_grad_r > 0.0, tol_grad_r, jnp.sqrt(jnp.maximum(tol_E_r, 0.0))),
        jnp.where(
            tol_grad_r > 0.0,
            jnp.asarray(100.0 * jnp.finfo(real_dtype).eps, dtype=real_dtype),
            jnp.asarray(jnp.sqrt(jnp.finfo(real_dtype).eps), dtype=real_dtype),
        ),
    )
    converged = jnp.logical_and(
        final_res <= stationarity_tol,
        jnp.where(tol_grad_r > 0.0, grad_fin <= tol_grad_r, k_fin < max_iter),
    )

    return SolveResult(
        Q=Q_fin,
        p=p_fin,
        mu=mu_fin,
        density=P_fin,
        fock=F_fin,
        energy=E_fin,
        n_iter=k_fin,
        converged=converged,
        history=dict(E=hist_E, free_energy=hist_free, grad_norm=hist_grad),
    )


_solve_jitted = jax.jit(
    _solve_impl,
    static_argnames=(
        # Kernel-side static flags (affect Fock build dispatch)
        "include_hartree",
        "include_exchange",
        "exchange_hermitian_channel_packing",
        # SolverConfig fields that are structurally static (integers used as
        # loop bounds, callables, or tuples affecting graph shape)
        "max_iter",
        "bt_max",
        "cg_restart",
        "mu_maxiter",
        "plateau_window",
        "block_sizes",
        "project_fn",
        # Optional Fock build override (e.g. superlattice streaming Fock).
        # Captured by identity, so passing the same closure is cache-friendly.
        "fock_build_fn",
    ),
)


def _solve_newton_impl(
    P0: jax.Array,
    n_electrons: float,
    tol_grad: float,
    tr_delta0: float,
    denom_scale: float,
    *,
    h: jax.Array,
    weights_b: jax.Array,
    weight_sum: jax.Array,
    VR: jax.Array,
    T: float,
    refP: jax.Array,
    HH: jax.Array,
    include_hartree: bool,
    include_exchange: bool,
    exchange_hermitian_channel_packing: bool,
    contact_g: jax.Array,
    contact_Oi: jax.Array,
    contact_Oj: jax.Array,
    deflation_targets: jax.Array,
    deflation_sigma: float,
    deflation_length: float,
    has_deflation: bool,
    max_iter: int,
    tr_cg_max: int,
    mu_maxiter: int,
    block_sizes: tuple | None,
    project_fn,
    fock_build_fn=None,
) -> SolveResult:
    """Trust-region Newton via Steihaug truncated-CG on the joint (Q, p) Hessian.

    Minimizes the same free energy as :func:`_solve_impl`, but with a
    second-order step: the joint (Q, p) Hessian-vector product uses the exact
    interaction response (one Fock build, or a JVP for a custom builder),
    and a Steihaug truncated-CG solves the trust-region subproblem. Convergence
    requires both the orbital gradient and the full Fermi-Dirac fixed-point
    residual. It often needs fewer outer iterations than CG on stiff problems.
    """
    target_dtype = h.dtype
    real_dtype = jnp.zeros((), dtype=target_dtype).real.dtype
    tiny = jnp.asarray(1e-30, dtype=real_dtype)

    _project = project_fn if project_fn is not None else (lambda A: A)
    _fock_fn = fock_build_fn if fock_build_fn is not None else build_fock
    block_specs = _exchange_block_specs(block_sizes)

    w2d = jnp.asarray(weights_b[..., 0, 0], dtype=real_dtype)
    weight_sum_r = jnp.asarray(weight_sum, dtype=real_dtype)
    w_norm = w2d / jnp.maximum(weight_sum_r, tiny)

    n_target = jnp.asarray(n_electrons, dtype=real_dtype)
    n_target_norm = n_target / jnp.maximum(weight_sum_r, tiny)

    T_r = jnp.maximum(jnp.asarray(T, dtype=real_dtype), jnp.asarray(1e-12, dtype=real_dtype))
    tol_g = jnp.asarray(tol_grad, dtype=real_dtype)
    denom_scale_r = jnp.asarray(denom_scale, dtype=real_dtype)
    delta0 = jnp.asarray(tr_delta0, dtype=real_dtype)
    delta_min = jnp.asarray(1e-10, dtype=real_dtype)
    # Relative energy-noise floor for the trust-region ratio: ~4500 * machine
    # epsilon, i.e. ~1e-12 in float64 (cpp_hf's hard-coded value) and ~5e-4 in
    # float32, tracking the precision of the Fock build that produces E.
    noise_floor = jnp.maximum(
        jnp.asarray(1e-12, dtype=real_dtype),
        jnp.asarray(4500.0 * jnp.finfo(real_dtype).eps, dtype=real_dtype),
    )

    nb = h.shape[-1]
    offdiag = (1.0 - jnp.eye(nb, dtype=real_dtype))[None, None, ...]
    refP_zero = jnp.zeros_like(refP)

    def _fock_full(P):
        return _fock_fn(
            P, h=h, VR=VR, refP=refP, HH=HH, w2d=w2d,
            include_exchange=include_exchange, include_hartree=include_hartree,
            exchange_hermitian_channel_packing=exchange_hermitian_channel_packing,
            contact_g=contact_g, contact_Oi=contact_Oi, contact_Oj=contact_Oj,
            exchange_block_specs=block_specs, project_fn=None,
        )

    def _fock_response(P, dP):
        # A custom builder may capture reference/one-body terms internally.
        # Differentiate its actual Fock instead of assuming F[dP] - h.
        if fock_build_fn is not None:
            return jax.jvp(lambda density: _fock_full(density)[2], (P,), (dP,))[1]
        # Standard HF response is linear: omit the one-body/reference terms
        # directly to avoid cancellation against a large one-body matrix.
        _, _, F_resp = _fock_fn(
            dP, h=jnp.zeros_like(h), VR=VR, refP=refP_zero, HH=HH, w2d=w2d,
            include_exchange=include_exchange, include_hartree=include_hartree,
            exchange_hermitian_channel_packing=exchange_hermitian_channel_packing,
            contact_g=contact_g, contact_Oi=contact_Oi, contact_Oj=contact_Oj,
            exchange_block_specs=block_specs, project_fn=None,
        )
        return F_resp

    # The gradient and Hessian represent Omega / weight_sum in this metric.
    # Different weights in the two sectors break Hessian self-adjointness.
    def _jip(aX, ap, bX, bp):
        return _joint_inner_product(aX, ap, bX, bp, w_norm)

    def _jip_norm(aX, ap):
        return jnp.sqrt(jnp.maximum(0.0, _jip(aX, ap, aX, ap)))

    # Deflation bias (Newton-only): a repulsive Gaussian hill around each
    # already-found density, so a re-solve is pushed out of those basins and
    # converges to a DISTINCT HF solution.
    #   Phi      = sigma * sum_i exp(-d_i^2 / (2 L^2)),
    #   d_i^2    = sum_k w_k ||P_k - target_{i,k}||_F^2   (gauge-invariant in P)
    #   S_pen    = sum_i 2 phi'(d_i^2) (P - target_i),    phi'(x) = -phi(x)/(2L^2)
    # The force S_pen is folded into the Fock (F_eff = F + S_pen) so the gradient
    # sees the bias. Both diagonal and outer-product penalty curvature are
    # included in the Hessian-vector product. ``has_deflation`` is a
    # static flag, so a normal Newton solve compiles with none of this.
    defl_sigma = jnp.asarray(deflation_sigma, dtype=real_dtype)

    def _eval_state(Q, p):
        """Objectives, orbital-basis Fock, stationarity, and penalty curvature.

        One Fock build.  Convergence is judged on the orbital gradient
        (equivalently the commutator ``[F, P]``) and the occupation
        Fermi-Dirac residual.  With deflation the Fock carries the penalty force
        and ``Omega`` is the biased free energy (the physical internal energy
        is recomputed unbiased at finalize).
        """
        P, Q, p, _ = _project_density_eigensystem(Q, p, project_fn)
        Sigma, H_h, F = _fock_full(P)
        E = hf_energy(
            P, h=h, Sigma=Sigma, H=H_h, weights_b=weights_b, refP=refP,
        )
        Omega = free_energy(E, p, w2d, T_r)
        if has_deflation:
            Phi, S_pen, pen_diag = _deflation_bias(
                P, deflation_targets, w2d, defl_sigma, deflation_length,
            )
            Omega = Omega + Phi
            F = F + S_pen
        else:
            pen_diag = jnp.zeros((), dtype=real_dtype)
        Ft = _fock_in_orbital_basis(Q, _herm(_project(F)))
        diff_p = p[..., None, :] - p[..., :, None]
        G = _skew_hermitian(diff_p * Ft) * offdiag
        eps, V_fd = jnp.linalg.eigh(Ft)
        eps = eps.astype(real_dtype)
        mu_fd = _solve_mu(
            eps, w_norm, n_target_norm, jnp.asarray(0.0, dtype=real_dtype),
            T_r, maxiter=mu_maxiter,
        )
        p_fd = jax.nn.sigmoid((mu_fd - eps) / T_r).astype(real_dtype)
        Q_fd = Q @ V_fd.astype(target_dtype)
        P_fd = _herm(_project(_density_from_Qp(Q_fd, p_fd)))
        grad_norm = jnp.maximum(
            _norm_matrix(G, w_norm),
            _norm_matrix(P - P_fd, w_norm),
        )
        return E, Omega, Ft, grad_norm, pen_diag, Q, p

    # ---- Initialize (Q, p) from Fock at P0 ----
    P0_h = _herm(_project(jnp.asarray(P0, dtype=target_dtype)))
    _, _, F0 = _fock_full(P0_h)
    F0 = _herm(_project(F0))
    eps0, Q0 = jnp.linalg.eigh(F0)
    eps0 = eps0.astype(real_dtype)
    Q0 = Q0.astype(target_dtype)
    mu0 = _solve_mu(eps0, w_norm, n_target_norm,
                    jnp.asarray(0.0, dtype=real_dtype), T_r, maxiter=mu_maxiter)
    p0 = jax.nn.sigmoid((mu0 - eps0) / T_r).astype(real_dtype)
    E0, Omega0, Ft0, grad0, pen0, Q0, p0 = _eval_state(Q0, p0)

    hist_E = jnp.zeros(max_iter, dtype=real_dtype)
    hist_free = jnp.zeros(max_iter, dtype=real_dtype)
    hist_grad = jnp.zeros(max_iter, dtype=real_dtype)

    # carry: (k, Q, p, mu, Ft, E, Omega, Delta, grad_norm, pen_diag,
    #         hist_E, hist_free, hist_grad)
    carry0 = (
        jnp.int32(0), Q0, p0, mu0, Ft0, E0, Omega0, delta0, grad0, pen0,
        hist_E, hist_free, hist_grad,
    )

    def outer_cond(carry):
        k, _, _, _, _, _, _, Delta, grad_norm, _, _, _, _ = carry
        return jnp.logical_and(
            k < max_iter,
            jnp.logical_and(grad_norm > tol_g, Delta > delta_min),
        )

    def outer_body(carry):
        (
            k, Q, p, mu, Ft, E, Omega, Delta, grad_norm, pen_diag,
            hE, hO, hG,
        ) = carry
        hE = hE.at[k].set(E)
        hO = hO.at[k].set(Omega)
        hG = hG.at[k].set(grad_norm)

        eps = jnp.real(jnp.diagonal(Ft, axis1=-2, axis2=-1)).astype(real_dtype)
        diff_p = p[..., None, :] - p[..., :, None]
        G = _skew_hermitian(diff_p * Ft) * offdiag
        g_p, _h_p, curv_p, _active_p = _occupation_gradient(eps, p, T_r, w2d)

        eps_scale = jnp.sqrt(jnp.mean(eps ** 2) + tiny)
        lam_pc = jnp.maximum(T_r, denom_scale_r * eps_scale)

        # Block preconditioner: orbital 1/sqrt(gap^2 + lam^2), occupation
        # p(1-p)/T, then project the occupation onto particle conservation.
        gap = eps[..., :, None] - eps[..., None, :]
        denomX = jnp.sqrt(gap ** 2 + lam_pc ** 2)

        def _prec(aX, ap):
            return aX / denomX, _project_occ_curvature(ap, w2d, curv_p)

        P_base = _herm(_project(_density_from_Qp(Q, p)))

        def _response(dP):
            resp = _fock_response(P_base, dP)
            if has_deflation:
                resp = resp + _deflation_response(
                    P_base, dP, deflation_targets, w2d, defl_sigma, deflation_length,
                )
            return resp

        def _hvp(X, dp):
            return _joint_hvp(Q, p, Ft, X, dp, T_r, w2d, _response, project_fn)

        # ---- Steihaug truncated CG on the joint quadratic model ----
        rX0 = -G
        rp0 = -g_p
        zX0, zp0 = _prec(rX0, rp0)
        rz0 = _jip(rX0, rp0, zX0, zp0)
        residual0 = _jip_norm(rX0, rp0)
        zerosQ = jnp.zeros_like(Q)
        zerosp = jnp.zeros_like(p)
        # carry: (cg, done, vX, vp, HvX, Hvp, rX, rp, dX, dp, rz)
        cg_carry0 = (jnp.int32(0), jnp.bool_(False),
                     zerosQ, zerosp, zerosQ, zerosp,
                     rX0, rp0, zX0, zp0, rz0)

        def cg_cond(c):
            return jnp.logical_and(c[0] < tr_cg_max, jnp.logical_not(c[1]))

        def cg_body(c):
            cg, _done, vX, vp, HvX, Hvp, rX, rp, dX, dp, rz = c
            HdX, Hdp = _hvp(dX, dp)
            dHd = _jip(dX, dp, HdX, Hdp)
            dd = _jip(dX, dp, dX, dp)
            neg_curv = dHd <= 1e-14 * dd

            # to_boundary: positive root t of ||v + t d||^2 = Delta^2.
            b_coef = 2.0 * _jip(vX, vp, dX, dp)
            cc = _jip(vX, vp, vX, vp) - Delta * Delta
            disc = jnp.maximum(b_coef ** 2 - 4.0 * dd * cc, 0.0)
            t_bd = (-b_coef + jnp.sqrt(disc)) / (2.0 * dd + 1e-30)
            vX_bd, vp_bd = vX + t_bd * dX, vp + t_bd * dp
            HvX_bd, Hvp_bd = HvX + t_bd * HdX, Hvp + t_bd * Hdp

            # Full CG step (guard alpha denom so negative-curvature stays finite).
            alpha = rz / jnp.where(neg_curv, jnp.ones_like(dHd), dHd)
            vX_a, vp_a = vX + alpha * dX, vp + alpha * dp
            norm_va = _jip_norm(vX_a, vp_a)
            cross = norm_va >= Delta
            take_boundary = jnp.logical_or(neg_curv, cross)

            HvX_a, Hvp_a = HvX + alpha * HdX, Hvp + alpha * Hdp
            rX_a, rp_a = rX - alpha * HdX, rp - alpha * Hdp
            rn = _jip_norm(rX_a, rp_a)
            inner_conv = rn < jnp.minimum(0.5, jnp.sqrt(residual0)) * residual0
            zX_a, zp_a = _prec(rX_a, rp_a)
            rz_new = _jip(rX_a, rp_a, zX_a, zp_a)
            beta = rz_new / jnp.where(jnp.abs(rz) > tiny, rz, jnp.ones_like(rz))
            dX_a, dp_a = zX_a + beta * dX, zp_a + beta * dp

            stop = jnp.logical_or(take_boundary, inner_conv)
            vX_n = jnp.where(take_boundary, vX_bd, vX_a)
            vp_n = jnp.where(take_boundary, vp_bd, vp_a)
            HvX_n = jnp.where(take_boundary, HvX_bd, HvX_a)
            Hvp_n = jnp.where(take_boundary, Hvp_bd, Hvp_a)
            rX_n = jnp.where(take_boundary, rX, rX_a)
            rp_n = jnp.where(take_boundary, rp, rp_a)
            dX_n = jnp.where(stop, dX, dX_a)
            dp_n = jnp.where(stop, dp, dp_a)
            rz_n = jnp.where(stop, rz, rz_new)
            return (cg + 1, stop, vX_n, vp_n, HvX_n, Hvp_n,
                    rX_n, rp_n, dX_n, dp_n, rz_n)

        cg_fin = lax.while_loop(cg_cond, cg_body, cg_carry0)
        _, _, vX, vp, HvX, Hvp, _, _, _, _, _ = cg_fin

        # Preserve the coupled Newton direction. One fraction-to-boundary
        # factor keeps the occupations feasible and cannot enlarge the trust
        # region step; scale its stored Hessian action by the same factor.
        vX, vp, HvX, Hvp = _scale_joint_step_to_bounds(p, vX, vp, HvX, Hvp)
        p_trial = p + vp
        V_v, lam_v = _cayley_spectral_setup(vX)
        U = _cayley_unitary_from_spectrum(
            V_v, lam_v, jnp.asarray(-1.0, dtype=real_dtype),
        )
        Qt = Q @ U

        # Predicted reduction m(0) - m(v) in TRUE energy units: the orbital
        # gradient/Hessian action are per-k (unweighted) but the energy carries
        # the BZ measure w2d, so both terms here must be w2d-weighted for the
        # trust-region ratio to compare against the actual energy change.
        pred = _joint_predicted_reduction(
            G, g_p, vX, vp, HvX, Hvp, w2d,
        )

        # True free energy + gradient at the trial point (one Fock build for
        # this evaluation).
        E_trial, Omega_trial, Ft_trial, grad_trial, pen_trial, Qt, p_trial = _eval_state(Qt, p_trial)
        eps_t = jnp.real(
            jnp.diagonal(Ft_trial, axis1=-2, axis2=-1)
        ).astype(real_dtype)
        mu_t = _solve_mu(eps_t, w_norm, n_target_norm, mu, T_r, maxiter=mu_maxiter)

        # Energy-noise floor below which the predicted reduction is meaningless
        # and the step is judged on the gradient instead.  Scaled to the working
        # precision: ~1e-12 in float64 (matching cpp_hf), ~5e-4 in float32, so
        # the trust-region ratio is never trusted below the actual Fock-build
        # round-off (otherwise float32 noise makes every ratio garbage).
        noise = noise_floor * jnp.maximum(jnp.abs(Omega), 1.0)
        use_grad_branch = pred <= noise

        # Branch 1 (pred below energy noise): judge by the gradient, never climb.
        accept_g = jnp.logical_and(
            grad_trial <= grad_norm, Omega_trial <= Omega + noise,
        )
        Delta_g = jnp.where(accept_g, Delta, 0.5 * Delta)
        # Branch 2 (standard trust-region ratio).
        ratio = (Omega - Omega_trial) / jnp.where(
            use_grad_branch, jnp.ones_like(pred), pred,
        )
        step_norm = _jip_norm(vX, vp)
        Delta_r = jnp.where(
            ratio < 0.25, 0.25 * Delta,
            jnp.where(jnp.logical_and(ratio > 0.75, step_norm > 0.9 * Delta),
                      jnp.minimum(2.0 * Delta, 5.0), Delta),
        )
        accept_r = ratio > 0.1

        accept = jnp.where(use_grad_branch, accept_g, accept_r)
        Delta_new = jnp.where(use_grad_branch, Delta_g, Delta_r)

        # If the local (Q,p) model rejects its step, fall back to a damped
        # full Fermi-Dirac fixed-point direction.  Near-equal occupations can
        # suppress ``(p_j-p_i) F_ij`` and make the parameterization gradient
        # tiny while ``||P-P_FD(F)||`` is still O(1).  Each fallback candidate
        # preserves the particle count (convex density mixing), is represented
        # by a consistent eigendecomposition (Q,p), and is accepted only on
        # true free-energy descent.
        eps_fd, V_fd = jnp.linalg.eigh(Ft)
        mu_fd = _solve_mu(
            eps_fd.astype(real_dtype), w_norm, n_target_norm, mu, T_r,
            maxiter=mu_maxiter,
        )
        p_fd = jax.nn.sigmoid((mu_fd - eps_fd) / T_r).astype(real_dtype)
        Q_fd = Q @ V_fd.astype(target_dtype)
        P_cur = _herm(_project(_density_from_Qp(Q, p)))
        P_fd = _herm(_project(_density_from_Qp(Q_fd, p_fd)))

        def fb_cond(state):
            i, _lam, accepted_fb, *_ = state
            return jnp.logical_and(i < 10, jnp.logical_not(accepted_fb))

        def fb_body(state):
            i, lam_fb, _accepted_fb, *_rest = state
            P_mix = _herm(_project(P_cur + lam_fb.astype(target_dtype) * (P_fd - P_cur)))
            p_mix, Q_mix = jnp.linalg.eigh(P_mix)
            p_mix = _project_capped_occupations(
                p_mix.astype(real_dtype), w2d, n_target,
            )
            Q_mix = Q_mix.astype(target_dtype)
            E_mix, O_mix, Ft_mix, g_mix, pen_mix, Q_mix, p_mix = _eval_state(Q_mix, p_mix)
            eps_mix = jnp.real(
                jnp.diagonal(Ft_mix, axis1=-2, axis2=-1)
            ).astype(real_dtype)
            mu_mix = _solve_mu(
                eps_mix, w_norm, n_target_norm, mu, T_r, maxiter=mu_maxiter,
            )
            delta_O = O_mix - Omega
            accepted_fb = jnp.logical_or(
                delta_O < 0.0,
                jnp.logical_and(
                    jnp.abs(delta_O) <= noise,
                    jnp.logical_and(g_mix < grad_norm, O_mix <= Omega + noise),
                ),
            )
            lam_next = jnp.where(accepted_fb, lam_fb, 0.5 * lam_fb)
            return (
                i + 1, lam_next, accepted_fb, Q_mix, p_mix, E_mix, O_mix,
                Ft_mix, g_mix, pen_mix, mu_mix,
            )

        fb0 = (
            jnp.int32(0), jnp.asarray(1.0, real_dtype), jnp.bool_(False),
            Q, p, E, Omega, Ft, grad_norm, pen_diag, mu,
        )
        (
            _fb_i, _fb_lam, fb_accept, Q_fb, p_fb, E_fb, Omega_fb, Ft_fb,
            grad_fb, pen_fb, mu_fb,
        ) = lax.while_loop(fb_cond, fb_body, fb0)
        use_fallback = jnp.logical_and(
            fb_accept,
            jnp.logical_or(jnp.logical_not(accept), grad_fb < grad_trial),
        )

        Q_new = jnp.where(use_fallback, Q_fb, jnp.where(accept, Qt, Q))
        p_new = jnp.where(use_fallback, p_fb, jnp.where(accept, p_trial, p))
        Ft_new = jnp.where(use_fallback, Ft_fb, jnp.where(accept, Ft_trial, Ft))
        E_new = jnp.where(use_fallback, E_fb, jnp.where(accept, E_trial, E))
        Omega_new = jnp.where(
            use_fallback, Omega_fb, jnp.where(accept, Omega_trial, Omega),
        )
        mu_new = jnp.where(use_fallback, mu_fb, jnp.where(accept, mu_t, mu))
        grad_new = jnp.where(
            use_fallback, grad_fb, jnp.where(accept, grad_trial, grad_norm),
        )
        pen_new = jnp.where(
            use_fallback, pen_fb, jnp.where(accept, pen_trial, pen_diag),
        )
        Delta_new = jnp.where(
            use_fallback, jnp.maximum(Delta, jnp.asarray(0.1, real_dtype)),
            Delta_new,
        )

        return (
            k + 1, Q_new, p_new, mu_new, Ft_new, E_new, Omega_new, Delta_new,
            grad_new, pen_new, hE, hO, hG,
        )

    (
        k_fin, Q_fin, p_fin, mu_fin, _Ft_fin, _E_loop, _Omega_loop,
        _Delta_fin, grad_fin, _pen_fin, hist_E, hist_free, hist_grad,
    ) = lax.while_loop(outer_cond, outer_body, carry0)

    # ---- Finalize: rebuild the Fock at the converged (Q, p) and report.
    # Unlike the CG path we do NOT re-diagonalize and re-occupy here: at finite
    # temperature with near-degenerate occupations the orbital gradient
    # diff_p * Ft can mask sizeable off-diagonal Ft entries, so an eigh +
    # Fermi-Dirac re-occupation acts as one extra (unconverged) SCF step that
    # would move the density away from the stationary point the loop reached.
    # Reporting the loop's own (Q, p) keeps the solution self-consistent
    # (mirrors cpp_hf's solve_rtr finalize).
    P_fin, Q_fin, p_fin, _ = _project_density_eigensystem(Q_fin, p_fin, project_fn)
    Sigma_fin, H_fin, F_fin = _fock_full(P_fin)
    E_fin = hf_energy(
        P_fin, h=h, Sigma=Sigma_fin, H=H_fin, weights_b=weights_b, refP=refP,
    )
    F_fin = _herm(_project(F_fin))
    eps_check, Q_check = jnp.linalg.eigh(F_fin)
    mu_check = _solve_mu(
        eps_check.astype(real_dtype), w_norm, n_target_norm, mu_fin, T_r,
        maxiter=mu_maxiter,
    )
    p_check = jax.nn.sigmoid((mu_check - eps_check) / T_r).astype(real_dtype)
    P_check = _herm(_project(_density_from_Qp(Q_check.astype(target_dtype), p_check)))
    final_res = _norm_matrix(P_fin - P_check, w_norm)
    Ft_return = _fock_in_orbital_basis(Q_fin, F_fin)
    eps_return = jnp.real(
        jnp.diagonal(Ft_return, axis1=-2, axis2=-1)
    ).astype(real_dtype)
    mu_fin = _solve_mu(
        eps_return, w_norm, n_target_norm, mu_fin, T_r, maxiter=mu_maxiter,
    )

    converged = jnp.logical_and(
        grad_fin <= tol_g,
        jnp.where(has_deflation, jnp.bool_(True), final_res <= tol_g),
    )

    return SolveResult(
        Q=Q_fin,
        p=p_fin,
        mu=mu_fin,
        density=P_fin,
        fock=F_fin,
        energy=E_fin,
        n_iter=k_fin,
        converged=converged,
        history=dict(E=hist_E, free_energy=hist_free, grad_norm=hist_grad),
    )


_solve_newton_jitted = jax.jit(
    _solve_newton_impl,
    static_argnames=(
        "include_hartree",
        "include_exchange",
        "exchange_hermitian_channel_packing",
        "has_deflation",
        "max_iter",
        "tr_cg_max",
        "mu_maxiter",
        "block_sizes",
        "project_fn",
        "fock_build_fn",
    ),
)


def solve(
    kernel,
    P0,
    n_electrons: float,
    *,
    config: SolverConfig | None = None,
    fock_build_fn=None,
) -> SolveResult:
    """Direct-minimization Hartree-Fock solver.

    With ``config.optimizer="cg"`` (default): preconditioned Riemannian CG on
    Stiefel x capped simplex with true-free-energy backtracking (one
    current-state Fock build plus trial builds).  With
    ``config.optimizer="newton"``: trust-region Newton (Steihaug truncated-CG
    on the joint (Q, p) response Hessian), which converges on the gradient norm
    and typically needs fewer outer iterations on stiff problems.  See
    :class:`SolverConfig` for tuning knobs.

    Parameters
    ----------
    kernel
        :class:`HartreeFockKernel` holding problem arrays (pre-computed FFT
        of the interaction kernel, Hamiltonian, reference density, etc).
    P0
        Initial density matrix, shape ``(nk1, nk2, nb, nb)``.  If the seed
        is arbitrary (e.g. zeros), the solver auto-diagonalises ``F[P0]``
        internally before starting the CG loop.
    n_electrons
        Target electron count (sum of weights × occupations).
    config
        Solver controls.  Defaults to :class:`SolverConfig()`.

    Returns
    -------
    SolveResult
        Named tuple with ``Q, p, mu, density, fock, energy, n_iter,
        converged, history``.  ``density`` = ``Q @ diag(p) @ Q†`` is the
        self-consistent density matrix represented by the returned optimized
        ``Q`` and ``p``.

    Notes
    -----
    The inner loop uses Cayley retractions for orbitals and capped-simplex
    occupation steps.  The returned ``Q`` and ``p`` represent the optimized
    density directly. There is no unconditional final re-occupation; a single
    canonical fixed-point polish is used only when it reduces an already-small
    residual and the polished state is rechecked against its rebuilt Fock
    matrix.

    JIT cache strategy: only *structurally static* config fields trigger
    recompilation — loop-bound integers (``max_iter``, ``bt_max``,
    ``cg_restart``, ``mu_maxiter``), the optional ``block_sizes`` tuple,
    and ``project_fn`` (callable identity).  Float tolerances and step
    sizes (``tol_E``, ``tol_grad``, ``max_step``, ``bt_shrink``,
    ``denom_scale``) stay dynamic and can be tuned between calls without
    triggering a recompile.
    """
    if config is None:
        config = SolverConfig()
    P0_arr = jnp.asarray(P0)
    if P0_arr.shape != kernel.h.shape:
        raise ValueError(
            f"P0 must have shape {tuple(kernel.h.shape)}, got {tuple(P0_arr.shape)}."
        )
    validate_electron_count(
        kernel.w2d,
        kernel.h.shape[-1],
        n_electrons,
        context="n_electrons",
    )
    validate_temperature(kernel.T, context="kernel.T")
    if config.block_sizes is not None and sum(config.block_sizes) != kernel.h.shape[-1]:
        raise ValueError(
            f"block_sizes must sum to {kernel.h.shape[-1]}, got {sum(config.block_sizes)}."
        )

    optimizer = str(config.optimizer).lower()
    if optimizer not in ("cg", "newton"):
        raise ValueError(
            f"Unknown optimizer {config.optimizer!r}; expected 'cg' or 'newton'."
        )

    deflation_on = (
        config.deflation_targets is not None and float(config.deflation_sigma) > 0.0
    )
    if deflation_on and optimizer != "newton":
        raise ValueError(
            "deflation is only supported with optimizer='newton' "
            f"(got optimizer={config.optimizer!r})."
        )

    if optimizer == "newton":
        # Trust-region Newton is a second-order method: its Steihaug Hessian-
        # vector products go through the FFT Fock build, whose float32 round-off
        # (~1e-6) swamps the small Hessian eigenvalues on near-degenerate
        # problems and yields ascent directions.  cpp_hf sidesteps this by
        # computing in double precision throughout; warn so a float32 caller
        # knows to enable x64 rather than silently get converged=False.
        if jnp.finfo(jnp.asarray(kernel.h).real.dtype).bits < 64:
            warnings.warn(
                "optimizer='newton' needs float64 precision to converge "
                "reliably; the kernel is float32. Enable x64 via "
                "jax.config.update('jax_enable_x64', True) (and build the "
                "kernel in float64). The CG optimizer works in float32.",
                stacklevel=2,
            )
        # Trust-region Newton converges on the gradient norm; default tol_grad
        # to 1e-6 when the caller leaves it at 0 (mirrors cpp_hf's solve_rtr).
        tol_grad = float(config.tol_grad) if config.tol_grad > 0.0 else 1e-6

        # Deflation targets -> (n_solutions, nk1, nk2, nb, nb), matching the
        # kernel dtype.  An empty (0, ...) stack with has_deflation=False is the
        # no-deflation path (the bias machinery is then compiled out entirely).
        h_dtype = jnp.asarray(kernel.h).dtype
        nk1, nk2, nb = kernel.h.shape[0], kernel.h.shape[1], kernel.h.shape[-1]
        if deflation_on:
            defl_arr = jnp.asarray(config.deflation_targets, dtype=h_dtype)
            if defl_arr.ndim == 4:
                defl_arr = defl_arr[None, ...]
            if defl_arr.ndim != 5 or defl_arr.shape[1:] != (nk1, nk2, nb, nb):
                raise ValueError(
                    "deflation_targets must have shape "
                    "(n_solutions, nk1, nk2, nb, nb) matching the kernel; got "
                    f"{tuple(jnp.asarray(config.deflation_targets).shape)}."
                )
        else:
            defl_arr = jnp.zeros((0, nk1, nk2, nb, nb), dtype=h_dtype)

        return drop_unrequested(_solve_newton_jitted(
            P0_arr,
            float(n_electrons),
            tol_grad,
            float(config.tr_delta0),
            float(config.denom_scale),
            **device_args(kernel),
            deflation_targets=defl_arr,
            deflation_sigma=float(config.deflation_sigma),
            deflation_length=float(config.deflation_length),
            has_deflation=bool(deflation_on),
            max_iter=int(config.max_iter),
            tr_cg_max=int(config.tr_cg_max),
            mu_maxiter=int(config.mu_maxiter),
            block_sizes=config.block_sizes,
            project_fn=config.project_fn,
            fock_build_fn=fock_build_fn,
        ), config)

    return drop_unrequested(_solve_jitted(
        P0_arr,
        float(n_electrons),
        float(config.tol_E),
        float(config.tol_grad),
        float(config.max_step),
        float(config.bt_shrink),
        float(config.denom_scale),
        **device_args(kernel),
        max_iter=int(config.max_iter),
        bt_max=int(config.bt_max),
        cg_restart=int(config.cg_restart),
        mu_maxiter=int(config.mu_maxiter),
        plateau_window=int(config.plateau_window),
        block_sizes=config.block_sizes,
        project_fn=config.project_fn,
        fock_build_fn=fock_build_fn,
    ), config)

