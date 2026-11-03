"""Preconditioned Riemannian CG on Stiefel x capped simplex, and the manifold
helpers (retractions, Cayley spectral forms, inner products) it shares with
the Newton solver.

Minimizes the Hartree-Fock free energy Omega(Q, p) = E[P] - T*S(p) over
unitary orbitals Q and occupations p:

  1. Build F[P] (an exchange FFT — the dominant cost)
  2. Analytical Riemannian gradient from Fock matrix (no autodiff)
  3. Energy-gap preconditioning (diagonal Hessian)
  4. Polak-Ribiere CG direction with periodic restart
  5. Backtracking line search on the true nonlinear free energy
  6. Cayley retraction for orbitals, simplex projection for occupations
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from ..config import SolveResult
from .fock import build_fock, free_energy, hf_energy
from .linalg import adjoint, bmm, eigh, hermitian_eigh
from .occupations import (
    _occupation_gradient,
    _project_capped_occupations,
)
from .utils import chemical_potential
from .utils import hermitize as _herm

# ---------------------------------------------------------------------------
# Small linear algebra helpers
# ---------------------------------------------------------------------------


def _skew_hermitian(A: jax.Array) -> jax.Array:
    return 0.5 * (A - jnp.conj(jnp.swapaxes(A, -1, -2)))


def _density_from_Qp(Q: jax.Array, p: jax.Array) -> jax.Array:
    """P = Q diag(p) Q†."""
    return bmm(Q * p[..., None, :].astype(Q.dtype), adjoint(Q))


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
        p_new, Q_new = hermitian_eigh(P)
        return Q_new.astype(Q.dtype), p_new.astype(p.dtype)

    Q, p = lax.cond(changed, repair, lambda _: (Q, p), operand=None)
    return P, Q, p, changed


def _fock_in_orbital_basis(Q: jax.Array, F: jax.Array) -> jax.Array:
    """Ft = Q† F Q."""
    return bmm(bmm(adjoint(Q), F), Q)


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
    lam, V = hermitian_eigh(iA)
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
    return bmm(V * c[..., None, :], adjoint(V))


def _diag_UFU_from_spectrum(V: jax.Array, Ft_eig: jax.Array,
                            lam: jax.Array, tau: jax.Array) -> jax.Array:
    """Compute diag(U(τ)† Ft U(τ)) using precomputed Ft_eig = V† Ft V.

    Cost: one Hadamard (nb²) plus one matmul (nb³) — versus the LU+two-matmul
    path in :func:`_cayley_retract` followed by a full U†FtU triple product.
    """
    c = _cayley_factor(lam, tau, V.dtype)
    cbar = jnp.conj(c)
    M = (cbar[..., :, None]) * Ft_eig * (c[..., None, :])
    A = bmm(V, M)
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
    sectors: tuple | None,
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
    ``cg_restart``, ``mu_maxiter``, ``sectors``, ``project_fn``, plus the
    Hartree/exchange/hcp flags from the kernel) are marked as static in the
    JIT decorator.  Float tolerances and step sizes stay dynamic so that
    tuning them does not trigger recompilation.
    """
    target_dtype = h.dtype
    real_dtype = jnp.zeros((), dtype=target_dtype).real.dtype
    tiny = jnp.asarray(1e-30, dtype=real_dtype)

    _project = project_fn if project_fn is not None else (lambda A: A)
    _fock_fn = fock_build_fn if fock_build_fn is not None else build_fock

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
        sectors=sectors,
        project_fn=project_fn,
    )
    eps0, Q0 = eigh(F0, sectors)
    eps0 = eps0.astype(real_dtype)
    Q0 = Q0.astype(target_dtype)
    mu0 = chemical_potential(eps0, w_norm, n_target_norm, T_r, mu0=jnp.asarray(0.0, dtype=real_dtype), maxiter=mu_maxiter)
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
        # Stopping rule:
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
            sectors=sectors,
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
        eps_fd, Q_fd = eigh(F, sectors)
        mu_fd = chemical_potential(eps_fd.astype(real_dtype), w_norm, n_target_norm, T_r, mu0=mu, maxiter=mu_maxiter)
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
        Ft_eig = bmm(bmm(adjoint(V_d), Ft), V_d)

        d_Q_norm = _norm_matrix(d_Q, w_norm)
        tau0 = jnp.minimum(1.0, max_step_r / jnp.maximum(d_Q_norm, tiny))

        def trial_free_energy(tau_val):
            U_trial = _cayley_unitary_from_spectrum(V_d, lam_d, tau_val)
            Q_trial = bmm(Q, U_trial)
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
                sectors=sectors, project_fn=None,
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
            Q_new = bmm(Q, U)
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
        mu_new = chemical_potential(eps_new, w_norm, n_target_norm, T_r, mu0=mu, maxiter=mu_maxiter)

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
            p_mix, Q_mix = eigh(P_mix, sectors)
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
                sectors=sectors, project_fn=None,
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
        mu_fb = chemical_potential(eps_fb, w_norm, n_target_norm, T_r, mu0=mu, maxiter=mu_maxiter)
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
        sectors=sectors,
        project_fn=project_fn,
    )
    E_fin = hf_energy(
        P_fin, h=h, Sigma=Sigma_fin, H=H_fin, weights_b=weights_b, refP=refP,
    )

    # Recheck the full self-consistency residual at the exact returned state.
    eps_check, Q_check = eigh(F_fin, sectors)
    mu_check = chemical_potential(eps_check.astype(real_dtype), w_norm, n_target_norm, T_r, mu0=mu_fin, maxiter=mu_maxiter)
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
            sectors=sectors, project_fn=None,
        )
        eps_next, Q_next = eigh(F_polish, sectors)
        mu_next = chemical_potential(eps_next.astype(real_dtype), w_norm, n_target_norm, T_r, mu0=mu_check, maxiter=mu_maxiter)
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
    mu_fin = chemical_potential(eps_return, w_norm, n_target_norm, T_r, mu0=mu_fin, maxiter=mu_maxiter)

    # Convergence flag.  Gradient-primary when tol_grad > 0: grad_fin is the
    # last measured gradient — the entry the loop stopped on, or the final
    # recorded one when max_iter ran out (the plain k_fin < max_iter test
    # would miss it).
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
        # Config fields that are structurally static (integers used as
        # loop bounds, callables, or tuples affecting graph shape)
        "max_iter",
        "bt_max",
        "cg_restart",
        "mu_maxiter",
        "plateau_window",
        "sectors",
        "project_fn",
        # Optional Fock build override (e.g. superlattice streaming Fock).
        # Captured by identity, so passing the same closure is cache-friendly.
        "fock_build_fn",
    ),
)
