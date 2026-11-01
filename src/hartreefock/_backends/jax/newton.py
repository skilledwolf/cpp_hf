"""Trust-region Newton on the joint (Q, p) response Hessian, with optional
deflation penalties."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from ...config import SolveResult
from .fock import build_fock, free_energy, hf_energy
from .linalg import adjoint, bmm, hermitian_eigh
from .occupations import (
    _active_occupations,
    _logit_clip_eps,
    _occupation_gradient,
    _project_capped_occupations,
    _project_occ_curvature,
    _project_occ_masked,
    _solve_mu,
)
from .riemannian import (
    _cayley_spectral_setup,
    _cayley_unitary_from_spectrum,
    _density_from_Qp,
    _exchange_block_specs,
    _fock_in_orbital_basis,
    _herm,
    _ip_matrix,
    _ip_vec,
    _norm_matrix,
    _project_density_eigensystem,
    _skew_hermitian,
)


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
    dP = _herm(project(bmm(bmm(Q, M), adjoint(Q))))
    St = _fock_in_orbital_basis(Q, _herm(project(response_fn(dP))))
    A = bmm(Ft, X) - bmm(X, Ft) + St
    G = diff_p * Ft
    C = (dp[..., None, :] - dp[..., :, None]) * Ft + diff_p * A
    # Parallel transport from Q exp(tX) back to Q contributes 1/2 [X,G].
    # Without this connection the operator is not self-adjoint off stationarity.
    C = C + 0.5 * (bmm(X, G) - bmm(G, X))
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
    # epsilon, i.e. ~1e-12 in float64 (the native hard-coded value) and ~5e-4 in
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
        eps, V_fd = hermitian_eigh(Ft)
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
    eps0, Q0 = hermitian_eigh(F0)
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
        Qt = bmm(Q, U)

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
        # precision: ~1e-12 in float64 (matching the native backend), ~5e-4 in float32, so
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
        eps_fd, V_fd = hermitian_eigh(Ft)
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
            p_mix, Q_mix = hermitian_eigh(P_mix)
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
    # (mirrors the native solve_rtr finalize).
    P_fin, Q_fin, p_fin, _ = _project_density_eigensystem(Q_fin, p_fin, project_fn)
    Sigma_fin, H_fin, F_fin = _fock_full(P_fin)
    E_fin = hf_energy(
        P_fin, h=h, Sigma=Sigma_fin, H=H_fin, weights_b=weights_b, refP=refP,
    )
    F_fin = _herm(_project(F_fin))
    eps_check, Q_check = hermitian_eigh(F_fin)
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
