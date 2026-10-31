// Trust-region Newton solver for HF direct minimization.
//
// Minimises the same free energy Ω(Q,p)=E[P]-T·S(p) as solve_dm, but with a
// second-order step: the joint (Q,p) Hessian-vector product uses the EXACT
// linear interaction response Σ[δP] = F[δP]-h (one Fock build each), and a
// Steihaug truncated-CG solves the trust-region subproblem.
#pragma once

#include "cpp_hf/fock.hpp"
#include "cpp_hf/kernel.hpp"
#include "cpp_hf/linalg.hpp"
#include "cpp_hf/solver_dm.hpp"
#include "cpp_hf/utils.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace cpp_hf {

namespace rtr_internal {

// Interaction response Σ[δP] = F[δP] − h, treating δP as a pure density
// variation (no reference density).  Σ is linear in P so this is exact.
inline void fock_response(const HFKernel& K, const c64* dP, c64* resp,
                          c64* Sigma_buf, c64* F_buf, f32* hartree_buf,
                          std::size_t n_tot) {
    HFKernel Kr = K;
    Kr.has_refP = false;  // no refP subtraction → linear response of δP
    build_fock_compact(Kr, dP, Sigma_buf, F_buf, hartree_buf, nullptr, false);
    std::memcpy(resp, F_buf, n_tot * sizeof(c64));
}

// Joint (Q,p) Hessian-vector product.  Inputs X (skew-Herm off-diagonal,
// orbital basis), dp (real per-(k,b)); outputs HX (skew-Herm off-diagonal) and
// Hp (real, particle-conserving).  One Fock build via fock_response.
inline void joint_hvp(const HFKernel& K, const c64* X, const f32* dp,
                      const c64* Q, const f32* p, const c64* Ft, f32 T_r,
                      const f32* w2d, std::size_t nk, std::size_t nb,
                      c64* HX, f32* Hp, c64* dP_buf, c64* resp_buf,
                      c64* Sigma_buf, c64* F_buf, f32* hartree_buf,
                      f32 pen_diag_coeff = 0.0,
                      const c64* penalty_density = nullptr,
                      const c64* penalty_targets = nullptr,
                      std::size_t n_penalty = 0, f32 penalty_sigma = 0.0,
                      f32 penalty_length = 1.0,
                      const ProjectFn* project_fn = nullptr) {
    const std::size_t nb2 = nb * nb;
    // 1. Build the combined density variation δP = Q ([X,diag p] + diag(dp)) Q†.
    parallel_for(nk, [&](std::size_t k) {
        ConstMapMatXcf Xk(X + k * nb2, nb, nb);
        ConstMapMatXcf Qk(Q + k * nb2, nb, nb);
        const f32* pk = p + k * nb;
        const f32* dpk = dp + k * nb;
        MatXcf M(nb, nb);
        for (std::size_t i = 0; i < nb; ++i)
            for (std::size_t j = 0; j < nb; ++j)
                M(i, j) = (i == j) ? c64(dpk[i], 0.0)
                                   : Xk(i, j) * static_cast<f32>(pk[j] - pk[i]);
        MatXcf dPk = Qk * M * Qk.adjoint();
        dPk = (0.5 * (dPk + dPk.adjoint())).eval();  // exact Hermitisation
        std::memcpy(dP_buf + k * nb2, dPk.data(), nb2 * sizeof(c64));
    });
    if (project_fn && *project_fn) (*project_fn)(dP_buf, K.nk1, K.nk2, nb);
    // 2. One Fock build: the interaction response.
    fock_response(K, dP_buf, resp_buf, Sigma_buf, F_buf, hartree_buf, nk * nb2);
    // 2b. Deflation penalty curvature (diagonal part): add pen_diag_coeff * δP to
    // the self-energy response (Fock-free).  pen_diag_coeff = 2·Σ_i φ'(d_i²) < 0
    // near a deflated basin, injecting the negative curvature that pushes the
    // Newton step out of it.  Strict no-op when 0.
    if (pen_diag_coeff != 0.0) {
        const c64 c(static_cast<double>(pen_diag_coeff), 0.0);
        for (std::size_t i = 0; i < nk * nb2; ++i) resp_buf[i] += c * dP_buf[i];
    }
    if (penalty_density && penalty_targets && n_penalty > 0) {
        const std::size_t n = nk*nb2;
        const double invL2 = 1.0 / (penalty_length*penalty_length);
        for (std::size_t target = 0; target < n_penalty; ++target) {
            const c64* base = penalty_targets+target*n;
            double distance2 = 0.0, directional = 0.0;
            for (std::size_t k = 0; k < nk; ++k)
                for (std::size_t i = 0; i < nb2; ++i) {
                    const std::size_t a = k*nb2+i;
                    const c64 difference = penalty_density[a]-base[a];
                    distance2 += w2d[k]*std::norm(difference);
                    directional += w2d[k]*(std::conj(difference)*dP_buf[a]).real();
                }
            const double coefficient = penalty_sigma*std::exp(-0.5*distance2*invL2)
                                       *invL2*invL2*directional;
            for (std::size_t i = 0; i < n; ++i)
                resp_buf[i] += coefficient*(penalty_density[i]-base[i]);
        }
    }
    if (project_fn && *project_fn) (*project_fn)(resp_buf, K.nk1, K.nk2, nb);
    // 3. Assemble HX and Hp per k.
    parallel_for(nk, [&](std::size_t k) {
        ConstMapMatXcf Xk(X + k * nb2, nb, nb);
        ConstMapMatXcf Qk(Q + k * nb2, nb, nb);
        ConstMapMatXcf Ftk(Ft + k * nb2, nb, nb);
        ConstMapMatXcf Rk(resp_buf + k * nb2, nb, nb);
        const f32* pk = p + k * nb;
        const f32* dpk = dp + k * nb;
        MatXcf St = Qk.adjoint() * Rk * Qk;       // response in orbital basis
        MatXcf A = Ftk * Xk - Xk * Ftk + St;       // [Ft,X] + St
        MatXcf C(nb, nb);
        for (std::size_t i = 0; i < nb; ++i)
            for (std::size_t j = 0; j < nb; ++j)
                C(i, j) = static_cast<f32>(dpk[j] - dpk[i]) * Ftk(i, j)
                          + static_cast<f32>(pk[j] - pk[i]) * A(i, j);
        // Covariant derivative in left-trivialized unitary coordinates.
        // Differentiating the body gradient alone omits the connection and
        // gives a nonsymmetric operator away from stationarity.  Parallel
        // transport back from Q exp(tX) contributes 1/2 [X,G].
        MatXcf Gk(nb, nb);
        for (std::size_t i = 0; i < nb; ++i)
            for (std::size_t j = 0; j < nb; ++j)
                Gk(i, j) = (pk[j] - pk[i]) * Ftk(i, j);
        C += 0.5 * (Xk * Gk - Gk * Xk);
        MatXcf HXk = 0.5 * (C - C.adjoint());
        for (std::size_t d = 0; d < nb; ++d) HXk(d, d) = c64(0.0, 0.0);
        std::memcpy(HX + k * nb2, HXk.data(), nb2 * sizeof(c64));
        for (std::size_t b = 0; b < nb; ++b) {
            const f32 pc = std::clamp(pk[b], LOGIT_CLIP, 1.0 - LOGIT_CLIP);
            Hp[k * nb + b] = static_cast<f32>(A(b, b).real())
                             + T_r / (pc * (1.0 - pc)) * dpk[b];
        }
    });
    // Saturated occupations are active bound constraints: project the Hessian
    // action onto the particle-conserving subspace of the resolvable bands
    // only, so the Steihaug Krylov space never leaves it (the residual and the
    // preconditioner are masked the same way; see dm_internal::occupation_gradient).
    std::vector<unsigned char> active(nk * nb);
    dm_internal::active_occupations(p, active.data(), nk * nb);
    dm_internal::project_occupation_tangent_masked(Hp, w2d, active.data(), nk, nb);
}

// Deflation bias: a Gaussian repulsion hill around each already-found density.
//   Φ        = σ · Σ_i exp(-d_i²/(2L²)),   d_i² = Σ_k w_k ‖P_k - P*_{i,k}‖_F²
//   S_pen,k  = (∂Φ/∂P_k)/w_k = 2 Σ_i φ'(d_i²) (P_k - P*_{i,k})   (per-k, no w2d —
//              mirrors the role of the physical Fock F so F_eff = F + S_pen folds
//              the bias into the gradient with the energy's w2d applied outside)
//   diag     = 2 Σ_i φ'(d_i²)  (the local Hessian-response scalar for joint_hvp)
// with φ(x)=σe^{-x/(2L²)}, φ'(x)=-φ(x)/(2L²) (<0 ⇒ force points away from P*).
// Returns Φ; writes S_pen (n_tot) and diag_coeff.  Strict no-op (Φ=0, S_pen=0,
// diag_coeff=0) when n_defl==0 or σ==0.
inline f32 deflation_bias(const c64* P, const c64* targets, std::size_t n_defl,
                          const f32* w2d, std::size_t nk, std::size_t nb,
                          f32 sigma, f32 length, c64* S_pen, f32& diag_coeff) {
    const std::size_t nb2 = nb * nb;
    const std::size_t n_tot = nk * nb2;
    std::fill(S_pen, S_pen + n_tot, c64(0.0, 0.0));
    diag_coeff = 0.0;
    if (n_defl == 0 || sigma == 0.0 || targets == nullptr) return 0.0;
    const double inv2L2 =
        1.0 / (2.0 * static_cast<double>(length) * static_cast<double>(length));
    double Phi = 0.0, diag = 0.0;
    for (std::size_t i = 0; i < n_defl; ++i) {
        const c64* Ti = targets + i * n_tot;
        double d2 = 0.0;
        for (std::size_t k = 0; k < nk; ++k) {
            const c64* pk = P + k * nb2;
            const c64* tk = Ti + k * nb2;
            double per_k = 0.0;
            for (std::size_t a = 0; a < nb2; ++a) per_k += std::norm(pk[a] - tk[a]);
            d2 += static_cast<double>(w2d[k]) * per_k;
        }
        const double phi = static_cast<double>(sigma) * std::exp(-d2 * inv2L2);
        const double phip = -phi * inv2L2;            // φ'(d²) < 0
        Phi += phi;
        diag += phip;
        const c64 coeff(2.0 * phip, 0.0);             // 2 φ'(d²)
        for (std::size_t k = 0; k < nk; ++k) {
            const c64* pk = P + k * nb2;
            const c64* tk = Ti + k * nb2;
            c64* sk = S_pen + k * nb2;
            for (std::size_t a = 0; a < nb2; ++a) sk[a] += coeff * (pk[a] - tk[a]);
        }
    }
    diag_coeff = static_cast<f32>(2.0 * diag);         // 2 Σ_i φ'(d_i²)
    return static_cast<f32>(Phi);
}

}  // namespace rtr_internal

inline DMResult solve_rtr(const HFKernel& K, const c64* P0, f32 n_e,
                          const SolverConfig& cfg,
                          const ProjectFn* project_fn = nullptr) {
    using namespace dm_internal;
    using namespace rtr_internal;

    const std::size_t nk1 = K.nk1, nk2 = K.nk2, nb = K.nb;
    const std::size_t nk = nk1 * nk2, nb2 = nb * nb, n_tot = nk * nb2;
    const std::size_t npb = nk * nb;
    const f32 T_r = std::max(K.T, 1.0e-12);
    const f32 tol_g = (cfg.tol_grad > 0.0) ? cfg.tol_grad : 1.0e-6;

    std::vector<f32> w_norm(nk);
    const f32 ws = std::max(K.weight_sum, TINY_REAL);
    for (std::size_t i = 0; i < nk; ++i) w_norm[i] = K.w2d[i] / ws;
    const f32 n_target_norm = n_e / ws;

    // State + workspaces.
    std::vector<c64> Q(n_tot), P_cur(n_tot), Sigma(n_tot), F(n_tot), Ft(n_tot);
    std::vector<f32> p(npb), eps(npb), hartree_diag(nb, 0.0);
    std::vector<c64> G(n_tot);                       // orbital gradient
    std::vector<f32> g_p(npb), h_p(npb);            // occupation gradient (+ preconditioned form)
    std::vector<f32> curv_p(npb);                    // FD curvature p(1-p)/T, 0 on saturated bands
    std::vector<unsigned char> active_p(npb);        // resolvable-occupation mask
    std::vector<c64> vX(n_tot), dPb(n_tot), respb(n_tot), Fb(n_tot), Sb(n_tot);
    std::vector<f32> vp(npb), hb(nb, 0.0);
    std::vector<c64> rX(n_tot), dX(n_tot), zX(n_tot), HvX(n_tot), HDX(n_tot);
    std::vector<f32> rp(npb), dp(npb), zp(npb), Hvp(npb), HDp(npb);
    std::vector<c64> Vd(n_tot), Qt(n_tot), Ub(n_tot);
    std::vector<f32> lamd(npb);
    std::vector<c64> Gsave(n_tot), Ftsave(n_tot);
    std::vector<f32> epssave(npb), eps_t(npb), p_trial(npb);
    std::vector<c64> Vfd(n_tot), Qfd(n_tot), Pfd(n_tot), Pfd_save(n_tot);
    std::vector<f32> eps_fd(npb), p_fd(npb);

    // Deflation (Newton path only): repulsive Gaussian bias around found densities.
    const std::size_t n_defl = cfg.n_deflation;
    const c64* defl_targets =
        cfg.deflation_targets.empty() ? nullptr : cfg.deflation_targets.data();
    const f32 defl_sigma = cfg.deflation_sigma;
    const f32 defl_length = std::max(cfg.deflation_length, static_cast<f32>(1.0e-12));
    const bool deflation_on =
        (n_defl > 0 && defl_sigma > 0.0 && defl_targets != nullptr);
    if (deflation_on && cfg.deflation_targets.size() < n_defl * n_tot)
        throw std::invalid_argument(
            "deflation_targets buffer smaller than n_deflation * nk * nb * nb.");
    std::vector<c64> S_pen(deflation_on ? n_tot : 0);
    std::vector<c64> penalty_base(deflation_on ? n_tot : 0);
    f32 pen_diag_coeff = 0.0;  // 2·Σ_i φ'(d_i²) at the current density (frozen per outer)

    DMResult out;
    out.hist_E.assign(cfg.max_iter, 0.0);
    out.hist_free.assign(cfg.max_iter, 0.0);
    out.hist_grad.assign(cfg.max_iter, 0.0);
    std::size_t n_builds = 0;  // Fock-build counter (env-gated diagnostic)

    // The same normalized BZ metric in BOTH sectors.  The body gradient
    // and Hessian below represent Omega / weight_sum in this metric.  Mixing
    // an unweighted orbital sector with weighted occupations destroys Hessian
    // self-adjointness, even for uniform weights different from one.
    auto jip = [&](const c64* aX, const f32* ap, const c64* bX, const f32* bp) {
        return ip_matrix(aX, bX, w_norm.data(), nk, nb)
               + ip_vec(ap, bp, w_norm.data(), nk, nb);
    };
    auto jfro = [&](const c64* aX, const f32* ap) {
        return std::sqrt(std::max(0.0, jip(aX, ap, aX, ap)));
    };
    // Block preconditioner: orbital 1/√(gap²+λ²); occupation p(1-p)/T on the
    // resolvable bands (zero on saturated ones), projected onto particle
    // conservation under the 1/curv metric.
    auto japply_prec = [&](const c64* aX, const f32* ap, c64* oX, f32* op, f32 lam) {
        for (std::size_t k = 0; k < nk; ++k) {
            const f32* eb = eps.data() + k * nb;
            for (std::size_t i = 0; i < nb; ++i)
                for (std::size_t j = 0; j < nb; ++j) {
                    const f32 gap = eb[i] - eb[j];
                    oX[k * nb2 + i * nb + j] =
                        aX[k * nb2 + i * nb + j] / std::sqrt(gap * gap + lam * lam);
                }
        }
        project_occupation_curvature(ap, K.w2d, curv_p.data(), op, nk, nb);
    };
    // Symmetry constraints require an orthogonal, trace-preserving projection
    // onto a matrix algebra. Project the Fock gradient as well as the density;
    // merely deleting density coherences would optimize the wrong entropy.
    auto project_in_place = [&](c64* M) {
        if (project_fn && *project_fn) (*project_fn)(M, nk1, nk2, nb);
    };

    // ---- initialise (Q, p) from F[P0] ----
    std::memcpy(P_cur.data(), P0, n_tot * sizeof(c64));
    project_in_place(P_cur.data());
    hermitize_inplace(P_cur.data(), nk, nb);
    build_fock_compact(K, P_cur.data(), Sigma.data(), F.data(), hartree_diag.data(), project_fn);
    ++n_builds;
    eigh_checked(F.data(), eps.data(), Q.data(), nk1, nk2, nb, cfg.block_sizes);
    {
        const f32 mu0 = solve_mu_inloop(eps.data(), npb, w_norm.data(), nk, nb,
                                        n_target_norm, 0.0, T_r, cfg.mu_maxiter);
        for (std::size_t i = 0; i < npb; ++i) p[i] = expit_scalar((mu0 - eps[i]) / T_r);
        out.mu = mu0;
    }
    if (cfg.preserve_initial_density) {
        eigh_checked(P_cur.data(), p.data(), Q.data(), nk1, nk2, nb, cfg.block_sizes);
        project_capped_occupations(p.data(), w_norm.data(), nk, nb, n_target_norm);
    }

    auto eval_state = [&](c64* Qs, f32* ps,
                          f32& E_out, f32& Omega_out, f32& grad_out) {
        density_from_Qp(Qs, ps, P_cur.data(), nk, nb, cfg.block_sizes);
        project_density_eigensystem(
            P_cur.data(), Qs, ps, Sigma.data(), project_fn,
            nk1, nk2, nb, cfg.block_sizes);
        build_fock_compact(K, P_cur.data(), Sigma.data(), F.data(), hartree_diag.data(), project_fn);
        ++n_builds;
        E_out = hf_energy_with_hartree_diag(K, P_cur.data(), Sigma.data(), hartree_diag.data());
        Omega_out = free_energy_value(E_out, ps, K.w2d, nk, nb, T_r);
        if (deflation_on) {
            const f32 Phi = deflation_bias(P_cur.data(), defl_targets, n_defl,
                                           K.w2d, nk, nb, defl_sigma, defl_length,
                                           S_pen.data(), pen_diag_coeff);
            Omega_out += Phi;                          // biased free energy Ω + Φ
            for (std::size_t i = 0; i < n_tot; ++i)
                F[i] += S_pen[i];                      // F_eff = F + S_pen
        }
        project_in_place(F.data());
        fock_in_orbital_basis(Qs, F.data(), Ft.data(), nk, nb, cfg.block_sizes);
        for (std::size_t k = 0; k < nk; ++k)
            for (std::size_t b = 0; b < nb; ++b)
                eps[k * nb + b] = Ft[k * nb2 + b * nb + b].real();
        compute_orbital_gradient(Ft.data(), ps, G.data(), nk, nb);

        // Full FD fixed point of the effective Fock (physical Fock plus
        // deflation force when active).  This catches non-stationary
        // occupations even when the orbital gradient vanishes identically.
        eigh_checked(Ft.data(), eps_fd.data(), Vfd.data(), nk1, nk2, nb, cfg.block_sizes);
        const f32 mu_fd = solve_mu_inloop(
            eps_fd.data(), npb, w_norm.data(), nk, nb,
            n_target_norm, 0.0, T_r, cfg.mu_maxiter);
        for (std::size_t i = 0; i < npb; ++i)
            p_fd[i] = expit_scalar((mu_fd - eps_fd[i]) / T_r);
        Q_times_U(Qs, Vfd.data(), Qfd.data(), nk, nb);
        density_from_Qp(Qfd.data(), p_fd.data(), Pfd.data(), nk, nb, cfg.block_sizes);
        project_in_place(Pfd.data());
        hermitize_inplace(Pfd.data(), nk, nb);
        grad_out = std::max(
            norm_matrix(G.data(), w_norm.data(), nk, nb),
            norm_matrix_difference(
                P_cur.data(), Pfd.data(), w_norm.data(), nk, nb));
    };

    f32 E = 0.0, Omega = 0.0;
    f32 grad_norm = std::numeric_limits<f32>::infinity();
    eval_state(Q.data(), p.data(), E, Omega, grad_norm);
    f32 mu = out.mu, Delta = cfg.tr_delta0;
    std::size_t k_outer = 0;
    bool converged = false;

    while (k_outer < cfg.max_iter) {
        out.hist_E[k_outer] = E;
        out.hist_free[k_outer] = Omega;
        out.hist_grad[k_outer] = grad_norm;
        if (grad_norm <= tol_g) { converged = true; ++k_outer; break; }

        // Occupation gradient on the fixed-particle tangent space with
        // saturated bands masked out, plus the Fermi-Dirac curvature the
        // occupation preconditioner uses (see dm_internal::occupation_gradient).
        occupation_gradient(eps.data(), p.data(), T_r, K.w2d, nk, nb,
                            g_p.data(), h_p.data(), curv_p.data(),
                            active_p.data());

        double eps_sq = 0.0;
        for (std::size_t i = 0; i < npb; ++i) eps_sq += double(eps[i]) * eps[i];
        const f32 lam_pc = std::max(T_r, cfg.denom_scale *
            static_cast<f32>(std::sqrt(eps_sq / std::max<std::size_t>(npb, 1) + TINY_REAL)));

        if (deflation_on) {
            density_from_Qp(Q.data(), p.data(), penalty_base.data(), nk, nb, cfg.block_sizes);
            project_in_place(penalty_base.data());
        }

        // ---- Steihaug truncated CG: minimise the joint quadratic model ----
        std::fill(vX.begin(), vX.end(), c64(0.0, 0.0)); std::fill(vp.begin(), vp.end(), 0.0);
        std::fill(HvX.begin(), HvX.end(), c64(0.0, 0.0)); std::fill(Hvp.begin(), Hvp.end(), 0.0);
        for (std::size_t i = 0; i < n_tot; ++i) rX[i] = -G[i];
        for (std::size_t i = 0; i < npb; ++i) rp[i] = -g_p[i];
        japply_prec(rX.data(), rp.data(), zX.data(), zp.data(), lam_pc);
        std::memcpy(dX.data(), zX.data(), n_tot * sizeof(c64));
        std::memcpy(dp.data(), zp.data(), npb * sizeof(f32));
        f32 rz = jip(rX.data(), rp.data(), zX.data(), zp.data());
        const f32 residual0 = jfro(rX.data(), rp.data());

        for (std::size_t cg = 0; cg < cfg.tr_cg_max; ++cg) {
            joint_hvp(K, dX.data(), dp.data(), Q.data(), p.data(), Ft.data(), T_r,
                      K.w2d, nk, nb, HDX.data(), HDp.data(), dPb.data(), respb.data(),
                      Sb.data(), Fb.data(), hb.data(), pen_diag_coeff,
                      deflation_on ? penalty_base.data() : nullptr, defl_targets,
                      n_defl, defl_sigma, defl_length, project_fn);
            ++n_builds;
            const f32 dHd = jip(dX.data(), dp.data(), HDX.data(), HDp.data());
            const f32 dd = jip(dX.data(), dp.data(), dX.data(), dp.data());
            auto to_boundary = [&]() {
                // solve ‖v + t·d‖² = Δ² for the positive root.
                const f32 a = dd;
                const f32 b = 2.0 * jip(vX.data(), vp.data(), dX.data(), dp.data());
                const f32 c = jfro(vX.data(), vp.data());
                const f32 cc = c * c - Delta * Delta;
                const f32 t = (-b + std::sqrt(std::max(b * b - 4.0 * a * cc, 0.0)))
                              / (2.0 * a + 1e-30);
                for (std::size_t i = 0; i < n_tot; ++i) { vX[i] += t * dX[i]; HvX[i] += t * HDX[i]; }
                for (std::size_t i = 0; i < npb; ++i) { vp[i] += t * dp[i]; Hvp[i] += t * HDp[i]; }
            };
            if (dHd <= 1e-14 * dd) { to_boundary(); break; }  // negative curvature
            const f32 alpha = rz / dHd;
            // Joint norm in the same metric used by the Krylov solve.
            double s2 = 0.0;
            for (std::size_t kk = 0; kk < nk; ++kk)
                for (std::size_t i = 0; i < nb2; ++i)
                    s2 += w_norm[kk] * std::norm(
                        vX[kk * nb2 + i] + alpha * dX[kk * nb2 + i]);
            for (std::size_t kk = 0; kk < nk; ++kk) {
                double per = 0.0;
                for (std::size_t b = 0; b < nb; ++b) {
                    const f32 e = vp[kk * nb + b] + alpha * dp[kk * nb + b];
                    per += double(e) * e;
                }
                s2 += double(w_norm[kk]) * per;
            }
            if (std::sqrt(std::max(0.0, s2)) >= Delta) { to_boundary(); break; }
            for (std::size_t i = 0; i < n_tot; ++i) { vX[i] += alpha * dX[i]; HvX[i] += alpha * HDX[i]; rX[i] -= alpha * HDX[i]; }
            for (std::size_t i = 0; i < npb; ++i) { vp[i] += alpha * dp[i]; Hvp[i] += alpha * HDp[i]; rp[i] -= alpha * HDp[i]; }
            const f32 rn = std::sqrt(std::max(0.0, jip(rX.data(), rp.data(), rX.data(), rp.data())));
            if (rn < std::min(0.5, std::sqrt(residual0)) * residual0) break;
            japply_prec(rX.data(), rp.data(), zX.data(), zp.data(), lam_pc);
            const f32 rz_new = jip(rX.data(), rp.data(), zX.data(), zp.data());
            const f32 beta = rz_new / (
                std::abs(rz) > TINY_REAL ? rz : static_cast<f32>(1.0));
            for (std::size_t i = 0; i < n_tot; ++i) dX[i] = zX[i] + beta * dX[i];
            for (std::size_t i = 0; i < npb; ++i) dp[i] = zp[i] + beta * dp[i];
            rz = rz_new;
        }

        if (Delta < 1e-10) break;

        // Preserve the computed joint step.  A single fraction-to-boundary
        // factor keeps occupations feasible and scales the orbital sector and
        // its already computed Hessian action by the SAME amount.  It cannot
        // enlarge the trust-region step or violate particle conservation.
        f32 fraction = 1.0;
        for (std::size_t i = 0; i < npb; ++i) {
            if (vp[i] < 0.0) fraction = std::min(fraction, -0.99 * p[i] / vp[i]);
            if (vp[i] > 0.0) fraction = std::min(fraction, 0.99 * (1.0 - p[i]) / vp[i]);
        }
        for (std::size_t i = 0; i < n_tot; ++i) {
            vX[i] *= fraction;
            HvX[i] *= fraction;
        }
        for (std::size_t i = 0; i < npb; ++i) {
            vp[i] *= fraction;
            Hvp[i] *= fraction;
            p_trial[i] = p[i] + vp[i];
        }
        // Cayley(tau,X) = I - tau X + O(tau^2); tau=-1 moves along +vX.
        cayley_spectral_setup(vX.data(), Vd.data(), lamd.data(), nk, nb);
        cayley_unitary_from_spectrum(Vd.data(), lamd.data(), -1.0, Ub.data(), nk, nb);
        Q_times_U(Q.data(), Ub.data(), Qt.data(), nk, nb);

        // Predicted reduction m(0)-m(v) for the same applied joint step, in
        // true thermodynamic units.  Both linear sectors and both Hessian
        // sectors carry the raw BZ measure, matching Omega=E-T*S.
        const f32 linear =
            ip_matrix(G.data(), vX.data(), K.w2d, nk, nb)
            + ip_vec(g_p.data(), vp.data(), K.w2d, nk, nb);
        const f32 quadratic =
            ip_matrix(vX.data(), HvX.data(), K.w2d, nk, nb)
            + ip_vec(vp.data(), Hvp.data(), K.w2d, nk, nb);
        const f32 pred = -(linear + 0.5 * quadratic);

        // Save current state-dependent workspaces for a rejected trial.
        std::memcpy(Gsave.data(), G.data(), n_tot * sizeof(c64));
        std::memcpy(Ftsave.data(), Ft.data(), n_tot * sizeof(c64));
        std::memcpy(epssave.data(), eps.data(), npb * sizeof(f32));
        std::memcpy(Pfd_save.data(), Pfd.data(), n_tot * sizeof(c64));
        const f32 pen_diag_save = pen_diag_coeff;  // eval_state(trial) overwrites it

        // True free energy + full stationarity at the trial point.
        f32 E_trial = 0.0, Omega_trial = 0.0;
        f32 grad_trial = std::numeric_limits<f32>::infinity();
        eval_state(
            Qt.data(), p_trial.data(), E_trial, Omega_trial, grad_trial);

        const f32 noise = 1e-12 * std::max(std::abs(Omega), 1.0);
        bool accept;
        if (pred <= noise) {
            // Below the energy noise floor, judge by stationarity but never
            // accept a true free-energy increase.
            accept = (grad_trial < grad_norm)
                  && (Omega_trial <= Omega + noise);
            if (!accept) Delta *= 0.5;
        } else {
            const f32 ratio = (Omega - Omega_trial) / pred;
            if (ratio < 0.25) Delta *= 0.25;
            else if (ratio > 0.75 && jfro(vX.data(), vp.data()) > 0.9 * Delta)
                Delta = std::min(2.0 * Delta, 5.0);
            accept = (ratio > 0.1);
        }

        if (std::getenv("CPP_HF_DM_DEBUG")) {
            std::fprintf(stderr,
                "[rtr] k=%zu g=%.3e |v|=%.3e Delta=%.3e pred=%.3e dO=%+.3e gtr=%.3e accept=%d branch=%s\n",
                k_outer, grad_norm, jfro(vX.data(), vp.data()), Delta, pred,
                (Omega_trial - Omega), grad_trial, static_cast<int>(accept),
                (pred <= noise ? "grad" : "ratio"));
        }

        // A local joint model may stall when nearly equal occupations suppress
        // the orbital parameterization.  On rejection, backtrack along the
        // full FD density fixed-point direction and accept only true
        // free-energy descent (or noise-level equality with lower residual).
        f32 mu_accept = mu;
        if (!accept) {
            f32 lam_fb = 1.0;
            bool fallback_accepted = false;
            for (std::size_t fb = 0; fb < 10; ++fb) {
                density_from_Qp(Q.data(), p.data(), P_cur.data(), nk, nb, cfg.block_sizes);
                project_in_place(P_cur.data());
                hermitize_inplace(P_cur.data(), nk, nb);
                for (std::size_t i = 0; i < n_tot; ++i)
                    P_cur[i] += c64(lam_fb, 0.0)
                              * (Pfd_save[i] - P_cur[i]);
                project_in_place(P_cur.data());
                hermitize_inplace(P_cur.data(), nk, nb);

                eigh_checked(
                    P_cur.data(), p_trial.data(), Qt.data(), nk1, nk2, nb, cfg.block_sizes);
                project_capped_occupations(
                    p_trial.data(), K.w2d, nk, nb, n_e);
                eval_state(
                    Qt.data(), p_trial.data(),
                    E_trial, Omega_trial, grad_trial);
                const f32 dOmega = Omega_trial - Omega;
                fallback_accepted =
                    dOmega < 0.0
                    || (std::abs(dOmega) <= noise
                        && grad_trial < grad_norm
                        && Omega_trial <= Omega + noise);
                if (fallback_accepted) {
                    mu_accept = solve_mu_inloop(
                        eps.data(), npb, w_norm.data(), nk, nb,
                        n_target_norm, mu, T_r, cfg.mu_maxiter);
                    Delta = std::max(Delta, static_cast<f32>(0.1));
                    accept = true;
                    break;
                }
                lam_fb *= 0.5;
            }
        }

        if (accept) {
            std::memcpy(Q.data(), Qt.data(), n_tot * sizeof(c64));
            std::memcpy(p.data(), p_trial.data(), npb * sizeof(f32));
            E = E_trial;
            Omega = Omega_trial;
            grad_norm = grad_trial;
            mu = mu_accept;  // G/Ft/eps already match the accepted state
        } else {
            std::memcpy(G.data(), Gsave.data(), n_tot * sizeof(c64));
            std::memcpy(Ft.data(), Ftsave.data(), n_tot * sizeof(c64));
            std::memcpy(eps.data(), epssave.data(), npb * sizeof(f32));
            std::memcpy(Pfd.data(), Pfd_save.data(), n_tot * sizeof(c64));
            pen_diag_coeff = pen_diag_save;
        }
        ++k_outer;
    }

    // ---- Finalize without changing the optimized state ----
    density_from_Qp(Q.data(), p.data(), P_cur.data(), nk, nb, cfg.block_sizes);
    project_density_eigensystem(
        P_cur.data(), Q.data(), p.data(), Sigma.data(), project_fn,
        nk1, nk2, nb, cfg.block_sizes);
    build_fock_compact(K, P_cur.data(), Sigma.data(), F.data(), hartree_diag.data(), project_fn);
    ++n_builds;
    const f32 E_fin = hf_energy_with_hartree_diag(K, P_cur.data(), Sigma.data(), hartree_diag.data());

    // Recheck the full physical FD fixed point at the exact returned density.
    eigh_checked(F.data(), eps_fd.data(), Vfd.data(), nk1, nk2, nb, cfg.block_sizes);
    const f32 mu_check = solve_mu_inloop(
        eps_fd.data(), npb, w_norm.data(), nk, nb,
        n_target_norm, mu, T_r, cfg.mu_maxiter);
    for (std::size_t i = 0; i < npb; ++i)
        p_fd[i] = expit_scalar((mu_check - eps_fd[i]) / T_r);
    density_from_Qp(Vfd.data(), p_fd.data(), Pfd.data(), nk, nb, cfg.block_sizes);
    project_in_place(Pfd.data());
    hermitize_inplace(Pfd.data(), nk, nb);
    const f32 final_res = norm_matrix_difference(
        P_cur.data(), Pfd.data(), w_norm.data(), nk, nb);
    converged = grad_norm <= tol_g
             && (deflation_on || final_res <= tol_g);

    // Report a chemical potential in the returned orbital gauge without
    // changing its occupations.
    fock_in_orbital_basis(Q.data(), F.data(), Ft.data(), nk, nb, cfg.block_sizes);
    for (std::size_t kk = 0; kk < nk; ++kk)
        for (std::size_t b = 0; b < nb; ++b)
            eps[kk * nb + b] = Ft[kk * nb2 + b * nb + b].real();
    mu = solve_mu_inloop(
        eps.data(), npb, w_norm.data(), nk, nb,
        n_target_norm, mu, T_r, cfg.mu_maxiter);

    if (std::getenv("CPP_HF_DM_DEBUG"))
        std::fprintf(stderr, "[rtr] outer=%zu fock_builds=%zu converged=%d\n",
                     k_outer, n_builds, static_cast<int>(converged));

    if (cfg.return_Q) out.Q = std::move(Q);
    out.p = std::move(p);
    if (cfg.return_density) out.density = std::move(P_cur);
    if (cfg.return_fock) out.fock = std::move(F);
    out.mu = mu;
    out.energy = E_fin;
    out.n_iter = k_outer;
    out.converged = converged;
    out.hist_E.resize(k_outer);
    out.hist_free.resize(k_outer);
    out.hist_grad.resize(k_outer);
    return out;
}

}  // namespace cpp_hf
