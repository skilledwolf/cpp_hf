// Trust-region Newton solver for HF direct minimization.
//
// Minimises the same free energy Ω(Q,p)=E[P]-T·S(p) as solve_dm, but with a
// second-order step: the joint (Q,p) Hessian-vector product uses the EXACT
// linear interaction response Σ[δP] = F[δP]-h (one Fock build each), and a
// Steihaug truncated-CG solves the trust-region subproblem.  Far fewer Fock
// builds than CG on stiff problems (superlinear outer convergence).
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
    build_fock_compact(Kr, dP, Sigma_buf, F_buf, hartree_buf, nullptr);
    for (std::size_t i = 0; i < n_tot; ++i) resp[i] = F_buf[i] - K.h[i];
}

// Project occupation vector onto the particle-conserving subspace Σ_k w_k δp = 0.
inline void project_occ(f32* dp, const f32* w2d, std::size_t nk, std::size_t nb) {
    double num = 0.0, den = 0.0;
    for (std::size_t k = 0; k < nk; ++k) {
        for (std::size_t b = 0; b < nb; ++b) num += static_cast<double>(w2d[k]) * dp[k * nb + b];
        den += static_cast<double>(w2d[k]) * static_cast<double>(nb);
    }
    const f32 shift = (den > 0.0) ? static_cast<f32>(num / den) : 0.0;
    for (std::size_t i = 0; i < nk * nb; ++i) dp[i] -= shift;
}

// Joint (Q,p) Hessian-vector product.  Inputs X (skew-Herm off-diagonal,
// orbital basis), dp (real per-(k,b)); outputs HX (skew-Herm off-diagonal) and
// Hp (real, particle-conserving).  One Fock build via fock_response.
inline void joint_hvp(const HFKernel& K, const c64* X, const f32* dp,
                      const c64* Q, const f32* p, const c64* Ft, f32 T_r,
                      const f32* w2d, std::size_t nk, std::size_t nb,
                      c64* HX, f32* Hp, c64* dP_buf, c64* resp_buf,
                      c64* Sigma_buf, c64* F_buf, f32* hartree_buf) {
    const std::size_t nb2 = nb * nb;
    // 1. Build the combined density variation δP = Q ([X,diag p] + diag(dp)) Q†.
    for (std::size_t k = 0; k < nk; ++k) {
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
        dPk = 0.5 * (dPk + dPk.adjoint());  // exact Hermitisation
        std::memcpy(dP_buf + k * nb2, dPk.data(), nb2 * sizeof(c64));
    }
    // 2. One Fock build: the interaction response.
    fock_response(K, dP_buf, resp_buf, Sigma_buf, F_buf, hartree_buf, nk * nb2);
    // 3. Assemble HX and Hp per k.
    for (std::size_t k = 0; k < nk; ++k) {
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
        MatXcf HXk = 0.5 * (C - C.adjoint());
        for (std::size_t d = 0; d < nb; ++d) HXk(d, d) = c64(0.0, 0.0);
        std::memcpy(HX + k * nb2, HXk.data(), nb2 * sizeof(c64));
        for (std::size_t b = 0; b < nb; ++b) {
            const f32 pc = std::clamp(pk[b], OCC_CLIP, 1.0 - OCC_CLIP);
            Hp[k * nb + b] = static_cast<f32>(A(b, b).real())
                             + T_r / (pc * (1.0 - pc)) * dpk[b];
        }
    }
    project_occ(Hp, w2d, nk, nb);
}

}  // namespace rtr_internal

inline DMResult solve_rtr(const HFKernel& K, const c64* P0, f32 n_e,
                          const SolverConfig& cfg,
                          const ProjectFn* /*project_fn*/ = nullptr) {
    using namespace dm_internal;
    using namespace rtr_internal;

    const std::size_t nk1 = K.nk1, nk2 = K.nk2, nb = K.nb;
    const std::size_t nk = nk1 * nk2, nb2 = nb * nb, n_tot = nk * nb2;
    const std::size_t npb = nk * nb;
    const f32 T_r = std::max(K.T, 1.0e-12);
    const f32 tol_g = (cfg.tol_grad > 0.0) ? cfg.tol_grad : 1.0e-6;

    std::vector<f32> w_norm(nk), ones(nk, 1.0);
    const f32 ws = std::max(K.weight_sum, TINY_REAL);
    for (std::size_t i = 0; i < nk; ++i) w_norm[i] = K.w2d[i] / ws;
    const f32 n_target_norm = n_e / ws;

    // State + workspaces.
    std::vector<c64> Q(n_tot), P_cur(n_tot), Sigma(n_tot), F(n_tot), Ft(n_tot);
    std::vector<f32> p(npb), eps(npb), hartree_diag(nb, 0.0);
    std::vector<c64> G(n_tot);                       // orbital gradient
    std::vector<c64> vX(n_tot), dPb(n_tot), respb(n_tot), Fb(n_tot), Sb(n_tot);
    std::vector<f32> vp(npb), hb(nb, 0.0);
    std::vector<c64> rX(n_tot), dX(n_tot), zX(n_tot), HvX(n_tot), HDX(n_tot);
    std::vector<f32> rp(npb), dp(npb), zp(npb), Hvp(npb), HDp(npb);
    std::vector<c64> Vd(n_tot), Qt(n_tot), Ub(n_tot);
    std::vector<f32> lamd(npb);
    std::vector<c64> Gsave(n_tot), Ftsave(n_tot);
    std::vector<f32> epssave(npb), eps_t(npb), p_trial(npb);

    DMResult out;
    out.hist_E.assign(cfg.max_iter, 0.0);
    out.hist_grad.assign(cfg.max_iter, 0.0);
    std::size_t n_builds = 0;  // Fock-build counter (env-gated diagnostic)

    // Joint metric (mirrors the validated prototype): orbital part unweighted,
    // occupation part weighted by w2d.
    auto jip = [&](const c64* aX, const f32* ap, const c64* bX, const f32* bp) {
        return ip_matrix(aX, bX, ones.data(), nk, nb)
               + ip_vec(ap, bp, K.w2d, nk, nb);
    };
    auto jfro = [&](const c64* aX, const f32* ap) {
        return std::sqrt(std::max(0.0, jip(aX, ap, aX, ap)));
    };
    // Block preconditioner: orbital 1/√(gap²+λ²), occupation p(1-p)/T (+ project).
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
        for (std::size_t k = 0; k < nk; ++k)
            for (std::size_t b = 0; b < nb; ++b) {
                const f32 pc = std::clamp(p[k * nb + b], OCC_CLIP, 1.0 - OCC_CLIP);
                op[k * nb + b] = ap[k * nb + b] * pc * (1.0 - pc) / T_r;
            }
        project_occ(op, K.w2d, nk, nb);
    };
    auto grad_norm_w = [&](const c64* g) {
        return norm_matrix(g, w_norm.data(), nk, nb);
    };

    // ---- initialise (Q, p) from F[P0] ----
    std::memcpy(P_cur.data(), P0, n_tot * sizeof(c64));
    hermitize_inplace(P_cur.data(), nk, nb);
    build_fock_compact(K, P_cur.data(), Sigma.data(), F.data(), hartree_diag.data(), nullptr);
    ++n_builds;
    eigh_batched(F.data(), eps.data(), Q.data(), nk1, nk2, nb);
    {
        const f32 mu0 = solve_mu_inloop(eps.data(), npb, w_norm.data(), nk, nb,
                                        n_target_norm, 0.0, T_r, cfg.mu_maxiter);
        for (std::size_t i = 0; i < npb; ++i) p[i] = expit_scalar((mu0 - eps[i]) / T_r);
        out.mu = mu0;
    }

    auto eval_state = [&](const c64* Qs, const f32* ps, f32& E_out) {
        density_from_Qp(Qs, ps, P_cur.data(), nk, nb);
        hermitize_inplace(P_cur.data(), nk, nb);
        build_fock_compact(K, P_cur.data(), Sigma.data(), F.data(), hartree_diag.data(), nullptr);
        ++n_builds;
        E_out = hf_energy_with_hartree_diag(K, P_cur.data(), Sigma.data(), hartree_diag.data());
        fock_in_orbital_basis(Qs, F.data(), Ft.data(), nk, nb);
        for (std::size_t k = 0; k < nk; ++k)
            for (std::size_t b = 0; b < nb; ++b)
                eps[k * nb + b] = Ft[k * nb2 + b * nb + b].real();
        compute_orbital_gradient(Ft.data(), ps, G.data(), nk, nb);
    };

    f32 E = 0.0;
    eval_state(Q.data(), p.data(), E);
    f32 mu = out.mu, Delta = cfg.tr_delta0, grad_norm = grad_norm_w(G.data());
    std::size_t k_outer = 0;
    bool converged = false;

    while (k_outer < cfg.max_iter) {
        grad_norm = grad_norm_w(G.data());
        out.hist_E[k_outer] = E;
        out.hist_grad[k_outer] = grad_norm;
        if (grad_norm < tol_g) { converged = true; ++k_outer; break; }

        double eps_sq = 0.0;
        for (std::size_t i = 0; i < npb; ++i) eps_sq += double(eps[i]) * eps[i];
        const f32 lam_pc = std::max(T_r, cfg.denom_scale *
            static_cast<f32>(std::sqrt(eps_sq / std::max<std::size_t>(npb, 1) + TINY_REAL)));

        // ---- Steihaug truncated CG: minimise the joint quadratic model ----
        std::fill(vX.begin(), vX.end(), c64(0.0, 0.0)); std::fill(vp.begin(), vp.end(), 0.0);
        std::fill(HvX.begin(), HvX.end(), c64(0.0, 0.0)); std::fill(Hvp.begin(), Hvp.end(), 0.0);
        for (std::size_t i = 0; i < n_tot; ++i) rX[i] = -G[i];
        std::fill(rp.begin(), rp.end(), 0.0);
        japply_prec(rX.data(), rp.data(), zX.data(), zp.data(), lam_pc);
        std::memcpy(dX.data(), zX.data(), n_tot * sizeof(c64));
        std::memcpy(dp.data(), zp.data(), npb * sizeof(f32));
        f32 rz = jip(rX.data(), rp.data(), zX.data(), zp.data());

        for (std::size_t cg = 0; cg < cfg.tr_cg_max; ++cg) {
            joint_hvp(K, dX.data(), dp.data(), Q.data(), p.data(), Ft.data(), T_r,
                      K.w2d, nk, nb, HDX.data(), HDp.data(), dPb.data(), respb.data(),
                      Sb.data(), Fb.data(), hb.data());
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
            // ‖v + α·d‖ in the joint metric (orbital unweighted, occupation w2d-weighted):
            double s2 = 0.0;
            for (std::size_t i = 0; i < n_tot; ++i) s2 += std::norm(vX[i] + alpha * dX[i]);
            for (std::size_t kk = 0; kk < nk; ++kk) {
                double per = 0.0;
                for (std::size_t b = 0; b < nb; ++b) {
                    const f32 e = vp[kk * nb + b] + alpha * dp[kk * nb + b];
                    per += double(e) * e;
                }
                s2 += double(K.w2d[kk]) * per;
            }
            if (std::sqrt(std::max(0.0, s2)) >= Delta) { to_boundary(); break; }
            for (std::size_t i = 0; i < n_tot; ++i) { vX[i] += alpha * dX[i]; HvX[i] += alpha * HDX[i]; rX[i] -= alpha * HDX[i]; }
            for (std::size_t i = 0; i < npb; ++i) { vp[i] += alpha * dp[i]; Hvp[i] += alpha * HDp[i]; rp[i] -= alpha * HDp[i]; }
            const f32 rn = std::sqrt(std::max(0.0, jip(rX.data(), rp.data(), rX.data(), rp.data())));
            if (rn < 0.05 * grad_norm) break;
            japply_prec(rX.data(), rp.data(), zX.data(), zp.data(), lam_pc);
            const f32 rz_new = jip(rX.data(), rp.data(), zX.data(), zp.data());
            const f32 beta = rz_new / rz;
            for (std::size_t i = 0; i < n_tot; ++i) dX[i] = zX[i] + beta * dX[i];
            for (std::size_t i = 0; i < npb; ++i) dp[i] = zp[i] + beta * dp[i];
            rz = rz_new;
        }

        // Predicted reduction m(0) − m(v) = −(⟨g,v⟩ + ½⟨v,Hv⟩), g = (G, 0), in
        // TRUE energy units.  The orbital gradient and Hessian-action are per-k
        // (unweighted), while the energy is Σ_k w2d·e_k, so BOTH the linear and
        // quadratic terms must carry the BZ measure w2d.  (The CG step above is
        // solved in the jip metric; only this predicted reduction — which the
        // trust-region ratio compares against the actual ΔE — must match the
        // energy's weighting.  Using the unweighted/mixed metric here makes pred
        // larger than ΔE by ~nk/weight_sum, so every ratio ≈ 0, the step is
        // rejected, and the trust radius collapses without making progress.)
        const f32 gv = ip_matrix(G.data(), vX.data(), K.w2d, nk, nb);
        const f32 vHv = ip_matrix(vX.data(), HvX.data(), K.w2d, nk, nb)
                      + ip_vec(vp.data(), Hvp.data(), K.w2d, nk, nb);
        const f32 pred = -(gv + 0.5 * vHv);
        if (Delta < 1e-10) break;

        // ---- retract Q ← Q·exp(+vX) ≈ Q·Cayley(τ=−1, vX); re-solve μ ----
        // The Cayley factor satisfies Cayley(τ,d) ≈ I − τd (solve_dm convention:
        // d=+gradient, τ>0 descends), so to move ALONG the Newton step +vX we
        // use τ = −1.  (Steihaug returns vX = −H⁻¹G, the descent direction.)
        cayley_spectral_setup(vX.data(), Vd.data(), lamd.data(), nk, nb);
        cayley_unitary_from_spectrum(Vd.data(), lamd.data(), -1.0, Ub.data(), nk, nb);
        Q_times_U(Q.data(), Ub.data(), Qt.data(), nk, nb);
        // Trial orbital energies diag(U† Ft U) from the FROZEN Ft (no Fock build).
        for (std::size_t kk = 0; kk < nk; ++kk) {
            ConstMapMatXcf Uk(Ub.data() + kk * nb2, nb, nb);
            ConstMapMatXcf Ftk(Ft.data() + kk * nb2, nb, nb);
            MatXcf Fttk = Uk.adjoint() * Ftk * Uk;
            for (std::size_t b = 0; b < nb; ++b) eps_t[kk * nb + b] = Fttk(b, b).real();
        }
        const f32 mu_t = solve_mu_inloop(eps_t.data(), npb, w_norm.data(), nk, nb,
                                         n_target_norm, mu, T_r, cfg.mu_maxiter);
        for (std::size_t i = 0; i < npb; ++i)
            p_trial[i] = expit_scalar((mu_t - eps_t[i]) / T_r);

        // Save current gradient/Ft/eps so a rejection costs no Fock build.
        std::memcpy(Gsave.data(), G.data(), n_tot * sizeof(c64));
        std::memcpy(Ftsave.data(), Ft.data(), n_tot * sizeof(c64));
        std::memcpy(epssave.data(), eps.data(), npb * sizeof(f32));

        // True energy + gradient at the trial point (the one Fock build per outer).
        f32 E_trial = 0.0;
        eval_state(Qt.data(), p_trial.data(), E_trial);
        const f32 grad_trial = grad_norm_w(G.data());

        const f32 noise = 1e-12 * std::max(std::abs(E), 1.0);
        bool accept;
        if (pred <= noise) {
            // Predicted reduction below float64 energy noise: judge by the
            // gradient, but never accept a genuine energy increase — a minimizer
            // must not climb to a saddle/maximum.
            accept = (grad_trial <= grad_norm) && (E_trial <= E + noise);
            if (!accept) Delta *= 0.5;
        } else {
            const f32 ratio = (E - E_trial) / pred;
            if (ratio < 0.25) Delta *= 0.25;
            else if (ratio > 0.75 && jfro(vX.data(), vp.data()) > 0.9 * Delta)
                Delta = std::min(2.0 * Delta, 5.0);
            accept = (ratio > 0.1);
        }

        if (std::getenv("CPP_HF_DM_DEBUG")) {
            std::fprintf(stderr,
                "[rtr] k=%zu g=%.3e |v|=%.3e Delta=%.3e pred=%.3e dE=%+.3e gtr=%.3e accept=%d branch=%s\n",
                k_outer, grad_norm, jfro(vX.data(), vp.data()), Delta, pred,
                (E_trial - E), grad_trial, static_cast<int>(accept),
                (pred <= noise ? "grad" : "ratio"));
        }

        if (accept) {
            std::memcpy(Q.data(), Qt.data(), n_tot * sizeof(c64));
            std::memcpy(p.data(), p_trial.data(), npb * sizeof(f32));
            E = E_trial; mu = mu_t;  // G/Ft/eps already at (Qt, p_trial)
        } else {
            std::memcpy(G.data(), Gsave.data(), n_tot * sizeof(c64));
            std::memcpy(Ft.data(), Ftsave.data(), n_tot * sizeof(c64));
            std::memcpy(eps.data(), epssave.data(), npb * sizeof(f32));
        }
        ++k_outer;
    }

    // ---- finalize ----
    density_from_Qp(Q.data(), p.data(), P_cur.data(), nk, nb);
    hermitize_inplace(P_cur.data(), nk, nb);
    build_fock_compact(K, P_cur.data(), Sigma.data(), F.data(), hartree_diag.data(), nullptr);
    ++n_builds;
    const f32 E_fin = hf_energy_with_hartree_diag(K, P_cur.data(), Sigma.data(), hartree_diag.data());
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
    out.hist_grad.resize(k_outer);
    return out;
}

}  // namespace cpp_hf
