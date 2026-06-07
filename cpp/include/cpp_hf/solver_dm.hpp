// Direct minimization solver: preconditioned Riemannian CG on Stiefel × simplex.
#pragma once

#include "cpp_hf/fock.hpp"
#include "cpp_hf/kernel.hpp"
#include "cpp_hf/linalg.hpp"
#include "cpp_hf/utils.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace cpp_hf {

struct SolverConfig {
    std::size_t max_iter = 200;
    f32 tol_E = 1.0e-7;
    f32 tol_grad = 0.0;
    // Energy convergence is judged over a sliding window: stop when the energy
    // has improved by less than tol_E over the last plateau_window iterations.
    // The DM line search makes E monotonically decreasing, so the windowed
    // improvement is a reliable, physically-meaningful stop — unlike a single-
    // iteration |dE| test, which dips below tol_E during CG warm-up / after a
    // backtracked step and stops prematurely.  plateau_window=0 restores the
    // per-iteration test.
    std::size_t plateau_window = 5;
    f32 denom_scale = 1.0e-3;
    f32 max_step = 0.6;
    std::size_t cg_restart = 10;
    f32 bt_shrink = 0.5;
    std::size_t bt_max = 8;
    int mu_maxiter = 25;
    std::vector<std::size_t> block_sizes;  // empty = no block structure
    bool return_Q = true;
    bool return_density = true;
    bool return_fock = true;
};

struct DMResult {
    std::vector<c64> Q;        // (nk, nb, nb)
    std::vector<f32> p;        // (nk, nb)
    std::vector<c64> density;  // (nk, nb, nb)
    std::vector<c64> fock;     // (nk, nb, nb)
    f32 mu = 0.0;
    f32 energy = 0.0;
    std::size_t n_iter = 0;
    bool converged = false;
    std::vector<f32> hist_E;
    std::vector<f32> hist_grad;
};

namespace dm_internal {

inline bool block_sizes_enabled(const std::vector<std::size_t>& sizes,
                                std::size_t nb) {
    if (sizes.empty()) return false;
    std::size_t total = 0;
    for (std::size_t s : sizes) {
        if (s == 0) throw std::invalid_argument("block_sizes entries must be positive.");
        total += s;
    }
    if (total != nb) throw std::invalid_argument("block_sizes must sum to nb.");
    return sizes.size() > 1;
}

inline bool block_structure_ok(const c64* M, std::size_t nk, std::size_t nb,
                               const std::vector<std::size_t>& sizes) {
    const f32 scale = max_abs(M, nk * nb * nb);
    const f32 off = max_offblock_sizes(M, nk, nb, sizes);
    const f32 tol = 1.0e-10 + 1.0e-10 * scale;
    return off <= tol;
}

inline bool block_problem_invariant(const HFKernel& K, const ProjectFn* project_fn,
                                    const std::vector<std::size_t>& sizes) {
    if (project_fn && *project_fn) return false;
    const std::size_t nk = K.nk();
    const std::size_t nb = K.nb;
    if (!block_structure_ok(K.h, nk, nb, sizes)) return false;
    if (K.has_refP && !block_structure_ok(K.refP, nk, nb, sizes)) return false;
    for (std::size_t t = 0; t < K.n_contact; ++t) {
        if (K.contact_g[t] == 0.0) continue;
        if (!block_structure_ok(K.contact_Oi + t * nb * nb, 1, nb, sizes))
            return false;
        if (!block_structure_ok(K.contact_Oj + t * nb * nb, 1, nb, sizes))
            return false;
    }
    return true;
}

inline void density_from_Qp(const c64* Q, const f32* p, c64* P,
                            std::size_t nk, std::size_t nb) {
    const std::size_t nb2 = nb * nb;
    for (std::size_t k = 0; k < nk; ++k) {
        ConstMapMatXcf Qk(Q + k * nb2, nb, nb);
        Eigen::Map<const Eigen::Array<f32, Eigen::Dynamic, 1>> pk(p + k * nb, nb);
        MatXcf Qp = Qk;
        for (std::size_t j = 0; j < nb; ++j) Qp.col(j) *= pk[j];
        MatXcf P_k = Qp * Qk.adjoint();
        std::memcpy(P + k * nb2, P_k.data(), nb2 * sizeof(c64));
    }
}

inline void fock_in_orbital_basis(const c64* Q, const c64* F, c64* Ft,
                                   std::size_t nk, std::size_t nb) {
    const std::size_t nb2 = nb * nb;
    for (std::size_t k = 0; k < nk; ++k) {
        ConstMapMatXcf Qk(Q + k * nb2, nb, nb);
        ConstMapMatXcf Fk(F + k * nb2, nb, nb);
        MatXcf Ftk = Qk.adjoint() * Fk * Qk;
        std::memcpy(Ft + k * nb2, Ftk.data(), nb2 * sizeof(c64));
    }
}

// G_ij = 0.5 * (A_ij - conj(A_ji)),  A = (p_j - p_i) * Ft_ij,  with off-diagonal mask.
inline void compute_orbital_gradient(const c64* Ft, const f32* p, c64* G,
                                     std::size_t nk, std::size_t nb) {
    const std::size_t nb2 = nb * nb;
    for (std::size_t k = 0; k < nk; ++k) {
        const c64* Ftk = Ft + k * nb2;
        const f32* pk = p + k * nb;
        c64* Gk = G + k * nb2;
        for (std::size_t i = 0; i < nb; ++i) {
            for (std::size_t j = 0; j < nb; ++j) {
                if (i == j) { Gk[i * nb + j] = c64(0.0, 0.0); continue; }
                const f32 dp_ij = pk[j] - pk[i];
                const f32 dp_ji = pk[i] - pk[j];
                const c64 a_ij = dp_ij * Ftk[i * nb + j];
                const c64 a_ji = dp_ji * Ftk[j * nb + i];
                Gk[i * nb + j] = 0.5 * (a_ij - std::conj(a_ji));
            }
        }
    }
}

// Weighted Frobenius inner product Σ_k w_k Σ_ij Re(conj(X)_ij * Y_ij)
inline f32 ip_matrix(const c64* X, const c64* Y, const f32* w_norm,
                     std::size_t nk, std::size_t nb) {
    const std::size_t nb2 = nb * nb;
    double total = 0.0;
    for (std::size_t k = 0; k < nk; ++k) {
        const c64* xk = X + k * nb2;
        const c64* yk = Y + k * nb2;
        double per_k = 0.0;
        for (std::size_t i = 0; i < nb2; ++i) {
            per_k += static_cast<double>((std::conj(xk[i]) * yk[i]).real());
        }
        total += static_cast<double>(w_norm[k]) * per_k;
    }
    return static_cast<f32>(total);
}

inline f32 ip_vec(const f32* x, const f32* y, const f32* w_norm,
                  std::size_t nk, std::size_t nb) {
    double total = 0.0;
    for (std::size_t k = 0; k < nk; ++k) {
        const f32* xk = x + k * nb;
        const f32* yk = y + k * nb;
        double per_k = 0.0;
        for (std::size_t b = 0; b < nb; ++b)
            per_k += static_cast<double>(xk[b]) * static_cast<double>(yk[b]);
        total += static_cast<double>(w_norm[k]) * per_k;
    }
    return static_cast<f32>(total);
}

inline f32 norm_matrix(const c64* X, const f32* w_norm,
                       std::size_t nk, std::size_t nb) {
    const f32 sq = ip_matrix(X, X, w_norm, nk, nb);
    return std::sqrt(std::max(0.0, sq));
}

// Cayley spectral setup: eigh(i * d) where d is skew-Hermitian.  Returns
// (V_per_k, lam_per_k).  Hermitises iA exactly first to suppress roundoff.
//
// IMPORTANT: assigning Eigen's eigenvector matrix to a row-major
// ``MapMatXcf`` (rather than memcpy'ing its raw buffer) is what guarantees
// the columns end up where we expect — Eigen's internal storage may not
// match the requested ``MatrixType`` row-majorness.
inline void cayley_spectral_setup(const c64* d, c64* V, f32* lam,
                                   std::size_t nk, std::size_t nb) {
    const std::size_t nb2 = nb * nb;
    std::vector<c64> iA(nb2);
    for (std::size_t k = 0; k < nk; ++k) {
        const c64* dk = d + k * nb2;
        for (std::size_t i = 0; i < nb2; ++i) iA[i] = c64(0.0, 1.0) * dk[i];
        MapMatXcf iAm(iA.data(), nb, nb);
        MatXcf iAh = 0.5 * (iAm + iAm.adjoint());
        Eigen::SelfAdjointEigenSolver<MatXcf> es(iAh);
        const auto& w = es.eigenvalues();
        const auto& Vk = es.eigenvectors();
        for (std::size_t i = 0; i < nb; ++i) lam[k * nb + i] = w[i];
        MapMatXcf VOut(V + k * nb2, nb, nb);
        VOut = Vk;
    }
}

inline void cayley_spectral_setup_block_sizes(const c64* d, c64* V, f32* lam,
                                               std::size_t nk, std::size_t nb,
                                               const std::vector<std::size_t>& sizes) {
    const std::size_t nb2 = nb * nb;
    std::vector<c64> iA(nb2);
    for (std::size_t k = 0; k < nk; ++k) {
        const c64* dk = d + k * nb2;
        for (std::size_t i = 0; i < nb2; ++i) iA[i] = c64(0.0, 1.0) * dk[i];
        MapMatXcf iAm(iA.data(), nb, nb);
        MapMatXcf VOut(V + k * nb2, nb, nb);
        VOut.setZero();

        std::size_t start = 0;
        for (std::size_t s : sizes) {
            MatXcf sub = iAm.block(start, start, s, s);
            sub = 0.5 * (sub + sub.adjoint());
            Eigen::SelfAdjointEigenSolver<MatXcf> es(sub);
            const auto& w = es.eigenvalues();
            const auto& Vk = es.eigenvectors();
            for (std::size_t i = 0; i < s; ++i) lam[k * nb + start + i] = w[i];
            VOut.block(start, start, s, s) = Vk;
            start += s;
        }
    }
}

// c_t = (1 + i τλ/2) / (1 − i τλ/2),  |c_t| = 1 by construction.
inline c64 cayley_factor(f32 lam, f32 tau) {
    const f32 arg = 0.5 * tau * lam;
    const c64 iexp(0.0, arg);
    return (c64(1.0, 0.0) + iexp) / (c64(1.0, 0.0) - iexp);
}

// U(τ) = V · diag(c(τ, λ)) · V†.  Output U_out is per-k (nb, nb).
inline void cayley_unitary_from_spectrum(const c64* V, const f32* lam, f32 tau,
                                          c64* U_out,
                                          std::size_t nk, std::size_t nb) {
    const std::size_t nb2 = nb * nb;
    std::vector<c64> Vc(nb2);
    for (std::size_t k = 0; k < nk; ++k) {
        ConstMapMatXcf Vk(V + k * nb2, nb, nb);
        MatXcf VcM = Vk;
        for (std::size_t j = 0; j < nb; ++j) {
            const c64 c = cayley_factor(lam[k * nb + j], tau);
            VcM.col(j) *= c;
        }
        MatXcf U = VcM * Vk.adjoint();
        std::memcpy(U_out + k * nb2, U.data(), nb2 * sizeof(c64));
    }
}

// diag(U(τ)† Ft U(τ)) = real diag of  V · M(τ) · V†, where
//   M(τ)[i, j] = c̄(τ, λ_i) · Ft_eig[i, j] · c(τ, λ_j)
// Output diag_out has shape (nk, nb).
inline void diag_UFU_from_spectrum(const c64* V, const c64* Ft_eig,
                                    const f32* lam, f32 tau, f32* diag_out,
                                    std::size_t nk, std::size_t nb) {
    const std::size_t nb2 = nb * nb;
    std::vector<c64> c_vec(nb);
    MatXcf M(nb, nb);
    for (std::size_t k = 0; k < nk; ++k) {
        ConstMapMatXcf Vk(V + k * nb2, nb, nb);
        ConstMapMatXcf Fe(Ft_eig + k * nb2, nb, nb);
        for (std::size_t i = 0; i < nb; ++i) c_vec[i] = cayley_factor(lam[k * nb + i], tau);

        for (std::size_t i = 0; i < nb; ++i) {
            const c64 cb_i = std::conj(c_vec[i]);
            for (std::size_t j = 0; j < nb; ++j) {
                M(i, j) = cb_i * Fe(i, j) * c_vec[j];
            }
        }
        MatXcf A = Vk * M;
        // diag_out[k, i] = Re Σ_j A[i, j] * conj(V[i, j])
        for (std::size_t i = 0; i < nb; ++i) {
            double s = 0.0;
            for (std::size_t j = 0; j < nb; ++j)
                s += static_cast<double>((A(i, j) * std::conj(Vk(i, j))).real());
            diag_out[k * nb + i] = static_cast<f32>(s);
        }
    }
}

inline void compute_Ft_eig(const c64* Ft, const c64* V_d, c64* Ft_eig,
                            std::size_t nk, std::size_t nb) {
    const std::size_t nb2 = nb * nb;
    for (std::size_t k = 0; k < nk; ++k) {
        ConstMapMatXcf Vk(V_d + k * nb2, nb, nb);
        ConstMapMatXcf Ftk(Ft + k * nb2, nb, nb);
        MatXcf out = Vk.adjoint() * Ftk * Vk;
        std::memcpy(Ft_eig + k * nb2, out.data(), nb2 * sizeof(c64));
    }
}

// Q_new = Q * U  (per k)
inline void Q_times_U(const c64* Q, const c64* U, c64* Q_new,
                       std::size_t nk, std::size_t nb) {
    const std::size_t nb2 = nb * nb;
    for (std::size_t k = 0; k < nk; ++k) {
        ConstMapMatXcf Qk(Q + k * nb2, nb, nb);
        ConstMapMatXcf Uk(U + k * nb2, nb, nb);
        MatXcf out = Qk * Uk;
        std::memcpy(Q_new + k * nb2, out.data(), nb2 * sizeof(c64));
    }
}

inline void Q_times_cayley_spectrum_inplace(c64* Q, const c64* V,
                                             const f32* lam, f32 tau,
                                             std::size_t nk, std::size_t nb) {
    const std::size_t nb2 = nb * nb;
    for (std::size_t k = 0; k < nk; ++k) {
        ConstMapMatXcf Qk(Q + k * nb2, nb, nb);
        ConstMapMatXcf Vk(V + k * nb2, nb, nb);
        MatXcf VcM = Vk;
        for (std::size_t j = 0; j < nb; ++j) {
            const c64 c = cayley_factor(lam[k * nb + j], tau);
            VcM.col(j) *= c;
        }
        MatXcf U = VcM * Vk.adjoint();
        MatXcf out = Qk * U;
        std::memcpy(Q + k * nb2, out.data(), nb2 * sizeof(c64));
    }
}

inline f32 frozen_free_energy_from_occ(const f32* eps, const f32* p,
                                       const f32* w_norm,
                                       std::size_t nk, std::size_t nb,
                                       f32 T) {
    double E = 0.0;
    double S = 0.0;
    for (std::size_t k = 0; k < nk; ++k) {
        const f32 wk = w_norm[k];
        for (std::size_t b = 0; b < nb; ++b) {
            const std::size_t idx = k * nb + b;
            const f32 pv = std::clamp(p[idx], OCC_CLIP, 1.0 - OCC_CLIP);
            E += static_cast<double>(wk) * static_cast<double>(pv)
               * static_cast<double>(eps[idx]);
            const double pd = static_cast<double>(pv);
            S -= static_cast<double>(wk) *
                 (pd * std::log(pd) + (1.0 - pd) * std::log1p(-pd));
        }
    }
    return static_cast<f32>(E - static_cast<double>(std::max(T, 1.0e-14)) * S);
}

inline f32 frozen_free_energy_trial(const f32* eps, const f32* p,
                                    const f32* d_p, f32 tau,
                                    const f32* w_norm,
                                    std::size_t nk, std::size_t nb,
                                    f32 T) {
    double E = 0.0;
    double S = 0.0;
    for (std::size_t k = 0; k < nk; ++k) {
        const f32 wk = w_norm[k];
        for (std::size_t b = 0; b < nb; ++b) {
            const std::size_t idx = k * nb + b;
            const f32 pv = std::clamp(p[idx] - tau * d_p[idx],
                                      OCC_CLIP, 1.0 - OCC_CLIP);
            E += static_cast<double>(wk) * static_cast<double>(pv)
               * static_cast<double>(eps[idx]);
            const double pd = static_cast<double>(pv);
            S -= static_cast<double>(wk) *
                 (pd * std::log(pd) + (1.0 - pd) * std::log1p(-pd));
        }
    }
    return static_cast<f32>(E - static_cast<double>(std::max(T, 1.0e-14)) * S);
}

}  // namespace dm_internal

inline DMResult solve_dm(const HFKernel& K, const c64* P0, f32 n_e,
                          const SolverConfig& cfg,
                          const ProjectFn* project_fn = nullptr) {
    using namespace dm_internal;

    const std::size_t nk1 = K.nk1;
    const std::size_t nk2 = K.nk2;
    const std::size_t nb = K.nb;
    const std::size_t nk = nk1 * nk2;
    const std::size_t nb2 = nb * nb;
    const std::size_t n_tot = nk * nb2;
    const bool block_cfg_enabled = block_sizes_enabled(cfg.block_sizes, nb);
    bool use_block_ops = block_cfg_enabled
        && block_problem_invariant(K, project_fn, cfg.block_sizes);

    // Normalised weights and target electron count
    std::vector<f32> w_norm(nk);
    const f32 ws = std::max(K.weight_sum, TINY_REAL);
    for (std::size_t i = 0; i < nk; ++i) w_norm[i] = K.w2d[i] / ws;
    const f32 n_target_norm = n_e / ws;

    const f32 T_r = std::max(K.T, 1.0e-12);

    // Workspaces
    std::vector<c64> Q(n_tot);
    std::vector<f32> p(nk * nb);
    std::vector<c64> P_cur(n_tot);
    std::vector<c64> Sigma(n_tot), F(n_tot);
    std::vector<f32> hartree_diag(nb, 0.0);
    std::vector<f32> eps(nk * nb);
    std::vector<c64> G_Q(n_tot);
    std::vector<f32> g_p(nk * nb), h_p(nk * nb);
    std::vector<c64> G_Q_prev(n_tot, c64(0.0, 0.0));
    std::vector<f32> g_p_prev(nk * nb, 0.0);
    std::vector<c64> d_Q(n_tot), d_Q_prev(n_tot, c64(0.0, 0.0));
    std::vector<f32> d_p(nk * nb), d_p_prev(nk * nb, 0.0);
    std::vector<c64> V_d(n_tot);
    std::vector<f32> lam_d(nk * nb);
    std::vector<f32> eps_work(nk * nb);

    DMResult out;
    out.hist_E.assign(cfg.max_iter, 0.0);
    out.hist_grad.assign(cfg.max_iter, 0.0);

    // ---- Initialize (Q, p) from Fock at P0 ----
    {
        std::memcpy(P_cur.data(), P0, n_tot * sizeof(c64));
        hermitize_inplace(P_cur.data(), nk, nb);
        // Use the physical one-body Fock for the initial eigensolve.  Symmetry
        // callbacks constrain densities below; projecting F here would also
        // project the bare continuum Hamiltonian, which is wrong for rhombic
        // C3 because that grid is C3-closed only as a torus.
        build_fock_compact(K, P_cur.data(), Sigma.data(), F.data(),
                           hartree_diag.data(), nullptr);
        if (use_block_ops && !block_structure_ok(F.data(), nk, nb, cfg.block_sizes))
            use_block_ops = false;
        if (use_block_ops)
            eigh_block_sizes_batched(F.data(), eps.data(), Q.data(), nk1, nk2, nb,
                                     cfg.block_sizes, false);
        else
            eigh_batched(F.data(), eps.data(), Q.data(), nk1, nk2, nb);
        const f32 mu0 = solve_mu_inloop(eps.data(), nk * nb, w_norm.data(),
                                         nk, nb, n_target_norm, 0.0, T_r,
                                         cfg.mu_maxiter);
        for (std::size_t i = 0; i < nk * nb; ++i)
            p[i] = expit_scalar((mu0 - eps[i]) / T_r);
        out.mu = mu0;
    }

    auto project_in_place = [&](c64* M) {
        if (project_fn && *project_fn) {
            (*project_fn)(M, nk1, nk2, nb);
        }
    };

    f32 grad_norm = std::numeric_limits<f32>::infinity();
    f32 E_prev = std::numeric_limits<f32>::infinity();
    f32 dE = std::numeric_limits<f32>::infinity();
    f32 E = 0.0;
    f32 mu = out.mu;
    std::size_t k = 0;

    while (k < cfg.max_iter) {
        bool energy_not_converged;
        if (cfg.plateau_window > 0 && k > cfg.plateau_window) {
            // Improvement over the last plateau_window recorded energies.
            const f32 e_now = out.hist_E[k - 1];
            const f32 e_past = out.hist_E[k - 1 - cfg.plateau_window];
            energy_not_converged = (e_past - e_now) > cfg.tol_E;
        } else {
            energy_not_converged = (dE > cfg.tol_E);
        }
        bool grad_not_converged = (cfg.tol_grad > 0.0) && (grad_norm > cfg.tol_grad);
        if (!(energy_not_converged || grad_not_converged)) break;

        // 1. Build Fock at projected hermitised density
        density_from_Qp(Q.data(), p.data(), P_cur.data(), nk, nb);
        project_in_place(P_cur.data());
        hermitize_inplace(P_cur.data(), nk, nb);

        const ProjectFn* fock_proj = nullptr;  // F-projection skipped (matches jax_hf)
        build_fock_compact(K, P_cur.data(), Sigma.data(), F.data(),
                           hartree_diag.data(), fock_proj);
        E = hf_energy_with_hartree_diag(K, P_cur.data(), Sigma.data(),
                                        hartree_diag.data());
        fock_in_orbital_basis(Q.data(), F.data(), F.data(), nk, nb);
        for (std::size_t kk = 0; kk < nk; ++kk) {
            const c64* Ftk = F.data() + kk * nb2;
            for (std::size_t b = 0; b < nb; ++b)
                eps[kk * nb + b] = Ftk[b * nb + b].real();
        }

        // 2. Gradient
        compute_orbital_gradient(F.data(), p.data(), G_Q.data(), nk, nb);
        for (std::size_t i = 0; i < nk * nb; ++i) {
            f32 pv = std::clamp(p[i], LOGIT_CLIP, 1.0 - LOGIT_CLIP);
            const f32 logit = std::log(pv) - std::log1p(-pv);
            g_p[i] = eps[i] + T_r * logit - mu;
        }
        grad_norm = norm_matrix(G_Q.data(), w_norm.data(), nk, nb);

        // 3. Precondition
        double eps_sq_sum = 0.0;
        for (std::size_t i = 0; i < nk * nb; ++i)
            eps_sq_sum += static_cast<double>(eps[i]) * static_cast<double>(eps[i]);
        const f32 eps_scale = static_cast<f32>(
            std::sqrt(eps_sq_sum / std::max<std::size_t>(nk * nb, 1) + static_cast<double>(TINY_REAL)));
        const f32 lam_pc = std::max(T_r, cfg.denom_scale * eps_scale);

        for (std::size_t kk = 0; kk < nk; ++kk) {
            const c64* Gk = G_Q.data() + kk * nb2;
            c64* Hk = d_Q.data() + kk * nb2;
            const f32* eb = eps.data() + kk * nb;
            for (std::size_t i = 0; i < nb; ++i) {
                for (std::size_t j = 0; j < nb; ++j) {
                    const f32 gap = eb[i] - eb[j];
                    const f32 base = std::sqrt(gap * gap + lam_pc * lam_pc);
                    Hk[i * nb + j] = Gk[i * nb + j] / base;
                }
            }
        }
        for (std::size_t kk = 0; kk < nk; ++kk) {
            for (std::size_t i = 0; i < nb; ++i) {
                const std::size_t idx = kk * nb + i;
                f32 ps = std::clamp(p[idx], OCC_CLIP, 1.0 - OCC_CLIP);
                const double curv = static_cast<double>(T_r) / (
                    static_cast<double>(ps) * static_cast<double>(1.0 - ps));
                h_p[idx] = g_p[idx] / static_cast<f32>(curv);
            }
        }

        // 4. CG direction (Polak-Ribière+)
        const f32 pr_num = (
            ip_matrix(G_Q.data(), d_Q.data(), w_norm.data(), nk, nb)
            - ip_matrix(G_Q_prev.data(), d_Q.data(), w_norm.data(), nk, nb)
            + ip_vec(g_p.data(), h_p.data(), w_norm.data(), nk, nb)
            - ip_vec(g_p_prev.data(), h_p.data(), w_norm.data(), nk, nb));
        const f32 pr_den = (
            ip_matrix(G_Q_prev.data(), G_Q_prev.data(), w_norm.data(), nk, nb)
            + ip_vec(g_p_prev.data(), g_p_prev.data(), w_norm.data(), nk, nb));
        f32 beta = (pr_den > TINY_REAL) ? std::max(0.0, pr_num / pr_den) : 0.0;
        beta = std::min(beta, 5.0);
        if (k == 0 || (cfg.cg_restart > 0 && k % cfg.cg_restart == 0)) beta = 0.0;

        for (std::size_t i = 0; i < n_tot; ++i)
            d_Q[i] = d_Q[i] + beta * d_Q_prev[i];
        for (std::size_t i = 0; i < nk * nb; ++i)
            d_p[i] = h_p[i] + beta * d_p_prev[i];

        // 5. Line search (frozen-F free energy, spectral Cayley)
        if (use_block_ops)
            cayley_spectral_setup_block_sizes(d_Q.data(), V_d.data(), lam_d.data(),
                                              nk, nb, cfg.block_sizes);
        else
            cayley_spectral_setup(d_Q.data(), V_d.data(), lam_d.data(), nk, nb);
        compute_Ft_eig(F.data(), V_d.data(), Sigma.data(), nk, nb);

        const f32 d_Q_norm = norm_matrix(d_Q.data(), w_norm.data(), nk, nb);
        f32 tau0 = std::min(1.0, cfg.max_step / std::max(d_Q_norm, TINY_REAL));

        const f32 Omega0 = frozen_free_energy_from_occ(
            eps.data(), p.data(), w_norm.data(), nk, nb, T_r);

        f32 tau_final = tau0;
        bool bt_accepted = false;
        for (std::size_t bt = 0; bt < cfg.bt_max; ++bt) {
            diag_UFU_from_spectrum(V_d.data(), Sigma.data(), lam_d.data(),
                                   tau_final, eps_work.data(), nk, nb);
            const f32 Omega_trial = frozen_free_energy_trial(
                eps_work.data(), p.data(), d_p.data(), tau_final,
                w_norm.data(), nk, nb, T_r);
            if (Omega_trial < Omega0) { bt_accepted = true; break; }
            tau_final *= cfg.bt_shrink;
        }
        if (!bt_accepted) {
            tau_final = tau0 * std::pow(cfg.bt_shrink,
                                        static_cast<f32>(cfg.bt_max));
        }
        // 6. Retraction
        Q_times_cayley_spectrum_inplace(Q.data(), V_d.data(), lam_d.data(),
                                        tau_final, nk, nb);
        diag_UFU_from_spectrum(V_d.data(), Sigma.data(), lam_d.data(),
                               tau_final, eps_work.data(), nk, nb);
        const f32 mu_new = solve_mu_inloop(eps_work.data(), nk * nb, w_norm.data(),
                                            nk, nb, n_target_norm, mu, T_r,
                                            cfg.mu_maxiter);
        for (std::size_t i = 0; i < nk * nb; ++i)
            p[i] = expit_scalar((mu_new - eps_work[i]) / T_r);
        mu = mu_new;

        // 7. Record + advance
        out.hist_E[k] = E;
        out.hist_grad[k] = grad_norm;
        dE = std::abs(E - E_prev);

        d_Q_prev = d_Q;
        d_p_prev = d_p;
        G_Q_prev = G_Q;
        g_p_prev = g_p;
        E_prev = E;
        ++k;
    }

    // ---- Finalize ----
    density_from_Qp(Q.data(), p.data(), P_cur.data(), nk, nb);
    project_in_place(P_cur.data());
    hermitize_inplace(P_cur.data(), nk, nb);
    build_fock_compact(K, P_cur.data(), Sigma.data(), F.data(),
                       hartree_diag.data(), nullptr);
    fock_in_orbital_basis(Q.data(), F.data(), F.data(), nk, nb);
    if (use_block_ops)
        eigh_block_sizes_batched(F.data(), eps_work.data(), V_d.data(),
                                 nk1, nk2, nb, cfg.block_sizes, false);
    else
        eigh_batched(F.data(), eps_work.data(), V_d.data(), nk1, nk2, nb);
    // Q = Q @ V_fin
    Q_times_U(Q.data(), V_d.data(), Q.data(), nk, nb);
    mu = solve_mu_inloop(eps_work.data(), nk * nb, w_norm.data(),
                          nk, nb, n_target_norm, mu, T_r, cfg.mu_maxiter);
    for (std::size_t i = 0; i < nk * nb; ++i)
        p[i] = expit_scalar((mu - eps_work[i]) / T_r);

    density_from_Qp(Q.data(), p.data(), P_cur.data(), nk, nb);
    project_in_place(P_cur.data());
    hermitize_inplace(P_cur.data(), nk, nb);
    build_fock_compact(K, P_cur.data(), Sigma.data(), F.data(),
                       hartree_diag.data(), nullptr);
    const f32 E_fin = hf_energy_with_hartree_diag(K, P_cur.data(), Sigma.data(),
                                                  hartree_diag.data());

    if (cfg.return_Q) out.Q = std::move(Q);
    out.p = std::move(p);
    if (cfg.return_density) out.density = std::move(P_cur);
    if (cfg.return_fock) out.fock = std::move(F);
    out.mu = mu;
    out.energy = E_fin;
    out.n_iter = k;
    out.converged = (k < cfg.max_iter);
    out.hist_E.resize(k);
    out.hist_grad.resize(k);
    return out;
}


}  // namespace cpp_hf
