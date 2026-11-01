// Reference SCF solver: Roothaan iteration with linear / DIIS / ODA
// acceleration and optional block-diagonal Fock eigh.
#pragma once

#include "cpp_hf/fock.hpp"
#include "cpp_hf/kernel.hpp"
#include "cpp_hf/linalg.hpp"
#include "cpp_hf/parallel.hpp"
#include "cpp_hf/utils.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace cpp_hf {

struct SCFConfig {
    std::size_t max_iter = 200;
    f32 density_tol = 1.0e-7;
    f32 comm_tol = 1.0e-6;
    f32 mixing = 0.5;
    f32 level_shift = 0.0;
    // Block-diagonal acceleration for the Fock eigh.  When non-empty + the
    // Fock has block structure, eigh runs per block (4× speedup for ABC5).
    std::vector<std::size_t> block_sizes;
    // Acceleration scheme: "linear" (default — α-mixing), "diis" (Pulay
    // commutator-DIIS extrapolation of Fock), or "oda" (optimal damping —
    // exact analytic line search on E((1−λ)P + λP_new) per iter).
    std::string acceleration = "linear";
    std::string preconditioner = "none"; // Anderson: none or diagonal Hartree response
    bool safeguard = true;              // Anderson: true free-energy line search
    std::size_t diis_size = 6;        // history depth (Pulay)
    std::size_t diis_start = 2;       // begin extrapolation after this many iter
    // DIIS damping: blend the extrapolated Fock with the current iter's Fock.
    //   F_used = damp · F_extrap + (1−damp) · F_current
    // damp=1.0 → pure DIIS (default).  damp ∈ [0.6, 0.9] suppresses the
    // late-iteration oscillation that prevents tight commutator convergence
    // when the residual norm has plateaued.
    f32 diis_damping = 1.0;
    // Trust radius on the density update.  After each iter the proposed
    // density step Δ = density_proposed − density is rescaled if its
    // weighted Frobenius norm exceeds ``trust_radius``.  0.0 = disabled
    // (default).  Useful when DIIS extrapolations want to take huge jumps
    // (e.g. ungated Coulomb at high doping) that drive the iteration to
    // unphysical states.
    f32 trust_radius = 0.0;
    bool return_density = true;
    bool return_fock = true;
};

struct SCFResult {
    std::vector<c64> density;     // (nk1, nk2, nb, nb) -- pre-final-build density
    std::vector<c64> fock;        // F at convergence (rebuilt on final density)
    f32 energy = 0.0;
    f32 mu = 0.0;
    std::size_t iterations = 0;
    bool converged = false;
    std::vector<f32> hist_E;
    std::vector<f32> hist_free;
    std::vector<f32> hist_density;
    std::vector<f32> hist_comm;
};

inline f32 weighted_matrix_norm(const c64* M, const f32* weights, std::size_t nk,
                                 std::size_t nb, f32 weight_sum) {
    const std::size_t nb2 = nb * nb;
    double total = 0.0;
    for (std::size_t k = 0; k < nk; ++k) {
        const c64* mk = M + k * nb2;
        double per_k = 0.0;
        for (std::size_t i = 0; i < nb2; ++i) per_k += static_cast<double>(std::norm(mk[i]));
        total += static_cast<double>(weights[k]) * per_k;
    }
    return static_cast<f32>(std::sqrt(total / std::max(static_cast<double>(weight_sum), 1.0e-30)));
}

inline f32 weighted_difference_norm(const c64* A, const c64* B,
                                    const f32* weights, std::size_t nk,
                                    std::size_t nb, f32 weight_sum) {
    const std::size_t nb2 = nb * nb;
    double total = 0.0;
    for (std::size_t k = 0; k < nk; ++k) {
        const c64* ak = A + k * nb2;
        const c64* bk = B + k * nb2;
        double per_k = 0.0;
        for (std::size_t i = 0; i < nb2; ++i) per_k += static_cast<double>(std::norm(ak[i] - bk[i]));
        total += static_cast<double>(weights[k]) * per_k;
    }
    return static_cast<f32>(std::sqrt(total / std::max(static_cast<double>(weight_sum), 1.0e-30)));
}

// Diagonalize F, find μ (with optional level shift), build P = V diag(f) V†.
// Output: P_new (overwritten), mu (returned).
// When ``block_sizes`` is non-empty AND F is block-diagonal in the same blocks,
// uses per-block eigh for a sizeable speedup (matches DM solver behavior).
inline f32 density_from_fock(const c64* F, std::size_t nk1, std::size_t nk2,
                              std::size_t nb,
                              const f32* weights,
                              f32 n_e, f32 T, f32 level_shift,
                              c64* P_new,
                              const std::vector<std::size_t>& block_sizes = {}) {
    const std::size_t nk = nk1 * nk2;
    const std::size_t nb2 = nb * nb;
    std::vector<f32> eps(nk * nb);
    std::vector<c64> V(nk * nb2);
    bool used_blocks = false;
    if (!block_sizes.empty()) {
        std::size_t total = 0;
        for (std::size_t s : block_sizes) total += s;
        if (total == nb && block_sizes.size() > 1) {
            // Quick off-block test: if F has off-block entries above tol, fall
            // back to the dense eigh.
            const f32 scale = max_abs(F, nk * nb2);
            const f32 off = max_offblock_sizes(F, nk, nb, block_sizes);
            const f32 tol = 1.0e-10 + 1.0e-10 * scale;
            if (off <= tol) {
                eigh_block_sizes_batched(F, eps.data(), V.data(),
                                          nk1, nk2, nb, block_sizes, false);
                used_blocks = true;
            }
        }
    }
    if (!used_blocks)
        eigh_batched(F, eps.data(), V.data(), nk1, nk2, nb);

    f32 mu = find_mu_bisection(eps.data(), nk * nb, weights, nk, nb, n_e, T);
    if (level_shift != 0.0) {
        for (std::size_t i = 0; i < nk * nb; ++i) {
            const f32 occ_raw = fermidirac_scalar(eps[i] - mu, T);
            eps[i] += level_shift * (1.0 - occ_raw);
        }
        mu = find_mu_bisection(eps.data(), nk * nb, weights, nk, nb, n_e, T);
    }

    // P = V diag(f) V†, hermitize
    std::vector<f32> occ(nk * nb);
    for (std::size_t i = 0; i < nk * nb; ++i)
        occ[i] = fermidirac_scalar(eps[i] - mu, T);

    parallel_for(nk, [&](std::size_t k) {
        ConstMapMatXcf Vk(V.data() + k * nb2, nb, nb);
        Eigen::Map<Eigen::Array<f32, Eigen::Dynamic, 1>> ok(occ.data() + k * nb, nb);
        MatXcf Vo = Vk;
        for (std::size_t j = 0; j < nb; ++j) Vo.col(j) *= ok[j];
        MatXcf P = Vo * Vk.adjoint();
        // hermitize one slice
        MatXcf Ph = 0.5 * (P + P.adjoint());
        std::memcpy(P_new + k * nb2, Ph.data(), nb2 * sizeof(c64));
    });
    return mu;
}

inline SCFResult solve_scf(const HFKernel& K, const c64* P0, f32 n_e,
                            const SCFConfig& cfg,
                            const ProjectFn* project_fn = nullptr) {
    SCFResult out;
    const std::size_t nk = K.nk();
    const std::size_t nb = K.nb;
    const std::size_t nb2 = K.nb2();
    const std::size_t n_tot = K.n_dense();

    out.hist_E.assign(cfg.max_iter, 0.0);
    out.hist_free.assign(cfg.max_iter, 0.0);
    out.hist_density.assign(cfg.max_iter, 0.0);
    out.hist_comm.assign(cfg.max_iter, 0.0);

    std::vector<c64> density(P0, P0 + n_tot);
    std::vector<c64> Sigma(n_tot), F(n_tot), P_new(n_tot);
    std::vector<f32> hartree_diag(nb, 0.0);
    std::vector<c64> delta(n_tot), comm(n_tot);
    std::vector<f32> density_eigs(nk * nb);
    std::vector<c64> density_eigvecs(n_tot);

    bool converged = false;
    std::size_t k = 0;

    auto density_free_energy = [&](f32 E_in, const c64* P_in) {
        eigh_batched(
            P_in, density_eigs.data(), density_eigvecs.data(),
            K.nk1, K.nk2, nb);
        return free_energy_value(
            E_in, density_eigs.data(), K.w2d, nk, nb, K.T);
    };

    auto build_and_occupy = [&](std::vector<c64>& d, c64* P_out,
                                f32& E_out, f32& Omega_out, f32& mu_out) {
        // Project + hermitize before Fock build
        if (project_fn && *project_fn) {
            (*project_fn)(d.data(), K.nk1, K.nk2, K.nb);
        }
        hermitize_inplace(d.data(), nk, nb);
        build_fock_compact(K, d.data(), Sigma.data(), F.data(),
                           hartree_diag.data(), project_fn);
        E_out = hf_energy_with_hartree_diag(K, d.data(), Sigma.data(),
                                            hartree_diag.data());
        Omega_out = density_free_energy(E_out, d.data());
        mu_out = density_from_fock(F.data(), K.nk1, K.nk2, K.nb,
                                   K.w2d, n_e, K.T, cfg.level_shift,
                                   P_out, cfg.block_sizes);
        if (project_fn && *project_fn) {
            (*project_fn)(P_out, K.nk1, K.nk2, K.nb);
        }
        hermitize_inplace(P_out, nk, nb);
    };

    // ---- Acceleration-scheme dispatch and bookkeeping ---------------------
    const bool use_diis = (cfg.acceleration == "diis");
    const bool use_oda  = (cfg.acceleration == "oda");
    if (cfg.acceleration != "linear" && !use_diis && !use_oda) {
        throw std::invalid_argument(
            "SCFConfig.acceleration must be 'linear', 'diis', or 'oda'");
    }
    if (use_diis && cfg.diis_size == 0) {
        throw std::invalid_argument(
            "SCFConfig.diis_size must be positive when acceleration='diis'");
    }
    // DIIS history (commutator C-DIIS). Stores Fock matrices and commutator
    // residuals; extrapolated F is diagonalized to give next density.
    std::vector<std::vector<c64>> diis_F, diis_R;
    // Stored residuals are immutable. Reuse their weighted overlaps until
    // eviction, so each iteration contracts only the newly added residual.
    std::vector<std::vector<f32>> diis_overlap;
    if (use_diis) {
        diis_F.reserve(cfg.diis_size);
        diis_R.reserve(cfg.diis_size);
    }
    // ODA needs to evaluate energy along (1−λ)P + λP_new. For HF this is
    // quadratic in λ — we compute c1, c2 analytically using one extra Fock
    // build at P_new (cost equal to one normal SCF iter).
    std::vector<c64> Sigma_new(use_oda ? n_tot : 0);
    std::vector<f32> hartree_diag_new(use_oda ? nb : 0);
    std::vector<c64> F_new(use_oda ? n_tot : 0);
    std::vector<c64> P_oda(use_oda ? n_tot : 0);

    while (k < cfg.max_iter && !converged) {
        f32 E_iter = 0.0, Omega_iter = 0.0, mu_iter = 0.0;
        build_and_occupy(
            density, P_new.data(), E_iter, Omega_iter, mu_iter);

        parallel_for(nk, [&](std::size_t kk) {
            const std::size_t off = kk * nb2;
            for (std::size_t i = 0; i < nb2; ++i)
                delta[off + i] = P_new[off + i] - density[off + i];
        });

        // commutator F P − P F
        parallel_for(nk, [&](std::size_t kk) {
            ConstMapMatXcf Fk(F.data() + kk * nb2, nb, nb);
            ConstMapMatXcf Pk(density.data() + kk * nb2, nb, nb);
            MatXcf C = Fk * Pk - Pk * Fk;
            std::memcpy(comm.data() + kk * nb2, C.data(), nb2 * sizeof(c64));
        });

        const f32 d_res = weighted_matrix_norm(delta.data(), K.w2d,
                                                nk, nb, K.weight_sum);
        const f32 c_res = weighted_matrix_norm(comm.data(), K.w2d,
                                                nk, nb, K.weight_sum);
        converged = (d_res <= cfg.density_tol) && (c_res <= cfg.comm_tol);

        out.hist_E[k] = E_iter;
        out.hist_free[k] = Omega_iter;
        out.hist_density[k] = d_res;
        out.hist_comm[k] = c_res;
        ++k;
        // Keep the density whose residuals were checked.  An undamped final
        // fixed-point step can amplify even a sub-tolerance residual in a
        // stiff problem, invalidating the convergence certificate.
        if (converged) break;

        // Snapshot density before the update so we can apply a trust-radius
        // clip on the proposed step at the end of the iteration block.
        std::vector<c64> density_pre;
        if (cfg.trust_radius > 0.0) {
            density_pre = density;
        }

        if (use_diis) {
            // ---- C-DIIS: extrapolate Fock from history --------------------
            // Add current (F, R) to history. R = F P − P F at this iter.
            if (diis_F.size() >= cfg.diis_size) {
                diis_F.erase(diis_F.begin());
                diis_R.erase(diis_R.begin());
                diis_overlap.erase(diis_overlap.begin());
                for (auto& row : diis_overlap) row.erase(row.begin());
            }
            diis_F.emplace_back(F.begin(), F.end());
            diis_R.emplace_back(comm.begin(), comm.end());
            const std::size_t m = diis_F.size();
            for (auto& row : diis_overlap) row.resize(m);
            diis_overlap.emplace_back(m, 0.0);
            for (std::size_t i = 0; i < m; ++i) {
                double total = 0.0;
                const c64* Ri = diis_R[i].data();
                const c64* Rj = diis_R.back().data();
                for (std::size_t kk = 0; kk < nk; ++kk) {
                    double per_k = 0.0;
                    const c64* rik = Ri + kk * nb2;
                    const c64* rjk = Rj + kk * nb2;
                    for (std::size_t a = 0; a < nb2; ++a)
                        per_k += static_cast<double>((std::conj(rik[a]) * rjk[a]).real());
                    total += static_cast<double>(K.w2d[kk]) * per_k;
                }
                diis_overlap[i][m-1] = static_cast<f32>(total);
                diis_overlap[m-1][i] = static_cast<f32>(total);
            }
            if (m >= std::max<std::size_t>(2, cfg.diis_start)) {
                // Build B[i,j] = Re Σ_k w_k tr(R_i_k^† R_j_k), augmented with
                // Lagrange multiplier:
                //   [B  -1] [c]      [0]
                //   [-1  0] [λ]  =   [-1]
                Eigen::Matrix<f32, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
                    B(m + 1, m + 1);
                Eigen::Matrix<f32, Eigen::Dynamic, 1> rhs(m + 1);
                B.setZero(); rhs.setZero();
                for (std::size_t i = 0; i < m; ++i) {
                    for (std::size_t j = i; j < m; ++j) {
                        B(i, j) = diis_overlap[i][j];
                        B(j, i) = B(i, j);
                    }
                    B(i, m) = -1.0;
                    B(m, i) = -1.0;
                }
                B(m, m) = 0.0;
                rhs[m] = -1.0;
                // Normalise the residual overlaps by their largest entry (the
                // Pulay solution is scale invariant) and solve with a
                // partial-pivot LU, as jax_hf's _diis_coefficients does.  A
                // numerically rank-deficient B is the normal case, not a
                // degenerate one: when the iteration is dominated by a single
                // slow mode the successive commutators are nearly parallel, the
                // augmented system is ill-conditioned but solvable, and the
                // large-coefficient extrapolation it yields is exactly what
                // removes that mode.  Gating on FullPivLU::isInvertible()
                // rejected those solves and left the run in plain linear mixing
                // (bilayer PM +0.05: 11 -> 100+ iterations once the symmetry
                // projector stopped supplying rounding noise to B).  Only a
                // non-finite solution now falls back to linear mixing.
                bool diis_ok = true;
                {
                    f32 bmax = 0.0;
                    for (std::size_t i = 0; i < m; ++i)
                        for (std::size_t j = 0; j < m; ++j)
                            bmax = std::max(bmax, std::abs(B(i, j)));
                    // A commutator residual already at tolerance carries no
                    // information to extrapolate on: diagonal F and P commute
                    // exactly while the occupations are still wrong, and the
                    // normalised B would then be pure rounding noise.  Mix
                    // linearly instead of fitting DIIS coefficients to it.
                    if (!(bmax > 0.0) || c_res <= cfg.comm_tol) diis_ok = false;
                    else B.topLeftCorner(m, m) /= bmax;
                }
                Eigen::Matrix<f32, Eigen::Dynamic, 1> sol;
                if (diis_ok) {
                    Eigen::PartialPivLU<decltype(B)> lu(B);
                    sol = lu.solve(rhs);
                    for (std::size_t i = 0; i < m; ++i)
                        diis_ok = diis_ok && std::isfinite(static_cast<double>(sol[i]));
                }
                if (diis_ok) {
                    // F_extrap = Σ_i c_i F_i. Build, hermitize, diagonalize.
                    std::vector<c64> F_extrap(n_tot, c64(0.0, 0.0));
                    for (std::size_t i = 0; i < m; ++i) {
                        const c64 ci(sol[i], 0.0);
                        const c64* Fi = diis_F[i].data();
                        for (std::size_t j = 0; j < n_tot; ++j)
                            F_extrap[j] += ci * Fi[j];
                    }
                    // Optional damping: blend extrapolated Fock with current
                    // iter's Fock.  Suppresses late-iteration oscillation when
                    // the residual has plateaued.
                    const f32 damp = std::clamp(cfg.diis_damping,
                                                 static_cast<f32>(0.0),
                                                 static_cast<f32>(1.0));
                    if (damp < 1.0) {
                        const c64 cd(damp, 0.0), cm1(1.0 - damp, 0.0);
                        const c64* Fcur = F.data();
                        for (std::size_t j = 0; j < n_tot; ++j)
                            F_extrap[j] = cd * F_extrap[j] + cm1 * Fcur[j];
                    }
                    hermitize_inplace(F_extrap.data(), nk, nb);
                    std::vector<c64> P_diis(n_tot);
                    density_from_fock(F_extrap.data(), K.nk1, K.nk2, K.nb,
                                       K.w2d, n_e, K.T, cfg.level_shift,
                                       P_diis.data(), cfg.block_sizes);
                    if (project_fn && *project_fn)
                        (*project_fn)(P_diis.data(), K.nk1, K.nk2, K.nb);
                    hermitize_inplace(P_diis.data(), nk, nb);
                    density = std::move(P_diis);
                } else {
                    // Fallback to linear mix when DIIS solve degenerate.
                    for (std::size_t i = 0; i < n_tot; ++i)
                        density[i] = density[i] + cfg.mixing * delta[i];
                    if (project_fn && *project_fn)
                        (*project_fn)(density.data(), K.nk1, K.nk2, K.nb);
                    hermitize_inplace(density.data(), nk, nb);
                }
            } else {
                // Not enough history yet — plain linear mix.
                for (std::size_t i = 0; i < n_tot; ++i)
                    density[i] = density[i] + cfg.mixing * delta[i];
                if (project_fn && *project_fn)
                    (*project_fn)(density.data(), K.nk1, K.nk2, K.nb);
                hermitize_inplace(density.data(), nk, nb);
            }
        } else if (use_oda) {
            // ---- ODA: optimal damping along (1−λ)P + λP_new --------------
            // For HF, E(P(λ)) = c0 + c1 λ + c2 λ². Compute c1, c2 analytically
            // using the existing E_iter at P_old and one Fock build at P_new.
            // Saves one search but costs an extra Fock build per iter.
            build_fock_compact(K, P_new.data(), Sigma_new.data(), F_new.data(),
                                hartree_diag_new.data(), project_fn);
            const f32 E_new = hf_energy_with_hartree_diag(K, P_new.data(),
                                                           Sigma_new.data(),
                                                           hartree_diag_new.data());
            // c2 = ½ ⟨Δn, HH Δn⟩ + ½ Σ_k w_k Re tr(ΔP_k Σ_Δ_k)  (Hartree quadratic
            // + Fock quadratic).  Easier: c2 = (E_new − E_iter − c1)/1 with
            // λ=1 endpoint, since c0+c1+c2 = E_new and c0 = E_iter.  We need
            // c1 = dE/dλ|_{λ=0} = Tr(F_iter · ΔP) summed weighted (i.e. the
            // gradient of E at P_old in direction ΔP).
            double c1_d = 0.0;
            for (std::size_t kk = 0; kk < nk; ++kk) {
                const c64* Fk = F.data() + kk * nb2;
                const c64* dk = delta.data() + kk * nb2;
                const f32 wk = K.w2d[kk];
                double per_k = 0.0;
                for (std::size_t a = 0; a < nb; ++a) {
                    for (std::size_t b = 0; b < nb; ++b)
                        per_k += static_cast<double>(
                            (Fk[a * nb + b] * dk[b * nb + a]).real());
                }
                c1_d += static_cast<double>(wk) * per_k;
            }
            const f32 c1 = static_cast<f32>(c1_d);
            const f32 c2 = (E_new - E_iter) - c1;
            f32 lam = 1.0;
            if (K.T > 0.0) {
                // At finite temperature the minimized objective is E-T*S.
                // E is exactly quadratic on this segment; evaluate only the
                // density-matrix entropy for each scalar backtracking trial.
                bool accepted = false;
                for (std::size_t bt = 0; bt < 12; ++bt) {
                    for (std::size_t i = 0; i < n_tot; ++i)
                        P_oda[i] = density[i] + c64(lam, 0.0) * delta[i];
                    hermitize_inplace(P_oda.data(), nk, nb);
                    const f32 E_lam =
                        E_iter + lam * c1 + lam * lam * c2;
                    const f32 Omega_lam =
                        density_free_energy(E_lam, P_oda.data());
                    if (Omega_lam < Omega_iter) {
                        accepted = true;
                        break;
                    }
                    lam *= 0.5;
                }
                if (!accepted) lam = 0.0;
            } else if (c2 > 1.0e-12) {
                lam = std::clamp(-c1 / (2.0 * c2), 0.0, 1.0);
            } else if (c1 < 0.0) {
                lam = 1.0;  // downhill, no curvature info — take full step
            } else {
                lam = 0.0;  // c1 ≥ 0 and c2 ≤ 0: don't move
            }
            for (std::size_t i = 0; i < n_tot; ++i)
                density[i] = density[i] + c64(lam, 0.0) * delta[i];
            if (project_fn && *project_fn)
                (*project_fn)(density.data(), K.nk1, K.nk2, K.nb);
            hermitize_inplace(density.data(), nk, nb);
        } else {
            // Linear mixing: density ← (1-mixing) density + mixing P_new.
            for (std::size_t i = 0; i < n_tot; ++i)
                density[i] = density[i] + cfg.mixing * delta[i];
            if (project_fn && *project_fn) {
                (*project_fn)(density.data(), K.nk1, K.nk2, K.nb);
            }
            hermitize_inplace(density.data(), nk, nb);
        }

        // Trust-region clip: if the proposed density step exceeds the trust
        // radius, rescale it to fit.  Bounds the per-iter motion to prevent
        // DIIS extrapolations from launching the iteration to unphysical
        // states (e.g. ungated Coulomb at high doping where the model has
        // weak/no minimum and the iteration would otherwise diverge).
        if (cfg.trust_radius > 0.0 && !density_pre.empty()) {
            const f32 step_norm = weighted_difference_norm(
                density.data(), density_pre.data(), K.w2d, nk, nb, K.weight_sum);
            if (step_norm > cfg.trust_radius) {
                const f32 scale = cfg.trust_radius / step_norm;
                for (std::size_t i = 0; i < n_tot; ++i)
                    density[i] = density_pre[i] + c64(scale, 0.0) * (density[i] - density_pre[i]);
                if (project_fn && *project_fn)
                    (*project_fn)(density.data(), K.nk1, K.nk2, K.nb);
                hermitize_inplace(density.data(), nk, nb);
            }
        }
    }

    // Final evaluation at converged density
    f32 E_final = 0.0, Omega_final = 0.0, mu_final = 0.0;
    build_and_occupy(
        density, P_new.data(), E_final, Omega_final, mu_final);

    if (cfg.return_density) out.density = std::move(density);
    if (cfg.return_fock) out.fock = std::move(F);
    out.energy = E_final;
    out.mu = mu_final;
    out.iterations = k;
    out.converged = converged;
    out.hist_E.resize(k);
    out.hist_free.resize(k);
    out.hist_density.resize(k);
    out.hist_comm.resize(k);
    return out;
}

}  // namespace cpp_hf
