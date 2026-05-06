// Newton-on-charge solver for Hartree-only HF problems.
//
// Exploits the rank-N structure of the q=0 Hartree self-consistency: the
// outer fixed-point lives entirely in an nb-dimensional charge subspace,
// while the inner problem at fixed orbital potential is a single non-
// interacting diagonalization.  Outer Newton with the exact Jacobian
//      r'(σ) = I + Π · HH        (nb × nb, PSD, eigenvalues ≥ 1)
// converges quadratically near the fixed point; the linear system is small
// and well-conditioned regardless of HH spectral norm.  Per-iter cost is
// dominated by one Fock build + one eigh, identical to one DIIS-SCF iter.
//
// Hartree-only.  ``include_hartree`` must be true; ``include_exchange`` is
// not supported by this solver — use solve_dm or solve_scf for HF.
#pragma once

#include "cpp_hf/fock.hpp"
#include "cpp_hf/kernel.hpp"
#include "cpp_hf/linalg.hpp"
#include "cpp_hf/solver_scf.hpp"   // density_from_fock helper
#include "cpp_hf/solver_dm.hpp"    // DMResult (reused for output)
#include "cpp_hf/utils.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace cpp_hf {

struct HartreeNewtonConfig {
    std::size_t max_iter = 30;
    f32 tol_E = 1.0e-7;       // |ΔE| convergence
    f32 tol_sigma = 1.0e-7;   // ‖σ_new − σ‖_∞ convergence
    int mu_maxiter = 25;
    f32 level_shift = 0.0;
    std::vector<std::size_t> block_sizes;
    // Damped Newton: try α ∈ {1, s, s², ...} until ‖r(σ + α·δ)‖ < ‖r(σ)‖.
    // backtrack_max=0 disables (always full step).
    std::size_t backtrack_max = 6;
    f32 backtrack_shrink = 0.5;
    // If true, freeze Π at the very first iteration's spectrum (computed at
    // initial σ).  Saves recomputation; OK when the spectrum doesn't change
    // dramatically between iterations.  Default: recompute Π each iter for
    // exact Newton.
    bool fix_pi_at_start = false;
};

namespace newton_internal {

// Compute Π_orb[α, β] = deg · Σ_kn w_k · (−df/dε)(ε_n − μ) · |V[k,α,n]|² · |V[k,β,n]|²
// from a fully-built spectrum (V is the eigenvector matrix per k).  Real,
// symmetric, PSD.  Output: Pi (nb × nb, row-major).
inline void compressibility_orbital(
    const c64* V, const f32* eps, const f32* w2d,
    f32 mu, f32 T, f32 weight_sum,
    std::size_t nk, std::size_t nb, std::size_t nb2,
    f32* Pi /* (nb, nb) */) {
    const f32 Tsafe = std::max(T, static_cast<f32>(1.0e-12));
    std::vector<f32> psi_sq(nb);
    for (std::size_t i = 0; i < nb * nb; ++i) Pi[i] = 0.0;
    for (std::size_t kk = 0; kk < nk; ++kk) {
        const c64* Vk = V + kk * nb2;
        const f32* eb = eps + kk * nb;
        const f32 wk = w2d[kk];
        for (std::size_t n = 0; n < nb; ++n) {
            const f32 occ = fermidirac_scalar(eb[n] - mu, Tsafe);
            const f32 neg_dfde = occ * (1.0 - occ) / Tsafe;
            if (neg_dfde <= 0.0) continue;
            for (std::size_t a = 0; a < nb; ++a) {
                // |V[k, a, n]|^2  (V is row-major (nb, nb), index a row, n col)
                const c64 v = Vk[a * nb + n];
                psi_sq[a] = static_cast<f32>(v.real() * v.real()
                                              + v.imag() * v.imag());
            }
            const f32 prefactor = wk * neg_dfde;
            for (std::size_t a = 0; a < nb; ++a) {
                const f32 pa = prefactor * psi_sq[a];
                for (std::size_t b = 0; b < nb; ++b)
                    Pi[a * nb + b] += pa * psi_sq[b];
            }
        }
    }
}

// Build h_eff = h + diag(V_orb) at every k.  V_orb is length nb.
inline void apply_orbital_shift(const c64* h, const f32* V_orb,
                                 c64* h_eff,
                                 std::size_t nk, std::size_t nb,
                                 std::size_t n_tot) {
    std::memcpy(h_eff, h, n_tot * sizeof(c64));
    const std::size_t nb2 = nb * nb;
    for (std::size_t kk = 0; kk < nk; ++kk) {
        c64* hk = h_eff + kk * nb2;
        for (std::size_t a = 0; a < nb; ++a)
            hk[a * nb + a] += c64(V_orb[a], 0.0);
    }
}

// Per-orbital weight-summed diagonal of P, real part.
//   sigma_orb[α] = Σ_k w_k Re(P[k][α,α])
inline void orbital_diagonal(const c64* P, const f32* w2d,
                              std::size_t nk, std::size_t nb,
                              f32* sigma_orb /* nb */) {
    const std::size_t nb2 = nb * nb;
    for (std::size_t a = 0; a < nb; ++a) sigma_orb[a] = 0.0;
    for (std::size_t kk = 0; kk < nk; ++kk) {
        const c64* Pk = P + kk * nb2;
        const f32 wk = w2d[kk];
        for (std::size_t a = 0; a < nb; ++a)
            sigma_orb[a] += wk * static_cast<f32>(Pk[a * nb + a].real());
    }
}

// Inner solve at fixed orbital shift V_orb: build h_eff, eigh per k, μ-search,
// fill, build P, return P + eigvals/eigvecs/μ.
struct InnerResult {
    std::vector<c64> P;
    std::vector<c64> V_eig;   // eigenvectors (per k)
    std::vector<f32> eps;     // eigenvalues
    f32 mu = 0.0;
    f32 E = 0.0;              // total Hartree energy at P
};

inline InnerResult inner_solve(
    const HFKernel& K, const f32* V_orb, f32 n_e, f32 T,
    const HartreeNewtonConfig& cfg, const ProjectFn* project_fn) {
    const std::size_t nk1 = K.nk1, nk2 = K.nk2, nk = nk1 * nk2;
    const std::size_t nb = K.nb, nb2 = nb * nb;
    const std::size_t n_tot = K.n_dense();

    InnerResult out;
    out.P.assign(n_tot, c64(0.0, 0.0));
    out.V_eig.assign(n_tot, c64(0.0, 0.0));
    out.eps.assign(nk * nb, 0.0);

    std::vector<c64> h_eff(n_tot);
    apply_orbital_shift(K.h.data(), V_orb, h_eff.data(), nk, nb, n_tot);
    hermitize_inplace(h_eff.data(), nk, nb);

    bool used_blocks = false;
    if (!cfg.block_sizes.empty()) {
        std::size_t total = 0;
        for (std::size_t s : cfg.block_sizes) total += s;
        if (total == nb && cfg.block_sizes.size() > 1) {
            const f32 scale = max_abs(h_eff.data(), n_tot);
            const f32 off = max_offblock_sizes(h_eff.data(), nk, nb, cfg.block_sizes);
            const f32 tol = 1.0e-10 + 1.0e-10 * scale;
            if (off <= tol) {
                eigh_block_sizes_batched(h_eff.data(), out.eps.data(),
                                          out.V_eig.data(), nk1, nk2, nb,
                                          cfg.block_sizes, false);
                used_blocks = true;
            }
        }
    }
    if (!used_blocks)
        eigh_batched(h_eff.data(), out.eps.data(), out.V_eig.data(), nk1, nk2, nb);

    out.mu = find_mu_bisection(out.eps.data(), nk * nb, K.w2d.data(),
                                nk, nb, n_e, T);
    if (cfg.level_shift != 0.0) {
        std::vector<f32> shifted(nk * nb);
        for (std::size_t i = 0; i < nk * nb; ++i) {
            const f32 occ_raw = fermidirac_scalar(out.eps[i] - out.mu, T);
            shifted[i] = out.eps[i] + cfg.level_shift * (1.0 - occ_raw);
        }
        out.mu = find_mu_bisection(shifted.data(), nk * nb, K.w2d.data(),
                                    nk, nb, n_e, T);
    }

    // Fill: occ_n = f(ε_n − μ), build P_k = V·diag(occ)·V†
    std::vector<f32> occ(nk * nb);
    for (std::size_t kk = 0; kk < nk; ++kk) {
        for (std::size_t n = 0; n < nb; ++n) {
            const f32 o = fermidirac_scalar(out.eps[kk * nb + n] - out.mu, T);
            occ[kk * nb + n] = o;
        }
    }

    for (std::size_t kk = 0; kk < nk; ++kk) {
        ConstMapMatXcf Vk(out.V_eig.data() + kk * nb2, nb, nb);
        Eigen::Map<const Eigen::Array<f32, Eigen::Dynamic, 1>>
            ok(occ.data() + kk * nb, nb);
        MatXcf Vo = Vk;
        for (std::size_t j = 0; j < nb; ++j) Vo.col(j) *= ok[j];
        MatXcf Pk = Vo * Vk.adjoint();
        MatXcf Pkh = 0.5 * (Pk + Pk.adjoint());
        std::memcpy(out.P.data() + kk * nb2, Pkh.data(), nb2 * sizeof(c64));
    }
    if (project_fn && *project_fn) {
        (*project_fn)(out.P.data(), nk1, nk2, nb);
        hermitize_inplace(out.P.data(), nk, nb);
    }

    std::vector<c64> Sigma(n_tot), F_tmp(n_tot);
    std::vector<f32> hartree_diag(nb, 0.0);
    build_fock_compact(K, out.P.data(), Sigma.data(), F_tmp.data(),
                       hartree_diag.data(), nullptr);
    out.E = hf_energy_with_hartree_diag(K, out.P.data(), Sigma.data(),
                                        hartree_diag.data());
    return out;
}

}  // namespace newton_internal

// Top-level Newton-on-charge Hartree solver.
//
// Initial σ = 0 (no Hartree shift).  Each outer step:
//   1. V_orb = HH · σ                                       (nb-vec)
//   2. inner_solve(h + diag(V_orb))  →  P_new, V_eig, eps, μ
//   3. σ_out_orb = Σ_k w_k Re(diag(P_new))                  (nb-vec)
//   4. residual r = σ − σ_out                               (nb-vec)
//   5. Π_orb = compressibility at current spectrum          (nb × nb)
//   6. Newton: δ = (I + Π · HH)⁻¹ (σ_out − σ)              (nb-vec)
//   7. backtracking line search: pick α so ‖r(σ + α·δ)‖_∞ < ‖r(σ)‖_∞
//   8. σ ← σ + α·δ
inline DMResult solve_hartree_newton(
    const HFKernel& K, const c64* P0, f32 n_e,
    const HartreeNewtonConfig& cfg,
    const ProjectFn* project_fn = nullptr) {
    using namespace newton_internal;

    if (!K.include_hartree) {
        throw std::invalid_argument(
            "solve_hartree_newton requires K.include_hartree=true");
    }
    if (K.include_exchange) {
        throw std::invalid_argument(
            "solve_hartree_newton is Hartree-only; use solve_dm or solve_scf "
            "for problems with exchange.");
    }
    (void)P0;  // initial guess unused — we always start from σ=0 (h_bare).

    const std::size_t nk1 = K.nk1, nk2 = K.nk2;
    const std::size_t nk = nk1 * nk2;
    const std::size_t nb = K.nb, nb2 = nb * nb;
    const f32 T = std::max(K.T, static_cast<f32>(1.0e-12));

    // Reference contribution to σ_out: σ_ref[α] = Σ_k w_k Re(refP[k][α,α]).
    // Subtracted from each σ_out so that σ measures deviation from refP.
    std::vector<f32> sigma_ref(nb, 0.0);
    orbital_diagonal(K.refP.data(), K.w2d.data(), nk, nb, sigma_ref.data());
    // Total-density constraint: σ_α must satisfy Σ_α σ_α = n_e − Σ_α σ_ref_α
    // because the inner μ-search enforces total electron count = n_e and
    // σ ≡ n − n_ref by construction.  We initialize σ uniformly to honour
    // this, and project Newton steps onto the trace-zero subspace so the
    // total stays right throughout the iteration.  Without this, the
    // Jacobian's enormous uniform-mode eigenvalue makes Newton refuse to
    // correct the constant-shift residual and the iteration never converges.
    f32 sigma_total_target = static_cast<f32>(n_e);
    for (std::size_t a = 0; a < nb; ++a) sigma_total_target -= sigma_ref[a];
    const f32 sigma_init_value = sigma_total_target / static_cast<f32>(nb);

    Eigen::Map<const Eigen::Matrix<f32, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        HHmat(K.HH.data(), nb, nb);

    std::vector<f32> sigma(nb, sigma_init_value);  // honour total constraint
    std::vector<f32> V_orb(nb, 0.0);
    std::vector<f32> Pi(nb * nb, 0.0);
    bool pi_built = false;

    DMResult out;
    out.hist_E.assign(cfg.max_iter, 0.0);
    out.hist_grad.assign(cfg.max_iter, 0.0);

    InnerResult inner;
    f32 E_prev = std::numeric_limits<f32>::infinity();
    f32 res_norm = std::numeric_limits<f32>::infinity();
    bool converged = false;
    std::size_t k = 0;

    auto eval_residual = [&](const std::vector<f32>& s,
                             InnerResult& res_out, f32& sigma_out_norm) {
        // V = HH · s
        for (std::size_t a = 0; a < nb; ++a) {
            f32 acc = 0.0;
            for (std::size_t b = 0; b < nb; ++b)
                acc += K.HH[a * nb + b] * s[b];
            V_orb[a] = acc;
        }
        res_out = inner_solve(K, V_orb.data(), n_e, T, cfg, project_fn);
        // σ_out[α] = Σ_k w_k Re(P_new[k][α,α]) − σ_ref[α]
        std::vector<f32> sigma_out(nb, 0.0);
        orbital_diagonal(res_out.P.data(), K.w2d.data(), nk, nb, sigma_out.data());
        for (std::size_t a = 0; a < nb; ++a)
            sigma_out[a] -= sigma_ref[a];
        // residual r = s - σ_out
        f32 rmax = 0.0;
        for (std::size_t a = 0; a < nb; ++a) {
            const f32 ri = s[a] - sigma_out[a];
            rmax = std::max(rmax, std::abs(ri));
        }
        sigma_out_norm = rmax;
        // Return the proposed (σ_out, σ-step direction) via members below.
        // Caller will recompute σ_out as needed; we just want the residual norm.
        return sigma_out;
    };

    while (k < cfg.max_iter) {
        // 1. Inner solve at current σ; compute residual.
        f32 cur_res = 0.0;
        std::vector<f32> sigma_out = eval_residual(sigma, inner, cur_res);
        const f32 dE = std::abs(inner.E - E_prev);

        out.hist_E[k] = inner.E;
        out.hist_grad[k] = cur_res;
        res_norm = cur_res;

        if (cur_res < cfg.tol_sigma && dE < cfg.tol_E) {
            converged = true;
            ++k;
            break;
        }

        // 2. Compressibility (always at iter 0; reuse if fix_pi_at_start).
        if (!pi_built || !cfg.fix_pi_at_start) {
            compressibility_orbital(inner.V_eig.data(), inner.eps.data(),
                                     K.w2d.data(), inner.mu, T, K.weight_sum,
                                     nk, nb, nb2, Pi.data());
            pi_built = true;
        }

        // 3. Newton direction: solve (I + Π·HH) δ = (σ_out − σ).
        Eigen::Map<const Eigen::Matrix<f32, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
            Pim(Pi.data(), nb, nb);
        Eigen::Matrix<f32, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> J =
            Eigen::Matrix<f32, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Identity(nb, nb)
            + Pim * HHmat;
        Eigen::Matrix<f32, Eigen::Dynamic, 1> rhs(nb);
        for (std::size_t a = 0; a < nb; ++a)
            rhs[a] = sigma_out[a] - sigma[a];
        Eigen::FullPivLU<decltype(J)> lu(J);
        Eigen::Matrix<f32, Eigen::Dynamic, 1> delta = lu.solve(rhs);
        bool delta_ok = lu.info() == Eigen::Success;
        for (std::size_t a = 0; a < nb && delta_ok; ++a)
            delta_ok = delta_ok && std::isfinite(static_cast<double>(delta[a]));
        if (!delta_ok) {
            // Fallback: damped fixed-point (use rhs directly with α=0.3).
            for (std::size_t a = 0; a < nb; ++a) delta[a] = 0.3 * rhs[a];
        }
        // Project δ onto the trace-zero subspace so σ_total stays at its
        // initial value (set by the total-density constraint).  Without this,
        // a tiny uniform drift each iter accumulates over many steps.
        f32 delta_mean = 0.0;
        for (std::size_t a = 0; a < nb; ++a) delta_mean += delta[a];
        delta_mean /= static_cast<f32>(nb);
        for (std::size_t a = 0; a < nb; ++a) delta[a] -= delta_mean;

        // 4. Backtracking line search.
        f32 alpha = 1.0;
        bool accepted = false;
        std::vector<f32> sigma_trial(nb);
        InnerResult inner_trial;
        f32 trial_res = 0.0;
        for (std::size_t bt = 0; bt <= cfg.backtrack_max; ++bt) {
            for (std::size_t a = 0; a < nb; ++a)
                sigma_trial[a] = sigma[a] + alpha * delta[a];
            (void)eval_residual(sigma_trial, inner_trial, trial_res);
            if (trial_res < cur_res || cfg.backtrack_max == 0) {
                accepted = true;
                break;
            }
            alpha *= cfg.backtrack_shrink;
        }
        if (!accepted) {
            // Even the smallest step didn't decrease residual — accept it
            // anyway and let the next iter try.
            for (std::size_t a = 0; a < nb; ++a)
                sigma_trial[a] = sigma[a] + alpha * delta[a];
            (void)eval_residual(sigma_trial, inner_trial, trial_res);
        }

        sigma = sigma_trial;
        inner = std::move(inner_trial);
        E_prev = inner.E;
        ++k;
    }

    // Final pass: ensure inner is built at final σ for return.
    if (!converged) {
        f32 dummy = 0.0;
        (void)eval_residual(sigma, inner, dummy);
    }

    out.density = inner.P;
    // Rebuild from the returned density so both the Fock matrix and total
    // energy correspond to out.density, even if Newton stopped unconverged.
    std::vector<c64> Sigma(K.n_dense()), F_eff(K.n_dense());
    std::vector<f32> hartree_diag(nb, 0.0);
    build_fock_compact(K, inner.P.data(), Sigma.data(), F_eff.data(),
                       hartree_diag.data(), nullptr);
    const f32 E_final = hf_energy_with_hartree_diag(K, inner.P.data(),
                                                    Sigma.data(),
                                                    hartree_diag.data());
    out.fock = std::move(F_eff);
    out.mu = inner.mu;
    out.energy = E_final;
    out.n_iter = k;
    out.converged = converged;
    out.hist_E.resize(k);
    out.hist_grad.resize(k);
    // Q and p are not produced by Newton (no orbital-rotation parameterisation);
    // leave them empty.
    return out;
}

}  // namespace cpp_hf
