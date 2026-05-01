// Reference SCF solver: linear-mixing Roothaan iteration.
#pragma once

#include "cpp_hf/fock.hpp"
#include "cpp_hf/kernel.hpp"
#include "cpp_hf/linalg.hpp"
#include "cpp_hf/utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace cpp_hf {

struct SCFConfig {
    std::size_t max_iter = 200;
    f32 density_tol = 1.0e-7;
    f32 comm_tol = 1.0e-6;
    f32 mixing = 0.5;
    f32 level_shift = 0.0;
};

struct SCFResult {
    std::vector<c64> density;     // (nk1, nk2, nb, nb) -- pre-final-build density
    std::vector<c64> fock;        // F at convergence (rebuilt on final density)
    f32 energy = 0.0;
    f32 mu = 0.0;
    std::size_t iterations = 0;
    bool converged = false;
    std::vector<f32> hist_E;
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

// Diagonalize F, find μ (with optional level shift), build P = V diag(f) V†.
// Output: P_new (overwritten), mu (returned).
inline f32 density_from_fock(const c64* F, std::size_t nk1, std::size_t nk2,
                              std::size_t nb,
                              const f32* weights,
                              f32 n_e, f32 T, f32 level_shift,
                              c64* P_new) {
    const std::size_t nk = nk1 * nk2;
    const std::size_t nb2 = nb * nb;
    std::vector<f32> eps(nk * nb);
    std::vector<c64> V(nk * nb2);
    // hermitize then eigh
    std::vector<c64> Fh(nk * nb2);
    std::memcpy(Fh.data(), F, nk * nb2 * sizeof(c64));
    hermitize_inplace(Fh.data(), nk, nb);
    eigh_batched(Fh.data(), eps.data(), V.data(), nk1, nk2, nb);

    f32 mu_raw = find_mu_bisection(eps.data(), nk * nb, weights, nk, nb, n_e, T);
    std::vector<f32> shifted(nk * nb);
    if (level_shift != 0.0) {
        for (std::size_t i = 0; i < nk * nb; ++i) {
            const f32 occ_raw = fermidirac_scalar(eps[i] - mu_raw, T);
            shifted[i] = eps[i] + level_shift * (1.0 - occ_raw);
        }
    } else {
        std::memcpy(shifted.data(), eps.data(), nk * nb * sizeof(f32));
    }
    f32 mu = find_mu_bisection(shifted.data(), nk * nb, weights, nk, nb, n_e, T);

    // P = V diag(f) V†, hermitize
    std::vector<f32> occ(nk * nb);
    for (std::size_t i = 0; i < nk * nb; ++i)
        occ[i] = fermidirac_scalar(shifted[i] - mu, T);

    for (std::size_t k = 0; k < nk; ++k) {
        ConstMapMatXcf Vk(V.data() + k * nb2, nb, nb);
        Eigen::Map<Eigen::Array<f32, Eigen::Dynamic, 1>> ok(occ.data() + k * nb, nb);
        MatXcf Vo = Vk;
        for (std::size_t j = 0; j < nb; ++j) Vo.col(j) *= ok[j];
        MatXcf P = Vo * Vk.adjoint();
        // hermitize one slice
        MatXcf Ph = 0.5 * (P + P.adjoint());
        std::memcpy(P_new + k * nb2, Ph.data(), nb2 * sizeof(c64));
    }
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

    out.density.resize(n_tot);
    out.fock.resize(n_tot);
    out.hist_E.assign(cfg.max_iter, 0.0);
    out.hist_density.assign(cfg.max_iter, 0.0);
    out.hist_comm.assign(cfg.max_iter, 0.0);

    std::vector<c64> density(P0, P0 + n_tot);
    std::vector<c64> fock(n_tot);
    std::vector<c64> Sigma(n_tot), F(n_tot), P_new(n_tot);
    std::vector<f32> hartree_diag(nb, 0.0);
    std::vector<c64> delta(n_tot), comm(n_tot);

    f32 mu = 0.0;
    f32 E = 0.0;
    bool converged = false;
    std::size_t k = 0;

    auto build_and_occupy = [&](std::vector<c64>& d, c64* P_out, c64* F_out,
                                f32& E_out, f32& mu_out) {
        // Project + hermitize before Fock build
        if (project_fn && *project_fn) {
            (*project_fn)(d.data(), K.nk1, K.nk2, K.nb);
        }
        hermitize_inplace(d.data(), nk, nb);
        build_fock_compact(K, d.data(), Sigma.data(), F.data(),
                           hartree_diag.data(), project_fn);
        E_out = hf_energy_with_hartree_diag(K, d.data(), Sigma.data(),
                                            hartree_diag.data());
        mu_out = density_from_fock(F.data(), K.nk1, K.nk2, K.nb,
                                   K.w2d.data(), n_e, K.T, cfg.level_shift,
                                   P_out);
        if (project_fn && *project_fn) {
            (*project_fn)(P_out, K.nk1, K.nk2, K.nb);
        }
        hermitize_inplace(P_out, nk, nb);
        std::memcpy(F_out, F.data(), n_tot * sizeof(c64));
    };

    while (k < cfg.max_iter && !converged) {
        f32 E_iter = 0.0, mu_iter = 0.0;
        build_and_occupy(density, P_new.data(), fock.data(), E_iter, mu_iter);

        for (std::size_t i = 0; i < n_tot; ++i) delta[i] = P_new[i] - density[i];

        // commutator F P − P F
        for (std::size_t kk = 0; kk < nk; ++kk) {
            ConstMapMatXcf Fk(fock.data() + kk * nb2, nb, nb);
            ConstMapMatXcf Pk(density.data() + kk * nb2, nb, nb);
            MatXcf C = Fk * Pk - Pk * Fk;
            std::memcpy(comm.data() + kk * nb2, C.data(), nb2 * sizeof(c64));
        }

        const f32 d_res = weighted_matrix_norm(delta.data(), K.w2d.data(),
                                                nk, nb, K.weight_sum);
        const f32 c_res = weighted_matrix_norm(comm.data(), K.w2d.data(),
                                                nk, nb, K.weight_sum);
        converged = (d_res <= cfg.density_tol) && (c_res <= cfg.comm_tol);

        if (converged) {
            density = P_new;
        } else {
            // mixed = project+hermitize(density + mixing * delta)
            for (std::size_t i = 0; i < n_tot; ++i)
                density[i] = density[i] + cfg.mixing * delta[i];
            if (project_fn && *project_fn) {
                (*project_fn)(density.data(), K.nk1, K.nk2, K.nb);
            }
            hermitize_inplace(density.data(), nk, nb);
        }

        out.hist_E[k] = E_iter;
        out.hist_density[k] = d_res;
        out.hist_comm[k] = c_res;
        E = E_iter;
        mu = mu_iter;
        ++k;
    }

    // Final evaluation at converged density
    f32 E_final = 0.0, mu_final = 0.0;
    build_and_occupy(density, P_new.data(), fock.data(), E_final, mu_final);

    out.density = density;
    out.fock = fock;
    out.energy = E_final;
    out.mu = mu_final;
    out.iterations = k;
    out.converged = converged;
    out.hist_E.resize(k);
    out.hist_density.resize(k);
    out.hist_comm.resize(k);
    return out;
}

}  // namespace cpp_hf
