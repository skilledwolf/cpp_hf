// Safeguarded, full-density-residual Anderson acceleration at fixed particle count.
#pragma once

#include "cpp_hf/solver_scf.hpp"
#include "cpp_hf/response.hpp"
#include <limits>

namespace cpp_hf {
namespace anderson_internal {

struct State {
    std::vector<c64> P, F, D, R, Q;
    std::vector<f32> eps, p;
    f32 E = 0.0, S = 0.0, mu = 0.0, density_res = 0.0, comm_res = 0.0;
    State(std::size_t nk, std::size_t nb)
        : P(nk*nb*nb), F(P.size()), D(P.size()), R(P.size()), Q(P.size()),
          eps(nk*nb), p(nk*nb) {}
    f32 residual() const { return std::max(density_res, comm_res); }
};

// A QR solve on residual differences avoids squaring the condition number in
// the Pulay Gram matrix.  The affine coefficients sum to one by construction;
// nearly dependent columns are removed by rank-revealing pivoting.
inline std::vector<f32> coefficients(const std::vector<std::vector<c64>>& history,
                                     const HFKernel& K) {
    const std::size_t m = history.size(), n = K.n_dense(), nb2 = K.nb2();
    std::vector<f32> c(m, 0.0);
    c.back() = 1.0;
    if (m < 2) return c;
    Eigen::MatrixXd A(2*n, m-1);
    Eigen::VectorXd b(2*n);
    for (std::size_t i = 0; i < n; ++i) {
        const f32 weight = std::sqrt(K.w2d[i / nb2] / K.weight_sum);
        b[2*i] = -weight * history.back()[i].real();
        b[2*i+1] = -weight * history.back()[i].imag();
        for (std::size_t j = 0; j < m-1; ++j) {
            const c64 d = weight * (history[j][i] - history.back()[i]);
            A(2*i, j) = d.real(); A(2*i+1, j) = d.imag();
        }
    }
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(A);
    qr.setThreshold(1e-10);
    const Eigen::VectorXd x = qr.solve(b);
    if (!x.allFinite()) return c;
    for (std::size_t j = 0; j < m-1; ++j) { c[j] = x[j]; c.back() -= x[j]; }
    return c;
}

inline void checked_eigh(const c64* A, f32* eigs, c64* Q, const HFKernel& K,
                          const std::vector<std::size_t>& blocks) {
    if (!blocks.empty() && dm_internal::block_structure_ok(A, K.nk(), K.nb, blocks))
        eigh_block_sizes_batched(A, eigs, Q, K.nk1, K.nk2, K.nb, blocks, false);
    else
        eigh_batched(A, eigs, Q, K.nk1, K.nk2, K.nb);
}

} // namespace anderson_internal

inline SCFResult solve_anderson(const HFKernel& K, const c64* P0, f32 n_e,
                                const SCFConfig& cfg, const ProjectFn* project_fn = nullptr) {
    using namespace anderson_internal;
    using namespace dm_internal;
    const auto nk = K.nk(), nb = K.nb, nb2 = nb*nb, n = K.n_dense();
    State current(nk, nb), candidate(nk, nb), trial(nk, nb);
    std::vector<c64> Sigma(n), vectors(n), preconditioned(n);
    std::vector<f32> hartree(nb), spectrum(nk*nb), w_norm(nk);
    for (std::size_t k = 0; k < nk; ++k) w_norm[k] = K.w2d[k] / K.weight_sum;
    const f32 temp = std::max(K.T, 1e-12);
    const f32 normalized_count = n_e / K.weight_sum;
    const bool use_prec = cfg.preconditioner == "hartree";
    const HartreeResponsePreconditioner prec(K, use_prec);
    std::vector<std::vector<c64>> history_R, history_update;
    SCFResult out;
    std::size_t rejected = 0;

    auto project = [&](c64* P) {
        if (project_fn && *project_fn) (*project_fn)(P, K.nk1, K.nk2, nb);
        hermitize_inplace(P, nk, nb);
    };
    auto entropy = [&](State& state, bool repair) {
        checked_eigh(state.P.data(), spectrum.data(), vectors.data(), K, cfg.block_sizes);
        double count = 0.0;
        bool bounded = true;
        for (std::size_t k = 0; k < nk; ++k)
            for (std::size_t i = 0; i < nb; ++i) {
                const f32 pi = spectrum[k*nb+i];
                count += w_norm[k] * pi;
                bounded = bounded && pi >= -1e-12 && pi <= 1.0+1e-12;
            }
        if (repair && (!bounded || std::abs(count-normalized_count) > 1e-12)) {
            project_capped_occupations(spectrum.data(), w_norm.data(), nk, nb, normalized_count);
            density_from_Qp(vectors.data(), spectrum.data(), state.P.data(), nk, nb, cfg.block_sizes);
            project(state.P.data());
            // Entropy belongs to the actual projected density, not the
            // unprojected orbitals that generated it.
            checked_eigh(state.P.data(), spectrum.data(), vectors.data(), K, cfg.block_sizes);
        }
        state.S = occupation_entropy(spectrum.data(), K.w2d, nk, nb);
    };
    auto fixed_point = [&](State& state) {
        checked_eigh(state.F.data(), state.eps.data(), state.Q.data(), K, cfg.block_sizes);
        state.mu = solve_mu_inloop(state.eps.data(), nk*nb, w_norm.data(), nk, nb,
                                  normalized_count, state.mu, temp, 100);
        for (std::size_t i = 0; i < nk*nb; ++i)
            state.p[i] = expit_scalar((state.mu-state.eps[i])/temp);
        density_from_Qp(state.Q.data(), state.p.data(), state.D.data(), nk, nb, cfg.block_sizes);
        project(state.D.data());
        for (std::size_t i = 0; i < n; ++i) state.R[i] = state.D[i]-state.P[i];
        state.density_res = norm_matrix(state.R.data(), w_norm.data(), nk, nb);
        parallel_for(nk, [&](std::size_t k) {
            ConstMapMatXcf F(state.F.data()+k*nb2, nb, nb), P(state.P.data()+k*nb2, nb, nb);
            MapMatXcf C(vectors.data()+k*nb2, nb, nb);
            C = F*P-P*F;
        });
        state.comm_res = norm_matrix(vectors.data(), w_norm.data(), nk, nb);
    };
    auto evaluate = [&](State& state) {
        project(state.P.data());
        entropy(state, true);
        build_fock_compact(K, state.P.data(), Sigma.data(), state.F.data(), hartree.data(), project_fn);
        state.E = hf_energy_with_hartree_diag(K, state.P.data(), Sigma.data(), hartree.data());
        fixed_point(state);
    };
    auto log = [&]() {
        out.hist_E.push_back(current.E);
        out.hist_free.push_back(current.E-K.T*current.S);
        out.hist_density.push_back(current.density_res);
        out.hist_comm.push_back(current.comm_res);
    };
    auto converged = [&]() {
        return current.density_res <= cfg.density_tol && current.comm_res <= cfg.comm_tol;
    };
    // Energy change evaluated from the affine Fock map.  Avoid subtracting
    // two large total energies when resolving a small residual near a minimum.
    auto delta_free = [&](const State& next) {
        long double delta = 0.0;
        for (std::size_t k = 0; k < nk; ++k)
            for (std::size_t i = 0; i < nb2; ++i) {
                const std::size_t a = k*nb2+i;
                delta += static_cast<long double>(K.w2d[k]) *
                    (std::conj(next.P[a]-current.P[a]) * (0.5*(next.F[a]+current.F[a]))).real();
            }
        return static_cast<f32>(delta) - K.T*(next.S-current.S);
    };
    auto acceptable = [&](const State& next) {
        const f32 d = delta_free(next);
        const f32 noise = 32.0*std::numeric_limits<f32>::epsilon()
                        * std::max(std::abs(current.E)+K.T*std::abs(current.S), K.weight_sum);
        return std::isfinite(d) && (d < -noise || (d <= noise && next.residual() < current.residual()));
    };
    // Every endpoint is admissible.  Along a density segment F and the
    // quadratic HF energy interpolate exactly; only entropy/eigensystems
    // need reevaluation.  Rejected scalar trials require no Fock builds.
    auto line_search = [&](State& endpoint) {
        if ((!cfg.safeguard && std::isfinite(endpoint.E)) || acceptable(endpoint)) {
            std::swap(current, endpoint);
            return true;
        }
        double slope = 0.0;
        for (std::size_t k = 0; k < nk; ++k)
            for (std::size_t i = 0; i < nb2; ++i) {
                const std::size_t a = k*nb2+i;
                slope += K.w2d[k]*(std::conj(endpoint.P[a]-current.P[a])*current.F[a]).real();
            }
        const f32 quadratic = endpoint.E-current.E-slope;
        f32 alpha = 0.5;
        for (std::size_t bt = 0; bt < 16; ++bt, alpha *= 0.5) {
            for (std::size_t i = 0; i < n; ++i) {
                trial.P[i] = current.P[i]+alpha*(endpoint.P[i]-current.P[i]);
                trial.F[i] = current.F[i]+alpha*(endpoint.F[i]-current.F[i]);
            }
            trial.E = current.E + alpha*slope + alpha*alpha*quadratic;
            entropy(trial, false);
            fixed_point(trial);
            if (acceptable(trial)) { std::swap(current, trial); return true; }
        }
        return false;
    };

    std::memcpy(current.P.data(), P0, n*sizeof(c64));
    evaluate(current);
    while (out.hist_E.size() < cfg.max_iter) {
        log();
        if (converged() || out.hist_E.size() == cfg.max_iter) break;
        if (use_prec)
            prec.apply(current.R.data(), current.Q.data(), current.eps.data(), current.p.data(), preconditioned.data());
        else
            preconditioned = current.R;
        if (history_R.size() >= cfg.diis_size) {
            history_R.erase(history_R.begin()); history_update.erase(history_update.begin());
        }
        history_R.push_back(preconditioned);
        history_update.emplace_back(n);
        for (std::size_t i = 0; i < n; ++i)
            history_update.back()[i] = cfg.acceleration == "rdiis"
                ? current.F[i] : current.P[i] + cfg.mixing*preconditioned[i];
        auto c = coefficients(history_R, K);
        if (history_R.size() < std::max<std::size_t>(2,cfg.diis_start)) {
            std::fill(c.begin(),c.end(),0.0); c.back()=1.0;
        }
        std::fill(candidate.P.begin(), candidate.P.end(), c64(0.0));
        for (std::size_t j = 0; j < c.size(); ++j)
            for (std::size_t i = 0; i < n; ++i) candidate.P[i] += c[j]*history_update[j][i];
        if (cfg.acceleration == "rdiis") {
            candidate.F = candidate.P;
            for (std::size_t i=0;i<n;++i)
                candidate.F[i] = cfg.diis_damping*candidate.F[i]+(1.0-cfg.diis_damping)*current.F[i];
            fixed_point(candidate);
            candidate.P = candidate.D;
        }
        if (cfg.trust_radius > 0.0) {
            const f32 distance = norm_matrix_difference(candidate.P.data(),current.P.data(),w_norm.data(),nk,nb);
            if (distance > cfg.trust_radius)
                for (std::size_t i=0;i<n;++i)
                    candidate.P[i] = current.P[i]+(cfg.trust_radius/distance)*(candidate.P[i]-current.P[i]);
        }
        evaluate(candidate);
        bool accepted = line_search(candidate);
        if (!accepted) {
            // The Fermi-density direction is a descent direction of the
            // finite-temperature objective.  Clear stale secants on fallback.
            history_R.clear(); history_update.clear();
            candidate.P = current.D;
            evaluate(candidate);
            accepted = line_search(candidate);
        }
        rejected = accepted ? 0 : rejected+1;
        if (rejected >= 2) break;
    }
    // No unchecked last update: report and check the exact returned state.
    evaluate(current); // remove affine-interpolation roundoff before reporting
    out.hist_E.back() = current.E;
    out.hist_free.back() = current.E-K.T*current.S;
    out.hist_density.back() = current.density_res;
    out.hist_comm.back() = current.comm_res;
    out.energy = current.E;
    out.mu = current.mu;
    out.converged = converged();
    out.iterations = out.hist_E.size();
    if (cfg.return_density) out.density = std::move(current.P);
    if (cfg.return_fock) out.fock = std::move(current.F);
    return out;
}
} // namespace cpp_hf
