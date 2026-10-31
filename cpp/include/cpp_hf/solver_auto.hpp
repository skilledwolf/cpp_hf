// Safeguarded full-residual DIIS, with coupled Newton recovery on stagnation.
#pragma once
#include "cpp_hf/solver_anderson.hpp"
#include "cpp_hf/solver_rtr.hpp"

namespace cpp_hf {
inline DMResult solve_auto(const HFKernel& K, const c64* P0, f32 n_e,
                           const SolverConfig& cfg, const ProjectFn* project_fn = nullptr) {
    SCFConfig scf;
    scf.acceleration = "rdiis";
    // Arbitrary symmetry callbacks need not preserve the orbital chart. Keep
    // their optimization in the projected density representation.
    scf.max_iter = project_fn && *project_fn ? cfg.max_iter
                                            : std::max<std::size_t>(1, cfg.max_iter - cfg.max_iter/5);
    scf.density_tol = scf.comm_tol = cfg.tol_grad > 0.0 ? cfg.tol_grad : 1e-6;
    scf.diis_size = 8;
    scf.mixing = 1.0;
    scf.block_sizes = cfg.block_sizes;
    auto initial = solve_anderson(K, P0, n_e, scf, project_fn);
    std::vector<f32> gradients(initial.hist_density.size());
    for (std::size_t i = 0; i < gradients.size(); ++i)
        gradients[i] = std::max(initial.hist_density[i], initial.hist_comm[i]);
    if (!initial.converged && initial.iterations < cfg.max_iter
        && !(project_fn && *project_fn)) {
        auto polish = cfg;
        polish.optimizer = 1;
        polish.preserve_initial_density = true;
        polish.max_iter -= initial.iterations;
        polish.tol_grad = scf.density_tol;
        auto result = solve_rtr(K, initial.density.data(), n_e, polish, project_fn);
        result.hist_E.insert(result.hist_E.begin(), initial.hist_E.begin(), initial.hist_E.end());
        result.hist_free.insert(result.hist_free.begin(), initial.hist_free.begin(), initial.hist_free.end());
        result.hist_grad.insert(result.hist_grad.begin(), gradients.begin(), gradients.end());
        result.n_iter += initial.iterations;
        return result;
    }
    DMResult result;
    result.Q.resize(K.n_dense()); result.p.resize(K.nk()*K.nb);
    dm_internal::eigh_checked(initial.density.data(), result.p.data(), result.Q.data(),
                              K.nk1,K.nk2,K.nb,cfg.block_sizes);
    if (!cfg.return_Q) result.Q.clear();
    if (cfg.return_density) result.density = std::move(initial.density);
    if (cfg.return_fock) result.fock = std::move(initial.fock);
    result.mu = initial.mu; result.energy = initial.energy;
    result.n_iter = initial.iterations; result.converged = initial.converged;
    result.hist_E = std::move(initial.hist_E); result.hist_free = std::move(initial.hist_free);
    result.hist_grad = std::move(gradients);
    return result;
}
} // namespace cpp_hf
