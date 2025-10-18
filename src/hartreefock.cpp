// src/hartreefock.cpp - implementation of pure C++ Hartree–Fock iteration

#include <vector>
#include <complex>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <span>
#include <mdspan>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "cpp_hf/hartreefock.hpp"
#include "cpp_hf/hf_kernel.hpp"
#include "cpp_hf/mixers.hpp"
#include "cpp_hf/utils.hpp"

namespace hf {

using MatC = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

HFResult hartreefock_iteration(
    const double* W,
    const cxd* H,
    const cxd* V,
    std::size_t nk1, std::size_t nk2, std::size_t d,
    std::size_t dv1, std::size_t dv2,
    const cxd* P0,
    double electron_density0,
    double T,
    std::size_t max_iter,
    double comm_tol,
    std::size_t diis_size,
    double mixing_alpha)
{
    const std::size_t nblocks = nk1*nk2;
    const std::size_t n_tot = nk1*nk2*d*d;
    if (!W) throw std::invalid_argument("weights null");
    if (!H) throw std::invalid_argument("H null");
    if (!V) throw std::invalid_argument("V null");
    if (!P0) throw std::invalid_argument("P0 null");

    std::span<const double> Wspan(W, nblocks);
    std::span<const cxd>    Hspan(H, n_tot);
    std::span<const cxd>    Vspan(V, (dv1==1 && dv2==1) ? nblocks : n_tot);
    std::vector<cxd>        P(P0, P0 + n_tot);

    HFKernel kernel(nk1, nk2, d, Wspan, Hspan, Vspan, dv1, dv2, T, electron_density0);
    DiisState  cdiis(diis_size);
    EdiisState ediis(diis_size);

    double e_fin = 0.0; std::size_t k_fin = 0; double mu_fin = 0.0;

    enum class Phase { EDIIS, CDIIS, BROYDEN };
    Phase last_phase = Phase::EDIIS;
    const std::size_t n_flat = n_tot;
    BroydenState bro_state(diis_size, n_flat);

    const double to_cdiis   = 9.0 * comm_tol;
    const double to_broyden = 1.5 * comm_tol;
    const double cdiis_blend_keep = 0.5, cdiis_blend_new = 0.5;

    for (std::size_t k=0; k<max_iter; ++k) {
        // 1) Diagonalize F[P] to build P_new and compute mu
        auto call_result = kernel.call(P);
        std::vector<cxd> P_new = std::move(call_result.first);
        const double mu = call_result.second;

        // 2) Build F[P_new] and energy once; also cache EVD(F[P_new]) for preconditioner
        std::vector<cxd> F_new;
        double e_new = 0.0;
        kernel.fock_energy_and_cache_evd(P_new, F_new, e_new);

        // 3) Commutator residual per k, and weighted RMS
        std::vector<cxd> comm(P_new.size());
        double sum_w_c2 = 0.0;

        using ext2 = std::extents<std::size_t, std::dynamic_extent, std::dynamic_extent>;
        auto Wv = std::mdspan<const double, ext2, std::layout_right>(kernel.weights.data(), nk1, nk2);

#ifdef _OPENMP
        #pragma omp parallel for collapse(2) reduction(+:sum_w_c2) schedule(static)
#endif
        for (long long k1i=0;k1i<(long long)nk1;++k1i)
            for (long long k2i=0;k2i<(long long)nk2;++k2i) {
                const std::size_t base = ::offset(nk2,d,(std::size_t)k1i,(std::size_t)k2i,0,0);
                Eigen::Map<const MatC> Fk(&F_new[base], d, d);
                Eigen::Map<const MatC> Pk(&P_new[base], d, d);
                Eigen::Map<      MatC> C (&comm [base], d, d);
                C.noalias() = Fk * Pk - Pk * Fk;
                const double wk = Wv.data_handle()[ Wv.mapping()((std::size_t)k1i,(std::size_t)k2i) ];
                sum_w_c2 += wk * C.squaredNorm();
            }

        const double comm_rms = std::sqrt(sum_w_c2 / std::max(1e-30, kernel.weight_sum));

        if (comm_rms < comm_tol) {
            P.swap(P_new); e_fin = e_new; k_fin = k; mu_fin = mu;
            break;
        }

        // 4) Mixer schedule with CDIIS in the middle
        const Phase phase_now = (comm_rms > to_cdiis) ? Phase::EDIIS
                               : (comm_rms > to_broyden) ? Phase::CDIIS
                               : Phase::BROYDEN;
        const bool switched = (phase_now != last_phase);

        std::vector<cxd> P_mix;

        if (phase_now == Phase::EDIIS) {
            auto ediis_result = ediis.update(P_new, F_new, e_new,
                                             kernel.weights, nk1, nk2, d,
                                             /*max_iter_qp=*/20, /*pg_tol=*/1e-7);
            P_mix = std::move(ediis_result.first);
        }
        else if (phase_now == Phase::CDIIS) {
            P_mix = cdiis.update_cdiis(P_new, comm, P,
                                       /*coeff_cap=*/5.0, /*eps_reg=*/1e-12,
                                       /*blend_keep=*/cdiis_blend_keep, /*blend_new=*/cdiis_blend_new);
        }
        else { // Phase::BROYDEN
            // Precondition C with cached eigen-decomposition of F_new
            std::vector<cxd> comm_pc;
            kernel.precondition_commutator_cached(F_new, comm, comm_pc, 5.0e-3);

            if (switched) bro_state.reset();
            const std::size_t bro_count_before = bro_state.count;

            auto upd = bro_state.update(P_new, comm_pc, mixing_alpha);
            bro_state = std::move(upd.first);
            std::vector<cxd> Praw = std::move(upd.second); // flat

            if (bro_count_before == 0) {
                const double beta = 0.35;
                P_mix.resize(P.size());
                Eigen::Map<      Eigen::ArrayXcd> Pm(P_mix.data(), (Eigen::Index)P_mix.size());
                Eigen::Map<const Eigen::ArrayXcd> Pc(P.data(),     (Eigen::Index)P.size());
                Eigen::Map<const Eigen::ArrayXcd> Rc(comm_pc.data(), (Eigen::Index)comm_pc.size());
                Pm = Pc - beta * Rc;

                const double w_keep = 0.7, w_new = 0.3;
                Pm = w_keep*Pc + w_new*Pm;
            } else {
                P_mix = std::move(Praw);
            }
        }

        last_phase = phase_now;
        P = std::move(P_mix);
        e_fin = e_new; k_fin = k; mu_fin = mu;
    }

    // Final Fock for output (single Σ)
    std::vector<cxd> F_fin;
    kernel.fock_of(P, F_fin);

    // Final μ from P (consistent)
    {
        std::vector<std::vector<double>> bands_final(nk1 * nk2);
        for_k(nk1, nk2, [&](std::size_t k1, std::size_t k2){
            const std::size_t base = offset(nk2, d, k1, k2, 0, 0);
            Eigen::Map<const MatC> Fk(&F_fin[base], d, d);
            Eigen::SelfAdjointEigenSolver<MatC> es;
            es.compute(Fk, Eigen::ComputeEigenvectors);
            if (es.info() != Eigen::Success) throw std::runtime_error("EVD failed (final mu)");
            const auto& ev = es.eigenvalues();
            bands_final[k1 * nk2 + k2] = std::vector<double>(ev.data(), ev.data() + d);
        });
        mu_fin = find_chemicalpotential(bands_final, kernel.weights, kernel.nk1, kernel.nk2, kernel.d, T, electron_density0);
    }

    HFResult out;
    out.P = std::move(P);
    out.F = std::move(F_fin);
    out.energy = e_fin;
    out.mu = mu_fin;
    out.iters = k_fin;
    return out;
}

} // namespace hf
