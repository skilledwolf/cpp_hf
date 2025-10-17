// hf_cpp_module.cpp (multicore-optimized; original mixing kept; 7 fixes applied)
// C++17 + pybind11 + Eigen + FFTW (guru, batched 2D) + optional Boost (toms748)
// Layout: (nk1, nk2, d, d) row-major (C-order)

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include "cpp_hf/mixers.hpp"
#include "cpp_hf/utils.hpp"
#include "cpp_hf/fftw_batched2d.hpp"
#include "cpp_hf/hf_kernel.hpp"

// fftw3.h is included by cpp_hf/fftw_batched2d.hpp

#include <vector>
#include <complex>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <limits>
#include <memory>
#include <mutex>
#include <cmath>

#ifdef _OPENMP
  #include <omp.h>
#endif

// ---- Boost root solver (optional, but recommended) ----
#include <boost/math/tools/toms748_solve.hpp>
#include <boost/math/tools/roots.hpp>

namespace py = pybind11;
using cxd  = std::complex<double>;
using MatC = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Vecd = Eigen::VectorXd;

// ---------------- Python-exposed function ----------------
py::tuple hartreefock_iteration_cpp(
    py::array_t<double, py::array::c_style | py::array::forcecast> weights,   // (nk1,nk2)
    py::array_t<cxd,    py::array::c_style | py::array::forcecast> hamiltonian, // (nk1,nk2,d,d)
    py::array_t<cxd,    py::array::c_style | py::array::forcecast> v_coulomb,   // (nk1,nk2,dv1,dv2)
    py::array_t<cxd,    py::array::c_style | py::array::forcecast> p0,          // (nk1,nk2,d,d)
    double electron_density0,
    double T,
    size_t max_iter,
    double comm_tol,
    size_t diis_size,
    double mixing_alpha
) {
    if (hamiltonian.ndim()!=4) throw std::invalid_argument("H must be (nk1,nk2,d,d)");
    const size_t nk1 = hamiltonian.shape(0), nk2 = hamiltonian.shape(1), d = hamiltonian.shape(2);
    if ((size_t)hamiltonian.shape(3)!=d) throw std::invalid_argument("H last two dims must be equal (d,d)");
    if (weights.ndim()!=2 || (size_t)weights.shape(0)!=nk1 || (size_t)weights.shape(1)!=nk2) throw std::invalid_argument("weights must be (nk1,nk2)");
    if (p0.ndim()!=4 || (size_t)p0.shape(0)!=nk1 || (size_t)p0.shape(1)!=nk2 || (size_t)p0.shape(2)!=d || (size_t)p0.shape(3)!=d)
        throw std::invalid_argument("p0 must be (nk1,nk2,d,d)");
    if (v_coulomb.ndim()!=4 || (size_t)v_coulomb.shape(0)!=nk1 || (size_t)v_coulomb.shape(1)!=nk2)
        throw std::invalid_argument("V must be (nk1,nk2,dv1,dv2)");

    hf::HFKernel kernel(nk1,nk2,d, weights, hamiltonian, v_coulomb, T, electron_density0);

    std::vector<cxd> P(p0.data(), p0.data()+ (nk1*nk2*d*d));
    hf::DiisState  cdiis(diis_size);     // wired in (fix #5)
    hf::EdiisState ediis(diis_size);

    double e_fin = 0.0; size_t k_fin = 0; double mu_fin = 0.0;

    py::gil_scoped_release nogil;

    enum class Phase { EDIIS, CDIIS, BROYDEN };
    Phase last_phase = Phase::EDIIS;
    const size_t n_flat = nk1*nk2*d*d;
    hf::BroydenState bro_state(diis_size, n_flat);

    // Relative thresholds based on target comm_tol; EDIIS -> CDIIS -> Broyden
    // Switch earlier to faster mixers to reduce iteration count
    const double to_cdiis    = 9.0 * comm_tol;
    const double to_broyden  = 1.5 * comm_tol;
    const double cdiis_blend_keep = 0.5, cdiis_blend_new = 0.5; // slightly more aggressive blend in CDIIS

    for (size_t k=0; k<max_iter; ++k) {
        // 1) Diagonalize F[P] to build P_new and compute μ
        auto call_result = kernel.call(P);
        std::vector<cxd> P_new = std::move(call_result.first);
        const double mu = call_result.second;

        // 2) Build F[P_new] and energy once; also cache EVD(F[P_new]) for preconditioner
        std::vector<cxd> F_new;
        double e_new = 0.0;
        kernel.fock_energy_and_cache_evd(P_new, F_new, e_new);

        // 3) Commutator residual per k, and weighted RMS (fix #7)
        std::vector<cxd> comm(P_new.size());
        double sum_w_c2 = 0.0;

#ifdef _OPENMP
        #pragma omp parallel for collapse(2) reduction(+:sum_w_c2) schedule(static)
#endif
        for (long long k1i=0;k1i<(long long)nk1;++k1i)
            for (long long k2i=0;k2i<(long long)nk2;++k2i) {
                const size_t base = offset(nk2,d,(size_t)k1i,(size_t)k2i,0,0);
                Eigen::Map<MatC> Fk(&F_new[base], d, d);
                Eigen::Map<MatC> Pk(&P_new[base], d, d);
                MatC C(d,d);
                C.noalias() = Fk * Pk - Pk * Fk;
                const double wk = kernel.weights[(size_t)k1i*nk2 + (size_t)k2i];
                sum_w_c2 += wk * C.cwiseAbs2().sum();
                Eigen::Map<MatC>(&comm[base], d, d) = C;
            }

        const double comm_rms = std::sqrt(sum_w_c2 / std::max(1e-30, kernel.weight_sum));

        if (comm_rms < comm_tol) {
            P.swap(P_new); e_fin = e_new; k_fin = k; mu_fin = mu;
            break;
        }

        // 4) Mixer schedule with CDIIS in the middle (fix #5)
        Phase phase_now = Phase::BROYDEN;
        if (comm_rms > to_cdiis)        phase_now = Phase::EDIIS;
        else if (comm_rms > to_broyden) phase_now = Phase::CDIIS;

        const bool switched = (phase_now != last_phase);

        std::vector<cxd> P_mix;

        if (phase_now == Phase::EDIIS) {
            auto ediis_result = ediis.update(P_new, F_new, e_new,
                                             kernel.weights, nk1, nk2, d,
                                             /*max_iter_qp=*/20, /*pg_tol=*/1e-7);
            P_mix = std::move(ediis_result.first);
        }
        else if (phase_now == Phase::CDIIS) {
            // CDIIS on commutator with a gentle blend
            P_mix = cdiis.update_cdiis(P_new, comm, P,
                                       /*coeff_cap=*/5.0, /*eps_reg=*/1e-12,
                                       /*blend_keep=*/cdiis_blend_keep, /*blend_new=*/cdiis_blend_new);
        }
        else { // Phase::BROYDEN
            // Precondition C with cached eigen-decomposition of F_new
            std::vector<cxd> comm_pc;
            kernel.precondition_commutator_cached(F_new, comm, comm_pc, 5.0e-3);

            if (switched) bro_state.reset();
            const size_t bro_count_before = bro_state.count;

            // Store / update LBFGS and get quasi-Newton proposal
            std::vector<cxd> Pflat(P_new.begin(), P_new.end());
            std::vector<cxd> Rflat(comm_pc.begin(), comm_pc.end());
            auto upd = bro_state.update(Pflat, Rflat, mixing_alpha);
            bro_state = std::move(upd.first);
            std::vector<cxd>& Praw = upd.second; // flat

            if (bro_count_before == 0) {
                // Fix #4: seed with a short preconditioned descent step on first Broyden iteration
                const double beta = 0.35;
                P_mix.resize(P.size());
#ifdef _OPENMP
                #pragma omp parallel for schedule(static)
#endif
                for (long long t=0;t<(long long)P_mix.size();++t)
                    P_mix[(size_t)t] = P[(size_t)t] - beta * comm_pc[(size_t)t];

                // Smooth transition from previous iterate
                const double w_keep = 0.7, w_new = 0.3;
#ifdef _OPENMP
                #pragma omp parallel for schedule(static)
#endif
                for (long long t=0;t<(long long)P_mix.size();++t)
                    P_mix[(size_t)t] = P[(size_t)t]*w_keep + P_mix[(size_t)t]*w_new;
            } else {
                // Use the LBFGS result
                P_mix.assign(Praw.begin(), Praw.end());
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
#ifdef _OPENMP
        #pragma omp parallel for collapse(2) schedule(static)
#endif
        for (long long k1 = 0; k1 < (long long)nk1; ++k1)
            for (long long k2 = 0; k2 < (long long)nk2; ++k2) {
                const size_t base = offset(nk2, d, (size_t)k1, (size_t)k2, 0, 0);
                Eigen::Map<MatC> Fk(&F_fin[base], d, d);
                Eigen::SelfAdjointEigenSolver<MatC> es;
                es.compute(Fk, Eigen::ComputeEigenvectors);
                if (es.info() != Eigen::Success) throw std::runtime_error("EVD failed (final mu)");
                const auto& ev = es.eigenvalues();
                bands_final[(size_t)k1 * nk2 + (size_t)k2] = std::vector<double>(ev.data(), ev.data() + d);
            }
        mu_fin = find_chemicalpotential(bands_final, kernel.weights, kernel.nk1, kernel.nk2, kernel.d, T, electron_density0);
    }

    py::gil_scoped_acquire gil;

    py::array_t<cxd> P_out({(py::ssize_t)nk1,(py::ssize_t)nk2,(py::ssize_t)d,(py::ssize_t)d});
    py::array_t<cxd> F_out({(py::ssize_t)nk1,(py::ssize_t)nk2,(py::ssize_t)d,(py::ssize_t)d});
    std::memcpy(P_out.mutable_data(), P.data(),    P.size()*sizeof(cxd));
    std::memcpy(F_out.mutable_data(), F_fin.data(),F_fin.size()*sizeof(cxd));

    return py::make_tuple(P_out, F_out, e_fin, mu_fin, k_fin);
}

PYBIND11_MODULE(cpp_hf, m) {
    m.doc() = "Hartree–Fock (k-grid) with FFTW + Eigen + OpenMP + EDIIS/CDIIS + preconditioned-LBFGS";
    m.def("hartreefock_iteration_cpp", &hartreefock_iteration_cpp,
          py::arg("weights"), py::arg("hamiltonian"), py::arg("v_coulomb"), py::arg("p0"),
          py::arg("electron_density0"), py::arg("T"),
          py::arg("max_iter"), py::arg("comm_tol"),
          py::arg("diis_size"), py::arg("mixing_alpha"));
}
