// hf_kernel.hpp - HFKernel class (FFTW + Eigen + utilities)
#pragma once

#include <vector>
#include <complex>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <limits>
#include <cmath>
#include <span>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include "cpp_hf/views.hpp"
#include "cpp_hf/platform.hpp"

#ifdef _OPENMP
  #include <omp.h>
#endif

#include "cpp_hf/utils.hpp"
#include "cpp_hf/fft_batched2d.hpp"
#include "cpp_hf/prof.hpp"

namespace hf {
using cxd  = std::complex<double>;
using MatC = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

struct HFKernel {
    std::size_t nk1, nk2, d;
    std::size_t n_tot;
    std::span<const double> weights; // (nk1*nk2)
    double weight_sum = 0.0;     // sum of weights
    double weight_mean = 0.0;    // scalar weight used in JAX path (dk^2/(2π)^2)
    std::span<const cxd> H;          // (nk1*nk2*d*d)
    std::vector<cxd> Vhat_full;   // (nk1*nk2*d*d)
    std::vector<cxd> Vhat_scalar; // (nk1*nk2)
    bool v_is_scalar = false;
    FftBatched2D plan;          // batched 2D FFT plan
    double T;
    double n_target = 0.0;
    // Cache from the last call_with_fock_and_energy() for reuse in preconditioning
    mutable std::vector<MatC> last_evecs;                 // per (k1,k2)
    mutable std::vector<std::vector<double>> last_bands;  // per (k1,k2)
    mutable bool last_cache_valid = false;

    // Reusable FFT scratch (aligned)
    cxd* scratch_fft = nullptr;

    HFKernel(std::size_t nk1_, std::size_t nk2_, std::size_t d_,
             std::span<const double> W,
             std::span<const cxd> H_in,
             std::span<const cxd> V_in,
             std::size_t dv1, std::size_t dv2,
             double T_,
             double n_target_)
        : nk1(nk1_), nk2(nk2_), d(d_),
          n_tot(nk1_*nk2_*d_*d_),
          plan(nk1_, nk2_, d_),
          T(T_),
          n_target(n_target_) {

        Eigen::setNbThreads(1); // let OpenMP own outer parallelism

        // weights
        if (W.size() != nk1*nk2)
            throw std::invalid_argument("weights must be length nk1*nk2");
        weights = W;
        weight_sum = std::accumulate(weights.begin(), weights.end(), 0.0);
        weight_mean = weight_sum / static_cast<double>(nk1*nk2);

        // H
        if (H_in.size() != n_tot)
            throw std::invalid_argument("H must be (nk1,nk2,d,d) flattened length nk1*nk2*d*d");
        H = H_in;

        // Scratch
        scratch_fft = new cxd[n_tot];
        if (!scratch_fft) throw std::bad_alloc{};

        // Vhat = FFT(weight_mean * V) along (0,1)
        if (!((dv1==1 && dv2==1) || (dv1==d && dv2==d)))
            throw std::invalid_argument("V last dims must be (1,1) or (d,d)");

        if (dv1==1 && dv2==1) {
            v_is_scalar = true;
            Vhat_scalar.resize(nk1*nk2);
            if (V_in.size() != nk1*nk2)
                throw std::invalid_argument("V scalar must be (nk1,nk2,1,1) flattened length nk1*nk2");
            const cxd* vptr = V_in.data();
            cxd* buf = reinterpret_cast<cxd*>(fftw_malloc(sizeof(cxd) * nk1 * nk2));
            if (!buf) throw std::bad_alloc{};
            {
                Eigen::Map<      Eigen::ArrayXcd> B(reinterpret_cast<cxd*>(buf), (Eigen::Index)(nk1*nk2));
                Eigen::Map<const Eigen::ArrayXcd> V(vptr,                         (Eigen::Index)(nk1*nk2));
                B = V * weight_mean;
            }
#  if defined(FFTW3_THREADS)
            FftwBatched2D::init_threads_once();
            fftw_plan_with_nthreads(plan.nthreads);
#  endif
            fftw_plan pf = fftw_plan_dft_2d((int)nk1, (int)nk2,
                                reinterpret_cast<fftw_complex*>(buf),
                                reinterpret_cast<fftw_complex*>(buf),
                                FFTW_FORWARD, FFTW_MEASURE);
            if (!pf) { fftw_free(buf); throw std::runtime_error("FFTW plan_dft_2d forward failed for scalar V"); }
            fftw_execute(pf);
            std::copy(buf, buf + nk1*nk2, Vhat_scalar.begin());
            fftw_destroy_plan(pf);
            fftw_free(buf);
        } else {
            if (V_in.size() != n_tot)
                throw std::invalid_argument("V full must be (nk1,nk2,d,d) flattened length nk1*nk2*d*d");
            Vhat_full.resize(n_tot);
            const cxd* vptr = V_in.data();
            Eigen::Map<      Eigen::ArrayXcd> D(Vhat_full.data(), (Eigen::Index)n_tot);
            Eigen::Map<const Eigen::ArrayXcd> S(reinterpret_cast<const cxd*>(vptr), (Eigen::Index)n_tot);
            D = S * weight_mean;
            plan.forward(Vhat_full.data());
        }
    }

    ~HFKernel() { if (scratch_fft) delete[] scratch_fft; }

    // Return (P_new, mu)
    std::pair<std::vector<cxd>, double> call(const std::vector<cxd>& P) const {
        std::vector<cxd> Sigma;
        { HF_PROFILE_SCOPE("self_energy_fft");
          ::self_energy_fft(v_is_scalar, Vhat_full, Vhat_scalar, nk1, nk2, d,
                            P, Sigma, plan, scratch_fft);
        }

        std::vector<cxd> Fock(n_tot);
        {
            HF_PROFILE_SCOPE("build_fock");
#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for (long long t=0;t<(long long)n_tot;++t) Fock[(std::size_t)t] = H[(std::size_t)t];
#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for (long long t=0;t<(long long)n_tot;++t) Fock[(std::size_t)t] += Sigma[(std::size_t)t];
        }

        // Diagonalize at each k
        std::vector<std::vector<double>> bands(nk1*nk2);
        std::vector<MatC> evecs(nk1*nk2);
        {
            HF_PROFILE_SCOPE("evd_all_k.call");
#ifdef _OPENMP
            #pragma omp parallel for collapse(2) schedule(static)
#endif
            for (long long k1i=0;k1i<(long long)nk1;++k1i)
                for (long long k2i=0;k2i<(long long)nk2;++k2i) {
                    const std::size_t base = ::offset(nk2,d,(std::size_t)k1i,(std::size_t)k2i,0,0);
                    Eigen::Map<MatC> Fk(&Fock[base], d, d);
                    Eigen::SelfAdjointEigenSolver<MatC> es;
                    es.compute(Fk, Eigen::ComputeEigenvectors);
                    if (es.info()!=Eigen::Success) throw std::runtime_error("EVD failed");
                    bands[(std::size_t)k1i*nk2+(std::size_t)k2i] = std::vector<double>(es.eigenvalues().data(), es.eigenvalues().data()+d);
                    evecs[(std::size_t)k1i*nk2+(std::size_t)k2i]  = es.eigenvectors();
                }
        }

        // Cache for potential reuse (e.g., preconditioning)
        last_evecs = evecs;
        last_bands = bands;
        last_cache_valid = true;

        double mu;
        {
            HF_PROFILE_SCOPE("find_mu");
            mu = ::find_chemicalpotential(bands, weights, nk1, nk2, d, T, n_target);
        }

        // P_new = U f(ε-μ) U^H
        std::vector<cxd> Pnew(n_tot);
        {
            HF_PROFILE_SCOPE("build_P");
#ifdef _OPENMP
            #pragma omp parallel for collapse(2) schedule(static)
#endif
            for (long long k1i=0;k1i<(long long)nk1;++k1i)
                for (long long k2i=0;k2i<(long long)nk2;++k2i) {
                    const auto& U = evecs[(std::size_t)k1i*nk2+(std::size_t)k2i];
                    Eigen::VectorXcd occ(d);
                    for (std::size_t j=0;j<d;++j) occ((Eigen::Index)j) = cxd(::fermi(bands[(std::size_t)k1i*nk2+(std::size_t)k2i][j]-mu, T), 0.0);
                    MatC D(d,d);
                    D.noalias() = U * occ.asDiagonal() * U.adjoint();
                    const std::size_t base = ::offset(nk2,d,(std::size_t)k1i,(std::size_t)k2i,0,0);
                    Eigen::Map<MatC>(&Pnew[base], d, d) = D;
                }
        }
        return {Pnew, mu};
    }

    // Build Fock Σ[P] and compute energy E = ∑_k w_k Re{(H + 0.5 Σ)^† · P}
    void fock_and_energy_of(const std::vector<cxd>& P, std::vector<cxd>& F, double& E) const {
        hf::Grid2<const double> Wv(weights.data(), nk1, nk2);
        std::vector<cxd> Sigma;
        { HF_PROFILE_SCOPE("self_energy_fft");
          ::self_energy_fft(v_is_scalar, Vhat_full, Vhat_scalar, nk1, nk2, d,
                            P, Sigma, plan, scratch_fft);
        }

        F.resize(n_tot);
        {
            HF_PROFILE_SCOPE("build_fock");
#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for (long long t=0;t<(long long)n_tot;++t)
                F[(std::size_t)t] = H[(std::size_t)t] + Sigma[(std::size_t)t];
        }

        double e = 0.0;
#ifdef _OPENMP
        #pragma omp parallel for collapse(2) reduction(+:e) schedule(static)
#endif
        for (long long k1i=0;k1i<(long long)nk1;++k1i)
            for (long long k2i=0;k2i<(long long)nk2;++k2i) {
                const double w = hf::mds_get(Wv, (std::size_t)k1i, (std::size_t)k2i);
                const std::size_t base = ::offset(nk2,d,(std::size_t)k1i,(std::size_t)k2i,0,0);
                Eigen::Map<const MatC> Hk(H.data() + base, d, d);
                Eigen::Map<const MatC> Sk(&Sigma[base], d, d);
                Eigen::Map<const MatC> Pk(&P[base], d, d);
                const double s = ((Hk + 0.5*Sk).adjoint() * Pk).trace().real();
                e += w * s;
            }
        E = e;
    }

    void fock_of(const std::vector<cxd>& P, std::vector<cxd>& F) const {
        double dummy; fock_and_energy_of(P, F, dummy);
    }

    // Combined path: build F=H+Σ[P], diagonalize, build P_new, compute energy once, and cache eigendecomps
    void call_with_fock_and_energy(const std::vector<cxd>& P,
                                   std::vector<cxd>& Pnew,
                                   std::vector<cxd>& F,
                                   double& E,
                                   double& mu) const {
        hf::Grid2<const double> Wv(weights.data(), nk1, nk2);
        std::vector<cxd> Sigma;
        ::self_energy_fft(v_is_scalar, Vhat_full, Vhat_scalar, nk1, nk2, d,
                          P, Sigma, plan, scratch_fft);

        F.resize(n_tot);
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (long long t=0;t<(long long)n_tot;++t)
            F[(std::size_t)t] = H[(std::size_t)t] + Sigma[(std::size_t)t];

        // Diagonalize and cache
        if (last_evecs.size() != nk1*nk2) last_evecs.assign(nk1*nk2, MatC());
        if (last_bands.size() != nk1*nk2) last_bands.assign(nk1*nk2, std::vector<double>());
        {
            HF_PROFILE_SCOPE("evd_all_k.fock_energy");
            ::for_k(nk1, nk2, [&](std::size_t k1i, std::size_t k2i){
                const std::size_t base = ::offset(nk2,d,k1i,k2i,0,0);
                Eigen::Map<const MatC> Fk(&F[base], d, d);
                Eigen::SelfAdjointEigenSolver<MatC> es;
                es.compute(Fk, Eigen::ComputeEigenvectors);
                if (es.info()!=Eigen::Success) throw std::runtime_error("EVD failed");
                last_bands[k1i*nk2+k2i] = std::vector<double>(es.eigenvalues().data(), es.eigenvalues().data()+d);
                last_evecs[k1i*nk2+k2i]  = es.eigenvectors();
            });
        }
        last_cache_valid = true;

        mu = ::find_chemicalpotential(last_bands, weights, nk1, nk2, d, T, n_target);

        // Build P_new = U f U^H
        Pnew.resize(n_tot);
        {
            HF_PROFILE_SCOPE("build_P");
            ::for_k(nk1, nk2, [&](std::size_t k1i, std::size_t k2i){
                const auto& U = last_evecs[k1i*nk2+k2i];
                Eigen::VectorXcd occ(d);
                for (std::size_t j=0;j<d;++j) occ((Eigen::Index)j) = cxd(::fermi(last_bands[k1i*nk2+k2i][j]-mu, T), 0.0);
                const std::size_t base = ::offset(nk2,d,k1i,k2i,0,0);
                Eigen::Map<MatC> Pk(&Pnew[base], d, d);
                Pk.noalias() = U * occ.asDiagonal() * U.adjoint();
            });
        }

        // Energy using (H + 0.5 Σ) = 0.5 (H + F)
        double e = 0.0;
        {
            HF_PROFILE_SCOPE("energy_sum");
#ifdef _OPENMP
            #pragma omp parallel for collapse(2) reduction(+:e) schedule(static)
#endif
            for (long long k1i=0;k1i<(long long)nk1;++k1i)
                for (long long k2i=0;k2i<(long long)nk2;++k2i) {
                    const double w = hf::mds_get(Wv, (std::size_t)k1i, (std::size_t)k2i);
                    const std::size_t base = ::offset(nk2,d,(std::size_t)k1i,(std::size_t)k2i,0,0);
                Eigen::Map<const MatC> Hk(H.data() + base, d, d);
                    Eigen::Map<const MatC> Fk(&F[base], d, d);
                    Eigen::Map<const MatC> Pk(&Pnew[base], d, d);
                    const MatC Hhalf = 0.5 * (Fk + Hk);
                    const double s = (Hhalf.adjoint() * Pk).trace().real();
                    e += w * s;
                }
        }
        E = e;
    }

    // Precondition commutator using cached (evecs, bands) from last call_with_fock_and_energy
    void precondition_commutator_cached(const std::vector<cxd>& F,
                                        const std::vector<cxd>& comm,
                                        std::vector<cxd>& out,
                                        double delta = 1e-3) const {
        if (!last_cache_valid || last_evecs.size() != nk1*nk2 || last_bands.size() != nk1*nk2)
            throw std::runtime_error("Preconditioner cache invalid; call call_with_fock_and_energy first");
        out.resize(F.size());
#ifdef _OPENMP
        #pragma omp parallel for collapse(2) schedule(static)
#endif
        for (long long k1=0;k1<(long long)nk1;++k1)
            for (long long k2=0;k2<(long long)nk2;++k2) {
                const size_t base = (static_cast<size_t>(k1)*nk2 + static_cast<size_t>(k2)) * d * d;
                const MatC& C = last_evecs[(std::size_t)k1*nk2+(std::size_t)k2];
                // eigenvalues vector
                Eigen::VectorXd eps = Eigen::Map<const Eigen::VectorXd>(last_bands[(std::size_t)k1*nk2+(std::size_t)k2].data(), (Eigen::Index)d);
                const Eigen::RowVectorXd er = eps.transpose();
                const Eigen::VectorXd    ec = eps;
                Eigen::MatrixXd denom = ec.rowwise().replicate((Eigen::Index)d) - er.colwise().replicate((Eigen::Index)d);
                denom.array() += delta;

                Eigen::Map<const MatC> Rk(&comm[base], d, d);
                MatC Rmo  = C.adjoint() * Rk * C;
                MatC Rtil = - Rmo.cwiseQuotient(denom.cast<cxd>());
                Eigen::Map<MatC> outk(&out[base], d, d);
                outk.noalias() = C * Rtil * C.adjoint();
            }
    }

    // Same as fock_and_energy_of; does NOT recompute/cached EVD (reuse cache from call())
    void fock_energy_and_cache_evd(const std::vector<cxd>& P,
                                   std::vector<cxd>& F,
                                   double& E) const {
        hf::Grid2<const double> Wv(weights.data(), nk1, nk2);
        std::vector<cxd> Sigma;
        { HF_PROFILE_SCOPE("self_energy_fft");
          ::self_energy_fft(v_is_scalar, Vhat_full, Vhat_scalar, nk1, nk2, d,
                            P, Sigma, plan, scratch_fft);
        }

        F.resize(n_tot);
        {
            HF_PROFILE_SCOPE("build_fock");
#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for (long long t=0;t<(long long)n_tot;++t)
                F[(std::size_t)t] = H[(std::size_t)t] + Sigma[(std::size_t)t];
        }

        double e = 0.0;
        {
            HF_PROFILE_SCOPE("energy_sum");
#ifdef _OPENMP
            #pragma omp parallel for collapse(2) reduction(+:e) schedule(static)
#endif
            for (long long k1i=0;k1i<(long long)nk1;++k1i)
                for (long long k2i=0;k2i<(long long)nk2;++k2i) {
                    const double w = hf::mds_get(Wv, (std::size_t)k1i, (std::size_t)k2i);
                    const std::size_t base = ::offset(nk2,d,(std::size_t)k1i,(std::size_t)k2i,0,0);
                    Eigen::Map<const MatC> Hk(H.data() + base, d, d);
                    Eigen::Map<const MatC> Sk(&Sigma[base], d, d);
                    Eigen::Map<const MatC> Pk(&P[base], d, d);
                    const double s = ((Hk + 0.5*Sk).adjoint() * Pk).trace().real();
                    e += w * s;
                }
        }
        E = e;

        // Reuse eigendecomposition cached in call(P) for preconditioning in this iteration
    }
};

} // namespace hf
