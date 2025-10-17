// hf_kernel.hpp - HFKernel class (FFTW + Eigen + utilities)
#pragma once

#include <vector>
#include <complex>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <limits>
#include <cmath>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <pybind11/numpy.h>

#ifdef _OPENMP
  #include <omp.h>
#endif

#include "cpp_hf/utils.hpp"
#include "cpp_hf/fftw_batched2d.hpp"

namespace hf {

namespace py = pybind11;
using cxd  = std::complex<double>;
using MatC = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

struct HFKernel {
    std::size_t nk1, nk2, d;
    std::size_t n_tot;
    std::vector<double> weights; // (nk1*nk2)
    double weight_sum = 0.0;     // sum of weights
    double weight_mean = 0.0;    // scalar weight used in JAX path (dk^2/(2π)^2)
    std::vector<cxd> H;          // (nk1*nk2*d*d)
    std::vector<cxd> Vhat_full;   // (nk1*nk2*d*d)
    std::vector<cxd> Vhat_scalar; // (nk1*nk2)
    bool v_is_scalar = false;
    FftwBatched2D plan;          // batched 2D FFT plan
    double T;
    double n_target = 0.0;
    // Cache from the last call_with_fock_and_energy() for reuse in preconditioning
    mutable std::vector<MatC> last_evecs;                 // per (k1,k2)
    mutable std::vector<std::vector<double>> last_bands;  // per (k1,k2)
    mutable bool last_cache_valid = false;

    // Reusable FFT scratch (aligned)
    cxd* scratch_fft = nullptr;

    HFKernel(std::size_t nk1_, std::size_t nk2_, std::size_t d_,
             const py::array_t<double>& W,
             const py::array_t<cxd>& H_in,
             const py::array_t<cxd>& V_in,
             double T_,
             double n_target_)
        : nk1(nk1_), nk2(nk2_), d(d_),
          n_tot(nk1_*nk2_*d_*d_),
          plan(nk1_, nk2_, d_),
          T(T_),
          n_target(n_target_) {

        Eigen::setNbThreads(1); // let OpenMP own outer parallelism

        // weights
        if (W.ndim()!=2 || (std::size_t)W.shape(0)!=nk1 || (std::size_t)W.shape(1)!=nk2) throw std::invalid_argument("weights must be (nk1,nk2)");
        weights.assign(W.data(), W.data()+ (nk1*nk2));
        weight_sum = std::accumulate(weights.begin(), weights.end(), 0.0);
        weight_mean = weight_sum / static_cast<double>(nk1*nk2);

        // H
        if (H_in.ndim()!=4 || (std::size_t)H_in.shape(0)!=nk1 || (std::size_t)H_in.shape(1)!=nk2
            || (std::size_t)H_in.shape(2)!=d || (std::size_t)H_in.shape(3)!=d)
            throw std::invalid_argument("H must be (nk1,nk2,d,d)");
        H.assign(H_in.data(), H_in.data() + n_tot);

        // Scratch
        scratch_fft = reinterpret_cast<cxd*>(fftw_malloc(sizeof(cxd) * n_tot));
        if (!scratch_fft) throw std::bad_alloc{};

        // Vhat = FFT(weight_mean * V) along (0,1)
        if (V_in.ndim()!=4 || (std::size_t)V_in.shape(0)!=nk1 || (std::size_t)V_in.shape(1)!=nk2)
            throw std::invalid_argument("V must be (nk1,nk2,dv1,dv2)");
        const std::size_t dv1 = V_in.shape(2), dv2 = V_in.shape(3);

        if (!((dv1==1 && dv2==1) || (dv1==d && dv2==d)))
            throw std::invalid_argument("V last dims must be (1,1) or (d,d)");

        if (dv1==1 && dv2==1) {
            v_is_scalar = true;
            Vhat_scalar.resize(nk1*nk2);
            cxd* buf = reinterpret_cast<cxd*>(fftw_malloc(sizeof(cxd) * nk1 * nk2));
            if (!buf) throw std::bad_alloc{};
            const cxd* vptr = V_in.data();

#ifdef _OPENMP
            #pragma omp parallel for collapse(2) schedule(static)
#endif
            for (long long k1i=0;k1i<(long long)nk1;++k1i)
                for (long long k2i=0;k2i<(long long)nk2;++k2i)
                    buf[(std::size_t)k1i*nk2 + (std::size_t)k2i] = vptr[(std::size_t)k1i*nk2 + (std::size_t)k2i] * weight_mean;

#if defined(FFTW3_THREADS)
            FftwBatched2D::init_threads_once();
            fftw_plan_with_nthreads(plan.nthreads);
#endif
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
            std::vector<cxd> Vfull(n_tot);
            const cxd* vptr = V_in.data();
#ifdef _OPENMP
            #pragma omp parallel for collapse(4) schedule(static)
#endif
            for (long long k1i=0;k1i<(long long)nk1;++k1i)
                for (long long k2i=0;k2i<(long long)nk2;++k2i)
                    for (long long i=0;i<(long long)d;++i)
                        for (long long j=0;j<(long long)d;++j) {
                            const std::size_t v_idx = ((((std::size_t)k1i)*nk2 + (std::size_t)k2i)*d + (std::size_t)i)*d + (std::size_t)j;
                            Vfull[::offset(nk2,d,(std::size_t)k1i,(std::size_t)k2i,(std::size_t)i,(std::size_t)j)] = vptr[v_idx] * weight_mean;
                        }
            Vhat_full = std::move(Vfull);
            plan.forward(Vhat_full.data());
        }
    }

    ~HFKernel() { if (scratch_fft) fftw_free(scratch_fft); }

    // Return (P_new, mu)
    std::pair<std::vector<cxd>, double> call(const std::vector<cxd>& P) const {
        std::vector<cxd> Sigma;
        ::self_energy_fft(v_is_scalar, Vhat_full, Vhat_scalar, nk1, nk2, d,
                          P, Sigma, plan, scratch_fft);

        std::vector<cxd> Fock = H;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (long long t=0;t<(long long)n_tot;++t) Fock[(std::size_t)t] += Sigma[(std::size_t)t];

        // Diagonalize at each k
        std::vector<std::vector<double>> bands(nk1*nk2);
        std::vector<MatC> evecs(nk1*nk2);
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

        // Cache for potential reuse (e.g., preconditioning)
        last_evecs = evecs;
        last_bands = bands;
        last_cache_valid = true;

        const double mu = ::find_chemicalpotential(bands, weights, nk1, nk2, d, T, n_target);

        // P_new = U f(ε-μ) U^H
        std::vector<cxd> Pnew(n_tot);
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
        return {Pnew, mu};
    }

    // Build Fock Σ[P] and compute energy E = ∑_k w_k Re{(H + 0.5 Σ)^† · P}
    void fock_and_energy_of(const std::vector<cxd>& P, std::vector<cxd>& F, double& E) const {
        std::vector<cxd> Sigma;
        ::self_energy_fft(v_is_scalar, Vhat_full, Vhat_scalar, nk1, nk2, d,
                          P, Sigma, plan, scratch_fft);

        F.resize(n_tot);
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (long long t=0;t<(long long)n_tot;++t)
            F[(std::size_t)t] = H[(std::size_t)t] + Sigma[(std::size_t)t];

        double e = 0.0;
#ifdef _OPENMP
        #pragma omp parallel for collapse(2) reduction(+:e) schedule(static)
#endif
        for (long long k1i=0;k1i<(long long)nk1;++k1i)
            for (long long k2i=0;k2i<(long long)nk2;++k2i) {
                const double w = weights[(std::size_t)k1i*nk2+(std::size_t)k2i];
                const std::size_t base = ::offset(nk2,d,(std::size_t)k1i,(std::size_t)k2i,0,0);
                Eigen::Map<const MatC> Hk(&H[base], d, d);
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
#ifdef _OPENMP
        #pragma omp parallel for collapse(2) schedule(static)
#endif
        for (long long k1i=0;k1i<(long long)nk1;++k1i)
            for (long long k2i=0;k2i<(long long)nk2;++k2i) {
                const std::size_t base = ::offset(nk2,d,(std::size_t)k1i,(std::size_t)k2i,0,0);
                Eigen::Map<MatC> Fk(&F[base], d, d);
                Eigen::SelfAdjointEigenSolver<MatC> es;
                es.compute(Fk, Eigen::ComputeEigenvectors);
                if (es.info()!=Eigen::Success) throw std::runtime_error("EVD failed");
                last_bands[(std::size_t)k1i*nk2+(std::size_t)k2i] = std::vector<double>(es.eigenvalues().data(), es.eigenvalues().data()+d);
                last_evecs[(std::size_t)k1i*nk2+(std::size_t)k2i]  = es.eigenvectors();
            }
        last_cache_valid = true;

        mu = ::find_chemicalpotential(last_bands, weights, nk1, nk2, d, T, n_target);

        // Build P_new = U f U^H
        Pnew.resize(n_tot);
#ifdef _OPENMP
        #pragma omp parallel for collapse(2) schedule(static)
#endif
        for (long long k1i=0;k1i<(long long)nk1;++k1i)
            for (long long k2i=0;k2i<(long long)nk2;++k2i) {
                const auto& U = last_evecs[(std::size_t)k1i*nk2+(std::size_t)k2i];
                Eigen::VectorXcd occ(d);
                for (std::size_t j=0;j<d;++j) occ((Eigen::Index)j) = cxd(::fermi(last_bands[(std::size_t)k1i*nk2+(std::size_t)k2i][j]-mu, T), 0.0);
                MatC D(d,d);
                D.noalias() = U * occ.asDiagonal() * U.adjoint();
                const std::size_t base = ::offset(nk2,d,(std::size_t)k1i,(std::size_t)k2i,0,0);
                Eigen::Map<MatC>(&Pnew[base], d, d) = D;
            }

        // Energy using (H + 0.5 Σ) = 0.5 (H + F)
        double e = 0.0;
#ifdef _OPENMP
        #pragma omp parallel for collapse(2) reduction(+:e) schedule(static)
#endif
        for (long long k1i=0;k1i<(long long)nk1;++k1i)
            for (long long k2i=0;k2i<(long long)nk2;++k2i) {
                const double w = weights[(std::size_t)k1i*nk2+(std::size_t)k2i];
                const std::size_t base = ::offset(nk2,d,(std::size_t)k1i,(std::size_t)k2i,0,0);
                Eigen::Map<const MatC> Hk(&H[base], d, d);
                Eigen::Map<const MatC> Fk(&F[base], d, d);
                Eigen::Map<const MatC> Pk(&Pnew[base], d, d);
                const MatC Hhalf = 0.5 * (Fk + Hk);
                const double s = (Hhalf.adjoint() * Pk).trace().real();
                e += w * s;
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
                Eigen::MatrixXd denom = eps.replicate(1, (Eigen::Index)d) - eps.transpose().replicate((Eigen::Index)d, 1);
                denom.array() += delta;

                Eigen::Map<const MatC> Rk(&comm[base], d, d);
                MatC Rmo  = C.adjoint() * Rk * C;
                MatC Rtil = - Rmo.cwiseQuotient(denom.cast<cxd>());
                Eigen::Map<MatC> outk(&out[base], d, d);
                outk.noalias() = C * Rtil * C.adjoint();
            }
    }

    // Same as fock_and_energy_of but also caches the eigendecomposition of F for reuse
    void fock_energy_and_cache_evd(const std::vector<cxd>& P,
                                   std::vector<cxd>& F,
                                   double& E) const {
        std::vector<cxd> Sigma;
        ::self_energy_fft(v_is_scalar, Vhat_full, Vhat_scalar, nk1, nk2, d,
                          P, Sigma, plan, scratch_fft);

        F.resize(n_tot);
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (long long t=0;t<(long long)n_tot;++t)
            F[(std::size_t)t] = H[(std::size_t)t] + Sigma[(std::size_t)t];

        double e = 0.0;
#ifdef _OPENMP
        #pragma omp parallel for collapse(2) reduction(+:e) schedule(static)
#endif
        for (long long k1i=0;k1i<(long long)nk1;++k1i)
            for (long long k2i=0;k2i<(long long)nk2;++k2i) {
                const double w = weights[(std::size_t)k1i*nk2+(std::size_t)k2i];
                const std::size_t base = ::offset(nk2,d,(std::size_t)k1i,(std::size_t)k2i,0,0);
                Eigen::Map<const MatC> Hk(&H[base], d, d);
                Eigen::Map<const MatC> Sk(&Sigma[base], d, d);
                Eigen::Map<const MatC> Pk(&P[base], d, d);
                const double s = ((Hk + 0.5*Sk).adjoint() * Pk).trace().real();
                e += w * s;
            }
        E = e;

        // Cache eigendecomposition of F
        if (last_evecs.size() != nk1*nk2) last_evecs.assign(nk1*nk2, MatC());
        if (last_bands.size() != nk1*nk2) last_bands.assign(nk1*nk2, std::vector<double>());
#ifdef _OPENMP
        #pragma omp parallel for collapse(2) schedule(static)
#endif
        for (long long k1i=0;k1i<(long long)nk1;++k1i)
            for (long long k2i=0;k2i<(long long)nk2;++k2i) {
                const std::size_t base = ::offset(nk2,d,(std::size_t)k1i,(std::size_t)k2i,0,0);
                Eigen::Map<MatC> Fk(&F[base], d, d);
                Eigen::SelfAdjointEigenSolver<MatC> es(Fk);
                if (es.info()!=Eigen::Success) throw std::runtime_error("EVD failed");
                last_bands[(std::size_t)k1i*nk2+(std::size_t)k2i] = std::vector<double>(es.eigenvalues().data(), es.eigenvalues().data()+d);
                last_evecs[(std::size_t)k1i*nk2+(std::size_t)k2i]  = es.eigenvectors();
            }
        last_cache_valid = true;
    }
};

} // namespace hf
