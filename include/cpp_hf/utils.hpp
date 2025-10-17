// utils.hpp - small inline helpers and thermodynamics utilities
#pragma once

#include <cstddef>
#include <cmath>
#include <algorithm>
#include <vector>
#include <limits>
#include <cstdint>
#include <complex>

// Optional Boost root finding
#include <boost/math/tools/toms748_solve.hpp>
#include <boost/math/tools/roots.hpp>

#include "cpp_hf/fftw_batched2d.hpp"

// Layout helper for (nk1, nk2, d, d) row-major (C-order)
inline std::size_t offset(std::size_t nk2, std::size_t d,
                          std::size_t k1, std::size_t k2,
                          std::size_t i, std::size_t j) {
    return ((k1 * nk2 + k2) * d + i) * d + j;
}

// Temperature-safe Fermi-Dirac function
inline double fermi(double x, double T) {
    const double tt = std::max(1e-12, std::abs(T));
    const double y  = x / tt;
    if (y >=  40.0) return 0.0;
    if (y <= -40.0) return 1.0;
    return 1.0 / (1.0 + std::exp(y));
}

// Find chemical potential μ so that total occupation matches target electrons
inline double find_chemicalpotential(const std::vector<std::vector<double>>& bands,
                                     const std::vector<double>& weights,
                                     std::size_t nk1, std::size_t nk2, std::size_t d,
                                     double T,
                                     double target_electrons) {
    double lo =  std::numeric_limits<double>::infinity();
    double hi = -std::numeric_limits<double>::infinity();
    for (const auto& v : bands)
        for (double x : v) { lo = std::min(lo, x); hi = std::max(hi, x); }
    const double pad = 10.0 * std::max(1e-6, T);
    lo -= pad; hi += pad;

    auto N = [&](double mu){
        double s = 0.0;
#ifdef _OPENMP
        #pragma omp parallel for reduction(+:s) schedule(static)
#endif
        for (long long k1=0;k1<(long long)nk1;++k1)
          for (long long k2=0;k2<(long long)nk2;++k2) {
              const std::size_t idx = (std::size_t)k1*nk2 + (std::size_t)k2;
              const double w = weights[idx];
              double occ = 0.0;
#ifdef _OPENMP
              #pragma omp simd reduction(+:occ)
#endif
              for (long long j=0;j<(long long)d;++j)
                  occ += fermi(bands[idx][(std::size_t)j] - mu, T);
              s += w * occ;
          }
        return s;
    };
    auto f = [&](double mu){ return N(mu) - target_electrons; };

    // Bracket
    double a = lo, b = hi, fa = f(a), fb = f(b);
    int expand = 0;
    while (!(fa <= 0.0 && fb >= 0.0) && expand < 60) {
        const double w = b - a; a -= w; b += w; fa = f(a); fb = f(b); ++expand;
    }
    if (!(fa <= 0.0 && fb >= 0.0))
        return (std::abs(fa) < std::abs(fb)) ? a : b;

    // Prefer Boost toms748
    {
        std::uintmax_t it = 200;
        auto rng = boost::math::tools::toms748_solve(f, a, b, fa, fb,
                     boost::math::tools::eps_tolerance<double>(52), it);
        const double mid = 0.5 * (rng.first + rng.second);
        if (std::isfinite(mid)) return mid;
    }
    // Fallback bisection
    const double tol = 1e-10;
    for (int it=0; it<200; ++it) {
        const double m = 0.5*(a+b);
        const double fm = f(m);
        if (std::abs(fm) < tol || (b-a) < tol) return m;
        if (fa*fm < 0.0) { b=m; fb=fm; } else { a=m; fa=fm; }
    }
    return 0.5*(a+b);
}

// out = fftshift( - IFFT( FFT(P) * Vhat ), axes=(0,1) ), with IFFT normalized by 1/(nk1*nk2)
inline void self_energy_fft(bool v_is_scalar,
                            const std::vector<std::complex<double>>& Vhat_full,
                            const std::vector<std::complex<double>>& Vhat_scalar,
                            std::size_t nk1, std::size_t nk2, std::size_t d,
                            const std::vector<std::complex<double>>& P,
                            std::vector<std::complex<double>>& out,
                            const FftwBatched2D& plan,
                            std::complex<double>* scratch_fft) {
    const std::size_t n_tot = nk1 * nk2 * d * d;
    std::copy(P.begin(), P.end(), scratch_fft);
    plan.forward(scratch_fft);

    if (v_is_scalar) {
#ifdef _OPENMP
        #pragma omp parallel for collapse(2) schedule(static)
#endif
        for (long long k1i=0;k1i<(long long)nk1;++k1i)
          for (long long k2i=0;k2i<(long long)nk2;++k2i) {
              const std::complex<double> v = Vhat_scalar[(std::size_t)k1i*nk2 + (std::size_t)k2i];
              const std::size_t base = (((std::size_t)k1i*nk2 + (std::size_t)k2i) * d) * d;
#ifdef _OPENMP
              #pragma omp simd
#endif
              for (long long t=0; t<(long long)(d*d); ++t)
                  scratch_fft[base + (std::size_t)t] *= v;
          }
    } else {
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (long long t=0; t<(long long)n_tot; ++t)
            scratch_fft[(std::size_t)t] *= Vhat_full[(std::size_t)t];
    }

    plan.backward(scratch_fft);
    const double norm = -1.0 / static_cast<double>(nk1*nk2);
    out.resize(n_tot);

    const std::size_t shift1 = nk1/2, shift2 = nk2/2;
#ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
#endif
    for (long long k1i=0;k1i<(long long)nk1;++k1i) {
        for (long long k2i=0;k2i<(long long)nk2;++k2i) {
            const std::size_t s1 = ((std::size_t)k1i + shift1) % nk1;
            const std::size_t s2 = ((std::size_t)k2i + shift2) % nk2;
            const std::size_t src_base = (((std::size_t)k1i*nk2 + (std::size_t)k2i) * d) * d;
            const std::size_t dst_base = (((std::size_t)s1*nk2 + (std::size_t)s2) * d) * d;
#ifdef _OPENMP
            #pragma omp simd
#endif
            for (long long t=0; t<(long long)(d*d); ++t)
                out[dst_base + (std::size_t)t] = scratch_fft[src_base + (std::size_t)t] * norm;
        }
    }
}
