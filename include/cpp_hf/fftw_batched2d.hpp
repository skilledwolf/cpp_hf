// fftw_batched2d.hpp - Thin wrapper around FFTW guru64 batched 2D plans
#pragma once

#include <complex>
#include <cstddef>
#include <stdexcept>
#include <mutex>

extern "C" {
#include <fftw3.h>
}

#ifdef _OPENMP
  #include <omp.h>
#endif

struct FftwBatched2D {
    fftw_plan fwd = nullptr;
    fftw_plan bwd = nullptr;
    std::size_t nk1{}, nk2{}, d{};
    std::complex<double>* plan_buf = nullptr;  // planning buffer (full-sized)
    std::size_t n_tot{};
    int nthreads{1};

    static void init_threads_once() {
#if defined(FFTW3_THREADS)
        static std::once_flag once;
        std::call_once(once, []{ fftw_init_threads(); });
#endif
    }

    static int choose_threads() {
#if defined(_OPENMP)
        return std::max(1, omp_get_max_threads());
#else
        return 1;
#endif
    }

    FftwBatched2D(std::size_t nk1_, std::size_t nk2_, std::size_t d_)
        : nk1(nk1_), nk2(nk2_), d(d_), n_tot(nk1_*nk2_*d_*d_), nthreads(choose_threads()) {
        plan_buf = reinterpret_cast<std::complex<double>*>(fftw_malloc(sizeof(std::complex<double>) * n_tot));
        if (!plan_buf) throw std::bad_alloc{};

        init_threads_once();
#if defined(FFTW3_THREADS)
        fftw_plan_with_nthreads(nthreads);
#endif

        // Strides for (nk1, nk2, d, d), C-order
        fftw_iodim64 dims[2];
        dims[0].n  = static_cast<long long>(nk2); dims[0].is = static_cast<long long>(d*d);          dims[0].os = dims[0].is;
        dims[1].n  = static_cast<long long>(nk1); dims[1].is = static_cast<long long>(nk2*d*d);     dims[1].os = dims[1].is;
        fftw_iodim64 how[2];
        how[0].n   = static_cast<long long>(d);   how[0].is  = static_cast<long long>(d);           how[0].os  = how[0].is; // i
        how[1].n   = static_cast<long long>(d);   how[1].is  = 1;                                   how[1].os  = 1;         // j

        fwd = fftw_plan_guru64_dft(2, dims, 2, how,
                reinterpret_cast<fftw_complex*>(plan_buf),
                reinterpret_cast<fftw_complex*>(plan_buf),
                FFTW_FORWARD, FFTW_MEASURE);
        if (!fwd) throw std::runtime_error("FFTW plan_guru64_dft forward failed");

        bwd = fftw_plan_guru64_dft(2, dims, 2, how,
                reinterpret_cast<fftw_complex*>(plan_buf),
                reinterpret_cast<fftw_complex*>(plan_buf),
                FFTW_BACKWARD, FFTW_MEASURE);
        if (!bwd) { fftw_destroy_plan(fwd); throw std::runtime_error("FFTW plan_guru64_dft backward failed"); }
    }

    void forward(std::complex<double>* buf) const  { fftw_execute_dft(fwd, reinterpret_cast<fftw_complex*>(buf), reinterpret_cast<fftw_complex*>(buf)); }
    void backward(std::complex<double>* buf) const { fftw_execute_dft(bwd, reinterpret_cast<fftw_complex*>(buf), reinterpret_cast<fftw_complex*>(buf)); }

    ~FftwBatched2D() { if (fwd) fftw_destroy_plan(fwd); if (bwd) fftw_destroy_plan(bwd); if (plan_buf) fftw_free(plan_buf); }
};

