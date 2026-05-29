// Batched 2D FFT wrapper around FFTW (double precision).
#pragma once

#include "cpp_hf/types.hpp"

#include <cstddef>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>

extern "C" {
#include <fftw3.h>
}

#ifdef _OPENMP
#include <omp.h>
#endif

namespace cpp_hf {

// Double-precision batched 2D FFT plan over arrays shaped (nk1, nk2, batch),
// flattened C-order with stride 1 along the batch axis.  Computes the FFT
// over axes (0, 1).  The plan operates in-place.
class FftBatched2D {
public:
    FftBatched2D(std::size_t nk1, std::size_t nk2, std::size_t batch)
        : nk1_(nk1), nk2_(nk2), batch_(batch), n_tot_(nk1 * nk2 * batch) {
        if (nk1_ == 0 || nk2_ == 0 || batch_ == 0)
            throw std::invalid_argument("FftBatched2D: zero-size dimensions are not allowed.");

        plan_buf_ = reinterpret_cast<c64*>(fftw_malloc(sizeof(c64) * n_tot_));
        if (!plan_buf_) throw std::bad_alloc{};

        init_threads_once();
        const int nthreads = choose_threads();
#if defined(FFTW3_THREADS)
        fftw_plan_with_nthreads(nthreads);
#else
        (void)nthreads;
#endif

        fftw_iodim64 dims[2];
        dims[0].n = static_cast<long long>(nk2_);
        dims[0].is = static_cast<long long>(batch_);
        dims[0].os = dims[0].is;
        dims[1].n = static_cast<long long>(nk1_);
        dims[1].is = static_cast<long long>(nk2_ * batch_);
        dims[1].os = dims[1].is;

        fftw_iodim64 how[1];
        how[0].n = static_cast<long long>(batch_);
        how[0].is = 1;
        how[0].os = 1;

        int plan_flag = FFTW_ESTIMATE;
        if (const char* p = std::getenv("HF_FFTW_PLAN")) {
            const std::string s(p);
            if (s == "estimate") plan_flag = FFTW_ESTIMATE;
            else if (s == "measure") plan_flag = FFTW_MEASURE;
            else if (s == "patient") plan_flag = FFTW_PATIENT;
            else if (s == "exhaustive") plan_flag = FFTW_EXHAUSTIVE;
        }

        fwd_ = fftw_plan_guru64_dft(
            2, dims, 1, how,
            reinterpret_cast<fftw_complex*>(plan_buf_),
            reinterpret_cast<fftw_complex*>(plan_buf_),
            FFTW_FORWARD, plan_flag);
        if (!fwd_) throw std::runtime_error("FFTW plan_guru64_dft forward failed");

        bwd_ = fftw_plan_guru64_dft(
            2, dims, 1, how,
            reinterpret_cast<fftw_complex*>(plan_buf_),
            reinterpret_cast<fftw_complex*>(plan_buf_),
            FFTW_BACKWARD, plan_flag);
        if (!bwd_) {
            fftw_destroy_plan(fwd_);
            throw std::runtime_error("FFTW plan_guru64_dft backward failed");
        }
        fftw_free(plan_buf_);
        plan_buf_ = nullptr;

        norm_ = 1.0 / static_cast<double>(nk1_ * nk2_);
    }

    ~FftBatched2D() {
        if (fwd_) fftw_destroy_plan(fwd_);
        if (bwd_) fftw_destroy_plan(bwd_);
        if (plan_buf_) fftw_free(plan_buf_);
    }

    FftBatched2D(const FftBatched2D&) = delete;
    FftBatched2D& operator=(const FftBatched2D&) = delete;

    void forward(c64* buf) const {
        fftw_execute_dft(fwd_,
                         reinterpret_cast<fftw_complex*>(buf),
                         reinterpret_cast<fftw_complex*>(buf));
    }

    // Backward (FFTW unnormalised) followed by normalisation, matching
    // numpy.fft.ifftn semantics: ifftn(x) == bwd(x) / (nk1 * nk2).
    void inverse(c64* buf) const {
        fftw_execute_dft(bwd_,
                         reinterpret_cast<fftw_complex*>(buf),
                         reinterpret_cast<fftw_complex*>(buf));
        const std::size_t n = n_tot_;
        const double s = norm_;
        for (std::size_t i = 0; i < n; ++i) buf[i] *= s;
    }

    std::size_t nk1() const { return nk1_; }
    std::size_t nk2() const { return nk2_; }
    std::size_t batch() const { return batch_; }

private:
    static void init_threads_once() {
#if defined(FFTW3_THREADS)
        static std::once_flag once;
        std::call_once(once, [] { fftw_init_threads(); });
#endif
    }

    static int choose_threads() {
        if (const char* env = std::getenv("HF_FFTW_THREADS")) {
            const int v = std::atoi(env);
            if (v > 0) return v;
        }
#if defined(_OPENMP)
        return std::max(1, omp_get_max_threads());
#else
        return 1;
#endif
    }

    std::size_t nk1_, nk2_, batch_, n_tot_;
    c64* plan_buf_ = nullptr;
    fftw_plan fwd_ = nullptr;
    fftw_plan bwd_ = nullptr;
    double norm_ = 1.0;
};

}  // namespace cpp_hf
