#pragma once
#include <complex>
#include <vector>

extern "C" { struct fftw_plan_s; typedef struct fftw_plan_s* fftw_plan; }

namespace cpp_hf_fft {

struct FftwBatched2D {
    fftw_plan fwd = nullptr;
    fftw_plan bwd = nullptr;
    size_t nk1{}, nk2{}, d{};
    std::complex<double>* plan_buf = nullptr;  // planning buffer (full-sized)
    size_t n_tot{};
    int nthreads{1};

    static void init_threads_once();
    static int  choose_threads();

    FftwBatched2D(size_t nk1_, size_t nk2_, size_t d_);
    void forward(std::complex<double>* buf) const;
    void backward(std::complex<double>* buf) const;
    ~FftwBatched2D();
};

} // namespace cpp_hf_fft
