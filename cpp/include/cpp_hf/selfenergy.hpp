// Exchange self-energy via FFT-based convolution.
#pragma once

#include "cpp_hf/fft.hpp"
#include "cpp_hf/types.hpp"

#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace cpp_hf {

// Plan cache keyed on (nk1, nk2, batch, effective thread count).  Keep these
// expensive plans between calls; the lock also coordinates parallel channels.
class FftPlanCache {
public:
    static FftPlanCache& instance() {
        // The cache is intentionally process-lifetime.  Letting its entries
        // destruct while a Python extension and FFTW's own globals are being
        // torn down creates an unspecified cross-DSO destruction order.
        static auto* cache = new FftPlanCache;
        return *cache;
    }

    std::shared_ptr<FftBatched2D> get(
        std::size_t nk1, std::size_t nk2, std::size_t batch) {
        const int nthreads = FftBatched2D::choose_threads();
        std::lock_guard<std::mutex> lock(mu_);
        for (auto& e : entries_) {
            if (e->nk1() == nk1 && e->nk2() == nk2 && e->batch() == batch
                && e->nthreads() == nthreads)
                return e;
        }
        const std::size_t max_n = max_entries();
        while (entries_.size() >= max_n) entries_.erase(entries_.begin());
        entries_.emplace_back(
            std::make_shared<FftBatched2D>(nk1, nk2, batch, nthreads));
        return entries_.back();
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mu_);
        entries_.clear();
    }

private:
    static std::size_t max_entries() {
        if (const char* env = std::getenv("HF_FFTW_CACHE_MAX")) {
            const int v = std::atoi(env);
            if (v > 0) return static_cast<std::size_t>(v);
        }
        return 4;
    }

    // Callers retain shared ownership for the complete FFT execution.  This
    // prevents eviction/clear from destroying a plan after get() returns but
    // before forward()/inverse() finish.
    std::vector<std::shared_ptr<FftBatched2D>> entries_;
    std::mutex mu_;
};

// In-place: P[k, ab] *= VR[k, ab'], with VR broadcasting from (dv1, dv2) to
// (nb, nb).  When (dv1, dv2) == (1, 1) the scalar VR[k, 0, 0] multiplies the
// whole (nb, nb) block; when (dv1, dv2) == (nb, nb) it broadcasts elementwise.
inline void multiply_VR_inplace(c64* P, const c64* VR,
                                std::size_t nk1, std::size_t nk2,
                                std::size_t nb,
                                std::size_t dv1, std::size_t dv2) {
    const std::size_t nk = nk1 * nk2;
    const std::size_t nb2 = nb * nb;
    if (dv1 == 1 && dv2 == 1) {
        for (std::size_t k = 0; k < nk; ++k) {
            const c64 v = VR[k];
            c64* p = P + k * nb2;
            for (std::size_t i = 0; i < nb2; ++i) p[i] *= v;
        }
    } else if (dv1 == nb && dv2 == nb) {
        for (std::size_t k = 0; k < nk; ++k) {
            c64* p = P + k * nb2;
            const c64* v = VR + k * nb2;
            for (std::size_t i = 0; i < nb2; ++i) p[i] *= v[i];
        }
    } else {
        throw std::invalid_argument("VR shape must be (..., 1, 1) or (..., nb, nb).");
    }
}

// Σ(k) = -FFT⁻¹[FFT(sigma) · VR] in place, no ifftshift (c64, C-order).
//   sigma : (nk1, nk2, nb, nb), holding the density on entry
//   VR    : (nk1, nk2, dv1, dv2)
inline void selfenergy_fft_full_inplace(c64* sigma, const c64* VR,
                                        std::size_t nk1, std::size_t nk2,
                                        std::size_t nb,
                                        std::size_t dv1, std::size_t dv2) {
    const std::size_t nb2 = nb * nb;
    const std::size_t n_tot = nk1 * nk2 * nb2;

    auto plan = FftPlanCache::instance().get(nk1, nk2, nb2);
    plan->forward(sigma);
    multiply_VR_inplace(sigma, VR, nk1, nk2, nb, dv1, dv2);
    plan->inverse(sigma);
    for (std::size_t i = 0; i < n_tot; ++i) sigma[i] = -sigma[i];
}

}  // namespace cpp_hf
