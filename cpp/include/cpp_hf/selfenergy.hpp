// Exchange self-energy via FFT-based convolution.
#pragma once

#include "cpp_hf/fft.hpp"
#include "cpp_hf/types.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace cpp_hf {

// Plan cache keyed on (nk1, nk2, batch).  Plans are expensive to build, so we
// keep them around for the lifetime of the process.  Concurrent access is
// rare; a small lock keeps the cache safe without measurable overhead.
class FftPlanCache {
public:
    static FftPlanCache& instance() {
        static FftPlanCache c;
        return c;
    }

    FftBatched2D& get(std::size_t nk1, std::size_t nk2, std::size_t batch) {
        std::lock_guard<std::mutex> lock(mu_);
        for (auto& e : entries_) {
            if (e->nk1() == nk1 && e->nk2() == nk2 && e->batch() == batch)
                return *e;
        }
        entries_.emplace_back(std::make_unique<FftBatched2D>(nk1, nk2, batch));
        return *entries_.back();
    }

private:
    std::vector<std::unique_ptr<FftBatched2D>> entries_;
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

inline void selfenergy_fft_full_inplace(c64* sigma, const c64* VR,
                                        std::size_t nk1, std::size_t nk2,
                                        std::size_t nb,
                                        std::size_t dv1, std::size_t dv2);

// Σ(k) = -FFT⁻¹[FFT(P) · VR], no ifftshift.  P, sigma, VR are c64* in C-order.
//   P, sigma   : (nk1, nk2, nb, nb)
//   VR         : (nk1, nk2, dv1, dv2)
inline void selfenergy_fft_full(const c64* P, c64* sigma, const c64* VR,
                                std::size_t nk1, std::size_t nk2,
                                std::size_t nb,
                                std::size_t dv1, std::size_t dv2) {
    const std::size_t nb2 = nb * nb;
    const std::size_t n_tot = nk1 * nk2 * nb2;
    std::memcpy(sigma, P, n_tot * sizeof(c64));

    selfenergy_fft_full_inplace(sigma, VR, nk1, nk2, nb, dv1, dv2);
}

// Same transform as selfenergy_fft_full, but the caller has already populated
// sigma with the input density difference.  This avoids one full-buffer copy in
// solver Fock builds.
inline void selfenergy_fft_full_inplace(c64* sigma, const c64* VR,
                                        std::size_t nk1, std::size_t nk2,
                                        std::size_t nb,
                                        std::size_t dv1, std::size_t dv2) {
    const std::size_t nb2 = nb * nb;
    const std::size_t n_tot = nk1 * nk2 * nb2;

    auto& plan = FftPlanCache::instance().get(nk1, nk2, nb2);
    plan.forward(sigma);
    multiply_VR_inplace(sigma, VR, nk1, nk2, nb, dv1, dv2);
    plan.inverse(sigma);
    for (std::size_t i = 0; i < n_tot; ++i) sigma[i] = -sigma[i];
}

// Hermitian-channel-packed variant: only the upper triangle of (nb, nb) is
// Fourier transformed; the lower triangle is reconstructed by conjugation.
// Requires VR to be scalar (dv1 == dv2 == 1).  Halves the number of FFTs for
// scalar interactions when P is Hermitian.
inline void selfenergy_fft_full_hcp(const c64* P, c64* sigma, const c64* VR,
                                    std::size_t nk1, std::size_t nk2,
                                    std::size_t nb) {
    const std::size_t nb2 = nb * nb;
    const std::size_t nk = nk1 * nk2;

    // Indices of the upper triangle (i <= j), and which of those are off-diag.
    std::vector<std::size_t> tri_i, tri_j;
    tri_i.reserve(nb * (nb + 1) / 2);
    tri_j.reserve(nb * (nb + 1) / 2);
    for (std::size_t i = 0; i < nb; ++i)
        for (std::size_t j = i; j < nb; ++j) {
            tri_i.push_back(i);
            tri_j.push_back(j);
        }
    const std::size_t n_pack = tri_i.size();

    // Pack: packed[k, t] = P[k, tri_i[t], tri_j[t]]
    std::vector<c64> packed(nk * n_pack);
    for (std::size_t k = 0; k < nk; ++k) {
        const c64* pk = P + k * nb2;
        c64* dst = packed.data() + k * n_pack;
        for (std::size_t t = 0; t < n_pack; ++t)
            dst[t] = pk[tri_i[t] * nb + tri_j[t]];
    }

    auto& plan = FftPlanCache::instance().get(nk1, nk2, n_pack);
    plan.forward(packed.data());
    // Multiply by scalar VR[k]
    for (std::size_t k = 0; k < nk; ++k) {
        const c64 v = VR[k];
        c64* p = packed.data() + k * n_pack;
        for (std::size_t t = 0; t < n_pack; ++t) p[t] *= v;
    }
    plan.inverse(packed.data());

    // Negate and unpack into sigma, mirroring conj into the lower triangle.
    for (std::size_t k = 0; k < nk; ++k) {
        const c64* src = packed.data() + k * n_pack;
        c64* sk = sigma + k * nb2;
        for (std::size_t t = 0; t < n_pack; ++t) {
            const std::size_t i = tri_i[t];
            const std::size_t j = tri_j[t];
            const c64 val = -src[t];
            sk[i * nb + j] = val;
            if (i != j) sk[j * nb + i] = std::conj(val);
        }
    }
}

// In-place ifftshift along axes (0, 1) for an array shaped (nk1, nk2, batch).
inline void ifftshift_2d_batch(c64* buf, std::size_t nk1, std::size_t nk2,
                               std::size_t batch) {
    const std::size_t row = nk2 * batch;
    const std::size_t s0 = nk1 / 2;
    const std::size_t s1 = nk2 / 2;
    std::vector<c64> tmp(nk1 * row);
    for (std::size_t i = 0; i < nk1; ++i) {
        const std::size_t i_src = (i + s0) % nk1;
        for (std::size_t j = 0; j < nk2; ++j) {
            const std::size_t j_src = (j + s1) % nk2;
            std::memcpy(&tmp[(i * nk2 + j) * batch],
                        &buf[(i_src * nk2 + j_src) * batch],
                        batch * sizeof(c64));
        }
    }
    std::memcpy(buf, tmp.data(), nk1 * row * sizeof(c64));
}

}  // namespace cpp_hf
