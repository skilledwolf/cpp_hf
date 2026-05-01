// build_fock, hf_energy, occupation_entropy, free_energy.
#pragma once

#include "cpp_hf/kernel.hpp"
#include "cpp_hf/linalg.hpp"
#include "cpp_hf/selfenergy.hpp"
#include "cpp_hf/types.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <vector>

namespace cpp_hf {

using ProjectFn = std::function<void(c64* /*F*/,
                                     std::size_t nk1, std::size_t nk2,
                                     std::size_t nb)>;

inline bool has_contact_terms(const HFKernel& K) {
    for (std::size_t t = 0; t < K.n_contact; ++t)
        if (K.contact_g[t] != 0.0) return true;
    return K.n_contact > 1;
}

// Solver-oriented Fock builder.  It writes Sigma and F, and returns the
// k-independent Hartree diagonal separately to avoid allocating and clearing a
// full dense Hartree array in every solver iteration.
inline void build_fock_compact(const HFKernel& K, const c64* P,
                               c64* Sigma, c64* F, f32* hartree_diag,
                               const ProjectFn* project_fn = nullptr) {
    const std::size_t nk = K.nk();
    const std::size_t nb = K.nb;
    const std::size_t nb2 = K.nb2();
    const std::size_t n_tot = K.n_dense();

    // Σ = -FFT⁻¹[FFT(P − refP) · VR]  (no ifftshift; VR is phase-shifted)
    if (K.include_exchange) {
        for (std::size_t i = 0; i < n_tot; ++i) Sigma[i] = P[i] - K.refP[i];

        if (K.exchange_hcp && K.dv1 == 1 && K.dv2 == 1) {
            selfenergy_fft_full_hcp(Sigma, Sigma, K.VR.data(),
                                    K.nk1, K.nk2, nb);
        } else {
            selfenergy_fft_full_inplace(Sigma, K.VR.data(),
                                        K.nk1, K.nk2, nb, K.dv1, K.dv2);
        }
    } else {
        std::memset(Sigma, 0, n_tot * sizeof(c64));
    }

    std::fill(hartree_diag, hartree_diag + nb, 0.0);
    if (K.include_hartree) {
        std::vector<f32> n_vec(nb, 0.0);
        for (std::size_t k = 0; k < nk; ++k) {
            const c64* pk = P + k * nb2;
            const c64* rk = K.refP.data() + k * nb2;
            const f32 wk = K.w2d[k];
            for (std::size_t i = 0; i < nb; ++i) {
                const f32 dr = pk[i * nb + i].real() - rk[i * nb + i].real();
                n_vec[i] += wk * dr;
            }
        }
        for (std::size_t i = 0; i < nb; ++i) {
            f32 s = 0.0;
            for (std::size_t j = 0; j < nb; ++j)
                s += K.HH[i * nb + j] * n_vec[j];
            hartree_diag[i] = s;
        }
    }

    // Contact terms: ρ̄ = Σ_k w_k (P_k − refP_k) is a single (nb, nb) matrix.
    if (has_contact_terms(K)) {
        // Compute rho_bar
        std::vector<c64> rho_bar(nb2, c64(0.0, 0.0));
        for (std::size_t k = 0; k < nk; ++k) {
            const c64* pk = P + k * nb2;
            const c64* rk = K.refP.data() + k * nb2;
            const f32 wk = K.w2d[k];
            for (std::size_t i = 0; i < nb2; ++i)
                rho_bar[i] += static_cast<c64>(wk) * (pk[i] - rk[i]);
        }
        // sigma_contact = sigma_h_contact + sigma_f_contact, (nb, nb)
        std::vector<c64> sigma_contact(nb2, c64(0.0, 0.0));
        ConstMapMatXcf rho(rho_bar.data(), nb, nb);
        MapMatXcf SC(sigma_contact.data(), nb, nb);
        for (std::size_t t = 0; t < K.n_contact; ++t) {
            const f32 g = K.contact_g[t];
            if (g == 0.0) continue;
            ConstMapMatXcf Oi(K.contact_Oi.data() + t * nb2, nb, nb);
            ConstMapMatXcf Oj(K.contact_Oj.data() + t * nb2, nb, nb);
            // tr(Oj * rho_bar)
            c64 tr_oj_rho(0.0, 0.0);
            for (std::size_t i = 0; i < nb; ++i)
                for (std::size_t j = 0; j < nb; ++j)
                    tr_oj_rho += Oj(i, j) * rho(j, i);
            // Hartree-channel: g * Oi * tr(Oj * rho_bar)
            MatXcf herm = Oi * tr_oj_rho * c64(g);
            // Fock-channel: -g * Oi * rho_bar * Oj
            MatXcf fk = -c64(g) * (Oi * rho * Oj);
            SC += herm + fk;
        }
        // Add sigma_contact to every k of Sigma
        for (std::size_t k = 0; k < nk; ++k) {
            c64* sk = Sigma + k * nb2;
            for (std::size_t i = 0; i < nb2; ++i) sk[i] += sigma_contact[i];
        }
    }

    // F = hermitize(h + Sigma + H)
    for (std::size_t k = 0; k < nk; ++k) {
        const c64* hk = K.h.data() + k * nb2;
        const c64* sk = Sigma + k * nb2;
        c64* fk = F + k * nb2;
        for (std::size_t i = 0; i < nb2; ++i)
            fk[i] = hk[i] + sk[i];
        if (K.include_hartree) {
            for (std::size_t i = 0; i < nb; ++i)
                fk[i * nb + i] += c64(hartree_diag[i], 0.0);
        }
    }
    hermitize_inplace(F, nk, nb);

    if (project_fn && *project_fn) {
        (*project_fn)(F, K.nk1, K.nk2, K.nb);
        hermitize_inplace(F, nk, nb);
    }
}

// Build Fock matrix at density P.  Outputs Sigma, H, F (each (nk1,nk2,nb,nb)).
inline void build_fock(const HFKernel& K, const c64* P,
                       c64* Sigma, c64* Hh, c64* F,
                       const ProjectFn* project_fn = nullptr) {
    const std::size_t nk = K.nk();
    const std::size_t nb = K.nb;
    const std::size_t nb2 = K.nb2();
    const std::size_t n_tot = K.n_dense();

    std::vector<f32> hartree_diag(nb, 0.0);
    build_fock_compact(K, P, Sigma, F, hartree_diag.data(), project_fn);

    std::memset(Hh, 0, n_tot * sizeof(c64));
    if (K.include_hartree) {
        for (std::size_t k = 0; k < nk; ++k) {
            c64* hk = Hh + k * nb2;
            for (std::size_t i = 0; i < nb; ++i)
                hk[i * nb + i] = c64(hartree_diag[i], 0.0);
        }
    }
}

// E = sum_k w_k Re Tr[(h + 0.5*(Sigma + H)) P]
inline f32 hf_energy(const HFKernel& K, const c64* P,
                     const c64* Sigma, const c64* Hh) {
    const std::size_t nk = K.nk();
    const std::size_t nb = K.nb;
    const std::size_t nb2 = K.nb2();
    double total = 0.0;
    for (std::size_t k = 0; k < nk; ++k) {
        const c64* hk = K.h.data() + k * nb2;
        const c64* sk = Sigma + k * nb2;
        const c64* hh = Hh + k * nb2;
        const c64* pk = P + k * nb2;
        const f32 wk = K.w2d[k];
        // Tr(M · P) = sum_i sum_j M_ij P_ji
        // Here M_ij = hk[ij] + 0.5*(sk[ij]+hh[ij])
        double sum_re = 0.0;
        for (std::size_t i = 0; i < nb; ++i) {
            for (std::size_t j = 0; j < nb; ++j) {
                const c64 m = hk[i * nb + j] + 0.5 * (sk[i * nb + j] + hh[i * nb + j]);
                const c64 p = pk[j * nb + i];
                sum_re += static_cast<double>((m * p).real());
            }
        }
        total += static_cast<double>(wk) * sum_re;
    }
    return static_cast<f32>(total);
}

// Same energy calculation as hf_energy, with the k-independent Hartree
// diagonal represented compactly.
inline f32 hf_energy_with_hartree_diag(const HFKernel& K, const c64* P,
                                       const c64* Sigma,
                                       const f32* hartree_diag) {
    const std::size_t nk = K.nk();
    const std::size_t nb = K.nb;
    const std::size_t nb2 = K.nb2();
    double total = 0.0;
    for (std::size_t k = 0; k < nk; ++k) {
        const c64* hk = K.h.data() + k * nb2;
        const c64* sk = Sigma + k * nb2;
        const c64* pk = P + k * nb2;
        const f32 wk = K.w2d[k];
        double sum_re = 0.0;
        for (std::size_t i = 0; i < nb; ++i) {
            for (std::size_t j = 0; j < nb; ++j) {
                c64 m = hk[i * nb + j] + 0.5 * sk[i * nb + j];
                if (K.include_hartree && i == j)
                    m += c64(0.5 * hartree_diag[i], 0.0);
                const c64 p = pk[j * nb + i];
                sum_re += static_cast<double>((m * p).real());
            }
        }
        total += static_cast<double>(wk) * sum_re;
    }
    return static_cast<f32>(total);
}

inline f32 occupation_entropy(const f32* p, const f32* w_norm,
                              std::size_t nk, std::size_t nb) {
    double S = 0.0;
    for (std::size_t k = 0; k < nk; ++k) {
        const f32 wk = w_norm[k];
        const f32* pk = p + k * nb;
        for (std::size_t b = 0; b < nb; ++b) {
            f32 pv = pk[b];
            if (pv < ENTROPY_CLIP) pv = ENTROPY_CLIP;
            if (pv > 1.0 - ENTROPY_CLIP) pv = 1.0 - ENTROPY_CLIP;
            const double s = static_cast<double>(pv) * std::log(static_cast<double>(pv))
                           + static_cast<double>(1.0 - pv) * std::log1p(static_cast<double>(-pv));
            S -= static_cast<double>(wk) * s;
        }
    }
    return static_cast<f32>(S);
}

inline f32 free_energy_value(f32 E, const f32* p, const f32* w_norm,
                              std::size_t nk, std::size_t nb, f32 T) {
    const f32 Tval = std::max(T, 1.0e-14);
    return E - Tval * occupation_entropy(p, w_norm, nk, nb);
}

}  // namespace cpp_hf
