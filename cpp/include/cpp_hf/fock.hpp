// build_fock, hf_energy, occupation_entropy, free_energy.
#pragma once

#include "cpp_hf/kernel.hpp"
#include "cpp_hf/linalg.hpp"
#include "cpp_hf/parallel.hpp"
#include "cpp_hf/selfenergy.hpp"
#include "cpp_hf/superlattice.hpp"
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

    // We may need rho_diff = P - refP for both the exchange transform AND
    // the superlattice Hartree (which needs rho_bar = Σ_k w_k rho_diff_k).
    // Compute it once and stash it in Sigma; the Fock transform consumes it
    // in-place, so we extract rho_bar first when the superlattice Hartree
    // path is active.
    const bool need_rho_diff_in_sigma =
        K.include_exchange || K.superlattice_hartree_active;

    std::vector<c64> sl_hartree_full;  // (nb, nb), populated when active
    bool sl_hartree_populated = false;

    if (need_rho_diff_in_sigma) {
        if (K.has_refP) {
            parallel_for(nk, [&](std::size_t k) {
                const std::size_t off = k * nb2;
                for (std::size_t i = 0; i < nb2; ++i)
                    Sigma[off + i] = P[off + i] - K.refP[off + i];
            });
        } else {
            std::memcpy(Sigma, P, n_tot * sizeof(c64));
        }

        // Superlattice Hartree: derive a k-independent (nb, nb) shift from
        // rho_bar.  Must run before the Fock transform overwrites Sigma.
        if (K.include_hartree && K.superlattice_hartree_active) {
            const std::size_t nG = K.n_G;
            const std::size_t orb = K.dim_orb;
            const std::size_t n_d = K.n_delta;

            // rho_bar = Σ_k w_k rho_diff_k   ((nb, nb))
            std::vector<c64> rho_bar(nb2, c64(0.0, 0.0));
            for (std::size_t k = 0; k < nk; ++k) {
                const c64* rk = Sigma + k * nb2;  // currently holds rho_diff
                const f32 wk = K.w2d[k];
                for (std::size_t i = 0; i < nb2; ++i)
                    rho_bar[i] += static_cast<c64>(wk) * rk[i];
            }

            // rho_GG[a, b] = Σ_xi rho_bar[(a*orb)+xi, (b*orb)+xi]
            std::vector<c64> rho_GG(nG * nG, c64(0.0, 0.0));
            for (std::size_t a = 0; a < nG; ++a) {
                for (std::size_t b = 0; b < nG; ++b) {
                    c64 s(0.0, 0.0);
                    for (std::size_t xi = 0; xi < orb; ++xi) {
                        const std::size_t row = a * orb + xi;
                        const std::size_t col = b * orb + xi;
                        s += rho_bar[row * nb + col];
                    }
                    rho_GG[a * nG + b] = s;
                }
            }

            // rho_for_delta[d] = Σ_{(i,j): pair_to_delta[i,j]==d} rho_GG[i, j]
            // This is the plane-wave Hartree analogue of the G_0 sum in the
            // Fock channel — multiple (G_a, G_b) pairs can share the same ΔG.
            std::vector<c64> rho_for_delta(n_d, c64(0.0, 0.0));
            for (std::size_t i = 0; i < nG; ++i) {
                for (std::size_t j = 0; j < nG; ++j) {
                    const std::size_t d =
                        static_cast<std::size_t>(K.pair_to_delta[i * nG + j]);
                    rho_for_delta[d] += rho_GG[i * nG + j];
                }
            }

            sl_hartree_full.assign(nb2, c64(0.0, 0.0));
            const f32 deg = K.hartree_degeneracy;

            if (K.HH_GG_orbital == nullptr) {
                // Scalar-V path:
                //   sigma_GG[a, b] = degeneracy · HH_GG[a, b]
                //                    · rho_for_delta[ΔG(a, b)]
                //   sigma_H[(a*orb+ξ), (b*orb+ξ)] = sigma_GG[a, b]   ∀ξ
                for (std::size_t a = 0; a < nG; ++a) {
                    for (std::size_t b = 0; b < nG; ++b) {
                        const std::size_t d =
                            static_cast<std::size_t>(K.pair_to_delta[a * nG + b]);
                        const c64 s =
                            static_cast<c64>(deg * K.HH_GG[a * nG + b])
                            * rho_for_delta[d];
                        for (std::size_t xi = 0; xi < orb; ++xi) {
                            const std::size_t row = a * orb + xi;
                            const std::size_t col = b * orb + xi;
                            sl_hartree_full[row * nb + col] = s;
                        }
                    }
                }
            } else {
                // Orbital-resolved V path.  We need the per-orbital ΔG
                // density (rho_perorbital[d, γ]) rather than the orbital-
                // traced one, then contract with the (α, γ) block of
                // HH_GG_orbital at each ΔG.
                //
                // rho_perorbital_for_delta[d, γ] =
                //     Σ_{(A,B): pair_to_delta(A,B) == d} rho_bar[(A,γ), (B,γ)]
                //
                // sigma_H[(a*orb+α), (b*orb+α)] =
                //     degeneracy · Σ_γ HH_GG_orbital[a, b, α, γ]
                //                      · rho_perorbital_for_delta[d(a,b), γ]
                std::vector<c64> rho_perorbital_for_delta(n_d * orb,
                                                          c64(0.0, 0.0));
                for (std::size_t A = 0; A < nG; ++A) {
                    for (std::size_t B = 0; B < nG; ++B) {
                        const std::size_t d =
                            static_cast<std::size_t>(K.pair_to_delta[A * nG + B]);
                        for (std::size_t gamma = 0; gamma < orb; ++gamma) {
                            const std::size_t row = A * orb + gamma;
                            const std::size_t col = B * orb + gamma;
                            rho_perorbital_for_delta[d * orb + gamma] +=
                                rho_bar[row * nb + col];
                        }
                    }
                }
                for (std::size_t a = 0; a < nG; ++a) {
                    for (std::size_t b = 0; b < nG; ++b) {
                        const std::size_t d =
                            static_cast<std::size_t>(K.pair_to_delta[a * nG + b]);
                        for (std::size_t alpha = 0; alpha < orb; ++alpha) {
                            c64 s(0.0, 0.0);
                            for (std::size_t gamma = 0; gamma < orb; ++gamma) {
                                const std::size_t v_idx =
                                    ((a * nG + b) * orb + alpha) * orb + gamma;
                                s += static_cast<c64>(
                                        deg * K.HH_GG_orbital[v_idx])
                                     * rho_perorbital_for_delta[d * orb + gamma];
                            }
                            const std::size_t row = a * orb + alpha;
                            const std::size_t col = b * orb + alpha;
                            sl_hartree_full[row * nb + col] = s;
                        }
                    }
                }
            }
            sl_hartree_populated = true;
        }
    }

    // Σ = -FFT⁻¹[FFT(P − refP) · VR]  (no ifftshift; VR is phase-shifted)
    // Or, for superlattice mode, Σ = streaming Fock on (P − refP).
    if (K.include_exchange) {
        if (K.superlattice_fock_active) {
            // The streaming Fock requires distinct input / output buffers
            // (it scatters from rho into a scratch grid before gathering).
            std::vector<c64> sigma_out(n_tot);
            // hermitian_rho=true: the HF Fock input (P − refP) is always
            // Hermitian, so the primitive can compute one of each (+ΔG, -ΔG)
            // conjugate-partner channel pair and fill the other by Hermiticity
            // (~2× fewer FFTs).  Same Hermitian-density assumption the regular
            // exchange_hcp path below already relies on.
            selfenergy_superlattice_streamed(
                Sigma, sigma_out.data(),
                K.V_lag_fft, K.V_lag_fft_orbital, K.g_a_off,
                K.pair_i, K.pair_j, K.pair_start,
                K.n_delta,
                K.N_ext_x, K.N_ext_y,
                K.nk1, K.nk2,
                K.n_G, K.dim_orb,
                /*hermitian_rho=*/true);
            std::memcpy(Sigma, sigma_out.data(), n_tot * sizeof(c64));
        } else if (K.exchange_hcp && K.dv1 == 1 && K.dv2 == 1) {
            selfenergy_fft_full_hcp(Sigma, Sigma, K.VR,
                                    K.nk1, K.nk2, nb);
        } else {
            selfenergy_fft_full_inplace(Sigma, K.VR,
                                        K.nk1, K.nk2, nb, K.dv1, K.dv2);
        }
    } else {
        std::memset(Sigma, 0, n_tot * sizeof(c64));
    }

    // Absorb the superlattice Hartree shift into Sigma so the existing
    // ``hf_energy_with_hartree_diag`` / SCF / DM machinery (which expects
    // Hartree only through ``hartree_diag``) keeps working unchanged.
    if (sl_hartree_populated) {
        parallel_for(nk, [&](std::size_t k) {
            c64* sk = Sigma + k * nb2;
            for (std::size_t i = 0; i < nb2; ++i)
                sk[i] += sl_hartree_full[i];
        });
    }

    std::fill(hartree_diag, hartree_diag + nb, 0.0);
    if (K.include_hartree && !K.superlattice_hartree_active) {
        std::vector<f32> n_vec(nb, 0.0);
        for (std::size_t k = 0; k < nk; ++k) {
            const c64* pk = P + k * nb2;
            const c64* rk = K.has_refP ? (K.refP + k * nb2) : nullptr;
            const f32 wk = K.w2d[k];
            for (std::size_t i = 0; i < nb; ++i) {
                const f32 rr = rk ? rk[i * nb + i].real() : 0.0;
                const f32 dr = pk[i * nb + i].real() - rr;
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
            const c64* rk = K.has_refP ? (K.refP + k * nb2) : nullptr;
            const f32 wk = K.w2d[k];
            for (std::size_t i = 0; i < nb2; ++i)
                rho_bar[i] += static_cast<c64>(wk) * (pk[i] - (rk ? rk[i] : c64(0.0, 0.0)));
        }
        // sigma_contact = sigma_h_contact + sigma_f_contact, (nb, nb)
        std::vector<c64> sigma_contact(nb2, c64(0.0, 0.0));
        ConstMapMatXcf rho(rho_bar.data(), nb, nb);
        MapMatXcf SC(sigma_contact.data(), nb, nb);
        for (std::size_t t = 0; t < K.n_contact; ++t) {
            const f32 g = K.contact_g[t];
            if (g == 0.0) continue;
            ConstMapMatXcf Oi(K.contact_Oi + t * nb2, nb, nb);
            ConstMapMatXcf Oj(K.contact_Oj + t * nb2, nb, nb);
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
        parallel_for(nk, [&](std::size_t k) {
            c64* sk = Sigma + k * nb2;
            for (std::size_t i = 0; i < nb2; ++i) sk[i] += sigma_contact[i];
        });
    }

    // F = hermitize(h + Sigma + H)
    parallel_for(nk, [&](std::size_t k) {
        const c64* hk = K.h + k * nb2;
        const c64* sk = Sigma + k * nb2;
        c64* fk = F + k * nb2;
        for (std::size_t i = 0; i < nb2; ++i)
            fk[i] = hk[i] + sk[i];
        if (K.include_hartree) {
            for (std::size_t i = 0; i < nb; ++i)
                fk[i * nb + i] += c64(hartree_diag[i], 0.0);
        }
    });
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

// E = sum_k w_k Re Tr[h P] + 0.5 * Re Tr[(Sigma + H) (P - refP)]
inline f32 hf_energy(const HFKernel& K, const c64* P,
                     const c64* Sigma, const c64* Hh) {
    const std::size_t nk = K.nk();
    const std::size_t nb = K.nb;
    const std::size_t nb2 = K.nb2();
    double total = 0.0;
    for (std::size_t k = 0; k < nk; ++k) {
        const c64* hk = K.h + k * nb2;
        const c64* sk = Sigma + k * nb2;
        const c64* hh = Hh + k * nb2;
        const c64* pk = P + k * nb2;
        const c64* rk = K.has_refP ? (K.refP + k * nb2) : nullptr;
        const f32 wk = K.w2d[k];
        // Tr(A · B) = sum_i sum_j A_ij B_ji.
        double sum_re = 0.0;
        for (std::size_t i = 0; i < nb; ++i) {
            for (std::size_t j = 0; j < nb; ++j) {
                const c64 p = pk[j * nb + i];
                const c64 dp = p - (rk ? rk[j * nb + i] : c64(0.0, 0.0));
                const c64 interaction = 0.5 * (sk[i * nb + j] + hh[i * nb + j]);
                sum_re += static_cast<double>((hk[i * nb + j] * p).real());
                sum_re += static_cast<double>((interaction * dp).real());
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
        const c64* hk = K.h + k * nb2;
        const c64* sk = Sigma + k * nb2;
        const c64* pk = P + k * nb2;
        const c64* rk = K.has_refP ? (K.refP + k * nb2) : nullptr;
        const f32 wk = K.w2d[k];
        double sum_re = 0.0;
        for (std::size_t i = 0; i < nb; ++i) {
            for (std::size_t j = 0; j < nb; ++j) {
                c64 interaction = 0.5 * sk[i * nb + j];
                if (K.include_hartree && i == j)
                    interaction += c64(0.5 * hartree_diag[i], 0.0);
                const c64 p = pk[j * nb + i];
                const c64 dp = p - (rk ? rk[j * nb + i] : c64(0.0, 0.0));
                sum_re += static_cast<double>((hk[i * nb + j] * p).real());
                sum_re += static_cast<double>((interaction * dp).real());
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
