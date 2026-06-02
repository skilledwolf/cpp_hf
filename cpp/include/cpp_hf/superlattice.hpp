// Streaming superlattice Fock self-energy.
//
// For Hamiltonians written in a plane-wave-like basis indexed by a
// moire/superlattice reciprocal-vector basis ``g_basis`` (TBG, modulated
// bilayer, twisted MoTe_2, generic SuperlatticeModel), the Fock self-energy
//
//   Σ_F[α G_a, β G_b](k) = -Σ_{q, G_0} V(|k - q + G_0|)
//                          · ρ[α (G_a+G_0), β (G_b+G_0)](q)
//
// is a single 2D convolution in absolute momentum p = k - G, decoupled into
// independent (α, β, ΔG = G_a - G_b) channels.  This routine streams those
// channels through one shared (N_ext_x, N_ext_y, dim_orb, dim_orb) scratch
// buffer: scatter ρ̃ onto the extended grid, FFT-convolve against the lag
// Coulomb kernel, gather σ̃ back to (k, G_a, G_b).  Peak memory is
// O(N_ext^2 · dim_orb^2), independent of n_delta.
#pragma once

#include "cpp_hf/selfenergy.hpp"
#include "cpp_hf/types.hpp"

#include <cstring>
#include <vector>

namespace cpp_hf {

// All arrays are C-contiguous, row-major.
//
//   rho        : (nkx, nky, n_G, dim_orb, n_G, dim_orb) complex128
//   sigma_out  : same shape as rho
//   VR_fft     : (N_ext_x, N_ext_y) complex128.  FFT of the lag Coulomb
//                kernel, already scaled by the BZ-integration weight scalar.
//   g_a_off    : (n_G, 2) int.  Per-G_a tile offset (in units of nkx, nky).
//   pair_i     : (n_pairs,) int.  Flattened (G_a, G_b) pair list grouped
//                by ΔG-index.
//   pair_j     : (n_pairs,) int.
//   pair_start : (n_delta + 1,) int.  CSR row pointer:
//                pair_i[pair_start[d] : pair_start[d+1]] lists all G_a values
//                that map to ΔG-index d (matched by pair_j entries).
// ``VR_fft`` is the scalar (N_ext_x, N_ext_y) lag Coulomb kernel.  When
// ``VR_fft_orbital`` is non-null it is interpreted as a
// (N_ext_x, N_ext_y, dim_orb, dim_orb) orbital-resolved kernel — the inner
// multiply step uses ``V_{αβ}(q)`` per orbital pair (α, β) instead of the
// scalar broadcast.  ``VR_fft`` is still required so callers that haven't
// migrated keep working unchanged; the orbital-resolved field overrides it
// when present.
inline void selfenergy_superlattice_streamed(
    const c64* rho,
    c64* sigma_out,
    const c64* VR_fft,
    const c64* VR_fft_orbital,
    const int64_t* g_a_off,
    const int64_t* pair_i,
    const int64_t* pair_j,
    const int64_t* pair_start,
    std::size_t n_delta,
    std::size_t N_ext_x,
    std::size_t N_ext_y,
    std::size_t nkx,
    std::size_t nky,
    std::size_t n_G,
    std::size_t dim_orb
) {
    const std::size_t orb2 = dim_orb * dim_orb;
    const std::size_t n_pix = N_ext_x * N_ext_y;
    const std::size_t channel_size = n_pix * orb2;
    const std::size_t row_rho = nky * n_G * dim_orb * n_G * dim_orb;  // stride along kx in rho
    const std::size_t row_dg = N_ext_y * orb2;                        // stride along px in rho_dg

    // VR for selfenergy_fft_full_inplace: (N_ext_x, N_ext_y, 1, 1) scalar.
    // The existing routine treats dv1 == dv2 == 1 as a scalar broadcast.
    // Zero output up front; the gather phase writes only the slots we touch
    // (only pairs that appear in the CSR), but a fresh sigma_out is simpler
    // and lets callers reuse the buffer between SCF iterations.
    std::memset(sigma_out, 0, nkx * nky * n_G * dim_orb * n_G * dim_orb * sizeof(c64));

    std::vector<c64> rho_dg(channel_size);

    for (std::size_t dg = 0; dg < n_delta; ++dg) {
        const std::size_t a = static_cast<std::size_t>(pair_start[dg]);
        const std::size_t b = static_cast<std::size_t>(pair_start[dg + 1]);
        if (a == b) continue;

        // --- 1) Zero the channel buffer. ---
        std::memset(rho_dg.data(), 0, channel_size * sizeof(c64));

        // --- 2) Scatter ρ[kx, ky, i, α, j, β] → ρ_dg[ox+kx, oy+ky, α, β]. ---
        for (std::size_t p = a; p < b; ++p) {
            const std::size_t i = static_cast<std::size_t>(pair_i[p]);
            const std::size_t j = static_cast<std::size_t>(pair_j[p]);
            const std::size_t ox = nkx * static_cast<std::size_t>(g_a_off[2 * i + 0]);
            const std::size_t oy = nky * static_cast<std::size_t>(g_a_off[2 * i + 1]);
            for (std::size_t kx = 0; kx < nkx; ++kx) {
                const c64* rho_row = rho + kx * row_rho + 0;  // start of (kx, *, ...)
                c64* dg_row = rho_dg.data() + (ox + kx) * row_dg;
                for (std::size_t ky = 0; ky < nky; ++ky) {
                    // Index into rho: (((kx*nky + ky)*n_G + i)*dim_orb + α)*n_G + j)*dim_orb + β
                    const c64* rho_block =
                        rho_row + (((ky * n_G + i) * dim_orb) * n_G * dim_orb) + j * dim_orb;
                    c64* dg_block = dg_row + (oy + ky) * orb2;
                    for (std::size_t aa = 0; aa < dim_orb; ++aa) {
                        const c64* src = rho_block + aa * (n_G * dim_orb);
                        c64* dst = dg_block + aa * dim_orb;
                        // Copy dim_orb complex doubles (the β-row).
                        std::memcpy(dst, src, dim_orb * sizeof(c64));
                    }
                }
            }
        }

        // --- 2.5) Skip channels whose scattered density is identically zero.
        // The FFT-convolution of an all-zero channel produces an all-zero
        // self-energy, and ``sigma_out`` is already zeroed, so the FFT +
        // gather would be pure wasted work.  Empty channels dominate for
        // sparse-coherence problems (e.g. a diagonal / no-CDW density
        // populates only the ΔG=0 channel, leaving every other ΔG empty),
        // so skipping their FFTs is the main speedup for the isolated-patch
        // (CDW) layout.  Bit-identical to computing them (zero in → zero out);
        // the early-exit scan is O(channel_size), negligible vs the FFT.
        bool channel_nonzero = false;
        for (std::size_t t = 0; t < channel_size; ++t) {
            if (rho_dg[t].real() != 0.0 || rho_dg[t].imag() != 0.0) {
                channel_nonzero = true;
                break;
            }
        }
        if (!channel_nonzero) continue;

        // --- 3) FFT, multiply by VR_fft, iFFT, negate. ---
        // selfenergy_fft_full_inplace expects (nk1, nk2, nb, nb) with nb*nb
        // batched FFTs.  Treat (N_ext_x, N_ext_y, dim_orb, dim_orb) as that.
        // VR shape (N_ext_x, N_ext_y, 1, 1) → scalar broadcast (canonical
        // moire case).  When ``VR_fft_orbital`` is non-null, use the
        // (N_ext_x, N_ext_y, dim_orb, dim_orb) full kernel — multiply
        // per orbital pair (α, β) at each q.
        if (VR_fft_orbital != nullptr) {
            selfenergy_fft_full_inplace(
                rho_dg.data(), VR_fft_orbital,
                N_ext_x, N_ext_y, dim_orb,
                /*dv1=*/dim_orb, /*dv2=*/dim_orb
            );
        } else {
            selfenergy_fft_full_inplace(
                rho_dg.data(), VR_fft,
                N_ext_x, N_ext_y, dim_orb,
                /*dv1=*/1, /*dv2=*/1
            );
        }

        // --- 4) Gather σ_dg[ox+kx, oy+ky, α, β] → σ[kx, ky, i, α, j, β]. ---
        for (std::size_t p = a; p < b; ++p) {
            const std::size_t i = static_cast<std::size_t>(pair_i[p]);
            const std::size_t j = static_cast<std::size_t>(pair_j[p]);
            const std::size_t ox = nkx * static_cast<std::size_t>(g_a_off[2 * i + 0]);
            const std::size_t oy = nky * static_cast<std::size_t>(g_a_off[2 * i + 1]);
            for (std::size_t kx = 0; kx < nkx; ++kx) {
                const c64* dg_row = rho_dg.data() + (ox + kx) * row_dg;
                c64* sig_row = sigma_out + kx * row_rho;
                for (std::size_t ky = 0; ky < nky; ++ky) {
                    const c64* dg_block = dg_row + (oy + ky) * orb2;
                    c64* sig_block =
                        sig_row + (((ky * n_G + i) * dim_orb) * n_G * dim_orb) + j * dim_orb;
                    for (std::size_t aa = 0; aa < dim_orb; ++aa) {
                        const c64* src = dg_block + aa * dim_orb;
                        c64* dst = sig_block + aa * (n_G * dim_orb);
                        std::memcpy(dst, src, dim_orb * sizeof(c64));
                    }
                }
            }
        }
    }
}

}  // namespace cpp_hf
