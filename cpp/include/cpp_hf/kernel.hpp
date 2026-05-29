// HFKernel data container: precomputed arrays consumed by the C++ solvers.
#pragma once

#include "cpp_hf/types.hpp"

#include <stdexcept>

namespace cpp_hf {

struct HFKernel {
    std::size_t nk1 = 0;
    std::size_t nk2 = 0;
    std::size_t nb = 0;
    std::size_t dv1 = 0;
    std::size_t dv2 = 0;

    bool include_hartree = false;
    bool include_exchange = true;
    bool exchange_hcp = false;
    bool has_refP = false;

    f32 T = 0.0;
    f32 weight_sum = 0.0;

    const c64* h = nullptr;        // (nk1, nk2, nb, nb)
    const c64* VR = nullptr;       // (nk1, nk2, dv1,dv2), shifted
    const c64* refP = nullptr;     // optional (nk1, nk2, nb, nb)
    const f32* w2d = nullptr;      // (nk1, nk2)
    const f32* HH = nullptr;       // (nb, nb)

    std::size_t n_contact = 1;
    const f32* contact_g = nullptr;      // (n_contact,)
    const c64* contact_Oi = nullptr;     // (n_contact, nb, nb)
    const c64* contact_Oj = nullptr;     // (n_contact, nb, nb)

    // -------- Superlattice Fock + Hartree (optional) --------
    //
    // When ``superlattice_fock_active`` is true, the Fock self-energy is
    // built by ``selfenergy_superlattice_streamed`` instead of the regular
    // k-space FFT exchange (``VR`` is ignored).  When
    // ``superlattice_hartree_active`` is true, the k-independent Hartree
    // shift is computed from the full (n_G, n_G) coupling matrix
    // ``HH_GG`` and added to ``Sigma`` (broadcast over k) — the existing
    // diagonal ``HH`` path is bypassed.  The ``hartree_diag`` output of
    // ``build_fock_compact`` is left at zero in that mode, so
    // ``hf_energy_with_hartree_diag`` works unchanged (the Hartree
    // contribution is absorbed into Sigma).

    bool superlattice_fock_active = false;
    bool superlattice_hartree_active = false;

    f32 hartree_degeneracy = 1.0;   // spin × valley factor for superlattice Hartree
    std::size_t n_G = 0;
    std::size_t dim_orb = 0;
    std::size_t n_delta = 0;
    std::size_t N_ext_x = 0;
    std::size_t N_ext_y = 0;

    const c64* V_lag_fft = nullptr;        // (N_ext_x, N_ext_y), complex
    // Optional orbital-resolved lag Coulomb FFT:
    //   ``V_lag_fft_orbital[k_ext, α, β]`` is V_{αβ}(q_lag) for each orbital
    // pair (α, β) at each lag-grid point k_ext.  When non-null, the
    // streaming Fock multiplies elementwise across orbital pairs instead
    // of using a scalar broadcast.  Layout: row-major
    // (N_ext_x, N_ext_y, dim_orb, dim_orb).
    const c64* V_lag_fft_orbital = nullptr;
    const int64_t* g_a_off = nullptr;      // (n_G, 2)
    const int64_t* pair_i = nullptr;       // (n_pairs,)
    const int64_t* pair_j = nullptr;       // (n_pairs,)
    const int64_t* pair_start = nullptr;   // (n_delta + 1,)
    const int64_t* pair_to_delta = nullptr;// (n_G, n_G)
    const f32* HH_GG = nullptr;            // (n_G, n_G), real Hartree coupling

    // Optional orbital-resolved Hartree coupling.  When non-null, the
    // Hartree path uses the per-orbital formula instead of the scalar
    // ``HH_GG`` one.  Layout: ``HH_GG_orbital[((a*n_G + b)*dim_orb + α)
    // *dim_orb + γ] = V_{αγ}(G_a − G_b)``.  Diagonal-G blocks (a == b)
    // must be zero (q=0 piece dropped).
    const f32* HH_GG_orbital = nullptr;

    std::size_t nk() const { return nk1 * nk2; }
    std::size_t nb2() const { return nb * nb; }
    std::size_t n_dense() const { return nk() * nb2(); }
};

}  // namespace cpp_hf
