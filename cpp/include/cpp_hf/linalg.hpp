// Batched Hermitian eigendecomposition and small dense linalg helpers.
#pragma once

#include "cpp_hf/types.hpp"

#include <Eigen/Dense>

#include <complex>
#include <cstring>
#include <vector>

namespace cpp_hf {

using MatXcf = Eigen::Matrix<c64, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;  // c64 == complex<double>
using MapMatXcf = Eigen::Map<MatXcf>;
using ConstMapMatXcf = Eigen::Map<const MatXcf>;

// In-place hermitization on the last two axes for a (nk1, nk2, nb, nb) array.
inline void hermitize_inplace(c64* M, std::size_t nk, std::size_t nb) {
    const std::size_t nb2 = nb * nb;
    for (std::size_t k = 0; k < nk; ++k) {
        c64* mk = M + k * nb2;
        for (std::size_t i = 0; i < nb; ++i) {
            for (std::size_t j = i; j < nb; ++j) {
                const c64 a = mk[i * nb + j];
                const c64 b = mk[j * nb + i];
                const c64 h = 0.5 * (a + std::conj(b));
                mk[i * nb + j] = h;
                mk[j * nb + i] = std::conj(h);
            }
        }
    }
}

// Compute Hermitian eigendecomposition of one (nb, nb) matrix.  Eigenvalues
// returned in ascending order; eigenvectors as columns of V.  Inputs and
// outputs are row-major C-order buffers.
inline void eigh_one(const c64* M_in, std::size_t nb, f32* w_out, c64* V_out) {
    ConstMapMatXcf M(M_in, nb, nb);
    MatXcf H = 0.5 * (M + M.adjoint());
    Eigen::SelfAdjointEigenSolver<MatXcf> es(H);
    if (es.info() != Eigen::Success) {
        // Fall back: should not happen for well-formed Hermitian inputs.
        for (std::size_t i = 0; i < nb; ++i) w_out[i] = 0.0;
        MatXcf eye = MatXcf::Identity(nb, nb);
        std::memcpy(V_out, eye.data(), nb * nb * sizeof(c64));
        return;
    }
    const auto& w = es.eigenvalues();
    const auto& V = es.eigenvectors();
    for (std::size_t i = 0; i < nb; ++i) w_out[i] = w[i];
    MapMatXcf VOut(V_out, nb, nb);
    VOut = V;
}

// Batched Hermitian eigh over a (nk1, nk2, nb, nb) array.  Outputs are
// w (nk1, nk2, nb) and V (nk1, nk2, nb, nb).
inline void eigh_batched(const c64* M, f32* w, c64* V,
                         std::size_t nk1, std::size_t nk2, std::size_t nb) {
    const std::size_t nk = nk1 * nk2;
    const std::size_t nb2 = nb * nb;
    for (std::size_t k = 0; k < nk; ++k) {
        eigh_one(M + k * nb2, nb, w + k * nb, V + k * nb2);
    }
}

// Block-diagonal eigh: ignore off-block coupling.  Used by the optional
// block-spec path; the caller is responsible for verifying block structure
// before calling.  Eigenvalues are concatenated in canonical block order, then
// optionally sorted globally with corresponding column permutation.
inline void eigh_block_sizes_one(const c64* M_in, std::size_t nb,
                                 const std::vector<std::size_t>& sizes,
                                 bool sort,
                                 f32* w_out, c64* V_out) {
    ConstMapMatXcf M(M_in, nb, nb);
    MapMatXcf VOut(V_out, nb, nb);
    VOut.setZero();

    std::size_t start = 0;
    std::size_t col_start = 0;
    for (std::size_t s : sizes) {
        MatXcf sub = M.block(start, start, s, s);
        sub = 0.5 * (sub + sub.adjoint());
        Eigen::SelfAdjointEigenSolver<MatXcf> es(sub);
        const auto& w = es.eigenvalues();
        const auto& V = es.eigenvectors();
        for (std::size_t i = 0; i < s; ++i) w_out[col_start + i] = w[i];
        VOut.block(start, col_start, s, s) = V;
        start += s;
        col_start += s;
    }

    if (sort) {
        std::vector<std::size_t> order(nb);
        for (std::size_t i = 0; i < nb; ++i) order[i] = i;
        std::sort(order.begin(), order.end(),
                  [&](std::size_t a, std::size_t b) { return w_out[a] < w_out[b]; });
        std::vector<f32> w_tmp(nb);
        MatXcf V_tmp(nb, nb);
        for (std::size_t i = 0; i < nb; ++i) {
            w_tmp[i] = w_out[order[i]];
            V_tmp.col(i) = VOut.col(order[i]);
        }
        for (std::size_t i = 0; i < nb; ++i) w_out[i] = w_tmp[i];
        VOut = V_tmp;
    }
}

inline void eigh_block_sizes_batched(const c64* M, f32* w, c64* V,
                                     std::size_t nk1, std::size_t nk2, std::size_t nb,
                                     const std::vector<std::size_t>& sizes,
                                     bool sort) {
    const std::size_t nk = nk1 * nk2;
    const std::size_t nb2 = nb * nb;
    for (std::size_t k = 0; k < nk; ++k)
        eigh_block_sizes_one(M + k * nb2, nb, sizes, sort,
                             w + k * nb, V + k * nb2);
}

// Maximum |M[i,j]| with (i,j) in the off-block region defined by `sizes`.
inline f32 max_offblock_sizes(const c64* M, std::size_t nk, std::size_t nb,
                              const std::vector<std::size_t>& sizes) {
    std::vector<bool> mask(nb * nb, true);
    std::size_t start = 0;
    for (std::size_t s : sizes) {
        for (std::size_t i = start; i < start + s; ++i)
            for (std::size_t j = start; j < start + s; ++j)
                mask[i * nb + j] = false;
        start += s;
    }
    f32 mx = 0.0;
    for (std::size_t k = 0; k < nk; ++k) {
        const c64* mk = M + k * nb * nb;
        for (std::size_t i = 0; i < nb * nb; ++i)
            if (mask[i]) mx = std::max(mx, std::abs(mk[i]));
    }
    return mx;
}

inline f32 max_abs(const c64* M, std::size_t n) {
    f32 mx = 0.0;
    for (std::size_t i = 0; i < n; ++i) mx = std::max(mx, std::abs(M[i]));
    return mx;
}

}  // namespace cpp_hf
