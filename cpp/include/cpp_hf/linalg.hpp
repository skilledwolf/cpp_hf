// Batched Hermitian eigendecomposition.
#pragma once

#include "cpp_hf/parallel.hpp"
#include "cpp_hf/types.hpp"

#include <Eigen/Dense>

#include <complex>
#include <cstring>
#include <vector>

namespace cpp_hf {

using MatXcf = Eigen::Matrix<c64, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;  // c64 == complex<double>
using MapMatXcf = Eigen::Map<MatXcf>;
using ConstMapMatXcf = Eigen::Map<const MatXcf>;

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
    parallel_for(nk, [&](std::size_t k) {
        eigh_one(M + k * nb2, nb, w + k * nb, V + k * nb2);
    });
}

}  // namespace cpp_hf
