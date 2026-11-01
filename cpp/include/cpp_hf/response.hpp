// Fixed-particle Fermi response and a low-rank diagonal-Hartree preconditioner.
#pragma once

#include "cpp_hf/solver_dm.hpp"

namespace cpp_hf {

// Frechet derivative of f_mu(F), including dmu so sum_k w_k tr(dP_k)=0.
// Handles degeneracies by the continuous divided-difference limit.  The
// caller provides the eigensystem/occupations of F; no eigensolve or Fock
// build is needed.  Input and output must not alias Q, eps, or p.
inline void fermi_response(const c64* dF, const c64* Q, const f32* eps,
                           const f32* p, const f32* w, f32 T,
                           std::size_t nk, std::size_t nb, c64* dP) {
    const std::size_t nb2 = nb * nb;
    const f32 temp = std::max(T, 1e-12);
    dm_internal::fock_in_orbital_basis(Q, dF, dP, nk, nb);
    double numerator = 0.0, denominator = 0.0;
    for (std::size_t k = 0; k < nk; ++k)
        for (std::size_t i = 0; i < nb; ++i) {
            const f32 pi = p[k * nb + i];
            const double curvature = w[k] * pi * (1.0 - pi) / temp;
            numerator += curvature * dP[k * nb2 + i * nb + i].real();
            denominator += curvature;
        }
    const f32 dmu = denominator > 0.0 ? numerator / denominator : 0.0;
    parallel_for(nk, [&](std::size_t k) {
        ConstMapMatXcf Qk(Q + k * nb2, nb, nb);
        MapMatXcf out(dP + k * nb2, nb, nb);
        MatXcf response = out;
        for (std::size_t i = 0; i < nb; ++i) {
            const f32 pi = p[k * nb + i];
            for (std::size_t j = 0; j < nb; ++j) {
                const f32 pj = p[k * nb + j];
                const f32 gap = eps[k * nb + i] - eps[k * nb + j];
                const f32 divided = std::abs(gap) > 1e-7 * temp
                    ? (pi - pj) / gap
                    : -0.5 * (pi * (1.0 - pi) + pj * (1.0 - pj)) / temp;
                response(i, j) = divided * (out(i, j) - (i == j ? dmu : 0.0));
            }
        }
        out = Qk * response * Qk.adjoint();
    });
    hermitize_inplace(dP, nk, nb);
}

// Woodbury inverse of I - chi_0 H.  H acts on integrated orbital charges.
// SVD keeps only the nonzero potential/charge modes of HH (usually the
// number of layers, not the number of bands).  The chemical-potential
// correction in chi_0 removes the uniform-charge mode automatically.
class HartreeResponsePreconditioner {
public:
    explicit HartreeResponsePreconditioner(const HFKernel& K, bool enabled = true) : K_(K) {
        if (!enabled || !K.include_hartree || K.superlattice_hartree_active) return;
        using RealMatrix = Eigen::Matrix<f32, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        Eigen::Map<const RealMatrix> HH(K.HH, K.nb, K.nb);
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(HH, Eigen::ComputeThinU | Eigen::ComputeThinV);
        if (svd.singularValues().size() == 0) return;
        const f32 cutoff = 1e-12 * svd.singularValues()[0];
        while (rank_ < K.nb && svd.singularValues()[rank_] > std::max(cutoff, 0.0)) ++rank_;
        U_ = svd.matrixU().leftCols(rank_);
        V_ = svd.matrixV().leftCols(rank_);
        values_ = svd.singularValues().head(rank_);
    }

    std::size_t rank() const { return rank_; }

    void apply(const c64* R, const c64* Q, const f32* eps, const f32* p, c64* out) const {
        const auto nk = K_.nk(), nb = K_.nb, nb2 = nb * nb, n = K_.n_dense();
        std::memcpy(out, R, n * sizeof(c64));
        if (!rank_) return;
        std::vector<c64> potential(n, 0.0), responses(rank_ * n);
        Eigen::MatrixXd C(rank_, rank_);
        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(rank_);
        for (std::size_t k = 0; k < nk; ++k)
            for (std::size_t i = 0; i < nb; ++i)
                for (std::size_t a = 0; a < rank_; ++a)
                    rhs[a] += K_.w2d[k] * V_(i, a) * R[k * nb2 + i * nb + i].real();
        for (std::size_t b = 0; b < rank_; ++b) {
            for (std::size_t k = 0; k < nk; ++k)
                for (std::size_t i = 0; i < nb; ++i)
                    potential[k * nb2 + i * nb + i] = U_(i, b);
            c64* chi = responses.data() + b * n;
            fermi_response(potential.data(), Q, eps, p, K_.w2d, K_.T, nk, nb, chi);
            for (std::size_t a = 0; a < rank_; ++a) {
                double sum = 0.0;
                for (std::size_t k = 0; k < nk; ++k)
                    for (std::size_t i = 0; i < nb; ++i)
                        sum += K_.w2d[k] * V_(i, a) * chi[k * nb2 + i * nb + i].real();
                C(a, b) = sum;
            }
        }
        Eigen::MatrixXd J = Eigen::MatrixXd::Identity(rank_, rank_) - C * values_.asDiagonal();
        Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(J);
        qr.setThreshold(1e-12);
        const Eigen::VectorXd y = qr.solve(rhs);
        if (!y.allFinite() || (J * y - rhs).norm() > 1e-8 * std::max(rhs.norm(), 1e-30)) return;
        for (std::size_t b = 0; b < rank_; ++b)
            for (std::size_t i = 0; i < n; ++i)
                out[i] += values_[b] * y[b] * responses[b * n + i];
    }

private:
    HFKernel K_;
    std::size_t rank_ = 0;
    Eigen::MatrixXd U_, V_;
    Eigen::VectorXd values_;
};

}  // namespace cpp_hf
