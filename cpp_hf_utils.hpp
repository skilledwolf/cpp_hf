#pragma once
#include <complex>
#include <vector>
#include <algorithm>
#include <cmath>

#include <Eigen/Core>

namespace cpp_hf_utils {

using cxd  = std::complex<double>;
using MatC = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

inline size_t offset(size_t nk2, size_t d, size_t k1, size_t k2, size_t i, size_t j) {
    return ((k1 * nk2 + k2) * d + i) * d + j; // C-order
}

inline double fermi(double x, double T) noexcept {
    const double tt = std::max(1e-12, std::abs(T));
    const double y  = std::clamp(x / tt, -40.0, 40.0);
    return 1.0 / (1.0 + std::exp(y));
}


inline double real_inner(const std::vector<cxd>& a, const std::vector<cxd>& b) {
    double s = 0.0; const size_t n = a.size();
    for (size_t i = 0; i < n; ++i) s += (a[i] * std::conj(b[i])).real();
    return s;
}

} // namespace cpp_hf_utils
