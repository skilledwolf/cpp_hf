// hartreefock.hpp - pure C++ Hartree–Fock iteration front-end (no pybind)
#pragma once

#include <vector>
#include <complex>
#include <cstddef>

namespace hf {

using cxd = std::complex<double>;

struct HFResult {
    std::vector<cxd> P;   // (nk1*nk2*d*d)
    std::vector<cxd> F;   // (nk1*nk2*d*d)
    double energy = 0.0;
    double mu = 0.0;
    std::size_t iters = 0;
};

HFResult hartreefock_iteration(
    const double* W,                              // (nk1,nk2) C-order
    const cxd* H,                                 // (nk1,nk2,d,d) C-order
    const cxd* V,                                 // (nk1,nk2,dv1,dv2) C-order
    std::size_t nk1, std::size_t nk2, std::size_t d,
    std::size_t dv1, std::size_t dv2,            // either (1,1) or (d,d)
    const cxd* P0,                                // (nk1,nk2,d,d) C-order
    double electron_density0,
    double T,
    std::size_t max_iter,
    double comm_tol,
    std::size_t diis_size,
    double mixing_alpha);

} // namespace hf
