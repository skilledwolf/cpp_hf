// HFKernel data container: precomputed arrays consumed by the C++ solvers.
#pragma once

#include "cpp_hf/types.hpp"

#include <cstring>
#include <stdexcept>
#include <vector>

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

    f32 T = 0.0;
    f32 weight_sum = 0.0;

    std::vector<c64> h;        // (nk1, nk2, nb, nb)
    std::vector<c64> VR;       // (nk1, nk2, dv1, dv2)  shifted (phase pre-absorbed)
    std::vector<c64> refP;     // (nk1, nk2, nb, nb)
    std::vector<f32> w2d;      // (nk1, nk2)
    std::vector<f32> HH;       // (nb, nb)

    std::size_t n_contact = 1;
    std::vector<f32> contact_g;          // (n_contact,)
    std::vector<c64> contact_Oi;         // (n_contact, nb, nb)
    std::vector<c64> contact_Oj;         // (n_contact, nb, nb)

    std::size_t nk() const { return nk1 * nk2; }
    std::size_t nb2() const { return nb * nb; }
    std::size_t n_dense() const { return nk() * nb2(); }
};

}  // namespace cpp_hf
