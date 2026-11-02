// pybind11 bindings for the cpp_hf C++ core.  Thin wrappers — the algorithms
// live in cpp/include/cpp_hf/.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "cpp_hf/fock.hpp"
#include "cpp_hf/kernel.hpp"
#include "cpp_hf/linalg.hpp"
#include "cpp_hf/selfenergy.hpp"
#include "cpp_hf/solver_dm.hpp"
#include "cpp_hf/solver_rtr.hpp"
#include "cpp_hf/solver_scf.hpp"
#include "cpp_hf/solver_anderson.hpp"
#include "cpp_hf/solver_auto.hpp"
#include "cpp_hf/superlattice.hpp"
#include "cpp_hf/types.hpp"
#include "cpp_hf/utils.hpp"
#ifdef HF_HAVE_XLA_FFI
#include "cpp_hf/xla_ffi.hpp"
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;
using namespace cpp_hf;

namespace {

template <typename T>
py::array_t<T> move_array(std::vector<T>&& v, std::vector<py::ssize_t> shape) {
    auto owner = std::make_unique<std::vector<T>>(std::move(v));
    T* data = owner->data();
    py::capsule cap(owner.release(), [](void* p) {
        delete static_cast<std::vector<T>*>(p);
    });
    return py::array_t<T>(shape, data, cap);
}

std::string shape_string(const py::array& array) {
    std::ostringstream out;
    out << '(';
    for (py::ssize_t axis = 0; axis < array.ndim(); ++axis) {
        if (axis) out << ", ";
        out << array.shape(axis);
    }
    if (array.ndim() == 1) out << ',';
    out << ')';
    return out.str();
}

void require_shape(
    const py::array& array,
    const std::vector<py::ssize_t>& expected,
    const char* name) {
    bool matches = array.ndim() == static_cast<py::ssize_t>(expected.size());
    if (matches) {
        for (py::ssize_t axis = 0; axis < array.ndim(); ++axis) {
            if (array.shape(axis) != expected[static_cast<std::size_t>(axis)]) {
                matches = false;
                break;
            }
        }
    }
    if (!matches) {
        std::ostringstream message;
        message << name << " must have shape (";
        for (std::size_t axis = 0; axis < expected.size(); ++axis) {
            if (axis) message << ", ";
            message << expected[axis];
        }
        if (expected.size() == 1) message << ',';
        message << "); got " << shape_string(array);
        throw std::invalid_argument(message.str());
    }
}

void require_dense_shape(
    const py::array& array,
    std::size_t nk1,
    std::size_t nk2,
    std::size_t nb,
    const char* name) {
    require_shape(
        array,
        {
            static_cast<py::ssize_t>(nk1),
            static_cast<py::ssize_t>(nk2),
            static_cast<py::ssize_t>(nb),
            static_cast<py::ssize_t>(nb),
        },
        name);
}

void require_particle_count(const HFKernel& kernel, f32 n_e) {
    if (!std::isfinite(n_e))
        throw std::invalid_argument("n_e must be finite");
    const f32 capacity =
        kernel.weight_sum * static_cast<f32>(kernel.nb);
    const f32 tolerance =
        std::max<f32>(1.0e-8, 1.0e-6 * std::max<f32>(1.0, std::abs(capacity)));
    if (n_e < -tolerance || n_e > capacity + tolerance)
        throw std::invalid_argument(
            "n_e is outside the physically reachable range");
}

void require_block_sizes(
    const std::vector<std::size_t>& sizes,
    std::size_t nb) {
    if (sizes.empty()) return;
    std::size_t total = 0;
    for (const std::size_t size : sizes) {
        if (size == 0)
            throw std::invalid_argument(
                "block_sizes entries must be positive");
        if (size > nb - std::min(total, nb))
            throw std::invalid_argument(
                "block_sizes must sum to nb");
        total += size;
    }
    if (total != nb)
        throw std::invalid_argument("block_sizes must sum to nb");
}

// Leading-axis layout shared by the per-band test hooks: ``bands`` is
// (..., nb) and ``weights`` carries its leading axes.  Returns (nk, nb).
std::pair<std::size_t, std::size_t> band_layout(
    const py::array& bands, const py::array& weights,
    const char* bands_name, const char* weights_name) {
    if (bands.ndim() < 1 || bands.shape(bands.ndim() - 1) <= 0)
        throw std::invalid_argument(
            std::string(bands_name) +
            " must have at least one dimension and a non-empty band axis");
    bool matches = weights.ndim() == bands.ndim() - 1;
    for (py::ssize_t axis = 0; matches && axis < weights.ndim(); ++axis)
        matches = weights.shape(axis) == bands.shape(axis);
    if (!matches)
        throw std::invalid_argument(
            std::string(weights_name) + " shape must match " + bands_name +
            " without the final band axis");
    std::size_t nk = 1;
    for (py::ssize_t axis = 0; axis < bands.ndim() - 1; ++axis)
        nk *= static_cast<std::size_t>(bands.shape(axis));
    return {nk, static_cast<std::size_t>(bands.shape(bands.ndim() - 1))};
}

// Build a HFKernel from raw NumPy arrays.  Validate again at the native
// boundary: the binding is public, and the solvers use fixed-size pointer
// arithmetic that must not rely on a particular Python wrapper having run.
HFKernel make_kernel(
    py::array_t<f32, py::array::c_style> w2d,
    py::array_t<c64, py::array::c_style> h,
    py::array_t<c64, py::array::c_style> VR_shifted,
    py::array_t<c64, py::array::c_style> refP,
    bool has_refP,
    py::array_t<f32, py::array::c_style> HH,
    py::array_t<f32, py::array::c_style> contact_g,
    py::array_t<c64, py::array::c_style> contact_Oi,
    py::array_t<c64, py::array::c_style> contact_Oj,
    f32 weight_sum,
    f32 T,
    bool include_hartree,
    bool include_exchange,
    bool exchange_hcp) {

    if (h.ndim() != 4) throw std::invalid_argument("h must be (nk1,nk2,nb,nb)");
    if (h.shape(0) <= 0 || h.shape(1) <= 0 || h.shape(2) <= 0)
        throw std::invalid_argument("h dimensions must be positive");
    HFKernel K;
    K.nk1 = static_cast<std::size_t>(h.shape(0));
    K.nk2 = static_cast<std::size_t>(h.shape(1));
    K.nb = static_cast<std::size_t>(h.shape(2));
    if (static_cast<std::size_t>(h.shape(3)) != K.nb)
        throw std::invalid_argument("h last two axes must be square (nb,nb)");
    require_shape(
        w2d,
        {
            static_cast<py::ssize_t>(K.nk1),
            static_cast<py::ssize_t>(K.nk2),
        },
        "w2d");
    f32 computed_weight_sum = 0.0;
    for (std::size_t k = 0; k < K.nk(); ++k) {
        if (!std::isfinite(w2d.data()[k]))
            throw std::invalid_argument("w2d must be finite");
        if (w2d.data()[k] < 0.0)
            throw std::invalid_argument("w2d must be non-negative");
        computed_weight_sum += w2d.data()[k];
    }
    if (!std::isfinite(computed_weight_sum) || computed_weight_sum <= 0.0)
        throw std::invalid_argument(
            "w2d must have a finite positive sum");
    const f32 weight_sum_tolerance =
        1.0e-8 * std::max<f32>(1.0, std::abs(computed_weight_sum));
    if (std::abs(weight_sum - computed_weight_sum) > weight_sum_tolerance)
        throw std::invalid_argument(
            "weight_sum must equal sum(w2d)");
    if (VR_shifted.ndim() != 4 ||
        static_cast<std::size_t>(VR_shifted.shape(0)) != K.nk1 ||
        static_cast<std::size_t>(VR_shifted.shape(1)) != K.nk2) {
        throw std::invalid_argument(
            "VR must have shape (nk1,nk2,1,1) or (nk1,nk2,nb,nb)");
    }
    K.dv1 = static_cast<std::size_t>(VR_shifted.shape(2));
    K.dv2 = static_cast<std::size_t>(VR_shifted.shape(3));
    if (!((K.dv1 == 1 && K.dv2 == 1) || (K.dv1 == K.nb && K.dv2 == K.nb)))
        throw std::invalid_argument("VR last two dims must be (1,1) or (nb,nb)");
    if (exchange_hcp && (K.dv1 != 1 || K.dv2 != 1))
        throw std::invalid_argument(
            "exchange_hcp requires VR with shape (nk1,nk2,1,1)");
    require_shape(
        HH,
        {
            static_cast<py::ssize_t>(K.nb),
            static_cast<py::ssize_t>(K.nb),
        },
        "HH");
    if (!std::isfinite(weight_sum) || weight_sum <= 0.0)
        throw std::invalid_argument("weight_sum must be finite and positive");
    if (!std::isfinite(T) || T < 0.0)
        throw std::invalid_argument("T must be finite and non-negative");

    K.include_hartree = include_hartree;
    K.include_exchange = include_exchange;
    K.exchange_hcp = exchange_hcp;
    K.has_refP = has_refP;
    K.T = T;
    K.weight_sum = weight_sum;

    if (has_refP) {
        if (refP.ndim() != 4 ||
            static_cast<std::size_t>(refP.shape(0)) != K.nk1 ||
            static_cast<std::size_t>(refP.shape(1)) != K.nk2 ||
            static_cast<std::size_t>(refP.shape(2)) != K.nb ||
            static_cast<std::size_t>(refP.shape(3)) != K.nb) {
            throw std::invalid_argument("refP must match h shape when has_refP is true");
        }
    } else if (refP.size() != 0) {
        throw std::invalid_argument(
            "refP must be empty when has_refP is false");
    }

    K.h = h.data();
    K.VR = VR_shifted.data();
    K.refP = has_refP ? refP.data() : nullptr;
    K.w2d = w2d.data();
    K.HH = HH.data();

    if (contact_g.ndim() != 1) throw std::invalid_argument("contact_g must be 1-D");
    K.n_contact = static_cast<std::size_t>(contact_g.shape(0));
    require_shape(
        contact_Oi,
        {
            static_cast<py::ssize_t>(K.n_contact),
            static_cast<py::ssize_t>(K.nb),
            static_cast<py::ssize_t>(K.nb),
        },
        "contact_Oi");
    require_shape(
        contact_Oj,
        {
            static_cast<py::ssize_t>(K.n_contact),
            static_cast<py::ssize_t>(K.nb),
            static_cast<py::ssize_t>(K.nb),
        },
        "contact_Oj");
    K.contact_g = contact_g.data();
    K.contact_Oi = contact_Oi.data();
    K.contact_Oj = contact_Oj.data();

    return K;
}

// Inject the optional superlattice Fock / Hartree configuration from a
// kernel_args dict into a HFKernel.  No-op when the dict lacks a truthy
// ``superlattice_fock_active`` / ``superlattice_hartree_active`` flag.  All
// array pointers are sourced from the kernel_args dict so they stay alive
// for the duration of the surrounding solver call (caller holds the dict).
void inject_superlattice_into_kernel(HFKernel& K, py::object kernel_args) {
    const bool fock_active = kernel_args.contains("superlattice_fock_active")
        ? py::cast<bool>(kernel_args["superlattice_fock_active"])
        : false;
    const bool hartree_active = kernel_args.contains("superlattice_hartree_active")
        ? py::cast<bool>(kernel_args["superlattice_hartree_active"])
        : false;
    if (!fock_active && !hartree_active) return;

    K.superlattice_fock_active = fock_active;
    K.superlattice_hartree_active = hartree_active;

    if (kernel_args.contains("hartree_degeneracy")) {
        K.hartree_degeneracy = py::cast<f32>(kernel_args["hartree_degeneracy"]);
    }
    K.n_G = py::cast<std::size_t>(kernel_args["n_G"]);
    K.dim_orb = py::cast<std::size_t>(kernel_args["dim_orb"]);
    K.n_delta = py::cast<std::size_t>(kernel_args["n_delta"]);
    K.N_ext_x = py::cast<std::size_t>(kernel_args["N_ext_x"]);
    K.N_ext_y = py::cast<std::size_t>(kernel_args["N_ext_y"]);

    if (K.n_G == 0 || K.dim_orb == 0 || K.n_delta == 0 ||
        K.N_ext_x == 0 || K.N_ext_y == 0) {
        throw std::invalid_argument(
            "superlattice dimensions must be positive");
    }
    if (K.n_G > K.nb / K.dim_orb ||
        K.n_G * K.dim_orb != K.nb) {
        throw std::invalid_argument(
            "superlattice mode requires n_G × dim_orb == nb");
    }
    if (fock_active) {
        const f32 first_weight = K.w2d[0];
        const f32 tolerance =
            1.0e-12 * std::max<f32>(1.0, std::abs(first_weight));
        for (std::size_t k = 1; k < K.nk(); ++k) {
            if (std::abs(K.w2d[k] - first_weight) > tolerance) {
                throw std::invalid_argument(
                    "superlattice Fock currently requires uniform k-point weights");
            }
        }
    }

    auto V_lag_fft = py::cast<py::array_t<c64, py::array::c_style>>(
        kernel_args["V_lag_fft"]);
    auto g_a_off = py::cast<py::array_t<std::int64_t, py::array::c_style>>(
        kernel_args["g_a_off"]);
    auto pair_i = py::cast<py::array_t<std::int64_t, py::array::c_style>>(
        kernel_args["pair_i"]);
    auto pair_j = py::cast<py::array_t<std::int64_t, py::array::c_style>>(
        kernel_args["pair_j"]);
    auto pair_start = py::cast<py::array_t<std::int64_t, py::array::c_style>>(
        kernel_args["pair_start"]);
    auto pair_to_delta = py::cast<py::array_t<std::int64_t, py::array::c_style>>(
        kernel_args["pair_to_delta"]);

    if (V_lag_fft.ndim() != 2 ||
        static_cast<std::size_t>(V_lag_fft.shape(0)) != K.N_ext_x ||
        static_cast<std::size_t>(V_lag_fft.shape(1)) != K.N_ext_y) {
        throw std::invalid_argument(
            "V_lag_fft must have shape (N_ext_x, N_ext_y)");
    }
    if (g_a_off.ndim() != 2 ||
        static_cast<std::size_t>(g_a_off.shape(0)) != K.n_G ||
        g_a_off.shape(1) != 2) {
        throw std::invalid_argument("g_a_off must be (n_G, 2)");
    }
    if (pair_i.ndim() != 1 || pair_j.ndim() != 1)
        throw std::invalid_argument("pair_i and pair_j must be 1-D");
    if (pair_start.ndim() != 1 ||
        static_cast<std::size_t>(pair_start.shape(0)) != K.n_delta + 1) {
        throw std::invalid_argument("pair_start must have length n_delta + 1");
    }
    if (pair_i.shape(0) != pair_j.shape(0)) {
        throw std::invalid_argument("pair_i and pair_j must have the same length");
    }
    if (pair_to_delta.ndim() != 2 ||
        static_cast<std::size_t>(pair_to_delta.shape(0)) != K.n_G ||
        static_cast<std::size_t>(pair_to_delta.shape(1)) != K.n_G) {
        throw std::invalid_argument("pair_to_delta must be (n_G, n_G)");
    }
    const auto n_pairs = static_cast<std::size_t>(pair_i.shape(0));
    if (pair_start.data()[0] != 0 ||
        pair_start.data()[K.n_delta] != static_cast<std::int64_t>(n_pairs)) {
        throw std::invalid_argument(
            "pair_start must begin at 0 and end at len(pair_i)");
    }
    for (std::size_t d = 0; d < K.n_delta; ++d) {
        if (pair_start.data()[d] < 0 ||
            pair_start.data()[d] > pair_start.data()[d + 1]) {
            throw std::invalid_argument(
                "pair_start must be non-negative and non-decreasing");
        }
    }
    for (std::size_t p = 0; p < n_pairs; ++p) {
        if (pair_i.data()[p] < 0 ||
            pair_i.data()[p] >= static_cast<std::int64_t>(K.n_G) ||
            pair_j.data()[p] < 0 ||
            pair_j.data()[p] >= static_cast<std::int64_t>(K.n_G)) {
            throw std::invalid_argument(
                "pair_i and pair_j entries must index [0, n_G)");
        }
    }
    for (std::size_t i = 0; i < K.n_G; ++i) {
        const std::int64_t ox = g_a_off.data()[2 * i];
        const std::int64_t oy = g_a_off.data()[2 * i + 1];
        const bool x_outside =
            ox < 0 || K.N_ext_x < K.nk1 ||
            static_cast<std::size_t>(ox) >
                (K.N_ext_x - K.nk1) / K.nk1;
        const bool y_outside =
            oy < 0 || K.N_ext_y < K.nk2 ||
            static_cast<std::size_t>(oy) >
                (K.N_ext_y - K.nk2) / K.nk2;
        if (x_outside || y_outside) {
            throw std::invalid_argument(
                "g_a_off entries place a k tile outside the extended grid");
        }
    }
    for (std::size_t p = 0; p < K.n_G * K.n_G; ++p) {
        if (pair_to_delta.data()[p] < 0 ||
            pair_to_delta.data()[p] >=
                static_cast<std::int64_t>(K.n_delta)) {
            throw std::invalid_argument(
                "pair_to_delta entries must index [0, n_delta)");
        }
    }

    K.V_lag_fft = V_lag_fft.data();
    K.g_a_off = g_a_off.data();
    K.pair_i = pair_i.data();
    K.pair_j = pair_j.data();
    K.pair_start = pair_start.data();
    K.pair_to_delta = pair_to_delta.data();

    if (kernel_args.contains("V_lag_fft_orbital")) {
        auto V_orb = py::cast<py::array_t<c64, py::array::c_style>>(
            kernel_args["V_lag_fft_orbital"]);
        if (V_orb.ndim() != 4 ||
            static_cast<std::size_t>(V_orb.shape(0)) != K.N_ext_x ||
            static_cast<std::size_t>(V_orb.shape(1)) != K.N_ext_y ||
            static_cast<std::size_t>(V_orb.shape(2)) != K.dim_orb ||
            static_cast<std::size_t>(V_orb.shape(3)) != K.dim_orb) {
            throw std::invalid_argument(
                "V_lag_fft_orbital must be "
                "(N_ext_x, N_ext_y, dim_orb, dim_orb)");
        }
        K.V_lag_fft_orbital = V_orb.data();
    }

    if (hartree_active) {
        auto HH_GG = py::cast<py::array_t<f32, py::array::c_style>>(
            kernel_args["HH_GG"]);
        if (HH_GG.ndim() != 2 ||
            static_cast<std::size_t>(HH_GG.shape(0)) != K.n_G ||
            static_cast<std::size_t>(HH_GG.shape(1)) != K.n_G) {
            throw std::invalid_argument("HH_GG must be (n_G, n_G)");
        }
        K.HH_GG = HH_GG.data();

        if (kernel_args.contains("HH_GG_orbital")) {
            auto HH_orb = py::cast<py::array_t<f32, py::array::c_style>>(
                kernel_args["HH_GG_orbital"]);
            if (HH_orb.ndim() != 4 ||
                static_cast<std::size_t>(HH_orb.shape(0)) != K.n_G ||
                static_cast<std::size_t>(HH_orb.shape(1)) != K.n_G ||
                static_cast<std::size_t>(HH_orb.shape(2)) != K.dim_orb ||
                static_cast<std::size_t>(HH_orb.shape(3)) != K.dim_orb) {
                throw std::invalid_argument(
                    "HH_GG_orbital must be (n_G, n_G, dim_orb, dim_orb)");
            }
            K.HH_GG_orbital = HH_orb.data();
        }
    }
}

ProjectFn wrap_project_fn(py::object py_fn) {
    if (py_fn.is_none()) return ProjectFn();
    return [py_fn](c64* M, std::size_t nk1, std::size_t nk2, std::size_t nb) {
        py::gil_scoped_acquire gil;
        std::vector<py::ssize_t> shape{
            static_cast<py::ssize_t>(nk1),
            static_cast<py::ssize_t>(nk2),
            static_cast<py::ssize_t>(nb),
            static_cast<py::ssize_t>(nb),
        };
        py::array_t<c64> view(shape, M);  // wraps existing buffer (no copy)
        py::object result = py_fn(view);
        py::array_t<c64, py::array::c_style | py::array::forcecast> out =
            py::cast<py::array_t<c64, py::array::c_style | py::array::forcecast>>(result);
        if (out.ndim() != 4 ||
            static_cast<std::size_t>(out.shape(0)) != nk1 ||
            static_cast<std::size_t>(out.shape(1)) != nk2 ||
            static_cast<std::size_t>(out.shape(2)) != nb ||
            static_cast<std::size_t>(out.shape(3)) != nb) {
            throw std::runtime_error("project_fn returned an array of wrong shape");
        }
        std::memcpy(M, out.data(), nk1 * nk2 * nb * nb * sizeof(c64));
    };
}

}  // namespace

PYBIND11_MODULE(_native, m) {
    m.doc() = "cpp_hf native core (FFTW + Eigen + pybind11)";

    // --- Self-energy / FFT ---
    m.def("selfenergy_fft_full",
          [](py::array_t<c64, py::array::c_style | py::array::forcecast> VR,
             py::array_t<c64, py::array::c_style | py::array::forcecast> P,
             bool apply_ifftshift, bool hcp) {
              if (P.ndim() != 4) throw std::invalid_argument("P must be (nk1,nk2,nb,nb)");
              if (VR.ndim() != 4) throw std::invalid_argument("VR must be (nk1,nk2,dv1,dv2)");
              if (P.shape(0) <= 0 || P.shape(1) <= 0 || P.shape(2) <= 0)
                  throw std::invalid_argument("P dimensions must be positive");
              if (P.shape(3) != P.shape(2))
                  throw std::invalid_argument("P last two dimensions must be square");
              const std::size_t nk1 = P.shape(0);
              const std::size_t nk2 = P.shape(1);
              const std::size_t nb = P.shape(2);
              if (VR.shape(0) != P.shape(0) || VR.shape(1) != P.shape(1))
                  throw std::invalid_argument(
                      "VR leading dimensions must match P");
              const std::size_t dv1 = VR.shape(2);
              const std::size_t dv2 = VR.shape(3);
              if (!((dv1 == 1 && dv2 == 1) ||
                    (dv1 == nb && dv2 == nb))) {
                  throw std::invalid_argument(
                      "VR last two dimensions must be (1,1) or match P");
              }
              std::vector<c64> sigma(nk1 * nk2 * nb * nb);
              {
                  py::gil_scoped_release release;
                  if (hcp) {
                      if (dv1 != 1 || dv2 != 1)
                          throw std::invalid_argument(
                              "hermitian_channel_packing requires VR with shape (...,1,1)");
                      selfenergy_fft_full_hcp(P.data(), sigma.data(), VR.data(), nk1, nk2, nb);
                  } else {
                      selfenergy_fft_full(P.data(), sigma.data(), VR.data(),
                                           nk1, nk2, nb, dv1, dv2);
                  }
                  if (apply_ifftshift) {
                      ifftshift_2d_batch(sigma.data(), nk1, nk2, nb * nb);
                  }
              }
              return move_array(std::move(sigma), {(py::ssize_t)nk1, (py::ssize_t)nk2,
                                                    (py::ssize_t)nb, (py::ssize_t)nb});
          },
          py::arg("VR"), py::arg("P"),
          py::arg("apply_ifftshift") = true,
          py::arg("hermitian_channel_packing") = false);

    m.def("clear_fft_plan_cache",
          []() {
              py::gil_scoped_release release;
              FftPlanCache::instance().clear();
          });

    // --- XLA FFI targets for the JAX backend (empty without the headers) ---
    m.def("xla_ffi_targets", []() {
        py::dict targets;
#ifdef HF_HAVE_XLA_FFI
        targets["hartreefock_eigh_c128"] =
            py::capsule(reinterpret_cast<void*>(cpp_hf::xla_ffi::EighC128));
        targets["hartreefock_eigvalsh_c128"] =
            py::capsule(reinterpret_cast<void*>(cpp_hf::xla_ffi::EigvalshC128));
        targets["hartreefock_matmul_c128"] =
            py::capsule(reinterpret_cast<void*>(cpp_hf::xla_ffi::MatmulC128));
        targets["hartreefock_superlattice_fock"] =
            py::capsule(reinterpret_cast<void*>(cpp_hf::xla_ffi::SuperlatticeFock));
#endif
        return targets;
    });

    // --- Hermitian eigh (batched + block-sizes) ---
    m.def("eigh_batched",
          [](py::array_t<c64, py::array::c_style | py::array::forcecast> M) {
              if (M.ndim() < 2) throw std::invalid_argument("M needs at least 2 dims");
              std::size_t nb = M.shape(M.ndim() - 1);
              if (static_cast<std::size_t>(M.shape(M.ndim() - 2)) != nb)
                  throw std::invalid_argument("M last two dims must be square");
              std::size_t nk = 1;
              for (int i = 0; i < M.ndim() - 2; ++i) nk *= M.shape(i);
              std::vector<f32> w(nk * nb);
              std::vector<c64> V(nk * nb * nb);
              {
                  py::gil_scoped_release release;
                  eigh_batched(M.data(), w.data(), V.data(), nk, 1, nb);
              }
              std::vector<py::ssize_t> w_shape, v_shape;
              for (int i = 0; i < M.ndim() - 2; ++i) {
                  w_shape.push_back(M.shape(i));
                  v_shape.push_back(M.shape(i));
              }
              w_shape.push_back(nb);
              v_shape.push_back(nb);
              v_shape.push_back(nb);
              return py::make_tuple(move_array(std::move(w), w_shape),
                                    move_array(std::move(V), v_shape));
          },
          py::arg("M"));

    m.def("eigh_block_sizes",
          [](py::array_t<c64, py::array::c_style | py::array::forcecast> M,
             std::vector<std::size_t> sizes,
             bool sort) {
              if (M.ndim() < 2) throw std::invalid_argument("M needs at least 2 dims");
              std::size_t nb = M.shape(M.ndim() - 1);
              std::size_t total = 0;
              for (auto s : sizes) total += s;
              if (total != nb) throw std::invalid_argument("block sizes must sum to nb");
              std::size_t nk = 1;
              for (int i = 0; i < M.ndim() - 2; ++i) nk *= M.shape(i);
              std::vector<f32> w(nk * nb);
              std::vector<c64> V(nk * nb * nb);
              {
                  py::gil_scoped_release release;
                  eigh_block_sizes_batched(M.data(), w.data(), V.data(),
                                            nk, 1, nb, sizes, sort);
              }
              std::vector<py::ssize_t> w_shape, v_shape;
              for (int i = 0; i < M.ndim() - 2; ++i) {
                  w_shape.push_back(M.shape(i));
                  v_shape.push_back(M.shape(i));
              }
              w_shape.push_back(nb);
              v_shape.push_back(nb);
              v_shape.push_back(nb);
              return py::make_tuple(move_array(std::move(w), w_shape),
                                    move_array(std::move(V), v_shape));
          },
          py::arg("M"), py::arg("sizes"), py::arg("sort") = true);

    m.def("max_offblock_sizes",
          [](py::array_t<c64, py::array::c_style | py::array::forcecast> M,
             std::vector<std::size_t> sizes) {
              if (M.ndim() < 2) throw std::invalid_argument("M needs at least 2 dims");
              std::size_t nb = M.shape(M.ndim() - 1);
              std::size_t nk = 1;
              for (int i = 0; i < M.ndim() - 2; ++i) nk *= M.shape(i);
              f32 mx = 0.0;
              {
                  py::gil_scoped_release release;
                  mx = max_offblock_sizes(M.data(), nk, nb, sizes);
              }
              return mx;
          });

    // --- μ solvers ---
    m.def("find_mu_bisection",
          [](py::array_t<f32, py::array::c_style | py::array::forcecast> bands,
             py::array_t<f32, py::array::c_style | py::array::forcecast> weights,
             f32 n_e, f32 T, int maxiter) {
              if (bands.ndim() < 1 || bands.shape(bands.ndim() - 1) <= 0)
                  throw std::invalid_argument(
                      "bands must have at least one dimension and a non-empty band axis");
              if (weights.ndim() != bands.ndim() - 1)
                  throw std::invalid_argument(
                      "weights shape must match bands without the final band axis");
              for (py::ssize_t axis = 0; axis < weights.ndim(); ++axis) {
                  if (weights.shape(axis) != bands.shape(axis))
                      throw std::invalid_argument(
                          "weights shape must match bands without the final band axis");
              }
              if (!std::isfinite(T) || T < 0.0)
                  throw std::invalid_argument("T must be finite and non-negative");
              if (!std::isfinite(n_e))
                  throw std::invalid_argument("n_e must be finite");
              if (maxiter < 0)
                  throw std::invalid_argument("maxiter must be non-negative");
              std::size_t nk = 1;
              for (int i = 0; i < bands.ndim() - 1; ++i) nk *= bands.shape(i);
              std::size_t nb = bands.shape(bands.ndim() - 1);
              f32 weight_sum = 0.0;
              for (std::size_t k = 0; k < nk; ++k) {
                  const f32 weight = weights.data()[k];
                  if (!std::isfinite(weight))
                      throw std::invalid_argument("weights must be finite");
                  weight_sum += weight;
              }
              const f32 capacity = weight_sum * static_cast<f32>(nb);
              const f32 count_tol =
                  std::max<f32>(1.0e-8, 1.0e-6 * std::max<f32>(1.0, std::abs(capacity)));
              if (!std::isfinite(weight_sum) || weight_sum <= 0.0)
                  throw std::invalid_argument(
                      "weights must have a finite positive sum");
              if (n_e < -count_tol || n_e > capacity + count_tol)
                  throw std::invalid_argument(
                      "n_e is outside the physically reachable range");
              f32 mu = 0.0;
              {
                  py::gil_scoped_release release;
                  mu = find_mu_bisection(bands.data(), nk * nb,
                                          weights.data(), nk, nb, n_e, T, maxiter);
              }
              return mu;
          },
          py::arg("bands"), py::arg("weights"),
          py::arg("n_e"), py::arg("T"), py::arg("maxiter") = 0);

    // --- Build_fock & energy ---
    m.def("build_fock_apply",
          [](py::object kernel_args, py::array_t<c64, py::array::c_style | py::array::forcecast> P,
             py::object project_fn) {
              auto K = make_kernel(
                  py::cast<py::array_t<f32, py::array::c_style>>(kernel_args["w2d"]),
                  py::cast<py::array_t<c64, py::array::c_style>>(kernel_args["h"]),
                  py::cast<py::array_t<c64, py::array::c_style>>(kernel_args["VR"]),
                  py::cast<py::array_t<c64, py::array::c_style>>(kernel_args["refP"]),
                  py::cast<bool>(kernel_args["has_refP"]),
                  py::cast<py::array_t<f32, py::array::c_style>>(kernel_args["HH"]),
                  py::cast<py::array_t<f32, py::array::c_style>>(kernel_args["contact_g"]),
                  py::cast<py::array_t<c64, py::array::c_style>>(kernel_args["contact_Oi"]),
                  py::cast<py::array_t<c64, py::array::c_style>>(kernel_args["contact_Oj"]),
                  py::cast<f32>(kernel_args["weight_sum"]),
                  py::cast<f32>(kernel_args["T"]),
                  py::cast<bool>(kernel_args["include_hartree"]),
                  py::cast<bool>(kernel_args["include_exchange"]),
                  py::cast<bool>(kernel_args["exchange_hcp"]));
              inject_superlattice_into_kernel(K, kernel_args);
              require_dense_shape(P, K.nk1, K.nk2, K.nb, "P");
              const std::size_t n_tot = K.n_dense();
              std::vector<c64> Sigma(n_tot), Hh(n_tot), F(n_tot);
              ProjectFn pf = wrap_project_fn(project_fn);
              const ProjectFn* pfp = pf ? &pf : nullptr;
              f32 E = 0.0;
              {
                  py::gil_scoped_release release;
                  build_fock(K, P.data(), Sigma.data(), Hh.data(), F.data(), pfp);
                  E = hf_energy(K, P.data(), Sigma.data(), Hh.data());
              }
              std::vector<py::ssize_t> shape{(py::ssize_t)K.nk1, (py::ssize_t)K.nk2,
                                              (py::ssize_t)K.nb, (py::ssize_t)K.nb};
              return py::make_tuple(move_array(std::move(Sigma), shape),
                                     move_array(std::move(Hh), shape),
                                     move_array(std::move(F), shape),
                                     E);
          });

    // --- SCF solver ---
    m.def("solve_scf",
          [](py::object kernel_args,
             py::array_t<c64, py::array::c_style | py::array::forcecast> P0,
             f32 n_e,
             std::size_t max_iter,
             f32 density_tol, f32 comm_tol,
             f32 mixing, f32 level_shift,
             py::object project_fn,
             std::vector<std::size_t> block_sizes,
             std::string acceleration,
             std::size_t diis_size, std::size_t diis_start,
             f32 diis_damping, f32 trust_radius,
             bool return_density, bool return_fock) {
              auto K = make_kernel(
                  py::cast<py::array_t<f32, py::array::c_style>>(kernel_args["w2d"]),
                  py::cast<py::array_t<c64, py::array::c_style>>(kernel_args["h"]),
                  py::cast<py::array_t<c64, py::array::c_style>>(kernel_args["VR"]),
                  py::cast<py::array_t<c64, py::array::c_style>>(kernel_args["refP"]),
                  py::cast<bool>(kernel_args["has_refP"]),
                  py::cast<py::array_t<f32, py::array::c_style>>(kernel_args["HH"]),
                  py::cast<py::array_t<f32, py::array::c_style>>(kernel_args["contact_g"]),
                  py::cast<py::array_t<c64, py::array::c_style>>(kernel_args["contact_Oi"]),
                  py::cast<py::array_t<c64, py::array::c_style>>(kernel_args["contact_Oj"]),
                  py::cast<f32>(kernel_args["weight_sum"]),
                  py::cast<f32>(kernel_args["T"]),
                  py::cast<bool>(kernel_args["include_hartree"]),
                  py::cast<bool>(kernel_args["include_exchange"]),
                  py::cast<bool>(kernel_args["exchange_hcp"]));
              inject_superlattice_into_kernel(K, kernel_args);
              require_dense_shape(P0, K.nk1, K.nk2, K.nb, "P0");
              require_particle_count(K, n_e);
              require_block_sizes(block_sizes, K.nb);
              if (max_iter == 0)
                  throw std::invalid_argument("max_iter must be positive");
              if (!std::isfinite(density_tol) || density_tol < 0.0 ||
                  !std::isfinite(comm_tol) || comm_tol < 0.0)
                  throw std::invalid_argument(
                      "SCF tolerances must be finite and non-negative");
              if (!std::isfinite(mixing) || mixing < 0.0 || mixing > 1.0)
                  throw std::invalid_argument("mixing must be between 0 and 1");
              if (!std::isfinite(level_shift))
                  throw std::invalid_argument("level_shift must be finite");
              if (acceleration != "linear" && acceleration != "diis" &&
                  acceleration != "oda" && acceleration != "anderson" && acceleration != "rdiis")
                  throw std::invalid_argument(
                      "acceleration must be linear, diis, oda, or anderson");
              if (diis_size == 0)
                  throw std::invalid_argument("diis_size must be positive");
              if (!std::isfinite(diis_damping) ||
                  diis_damping < 0.0 || diis_damping > 1.0)
                  throw std::invalid_argument(
                      "diis_damping must be between 0 and 1");
              if (!std::isfinite(trust_radius) || trust_radius < 0.0)
                  throw std::invalid_argument(
                      "trust_radius must be finite and non-negative");
              SCFConfig cfg;
              cfg.max_iter = max_iter;
              cfg.density_tol = density_tol;
              cfg.comm_tol = comm_tol;
              cfg.mixing = mixing;
              cfg.level_shift = level_shift;
              cfg.block_sizes = block_sizes;
              cfg.acceleration = acceleration;
              if (kernel_args.contains("preconditioner"))
                  cfg.preconditioner = py::cast<std::string>(kernel_args["preconditioner"]);
              if (kernel_args.contains("safeguard"))
                  cfg.safeguard = py::cast<bool>(kernel_args["safeguard"]);
              if (cfg.preconditioner != "none" && cfg.preconditioner != "hartree")
                  throw std::invalid_argument("preconditioner must be none or hartree");
              if ((acceleration == "anderson" || acceleration == "rdiis") && level_shift != 0.0)
                  throw std::invalid_argument("Anderson requires level_shift=0");
              cfg.diis_size = diis_size;
              cfg.diis_start = diis_start;
              cfg.diis_damping = diis_damping;
              cfg.trust_radius = trust_radius;
              cfg.return_density = return_density;
              cfg.return_fock = return_fock;
              ProjectFn pf = wrap_project_fn(project_fn);
              const ProjectFn* pfp = pf ? &pf : nullptr;
              SCFResult res;
              std::size_t n_fock_builds = 0;
              K.fock_builds = &n_fock_builds;
              K.exchange_blocks = &cfg.block_sizes;
              {
                  py::gil_scoped_release release;
                  res = (acceleration == "anderson" || acceleration == "rdiis")
                      ? solve_anderson(K, P0.data(), n_e, cfg, pfp)
                      : solve_scf(K, P0.data(), n_e, cfg, pfp);
              }

              if (kernel_args.contains("diagnostics"))
                  kernel_args["diagnostics"]["fock_builds"] = py::int_(n_fock_builds);

              std::vector<py::ssize_t> dense_shape{(py::ssize_t)K.nk1, (py::ssize_t)K.nk2,
                                                    (py::ssize_t)K.nb, (py::ssize_t)K.nb};
              std::vector<py::ssize_t> hE_shape{(py::ssize_t)res.hist_E.size()};
              std::vector<py::ssize_t> hF_shape{(py::ssize_t)res.hist_free.size()};
              std::vector<py::ssize_t> hD_shape{(py::ssize_t)res.hist_density.size()};
              std::vector<py::ssize_t> hC_shape{(py::ssize_t)res.hist_comm.size()};
              py::object density_obj = py::none();
              py::object fock_obj = py::none();
              if (return_density) density_obj = move_array(std::move(res.density), dense_shape);
              if (return_fock) fock_obj = move_array(std::move(res.fock), dense_shape);
              return py::make_tuple(
                  density_obj,
                  fock_obj,
                  res.energy, res.mu, res.iterations, res.converged,
                  move_array(std::move(res.hist_E), hE_shape),
                  move_array(std::move(res.hist_free), hF_shape),
                  move_array(std::move(res.hist_density), hD_shape),
                  move_array(std::move(res.hist_comm), hC_shape));
          });

    // --- Direct minimization solver ---
    m.def("solve_dm",
          [](py::object kernel_args,
             py::array_t<c64, py::array::c_style | py::array::forcecast> P0,
             f32 n_e,
             std::size_t max_iter, f32 tol_E, f32 tol_grad,
             f32 max_step, f32 bt_shrink, f32 denom_scale,
             std::size_t bt_max, std::size_t cg_restart,
             std::size_t plateau_window, int mu_maxiter,
             int optimizer,
             f32 tr_delta0, std::size_t tr_cg_max,
             py::array_t<c64, py::array::c_style | py::array::forcecast> deflation_targets,
             f32 deflation_sigma, f32 deflation_length,
             std::vector<std::size_t> block_sizes,
             py::object project_fn,
             bool return_Q, bool return_density, bool return_fock) {
              auto K = make_kernel(
                  py::cast<py::array_t<f32, py::array::c_style>>(kernel_args["w2d"]),
                  py::cast<py::array_t<c64, py::array::c_style>>(kernel_args["h"]),
                  py::cast<py::array_t<c64, py::array::c_style>>(kernel_args["VR"]),
                  py::cast<py::array_t<c64, py::array::c_style>>(kernel_args["refP"]),
                  py::cast<bool>(kernel_args["has_refP"]),
                  py::cast<py::array_t<f32, py::array::c_style>>(kernel_args["HH"]),
                  py::cast<py::array_t<f32, py::array::c_style>>(kernel_args["contact_g"]),
                  py::cast<py::array_t<c64, py::array::c_style>>(kernel_args["contact_Oi"]),
                  py::cast<py::array_t<c64, py::array::c_style>>(kernel_args["contact_Oj"]),
                  py::cast<f32>(kernel_args["weight_sum"]),
                  py::cast<f32>(kernel_args["T"]),
                  py::cast<bool>(kernel_args["include_hartree"]),
                  py::cast<bool>(kernel_args["include_exchange"]),
                  py::cast<bool>(kernel_args["exchange_hcp"]));
              inject_superlattice_into_kernel(K, kernel_args);
              require_dense_shape(P0, K.nk1, K.nk2, K.nb, "P0");
              require_particle_count(K, n_e);
              require_block_sizes(block_sizes, K.nb);
              if (max_iter == 0)
                  throw std::invalid_argument("max_iter must be positive");
              if (!std::isfinite(tol_E) || tol_E < 0.0 ||
                  !std::isfinite(tol_grad) || tol_grad < 0.0)
                  throw std::invalid_argument(
                      "solver tolerances must be finite and non-negative");
              if (!std::isfinite(max_step) || max_step <= 0.0)
                  throw std::invalid_argument(
                      "max_step must be finite and positive");
              if (!std::isfinite(bt_shrink) ||
                  bt_shrink <= 0.0 || bt_shrink >= 1.0)
                  throw std::invalid_argument(
                      "bt_shrink must be strictly between 0 and 1");
              if (!std::isfinite(denom_scale) || denom_scale < 0.0)
                  throw std::invalid_argument(
                      "denom_scale must be finite and non-negative");
              if (bt_max == 0 || cg_restart == 0 || mu_maxiter <= 0)
                  throw std::invalid_argument(
                      "bt_max, cg_restart, and mu_maxiter must be positive");
              if (optimizer != 0 && optimizer != 1 && optimizer != 2)
                  throw std::invalid_argument(
                      "optimizer must select CG (0), Newton (1), or auto (2)");
              if (!std::isfinite(tr_delta0) || tr_delta0 <= 0.0 ||
                  tr_cg_max == 0)
                  throw std::invalid_argument(
                      "Newton trust-region controls must be positive");
              if (!std::isfinite(deflation_sigma) || deflation_sigma < 0.0 ||
                  !std::isfinite(deflation_length) || deflation_length <= 0.0)
                  throw std::invalid_argument(
                      "deflation controls are outside their valid domain");
              SolverConfig cfg;
              cfg.max_iter = max_iter;
              cfg.tol_E = tol_E;
              cfg.tol_grad = tol_grad;
              cfg.max_step = max_step;
              cfg.bt_shrink = bt_shrink;
              cfg.denom_scale = denom_scale;
              cfg.bt_max = bt_max;
              cfg.cg_restart = cg_restart;
              cfg.plateau_window = plateau_window;
              cfg.mu_maxiter = mu_maxiter;
              cfg.optimizer = optimizer;
              cfg.tr_delta0 = tr_delta0;
              cfg.tr_cg_max = tr_cg_max;
              {
                  // deflation_targets: (n_defl, nk1, nk2, nb, nb) complex128, or
                  // empty when no deflation.  Copy the flat buffer; solve_rtr
                  // validates the size against n_deflation * nk * nb * nb.
                  std::size_t n_defl = 0;
                  if (deflation_targets.size() != 0) {
                      if (deflation_targets.ndim() != 5 ||
                          deflation_targets.shape(1) !=
                              static_cast<py::ssize_t>(K.nk1) ||
                          deflation_targets.shape(2) !=
                              static_cast<py::ssize_t>(K.nk2) ||
                          deflation_targets.shape(3) !=
                              static_cast<py::ssize_t>(K.nb) ||
                          deflation_targets.shape(4) !=
                              static_cast<py::ssize_t>(K.nb)) {
                          throw std::invalid_argument(
                              "deflation_targets must be "
                              "(n_solutions,nk1,nk2,nb,nb)");
                      }
                      n_defl = static_cast<std::size_t>(deflation_targets.shape(0));
                      const c64* dptr = deflation_targets.data();
                      cfg.deflation_targets.assign(
                          dptr, dptr + static_cast<std::size_t>(deflation_targets.size()));
                  }
                  cfg.n_deflation = n_defl;
              }
              cfg.deflation_sigma = deflation_sigma;
              cfg.deflation_length = deflation_length;
              cfg.block_sizes = block_sizes;
              cfg.return_Q = return_Q;
              cfg.return_density = return_density;
              cfg.return_fock = return_fock;
              ProjectFn pf = wrap_project_fn(project_fn);
              const ProjectFn* pfp = pf ? &pf : nullptr;
              DMResult res;
              std::size_t n_fock_builds = 0;
              K.fock_builds = &n_fock_builds;
              K.exchange_blocks = cfg.optimizer == 0 ? nullptr : &cfg.block_sizes;
              {
                  py::gil_scoped_release release;
                  if (optimizer == 2) res = solve_auto(K, P0.data(), n_e, cfg, pfp);
                  else if (optimizer == 1) res = solve_rtr(K, P0.data(), n_e, cfg, pfp);
                  else res = solve_dm(K, P0.data(), n_e, cfg, pfp);
              }

              if (kernel_args.contains("diagnostics"))
                  kernel_args["diagnostics"]["fock_builds"] = py::int_(n_fock_builds);

              std::vector<py::ssize_t> dense_shape{(py::ssize_t)K.nk1, (py::ssize_t)K.nk2,
                                                    (py::ssize_t)K.nb, (py::ssize_t)K.nb};
              std::vector<py::ssize_t> p_shape{(py::ssize_t)K.nk1, (py::ssize_t)K.nk2,
                                                (py::ssize_t)K.nb};
              std::vector<py::ssize_t> hE_shape{(py::ssize_t)res.hist_E.size()};
              std::vector<py::ssize_t> hF_shape{(py::ssize_t)res.hist_free.size()};
              std::vector<py::ssize_t> hG_shape{(py::ssize_t)res.hist_grad.size()};
              py::object Q_obj = py::none();
              py::object density_obj = py::none();
              py::object fock_obj = py::none();
              if (return_Q) Q_obj = move_array(std::move(res.Q), dense_shape);
              if (return_density) density_obj = move_array(std::move(res.density), dense_shape);
              if (return_fock) fock_obj = move_array(std::move(res.fock), dense_shape);
              return py::make_tuple(
                  Q_obj,
                  move_array(std::move(res.p), p_shape),
                  density_obj,
                  fock_obj,
                  res.mu, res.energy, res.n_iter, res.converged,
                  move_array(std::move(res.hist_E), hE_shape),
                  move_array(std::move(res.hist_free), hF_shape),
                  move_array(std::move(res.hist_grad), hG_shape));
          });

    // --- DM solver test hooks: cayley spectral helpers, exposed for diagnostics ---
    m.def("_cayley_spectral_setup",
          [](py::array_t<c64, py::array::c_style | py::array::forcecast> d) {
              if (d.ndim() < 2) throw std::invalid_argument("d needs at least 2 dims");
              std::size_t nb = d.shape(d.ndim() - 1);
              if (static_cast<std::size_t>(d.shape(d.ndim() - 2)) != nb)
                  throw std::invalid_argument("d last two dims must match");
              std::size_t nk = 1;
              for (int i = 0; i < d.ndim() - 2; ++i) nk *= d.shape(i);
              std::vector<c64> V(nk * nb * nb);
              std::vector<f32> lam(nk * nb);
              {
                  py::gil_scoped_release release;
                  cpp_hf::dm_internal::cayley_spectral_setup(d.data(), V.data(), lam.data(), nk, nb);
              }
              std::vector<py::ssize_t> v_shape, l_shape;
              for (int i = 0; i < d.ndim() - 2; ++i) {
                  v_shape.push_back(d.shape(i));
                  l_shape.push_back(d.shape(i));
              }
              v_shape.push_back(nb);
              v_shape.push_back(nb);
              l_shape.push_back(nb);
              return py::make_tuple(move_array(std::move(V), v_shape),
                                    move_array(std::move(lam), l_shape));
          });

    m.def("_cayley_unitary_from_spectrum",
          [](py::array_t<c64, py::array::c_style | py::array::forcecast> V,
             py::array_t<f32, py::array::c_style | py::array::forcecast> lam,
             f32 tau) {
              if (V.ndim() < 2) throw std::invalid_argument("V needs at least 2 dims");
              std::size_t nb = V.shape(V.ndim() - 1);
              std::size_t nk = 1;
              for (int i = 0; i < V.ndim() - 2; ++i) nk *= V.shape(i);
              std::vector<c64> U(nk * nb * nb);
              {
                  py::gil_scoped_release release;
                  cpp_hf::dm_internal::cayley_unitary_from_spectrum(
                      V.data(), lam.data(), tau, U.data(), nk, nb);
              }
              std::vector<py::ssize_t> u_shape;
              for (int i = 0; i < V.ndim() - 2; ++i) u_shape.push_back(V.shape(i));
              u_shape.push_back(nb);
              u_shape.push_back(nb);
              return move_array(std::move(U), u_shape);
          });

    m.def("_diag_UFU_from_spectrum",
          [](py::array_t<c64, py::array::c_style | py::array::forcecast> V,
             py::array_t<c64, py::array::c_style | py::array::forcecast> Ft_eig,
             py::array_t<f32, py::array::c_style | py::array::forcecast> lam,
             f32 tau) {
              if (V.ndim() < 2) throw std::invalid_argument("V needs at least 2 dims");
              std::size_t nb = V.shape(V.ndim() - 1);
              std::size_t nk = 1;
              for (int i = 0; i < V.ndim() - 2; ++i) nk *= V.shape(i);
              std::vector<f32> diag(nk * nb);
              {
                  py::gil_scoped_release release;
                  cpp_hf::dm_internal::diag_UFU_from_spectrum(
                      V.data(), Ft_eig.data(), lam.data(), tau, diag.data(), nk, nb);
              }
              std::vector<py::ssize_t> shape;
              for (int i = 0; i < V.ndim() - 2; ++i) shape.push_back(V.shape(i));
              shape.push_back(nb);
              return move_array(std::move(diag), shape);
          });

    // --- RTR joint (Q,p) Hessian-vector product (for finite-diff validation) ---
    m.def("_rtr_joint_hvp",
          [](py::object kernel_args,
             py::array_t<c64, py::array::c_style | py::array::forcecast> X,
             py::array_t<f32, py::array::c_style | py::array::forcecast> dp,
             py::array_t<c64, py::array::c_style | py::array::forcecast> Q,
             py::array_t<f32, py::array::c_style | py::array::forcecast> p,
             py::array_t<c64, py::array::c_style | py::array::forcecast> Ft) {
              auto K = make_kernel(
                  py::cast<py::array_t<f32, py::array::c_style>>(kernel_args["w2d"]),
                  py::cast<py::array_t<c64, py::array::c_style>>(kernel_args["h"]),
                  py::cast<py::array_t<c64, py::array::c_style>>(kernel_args["VR"]),
                  py::cast<py::array_t<c64, py::array::c_style>>(kernel_args["refP"]),
                  py::cast<bool>(kernel_args["has_refP"]),
                  py::cast<py::array_t<f32, py::array::c_style>>(kernel_args["HH"]),
                  py::cast<py::array_t<f32, py::array::c_style>>(kernel_args["contact_g"]),
                  py::cast<py::array_t<c64, py::array::c_style>>(kernel_args["contact_Oi"]),
                  py::cast<py::array_t<c64, py::array::c_style>>(kernel_args["contact_Oj"]),
                  py::cast<f32>(kernel_args["weight_sum"]),
                  py::cast<f32>(kernel_args["T"]),
                  py::cast<bool>(kernel_args["include_hartree"]),
                  py::cast<bool>(kernel_args["include_exchange"]),
                  py::cast<bool>(kernel_args["exchange_hcp"]));
              inject_superlattice_into_kernel(K, kernel_args);
              const std::size_t nk = K.nk(), nb = K.nb, n_tot = K.n_dense();
              std::vector<c64> HX(n_tot), dPb(n_tot), respb(n_tot), Sb(n_tot), Fb(n_tot);
              std::vector<f32> Hp(nk * nb), hb(nb, 0.0);
              require_dense_shape(X,K.nk1,K.nk2,nb,"X");
              require_dense_shape(Q,K.nk1,K.nk2,nb,"Q");
              require_dense_shape(Ft,K.nk1,K.nk2,nb,"Ft");
              const std::vector<py::ssize_t> band_shape{(py::ssize_t)K.nk1,(py::ssize_t)K.nk2,(py::ssize_t)nb};
              require_shape(dp,band_shape,"dp"); require_shape(p,band_shape,"p");
              const f32 T_r = std::max(K.T, static_cast<f32>(1.0e-12));
              py::array_t<c64, py::array::c_style | py::array::forcecast> targets;
              std::size_t n_penalty = 0;
              f32 sigma = 0.0, length = 1.0, diagonal = 0.0;
              std::vector<c64> density(n_tot), penalty(n_tot);
              if (py::cast<py::dict>(kernel_args).contains("test_deflation_targets")) {
                  targets = py::cast<decltype(targets)>(kernel_args["test_deflation_targets"]);
                  if (targets.ndim() != 5) throw std::invalid_argument("targets must have five axes");
                  n_penalty = targets.shape(0);
                  require_shape(targets,{(py::ssize_t)n_penalty,(py::ssize_t)K.nk1,
                      (py::ssize_t)K.nk2,(py::ssize_t)nb,(py::ssize_t)nb},"targets");
                  sigma = py::cast<f32>(kernel_args["test_deflation_sigma"]);
                  length = py::cast<f32>(kernel_args["test_deflation_length"]);
                  if (!std::isfinite(sigma) || !std::isfinite(length) || length <= 0)
                      throw std::invalid_argument("invalid test deflation parameters");
                  cpp_hf::dm_internal::density_from_Qp(Q.data(),p.data(),density.data(),nk,nb);
                  cpp_hf::rtr_internal::deflation_bias(density.data(),targets.data(),n_penalty,
                      K.w2d,nk,nb,sigma,length,penalty.data(),diagonal);
              }
              {
                  py::gil_scoped_release release;
                  cpp_hf::rtr_internal::joint_hvp(
                      K, X.data(), dp.data(), Q.data(), p.data(), Ft.data(), T_r,
                      K.w2d, nk, nb, HX.data(), Hp.data(), dPb.data(), respb.data(),
                      Sb.data(), Fb.data(), hb.data(), diagonal,
                      n_penalty ? density.data() : nullptr, n_penalty ? targets.data() : nullptr,
                      n_penalty, sigma, length);
              }
              std::vector<py::ssize_t> xs{(py::ssize_t)K.nk1, (py::ssize_t)K.nk2,
                                          (py::ssize_t)nb, (py::ssize_t)nb};
              std::vector<py::ssize_t> ps{(py::ssize_t)K.nk1, (py::ssize_t)K.nk2,
                                          (py::ssize_t)nb};
              return py::make_tuple(move_array(std::move(HX), xs),
                                    move_array(std::move(Hp), ps));
          });

    m.def("_fermi_response",
          [](py::array_t<c64, py::array::c_style | py::array::forcecast> dF,
             py::array_t<c64, py::array::c_style | py::array::forcecast> Q,
             py::array_t<f32, py::array::c_style | py::array::forcecast> eps,
             py::array_t<f32, py::array::c_style | py::array::forcecast> p,
             py::array_t<f32, py::array::c_style | py::array::forcecast> w, f32 T) {
              if (dF.ndim() != 4 || dF.shape(0) <= 0 || dF.shape(1) <= 0 || dF.shape(2) <= 0)
                  throw std::invalid_argument("dF must be a nonempty (nk1,nk2,nb,nb) array");
              const std::size_t nk1=dF.shape(0), nk2=dF.shape(1), nb=dF.shape(2);
              require_dense_shape(dF,nk1,nk2,nb,"dF");
              require_dense_shape(Q,nk1,nk2,nb,"Q");
              require_shape(eps,{(py::ssize_t)nk1,(py::ssize_t)nk2,(py::ssize_t)nb},"eps");
              require_shape(p,{(py::ssize_t)nk1,(py::ssize_t)nk2,(py::ssize_t)nb},"p");
              require_shape(w,{(py::ssize_t)nk1,(py::ssize_t)nk2},"w");
              if (!std::isfinite(T) || T <= 0.0)
                  throw std::invalid_argument("T must be finite and positive");
              std::vector<c64> out(nk1*nk2*nb*nb);
              {
                  py::gil_scoped_release release;
                  fermi_response(dF.data(),Q.data(),eps.data(),p.data(),w.data(),T,nk1*nk2,nb,out.data());
              }
              return move_array(std::move(out),{(py::ssize_t)nk1,(py::ssize_t)nk2,(py::ssize_t)nb,(py::ssize_t)nb});
          });

    // --- Occupation-sector test hooks: saturation handling and the in-loop
    //     chemical-potential solve (mirrors jax_hf.solver._solve_mu) ---
    m.def("_solve_mu_inloop",
          [](py::array_t<f32, py::array::c_style | py::array::forcecast> bands,
             py::array_t<f32, py::array::c_style | py::array::forcecast> w_norm,
             f32 n_target_norm, f32 mu0, f32 T, int maxiter) {
              const auto layout = band_layout(bands, w_norm, "bands", "w_norm");
              const std::size_t nk = layout.first, nb = layout.second;
              if (!std::isfinite(T) || T < 0.0)
                  throw std::invalid_argument("T must be finite and non-negative");
              if (!std::isfinite(n_target_norm) || !std::isfinite(mu0))
                  throw std::invalid_argument(
                      "n_target_norm and mu0 must be finite");
              if (maxiter <= 0)
                  throw std::invalid_argument("maxiter must be positive");
              f32 mu = 0.0;
              {
                  py::gil_scoped_release release;
                  mu = solve_mu_inloop(bands.data(), nk * nb, w_norm.data(),
                                       nk, nb, n_target_norm, mu0, T, maxiter);
              }
              return mu;
          },
          py::arg("bands"), py::arg("w_norm"), py::arg("n_target_norm"),
          py::arg("mu0"), py::arg("T"), py::arg("maxiter") = 25);

    m.def("_occupation_gradient",
          [](py::array_t<f32, py::array::c_style | py::array::forcecast> eps,
             py::array_t<f32, py::array::c_style | py::array::forcecast> p,
             f32 T,
             py::array_t<f32, py::array::c_style | py::array::forcecast> weights) {
              const auto layout = band_layout(eps, weights, "eps", "weights");
              const std::size_t nk = layout.first, nb = layout.second;
              std::vector<py::ssize_t> shape(eps.shape(), eps.shape() + eps.ndim());
              require_shape(p, shape, "p");
              if (!std::isfinite(T) || T < 0.0)
                  throw std::invalid_argument("T must be finite and non-negative");
              const f32 T_r = std::max(T, static_cast<f32>(1.0e-12));
              const std::size_t n = nk * nb;
              std::vector<f32> g(n), h(n), curv(n);
              std::vector<unsigned char> active(n);
              {
                  py::gil_scoped_release release;
                  cpp_hf::dm_internal::occupation_gradient(
                      eps.data(), p.data(), T_r, weights.data(), nk, nb,
                      g.data(), h.data(), curv.data(), active.data());
              }
              py::array_t<bool> active_arr(shape);
              bool* act = active_arr.mutable_data();
              for (std::size_t i = 0; i < n; ++i) act[i] = active[i] != 0;
              return py::make_tuple(move_array(std::move(g), shape),
                                    move_array(std::move(h), shape),
                                    move_array(std::move(curv), shape),
                                    active_arr);
          },
          py::arg("eps"), py::arg("p"), py::arg("T"), py::arg("weights"));

    m.def("_project_occupation_tangent",
          [](py::array_t<f32, py::array::c_style | py::array::forcecast> x,
             py::array_t<f32, py::array::c_style | py::array::forcecast> weights) {
              const auto layout = band_layout(x, weights, "x", "weights");
              const std::size_t nk = layout.first, nb = layout.second;
              std::vector<py::ssize_t> shape(x.shape(), x.shape() + x.ndim());
              std::vector<f32> out(x.data(), x.data() + nk * nb);
              cpp_hf::dm_internal::project_occupation_tangent(
                  out.data(), weights.data(), nk, nb);
              return move_array(std::move(out), shape);
          },
          py::arg("x"), py::arg("weights"));

    // --- Deflation bias (Phi, S_pen, diag_coeff) for finite-diff validation ---
    m.def("_deflation_bias",
          [](py::array_t<c64, py::array::c_style | py::array::forcecast> P,
             py::array_t<c64, py::array::c_style | py::array::forcecast> targets,
             py::array_t<f32, py::array::c_style | py::array::forcecast> w2d,
             f32 sigma, f32 length) {
              const std::size_t nk1 = static_cast<std::size_t>(P.shape(0));
              const std::size_t nk2 = static_cast<std::size_t>(P.shape(1));
              const std::size_t nb = static_cast<std::size_t>(P.shape(2));
              const std::size_t nk = nk1 * nk2, n_tot = nk * nb * nb;
              const std::size_t n_defl =
                  (targets.ndim() == 5) ? static_cast<std::size_t>(targets.shape(0)) : 0;
              std::vector<c64> S_pen(n_tot);
              f32 diag = 0.0;
              const f32 Phi = cpp_hf::rtr_internal::deflation_bias(
                  P.data(), n_defl ? targets.data() : nullptr, n_defl,
                  w2d.data(), nk, nb, sigma, length, S_pen.data(), diag);
              std::vector<py::ssize_t> xs{(py::ssize_t)nk1, (py::ssize_t)nk2,
                                          (py::ssize_t)nb, (py::ssize_t)nb};
              return py::make_tuple(Phi, move_array(std::move(S_pen), xs), diag);
          });

    // --- Superlattice Fock (ΔG-streamed) ---
    m.def("selfenergy_superlattice_streamed",
          [](py::array_t<c64, py::array::c_style | py::array::forcecast> rho,
             py::array_t<c64, py::array::c_style | py::array::forcecast> VR_fft,
             py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> g_a_off,
             py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> pair_i,
             py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> pair_j,
             py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> pair_start,
             std::size_t nkx, std::size_t nky,
             std::size_t n_G, std::size_t dim_orb,
             std::size_t N_ext_x, std::size_t N_ext_y,
             py::object VR_fft_orbital_obj,
             bool hermitian_rho) {
              if (rho.ndim() != 6) {
                  throw std::invalid_argument(
                      "rho must be (nkx, nky, n_G, dim_orb, n_G, dim_orb)");
              }
              if (VR_fft.ndim() != 2) {
                  throw std::invalid_argument("VR_fft must be (N_ext_x, N_ext_y)");
              }
              if (static_cast<std::size_t>(VR_fft.shape(0)) != N_ext_x ||
                  static_cast<std::size_t>(VR_fft.shape(1)) != N_ext_y) {
                  throw std::invalid_argument(
                      "VR_fft shape does not match (N_ext_x, N_ext_y)");
              }
              if (g_a_off.ndim() != 2 ||
                  static_cast<std::size_t>(g_a_off.shape(0)) != n_G ||
                  g_a_off.shape(1) != 2) {
                  throw std::invalid_argument("g_a_off must be (n_G, 2)");
              }
              if (pair_start.ndim() != 1 || pair_start.shape(0) < 1) {
                  throw std::invalid_argument("pair_start must be 1-D, len >= 1");
              }
              const std::size_t n_delta =
                  static_cast<std::size_t>(pair_start.shape(0)) - 1;
              if (pair_i.ndim() != 1 || pair_j.ndim() != 1 ||
                  pair_i.shape(0) != pair_j.shape(0)) {
                  throw std::invalid_argument(
                      "pair_i and pair_j must be 1-D with the same length");
              }

              const c64* VR_orb_ptr = nullptr;
              py::array_t<c64, py::array::c_style | py::array::forcecast>
                  VR_orb_arr;
              if (!VR_fft_orbital_obj.is_none()) {
                  VR_orb_arr = py::cast<
                      py::array_t<c64, py::array::c_style | py::array::forcecast>
                  >(VR_fft_orbital_obj);
                  if (VR_orb_arr.ndim() != 4 ||
                      static_cast<std::size_t>(VR_orb_arr.shape(0)) != N_ext_x ||
                      static_cast<std::size_t>(VR_orb_arr.shape(1)) != N_ext_y ||
                      static_cast<std::size_t>(VR_orb_arr.shape(2)) != dim_orb ||
                      static_cast<std::size_t>(VR_orb_arr.shape(3)) != dim_orb) {
                      throw std::invalid_argument(
                          "VR_fft_orbital must be "
                          "(N_ext_x, N_ext_y, dim_orb, dim_orb)");
                  }
                  VR_orb_ptr = VR_orb_arr.data();
              }

              // Allocate output sigma with the same layout as rho.
              std::vector<py::ssize_t> out_shape(rho.shape(), rho.shape() + 6);
              std::vector<c64> sigma_buf(
                  static_cast<std::size_t>(rho.size())
              );
              {
                  py::gil_scoped_release release;
                  selfenergy_superlattice_streamed(
                      rho.data(),
                      sigma_buf.data(),
                      VR_fft.data(),
                      VR_orb_ptr,
                      g_a_off.data(),
                      pair_i.data(),
                      pair_j.data(),
                      pair_start.data(),
                      n_delta,
                      N_ext_x, N_ext_y,
                      nkx, nky,
                      n_G, dim_orb,
                      hermitian_rho
                  );
              }
              return move_array(std::move(sigma_buf), out_shape);
          },
          py::arg("rho"),
          py::arg("VR_fft"),
          py::arg("g_a_off"),
          py::arg("pair_i"),
          py::arg("pair_j"),
          py::arg("pair_start"),
          py::arg("nkx"), py::arg("nky"),
          py::arg("n_G"), py::arg("dim_orb"),
          py::arg("N_ext_x"), py::arg("N_ext_y"),
          py::arg("VR_fft_orbital") = py::none(),
          py::arg("hermitian_rho") = false,
          "Streaming superlattice Fock self-energy (per-ΔG FFT convolution). "
          "hermitian_rho=True opts into ΔG<->-ΔG conjugate halving (valid only "
          "for Hermitian rho; ~2x fewer FFTs).");

    // --- Resample k-grid (linear, periodic) ---
    m.def("resample_kgrid_2d",
          [](py::array_t<c64, py::array::c_style | py::array::forcecast> values,
             std::size_t nk_new) {
              if (values.ndim() < 2) throw std::invalid_argument("need >= 2 dims");
              const std::size_t nk_old = values.shape(0);
              if (static_cast<std::size_t>(values.shape(1)) != nk_old)
                  throw std::invalid_argument("First two axes must be square (nk_old, nk_old)");
              std::size_t inner = 1;
              std::vector<py::ssize_t> out_shape;
              out_shape.push_back((py::ssize_t)nk_new);
              out_shape.push_back((py::ssize_t)nk_new);
              for (int i = 2; i < values.ndim(); ++i) {
                  inner *= values.shape(i);
                  out_shape.push_back(values.shape(i));
              }
              std::vector<c64> out;
              {
                  py::gil_scoped_release release;
                  out = resample_kgrid_2d(values.data(), nk_old, nk_new, inner);
              }
              return move_array(std::move(out), out_shape);
          });
}
