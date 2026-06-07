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
#include "cpp_hf/solver_scf.hpp"
#include "cpp_hf/superlattice.hpp"
#include "cpp_hf/types.hpp"
#include "cpp_hf/utils.hpp"

#include <cstring>
#include <memory>
#include <stdexcept>
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

// Build a HFKernel from raw numpy arrays.  All inputs are validated at the
// Python level before being passed in here.
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
    HFKernel K;
    K.nk1 = static_cast<std::size_t>(h.shape(0));
    K.nk2 = static_cast<std::size_t>(h.shape(1));
    K.nb = static_cast<std::size_t>(h.shape(2));
    if (static_cast<std::size_t>(h.shape(3)) != K.nb)
        throw std::invalid_argument("h last two axes must be square (nb,nb)");
    if (VR_shifted.ndim() != 4)
        throw std::invalid_argument("VR must be (nk1,nk2,dv1,dv2)");
    K.dv1 = static_cast<std::size_t>(VR_shifted.shape(2));
    K.dv2 = static_cast<std::size_t>(VR_shifted.shape(3));
    if (!((K.dv1 == 1 && K.dv2 == 1) || (K.dv1 == K.nb && K.dv2 == K.nb)))
        throw std::invalid_argument("VR last two dims must be (1,1) or (nb,nb)");

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
    }

    K.h = h.data();
    K.VR = VR_shifted.data();
    K.refP = has_refP ? refP.data() : nullptr;
    K.w2d = w2d.data();
    K.HH = HH.data();

    if (contact_g.ndim() != 1) throw std::invalid_argument("contact_g must be 1-D");
    K.n_contact = static_cast<std::size_t>(contact_g.shape(0));
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

    if (K.n_G * K.dim_orb != K.nb) {
        throw std::invalid_argument(
            "superlattice mode requires n_G × dim_orb == nb");
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

    if (static_cast<std::size_t>(V_lag_fft.shape(0)) != K.N_ext_x ||
        static_cast<std::size_t>(V_lag_fft.shape(1)) != K.N_ext_y) {
        throw std::invalid_argument(
            "V_lag_fft must have shape (N_ext_x, N_ext_y)");
    }
    if (static_cast<std::size_t>(g_a_off.shape(0)) != K.n_G ||
        g_a_off.shape(1) != 2) {
        throw std::invalid_argument("g_a_off must be (n_G, 2)");
    }
    if (static_cast<std::size_t>(pair_start.shape(0)) != K.n_delta + 1) {
        throw std::invalid_argument("pair_start must have length n_delta + 1");
    }
    if (pair_i.shape(0) != pair_j.shape(0)) {
        throw std::invalid_argument("pair_i and pair_j must have the same length");
    }
    if (static_cast<std::size_t>(pair_to_delta.shape(0)) != K.n_G ||
        static_cast<std::size_t>(pair_to_delta.shape(1)) != K.n_G) {
        throw std::invalid_argument("pair_to_delta must be (n_G, n_G)");
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
        if (static_cast<std::size_t>(HH_GG.shape(0)) != K.n_G ||
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
              const std::size_t nk1 = P.shape(0);
              const std::size_t nk2 = P.shape(1);
              const std::size_t nb = P.shape(2);
              const std::size_t dv1 = VR.shape(2);
              const std::size_t dv2 = VR.shape(3);
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
              std::size_t nk = 1;
              for (int i = 0; i < bands.ndim() - 1; ++i) nk *= bands.shape(i);
              std::size_t nb = bands.shape(bands.ndim() - 1);
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
              SCFConfig cfg;
              cfg.max_iter = max_iter;
              cfg.density_tol = density_tol;
              cfg.comm_tol = comm_tol;
              cfg.mixing = mixing;
              cfg.level_shift = level_shift;
              cfg.block_sizes = block_sizes;
              cfg.acceleration = acceleration;
              cfg.diis_size = diis_size;
              cfg.diis_start = diis_start;
              cfg.diis_damping = diis_damping;
              cfg.trust_radius = trust_radius;
              cfg.return_density = return_density;
              cfg.return_fock = return_fock;
              ProjectFn pf = wrap_project_fn(project_fn);
              const ProjectFn* pfp = pf ? &pf : nullptr;
              SCFResult res;
              {
                  py::gil_scoped_release release;
                  res = solve_scf(K, P0.data(), n_e, cfg, pfp);
              }

              std::vector<py::ssize_t> dense_shape{(py::ssize_t)K.nk1, (py::ssize_t)K.nk2,
                                                    (py::ssize_t)K.nb, (py::ssize_t)K.nb};
              std::vector<py::ssize_t> hE_shape{(py::ssize_t)res.hist_E.size()};
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
              cfg.block_sizes = block_sizes;
              cfg.return_Q = return_Q;
              cfg.return_density = return_density;
              cfg.return_fock = return_fock;
              ProjectFn pf = wrap_project_fn(project_fn);
              const ProjectFn* pfp = pf ? &pf : nullptr;
              DMResult res;
              {
                  py::gil_scoped_release release;
                  res = solve_dm(K, P0.data(), n_e, cfg, pfp);
              }

              std::vector<py::ssize_t> dense_shape{(py::ssize_t)K.nk1, (py::ssize_t)K.nk2,
                                                    (py::ssize_t)K.nb, (py::ssize_t)K.nb};
              std::vector<py::ssize_t> p_shape{(py::ssize_t)K.nk1, (py::ssize_t)K.nk2,
                                                (py::ssize_t)K.nb};
              std::vector<py::ssize_t> hE_shape{(py::ssize_t)res.hist_E.size()};
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
