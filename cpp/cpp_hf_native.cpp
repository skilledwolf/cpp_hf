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
#include "cpp_hf/types.hpp"
#include "cpp_hf/utils.hpp"

#include <cstring>
#include <stdexcept>
#include <vector>

namespace py = pybind11;
using namespace cpp_hf;

namespace {

template <typename T>
py::array_t<T> make_array(const std::vector<T>& v, std::vector<py::ssize_t> shape) {
    py::array_t<T> arr(shape);
    std::memcpy(arr.mutable_data(), v.data(), v.size() * sizeof(T));
    return arr;
}

template <typename T>
py::array_t<T> make_array_from(const T* data, std::vector<py::ssize_t> shape) {
    py::ssize_t total = 1;
    for (auto s : shape) total *= s;
    py::array_t<T> arr(shape);
    std::memcpy(arr.mutable_data(), data, total * sizeof(T));
    return arr;
}

// Build a HFKernel from raw numpy arrays.  All inputs are validated at the
// Python level before being passed in here.
HFKernel make_kernel(
    py::array_t<f32, py::array::c_style | py::array::forcecast> w2d,
    py::array_t<c64, py::array::c_style | py::array::forcecast> h,
    py::array_t<c64, py::array::c_style | py::array::forcecast> VR_shifted,
    py::array_t<c64, py::array::c_style | py::array::forcecast> refP,
    py::array_t<f32, py::array::c_style | py::array::forcecast> HH,
    py::array_t<f32, py::array::c_style | py::array::forcecast> contact_g,
    py::array_t<c64, py::array::c_style | py::array::forcecast> contact_Oi,
    py::array_t<c64, py::array::c_style | py::array::forcecast> contact_Oj,
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
    K.T = T;
    K.weight_sum = weight_sum;

    const std::size_t n_dense = K.n_dense();
    const std::size_t n_VR = K.nk1 * K.nk2 * K.dv1 * K.dv2;

    K.h.assign(h.data(), h.data() + n_dense);
    K.VR.assign(VR_shifted.data(), VR_shifted.data() + n_VR);
    K.refP.assign(refP.data(), refP.data() + n_dense);
    K.w2d.assign(w2d.data(), w2d.data() + K.nk1 * K.nk2);
    K.HH.assign(HH.data(), HH.data() + K.nb * K.nb);

    if (contact_g.ndim() != 1) throw std::invalid_argument("contact_g must be 1-D");
    K.n_contact = static_cast<std::size_t>(contact_g.shape(0));
    K.contact_g.assign(contact_g.data(), contact_g.data() + K.n_contact);
    K.contact_Oi.assign(contact_Oi.data(), contact_Oi.data() + K.n_contact * K.nb * K.nb);
    K.contact_Oj.assign(contact_Oj.data(), contact_Oj.data() + K.n_contact * K.nb * K.nb);

    return K;
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
              return make_array(sigma, {(py::ssize_t)nk1, (py::ssize_t)nk2,
                                         (py::ssize_t)nb, (py::ssize_t)nb});
          },
          py::arg("VR"), py::arg("P"),
          py::arg("apply_ifftshift") = true,
          py::arg("hermitian_channel_packing") = false);

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
              return py::make_tuple(make_array(w, w_shape), make_array(V, v_shape));
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
              return py::make_tuple(make_array(w, w_shape), make_array(V, v_shape));
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
                  py::cast<py::array_t<f32, py::array::c_style | py::array::forcecast>>(kernel_args["w2d"]),
                  py::cast<py::array_t<c64, py::array::c_style | py::array::forcecast>>(kernel_args["h"]),
                  py::cast<py::array_t<c64, py::array::c_style | py::array::forcecast>>(kernel_args["VR"]),
                  py::cast<py::array_t<c64, py::array::c_style | py::array::forcecast>>(kernel_args["refP"]),
                  py::cast<py::array_t<f32, py::array::c_style | py::array::forcecast>>(kernel_args["HH"]),
                  py::cast<py::array_t<f32, py::array::c_style | py::array::forcecast>>(kernel_args["contact_g"]),
                  py::cast<py::array_t<c64, py::array::c_style | py::array::forcecast>>(kernel_args["contact_Oi"]),
                  py::cast<py::array_t<c64, py::array::c_style | py::array::forcecast>>(kernel_args["contact_Oj"]),
                  py::cast<f32>(kernel_args["weight_sum"]),
                  py::cast<f32>(kernel_args["T"]),
                  py::cast<bool>(kernel_args["include_hartree"]),
                  py::cast<bool>(kernel_args["include_exchange"]),
                  py::cast<bool>(kernel_args["exchange_hcp"]));
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
              return py::make_tuple(make_array(Sigma, shape),
                                     make_array(Hh, shape),
                                     make_array(F, shape),
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
             py::object project_fn) {
              auto K = make_kernel(
                  py::cast<py::array_t<f32, py::array::c_style | py::array::forcecast>>(kernel_args["w2d"]),
                  py::cast<py::array_t<c64, py::array::c_style | py::array::forcecast>>(kernel_args["h"]),
                  py::cast<py::array_t<c64, py::array::c_style | py::array::forcecast>>(kernel_args["VR"]),
                  py::cast<py::array_t<c64, py::array::c_style | py::array::forcecast>>(kernel_args["refP"]),
                  py::cast<py::array_t<f32, py::array::c_style | py::array::forcecast>>(kernel_args["HH"]),
                  py::cast<py::array_t<f32, py::array::c_style | py::array::forcecast>>(kernel_args["contact_g"]),
                  py::cast<py::array_t<c64, py::array::c_style | py::array::forcecast>>(kernel_args["contact_Oi"]),
                  py::cast<py::array_t<c64, py::array::c_style | py::array::forcecast>>(kernel_args["contact_Oj"]),
                  py::cast<f32>(kernel_args["weight_sum"]),
                  py::cast<f32>(kernel_args["T"]),
                  py::cast<bool>(kernel_args["include_hartree"]),
                  py::cast<bool>(kernel_args["include_exchange"]),
                  py::cast<bool>(kernel_args["exchange_hcp"]));
              SCFConfig cfg;
              cfg.max_iter = max_iter;
              cfg.density_tol = density_tol;
              cfg.comm_tol = comm_tol;
              cfg.mixing = mixing;
              cfg.level_shift = level_shift;
              ProjectFn pf = wrap_project_fn(project_fn);
              const ProjectFn* pfp = pf ? &pf : nullptr;
              SCFResult res;
              {
                  py::gil_scoped_release release;
                  res = solve_scf(K, P0.data(), n_e, cfg, pfp);
              }

              std::vector<py::ssize_t> dense_shape{(py::ssize_t)K.nk1, (py::ssize_t)K.nk2,
                                                    (py::ssize_t)K.nb, (py::ssize_t)K.nb};
              return py::make_tuple(
                  make_array(res.density, dense_shape),
                  make_array(res.fock, dense_shape),
                  res.energy, res.mu, res.iterations, res.converged,
                  make_array(res.hist_E, {(py::ssize_t)res.hist_E.size()}),
                  make_array(res.hist_density, {(py::ssize_t)res.hist_density.size()}),
                  make_array(res.hist_comm, {(py::ssize_t)res.hist_comm.size()}));
          });

    // --- Direct minimization solver ---
    m.def("solve_dm",
          [](py::object kernel_args,
             py::array_t<c64, py::array::c_style | py::array::forcecast> P0,
             f32 n_e,
             std::size_t max_iter, f32 tol_E, f32 tol_grad,
             f32 max_step, f32 bt_shrink, f32 denom_scale,
             std::size_t bt_max, std::size_t cg_restart, int mu_maxiter,
             std::vector<std::size_t> block_sizes,
             py::object project_fn) {
              auto K = make_kernel(
                  py::cast<py::array_t<f32, py::array::c_style | py::array::forcecast>>(kernel_args["w2d"]),
                  py::cast<py::array_t<c64, py::array::c_style | py::array::forcecast>>(kernel_args["h"]),
                  py::cast<py::array_t<c64, py::array::c_style | py::array::forcecast>>(kernel_args["VR"]),
                  py::cast<py::array_t<c64, py::array::c_style | py::array::forcecast>>(kernel_args["refP"]),
                  py::cast<py::array_t<f32, py::array::c_style | py::array::forcecast>>(kernel_args["HH"]),
                  py::cast<py::array_t<f32, py::array::c_style | py::array::forcecast>>(kernel_args["contact_g"]),
                  py::cast<py::array_t<c64, py::array::c_style | py::array::forcecast>>(kernel_args["contact_Oi"]),
                  py::cast<py::array_t<c64, py::array::c_style | py::array::forcecast>>(kernel_args["contact_Oj"]),
                  py::cast<f32>(kernel_args["weight_sum"]),
                  py::cast<f32>(kernel_args["T"]),
                  py::cast<bool>(kernel_args["include_hartree"]),
                  py::cast<bool>(kernel_args["include_exchange"]),
                  py::cast<bool>(kernel_args["exchange_hcp"]));
              SolverConfig cfg;
              cfg.max_iter = max_iter;
              cfg.tol_E = tol_E;
              cfg.tol_grad = tol_grad;
              cfg.max_step = max_step;
              cfg.bt_shrink = bt_shrink;
              cfg.denom_scale = denom_scale;
              cfg.bt_max = bt_max;
              cfg.cg_restart = cg_restart;
              cfg.mu_maxiter = mu_maxiter;
              cfg.block_sizes = block_sizes;
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
              return py::make_tuple(
                  make_array(res.Q, dense_shape),
                  make_array(res.p, p_shape),
                  make_array(res.density, dense_shape),
                  make_array(res.fock, dense_shape),
                  res.mu, res.energy, res.n_iter, res.converged,
                  make_array(res.hist_E, {(py::ssize_t)res.hist_E.size()}),
                  make_array(res.hist_grad, {(py::ssize_t)res.hist_grad.size()}));
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
              return py::make_tuple(make_array(V, v_shape), make_array(lam, l_shape));
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
              return make_array(U, u_shape);
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
              return make_array(diag, shape);
          });

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
              return make_array(out, out_shape);
          });
}
