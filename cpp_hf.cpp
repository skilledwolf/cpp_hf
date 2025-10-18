// Thin Python bindings only. Core logic lives in src/hartreefock.cpp

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <vector>
#include <complex>
#include <cstring>

#include "cpp_hf/hartreefock.hpp"

namespace py = pybind11;
using cxd  = std::complex<double>;

static py::tuple hartreefock_iteration_cpp(
    py::array_t<double, py::array::c_style | py::array::forcecast> weights,   // (nk1,nk2)
    py::array_t<cxd,    py::array::c_style | py::array::forcecast> hamiltonian, // (nk1,nk2,d,d)
    py::array_t<cxd,    py::array::c_style | py::array::forcecast> v_coulomb,   // (nk1,nk2,dv1,dv2)
    py::array_t<cxd,    py::array::c_style | py::array::forcecast> p0,          // (nk1,nk2,d,d)
    double electron_density0,
    double T,
    std::size_t max_iter,
    double comm_tol,
    std::size_t diis_size,
    double mixing_alpha
) {
    if (hamiltonian.ndim()!=4) throw std::invalid_argument("H must be (nk1,nk2,d,d)");
    const std::size_t nk1 = (std::size_t)hamiltonian.shape(0);
    const std::size_t nk2 = (std::size_t)hamiltonian.shape(1);
    const std::size_t d   = (std::size_t)hamiltonian.shape(2);
    if ((std::size_t)hamiltonian.shape(3)!=d) throw std::invalid_argument("H last two dims must be equal (d,d)");
    if (weights.ndim()!=2 || (std::size_t)weights.shape(0)!=nk1 || (std::size_t)weights.shape(1)!=nk2) throw std::invalid_argument("weights must be (nk1,nk2)");
    if (p0.ndim()!=4 || (std::size_t)p0.shape(0)!=nk1 || (std::size_t)p0.shape(1)!=nk2 || (std::size_t)p0.shape(2)!=d || (std::size_t)p0.shape(3)!=d)
        throw std::invalid_argument("p0 must be (nk1,nk2,d,d)");
    if (v_coulomb.ndim()!=4 || (std::size_t)v_coulomb.shape(0)!=nk1 || (std::size_t)v_coulomb.shape(1)!=nk2)
        throw std::invalid_argument("V must be (nk1,nk2,dv1,dv2)");

    const std::size_t dv1 = (std::size_t)v_coulomb.shape(2);
    const std::size_t dv2 = (std::size_t)v_coulomb.shape(3);
    if (!((dv1==1 && dv2==1) || (dv1==d && dv2==d)))
        throw std::invalid_argument("V last dims must be (1,1) or (d,d)");

    hf::HFResult res;
    {
        py::gil_scoped_release nogil;
        res = hf::hartreefock_iteration(weights.data(), hamiltonian.data(), v_coulomb.data(),
                                         nk1, nk2, d, dv1, dv2,
                                         p0.data(), electron_density0, T,
                                         max_iter, comm_tol, diis_size, mixing_alpha);
    }

    py::array_t<cxd> P_out({(py::ssize_t)nk1,(py::ssize_t)nk2,(py::ssize_t)d,(py::ssize_t)d});
    py::array_t<cxd> F_out({(py::ssize_t)nk1,(py::ssize_t)nk2,(py::ssize_t)d,(py::ssize_t)d});
    std::memcpy(P_out.mutable_data(), res.P.data(), res.P.size()*sizeof(cxd));
    std::memcpy(F_out.mutable_data(), res.F.data(), res.F.size()*sizeof(cxd));

    return py::make_tuple(P_out, F_out, res.energy, res.mu, res.iters);
}

PYBIND11_MODULE(cpp_hf, m) {
    m.doc() = "Hartree–Fock (k-grid) with FFTW + Eigen + OpenMP + EDIIS/CDIIS + preconditioned-LBFGS";
    m.def("hartreefock_iteration_cpp", &hartreefock_iteration_cpp,
          py::arg("weights"), py::arg("hamiltonian"), py::arg("v_coulomb"), py::arg("p0"),
          py::arg("electron_density0"), py::arg("T"),
          py::arg("max_iter"), py::arg("comm_tol"),
          py::arg("diis_size"), py::arg("mixing_alpha"));
}
