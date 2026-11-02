// The extension module: it hands the XLA FFI handlers in cpp/include/cpp_hf/
// to hartreefock._jax.ffi, which registers them with XLA for CPU.

#include <pybind11/pybind11.h>

#include <string>

#ifdef HF_HAVE_XLA_FFI
#include "cpp_hf/xla_ffi.hpp"
#endif

namespace py = pybind11;

PYBIND11_MODULE(_native, m) {
    m.doc() = "hartreefock C++ kernels for jax.ffi (Eigen + FFTW)";

    // name -> handler capsule; empty when built without the jaxlib FFI headers.
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

    // The jaxlib the handlers were compiled against ("" without them).
    m.def("xla_ffi_jaxlib_version", []() {
#ifdef HF_HAVE_XLA_FFI
        return std::string(HF_XLA_FFI_JAXLIB);
#else
        return std::string();
#endif
    });
}
