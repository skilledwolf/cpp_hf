// XLA FFI handlers: the C++ kernels the JAX backend calls through jax.ffi.
//
// Only compiled when the build finds the XLA FFI headers that ship with
// jaxlib (HF_HAVE_XLA_FFI).  The handlers wrap the same kernels the native
// backend uses, so JAX gets OpenMP-threaded per-k linear algebra instead of
// XLA CPU's serial loop over the batch.
#pragma once

#include "cpp_hf/linalg.hpp"
#include "cpp_hf/types.hpp"

#include "xla/ffi/api/ffi.h"

#include <cstddef>

namespace cpp_hf {
namespace xla_ffi {

namespace ffi = ::xla::ffi;

// Batched Hermitian eigh of a complex128 (..., n, n) array: ascending
// eigenvalues (..., n) and eigenvectors as columns (..., n, n).  Reads the
// Hermitian part of each matrix.
inline ffi::Error EighC128Impl(ffi::Buffer<ffi::C128> a,
                               ffi::ResultBuffer<ffi::F64> w,
                               ffi::ResultBuffer<ffi::C128> v) {
    const auto dims = a.dimensions();
    if (dims.size() < 2) {
        return ffi::Error::InvalidArgument("eigh needs an array of at least 2 dimensions");
    }
    const std::size_t n = static_cast<std::size_t>(dims.back());
    if (static_cast<std::size_t>(dims[dims.size() - 2]) != n) {
        return ffi::Error::InvalidArgument("eigh needs square matrices on the last two axes");
    }
    std::size_t batch = 1;
    for (std::size_t i = 0; i + 2 < dims.size(); ++i) batch *= static_cast<std::size_t>(dims[i]);
    eigh_batched(reinterpret_cast<const c64*>(a.typed_data()), w->typed_data(),
                 reinterpret_cast<c64*>(v->typed_data()), batch, 1, n);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(EighC128, EighC128Impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::C128>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::C128>>());

}  // namespace xla_ffi
}  // namespace cpp_hf
