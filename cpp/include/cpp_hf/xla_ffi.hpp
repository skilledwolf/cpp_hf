// XLA FFI handlers: the C++ kernels the JAX backend calls through jax.ffi.
//
// Only compiled when the build finds the XLA FFI headers that ship with
// jaxlib (HF_HAVE_XLA_FFI).  The handlers wrap the same kernels the native
// backend uses, so JAX gets OpenMP-threaded per-k linear algebra instead of
// XLA CPU's serial loop over the batch.
#pragma once

#include "cpp_hf/linalg.hpp"
#include "cpp_hf/superlattice.hpp"
#include "cpp_hf/types.hpp"

#include "xla/ffi/api/ffi.h"

#include <cstddef>
#include <cstdint>

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

// Streamed superlattice Fock self-energy Σ_F[rho] of a (nkx, nky, D, D)
// complex128 density with D = n_G · dim_orb (see superlattice.hpp).  The
// layout tables come in as buffers: the lag Coulomb FFT (N_ext_x, N_ext_y),
// the orbital table (N_ext_x, N_ext_y, dim_orb, dim_orb) or an empty array,
// the per-G offsets (n_G, 2) and the ΔG pair CSR.  ``hermitian_rho`` opts
// into the ΔG <-> -ΔG conjugate halving, valid only for a Hermitian rho.
inline ffi::Error SuperlatticeFockImpl(ffi::Buffer<ffi::C128> rho,
                                       ffi::Buffer<ffi::C128> v_fft,
                                       ffi::Buffer<ffi::C128> v_fft_orbital,
                                       ffi::Buffer<ffi::S64> g_a_off,
                                       ffi::Buffer<ffi::S64> pair_i,
                                       ffi::Buffer<ffi::S64> pair_j,
                                       ffi::Buffer<ffi::S64> pair_start,
                                       ffi::ResultBuffer<ffi::C128> sigma,
                                       bool hermitian_rho) {
    const auto rd = rho.dimensions();
    const auto vd = v_fft.dimensions();
    const auto gd = g_a_off.dimensions();
    if (rd.size() != 4 || rd[2] != rd[3]) {
        return ffi::Error::InvalidArgument("superlattice Fock needs rho of shape (nkx, nky, D, D)");
    }
    if (vd.size() != 2 || gd.size() != 2 || gd[1] != 2 || pair_start.dimensions().size() != 1 ||
        pair_start.element_count() < 1) {
        return ffi::Error::InvalidArgument("superlattice Fock got malformed layout tables");
    }
    const std::size_t nkx = rd[0], nky = rd[1], D = rd[2];
    const std::size_t n_G = gd[0];
    if (n_G == 0 || D % n_G != 0) {
        return ffi::Error::InvalidArgument("superlattice Fock needs D to be a multiple of n_G");
    }
    const std::size_t dim_orb = D / n_G;
    const std::size_t nx = vd[0], ny = vd[1];
    const c64* orbital = nullptr;
    if (v_fft_orbital.element_count() != 0) {
        if (v_fft_orbital.element_count() != nx * ny * dim_orb * dim_orb) {
            return ffi::Error::InvalidArgument(
                "superlattice Fock orbital table must be (N_ext_x, N_ext_y, dim_orb, dim_orb)");
        }
        orbital = reinterpret_cast<const c64*>(v_fft_orbital.typed_data());
    }
    const std::size_t n_delta = pair_start.element_count() - 1;
    selfenergy_superlattice_streamed(
        reinterpret_cast<const c64*>(rho.typed_data()),
        reinterpret_cast<c64*>(sigma->typed_data()),
        reinterpret_cast<const c64*>(v_fft.typed_data()), orbital,
        g_a_off.typed_data(), pair_i.typed_data(), pair_j.typed_data(),
        pair_start.typed_data(), n_delta, nx, ny, nkx, nky, n_G, dim_orb, hermitian_rho);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(SuperlatticeFock, SuperlatticeFockImpl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::C128>>()
                                  .Arg<ffi::Buffer<ffi::C128>>()
                                  .Arg<ffi::Buffer<ffi::C128>>()
                                  .Arg<ffi::Buffer<ffi::S64>>()
                                  .Arg<ffi::Buffer<ffi::S64>>()
                                  .Arg<ffi::Buffer<ffi::S64>>()
                                  .Arg<ffi::Buffer<ffi::S64>>()
                                  .Ret<ffi::Buffer<ffi::C128>>()
                                  .Attr<bool>("hermitian_rho"));

}  // namespace xla_ffi
}  // namespace cpp_hf
