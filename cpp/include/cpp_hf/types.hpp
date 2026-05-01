// Common types and constants for the cpp_hf C++ core (double precision).
#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>

namespace cpp_hf {

using c64 = std::complex<double>;
using f32 = double;

inline constexpr f32 TINY_REAL = 1.0e-30;
inline constexpr f32 LOGIT_CLIP = 1.0e-6;
inline constexpr f32 ENTROPY_CLIP = 1.0e-14;
inline constexpr f32 OCC_CLIP = 1.0e-8;

}  // namespace cpp_hf
