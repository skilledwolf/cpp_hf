// mdspan_compat.hpp - unified include for std::mdspan or backport
#pragma once

#if __has_include(<version>)
#  include <version>
#endif

#if defined(__cpp_lib_mdspan) && __cpp_lib_mdspan >= 202207L && __has_include(<mdspan>)
#  include <mdspan>
namespace md = std;
#else
// Fallback to backport (kokkos/mdspan) which provides <experimental/mdspan>
#  include <experimental/mdspan>
namespace md = std::experimental;
#endif

