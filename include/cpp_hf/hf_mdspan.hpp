#pragma once

// Pull in feature-test macros if available
#if __has_include(<version>)
  #include <version>
#endif

// Allow CMake to tell us exactly which path to take
#if defined(HF_FORCE_STD_MDSPAN)
  #include <mdspan>
  namespace stdx = std;

#elif defined(HF_FORCE_EXP_MDSPAN)
  #include <experimental/mdspan>
  namespace stdx = std::experimental;

#elif defined(HF_FORCE_REF_MDSPAN)
  #include <mdspan/mdspan.hpp>
  // Recent Kokkos mdspan places the API in std::; prefer that.
  // If your vendored mdspan still uses std::experimental, switch the alias here.
  namespace stdx = std;

// Auto-detect
#else
  // True only when the <mdspan> header is present *and* actually defines the feature.
  #if __has_include(<mdspan>) && defined(__cpp_lib_mdspan) && (__cpp_lib_mdspan >= 202207L)
    #include <mdspan>
    namespace stdx = std;
  #elif __has_include(<experimental/mdspan>)
    #include <experimental/mdspan>
    namespace stdx = std::experimental;
  #elif __has_include(<mdspan/mdspan.hpp>)
    #include <mdspan/mdspan.hpp>
    // Default to std namespace for modern reference mdspan
    namespace stdx = std;
  #else
    #error "No usable mdspan header found. Install libstdc++/libc++ with mdspan, or vendor Kokkos mdspan."
  #endif
#endif
