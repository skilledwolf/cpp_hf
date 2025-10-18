// hf_mdspan.hpp - portable mdspan include shim
#pragma once

#if __has_include(<mdspan>)
  #include <mdspan>
  namespace stdx = std;
#elif __has_include(<experimental/mdspan>)
  #include <experimental/mdspan>
  namespace stdx = std::experimental;
#elif __has_include(<mdspan/mdspan.hpp>)
  // Reference implementation vendored via CMake FetchContent (Kokkos mdspan)
  #include <mdspan/mdspan.hpp>
  namespace stdx = std;  // ref impl exports into std
#else
  #error "No mdspan found: need <mdspan>, <experimental/mdspan>, or a vendored mdspan."
#endif

