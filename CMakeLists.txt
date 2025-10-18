cmake_minimum_required(VERSION 3.21)

# Version from scikit-build-core (pyproject.toml) if present
if (DEFINED SKBUILD_PROJECT_VERSION AND NOT "${SKBUILD_PROJECT_VERSION}" STREQUAL "")
  set(CPP_HF_VERSION "${SKBUILD_PROJECT_VERSION}")
else()
  set(CPP_HF_VERSION "0.0")
endif()

project(cpp_hf VERSION ${CPP_HF_VERSION} LANGUAGES CXX)

# Policies
if (POLICY CMP0167)  # Prefer CONFIG for Boost on newer CMake
  cmake_policy(SET CMP0167 NEW)
endif()

# Options
option(HF_USE_OPENMP       "Enable OpenMP parallelization over k-points" ON)
option(HF_USE_FFTW_THREADS "Link FFTW threads and enable fftw_init_threads()" ON)
option(HF_AGGRESSIVE_OPT   "Enable aggressive CPU/FP opts" OFF)
option(HF_LTO              "Enable link-time optimization (IPO) if supported" OFF)
option(HF_BUILD_BENCH      "Build the FFTW benchmark helper" OFF)

# Python + NumPy + pybind11
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module NumPy)

find_package(pybind11 CONFIG QUIET)
if (NOT pybind11_FOUND)
  include(FetchContent)
  FetchContent_Declare(pybind11 GIT_REPOSITORY https://github.com/pybind/pybind11.git GIT_TAG v2.12.0)
  FetchContent_MakeAvailable(pybind11)
endif()

# Eigen3 (header-only)
find_package(Eigen3 QUIET NO_MODULE)
if (NOT Eigen3_FOUND)
  include(FetchContent)
  FetchContent_Declare(Eigen GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git GIT_TAG 3.4.0)
  FetchContent_MakeAvailable(Eigen)
  if (NOT TARGET Eigen3::Eigen)
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    target_include_directories(Eigen3::Eigen INTERFACE "${eigen_SOURCE_DIR}")
  endif()
endif()

# Boost headers (for Boost.Math)
find_package(Boost 1.66.0 REQUIRED COMPONENTS headers)

# mdspan availability and optional vendoring
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_QUIET ON)

# Ensure the test compiles with at least C++23 so <mdspan> is detectable where supported
if (NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 23)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
endif()

check_cxx_source_compiles(
  "#include <mdspan>\nint main(){ std::mdspan<int, std::extents<size_t,2>> m(nullptr, std::extents<size_t,2>{}); return 0; }"
  HAVE_STD_MDSPAN)

if (NOT HAVE_STD_MDSPAN)
  check_cxx_source_compiles(
    "#include <experimental/mdspan>\nusing mdspan_t = std::experimental::mdspan<int, std::experimental::extents<size_t,2>>; int main(){ return 0; }"
    HAVE_EXP_MDSPAN)
endif()

if (NOT HAVE_STD_MDSPAN AND NOT HAVE_EXP_MDSPAN)
  include(FetchContent)
  FetchContent_Declare(
    mdspan_ref
    GIT_REPOSITORY https://github.com/kokkos/mdspan.git
    GIT_TAG stable
  )
  FetchContent_MakeAvailable(mdspan_ref)
  add_library(mdspan_ref INTERFACE)
  # Reference impl installs headers under include/mdspan/...
  target_include_directories(mdspan_ref INTERFACE "${mdspan_ref_SOURCE_DIR}/include")
endif()

# FFTW: prefer CMake package config, then fall back to pkg-config
set(HF_FFTW_LIBS "")
set(HF_HAVE_FFTW_THREADS FALSE)

# 1) Try CMake package config only if explicitly pointed to via env/var
if (DEFINED ENV{FFTW3_DIR})
  set(_FFTW3_HINT "$ENV{FFTW3_DIR}")
  if (EXISTS "${_FFTW3_HINT}/FFTW3Config.cmake")
    find_package(FFTW3 CONFIG QUIET PATHS "${_FFTW3_HINT}" NO_DEFAULT_PATH)
    if (TARGET FFTW3::fftw3)
      list(APPEND HF_FFTW_LIBS FFTW3::fftw3)
    endif()
    if (HF_USE_FFTW_THREADS AND TARGET FFTW3::fftw3_threads)
      list(APPEND HF_FFTW_LIBS FFTW3::fftw3_threads)
      set(HF_HAVE_FFTW_THREADS TRUE)
    endif()
  endif()
endif()

# 2) Fallback to pkg-config only if CMake package not found
if (HF_FFTW_LIBS STREQUAL "")
  find_package(PkgConfig QUIET)
  if (PkgConfig_FOUND)
    pkg_check_modules(FFTW3 QUIET IMPORTED_TARGET fftw3)
    if (FFTW3_FOUND)
      # Guard against stale/broken pc files; accept if at least one include dir exists
      get_target_property(_fftw_inc PkgConfig::FFTW3 INTERFACE_INCLUDE_DIRECTORIES)
      set(_fftw_has_valid_inc FALSE)
      foreach(dir IN LISTS _fftw_inc)
        if (dir AND EXISTS "${dir}")
          set(_fftw_has_valid_inc TRUE)
        endif()
      endforeach()
      if (_fftw_has_valid_inc)
        list(APPEND HF_FFTW_LIBS PkgConfig::FFTW3)
      else()
        message(WARNING "Ignoring pkg-config FFTW3: no valid include dirs found")
      endif()
    endif()
    if (HF_USE_FFTW_THREADS)
      pkg_check_modules(FFTW3_THREADS QUIET IMPORTED_TARGET fftw3_threads)
      if (FFTW3_THREADS_FOUND)
        # Similar sanity check for threads target; accept if any include dir exists
        get_target_property(_fftw_thr_inc PkgConfig::FFTW3_THREADS INTERFACE_INCLUDE_DIRECTORIES)
        set(_fftw_thr_has_valid_inc FALSE)
        foreach(dir IN LISTS _fftw_thr_inc)
          if (dir AND EXISTS "${dir}")
            set(_fftw_thr_has_valid_inc TRUE)
          endif()
        endforeach()
        if (_fftw_thr_has_valid_inc)
          list(APPEND HF_FFTW_LIBS PkgConfig::FFTW3_THREADS)
          set(HF_HAVE_FFTW_THREADS TRUE)
        else()
          message(WARNING "Ignoring pkg-config FFTW3_THREADS: no valid include dirs found")
        endif()
      endif()
    endif()
  endif()
endif()

# 3) If still not found, error with a helpful message
if (HF_FFTW_LIBS STREQUAL "")
  message(FATAL_ERROR "FFTW3 not found. Please install FFTW and set FFTW3_DIR to the directory containing FFTW3Config.cmake (e.g., /opt/homebrew/lib/cmake/fftw3 or $HOME/.local/lib/cmake/fftw3), or ensure pkg-config has a valid fftw3.pc.")
endif()

# POSIX threads may be required by FFTW threads
if (HF_HAVE_FFTW_THREADS)
  find_package(Threads QUIET)
  if (Threads_FOUND)
    list(APPEND HF_FFTW_LIBS Threads::Threads)
  endif()
endif()

# OpenMP (optional)
set(HF_OPENMP_LIB "")
if (HF_USE_OPENMP)
  find_package(OpenMP QUIET)
  if (OpenMP_CXX_FOUND)
    set(HF_OPENMP_LIB OpenMP::OpenMP_CXX)
  endif()
endif()

# The Python extension module
pybind11_add_module(cpp_hf cpp_hf.cpp src/hartreefock.cpp)
target_compile_features(cpp_hf PRIVATE cxx_std_23)
target_compile_options(cpp_hf PRIVATE -O3 -DNDEBUG -fvisibility=hidden)

# Aggressive opts are OFF by default to keep wheels portable/reproducible
if (HF_AGGRESSIVE_OPT AND NOT MSVC)
  include(CheckCXXCompilerFlag)
  foreach(flag IN ITEMS -Ofast -ffast-math -ffp-contract=fast -funroll-loops -fopenmp-simd)
    string(REPLACE "-" "_" key ${flag})
    check_cxx_compiler_flag("${flag}" HAVE_${key})
    if (HAVE_${key})
      target_compile_options(cpp_hf PRIVATE "${flag}")
    endif()
  endforeach()
  # Arch tuning
  foreach(flag IN ITEMS -march=native -mcpu=native)
    string(REPLACE "-" "_" key ${flag})
    check_cxx_compiler_flag("${flag}" HAVE_${key})
    if (HAVE_${key})
      target_compile_options(cpp_hf PRIVATE "${flag}")
      break()
    endif()
  endforeach()
endif()

# LTO via IPO
if (HF_LTO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT _ipo_ok OUTPUT _ipo_msg)
  if (_ipo_ok)
    set_property(TARGET cpp_hf PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
  else()
    message(WARNING "IPO not supported: ${_ipo_msg}")
  endif()
endif()

# Includes & link
target_include_directories(cpp_hf PRIVATE
  ${Python_NumPy_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(cpp_hf PRIVATE
  Eigen3::Eigen
  Boost::headers
  ${HF_FFTW_LIBS}
  ${HF_OPENMP_LIB}
)
if (TARGET mdspan_ref)
  target_link_libraries(cpp_hf PRIVATE mdspan_ref)
endif()
if (HF_HAVE_FFTW_THREADS)
  target_compile_definitions(cpp_hf PRIVATE FFTW3_THREADS=1)
endif()

# Install into scikit-build-core’s platlib dir (fallback if not set)
if (NOT DEFINED SKBUILD_PLATLIB_DIR OR SKBUILD_PLATLIB_DIR STREQUAL "")
  set(SKBUILD_PLATLIB_DIR "${CMAKE_CURRENT_BINARY_DIR}/python" CACHE PATH "Python package output dir")
endif()
install(TARGETS cpp_hf
  LIBRARY DESTINATION ${SKBUILD_PLATLIB_DIR}
  RUNTIME DESTINATION ${SKBUILD_PLATLIB_DIR}
  ARCHIVE DESTINATION ${SKBUILD_PLATLIB_DIR}
)

# Optional: FFTW benchmark
if (HF_BUILD_BENCH)
  add_executable(bench_fftw tests/bench_fftw.cpp)
  target_compile_features(bench_fftw PRIVATE cxx_std_23)
  target_include_directories(bench_fftw PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(bench_fftw PRIVATE ${HF_FFTW_LIBS})
  if (TARGET mdspan_ref)
    target_link_libraries(bench_fftw PRIVATE mdspan_ref)
  endif()
endif()
