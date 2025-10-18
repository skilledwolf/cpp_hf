cmake_minimum_required(VERSION 3.21)

# Prefer the project version provided by scikit-build-core when building wheels.
# This lets you set the version in one place (pyproject.toml), and CMake picks it up
# via the SKBUILD_PROJECT_VERSION cache variable. Fallback used for direct CMake builds.
if (DEFINED SKBUILD_PROJECT_VERSION AND NOT "${SKBUILD_PROJECT_VERSION}" STREQUAL "")
  set(CPP_HF_VERSION "${SKBUILD_PROJECT_VERSION}")
else()
  # Fallback for plain CMake builds without scikit-build-core
  set(CPP_HF_VERSION "0.0")
endif()

project(cpp_hf VERSION ${CPP_HF_VERSION} LANGUAGES CXX)

# Prefer CONFIG packages for Boost on newer CMake
if (POLICY CMP0167)
  cmake_policy(SET CMP0167 NEW)
endif()

# Options
option(HF_USE_OPENMP "Enable OpenMP parallelization over k-points" ON)
option(HF_USE_FFTW_THREADS "Link FFTW threads and enable fftw_init_threads()" ON)

# Python + NumPy
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module NumPy)

# pybind11
find_package(pybind11 CONFIG QUIET)
if (NOT pybind11_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.12.0
  )
  FetchContent_MakeAvailable(pybind11)
endif()

# Eigen3
find_package(Eigen3 QUIET NO_MODULE)
if (NOT Eigen3_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    Eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
  )
  FetchContent_MakeAvailable(Eigen)
  # Some Eigen builds already define Eigen3::Eigen; only create an interface
  # target if it is missing (header-only include target).
  if (NOT TARGET Eigen3::Eigen)
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    target_include_directories(Eigen3::Eigen INTERFACE "${eigen_SOURCE_DIR}")
  endif()
endif()

# Boost headers (for Boost.Math). Try CONFIG first; fallback to include path.
set(BOOST_HEADERS_TARGET "")
find_package(Boost 1.70 QUIET CONFIG COMPONENTS headers)
if (Boost_FOUND)
  if (TARGET Boost::headers)
    set(BOOST_HEADERS_TARGET Boost::headers)
  elseif (TARGET Boost::boost)
    set(BOOST_HEADERS_TARGET Boost::boost)
  endif()
else()
  find_path(BOOST_INCLUDE_DIR
            NAMES boost/math/tools/toms748_solve.hpp
            PATHS /opt/homebrew/include /usr/local/include
            DOC "Path to Boost headers")
  if (NOT BOOST_INCLUDE_DIR)
    message(FATAL_ERROR "Boost headers not found; install Boost (brew install boost) or set CMAKE_PREFIX_PATH.")
  endif()
endif()

# FFTW (pkg-config preferred; fallback to CMake config and manual find)
set(HF_FFTW_LIBS "")
set(HF_HAVE_FFTW_THREADS FALSE)
find_package(PkgConfig QUIET)
if (PkgConfig_FOUND)
  pkg_check_modules(FFTW3 QUIET IMPORTED_TARGET fftw3)
  if (FFTW3_FOUND)
    list(APPEND HF_FFTW_LIBS PkgConfig::FFTW3)
  endif()
  if (HF_USE_FFTW_THREADS)
    pkg_check_modules(FFTW3_THREADS QUIET IMPORTED_TARGET fftw3_threads)
    if (FFTW3_THREADS_FOUND)
      list(APPEND HF_FFTW_LIBS PkgConfig::FFTW3_THREADS)
      set(HF_HAVE_FFTW_THREADS TRUE)
    endif()
  endif()
endif()
if (HF_FFTW_LIBS STREQUAL "")
  find_package(FFTW3 CONFIG QUIET)
  if (FFTW3_FOUND)
    if (TARGET FFTW3::fftw3)
      list(APPEND HF_FFTW_LIBS FFTW3::fftw3)
    endif()
    if (HF_USE_FFTW_THREADS AND TARGET FFTW3::fftw3_threads)
      list(APPEND HF_FFTW_LIBS FFTW3::fftw3_threads)
      set(HF_HAVE_FFTW_THREADS TRUE)
    endif()
  endif()
endif()
if (HF_FFTW_LIBS STREQUAL "")
  message(FATAL_ERROR "FFTW3 not found. Install with 'brew install fftw' or set PKG_CONFIG_PATH/FFTW3_DIR.")
endif()
if (HF_USE_FFTW_THREADS AND NOT HF_HAVE_FFTW_THREADS)
  find_library(FFTW3_THREADS_LIB NAMES fftw3_threads fftw_threads PATHS /opt/homebrew/lib /usr/local/lib ENV LIBRARY_PATH)
  if (FFTW3_THREADS_LIB)
    list(APPEND HF_FFTW_LIBS ${FFTW3_THREADS_LIB})
    set(HF_HAVE_FFTW_THREADS TRUE)
  else()
    message(WARNING "Requested HF_USE_FFTW_THREADS=ON but libfftw3_threads not found; building without FFTW threading.")
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

# Build module (Python name: cpp_hf)
pybind11_add_module(cpp_hf cpp_hf.cpp src/hartreefock.cpp)
target_compile_features(cpp_hf PRIVATE cxx_std_23)
if (MSVC)
  target_compile_options(cpp_hf PRIVATE /O2 /DNDEBUG)
  target_compile_definitions(cpp_hf PRIVATE _SILENCE_CXX17_RESULT_OF_DEPRECATION_WARNING)
else()
  target_compile_options(cpp_hf PRIVATE -O3 -DNDEBUG -fvisibility=hidden)
  include(CheckCXXCompilerFlag)
  # Optional, CPU-tuned flags if supported by the host compiler
  check_cxx_compiler_flag("-Ofast" HAVE_OFAST)
  if (HAVE_OFAST)
    target_compile_options(cpp_hf PRIVATE -Ofast)
  endif()
  check_cxx_compiler_flag("-ffast-math" HAVE_FFAST_MATH)
  if (HAVE_FFAST_MATH)
    target_compile_options(cpp_hf PRIVATE -ffast-math)
  endif()
  check_cxx_compiler_flag("-ffp-contract=fast" HAVE_FFP_CONTRACT_FAST)
  if (HAVE_FFP_CONTRACT_FAST)
    target_compile_options(cpp_hf PRIVATE -ffp-contract=fast)
  endif()
  check_cxx_compiler_flag("-march=native" HAVE_MARCH_NATIVE)
  if (HAVE_MARCH_NATIVE)
    target_compile_options(cpp_hf PRIVATE -march=native)
  else()
    check_cxx_compiler_flag("-mcpu=native" HAVE_MCPU_NATIVE)
    if (HAVE_MCPU_NATIVE)
      target_compile_options(cpp_hf PRIVATE -mcpu=native)
    else()
      # Try Apple Silicon specific if available
      check_cxx_compiler_flag("-mcpu=apple-m1" HAVE_MCPU_APPLE_M1)
      if (HAVE_MCPU_APPLE_M1)
        target_compile_options(cpp_hf PRIVATE -mcpu=apple-m1)
      endif()
      check_cxx_compiler_flag("-mcpu=apple-m2" HAVE_MCPU_APPLE_M2)
      if (HAVE_MCPU_APPLE_M2)
        target_compile_options(cpp_hf PRIVATE -mcpu=apple-m2)
      endif()
    endif()
  endif()
  check_cxx_compiler_flag("-funroll-loops" HAVE_FUNROLL_LOOPS)
  if (HAVE_FUNROLL_LOOPS)
    target_compile_options(cpp_hf PRIVATE -funroll-loops)
  endif()
  check_cxx_compiler_flag("-fopenmp-simd" HAVE_FOPENMP_SIMD)
  if (HAVE_FOPENMP_SIMD)
    target_compile_options(cpp_hf PRIVATE -fopenmp-simd)
  endif()
  check_cxx_compiler_flag("-flto" HAVE_FLTO)
  if (HAVE_FLTO)
    target_compile_options(cpp_hf PRIVATE -flto)
    target_link_options(cpp_hf PRIVATE -flto)
  endif()
endif()
if (APPLE)
  set_target_properties(cpp_hf PROPERTIES MACOSX_RPATH ON)
endif()

# Headers
target_include_directories(cpp_hf PRIVATE
  ${Python_NumPy_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
if (DEFINED BOOST_INCLUDE_DIR)
  target_include_directories(cpp_hf PRIVATE ${BOOST_INCLUDE_DIR})
endif()

# Link
target_link_libraries(cpp_hf PRIVATE
  Eigen3::Eigen
  ${BOOST_HEADERS_TARGET}
  ${HF_FFTW_LIBS}
  ${HF_OPENMP_LIB}
)
if (HF_HAVE_FFTW_THREADS)
  target_compile_definitions(cpp_hf PRIVATE FFTW3_THREADS=1)
endif()

# Install
if (NOT DEFINED SKBUILD_PLATLIB_DIR OR SKBUILD_PLATLIB_DIR STREQUAL "")
  set(SKBUILD_PLATLIB_DIR "${CMAKE_CURRENT_BINARY_DIR}/python" CACHE PATH "Python package output dir")
endif()
install(TARGETS cpp_hf
  LIBRARY DESTINATION ${SKBUILD_PLATLIB_DIR}
  RUNTIME DESTINATION ${SKBUILD_PLATLIB_DIR}
  ARCHIVE DESTINATION ${SKBUILD_PLATLIB_DIR}
)

# Optional FFTW benchmark executable
add_executable(bench_fftw tests/bench_fftw.cpp)
target_compile_features(bench_fftw PRIVATE cxx_std_23)
target_include_directories(bench_fftw PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(bench_fftw PRIVATE ${HF_FFTW_LIBS})
