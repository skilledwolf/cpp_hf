cmake_minimum_required(VERSION 3.21)

# Project
project(cpp_hf LANGUAGES CXX)

# -----------------------------
# Options (simple & explicit)
# -----------------------------
option(HF_USE_OPENMP         "Enable OpenMP parallelization"          ON)
option(HF_USE_FFTW_THREADS   "Link FFTW threads and call init"        ON)
# For wheel builds we provide FFTW via CI 'before-all'; keep OFF by default.
option(HF_FETCH_FFTW         "Auto-download & build FFTW if not found" OFF)

# -----------------------------
# C++ standard (lean default)
# -----------------------------
if (NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 23)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# -----------------------------
# Python + pybind11
# -----------------------------
find_package(Python   REQUIRED COMPONENTS Interpreter Development.Module NumPy)
find_package(pybind11 CONFIG    QUIET)
if (NOT pybind11_FOUND)
  include(FetchContent)
  FetchContent_Declare(pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.12.0
  )
  FetchContent_MakeAvailable(pybind11)
endif()

# -----------------------------
# Eigen (header-only)
# -----------------------------
find_package(Eigen3 QUIET NO_MODULE)
if (NOT Eigen3_FOUND)
  include(FetchContent)
  FetchContent_Declare(eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        3.4.0
  )
  FetchContent_MakeAvailable(eigen)
endif()

# -----------------------------
# Boost (headers only)
# -----------------------------
find_package(Boost 1.66 QUIET)
if(NOT TARGET Boost::headers)
  add_library(Boost::headers INTERFACE IMPORTED)
endif()
if(Boost_FOUND)
  set_target_properties(Boost::headers PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${Boost_INCLUDE_DIRS}")
elseif(DEFINED ENV{BOOST_INCLUDEDIR})
  set_target_properties(Boost::headers PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "$ENV{BOOST_INCLUDEDIR}")
endif()


# ---- FFTW via pkg-config (preferred in CI) ----
find_package(PkgConfig QUIET)

set(FFTW_LIBS "")
set(FFTW_INCLUDE_DIRS "")

if (PKG_CONFIG_FOUND)
  pkg_check_modules(PC_FFTW3 QUIET fftw3)
  if (PC_FFTW3_FOUND)
    list(APPEND FFTW_LIBS         ${PC_FFTW3_LINK_LIBRARIES})
    list(APPEND FFTW_INCLUDE_DIRS ${PC_FFTW3_INCLUDE_DIRS})
  endif()

  if (HF_USE_FFTW_THREADS)
    pkg_check_modules(PC_FFTW3_THREADS QUIET fftw3_threads)
    if (PC_FFTW3_THREADS_FOUND)
      list(APPEND FFTW_LIBS         ${PC_FFTW3_THREADS_LINK_LIBRARIES})
      list(APPEND FFTW_INCLUDE_DIRS ${PC_FFTW3_THREADS_INCLUDE_DIRS})
      add_compile_definitions(FFTW3_THREADS=1)
    endif()
  endif()
endif()

# Fallback: raw find_path/find_library (for local dev)
if (FFTW_LIBS STREQUAL "")
  find_path(FFTW3_INCLUDE_DIR fftw3.h)
  find_library(FFTW3_LIBRARY NAMES fftw3)
  if (FFTW3_INCLUDE_DIR AND FFTW3_LIBRARY)
    list(APPEND FFTW_INCLUDE_DIRS "${FFTW3_INCLUDE_DIR}")
    list(APPEND FFTW_LIBS         "${FFTW3_LIBRARY}")
  endif()
  if (HF_USE_FFTW_THREADS)
    find_library(FFTW3_THREADS_LIBRARY NAMES fftw3_threads)
    if (FFTW3_THREADS_LIBRARY)
      list(APPEND FFTW_LIBS "${FFTW3_THREADS_LIBRARY}")
      add_compile_definitions(FFTW3_THREADS=1)
    endif()
  endif()
endif()

if (FFTW_LIBS STREQUAL "")
  message(FATAL_ERROR "FFTW not found. Ensure pkg-config files or libraries are visible.")
endif()


# -----------------------------
# OpenMP (optional)
# -----------------------------
set(HF_OPENMP_LIB "")
if (HF_USE_OPENMP)
  find_package(OpenMP QUIET)
  if (OpenMP_CXX_FOUND)
    set(HF_OPENMP_LIB OpenMP::OpenMP_CXX)
  endif()
endif()

# ---- mdspan detection & fallback ----
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_QUIET ON)

# Try <mdspan>
check_cxx_source_compiles([=[
  #include <mdspan>
  int main() {
    std::mdspan<double, std::extents<size_t, 2>> a(nullptr, std::extents<size_t,2>{});
    return 0;
  }
]=] HAVE_STD_MDSPAN)

# Try <experimental/mdspan> if needed
if (NOT HAVE_STD_MDSPAN)
  check_cxx_source_compiles([=[
    #include <experimental/mdspan>
    using E = std::experimental::extents<size_t,2>;
    int main() { std::experimental::mdspan<double, E> a(nullptr, E{}); return 0; }
  ]=] HAVE_EXP_MDSPAN)
endif()

# Vendor reference mdspan if neither header is available
if (NOT HAVE_STD_MDSPAN AND NOT HAVE_EXP_MDSPAN)
  include(FetchContent)
  FetchContent_Declare(kokkos_mdspan
    GIT_REPOSITORY https://github.com/kokkos/mdspan.git
    GIT_TAG        stable
  )
  FetchContent_MakeAvailable(kokkos_mdspan)
  # It is header-only; just add the include dir where mdspan/mdspan.hpp lives
  set(MDSPAN_INCLUDE_DIR "${kokkos_mdspan_SOURCE_DIR}/include")
endif()


# -----------------------------
# Sources
# -----------------------------
# Keep it robust for small projects: auto-collect .cpp files in tree
file(GLOB_RECURSE CPP_HF_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

# -----------------------------
# Extension target
# -----------------------------
pybind11_add_module(cpp_hf ${CPP_HF_SOURCES})
# Respect CMAKE_CXX_STANDARD set above / via scikit-build-core
# If we vendored mdspan, add its include dir
if (DEFINED MDSPAN_INCLUDE_DIR)
  target_include_directories(cpp_hf PRIVATE "${MDSPAN_INCLUDE_DIR}")
endif()
target_include_directories(cpp_hf PRIVATE ${FFTW_INCLUDE_DIRS})
target_link_libraries(cpp_hf PRIVATE ${FFTW_LIBS})
target_include_directories(cpp_hf
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)
target_compile_options(cpp_hf PRIVATE -fvisibility=hidden)
target_link_libraries(cpp_hf PRIVATE Eigen3::Eigen Boost::headers ${HF_OPENMP_LIB})

# Install the binary into the python package so the wheel includes it
# (place the extension inside the 'cpp_hf' package directory)
install(TARGETS cpp_hf
  LIBRARY DESTINATION "${SKBUILD_PLATLIB_DIR}/cpp_hf"
)
