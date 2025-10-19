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
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
endif()

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
if (NOT Boost_FOUND)
  # Create a header-only interface target if not present
  add_library(Boost::headers INTERFACE IMPORTED)
else()
  # CMake >=3.70 has Boost::headers; if not, emulate via INTERFACE_INCLUDE_DIRECTORIES
  if(NOT TARGET Boost::headers)
    add_library(Boost::headers INTERFACE IMPORTED)
    set_target_properties(Boost::headers PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${Boost_INCLUDE_DIRS}")
  endif()
endif()

# -----------------------------
# FFTW (pkg-config preferred)
# -----------------------------
set(HF_FFTW_LIBS "")
set(HF_FFTW_INCLUDE_DIRS "")
find_package(PkgConfig QUIET)

if (PKG_CONFIG_FOUND)
  # Prefer pkg-config (matches CI install location cleanly)
  pkg_check_modules(FFTW3 QUIET fftw3)
  if (FFTW3_FOUND)
    list(APPEND HF_FFTW_LIBS ${FFTW3_LINK_LIBRARIES})
    list(APPEND HF_FFTW_INCLUDE_DIRS ${FFTW3_INCLUDE_DIRS})
  endif()

  if (HF_USE_FFTW_THREADS)
    pkg_check_modules(FFTW3_THREADS QUIET fftw3_threads)
    if (FFTW3_THREADS_FOUND)
      list(APPEND HF_FFTW_LIBS ${FFTW3_THREADS_LINK_LIBRARIES})
      list(APPEND HF_FFTW_INCLUDE_DIRS ${FFTW3_THREADS_INCLUDE_DIRS})
      add_compile_definitions(FFTW3_THREADS=1)
    endif()
  endif()
endif()

# Fallback: try raw libraries if pkg-config didn’t find them
if (HF_FFTW_LIBS STREQUAL "")
  find_path(FFTW3_INCLUDE_DIR fftw3.h)
  find_library(FFTW3_LIBRARY     NAMES fftw3)
  if (FFTW3_INCLUDE_DIR AND FFTW3_LIBRARY)
    list(APPEND HF_FFTW_INCLUDE_DIRS "${FFTW3_INCLUDE_DIR}")
    list(APPEND HF_FFTW_LIBS         "${FFTW3_LIBRARY}")
  endif()

  if (HF_USE_FFTW_THREADS)
    find_library(FFTW3_THREADS_LIBRARY NAMES fftw3_threads)
    if (FFTW3_THREADS_LIBRARY)
      list(APPEND HF_FFTW_LIBS "${FFTW3_THREADS_LIBRARY}")
      add_compile_definitions(FFTW3_THREADS=1)
    endif()
  endif()
endif()

# Optional: developer convenience fetch (kept OFF in CI)
if (HF_FETCH_FFTW AND HF_FFTW_LIBS STREQUAL "")
  include(ExternalProject)
  set(_fftw_url "https://www.fftw.org/fftw-3.3.10.tar.gz")
  ExternalProject_Add(ext_fftw
    URL               ${_fftw_url}
    PREFIX            ${CMAKE_BINARY_DIR}/ext_fftw
    CONFIGURE_COMMAND ./configure --prefix=${CMAKE_BINARY_DIR}/ext_fftw/prefix
                                  --enable-shared --disable-static
                                  --disable-fortran --enable-threads
    BUILD_COMMAND     $(MAKE) -j
    INSTALL_COMMAND   $(MAKE) install
  )
  add_library(fftw3 SHARED IMPORTED)
  set_target_properties(fftw3 PROPERTIES
    IMPORTED_LOCATION "${CMAKE_BINARY_DIR}/ext_fftw/prefix/lib/libfftw3.${CMAKE_SHARED_LIBRARY_SUFFIX}")
  add_dependencies(fftw3 ext_fftw)
  list(APPEND HF_FFTW_LIBS fftw3)
  include_directories("${CMAKE_BINARY_DIR}/ext_fftw/prefix/include")
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
target_compile_features(cpp_hf PRIVATE cxx_std_23)
target_include_directories(cpp_hf
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)
target_compile_options(cpp_hf PRIVATE -fvisibility=hidden)
target_include_directories(cpp_hf PRIVATE ${HF_FFTW_INCLUDE_DIRS})
target_link_libraries(cpp_hf PRIVATE
  Eigen3::Eigen
  Boost::headers
  ${HF_FFTW_LIBS}
  ${HF_OPENMP_LIB}
)

# scikit-build-core will handle installation into the wheel; no manual install()
