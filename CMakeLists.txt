cmake_minimum_required(VERSION 3.21)

project(cpp_hf LANGUAGES CXX)

# -----------------------------
# Options
# -----------------------------
option(HF_USE_OPENMP       "Enable OpenMP parallelization"   ON)
option(HF_USE_FFTW_THREADS "Link FFTW threads and call init" ON)
option(HF_USE_ACCELERATE   "Route Eigen dense products to Apple Accelerate (AMX) on macOS" ON)

# -----------------------------
# C++ standard
# -----------------------------
if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# -----------------------------
# Python + pybind11
# -----------------------------
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module NumPy)
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
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
if(NOT Eigen3_FOUND)
    include(FetchContent)
    FetchContent_Declare(eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG        3.4.0
    )
    FetchContent_MakeAvailable(eigen)
endif()

# -----------------------------
# FFTW (single + double; we use single)
# -----------------------------
find_package(PkgConfig QUIET)

set(FFTW_LIBS "")
set(FFTW_INCLUDE_DIRS "")

if(PKG_CONFIG_FOUND)
    pkg_check_modules(PC_FFTW3 QUIET fftw3)
    if(PC_FFTW3_FOUND)
        list(APPEND FFTW_LIBS         ${PC_FFTW3_LINK_LIBRARIES})
        list(APPEND FFTW_INCLUDE_DIRS ${PC_FFTW3_INCLUDE_DIRS})
    endif()
    if(HF_USE_FFTW_THREADS)
        pkg_check_modules(PC_FFTW3_THREADS QUIET fftw3_threads)
        if(PC_FFTW3_THREADS_FOUND)
            list(APPEND FFTW_LIBS         ${PC_FFTW3_THREADS_LINK_LIBRARIES})
            list(APPEND FFTW_INCLUDE_DIRS ${PC_FFTW3_THREADS_INCLUDE_DIRS})
            add_compile_definitions(FFTW3_THREADS=1)
        endif()
    endif()
endif()

if(FFTW_LIBS STREQUAL "")
    find_path(FFTW3_INCLUDE_DIR fftw3.h)
    find_library(FFTW3_LIBRARY NAMES fftw3)
    if(FFTW3_INCLUDE_DIR AND FFTW3_LIBRARY)
        list(APPEND FFTW_INCLUDE_DIRS "${FFTW3_INCLUDE_DIR}")
        list(APPEND FFTW_LIBS         "${FFTW3_LIBRARY}")
    endif()
    if(HF_USE_FFTW_THREADS)
        find_library(FFTW3_THREADS_LIBRARY NAMES fftw3_threads)
        if(FFTW3_THREADS_LIBRARY)
            list(APPEND FFTW_LIBS "${FFTW3_THREADS_LIBRARY}")
            add_compile_definitions(FFTW3_THREADS=1)
        endif()
    endif()
endif()

if(FFTW_LIBS STREQUAL "")
    message(FATAL_ERROR "FFTW (double precision) not found. Install fftw3.")
endif()

# -----------------------------
# OpenMP (optional)
# -----------------------------
set(HF_OPENMP_LIB "")
if(HF_USE_OPENMP)
    find_package(OpenMP QUIET)
    if(NOT OpenMP_CXX_FOUND AND APPLE)
        # Apple clang ships without the OpenMP driver flag; wire homebrew
        # libomp explicitly (-Xpreprocessor -fopenmp + libomp).
        find_path(HF_LIBOMP_INCLUDE omp.h
                  HINTS /opt/homebrew/opt/libomp/include /usr/local/opt/libomp/include)
        find_library(HF_LIBOMP_LIB omp
                     HINTS /opt/homebrew/opt/libomp/lib /usr/local/opt/libomp/lib)
        if(HF_LIBOMP_INCLUDE AND HF_LIBOMP_LIB)
            set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${HF_LIBOMP_INCLUDE}")
            set(OpenMP_CXX_LIB_NAMES "omp")
            set(OpenMP_omp_LIBRARY "${HF_LIBOMP_LIB}")
            find_package(OpenMP QUIET)
        endif()
    endif()
    if(OpenMP_CXX_FOUND)
        set(HF_OPENMP_LIB OpenMP::OpenMP_CXX)
        message(STATUS "cpp_hf: OpenMP enabled (${OpenMP_CXX_FLAGS})")
    else()
        message(STATUS "cpp_hf: OpenMP NOT found — building serial kernels")
    endif()
endif()

# -----------------------------
# Native extension
# -----------------------------
pybind11_add_module(_native ${CMAKE_CURRENT_SOURCE_DIR}/cpp/cpp_hf_native.cpp)
target_include_directories(_native PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include
    ${FFTW_INCLUDE_DIRS}
)
target_link_libraries(_native PRIVATE
    Eigen3::Eigen
    ${FFTW_LIBS}
    ${HF_OPENMP_LIB}
)
target_compile_options(_native PRIVATE -fvisibility=hidden -O3)

# Apple Accelerate: EIGEN_USE_BLAS routes Eigen's dynamic-size complex matrix
# products (the per-k Q·diag·Q†, Q†FQ, V†FV, retraction gemms that dominate
# the DM solver's non-Fock time) to Accelerate's AMX-backed zgemm — measured
# 3.3× over Eigen's own kernels at the (64, 152, 152) CDW workload.  The
# Hermitian eigensolver stays Eigen (Accelerate zheevd measured ~1.1×, not
# worth a LAPACK path).  Single-threaded Accelerate (VECLIB_MAXIMUM_THREADS)
# still uses AMX, so this composes with process-level sweep parallelism.
if(APPLE AND HF_USE_ACCELERATE)
    target_compile_definitions(_native PRIVATE EIGEN_USE_BLAS)
    target_link_libraries(_native PRIVATE "-framework Accelerate")
endif()

install(TARGETS _native DESTINATION cpp_hf)
