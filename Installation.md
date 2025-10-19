# Installation Guide

This document covers advanced installation and build details beyond the quick start shown in the README.

- Supported wheels
- Building from source
- FFTW discovery tips
- Parallelization notes (OpenMP and FFTW threads)
- Profiling
- Development setup
- Troubleshooting

## Supported Wheels

Prebuilt wheels are provided via GitHub Actions for:
- Linux: `x86_64`, `aarch64` (manylinux_2_28)
- macOS: `x86_64`, `arm64`
- Python: 3.11, 3.12

PyPy and musllinux are currently skipped. Other platforms/Python versions build from source.

## Build From Source

Requirements:
- Python 3.8+
- CMake ≥ 3.21
- A C++23 compiler/toolchain
- FFTW 3 (shared libraries and headers)

The build uses scikit-build-core and pybind11. Eigen and Boost headers are fetched automatically if not available.

Recommended pip invocation (Release build with common toggles):

```
pip install . \
  -C cmake.build-type=Release \
  -C cmake.define.HF_USE_OPENMP=ON \
  -C cmake.define.HF_USE_FFTW_THREADS=ON
```

mdspan: If your standard library lacks `<mdspan>`, the build automatically uses the Kokkos mdspan backport. You can force it with:

```
pip install . -C cmake.define.HF_FORCE_MDSPAN_BACKPORT=ON
```

### Platform Tips

- macOS (Homebrew):
  - Install build deps: `brew install fftw libomp pkg-config cmake`
  - If needed, expose Homebrew prefixes to CMake/pkg-config:
    - `export PKG_CONFIG_PATH="$(brew --prefix)/lib/pkgconfig:${PKG_CONFIG_PATH:-}"`
    - `export CMAKE_PREFIX_PATH="$(brew --prefix):${CMAKE_PREFIX_PATH:-}"`
  - Apple Silicon: see `ci/macos_arm_fftw.sh` for a scripted FFTW build/caching approach.

- Linux:
  - See `ci/linux_before_all.sh` for a reference of building FFTW from source and vendor Boost headers for manylinux-style environments.

## FFTW Discovery

The build prefers pkg-config and falls back to direct library/header search. If FFTW is not found, point the build to your installation:

- Set `PKG_CONFIG_PATH` to the FFTW `lib/pkgconfig` directory.
- Optionally set `CMAKE_PREFIX_PATH` to the FFTW prefix.
- Compiler search paths can be aided via `CPATH` (headers) and `LIBRARY_PATH` (libraries).

Examples (adjust prefix):

```
export PKG_CONFIG_PATH=/opt/fftw/lib/pkgconfig:${PKG_CONFIG_PATH:-}
export CMAKE_PREFIX_PATH=/opt/fftw:${CMAKE_PREFIX_PATH:-}
export CPATH=/opt/fftw/include:${CPATH:-}
export LIBRARY_PATH=/opt/fftw/lib:${LIBRARY_PATH:-}
```

## Parallelization

- `HF_USE_OPENMP=ON` enables OpenMP parallelization over k-points.
- `HF_USE_FFTW_THREADS=ON` links FFTW’s threaded library and initializes it.

If both are enabled and you observe oversubscription, consider:
- `export OMP_NUM_THREADS=1` (let FFTW handle parallelism), or
- build with `-C cmake.define.HF_USE_FFTW_THREADS=OFF`, or
- tune OpenMP: `OMP_PROC_BIND=spread`, `OMP_PLACES=cores`.

## Profiling

Lightweight C++ profiling is available behind a compile-time flag:

- Build with profiling: `-C cmake.define.HF_ENABLE_PROFILING=ON`.
- Auto-dump at process end: `export HF_PROFILE=1`.
- Manual dump: `import cpp_hf; cpp_hf.prof_dump()`.

Note: `prof_dump` is exported by the C++ extension module.

## Development

- Editable install for local iteration:
  - `pip install -e . -C cmake.build-type=Release`
- Convenience script that prefers local/user prefixes and skips build isolation:
  - `./build_install.sh`
- Apple Silicon: see `build_fftw_macos_arm.sh` for a step-by-step FFTW build to `$HOME/.local`.

## Troubleshooting

- FFTW not found:
  - Set `PKG_CONFIG_PATH` and `CMAKE_PREFIX_PATH` to your FFTW install prefix.
  - On Homebrew, use `brew --prefix fftw` to discover the prefix and add its `lib/pkgconfig`.

- OpenMP linkage errors on macOS:
  - `brew install libomp` and ensure Clang can find it, or disable OpenMP via `-C cmake.define.HF_USE_OPENMP=OFF`.

- Excess threads / poor scaling:
  - Limit OpenMP threads: `OMP_NUM_THREADS=1`, or disable FFTW threads as above.

- Older compilers/standard libraries:
  - Prefer prebuilt wheels where possible. For source builds, ensure a C++23-capable toolchain; use mdspan backport if `<mdspan>` is missing.

