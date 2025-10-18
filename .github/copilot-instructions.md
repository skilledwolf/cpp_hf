# Copilot Instructions for cpp_hf

## Project Overview

cpp_hf is a high-performance Hartree–Fock self-consistent field (SCF) solver for uniform 2D k-grids in condensed matter physics. The project provides a compiled Python extension module that exposes fast C++ implementations to Python.

**Key Technologies:**
- **Language:** C++23 with Python bindings
- **Build System:** CMake 3.21+ with scikit-build-core
- **Core Dependencies:**
  - FFTW3 (batched 2D complex DFTs via guru interface)
  - Eigen (Hermitian eigendecompositions)
  - pybind11 (Python bindings)
  - Boost (header-only utilities)
  - mdspan (for multi-dimensional array views)
- **Python:** 3.10+

## Architecture

### Source Structure
```
cpp_hf/
├── cpp_hf.cpp              # pybind11 module entry point
├── src/
│   └── hartreefock.cpp     # Main HF iteration implementation
├── include/cpp_hf/
│   ├── hartreefock.hpp     # HF algorithm headers
│   ├── fftw_batched2d.hpp  # FFTW wrapper for batched 2D FFTs
│   ├── fft_batched2d.hpp   # FFT abstractions
│   ├── mixers.hpp          # EDIIS/Broyden mixing algorithms
│   ├── hf_kernel.hpp       # Core HF computation kernels
│   ├── hf_mdspan.hpp       # mdspan utilities
│   ├── views.hpp           # Array view utilities
│   ├── utils.hpp           # General utilities
│   ├── prof.hpp            # Profiling utilities
│   └── platform.hpp        # Platform-specific macros
├── CMakeLists.txt          # CMake build configuration
├── pyproject.toml          # Python packaging & build config
└── ci/                     # CI scripts and smoke tests
```

### Key Components

1. **FFT Operations:** Batched 2D complex FFTs using FFTW guru interface for computing Coulomb interactions in reciprocal space
2. **Eigensolvers:** Dense Hermitian diagonalization via Eigen for solving Fock matrix at each k-point
3. **Convergence Acceleration:** EDIIS (Energy-based DIIS) and Broyden mixing for robust SCF convergence
4. **Python Interface:** Single function `hartreefock_iteration_cpp` exposed via pybind11

## Build Instructions

### Quick Build (Local Development)
```bash
# Using convenience script (activates conda env if HF_CONDA_ENV is set)
sh build_install.sh

# Or manually with pip
pip install . --no-build-isolation \
  -C cmake.build-type=Release \
  -C cmake.define.HF_USE_OPENMP=ON \
  -C cmake.define.HF_USE_FFTW_THREADS=ON
```

### Platform-Specific Setup

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install libfftw3-dev cmake ninja-build pkg-config libboost-dev
pip install .
```

**macOS (Homebrew):**
```bash
brew install fftw eigen cmake ninja libomp boost
pip install .
```

**macOS ARM (Apple Silicon):**
```bash
# Build FFTW from source (cached in ~/.local)
bash ci/macos_arm_fftw.sh
# Set environment variables for local FFTW
export PKG_CONFIG_PATH="$HOME/.local/lib/pkgconfig:$PKG_CONFIG_PATH"
export CMAKE_PREFIX_PATH="$HOME/.local:$CMAKE_PREFIX_PATH"
pip install .
```

### CMake Build Options

Configure via `-C cmake.define.<OPTION>=<VALUE>`:
- `HF_USE_OPENMP`: Enable OpenMP parallelization (default: ON)
- `HF_USE_FFTW_THREADS`: Link FFTW threads variant (default: ON)
- `HF_AGGRESSIVE_OPT`: Enable `-march=native`, `-Ofast`, etc. (default: OFF, for portability)
- `HF_LTO`: Enable link-time optimization (default: OFF)
- `HF_FETCH_FFTW`: Auto-download & build FFTW if not found (default: ON)

## Testing

### Smoke Test
```bash
# After installation
python ci/smoke_test.py
```

The smoke test runs a minimal 4×4 k-grid HF calculation to verify:
- Module can be imported
- Core function executes without errors
- Returns expected tuple structure

### Manual Testing
```python
import numpy as np
import cpp_hf

nk, d = 128, 2  # 128×128 k-grid, 2 bands
weights = np.ones((nk, nk)) * ((2/nk)*(2/nk) / (2*np.pi)**2)
H = np.zeros((nk, nk, d, d), dtype=np.complex128)
K = np.linspace(-1, 1, nk)
Vq = (1.0/np.sqrt((K[:,None]**2 + K[None,:]**2) + 0.1)).astype(np.complex128)[..., None, None]
P0 = np.zeros_like(H)
ne_target = 0.5 * d * weights.sum()

P, F, E, mu, n_iter = cpp_hf.hartreefock_iteration_cpp(
    weights, H, Vq, P0, ne_target,
    0.5,    # temperature
    50,     # max_iter
    1e-3,   # commutator tolerance
    6,      # mixing history size
    1.0     # mixing alpha
)
print(f"Converged in {n_iter} iterations, μ={mu:.6f}, E={E:.6f}")
```

## Coding Conventions

### C++ Style
- **Standard:** C++23 (required for std::mdspan when available)
- **Headers:** Use `#pragma once`
- **Namespaces:** No global namespace pollution; prefer unnamed namespaces for internal helpers
- **Templates:** Heavy use of templates for array views and compile-time optimization
- **Error Handling:** Use exceptions for Python binding; propagate via pybind11
- **Memory:** Eigen matrices and FFTW-managed buffers; no manual memory management
- **Formatting:** Keep consistent with existing code (no auto-formatter configured)

### Key Patterns
- **mdspan:** Use `std::mdspan` or vendor fallback for multi-dimensional array views
- **FFTW Planning:** Plans are created with `FFTW_ESTIMATE` for fast setup (repeated solves use same plans)
- **OpenMP:** k-point parallelism via `#pragma omp parallel for` when `HF_USE_OPENMP=ON`
- **NumPy Interop:** All array parameters expect C-contiguous NumPy arrays; shapes validated at runtime

### What NOT to Change
- **API Signature:** The `hartreefock_iteration_cpp` function signature is stable; changes break downstream users
- **Build Requirements:** Minimum CMake 3.21, Python 3.10, C++23 (C++17 with vendor mdspan)
- **Licenses:** GPLv2+ (main code), respect third-party licenses (see THIRD_PARTY_NOTICES.md)

## CI/CD

### Workflows (`.github/workflows/ci-and-release.yml`)

**On PR/Push:**
- `build-test`: Builds wheels and runs smoke tests on Linux, macOS Intel, macOS ARM for Python 3.10-3.12

**On Tag (`v*`):**
- `wheels`: Builds release wheels via cibuildwheel for multiple platforms
- `sdist`: Creates source distribution
- `publish`: Publishes to PyPI using trusted publishing

### Cibuildwheel Configuration
- **Targets:** CPython 3.10, 3.11, 3.12 (no PyPy, no musllinux)
- **Linux:** manylinux_2_28 for x86_64 and aarch64
- **macOS:** Native arch builds (Intel on Intel runners, ARM on ARM runners)
- **Smoke Test:** Runs `ci/smoke_test.py` on every built wheel

## Common Development Tasks

### Adding a New C++ Function
1. Declare in appropriate header under `include/cpp_hf/`
2. Implement in `src/` or inline if template
3. Bind in `cpp_hf.cpp` using pybind11 syntax
4. Add to module exports with docstrings
5. Update smoke test if needed

### Modifying the HF Algorithm
1. Edit `src/hartreefock.cpp` or related headers
2. Rebuild: `pip install . --no-build-isolation --force-reinstall`
3. Test with `ci/smoke_test.py` and manual validation
4. Ensure convergence behavior is preserved

### Updating Dependencies
- **FFTW:** Modify version in `CMakeLists.txt` `HF_FFTW_VERSION` and update SHA256
- **Eigen/Boost/pybind11:** Update FetchContent version/tag in `CMakeLists.txt`
- **Python packages:** Update `pyproject.toml` `requires` or `dependencies`

### Performance Optimization
- Most critical: FFT batching, Eigen eigensolve, mixing overhead
- Profile with built-in `prof.hpp` utilities or external tools (gprof, perf)
- Aggressive optimizations: `-DHF_AGGRESSIVE_OPT=ON -DHF_LTO=ON` (local builds only)
- Verify correctness before and after optimizations

## Troubleshooting

### FFTW Not Found
- **Linux:** `sudo apt-get install libfftw3-dev`
- **macOS:** `brew install fftw` or `bash ci/macos_arm_fftw.sh` for ARM
- **Auto-build:** Ensure `HF_FETCH_FFTW=ON` (default) and internet access

### Import Errors (macOS)
- Wheel may not find FFTW/OpenMP dylibs; set `DYLD_FALLBACK_LIBRARY_PATH`:
  ```bash
  export DYLD_FALLBACK_LIBRARY_PATH="$HOME/.local/lib:/opt/homebrew/lib:/usr/local/lib"
  ```

### Build Failures
- **C++23 not supported:** Fallback to vendor mdspan (automatic via CMake)
- **Missing boost:** Install or rely on FetchContent (requires internet)
- **OpenMP not found:** Disable with `-DHF_USE_OPENMP=OFF`

## Additional Resources

- **PyPI:** https://pypi.org/project/cpp-hf/
- **Repository:** https://github.com/skilledwolf/cpp_hf
- **Issues:** https://github.com/skilledwolf/cpp_hf/issues
- **License:** GPLv2+ (see LICENSE)
- **Related:** Designed to interoperate with ContiMod (separate project)
