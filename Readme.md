# cpp_hf — Fast Hartree–Fock on k‑grids (FFTW + Eigen + pybind11)

cpp_hf provides a compiled extension module exposing a fast Hartree–Fock
self‑consistent field (SCF) loop for uniform 2D k‑grids. It uses
FFTW (batched 2D complex DFTs), Eigen (Hermitian eigendecompositions), and
pybind11 to bind the C++ implementation to Python.

- Batched 2D FFTs (FFTW guru interface)
- Dense Hermitian diagonalization (Eigen)
- EDIIS + Broyden mixing (robust convergence)
- Simple, NumPy‑friendly API

This package is designed to interoperate with ContiMod, but it can be used on
its own as a lightweight HF kernel.

## Installation

Prerequisites:
- Python 3.8+
- A C++17 compiler
- FFTW (with threads recommended) and headers
- CMake ≥ 3.21

Install from source (recommended during development):

```
# Optional: choose features via -C flags
pip install . \
  -C cmake.define.HF_USE_OPENMP=ON \
  -C cmake.define.HF_USE_FFTW_THREADS=ON
```

If you have Homebrew on macOS:
```
brew install fftw eigen cmake
```

The repo includes a convenience script that activates a conda env and builds:
```
sh build_install.sh
```

## Quick start

```python
import numpy as np
import cpp_hf

# Grid and shapes
nk = 128; d = 2
weights = np.ones((nk, nk)) * ( (2/nk)*(2/nk) / (2*np.pi)**2 )  # scalar mesh measure
H = np.zeros((nk, nk, d, d), dtype=np.complex128)
Vq = (1.0/np.sqrt((np.linspace(-1,1,nk)[:,None]**2 + np.linspace(-1,1,nk)[None,:]**2) + 0.1)).astype(np.complex128)[...,None,None]
P0 = np.zeros_like(H)

# Target electron density (half‑filling)
ne_target = 0.5 * d * weights.sum()

P_fin, F_fin, E_fin, mu_fin, n_iter = cpp_hf.hartreefock_iteration_cpp(
    weights, H, Vq, P0,
    float(ne_target),  # electron density target
    0.5,               # temperature
    50,                # max_iter
    1e-3,              # commutator tolerance
    6,                 # mixing history (DIIS/EDIIS/Broyden)
    1.0,               # mixing alpha (kept for compatibility)
)
print("iters:", n_iter, "mu:", mu_fin, "E:", E_fin)
```

## API

```python
hartreefock_iteration_cpp(
    weights:      np.ndarray[(nk,nk), float64],
    hamiltonian:  np.ndarray[(nk,nk,d,d), complex128],
    v_coulomb:    np.ndarray[(nk,nk,1,1) or (nk,nk,d,d), complex128],
    p0:           np.ndarray[(nk,nk,d,d), complex128],
    electron_density0: float,
    T: float,
    max_iter: int,
    comm_tol: float,
    diis_size: int,
    mixing_alpha: float,
) -> tuple[P, F, E, mu, n_iter]
```

- `weights` is the uniform k‑mesh measure (e.g., `dkx*dky/(2π)^2`).
- `v_coulomb` can be scalar per‑k (`[...,1,1]`) or a full matrix per‑k (`[...,d,d]`).
- Returns the converged density `P`, mean‑field `F`, total energy, chemical
  potential, and the iteration count.

## Building wheels / releasing to PyPI

We use scikit‑build‑core, so building wheels is standard:

```
python -m pip install build twine
python -m build  # creates dist/*.whl and dist/*.tar.gz
python -m twine upload dist/*
```

To test upload to TestPyPI first:
```
python -m twine upload --repository testpypi dist/*
# then install with
pip install -i https://test.pypi.org/simple/ cpp_hf
```

Note: the name `cpp_hf` must be available on PyPI. If it’s taken, consider a
unique name such as `contimod-cpp-hf` and update `pyproject.toml` accordingly.

## License

MIT — see `LICENSE`.

## Acknowledgments

- FFTW — MIT‑like license (www.fftw.org)
- Eigen — MPL2
- pybind11 — BSD‑style

