# cpp_hf — Hartree–Fock on k‑grids (C++ reimplementation of jax_hf)

`cpp_hf` is a C++ reimplementation of [`jax_hf`](https://github.com/skilledwolf/jax_hf):
two Hartree–Fock solvers on uniform 2D k‑meshes that share the same kernel,
the same public API, and pass the same regression tests, with the JAX
dependency replaced by a single pybind11 extension built on FFTW + Eigen.

* **Direct minimisation** (default) — preconditioned Riemannian CG on
  Stiefel × capped simplex, eigen‑free inner loop, Cayley retraction,
  one Fock build per iteration.
* **Reference SCF** (baseline / fallback) — standard Roothaan iteration
  with linear mixing.

Both solvers run in **double precision** throughout (complex128 / float64).

## Install

Requires FFTW (double precision: `fftw3`) and a C++17 compiler.

```bash
# macOS
brew install fftw eigen
pip install -e .

# Linux
sudo apt install libfftw3-dev libeigen3-dev
pip install -e .
```

## Minimal example

```python
import numpy as np
import cpp_hf

# Build a HartreeFockKernel: precomputes the FFT of the interaction kernel,
# the Hartree matrix, etc., ready for the solver.
kernel = cpp_hf.HartreeFockKernel(
    weights=weights,          # (nk1, nk2) k-point weights
    hamiltonian=hamiltonian,  # (nk1, nk2, nb, nb) single-particle Hamiltonian
    coulomb_q=coulomb_q,      # (nk1, nk2, 1, 1) scalar or (nk1, nk2, nb, nb) layer-resolved
    T=0.1,
    include_hartree=False,    # set True for Hartree; also pass reference_density + hartree_matrix
    include_exchange=True,
)

# Solve (direct minimisation, default)
result = cpp_hf.solve(kernel, P0=np.zeros_like(hamiltonian), n_electrons=N)
print(result.energy, result.converged, result.n_iter)
# result.density, result.fock, result.Q, result.p, result.mu, result.history

# Or use SCF as a fallback baseline
result_scf = cpp_hf.solve_scf(kernel, P0=np.zeros_like(hamiltonian), n_electrons=N)
```

## Architecture

| Layer | Where it lives | What it does |
|---|---|---|
| **C++ core** | `cpp/include/cpp_hf/*.hpp` + `cpp/cpp_hf_native.cpp` | FFT‑based exchange (`selfenergy_fft`), batched Hermitian eigendecomposition, contact‑term Fock construction, SCF main loop, direct‑minimisation main loop (preconditioned Riemannian CG with spectral Cayley line search), k‑grid resampling — all in double precision. |
| **Native extension** | `cpp_hf._native` (compiled `.so` inside the package) | pybind11 wrapper exposing the C++ entry points; the GIL is released for the duration of every solver call. |
| **Python surface** | `src/cpp_hf/*.py` | Public dataclasses (`SolverConfig`, `SCFConfig`, `SolveResult`, `SCFResult`, `ContinuationResult`), the `HartreeFockKernel` constructor (input validation + kernel precomputation), the symmetry projector framework (passed to the C++ solver as a Python callback), and the continuation driver (composes two C++ solver calls + `resample_kgrid`). |

The Python surface is intentionally thin: it validates inputs, packs them
into the dict shape the C++ binding expects, then hands off.  Every
production code path runs in C++; importing the package without the
compiled extension raises a clear `RuntimeError` from `native_required()`.

## Public API

| Name | Purpose |
|---|---|
| `HartreeFockKernel` | Problem definition + precomputed arrays |
| `solve` (alias `solve_direct_minimization`), `SolverConfig`, `SolveResult` | Primary solver |
| `solve_scf`, `SCFConfig`, `SCFResult` | Reference SCF solver |
| `build_fock`, `hf_energy`, `free_energy`, `occupation_entropy` | HF objective building blocks |
| `solve_continuation`, `ContinuationResult`, `resample_kgrid` | Coarse → fine multigrid driver + k‑grid resampler |
| `cpp_hf.symmetry.make_project_fn` | Symmetry projector builder (unitary / spatial / time‑reversal) |
| `cpp_hf.linalg.eigh` | Block‑diagonal Hermitian eigh with optional structure check |

The API mirrors `jax_hf` exactly so that scripts written against `jax_hf`
work against `cpp_hf` by changing only the import line.

## Coarse → fine continuation

```python
from cpp_hf import HartreeFockKernel, SolverConfig, SCFConfig, solve_continuation

coarse = HartreeFockKernel(weights_c, h_c, Vq_c, T=0.1)
fine   = HartreeFockKernel(weights_f, h_f, Vq_f, T=0.1)

result = solve_continuation(
    coarse, fine, P0_coarse=np.zeros_like(h_c),
    n_electrons_coarse=N, n_electrons_fine=N,
    coarse_config=SCFConfig(max_iter=50, mixing=0.5),
    fine_config=SolverConfig(max_iter=200, tol_E=1e-8),
)
# result.coarse, result.fine, result.P0_fine
```

## Tests

```bash
pip install -e .
python -m pytest tests/
```

The suite is a port of `jax_hf/tests/`; it covers the chemical‑potential
solver, block‑diagonal eigh and self‑energy, the symmetry projectors,
direct minimisation (basic convergence, multi‑k, contact terms, edge
cases, Cayley spectral identities), SCF, continuation, and package‑import
contracts.  The bilayer regression test (`test_bilayer_regression.py`,
ported from `jax_hf`) is not included by default since it requires
`contimod`.

## License

GPLv2+ — see `LICENSE`.

## Acknowledgments

- FFTW — GPLv2+ (www.fftw.org)
- Eigen — MPL2
- pybind11 — BSD‑style
