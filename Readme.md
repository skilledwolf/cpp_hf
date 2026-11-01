# hartreefock: Hartree-Fock on 2D k-meshes

`hartreefock` solves the finite-temperature Hartree-Fock problem on uniform 2D
k-meshes. One kernel object describes the problem, and two interchangeable
backends solve it:

* `backend="native"` (default): a pybind11 extension on FFTW + Eigen, double
  precision, GIL released for every solver call.
* `backend="jax"` (extra `hartreefock[jax]`): jitted JAX solvers. It adds the
  RDIIS / Anderson SCF schemes, and JAX powers `hf.stability_analysis` and
  `hf.solve_minimum`.

Two kernel types cover the problems:

* `HartreeFockKernel`: dense meshgrid problems (FFT exchange, optional Hartree
  and contact terms, reference-density embedding).
* `SuperlatticeHartreeFockKernel`: moiré plane-wave bases (`g_basis`), with
  streamed exchange and scalar or orbital-resolved interactions, plus
  `potential_at` for off-mesh self-energies.

Solvers: direct minimization (`solve`, Riemannian CG or trust-region Newton),
Roothaan SCF (`solve_scf`, linear / DIIS / ODA), coarse-to-fine
`solve_continuation`, and `solve_deflated` for several distinct minima.

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

For the JAX backend: `pip install -e ".[jax]"`.

## Minimal example

```python
import numpy as np
import hartreefock as hf

kernel = hf.HartreeFockKernel(weights, h, Vq, T=0.1)
P0 = np.zeros_like(h)

dm = hf.solve(kernel, P0, n_electrons, config=hf.SolverConfig(tol_E=1e-8))
scf = hf.solve_scf(kernel, P0, n_electrons, config=hf.SCFConfig(acceleration="diis"), backend="jax")
# dm.density, dm.fock, dm.energy, dm.mu, dm.n_iter, dm.converged, dm.history
```

Coarse-to-fine continuation runs `solve_scf` for stages configured with an
`SCFConfig` and `solve` otherwise:

```python
result = hf.solve_continuation(
    coarse, fine, P0_coarse, n_electrons_coarse, n_electrons_fine,
    coarse_config=hf.SCFConfig(max_iter=50, mixing=0.5),
    fine_config=hf.SolverConfig(max_iter=200, tol_E=1e-8),
)
```

Both kernels expose the interaction as NumPy linear maps, `kernel.exchange(rho)`,
`kernel.hartree(rho)` and `kernel.potential(rho)`; the solvers' Fock matrix is
`h + kernel.potential(P - refP)`. `hartreefock._backends` is private.

Energies are the variational functional
`E_var = Tr(h P) + 1/2 Tr(V_HF[P - P_ref] (P - P_ref))`; an embedded kernel
reports `kernel.embedded_energy_offset`, which callers subtract to get the
standard HF energy.

## Tests

```bash
python -m pytest -n 4 tests --deselect tests/test_bilayer_regression.py
```

`tests/` holds the backend-independent checks (superlattice, continuation,
symmetry, backend parity); `tests/native` and `tests/jax` hold the
backend-specific ones. `test_bilayer_regression.py` (about 12 minutes) pins
energies against `tests/data/bilayer_reference.npz`.

## License

MIT, see `LICENSE`.

The native backend links FFTW, which is GPLv2+. Binary wheels that bundle
FFTW are therefore distributed under the GPL as a whole; the source code and
the JAX backend alone are MIT.

## Acknowledgments

- FFTW — GPLv2+ (www.fftw.org)
- Eigen — MPL2
- pybind11 — BSD‑style
