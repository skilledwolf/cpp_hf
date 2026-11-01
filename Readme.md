# hartreefock: Hartree-Fock on 2D k-meshes

`hartreefock` solves the finite-temperature Hartree-Fock problem on uniform 2D
k-meshes. One kernel object describes the problem, and two interchangeable
backends solve it:

* `backend="native"` (default): a pybind11 extension on FFTW + Eigen, double
  precision, GIL released for every solver call.
* `backend="jax"` (extra `hartreefock[jax]`): jitted JAX solvers,
  differentiable and GPU-capable.

Two kernel types cover the problems:

* `HartreeFockKernel`: dense meshgrid problems (FFT exchange, optional Hartree
  and contact terms, reference-density embedding).
* `SuperlatticeHartreeFockKernel`: moiré plane-wave bases (`g_basis`), with
  streamed exchange and scalar or orbital-resolved interactions, plus
  `potential_at` for off-mesh self-energies.

Solvers, all available on both backends unless noted:

* `solve`: direct minimization by Riemannian CG or trust-region Newton, or
  (native) `optimizer="auto"`, which hands CG over to Newton near convergence.
* `solve_scf`: Roothaan SCF with linear, DIIS or ODA mixing, or the
  free-energy-safeguarded full-residual `rdiis` (recommended at finite T) and
  `anderson` schemes; native adds `preconditioner="hartree"`.
* `solve_minimum`: RDIIS plus a local stability check and bounded saddle
  recovery; `stability_analysis` is the standalone curvature test.
* `solve_continuation` (coarse to fine) and `solve_deflated` (several distinct
  minima).

## Documentation

The Sphinx site in `docs/` has install notes, a quickstart, guides to the
kernel, the solvers, superlattices and the backends, and the full API
reference. It is published at https://skilledwolf.github.io/cpp_hf/ from
`main`; build it locally with `pip install -e ".[docs]"` and
`sphinx-build -W -b html docs docs/_build/html`.

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

A converged residual alone does not make a solution a minimum. `solve_minimum`
runs safeguarded RDIIS, tests the free-energy curvature, and tries finite
descent steps along an unstable mode (first through a fixed-particle Fermi
map of the mode's interaction response, then projected density steps); each
accepted step must lower the actual free energy:

```python
minimum = hf.solve_minimum(kernel, P0, n_electrons)
print(minimum.converged, minimum.solution.energy, minimum.escapes, minimum.fock_builds)
check = hf.stability_analysis(kernel, minimum.solution.density, n_electrons)
```

Both are local checks in the supplied basis, not a global-minimum certificate;
compare free energies across seeds for phase comparisons. SCF and `auto`
histories report `free_energy` and the total `fock_builds`.

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
`E_var = Tr(h P) + 1/2 Tr(V_HF[P - P_ref] (P - P_ref))`. For an embedded
kernel, `E_standard = E - kernel.embedded_energy_offset +
kernel.embedded_hartree_shift * n_electrons` and
`mu_standard = mu + kernel.embedded_hartree_shift`.

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
