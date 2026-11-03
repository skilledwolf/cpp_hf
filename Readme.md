# hartreefock: Hartree-Fock on 2D k-meshes

`hartreefock` solves the finite-temperature Hartree-Fock problem on uniform 2D
k-meshes. One kernel object describes the problem, and jitted JAX solvers
(differentiable, GPU-capable) solve it. On the CPU the batched eigensolvers,
complex matrix products and the superlattice exchange run as OpenMP C++
kernels (FFTW + Eigen) through `jax.ffi`. Importing hartreefock turns on
`jax_enable_x64`; a solve runs in the dtype of the kernel's arrays.

A `Mesh` holds the k-points and their weights. One `HartreeFockKernel(mesh,
h, V, T=...)` covers dense meshgrid problems (FFT exchange, optional Hartree
and contact terms, reference-density embedding) and, with `plane_waves=`,
moiré plane-wave bases with streamed exchange and `potential_at` for
off-mesh self-energies. `V` is a callable of |q| or a table on the mesh,
scalar or orbital-resolved.

Solvers:

* `solve(kernel, P0, n, config=...)`: Roothaan SCF with the
  free-energy-safeguarded full-residual `hf.RDIIS()` (the default) or
  `hf.Linear()` mixing, or direct minimization by Riemannian `hf.CG()` or
  trust-region `hf.Newton()`; each config takes only its method's options.
* `solve_minimum`: RDIIS plus a local stability check and bounded saddle
  recovery; `stability_analysis` is the standalone curvature test.
* `solve(..., coarse=kernel)` (coarse to fine) and `solve_deflated`
  (several distinct minima).

## Documentation

The Sphinx site in `docs/` has install notes, a quickstart, guides to the
kernel, the solvers, superlattices, precision and speed, and the full API
reference. It is published at https://skilledwolf.github.io/cpp_hf/ from
`main`; build it locally with `pip install -e ".[docs]"` and
`sphinx-build -W -b html docs docs/_build/html`.

## Install

Requires JAX, FFTW (double precision: `fftw3`), Eigen and a C++17 compiler.

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
import hartreefock as hf

mesh = hf.Mesh.square(kmax=1.0, nk=32)
# h: (32, 32, D, D) Hamiltonian at mesh.ks
kernel = hf.HartreeFockKernel(mesh, h, lambda q: 2 * np.pi / (q + 0.1), T=0.1)
P0 = np.zeros_like(h)

r = hf.solve(kernel, P0, n_electrons)  # RDIIS
dm = hf.solve(kernel, P0, n_electrons, config=hf.CG(tol_E=1e-8))
# r.density, r.fock, r.energy, r.mu, r.n_iter, r.converged, r.history
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
compare free energies across seeds for phase comparisons. SCF histories
report `free_energy` and the total `fock_builds`.

Coarse-to-fine continuation solves the coarse kernel first and starts the
fine solve from its resampled density:

```python
result = hf.solve(fine, P0, n_electrons, config=hf.CG(tol_E=1e-8), coarse=coarse)
```

Both kernels expose the interaction as NumPy linear maps, `kernel.exchange(rho)`,
`kernel.hartree(rho)` and `kernel.potential(rho)`; the solvers' Fock matrix is
`h + kernel.potential(P - refP)`. `hartreefock._jax` is private.

Energies are the variational functional
`E_var = Tr(h P) + 1/2 Tr(V_HF[P - P_ref] (P - P_ref))`. For an embedded
kernel, `E_standard = E - kernel.embedded_energy_offset +
kernel.embedded_hartree_shift * n_electrons` and
`mu_standard = mu + kernel.embedded_hartree_shift`.

## Tests

```bash
python -m pytest -n 4 tests --deselect tests/test_bilayer_regression.py
```

`test_bilayer_regression.py` (about a minute on 4 cores; needs contimod) pins
energies against `tests/data/bilayer_reference.npz`.

## License

MIT, see `LICENSE`.

The extension links FFTW, which is GPLv2+. Binary wheels that bundle FFTW
are therefore distributed under the GPL as a whole; the source code alone is
MIT.

## Acknowledgments

- FFTW — GPLv2+ (www.fftw.org)
- Eigen — MPL2
- pybind11 — BSD‑style
