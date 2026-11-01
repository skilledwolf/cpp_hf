# hartreefock

hartreefock solves the finite-temperature Hartree-Fock problem on uniform 2D
k-meshes. You describe the problem once as a kernel (the single-particle
Hamiltonian, the interaction and the quadrature weights) and solve it with one
of two interchangeable backends: a C++ extension on FFTW and Eigen, or jitted
JAX.

```python
import hartreefock as hf

kernel = hf.HartreeFockKernel(weights, h, coulomb_q, T=0.005)
result = hf.solve_scf(kernel, P0, n_electrons, config=hf.SCFConfig(acceleration="rdiis"))
check = hf.stability_analysis(kernel, result.density, n_electrons)
```

It is the engine behind the mean-field solvers of
[contimod](https://github.com/skilledwolf/contimod), and works on any
Hamiltonian you can put on a k-mesh.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Get started
:link: get-started/quickstart
:link-type: doc

Install it, then find the spin-polarized ground state of a dilute electron gas.
:::

:::{grid-item-card} The kernel
:link: guides/kernel
:link-type: doc

Hamiltonian, interaction, reference density, embedding, and what the energy means.
:::

:::{grid-item-card} Solvers
:link: guides/solvers
:link-type: doc

Direct minimization, SCF, stability, continuation and deflation: which to use when.
:::

:::{grid-item-card} Python API
:link: reference/python/index
:link-type: doc

Every public name, generated from the docstrings.
:::
::::

```{toctree}
:hidden:

get-started/index
guides/index
reference/python/index
development/index
```
