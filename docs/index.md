---
file_format: mystnb
kernelspec:
  name: python3
---

# hartreefock

hartreefock solves the finite-temperature Hartree-Fock problem on uniform 2D
k-meshes. You describe the problem once as a kernel (a k-mesh, the
single-particle Hamiltonian on it and the interaction) and solve it with
jitted JAX solvers, which call C++ kernels for the hot loops on the CPU. It is the engine behind the mean-field solvers of
[contimod](https://github.com/skilledwolf/contimod), and works on any
Hamiltonian you can put on a k-mesh.

Here is the exchange self-energy of the two-dimensional electron gas at
$r_s = 1$, against its closed form in elliptic integrals
{cite}`Giuliani2005` ([tutorial step 1](tutorial/01-exchange.md)). The code
is under the figure.

```{code-cell} ipython3
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipe, ellipk
import hartreefock as hf

nk, kmax, rs = 64, 2.1, 1.0                      # units of k_F and E_F
mesh = hf.Mesh.square(kmax, nk)
eps = np.sum(mesh.ks**2, axis=-1)
h = eps[..., None, None] * np.eye(2)
dk = mesh.dk[0]


def V(q):  # bare Coulomb; at q = 0 its average over one mesh cell
    return 2 * np.pi * np.sqrt(2) * rs * np.where(q > 0, 1 / np.where(q > 0, q, 1), 4 * np.log(1 + np.sqrt(2)) / dk)


kernel = hf.HartreeFockKernel(mesh, h, V, T=0.002)
result = hf.solve(kernel, None, 1 / (2 * np.pi))
sigma = (np.asarray(result.fock) - h)[:, nk // 2, 0, 0].real

kk = np.linspace(0, 2.1, 400)
x = np.where(kk > 1, 1 / np.where(kk > 1, kk, 1), 0.5)
exact = -2 * np.sqrt(2) * rs / np.pi * np.where(
    kk < 1, ellipe(np.minimum(kk, 1) ** 2), kk * (ellipe(x**2) - (1 - x**2) * ellipk(x**2)))

fig, ax = plt.subplots()
ax.plot(np.abs(mesh.kx), sigma, "o", ms=3, label="hartreefock, 64 × 64 mesh")
ax.plot(kk, exact, color="C1", label="closed form")
ax.set(xlabel="$k / k_F$", ylabel=r"$\Sigma_x / E_F$", xlim=(0, 1.8))
ax.legend();
```

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Get started
:link: get-started/quickstart
:link-type: doc

Install it, then find the spin-polarized ground state of a gated electron gas.
:::

:::{grid-item-card} Tutorial
:link: tutorial/index
:link-type: doc

Three steps on the electron gas: exchange, Bloch ferromagnetism, the
Stoner instability. Each checks its result against a closed form.
:::

:::{grid-item-card} Guides
:link: guides/index
:link-type: doc

The kernel, the solvers, superlattices, and precision and speed, each with
a check.
:::

:::{grid-item-card} Python API
:link: reference/index
:link-type: doc

The functions you call, grouped by task, then every public name by module.
:::
::::

```{toctree}
:hidden:

get-started/index
tutorial/index
guides/index
reference/index
development/index
project/index
```
