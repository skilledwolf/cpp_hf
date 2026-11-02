---
file_format: mystnb
kernelspec:
  name: python3
---

# 1. Exchange in the electron gas

This page builds the first problem hartreefock solves: the
two-dimensional electron gas with the Coulomb interaction. It solves it,
compares the exchange self-energy with its closed form, and converges the
exchange energy to five digits by extrapolating in the mesh.

:::{admonition} Physics
In Hartree-Fock the plane waves of the uniform gas are exact
self-consistent orbitals: the Hartree term cancels against the positive
background, and only exchange is left {cite}`Giuliani2005`. With energies
in units of the Fermi energy $E_F$ and momenta in units of $k_F$, the
dispersion is $k^2$ and the Coulomb interaction is

$$
V(q) = \frac{2\pi\sqrt2\, r_s}{q},
$$

where $r_s$ is the mean electron spacing in Bohr radii. The exchange
self-energy of a filled Fermi disc is

$$
\Sigma_x(k) = -\frac{2\sqrt2\, r_s}{\pi}\times
\begin{cases} \mathrm E(k), & k < 1,\\
k\big[\mathrm E(1/k) - (1 - 1/k^2)\,\mathrm K(1/k)\big], & k > 1,\end{cases}
$$

with $\mathrm K$ and $\mathrm E$ the complete elliptic integrals, and the
exchange energy per electron is $-8\sqrt2/(3\pi r_s)$ Ry $= -1.2004/r_s$ Ry.

**You will use:** {class}`~hartreefock.Mesh`, {class}`~hartreefock.HartreeFockKernel`,
{func}`~hartreefock.solve`, {class}`~hartreefock.SolverConfig`.
:::

## The mesh, the Hamiltonian and the interaction

Everything lives on an `nk × nk` {class}`~hartreefock.Mesh`. The
Hamiltonian `h` has shape `(nk, nk, nb, nb)`, here two bands for the two
spins. The mesh weights make a k-sum an integral,
$\sum_k w_k f(\mathbf k) = \int d^2k/(2\pi)^2 f(\mathbf k)$, and the
particle number is counted with them.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipe, ellipk
import hartreefock as hf


def electron_gas(nk, kmax, rs):
    """The 2D electron gas on an nk x nk mesh over [-kmax, kmax]², in units of k_F and E_F."""
    mesh = hf.Mesh.square(kmax, nk)
    eps = np.sum(mesh.ks**2, axis=-1)
    dk = mesh.dk[0]

    def V(q):  # 2π e²/q; at q = 0 the average of 1/q over a cell, 4 ln(1 + √2) / dk
        return 2 * np.pi * np.sqrt(2) * rs * np.where(q > 0, 1 / np.where(q > 0, q, 1), 4 * np.log(1 + np.sqrt(2)) / dk)

    return mesh, eps[..., None, None] * np.eye(2), V, eps


n = 1 / (2 * np.pi)            # k_F = 1 with two spins: n = k_F² / 2π
mesh, h, V, eps = electron_gas(nk=96, kmax=2.1, rs=1.0)
print(h.shape, mesh.transfer_distance_grid.shape)
```

The kernel samples `V(|q|)` on the momentum transfers of the mesh,
`mesh.transfer_grid`, with the spacing of the k-mesh and `q = 0` at index
`nk // 2`. A scalar `V` is one interaction for all orbitals. The
exchange term is a convolution over the mesh, so transfers wrap around:
the mesh must reach beyond $2k_F$ (here `kmax = 2.1`) for the transfers
between occupied states to be represented.

## Solving

{class}`~hartreefock.HartreeFockKernel` holds what does not change during
a solve; by default it keeps exchange only. The start is the filled Fermi
disc:

```{code-cell} ipython3
kernel = hf.HartreeFockKernel(mesh, h, V, T=0.002)
P0 = (eps < 1)[..., None, None] * np.eye(2)
result = hf.solve(kernel, P0, n)
print(result.converged, result.n_iter, "iterations,  mu =", round(float(result.mu), 4))
```

The Fock matrix minus `h` is the self-energy. Along $k_x$:

```{code-cell} ipython3
def sigma_exact(k, rs):
    k = np.abs(k)
    inside = ellipe(np.minimum(k, 1) ** 2)
    x = np.where(k > 1, 1 / np.where(k > 1, k, 1), 0.5)
    outside = k * (ellipe(x**2) - (1 - x**2) * ellipk(x**2))
    return -2 * np.sqrt(2) * rs / np.pi * np.where(k < 1, inside, outside)


sigma = (np.asarray(result.fock) - h)[:, mesh.center_index, 0, 0].real

fig, ax = plt.subplots()
ax.plot(mesh.kx, sigma, "o", ms=3, label="hartreefock")
kk = np.linspace(-2.1, 2.1, 801)
ax.plot(kk, sigma_exact(kk, 1.0), color="C1", label="closed form")
ax.set(xlabel="$k_x / k_F$", ylabel=r"$\Sigma_x / E_F$", title="$r_s = 1$")
ax.legend();
```

The self-energy is deepest at $k = 0$, where it equals $-\sqrt2\,r_s$, and
has a logarithmic kink at $k_F$, the reason the Hartree-Fock Fermi velocity
of the electron gas diverges. Near the edge of the mesh the computed values
drift from the closed form: there the transfers to the occupied disc wrap
around the periodic transfer grid. Push `kmax` out to move the drift away
from the states you need.

## The energy and the mesh

`result.energy` is the free energy per unit area, $\mathrm{Tr}(hP) + \tfrac12
\mathrm{Tr}(\Sigma P)$, in units of $E_F k_F^2$. Per electron and in
Rydberg, at $r_s = 1$ where $E_F = 2$ Ry:

```{code-cell} ipython3
def energies(nk, rs=1.0):
    mesh, h, V, eps = electron_gas(nk, kmax=2.1, rs=rs)
    kernel = hf.HartreeFockKernel(mesh, h, V, T=0.002)
    P0 = (eps < 1)[..., None, None] * np.eye(2)
    r = hf.solve(kernel, P0, n)
    parts = kernel.energies(r.density)
    to_ry = 2 / rs**2 / n        # E_F k_F² per area -> Ry per electron
    return parts["kinetic"] * to_ry, parts["exchange"] * to_ry


meshes = np.array([32, 48, 64, 96, 128])
kinetic, exchange = np.array([energies(nk) for nk in meshes]).T
for nk, t, x in zip(meshes, kinetic, exchange):
    print(f"nk = {nk:3d}:  kinetic {t:.5f}   exchange {x:.5f} Ry")
```

The kinetic energy is $1/r_s^2$ Ry on every mesh. The exchange energy
converges as $1/n_k$: the error comes from the cells around $q = 0$, where
$1/q$ is not smooth. A straight-line fit in $1/n_k$ removes it.

```{code-cell} ipython3
slope, extrapolated = np.polyfit(1 / meshes, exchange, 1)
exact = -8 * np.sqrt(2) / (3 * np.pi)

fig, ax = plt.subplots()
ax.plot(1 / meshes, exchange, "o", label="hartreefock")
x = np.linspace(0, 1 / meshes[0], 2)
ax.plot(x, extrapolated + slope * x, "--", color="C0", lw=1, label="linear fit")
ax.plot(0, exact, "*", ms=12, color="C1", label=r"$-8\sqrt{2}/3\pi$")
ax.set(xlabel="$1/n_k$", ylabel="exchange energy (Ry per electron)", xlim=(-0.002, None))
ax.legend();
```

## Check

```{code-cell} ipython3
print(f"extrapolated exchange energy: {extrapolated:.5f} Ry")
print(f"exact:                        {exact:.5f} Ry   (-8√2 / 3π)")
print(f"relative error:               {abs(extrapolated / exact - 1):.1e}")
inner = (np.abs(mesh.kx) < 1.5) & (np.abs(np.abs(mesh.kx) - 1) > 0.1)
dev = np.abs(sigma - sigma_exact(mesh.kx, 1.0))[inner].max()
print(f"largest Σ_x error for |k| < 1.5 k_F, away from k_F: {dev:.4f} E_F")
```

```{code-cell} ipython3
:tags: [remove-cell]
assert abs(extrapolated / exact - 1) < 5e-4
assert np.allclose(kinetic, 1.0, atol=5e-3)
assert dev < 0.01
```

## Next

The exchange energy grows as $1/r_s$ and the kinetic energy only as
$1/r_s^2$, so at low density the gas can gain by polarizing its spin.
[Step 2](02-bloch.md) finds where.
