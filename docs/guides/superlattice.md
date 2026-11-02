---
file_format: mystnb
kernelspec:
  name: python3
---

# Superlattices

{class}`~hartreefock.SuperlatticeHartreeFockKernel` handles moiré problems
in a plane-wave basis $|\mathbf k, \mathbf G, \alpha\rangle$: $\mathbf k$ runs
over a mesh of the superlattice Brillouin zone, $\mathbf G$ over a set of
its reciprocal-lattice vectors and $\alpha$ over orbitals, so the matrix
dimension is `D = n_G · dim_orb` with the `G` index slow. The solvers are
the same as for the dense kernel.

## Theory

The state $|\mathbf k, \mathbf G\rangle$ carries the absolute momentum
$\mathbf p = \mathbf k - \mathbf G$, and the exchange self-energy is

$$
\Sigma_F[(\mathbf G_a\alpha),(\mathbf G_b\beta)](\mathbf k) =
-\sum_{\mathbf q,\mathbf G_0} w\, V(|\mathbf k - \mathbf q + \mathbf G_0|)\,
\rho[(\mathbf G_a + \mathbf G_0)\alpha, (\mathbf G_b + \mathbf G_0)\beta](\mathbf q),
$$

one aperiodic convolution in $\mathbf p$ per channel
$(\Delta\mathbf G = \mathbf G_a - \mathbf G_b, \alpha, \beta)$, computed with
an FFT on a zero-padded grid {cite}`Bistritzer2011`. Unlike the dense
kernel's periodic convolution, it does not wrap around. The Hartree term
couples the $\Delta\mathbf G$ components of the zone-averaged density and is
the same at every $\mathbf k$.

**On this page:** {class}`~hartreefock.SuperlatticeHartreeFockKernel`,
`potential_at`.

## Arguments

| Argument | Meaning |
| --- | --- |
| `weights` | uniform `(nkx, nky)` quadrature weights on the superlattice zone |
| `hamiltonian` | `(nkx, nky, D, D)`, the G index slow |
| `coulomb_V` | a callable `|q| -> V(|q|)`; the kernel tabulates it on the momentum transfers it needs, `q = 0` included |
| `g_fractional` | `(n_G, 2)` integers: the G-basis in units of the reciprocal lattice vectors |
| `reciprocal_basis` | `(2, 2)` matrix `B` with `G_cart = B @ G_frac` |
| `coulomb_V_orbital` | optional `|q| -> V_αβ(|q|)` for an orbital-resolved interaction, such as between layers |
| `hartree_matrix`, `hartree_matrix_orbital` | optional Hartree couplings between G components; by default taken from the interaction with `q = 0` dropped (scalar) or kept (orbital-resolved) |
| `hartree_degeneracy` | the flavour multiplicity of the Hartree term, for spin and valley left out of the basis |
| `k_origin` | fractional coordinates of the first mesh point, `(-0.5, -0.5)` by default; the mesh is `k_origin + i / nk` |

## The empty lattice

A superlattice with no potential is the electron gas folded into a small
zone. In Hartree-Fock the plane waves stay exact, so the self-energy of
each folded state is the closed form of [tutorial step 1](../tutorial/01-exchange.md)
at its absolute momentum. One spin, $k_F = 1$, on a square superlattice
with $|\mathbf b| = k_F$:

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipe, ellipk
import hartreefock as hf

nk, rs = 16, 1.0
B = np.eye(2)                                            # reciprocal lattice vectors, units of k_F
g_frac = np.array([(i, j) for i in range(-3, 4) for j in range(-3, 4) if i * i + j * j <= 5])
f = -0.5 + np.arange(nk) / nk                            # the default k_origin
k = np.stack(np.meshgrid(f, f, indexing="ij"), axis=-1) @ B.T
p = k[:, :, None, :] - (g_frac @ B.T)[None, None]        # absolute momentum of each basis state
eps = (p**2).sum(-1)
diag = np.arange(len(g_frac))
h = np.zeros((nk, nk, len(g_frac), len(g_frac)), complex)
h[:, :, diag, diag] = eps

dk = 1.0 / nk
weights = np.full((nk, nk), dk**2 / (2 * np.pi) ** 2)
e2 = np.sqrt(2) * rs


def coulomb(q):
    # 2π e²/q, with the cell average 4 ln(1 + √2)/dk at q = 0 as on the dense mesh
    return 2 * np.pi * e2 * np.where(q > 0, 1 / np.where(q > 0, q, 1), 4 * np.log(1 + np.sqrt(2)) / dk)


kernel = hf.SuperlatticeHartreeFockKernel(weights, h, coulomb, g_frac, B, T=0.002)
n = 1 / (4 * np.pi)                                      # one spin with k_F = 1
P0 = np.zeros_like(h)
P0[:, :, diag, diag] = eps < 1
r = hf.solve_scf(kernel, P0, n, config=hf.SCFConfig(acceleration="rdiis"))
print(f"{len(g_frac)} plane waves, D = {h.shape[-1]}: converged {r.converged} in {r.n_iter} iterations")
```

```{code-cell} ipython3
def sigma_exact(k, rs):
    k = np.abs(k)
    x = np.where(k > 1, 1 / np.where(k > 1, k, 1), 0.5)
    outside = k * (ellipe(x**2) - (1 - x**2) * ellipk(x**2))
    return -2 * np.sqrt(2) * rs / np.pi * np.where(k < 1, ellipe(np.minimum(k, 1) ** 2), outside)


sigma = np.asarray(r.fock) - h
p_abs = np.linalg.norm(p, axis=-1).ravel()
on_mesh = sigma[:, :, diag, diag].real.ravel()
off_diagonal = np.abs(sigma - np.einsum("ija,ab->ijab", sigma[:, :, diag, diag], np.eye(len(g_frac)))).max()

line = np.stack([np.linspace(-0.5, 0.5, 41), np.zeros(41)], axis=-1)         # fractional, off the mesh
g0 = int(np.flatnonzero((g_frac == 0).all(axis=1))[0])
off_mesh = np.asarray(kernel.potential_at(r.density, line))[:, g0, g0].real

fig, ax = plt.subplots()
ax.plot(p_abs, on_mesh, ".", ms=2, label="mesh, every G")
ax.plot(np.abs(line[:, 0]), off_mesh, "s", ms=3, mfc="none", label="potential_at, G = 0")
pp = np.linspace(0, 2.6, 400)
ax.plot(pp, sigma_exact(pp, rs), color="C1", label="closed form")
ax.set(xlabel="$|\\mathbf{k} - \\mathbf{G}| / k_F$", ylabel="$\\Sigma_x / E_F$")
ax.legend();
```

The folded states line up on the closed form as a function of their
absolute momentum, and the off-diagonal blocks between plane waves vanish,
as they must without a potential. The exchange partners of every state are
the occupied ones, $|\mathbf p'| < k_F$, which the basis holds, so even the
outermost shell is exact; the largest errors are inside the Fermi disc,
from the $q = 0$ cell, as on the dense mesh.

## Off-mesh self-energies

Matrix elements of $\Sigma(\mathbf k)$ in a truncated G-basis are not
periodic in $\mathbf k$, so they cannot be interpolated from the mesh.
`kernel.potential_at(P, k_frac)` evaluates $V_{HF}[P - P^{\mathrm{ref}}]$ at
any fractional momenta, such as a band path, with a direct plane-wave sum.
The squares above are such points; between mesh points the $q = 0$ cell
average is not used, and the values scatter by about $0.01\,E_F$.

## Check

```{code-cell} ipython3
inner = np.abs(p_abs - 1) > 0.1
dev_mesh = np.abs(on_mesh - sigma_exact(p_abs, rs))[inner].max()
dev_line = np.abs(off_mesh - sigma_exact(line[:, 0], rs)).max()
print(f"on the mesh, away from k_F:          largest error {dev_mesh:.4f} E_F")
print(f"potential_at along the line:        largest error {dev_line:.4f} E_F")
print(f"largest off-diagonal element:       {off_diagonal:.1e}")
```

```{code-cell} ipython3
:tags: [remove-cell]
assert dev_mesh < 0.01 and dev_line < 0.03 and off_diagonal < 1e-10
```

For a real moiré problem, contimod builds the kernel from a superlattice
model: see its
[superlattice guide](https://skilledwolf.github.io/contimod/guides/superlattices.html).
