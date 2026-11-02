---
file_format: mystnb
kernelspec:
  name: python3
---

# The kernel

A kernel holds everything about the problem that does not change during a
solve: a {class}`~hartreefock.Mesh`, the Hamiltonian on it and an
interaction that depends on the momentum transfer. This page is the dense
case, one orbital basis at every k; `plane_waves=` turns the same
{class}`~hartreefock.HartreeFockKernel` into the moiré plane-wave case
([Superlattices](superlattice.md)).

## Theory

For a density matrix $P_k$ on the mesh, with quadrature weights $w_k$ and
a reference density $P^{\mathrm{ref}}_k$, the kernel builds

$$
F_k = h_k + \Sigma_H + \Sigma_{F,k}, \qquad
\Sigma_{F,k} = -\sum_{k'} w_{k'}\, V(k - k')\,\big[P_{k'} - P^{\mathrm{ref}}_{k'}\big],
$$

$$
(\Sigma_H)_{aa} = \sum_b H_{ab} \sum_k w_k \big[P_k - P^{\mathrm{ref}}_k\big]_{bb},
$$

and the solvers minimize the free energy

$$
\Omega = \mathrm{Tr}(hP) + \tfrac12 \mathrm{Tr}\big(V_{HF}[P - P^{\mathrm{ref}}]\,(P - P^{\mathrm{ref}})\big) - TS
$$

at fixed $n = \sum_k w_k \mathrm{Tr} P_k$ {cite}`Giuliani2005`. The weight
multiplies the density, never the interaction. The exchange is one FFT
convolution per orbital pair; the Hartree term is the same at every k.

**On this page:** {class}`~hartreefock.Mesh`,
{class}`~hartreefock.HartreeFockKernel` and its `exchange`, `hartree` and
`potential` maps.

## Arrays and conventions

| Argument | Shape | Meaning |
| --- | --- | --- |
| `mesh` | | the k-mesh: `Mesh.square(kmax, nk)` for a continuum patch, `Mesh.bz(B, nk)` for a Brillouin zone; its `weights` are the quadrature |
| `h` | `(nk, nk, nb, nb)` | the single-particle Hamiltonian at every k |
| `V` | callable, or `(nk, nk)`, `(nk, nk, 1, 1)`, `(nk, nk, nb, nb)` | `V(\|q\|)`, scalar or `(..., nb, nb)`; or its table on `mesh.transfer_grid`, `q = 0` at index `(nk // 2, nk // 2)`; `None` for none |
| `T` | scalar | temperature, in the energy unit of `h` |
| `hartree` | `True` or `(nb, nb)` | the Hartree coupling $H_{ab}$ between orbital occupations, `True` for `V(q = 0)`; needs a reference density |
| `exchange` | | on by default |
| `contact_terms` | `(g, O_i, O_j)` triples | q-independent couplings $\tfrac12 g\, O_i \otimes O_j$, for Hubbard-like terms ([tutorial step 3](../tutorial/03-stoner.md)) |
| `reference` | `(nk, nk, nb, nb)` | $P^{\mathrm{ref}}$ above |

The exchange is a periodic convolution on the mesh: transfers wrap around.
The mesh must be large enough that the transfers between the states that
matter do not wrap ([tutorial step 1](../tutorial/01-exchange.md)).

## A model with every channel

Two layers of electrons a distance $d$ apart, one orbital per layer and
one spin, with a layer bias $\pm\Delta/2$. The interaction is resolved by
layer, $V_{ll'}(q) = 2\pi e^2 e^{-qd|l - l'|}/q$, and the Hartree term is
the charging energy of a parallel-plate capacitor, $2\pi e^2 d$ per unit
layer imbalance squared:

```{code-cell} ipython3
import numpy as np
import hartreefock as hf

T, Delta, e2, d = 0.02, 0.4, 1.0, 1.0
mesh = hf.Mesh.square(kmax=2.5, nk=48)
eps = np.sum(mesh.ks**2, axis=-1)
h = eps[..., None, None] * np.eye(2) + np.diag([Delta / 2, -Delta / 2])
dk = mesh.dk[0]
layers = np.abs(np.subtract.outer([0, 1], [0, 1]))


def V(q):  # (..., 2, 2) by layer; 1/q averaged over the q = 0 cell
    inv_q = np.where(q > 0, 1 / np.where(q > 0, q, 1), 4 * np.log(1 + np.sqrt(2)) / dk)
    return 2 * np.pi * e2 * inv_q[..., None, None] * np.exp(-q[..., None, None] * d * layers)


capacitor = np.pi * e2 * d * np.array([[1.0, -1.0], [-1.0, 1.0]])

n = 1 / (2 * np.pi)
P_ref = np.broadcast_to(0.5 * n / mesh.weight_normalization * np.eye(2), h.shape).copy()   # equal layers
P0 = (eps < 1)[..., None, None] * np.eye(2)
print("orbital-resolved V:", V(mesh.transfer_distance_grid).shape, "  Hartree matrix:", capacitor.tolist())
```

## The reference density

Mean-field calculations in a truncated basis usually measure the
self-energy from a reference state whose interaction effects are already
in `h`, such as the charge-neutral density of a continuum model. The
kernel then iterates $V_{HF}[P - P^{\mathrm{ref}}]$. The Hartree channel
always needs one: without a positive background, $\Sigma_H$ of a charged
system diverges.

With `embed_reference=True` the kernel adds
$V_{HF}[P^{\mathrm{ref}}]$ to `h` once, so the loop solves standard
Hartree-Fock, $F = h + V_{HF}[P]$, and reports the standard energy and
chemical potential.

Standard Hartree-Fock is also a reference density of zero. Solving both
ways gives the same state:

```{code-cell} ipython3
config = hf.SolverConfig(density_tol=1e-10, comm_tol=1e-10)
common = dict(T=T, hartree=capacitor)
standard = hf.HartreeFockKernel(mesh, h, V, reference=np.zeros_like(h), **common)
embedded = hf.HartreeFockKernel(mesh, h, V, reference=P_ref, embed_reference=True, **common)
r_std = hf.solve(standard, P0, n, config=config)
r_emb = hf.solve(embedded, P0, n, config=config)

layer_imbalance = np.einsum("ij,ijaa->a", mesh.weights, r_std.density).real @ [1, -1] / n
print(f"layer imbalance (n1 - n2)/n = {layer_imbalance:+.4f}")
print(f"energy: standard {r_std.energy:.8f}   embedded {r_emb.energy:.8f}")
print(f"mu:     standard {r_std.mu:.8f}   embedded {r_emb.mu:.8f}")
print(f"largest density difference: {np.abs(np.asarray(r_std.density) - np.asarray(r_emb.density)).max():.1e}")
```

The bias $\Delta$ pushes charge into the lower layer. Without
interactions the imbalance would be $\Delta/2 = 0.2$ of the density (each
layer has the density of states $1/4\pi$); the capacitor term pushes
back, more than exchange helps.

```{code-cell} ipython3
:tags: [remove-cell]
assert abs(r_emb.energy - r_std.energy) < 1e-8
assert abs(r_emb.mu - r_std.mu) < 1e-7
assert np.abs(np.asarray(r_std.density) - np.asarray(r_emb.density)).max() < 1e-6
```

## The interaction as linear maps

`kernel.exchange(rho)`, `kernel.hartree(rho)` and `kernel.potential(rho)`
apply the interaction to any density-like array with NumPy. The solvers'
Fock matrix is `h + kernel.potential(P - P_ref)`, so these maps check a
solution or build a response function by hand:

```{code-cell} ipython3
P = np.asarray(r_std.density)
F = h + standard.potential(P)
print(f"|F - result.fock| = {np.abs(F - np.asarray(r_std.fock)).max():.1e}")
print(f"|[F, P]|          = {np.abs(F @ P - P @ F).max():.1e}")
```

```{code-cell} ipython3
:tags: [remove-cell]
assert np.abs(F - np.asarray(r_std.fock)).max() < 1e-8
```
