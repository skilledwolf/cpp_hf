---
file_format: mystnb
kernelspec:
  name: python3
---

# The kernel

A kernel holds everything about the problem that does not change during a
solve. {class}`~hartreefock.HartreeFockKernel` is the dense case: a
Hamiltonian on a k-mesh with an interaction that depends on the momentum
transfer. {class}`~hartreefock.SuperlatticeHartreeFockKernel` is the moiré
plane-wave case (see [Superlattices](superlattice.md)).

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

**On this page:** {class}`~hartreefock.HartreeFockKernel` and its
`exchange`, `hartree` and `potential` maps.

## Arrays and conventions

| Argument | Shape | Meaning |
| --- | --- | --- |
| `weights` | `(nk1, nk2)` | quadrature weights; non-negative with a positive sum |
| `hamiltonian` | `(nk1, nk2, nb, nb)` | the single-particle Hamiltonian at every k |
| `coulomb_q` | `(nk1, nk2, 1, 1)` or `(nk1, nk2, nb, nb)` | `V(q)` on the transfer grid, `q = 0` at index `(nk1 // 2, nk2 // 2)` |
| `T` | scalar | temperature, in the energy unit of `h` |
| `include_hartree`, `hartree_matrix` | `(nb, nb)` | the Hartree coupling $H_{ab}$ between orbital occupations; needs a reference density |
| `include_exchange` | | on by default |
| `contact_terms` | `(g, O_i, O_j)` triples | q-independent couplings $\tfrac12 g\, O_i \otimes O_j$, for Hubbard-like terms ([tutorial step 3](../tutorial/03-stoner.md)) |
| `reference_density` | `(nk1, nk2, nb, nb)` | $P^{\mathrm{ref}}$ above |

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

nk, kmax, T = 48, 2.5, 0.02
k = np.linspace(-kmax, kmax, nk, endpoint=False) + kmax / nk
kx, ky = np.meshgrid(k, k, indexing="ij")
dk = k[1] - k[0]
weights = np.full((nk, nk), dk**2 / (2 * np.pi) ** 2)
Delta, e2, d = 0.4, 1.0, 1.0
h = (kx**2 + ky**2)[..., None, None] * np.eye(2) + np.diag([Delta / 2, -Delta / 2])

q1 = dk * (np.arange(nk) - nk // 2)
q = np.hypot(*np.meshgrid(q1, q1, indexing="ij"))
inv_q = np.where(q > 0, 1 / np.where(q > 0, q, 1), 4 * np.log(1 + np.sqrt(2)) / dk)
layers = np.abs(np.subtract.outer([0, 1], [0, 1]))
V = 2 * np.pi * e2 * inv_q[..., None, None] * np.exp(-q[..., None, None] * d * layers)
capacitor = np.pi * e2 * d * np.array([[1.0, -1.0], [-1.0, 1.0]])

n = 1 / (2 * np.pi)
P_ref = np.broadcast_to(0.5 * n / weights.sum() * np.eye(2), h.shape).copy()   # equal layers
P0 = ((kx**2 + ky**2) < 1)[..., None, None] * np.eye(2)
print("orbital-resolved V:", V.shape, "  Hartree matrix:", capacitor.tolist())
```

## The reference density

Mean-field calculations in a truncated basis usually measure the
self-energy from a reference state whose interaction effects are already
in `h`, such as the charge-neutral density of a continuum model. The
kernel then iterates $V_{HF}[P - P^{\mathrm{ref}}]$. The Hartree channel
always needs one: without a positive background, $\Sigma_H$ of a charged
system diverges.

With `embed_reference_potential=True` the kernel adds
$V_{HF}[P^{\mathrm{ref}}]$ to `h` once, so the loop solves standard
Hartree-Fock, $F = h + V_{HF}[P]$. The reported energy and chemical
potential then differ from the standard ones by constants the kernel
keeps:

$$
E_{\text{standard}} = E - \texttt{embedded\_energy\_offset} + \texttt{embedded\_hartree\_shift}\cdot n,
\qquad
\mu_{\text{standard}} = \mu + \texttt{embedded\_hartree\_shift}.
$$

`center_embedded_hartree` (on by default) removes the orbital mean of the
embedded Hartree potential, which keeps the Fermi level near zero.

Standard Hartree-Fock is also a reference density of zero. Solving both
ways gives the same density, and the energies obey the relation:

```{code-cell} ipython3
config = hf.SCFConfig(acceleration="rdiis", density_tol=1e-10, comm_tol=1e-10)
common = dict(T=T, include_hartree=True, hartree_matrix=capacitor)
standard = hf.HartreeFockKernel(weights, h, V, reference_density=np.zeros_like(h), **common)
embedded = hf.HartreeFockKernel(weights, h, V, reference_density=P_ref,
                                embed_reference_potential=True, **common)
r_std = hf.solve_scf(standard, P0, n, config=config)
r_emb = hf.solve_scf(embedded, P0, n, config=config)

E_from_embedded = r_emb.energy - embedded.embedded_energy_offset + embedded.embedded_hartree_shift * n
mu_from_embedded = r_emb.mu + embedded.embedded_hartree_shift
layer_imbalance = np.einsum("ij,ijaa->a", weights, r_std.density).real @ [1, -1] / n
print(f"layer imbalance (n1 - n2)/n = {layer_imbalance:+.4f}")
print(f"energy: standard {r_std.energy:.8f}   from the embedded kernel {E_from_embedded:.8f}")
print(f"mu:     standard {r_std.mu:.8f}   from the embedded kernel {mu_from_embedded:.8f}")
print(f"largest density difference: {np.abs(np.asarray(r_std.density) - np.asarray(r_emb.density)).max():.1e}")
```

The bias $\Delta$ pushes charge into the lower layer. Without
interactions the imbalance would be $\Delta/2 = 0.2$ of the density (each
layer has the density of states $1/4\pi$); the capacitor term pushes
back, more than exchange helps.

```{code-cell} ipython3
:tags: [remove-cell]
assert abs(E_from_embedded - r_std.energy) < 1e-8
assert abs(mu_from_embedded - r_std.mu) < 1e-7
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
