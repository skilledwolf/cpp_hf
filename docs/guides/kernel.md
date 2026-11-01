# The kernel

A kernel holds everything about the problem that does not change during a
solve. {class}`~hartreefock.HartreeFockKernel` is the dense case: a
Hamiltonian on a k-mesh with an interaction that depends on the momentum
transfer. {class}`~hartreefock.SuperlatticeHartreeFockKernel` is the moiré
plane-wave case (see [Superlattices](superlattice.md)).

```python
kernel = hf.HartreeFockKernel(
    weights, h, coulomb_q, T,
    include_hartree=False, include_exchange=True,
    reference_density=None, hartree_matrix=None, contact_terms=None,
    embed_reference_potential=False,
)
```

## Arrays and conventions

| Argument | Shape | Meaning |
| --- | --- | --- |
| `weights` | `(nk1, nk2)` | quadrature weights; non-negative with a positive sum |
| `hamiltonian` | `(nk1, nk2, nb, nb)` | the single-particle Hamiltonian at every k |
| `coulomb_q` | `(nk1, nk2, 1, 1)` or `(nk1, nk2, nb, nb)` | `V(q)` on the transfer grid, `q = 0` at index `(nk1 // 2, nk2 // 2)` |
| `T` | scalar | temperature, in the energy unit of `h` |

The particle number counts with the weights, `n = Σ_k w_k Tr P_k`, and the
self-energies are

```text
Σ_F(k) = -Σ_q V(k - q) w(q) [P(q) - P_ref(q)]
```

so the weight multiplies the density, never the interaction. The exchange is
one FFT convolution per orbital pair. The Hartree term couples the orbital
occupations `Σ_k w(k) [P(k) - P_ref(k)]` through `hartree_matrix` and is the
same at every k.

## Channels

- **Exchange** (`include_exchange`, on by default) uses `coulomb_q`.
- **Hartree** (`include_hartree`) is k-independent. It needs a
  `hartree_matrix`, the `(nb, nb)` coupling between orbitals at `q = 0`, and a
  reference density.
- **Contact terms** (`contact_terms`) are q-independent interactions
  `½ g O_i ⊗ O_j`, given as `(g, O_i, O_j)` triples, for Hubbard-like couplings.

## The reference density

Mean-field calculations in a truncated basis usually measure the self-energy
from a reference state, such as the charge-neutral density of a continuum
model, whose interaction effects are already in `h`. The kernel then iterates
`V_HF[P - P_ref]`, and the energy is the variational functional

```text
E = Tr(h P) + ½ Tr(V_HF[P - P_ref] (P - P_ref))
```

With `embed_reference_potential=True`, the kernel adds `V_HF[P_ref]` to `h`
once, so the solve is standard Hartree-Fock, `F = h + V_HF[P]`. The reported
energy and chemical potential then differ from the standard ones by known
constants:

```text
E_standard  = E - kernel.embedded_energy_offset + kernel.embedded_hartree_shift * n_electrons
mu_standard = mu + kernel.embedded_hartree_shift
```

`center_embedded_hartree` (on by default) removes the orbital mean of the
embedded Hartree potential, which keeps the Fermi level near zero; fixed-N
energy differences do not change.

## The interaction as linear maps

`kernel.exchange(rho)`, `kernel.hartree(rho)` and `kernel.potential(rho)`
apply the interaction to any density-like array with NumPy. The solvers'
Fock matrix is `h + kernel.potential(P - P_ref)`, so these are the tools for
checking a solution or building response functions by hand.
