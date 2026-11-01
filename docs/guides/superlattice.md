# Superlattices

{class}`~hartreefock.SuperlatticeHartreeFockKernel` handles moiré problems in
a plane-wave basis `|k, G, α⟩`: `G` runs over a set of reciprocal-lattice
vectors and `α` over orbitals, so the matrix dimension is
`D = n_G · dim_orb` with the `G` index slow. The solvers are the same as for
the dense kernel, on both backends.

```python
kernel = hf.SuperlatticeHartreeFockKernel(
    weights, h, coulomb_V, g_fractional, reciprocal_basis, T=0.1,
    include_hartree=True, reference_density=P_ref,
)
```

| Argument | Meaning |
| --- | --- |
| `coulomb_V` | a callable `|q| -> V(|q|)`; the kernel tabulates it on the momentum transfers it needs |
| `g_fractional` | `(n_G, 2)` integers: the G-basis in units of the reciprocal lattice vectors |
| `reciprocal_basis` | `(2, 2)` matrix `B` with `G_cart = B @ G_frac` |
| `coulomb_V_orbital` | optional `|q| -> V_αβ(|q|)` for an orbital-resolved interaction, such as between layers |
| `hartree_matrix`, `hartree_matrix_orbital` | optional Hartree couplings between G components; by default taken from the interaction with `q = 0` dropped (scalar) or kept (orbital-resolved) |
| `hartree_degeneracy` | the flavour multiplicity of the Hartree term, for spin and valley left out of the basis |

The exchange self-energy is an aperiodic convolution in the absolute momentum
`k - G`, computed with one FFT per `(ΔG, α, β)` channel on a zero-padded grid.

## Off-mesh self-energies

Matrix elements of `Σ(k)` in a truncated G-basis are not periodic, so they
cannot be interpolated from the mesh. `kernel.potential_at(P, k_frac)`
evaluates `V_HF[P - P_ref]` at any fractional momenta, such as a band path,
with a direct plane-wave sum.
