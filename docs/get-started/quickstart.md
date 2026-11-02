---
file_format: mystnb
kernelspec:
  name: python3
---

# Quickstart

A dilute two-dimensional electron gas between metal gates lowers its
exchange energy by polarizing its spin. This page builds that problem,
solves it from a spin-balanced and a polarized start, and asks which
solution is a local minimum. It runs in a few seconds.

## The problem

Three things define it: a {class}`~hartreefock.Mesh` of `nk × nk` k
points, the Hamiltonian `h` on it, of shape `(nk, nk, nb, nb)`, and the
interaction `V` as a function of `|q|`. The units are yours; the solver
assumes none. Here $\hbar = m = 1$, two spin bands with
dispersion $k^2/2$, and a dual-gate interaction
$V(q) = 2\pi g \tanh(qd)/q$.

```{code-cell} ipython3
import numpy as np
import hartreefock as hf

mesh = hf.Mesh.square(kmax=1.2, nk=48)    # Σ_k mesh.weights f = ∫ d²k/(2π)² f
eps = 0.5 * np.sum(mesh.ks**2, axis=-1)
h = eps[..., None, None] * np.eye(2)
d, g = 10.0, 5.0


def V(q):
    return g * 2 * np.pi * np.where(q > 0, np.tanh(q * d) / np.where(q > 0, q, 1.0), d)


kernel = hf.HartreeFockKernel(mesh, h, V, T=0.005)
n_electrons = 0.02          # per unit area, counted with the mesh weights
```

{class}`~hartreefock.HartreeFockKernel` holds what does not change during
a solve. It samples `V` on the momentum transfers of the mesh,
`mesh.transfer_grid`, which has `q = 0` at index `nk // 2`. By default it
keeps the exchange term only; Hartree terms, a
reference density and contact couplings are options
([The kernel](../guides/kernel.md)).

## Solving and checking

```{code-cell} ipython3
# Filled Fermi discs with the right particle number: for k²/2, n_σ = ε_F / 2π.
starts = {
    "balanced": (eps < np.pi * n_electrons)[..., None, None] * np.eye(2),
    "polarized": (eps < 2 * np.pi * n_electrons)[..., None, None] * np.diag([1.0, 0.0]),
}
results = {}
for name, P0 in starts.items():
    r = hf.solve(kernel, P0, n_electrons)
    s = hf.stability_analysis(kernel, r.density, n_electrons)
    up, down = np.einsum("ij,ijaa->a", mesh.weights, r.density).real
    results[name] = (r, s)
    print(f"{name:>9}: E = {r.energy:.6f}, spin up/down = {up:.4f}/{down:.4f}, "
          f"stable = {s.stable} (lowest curvature {s.lowest_eigenvalue:+.1e})")
```

Both starts converge, to different solutions. The spin-balanced one is a
saddle: {func}`~hartreefock.stability_analysis` finds a direction of
negative free-energy curvature. The polarized one is lower and stable; its
lowest curvature is zero within precision, the Goldstone mode of rotating
the spin. A converged residual does not tell the two apart.
{func}`~hartreefock.solve_minimum` runs the test for you and follows an
unstable direction downhill:

```{code-cell} ipython3
m = hf.solve_minimum(kernel, starts["balanced"], n_electrons)
print(f"from the balanced start: {m.escapes} escape, E = {m.solution.energy:.6f}, stable = {m.stability.stable}")
```

```{code-cell} ipython3
:tags: [remove-cell]
(rb, sb), (rp, sp) = results["balanced"], results["polarized"]
assert sb.stable is False and sp.stable is True and rp.energy < rb.energy
assert m.stability.stable and abs(m.solution.energy - rp.energy) < 1e-6
```

## Next

The [tutorial](../tutorial/index.md) builds up the electron gas with the
Coulomb interaction and checks each result against a closed form.
