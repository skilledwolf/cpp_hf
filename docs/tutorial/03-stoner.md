---
file_format: mystnb
kernelspec:
  name: python3
---

# 3. The Stoner instability

A converged solve is a stationary point of the free energy, which can be a
saddle. This page takes a spin-balanced metal with an on-site repulsion,
measures the curvature of its free energy as the repulsion grows, and
finds the critical repulsion where it turns negative. Above it,
{func}`~hartreefock.solve_minimum` leaves the saddle for the ferromagnet.

:::{admonition} Physics
For a contact repulsion $U n_\uparrow n_\downarrow$ the paramagnet becomes
unstable to a uniform spin polarization when {cite}`Stoner1938`

$$
U\chi_0 = 1, \qquad \chi_0 = \sum_k w_k\, \frac{f_k(1 - f_k)}{T},
$$

with $\chi_0$ the spin susceptibility per spin at temperature $T$. For a
parabolic band $k^2$ in two dimensions the density of states per spin is
$1/4\pi$ and does not depend on energy, so $\chi_0 = 1/4\pi$ up to
$e^{-\mu/T}$ and $U_c = 4\pi$. Above $U_c$ each spin's density solves

$$
n_\sigma = \frac{T}{4\pi}\,\ln\!\big(1 + e^{(\mu - U n_{-\sigma})/T}\big),
\qquad n_\uparrow + n_\downarrow = n ,
$$

and because the density of states is constant, the polarization jumps to
nearly 1 right above $U_c$.

**You will use:** {func}`~hartreefock.stability_analysis`,
{class}`~hartreefock.StabilityResult`, {func}`~hartreefock.solve_minimum`,
{class}`~hartreefock.MinimumResult`, `contact_terms` of
{class}`~hartreefock.HartreeFockKernel`.
:::

## A contact interaction

`contact_terms` adds q-independent couplings $\tfrac12 g\,O_i\otimes O_j$
as `(g, O_i, O_j)` triples. The Hubbard term $U n_\uparrow n_\downarrow$ is two
of them, with the spin projectors; with no `V` there is no long-range term.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
import hartreefock as hf

T = 0.05
mesh = hf.Mesh.square(kmax=2.0, nk=64)
weights = mesh.weights
eps = np.sum(mesh.ks**2, axis=-1)
h = eps[..., None, None] * np.eye(2)
n = 1 / (2 * np.pi)                                  # k_F = 1, E_F = 1
up, down = np.diag([1.0, 0.0]), np.diag([0.0, 1.0])


def hubbard(U):
    return hf.HartreeFockKernel(mesh, h, T=T, contact_terms=[(U, up, down), (U, down, up)])


P_para = (eps < 1)[..., None, None] * np.eye(2)
```

## The curvature

{func}`~hartreefock.stability_analysis` takes a converged density and returns
the lowest eigenvalue of the free-energy Hessian there, over all
particle-conserving Hermitian perturbations, spin flips included. Solve the
paramagnet at a range of $U$ and test each solution:

```{code-cell} ipython3
Us = np.array([8.0, 10.0, 11.0, 12.0, 12.5, 13.0, 14.0, 16.0])
curvature, magnetization = [], []
for U in Us:
    r = hf.solve(hubbard(U), P_para, n)
    P = np.asarray(r.density)
    check = hf.stability_analysis(hubbard(U), P, n)
    curvature.append(check.lowest_eigenvalue)
    magnetization.append(np.einsum("ij,ij->", weights, (P[..., 0, 0] - P[..., 1, 1]).real) / n)
    print(f"U = {U:5.1f}:  magnetization {magnetization[-1]:.3f}   lowest curvature {check.lowest_eigenvalue:+.4f}   stable = {check.stable}")

f = P[..., 0, 0].real
chi0 = np.sum(weights * f * (1 - f)) / T              # the paramagnet's own occupations
U_c = 1 / chi0
```

The solve from the balanced start converges to the paramagnet at every
$U$; above about $12.6$ it is a saddle. The curvature falls linearly in
$U$: the paramagnet's density does not change with $U$, and the Hessian
there is the kinetic and entropy part plus $U$ times the interaction part.

```{code-cell} ipython3
fit = np.polyfit(Us, curvature, 1)
U_zero = -fit[1] / fit[0]

fig, ax = plt.subplots()
ax.plot(Us, curvature, "o", label="lowest curvature at the paramagnet")
ax.plot(Us, np.polyval(fit, Us), "--", lw=1)
ax.axhline(0, color="0.5", lw=0.8)
ax.axvline(U_c, ls=":", color="C1", label=r"$1/\chi_0$")
ax.set(xlabel="$U$ ($E_F/k_F^2$)", ylabel="curvature")
ax.legend();
```

## Leaving the saddle

{func}`~hartreefock.solve_minimum` solves, runs the same test, and when it
finds negative curvature, steps along the unstable direction and solves
again. Its result holds the last solve in `solution` and its test in
`stability`. The spin can point anywhere, so measure the length of the
magnetization vector:

```{code-cell} ipython3
def magnetization_length(P):
    M = np.einsum("ij,ijab->ab", weights, P)
    return np.linalg.norm([2 * M[0, 1].real, 2 * M[0, 1].imag, (M[0, 0] - M[1, 1]).real]) / n


above = [13.0, 13.5, 14.0, 15.0]
found = []
for U in above:
    m = hf.solve_minimum(hubbard(U), P_para, n)
    found.append(magnetization_length(np.asarray(m.solution.density)))
    print(f"U = {U:4.1f}:  {m.escapes} escape,  magnetization {found[-1]:.4f},  "
          f"stable = {m.stability.stable},  free energy {m.free_energies[0]:.5f} -> {m.free_energies[-1]:.5f}")
```

Each run starts at the balanced paramagnet, finds it unstable, takes one
step along the spin mode and converges to a polarized state that the test
then passes. The free energy drops by the condensation energy.

## Check

The continuum gives two predictions: $U_c = 4\pi$, and the magnetization
above it, from the two self-consistency equations of the box above.

```{code-cell} ipython3
from scipy.optimize import brentq


def continuum_magnetization(U):
    """Polarization z of the 2D parabolic band with contact U at temperature T, from n_σ(μ)."""
    def mismatch(z):
        n_up, n_down = n * (1 + z) / 2, n * (1 - z) / 2
        mu = T * np.log(np.expm1(4 * np.pi * n_up / T)) + U * n_down
        return T / (4 * np.pi) * np.logaddexp(0, (mu - U * n_up) / T) - n_down
    zs = np.linspace(1e-6, 1 - 1e-9, 2000)
    signs = np.sign([mismatch(z) for z in zs])
    i = np.flatnonzero(np.diff(signs))[-1]
    return brentq(mismatch, zs[i], zs[i + 1])


exact = [continuum_magnetization(U) for U in above]
print(f"curvature crosses zero at U = {U_zero:.3f};  1/χ0 on this mesh = {U_c:.3f};  continuum 4π = {4 * np.pi:.3f}")
for U, m, z in zip(above, found, exact):
    print(f"U = {U:4.1f}:  magnetization {m:.4f}   continuum {z:.4f}")
```

```{code-cell} ipython3
:tags: [remove-cell]
assert abs(U_zero / U_c - 1) < 2e-3
assert abs(U_c / (4 * np.pi) - 1) < 5e-3
assert np.allclose(found, exact, atol=2e-3)
```

The crossing matches $1/\chi_0$ of the mesh to the precision of the linear
fit, and that is $4\pi$ to 0.2%; the remaining difference is the mesh
resolving the thermal shell at the Fermi surface.

## Next

The [solvers guide](../guides/solvers.md) compares the other solvers on
these problems. [The kernel](../guides/kernel.md) covers Hartree terms,
reference densities and what the energy means.
