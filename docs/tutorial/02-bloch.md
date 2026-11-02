---
file_format: mystnb
kernelspec:
  name: python3
---

# 2. Bloch ferromagnetism

A dilute electron gas lowers its exchange energy by putting all electrons
in one spin. This page solves the gas from a spin-balanced and a fully
polarized start, follows both solutions in density, and finds the density
where the polarized one becomes the ground state.

:::{admonition} Physics
At fixed density, polarizing the spin raises $k_F$ by $\sqrt2$. In
Rydberg per electron the paramagnet and the ferromagnet have
{cite}`Bloch1929,Giuliani2005`

$$
E_P = \frac{1}{r_s^2} - \frac{8\sqrt2}{3\pi r_s}, \qquad
E_F = \frac{2}{r_s^2} - \frac{16}{3\pi r_s} .
$$

The kinetic energy doubles and the exchange energy grows by $\sqrt2$. In
Hartree-Fock both are plane-wave solutions at every $r_s$, so each energy
is exactly the sum of a $1/r_s^2$ and a $1/r_s$ term, and the two cross at

$$
r_s^\ast = \frac{3\pi}{16 - 8\sqrt2} = 2.011 .
$$

The transition is first order: the magnetization jumps from 0 to 1.

**You will use:** {func}`~hartreefock.solve_scf`, `SCFResult.density`.
:::

## Two solutions

The setup is the one of [step 1](01-exchange.md), with a mesh that reaches
past $2\sqrt2\,k_F$, twice the Fermi momentum of the polarized gas:

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
import hartreefock as hf


def electron_gas(nk, kmax, rs):
    """The 2D electron gas on an nk x nk mesh, in units of the paramagnetic k_F and E_F."""
    mesh = hf.Mesh.square(kmax, nk)
    eps = np.sum(mesh.ks**2, axis=-1)
    dk = mesh.dk[0]

    def V(q):
        return 2 * np.pi * np.sqrt(2) * rs * np.where(q > 0, 1 / np.where(q > 0, q, 1), 4 * np.log(1 + np.sqrt(2)) / dk)

    return mesh, eps[..., None, None] * np.eye(2), V, eps


n = 1 / (2 * np.pi)


def solve_both(nk, rs):
    """Paramagnetic and fully polarized solutions: (kinetic, exchange, magnetization) of each, Ry per electron."""
    mesh, h, V, eps = electron_gas(nk, kmax=3.0, rs=rs)
    kernel = hf.HartreeFockKernel(mesh, h, V, T=0.002)
    w = mesh.weights
    starts = {
        "para": (eps < 1)[..., None, None] * np.eye(2),
        "ferro": (eps < 2)[..., None, None] * np.diag([1.0, 0.0]),
    }
    out = {}
    for name, P0 in starts.items():
        r = hf.solve_scf(kernel, P0, n)
        P = np.asarray(r.density)
        up, down = np.einsum("ij,ijaa->a", w, P).real
        kinetic = np.einsum("ij,ijab,ijba->", w, h, P).real
        to_ry = 2 / rs**2 / n
        out[name] = (kinetic * to_ry, (float(r.energy) - kinetic) * to_ry, (up - down) / n)
    return out


for name, (t, x, m) in solve_both(nk=96, rs=2.0).items():
    print(f"{name:>5}:  kinetic {t:.4f}  exchange {x:.4f}  total {t + x:.4f} Ry,  magnetization {m:.3f}")
```

Both starts converge and keep their magnetization: each is a
self-consistent solution. At $r_s = 2$ they are within a few mRy of each
other.

## Following the solutions in density

Both energies scale exactly with $r_s$, so the coefficients of $1/r_s^2$
and $1/r_s$ found at one density give the whole curve. The exchange
coefficient converges as $1/n_k$ ([step 1](01-exchange.md)); extrapolate
it for each state:

```{code-cell} ipython3
rs0 = 2.0
meshes = np.array([48, 64, 96, 128])
coef = {"para": [], "ferro": []}
for nk in meshes:
    for name, (t, x, _) in solve_both(nk, rs0).items():
        coef[name].append((t * rs0**2, x * rs0))      # kinetic * r_s², exchange * r_s
a = {name: np.polyfit(1 / meshes, np.array(c)[:, 1], 1)[1] for name, c in coef.items()}
b = {name: np.array(c)[-1, 0] for name, c in coef.items()}
for name in a:
    print(f"{name:>5}:  E = {b[name]:.4f}/r_s² {a[name]:+.5f}/r_s")
```

```{code-cell} ipython3
rs = np.linspace(1.0, 4.0, 300)
E = {name: b[name] / rs**2 + a[name] / rs for name in a}
rs_star = (b["ferro"] - b["para"]) / (a["para"] - a["ferro"])

fig, ax = plt.subplots()
ax.plot(rs, E["para"], label="paramagnet")
ax.plot(rs, E["ferro"], label="ferromagnet")
ax.axvline(rs_star, ls=":", color="0.5")
ax.set(xlabel="$r_s$", ylabel="energy (Ry per electron)", ylim=(-0.45, 0.2))
ax.legend();
```

Below $r_s^\ast$ the paramagnet is lower; above it, the ferromagnet.
Hartree-Fock puts the transition far too early: correlation, which
Hartree-Fock leaves out, favours the paramagnet, and quantum Monte Carlo
places the polarized phase of the 2D gas near $r_s \approx 26$ {cite}`Attaccalite2002`.

## Metastability

A first-order transition has a window where the higher solution is still a
local minimum. For the paramagnet the uniform spin curvature changes sign
at $r_s = \pi/\sqrt2 = 2.22$ at zero temperature, so it is metastable from
2.01 to 2.22. Testing that with {func}`~hartreefock.stability_analysis` on
this mesh needs care: with a singular $V(q)$ the $q = 0$ cell couples each
k-point to itself, and at temperatures well below that coupling single
k-points at the Fermi surface gain by filling up. The test then reports
these mesh modes as unstable (see [Solvers](../guides/solvers.md)).
[Step 3](03-stoner.md) finds a spin instability with the test, on a model
where it is clean.

## Check

```{code-cell} ipython3
exact = 3 * np.pi / (16 - 8 * np.sqrt(2))
print(f"crossing: r_s* = {rs_star:.4f}, closed form 3π/(16 - 8√2) = {exact:.4f}")
print(f"exchange coefficients: para {a['para']:.5f} vs {-8 * np.sqrt(2) / (3 * np.pi):.5f}, "
      f"ferro {a['ferro']:.5f} vs {-16 / (3 * np.pi):.5f}")
```

```{code-cell} ipython3
:tags: [remove-cell]
assert abs(rs_star / exact - 1) < 1e-3
assert abs(b["para"] - 1) < 5e-3 and abs(b["ferro"] - 2) < 5e-3
```

## Next

[Step 3](03-stoner.md) asks when the paramagnet stops being a local
minimum, and how to get from it to the state below.
