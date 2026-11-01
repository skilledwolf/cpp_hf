---
file_format: mystnb
kernelspec:
  name: python3
---

# Solvers

Every solver takes a kernel, a starting density `P0` of shape
`(nk1, nk2, nb, nb)`, the particle number `n_electrons` (counted with the
weights), a config and a `backend`. The chemical potential is solved for at
every step. This page runs each of them on one problem and compares.

## Theory

Finite-temperature Hartree-Fock minimizes the free energy
$\Omega[P] = E_{HF}[P] - TS[P]$ over one-body densities with
$0 \le P \le 1$ and fixed $n$ {cite}`Mermin1965`. A stationary point
satisfies $P = f\big((F[P] - \mu)/T\big)$, with $F = \delta E_{HF}/\delta P$
the Fock matrix. Two families find it:

- **Roothaan iteration** (`solve_scf`): diagonalize $F[P]$, refill at fixed
  $n$, repeat. Plain iteration can oscillate; extrapolation over past
  iterates (DIIS {cite}`Pulay1980`) or a line search on $\Omega$ (optimal
  damping {cite}`Cances2000`) makes it converge.
- **Direct minimization** (`solve`): descend $\Omega$ on the manifold of
  densities, with conjugate gradients or a Newton trust region on the exact
  Hessian.

Both stop at any stationary point, minimum or saddle. The Hessian at the
solution tells them apart {cite}`Thouless1960`.

**On this page:** {func}`~hartreefock.solve_scf`, {class}`~hartreefock.SCFConfig`,
{func}`~hartreefock.solve`, {class}`~hartreefock.SolverConfig`,
{func}`~hartreefock.solve_minimum`, {func}`~hartreefock.stability_analysis`,
{func}`~hartreefock.solve_continuation`, {func}`~hartreefock.solve_deflated`.

## The test problem

The gated electron gas of the [quickstart](../get-started/quickstart.md),
started from a partially polarized density with the right particle number:

```{code-cell} ipython3
import time
import numpy as np
import hartreefock as hf


def gated_gas(nk, kmax=1.2, T=0.005):
    k = np.linspace(-kmax, kmax, nk, endpoint=False) + kmax / nk
    kx, ky = np.meshgrid(k, k, indexing="ij")
    weights = np.full((nk, nk), (2 * kmax / nk) ** 2 / (2 * np.pi) ** 2)
    eps = 0.5 * (kx**2 + ky**2)
    dk = k[1] - k[0]
    q1 = dk * (np.arange(nk) - nk // 2)
    q = np.hypot(*np.meshgrid(q1, q1, indexing="ij"))
    V = 5.0 * 2 * np.pi * np.where(q > 0, np.tanh(10 * q) / np.where(q > 0, q, 1.0), 10.0)
    return hf.HartreeFockKernel(weights, eps[..., None, None] * np.eye(2), V[..., None, None], T=T), eps


n = 0.02
kernel, eps = gated_gas(48)
disc = (eps < 2 * np.pi * n)[..., None, None]          # holds n electrons in one band
P0 = disc * np.diag([0.7, 0.3])


def magnetization(P, w=kernel.w2d):
    M = np.einsum("ij,ijab->ab", w, np.asarray(P))
    return np.linalg.norm([2 * M[0, 1].real, 2 * M[0, 1].imag, (M[0, 0] - M[1, 1]).real]) / n
```

## SCF

{class}`~hartreefock.SCFConfig` picks the acceleration. It stops when the
density change is below `density_tol` and the commutator $[F, P]$ below
`comm_tol`. {class}`~hartreefock.SCFResult` has the density, Fock matrix,
energy, chemical potential, iteration count, convergence flag, a message
and the per-iteration `history`.

```{code-cell} ipython3
rows, residuals = [], {}
for acceleration in ("linear", "diis", "oda", "rdiis", "anderson"):
    t = time.perf_counter()
    r = hf.solve_scf(kernel, P0, n, config=hf.SCFConfig(acceleration=acceleration))
    rows.append((acceleration, r.converged, r.n_iter, float(r.energy), time.perf_counter() - t))
    residuals[acceleration] = np.asarray(r.history["density_residual"])
    print(f"{acceleration:>9}: converged {str(r.converged):5}  {r.n_iter:3d} iterations  "
          f"E = {r.energy:.10f}  m = {magnetization(r.density):.4f}  {rows[-1][-1] * 1e3:5.0f} ms")
```

```{code-cell} ipython3
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
for acceleration, res in residuals.items():
    ax.semilogy(np.arange(1, len(res) + 1), res, "o-", ms=3, label=acceleration)
ax.axhline(1e-7, ls=":", color="0.5")
ax.set(xlabel="iteration", ylabel="density change")
ax.legend();
```

- `"linear"`: plain mixing with `mixing`.
- `"diis"`: Pulay DIIS on the Fock commutator.
- `"oda"`: optimal damping, an exact line search at $T = 0$ and free-energy
  backtracking at $T > 0$.
- `"rdiis"` (Fock) and `"anderson"` (density): full-residual schemes with a
  free-energy safeguard. RDIIS is the recommended default at finite
  temperature; the native backend adds `preconditioner="hartree"` for
  problems where the Hartree term makes the residual stiff.

## Direct minimization

{class}`~hartreefock.SolverConfig` picks the optimizer: `"cg"`, Riemannian
conjugate gradients; `"newton"`, a truncated-CG trust region on the exact
Hessian, quadratic near a solution and in need of double precision; `"auto"`
(native only), CG first and Newton once the iterate is close. With
`tol_grad > 0` the stationarity norm is the stopping test; otherwise CG
stops when the free energy improves by less than `tol_E` over
`plateau_window` iterations. Compare `result.energy`, which is recomputed at
the end, rather than the tail of `history["E"]`.

```{code-cell} ipython3
for optimizer in ("cg", "newton", "auto"):
    t = time.perf_counter()
    r = hf.solve(kernel, P0, n, config=hf.SolverConfig(optimizer=optimizer, tol_grad=1e-8, max_iter=500))
    rows.append((optimizer, bool(r.converged), int(r.n_iter), float(r.energy), time.perf_counter() - t))
    print(f"{optimizer:>9}: converged {str(bool(r.converged)):5}  {int(r.n_iter):3d} iterations  "
          f"E = {float(r.energy):.10f}  m = {magnetization(r.density):.4f}  {rows[-1][-1] * 1e3:5.0f} ms")
```

Every method reaches the same polarized state. RDIIS, ODA, Anderson,
Newton and `"auto"` take five to eight iterations, plain CG hundreds to
reach a gradient of $10^{-8}$, which is why `"auto"` hands over to Newton.
Linear mixing and DIIS lie on top of each other in the figure: both blend
the new density with the old one by `mixing` (0.5 by default), and on this
problem one Roothaan step is already almost exact, so the blend halves the
error per iteration. With `mixing=1.0` DIIS converges in 6 iterations
here; the default is safer when the Roothaan step overshoots.

```{code-cell} ipython3
:tags: [remove-cell]
E = np.array([row[3] for row in rows])
assert all(row[1] for row in rows)
assert np.ptp(E) < 1e-8
```

## Stability

A converged residual means a stationary point, which can be a saddle.
{func}`~hartreefock.stability_analysis` returns the lowest eigenvalue of the
free-energy Hessian at a density and the direction that goes with it;
`stable` is `True`, `False`, or `None` when the test is inconclusive. It
needs $T > 0$. {func}`~hartreefock.solve_minimum` solves with RDIIS, runs
the test, and on negative curvature steps downhill along that direction and
solves again, up to `max_escapes` times. [Tutorial step 3](../tutorial/03-stoner.md)
uses both on the Stoner instability. They are local tests in the basis you
give: compare energies across starts to compare phases.

:::{admonition} Singular interactions and the q = 0 cell
:class: warning
On a mesh, the $q = 0$ entry of `coulomb_q` couples each k-point to
itself with strength $w_k V(0)$. When that exceeds the thermal stiffness
of a Fermi-surface state, about $4T$, filling single k-points lowers the
free energy, and the test reports these mesh modes as unstable. For the
Coulomb gas of [tutorial step 1](../tutorial/01-exchange.md) at $r_s = 2$
with `nk = 64`, `kmax = 3` and $T = 0.002\,E_F$, the paramagnet's lowest
curvature is $-17$ with the cell-averaged $V(0)$ and $+0.11$ with
$V(0) = 0$. Raise $T$ or refine the mesh until $w_k V(0) \ll 4T$, or
compare both choices, before reading an instability of a Coulomb problem
as physical.
:::

## Continuation

{func}`~hartreefock.solve_continuation` solves a coarse kernel, resamples
the result to the fine mesh, and solves there. Each stage runs SCF or
direct minimization according to its config, so they can mix:

```{code-cell} ipython3
coarse, eps_coarse = gated_gas(24)
P0_coarse = (eps_coarse < 2 * np.pi * n)[..., None, None] * np.diag([0.7, 0.3])
c = hf.solve_continuation(coarse, kernel, P0_coarse, n, n,
                          coarse_config=hf.SCFConfig(acceleration="rdiis"),
                          fine_config=hf.SCFConfig(acceleration="rdiis"))
print(f"coarse: {c.coarse.n_iter} iterations, E = {c.coarse.energy:.6f}")
print(f"fine:   {c.fine.n_iter} iterations from the resampled seed, E = {c.fine.energy:.10f}")
```

Both stages land on the same state as a direct solve. Here the mesh is
small and the resampled seed takes more iterations than a fresh start;
continuation pays when each fine iteration is expensive, so time it on
your problem before relying on it.

## Deflation

{func}`~hartreefock.solve_deflated` collects distinct solutions: after each
one it adds a repulsive bias around the solutions found so far and solves
again with Newton, then polishes the result without the bias.
`result.best` is the lowest. From the balanced start:

```{code-cell} ipython3
start = (eps < np.pi * n)[..., None, None] * np.eye(2)
d = hf.solve_deflated(kernel, start, n, n_solutions=4)
for s in d.solutions:
    check = hf.stability_analysis(kernel, s.density, n)
    print(f"E = {float(s.energy):.8f}   |m| = {magnetization(s.density):.3f}   stable = {check.stable}")
```

It finds the paramagnetic saddle and the ferromagnet more than once. The
gas is spin-rotation invariant, so the ferromagnet is a sphere of
solutions, and each spin direction counts as distinct; the energy
differences of about $5\times10^{-7}$ between them come from where each
polish stopped. Compare an order
parameter that does not depend on the direction, here $|\mathbf m|$, to
tell physically different solutions apart.

```{code-cell} ipython3
:tags: [remove-cell]
assert abs(c.fine.energy - rows[3][3]) < 1e-8
ms = sorted(round(magnetization(s.density), 2) for s in d.solutions)
assert ms[0] == 0.0 and ms[-1] == 1.0
assert abs(float(d.best.energy) - rows[3][3]) < 1e-6
```

## Symmetries

`block_sizes` in either config tells the eigensolver that the Fock matrix is
block diagonal. `project_fn` averages every density and Fock matrix over a
symmetry group, which keeps a solve in a symmetry sector;
{func}`hartreefock.symmetry.make_project_fn` builds one from unitary,
spatial and time-reversal symmetries.
