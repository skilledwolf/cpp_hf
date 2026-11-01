# Solvers

Every solver takes a kernel, a starting density `P0` of shape
`(nk1, nk2, nb, nb)`, the particle number `n_electrons` (counted with the
weights), a config and a `backend`. The chemical potential is solved for at
each step.

| Function | Method | Use it for |
| --- | --- | --- |
| {func}`~hartreefock.solve_scf` | Roothaan iteration with mixing or acceleration | most problems; `acceleration="rdiis"` at finite temperature |
| {func}`~hartreefock.solve` | direct minimization of the free energy | hard cases, gradients, and the Newton polish |
| {func}`~hartreefock.solve_minimum` | safeguarded RDIIS, then a stability check and descent along unstable modes | a solution that is a local minimum, not a saddle |
| {func}`~hartreefock.stability_analysis` | the lowest free-energy curvature at a density | checking any solution |
| {func}`~hartreefock.solve_continuation` | a coarse mesh, then the fine one from its result | large meshes |
| {func}`~hartreefock.solve_deflated` | repeated Newton solves biased away from earlier solutions | finding several distinct solutions |

## SCF

{class}`~hartreefock.SCFConfig` picks the acceleration:

- `"linear"`: plain mixing with `mixing`.
- `"diis"`: Pulay DIIS on the Fock commutator.
- `"oda"`: optimal damping, an exact line search at `T = 0` and free-energy
  backtracking at `T > 0`.
- `"rdiis"` (Fock) and `"anderson"` (density): full-residual schemes with a
  free-energy safeguard. RDIIS is the recommended default at finite
  temperature; the native backend adds `preconditioner="hartree"` for
  problems where the Hartree term makes the residual stiff.

It stops when the density change is below `density_tol` and the commutator
`[F, P]` below `comm_tol`. {class}`~hartreefock.SCFResult` has the density,
Fock matrix, energy, chemical potential, iteration count, convergence flag,
a message and the per-iteration `history`.

## Direct minimization

{class}`~hartreefock.SolverConfig` picks the optimizer:

- `"cg"`: Riemannian conjugate gradients on the manifold of densities.
- `"newton"`: a truncated-CG trust region on the exact Hessian; quadratic
  convergence near a solution, needs double precision.
- `"auto"` (native only): CG first, Newton once the iterate is close.

With `tol_grad > 0` the stationarity norm is the stopping test; otherwise CG
stops when the free energy improves by less than `tol_E` over
`plateau_window` iterations. Compare `result.energy`, which is recomputed at
the end, rather than the tail of `history["E"]`.

## Stability

A converged residual means a stationary point, which can be a saddle. Symmetric
starts often converge to symmetric saddles when a broken-symmetry state is
lower, as in the [quickstart](../get-started/quickstart.md).
{func}`~hartreefock.stability_analysis` returns the lowest eigenvalue of the
free-energy Hessian at a density, and the direction that goes with it;
`stable` is `True`, `False`, or `None` when the test is inconclusive. It needs
`T > 0`.

{func}`~hartreefock.solve_minimum` puts the two together: it solves with
RDIIS, runs the test, and when it finds negative curvature takes a step along
that direction that lowers the free energy, then solves again, up to
`max_escapes` times. Both are local tests in the basis you give; compare
energies across starts to compare phases.

## Continuation and deflation

{func}`~hartreefock.solve_continuation` solves a coarse kernel, resamples
the result to the fine mesh, and solves there; each stage runs SCF or direct
minimization according to its config, so you can mix them.

{func}`~hartreefock.solve_deflated` collects up to `n_solutions` distinct
solutions by adding a repulsive bias around each one found so far and
re-solving with Newton; `result.best` is the lowest.

## Symmetries

`block_sizes` in either config tells the eigensolver that the Fock matrix is
block diagonal. `project_fn` averages every density and Fock matrix over a
symmetry group, which keeps a solve in a symmetry sector;
{func}`hartreefock.symmetry.make_project_fn` builds one from unitary,
spatial and time-reversal symmetries.
