# API at a glance

The functions and classes you call, grouped by what you are doing. Each
links to its full entry; the [module reference](python/index) lists
everything public.

## Describe the problem

| Name | What it is |
| --- | --- |
| {class}`~hartreefock.HartreeFockKernel` | Hamiltonian on a k-mesh, interaction `V(q)`, weights, temperature; Hartree, contact and reference terms |
| {class}`~hartreefock.SuperlatticeHartreeFockKernel` | The same for a moiré plane-wave basis `|k, G, α⟩` |

## Solve

| Name | What it does | Returns |
| --- | --- | --- |
| {func}`~hartreefock.solve_scf` | Roothaan iteration with mixing, DIIS, ODA, RDIIS or Anderson, chosen by {class}`~hartreefock.SCFConfig` | {class}`~hartreefock.SCFResult` |
| {func}`~hartreefock.solve` | Direct minimization, CG or Newton, chosen by {class}`~hartreefock.SolverConfig` | {class}`~hartreefock.SolveResult` |
| {func}`~hartreefock.solve_continuation` | A coarse mesh, then the fine one from its result | {class}`~hartreefock.ContinuationResult` |
| {func}`~hartreefock.solve_deflated` | Several distinct solutions | {class}`~hartreefock.DeflatedResult` |

## Check a solution

| Name | What it does | Returns |
| --- | --- | --- |
| {func}`~hartreefock.stability_analysis` | The lowest free-energy curvature at a density | {class}`~hartreefock.StabilityResult` |
| {func}`~hartreefock.solve_minimum` | Solve, test, step down unstable directions, repeat | {class}`~hartreefock.MinimumResult` |

## Symmetry and backends

| Name | What it does |
| --- | --- |
| {func}`~hartreefock.symmetry.make_project_fn` | A projection onto a symmetry sector, for `project_fn` |
| {func}`~hartreefock.available_backends` | The backends that import here |
| {func}`~hartreefock.get_backend` | One backend as a module |

```{toctree}
:hidden:

python/index
```
