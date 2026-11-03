# API at a glance

The functions and classes you call, grouped by what you are doing. Each
links to its full entry; the [module reference](python/index) lists
everything public.

## Describe the problem

| Name | What it is |
| --- | --- |
| {class}`~hartreefock.Mesh` | A k-mesh: cell, points, weights, transfer grid, −k map, resampling |
| {class}`~hartreefock.HartreeFockKernel` | Hamiltonian on a mesh, interaction `V(q)`, temperature; Hartree, contact and reference terms; `plane_waves=` for a moiré basis `|k, G, α⟩` |

## Solve

| Name | What it does | Returns |
| --- | --- | --- |
| {func}`~hartreefock.solve` | Roothaan iteration (RDIIS, linear mixing) or direct minimization (CG, Newton), chosen by the config class (`RDIIS`, `Linear`, `CG`, `Newton`) | {class}`~hartreefock.SolveResult` |
| {func}`~hartreefock.solve_deflated` | Several distinct solutions | {class}`~hartreefock.DeflatedResult` |

## Check a solution

| Name | What it does | Returns |
| --- | --- | --- |
| {func}`~hartreefock.stability_analysis` | The lowest free-energy curvature at a density | {class}`~hartreefock.StabilityResult` |
| {func}`~hartreefock.solve_minimum` | Solve, test, step down unstable directions, repeat | {class}`~hartreefock.MinimumResult` |

## Symmetry and the Fock map

| Name | What it does |
| --- | --- |
| {func}`~hartreefock.symmetry.make_project_fn` | A projection onto a symmetry sector, for `project_fn` |

```{toctree}
:hidden:

python/index
```
