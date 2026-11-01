# Quickstart

A dilute two-dimensional electron gas with a strong Coulomb repulsion lowers
its exchange energy by polarizing its spin. This page builds that problem,
solves it from a spin-balanced and a polarized start, and asks which solution
is stable. The script is
[`docs/examples/quickstart.py`](https://github.com/skilledwolf/cpp_hf/blob/main/docs/examples/quickstart.py)
and takes a few seconds.

## The mesh and the Hamiltonian

```{literalinclude} ../examples/quickstart.py
:language: python
:start-at: "# A square"
:end-at: h = eps
```

Everything lives on an `nk × nk` mesh: the Hamiltonian `h` has shape
`(nk, nk, nb, nb)`, and `weights` are the quadrature weights, so a k-sum
`Σ_k w(k) f(k)` is an integral. Densities and particle numbers are counted
with the same weights. The units are yours; the solver never assumes any.

## The interaction

```{literalinclude} ../examples/quickstart.py
:language: python
:start-at: "# The interaction"
:end-at: coulomb_q =
```

`V(q)` is given on the grid of momentum transfers, with the same spacing as
the k-mesh and `q = 0` at index `nk // 2`. Shape `(nk, nk, 1, 1)` is the same
interaction between all orbitals; `(nk, nk, nb, nb)` resolves it.

## The kernel

```{literalinclude} ../examples/quickstart.py
:language: python
:start-at: kernel = hf
:end-at: n_electrons =
```

{class}`~hartreefock.HartreeFockKernel` precomputes everything the solvers
need. By default it keeps the exchange term only; Hartree, a reference
density and contact terms are options (see [The kernel](../guides/kernel.md)).

## Solving and checking

```{literalinclude} ../examples/quickstart.py
:language: python
:start-at: "# Two starting"
```

```text
 balanced: E = -0.011321, spin up/down = 0.0100/0.0100, stable = False (lowest curvature -1.5e+00)
polarized: E = -0.016818, spin up/down = 0.0200/0.0000, stable = True (lowest curvature -7.6e-11)
```

Both starts converge, to different solutions. The spin-balanced one is a
saddle point: {func}`~hartreefock.stability_analysis` finds a direction of
negative free-energy curvature. The polarized one is lower in energy and
stable; its lowest curvature is zero within precision, the Goldstone mode of
rotating the spin. A converged residual does not tell these apart, which is
why the check exists; {func}`~hartreefock.solve_minimum` runs it for you and
follows the unstable direction downhill.
