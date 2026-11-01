# Guides

Each guide covers one part of hartreefock: the equations, what to call,
what comes back, and a check of the result. Every page is an executed
notebook, so its numbers come from the code shown, and the build fails
when a check stops holding.

| Guide | Covers | Check |
| --- | --- | --- |
| [The kernel](kernel.md) | arrays and conventions, Hartree, contact terms, reference densities, embedding | embedded and standard Hartree-Fock agree to $10^{-8}$ |
| [Solvers](solvers.md) | SCF accelerations, direct minimization, stability, continuation, deflation | eight methods, one energy |
| [Superlattices](superlattice.md) | the plane-wave kernel and off-mesh self-energies | the empty lattice reproduces the electron gas |
| [Backends](backends.md) | native and JAX, precision, speed | the two agree to $10^{-15}$ |

```{toctree}
:hidden:

kernel
solvers
superlattice
backends
```
