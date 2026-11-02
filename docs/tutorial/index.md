# Tutorial

Three steps on the two-dimensional electron gas, each one page, each
building on the last. Every page names the physics it computes and the
parts of hartreefock it uses, runs top to bottom when this site is built,
and ends with a **Check**: the computed number next to a closed form. A
failing check fails the build.

| Step | Physics | You will use | Check |
| --- | --- | --- | --- |
| [1. Exchange in the electron gas](01-exchange.md) | the exchange self-energy and energy of the Coulomb gas | `HartreeFockKernel`, `solve` | $-8\sqrt2/3\pi r_s$ Ry to $10^{-4}$ |
| [2. Bloch ferromagnetism](02-bloch.md) | paramagnet against ferromagnet, the first-order transition | two starts, mesh extrapolation | $r_s^\ast = 3\pi/(16 - 8\sqrt2)$ |
| [3. The Stoner instability](03-stoner.md) | a contact repulsion, the free-energy curvature, leaving a saddle | `contact_terms`, `stability_analysis`, `solve_minimum` | $U_c = 1/\chi_0$; $m(U)$ of the continuum |

Run the pages yourself: each is a Markdown notebook that
[Jupytext](https://jupytext.readthedocs.io) opens in Jupyter, or copy the
cells into a script. They need NumPy, SciPy and Matplotlib next to
hartreefock.

```{toctree}
:hidden:
:maxdepth: 1

01-exchange
02-bloch
03-stoner
```
