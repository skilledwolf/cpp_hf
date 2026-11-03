# Pitfalls

Things that are easy to get wrong, each with what goes wrong and what to
do. Each one was hit while writing these pages.

## The mesh must reach past twice the Fermi momentum

The dense kernel's exchange is a periodic convolution on the mesh, so a
transfer longer than the mesh wraps around to a short one. For the states
that matter to be right, `kmax` must exceed $2k_F$ of the most occupied
band; for a spin-polarized state that is $2\sqrt2\,k_F$ of the balanced
one. States near the edge of the mesh always see the wrap ([tutorial
step 1](../tutorial/01-exchange.md)). A plane-wave kernel's
convolution is zero-padded and does not wrap.

## The q = 0 cell of a singular interaction

A tabulated `V` at $q = 0$ must be finite. For $1/q$ the cell average
$4\ln(1 + \sqrt2)/\Delta k$ gives exchange energies that converge as
$1/n_k$; extrapolate them ([tutorial step 1](../tutorial/01-exchange.md)).
The same cell couples each k-point to itself, which at low temperature
makes {func}`~hartreefock.stability_analysis` report mesh modes as
unstable ([Solvers](../guides/solvers.md)).

## Single-precision kernels

The solvers compute in the dtype of the kernel's arrays, so a kernel built
from float32 or complex64 arrays solves in single precision.
{func}`~hartreefock.stability_analysis` and
{func}`~hartreefock.solve_minimum` refuse their default tolerances on
such a kernel, and `hf.Newton` needs double precision
([Precision and speed](../guides/performance.md)).

## A converged solve can be a saddle

A small residual means a stationary point. A symmetric start converges to
the symmetric solution even when a broken-symmetry one is lower. Run
{func}`~hartreefock.stability_analysis` on the result, or solve with
{func}`~hartreefock.solve_minimum` ([quickstart](../get-started/quickstart.md),
[tutorial step 3](../tutorial/03-stoner.md)).

## Deflation counts rotated copies

With a continuous symmetry, such as spin rotation, a broken-symmetry
solution is a family, and {func}`~hartreefock.solve_deflated` returns
several members of it as distinct. Compare a symmetry-invariant order
parameter, such as the length of the magnetization
([Solvers](../guides/solvers.md)).
