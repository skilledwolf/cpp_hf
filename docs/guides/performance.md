---
file_format: mystnb
kernelspec:
  name: python3
---

# Precision and speed

The solvers are jitted JAX, so they run on CPU or GPU and are
differentiable. Importing hartreefock turns on `jax_enable_x64`, and a
solve then runs in the dtype of the kernel's arrays: complex128 from NumPy
defaults, complex64 from single-precision arrays. Single precision halves
the memory and is faster on most GPUs; the Newton optimizer, deflation,
{func}`~hartreefock.stability_analysis` and
{func}`~hartreefock.solve_minimum` need double precision for their default
tolerances.

```{code-cell} ipython3
import time
import numpy as np
import hartreefock as hf

mesh = hf.Mesh.square(kmax=1.2, nk=48)
eps = 0.5 * np.sum(mesh.ks**2, axis=-1)
h = eps[..., None, None] * np.eye(2)


def V(q):
    return 5.0 * 2 * np.pi * np.where(q > 0, np.tanh(10 * q) / np.where(q > 0, q, 1.0), 10.0)


n = 0.02
P0 = (eps < 2 * np.pi * n)[..., None, None] * np.diag([0.7, 0.3])

results = {}
for dtype, tol in ((np.complex64, 1e-5), (np.complex128, 1e-9)):
    kernel = hf.HartreeFockKernel(mesh, h.astype(dtype), V, T=0.005)
    config = hf.SCFConfig(density_tol=tol)
    for attempt in ("first call", "second call"):
        t = time.perf_counter()
        r = hf.solve_scf(kernel, P0.astype(dtype), n, config=config)
        elapsed = time.perf_counter() - t
    results[dtype] = r
    print(f"{dtype.__name__}: {r.n_iter} iterations, E = {float(r.energy):.10f}, {elapsed * 1e3:.0f} ms (second call)")

gap = abs(float(results[np.complex128].energy) - float(results[np.complex64].energy))
print(f"energy difference {gap:.1e}")
```

```{code-cell} ipython3
:tags: [remove-cell]
assert results[np.complex128].converged and gap < 1e-5
```

The first call compiles; later calls with the same shapes reuse the
compiled code.

## C++ kernels on the CPU

XLA on the CPU runs a batch of small eigendecompositions or matrix
products one after another. On the CPU in double precision, hartreefock
therefore swaps four operations for OpenMP-threaded C++ kernels through
`jax.ffi`: the batched Hermitian eigensolver and eigenvalue solver, the
batched complex matrix product, and the superlattice Fock build. The
kernels load only under the jaxlib the extension was compiled against;
under another jaxlib hartreefock warns and runs plain JAX, and rebuilding
it in that environment turns them back on. Set `HF_JAX_FFI=0` to run plain
JAX regardless.

A source build compiles for the CPU it runs on (`-march=native`), which
makes the complex matrix products about 6× faster than a portable build on
an AVX-512 machine.

Measured on 2026-11-02 on a 4-core AVX-512 Xeon cloud machine without a
GPU (JAX 0.12), with contimod's bilayer and twisted-bilayer problems, as
time relative to the C++ solver loop hartreefock 3.0 removed:

| | With the kernels | Plain XLA |
| --- | --- | --- |
| One SCF-DIIS iteration, bilayer nk = 48 to 96 | 1.0× | 1.4–1.9× |
| One SCF-DIIS iteration, twisted bilayer nk = 9 | 1.0× | 2.7× |
| First call on the twisted-bilayer kernel (compilation) | 6 s | 22 s |

Set `OPENBLAS_NUM_THREADS=1` when running on the CPU: without it, BLAS
threads compete with JAX's own and the solve runs up to 1.7× slower.
