---
file_format: mystnb
kernelspec:
  name: python3
---

# Backends

Both backends solve the same equations from the same kernel; `backend=`
on every solver picks one.

| | `backend="native"` | `backend="jax"` |
| --- | --- | --- |
| Implementation | C++ on FFTW and Eigen, through pybind11 | jitted JAX |
| Precision | double | the dtype of the kernel's arrays |
| Extras | `optimizer="auto"`, `preconditioner="hartree"` | differentiable, runs on GPU |
| Threads | OpenMP and threaded FFTW; releases the GIL | JAX's own |

`native` is the default. {func}`~hartreefock.available_backends` lists the
backends that import, and {func}`~hartreefock.get_backend` returns one as a
module.

## The same answer

For JAX in double precision, enable `jax_enable_x64` before building the
kernel. Then the two backends agree to solver tolerance:

```{code-cell} ipython3
import jax

jax.config.update("jax_enable_x64", True)

import time
import numpy as np
import hartreefock as hf

print("available:", hf.available_backends())

nk, kmax = 48, 1.2
k = np.linspace(-kmax, kmax, nk, endpoint=False) + kmax / nk
kx, ky = np.meshgrid(k, k, indexing="ij")
weights = np.full((nk, nk), (2 * kmax / nk) ** 2 / (2 * np.pi) ** 2)
eps = 0.5 * (kx**2 + ky**2)
dk = k[1] - k[0]
q1 = dk * (np.arange(nk) - nk // 2)
q = np.hypot(*np.meshgrid(q1, q1, indexing="ij"))
V = 5.0 * 2 * np.pi * np.where(q > 0, np.tanh(10 * q) / np.where(q > 0, q, 1.0), 10.0)
kernel = hf.HartreeFockKernel(weights, eps[..., None, None] * np.eye(2), V[..., None, None], T=0.005)
n = 0.02
P0 = (eps < 2 * np.pi * n)[..., None, None] * np.diag([0.7, 0.3])

results = {}
for backend in ("native", "jax"):
    for attempt in ("first call", "second call"):
        t = time.perf_counter()
        r = hf.solve_scf(kernel, P0, n, config=hf.SCFConfig(acceleration="rdiis"), backend=backend)
        elapsed = time.perf_counter() - t
    results[backend] = r
    print(f"{backend:>6}: {r.n_iter} iterations, E = {float(r.energy):.12f}, {elapsed * 1e3:.0f} ms (second call)")

gap = abs(float(results["native"].energy) - float(results["jax"].energy))
dP = np.abs(np.asarray(results["native"].density) - np.asarray(results["jax"].density)).max()
print(f"energy difference {gap:.1e}, largest density difference {dP:.1e}")
```

```{code-cell} ipython3
:tags: [remove-cell]
assert gap < 1e-9 and dP < 1e-6
```

The first JAX call compiles; later calls with the same shapes reuse the
compiled code. `tests/test_backend_parity.py` runs this comparison on
every problem class.

## Speed

Measured on 2026-11-01 on a 4-core Xeon cloud machine without a GPU
(JAX 0.12, hartreefock at commit `9f82a69`), with contimod's bilayer and
twisted-bilayer problems:

| | JAX time relative to native |
| --- | --- |
| One SCF-DIIS iteration, 4 cores | 1.4–1.9× |
| One SCF-DIIS iteration, 1 core | 0.9–1.4× |
| Newton iteration | 3–6× |
| Time to solution | 1.7–4.6× |
| First call on a superlattice kernel (compilation) | 6–43 s on 4 cores |

The gap on several cores is the batched eigensolver: XLA on the CPU runs
the per-k diagonalizations one after another, while the native loop
spreads them over threads. Of the Fock builds, the dense one is faster in
JAX and the superlattice one 2–3× slower. On a problem as small as the one
above, JAX's per-call overhead dominates instead.
Set `OPENBLAS_NUM_THREADS=1` when running JAX on the CPU: without it,
BLAS threads compete with JAX's own and JAX runs up to 1.7× slower.
