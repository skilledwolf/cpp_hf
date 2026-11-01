# Backends

| | `backend="native"` | `backend="jax"` |
| --- | --- | --- |
| Implementation | C++ on FFTW and Eigen, through pybind11 | jitted JAX |
| Precision | double | the dtype of the kernel's arrays |
| Extras | `optimizer="auto"`, `preconditioner="hartree"` | differentiable, runs on GPU |
| Threads | OpenMP and threaded FFTW; releases the GIL | JAX's own |

Both solve the same equations from the same kernel and agree to solver
tolerance; `tests/test_backend_parity.py` checks them against each other. `native` is the
default. {func}`~hartreefock.available_backends` lists the backends that
import, and {func}`~hartreefock.get_backend` returns one as a module.

For JAX in double precision, set `jax.config.update("jax_enable_x64", True)`
before you build the kernel.
