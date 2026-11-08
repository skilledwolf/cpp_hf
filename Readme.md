# hartreefock-ffi: native CPU kernels for contimod.hf

`hartreefock-ffi` is the optional native backend of contimod's Hartree-Fock
engine, `contimod.hf`. It is one compiled module, `hartreefock_ffi._native`,
that registers four `jax.ffi` targets:

* `hartreefock_eigh_c128` and `hartreefock_eigvalsh_c128`: threaded batched
  Hermitian eigensolvers;
* `hartreefock_matmul_c128`: batched complex matrix products;
* `hartreefock_superlattice_fock`: the streamed plane-wave (superlattice)
  exchange.

The kernels are OpenMP C++ (FFTW + Eigen). contimod picks them up when the
module imports and falls back to plain JAX otherwise, so the results do not
depend on it, only the CPU speed. The Python engine, its documentation and
its history up to 3.0.0 live in [contimod](https://github.com/skilledwolf/contimod)
(see `CHANGELOG.md` here for the engine's earlier releases).

## Install

Requires JAX, FFTW (double precision: `fftw3`), Eigen and a C++17 compiler.

```bash
# macOS
brew install fftw eigen
# Linux
sudo apt install libfftw3-dev libeigen3-dev

pip install scikit-build-core pybind11 setuptools_scm
pip install --no-build-isolation -e .   # in this checkout, next to contimod
```

`--no-build-isolation` builds against the installed jaxlib's FFI headers; the
wheel requires exactly that jaxlib (an isolated build would pull the newest
one). Builds are portable; `-C cmake.define.HF_NATIVE_ARCH=ON` compiles for
this CPU (`-march=native`). `contimod.hf.native_kernels()` lists the kernels
in use. `ci/jaxlib_pin.py` writes the pin into the metadata;
`python ci/smoke_test.py` checks that the four targets register and that
jaxlib matches the build.

## Tests

`pytest tests` checks the module registers its targets. The numerical tests
of the kernels run in contimod (`tests/hf`), against the plain-JAX paths.
