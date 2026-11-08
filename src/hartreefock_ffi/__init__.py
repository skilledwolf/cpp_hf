"""OpenMP C++ kernels for contimod's Hartree-Fock engine, registered with XLA through ``jax.ffi``.

``contimod.hf`` registers them on first use when this package is installed
(``CONTIMOD_FFI=0`` turns them off); without it the engine runs plain JAX.
:mod:`hartreefock_ffi._native` exposes ``xla_ffi_targets()``, the batched
complex ``eigh``, ``eigvalsh`` and ``matmul`` and the superlattice Fock
build, and ``xla_ffi_jaxlib_version()``, the jaxlib they were compiled against.
"""
