"""The C++ kernels the JAX engine calls through ``jax.ffi`` on CPU.

The extension exposes its XLA FFI handlers through ``xla_ffi_targets()``
(empty when it was built without the jaxlib FFI headers).  They are
registered with XLA once, on first use.  ``HF_JAX_FFI=0`` turns them off, so
the engine falls back to plain JAX.
"""

from __future__ import annotations

import os

import jax

EIGH = "hartreefock_eigh_c128"
EIGVALSH = "hartreefock_eigvalsh_c128"
MATMUL = "hartreefock_matmul_c128"
SUPERLATTICE_FOCK = "hartreefock_superlattice_fock"

_registered: frozenset[str] | None = None


def available(name: str) -> bool:
    """Whether the native kernel ``name`` is registered with XLA for CPU."""
    global _registered
    if _registered is None:
        _registered = frozenset()
        if os.environ.get("HF_JAX_FFI", "1") == "0":
            return False
        try:
            from hartreefock import _native
        except ImportError:
            return False
        targets = dict(_native.xla_ffi_targets())
        if targets and not _same_jaxlib(_native.xla_ffi_jaxlib_version()):
            return False
        for target, capsule in targets.items():
            jax.ffi.register_ffi_target(target, capsule, platform="cpu")
        _registered = frozenset(targets)
    return name in _registered


def _same_jaxlib(built: str) -> bool:
    """Whether the running jaxlib is the one the kernels were compiled against.

    The XLA FFI ABI is not stable across jaxlib releases: a handler built
    against newer headers can crash an older runtime.  On a mismatch the
    engine warns once and runs plain JAX.
    """
    import jaxlib

    if built == jaxlib.__version__:
        return True
    import warnings

    warnings.warn(
        f"hartreefock's native JAX kernels were built against jaxlib {built or '(unknown)'} "
        f"but jaxlib {jaxlib.__version__} is installed; using plain JAX. Rebuild "
        "hartreefock in this environment to enable them.",
        RuntimeWarning,
        stacklevel=3,
    )
    return False


def on_cpu() -> bool:
    """Whether JAX computes on CPU by default (where the FFI kernels run)."""
    return jax.default_backend() == "cpu"
