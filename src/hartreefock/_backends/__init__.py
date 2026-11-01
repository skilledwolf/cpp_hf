"""Compute backends: ``native`` (C++ extension) and ``jax`` (optional).

Each backend module provides ``solve``, ``solve_scf``, ``build_fock`` and
``resample_kgrid`` for the shared :class:`hartreefock.HartreeFockKernel`.
"""

from __future__ import annotations

import importlib
from types import ModuleType

BACKENDS = ("native", "jax")


def get_backend(name: str) -> ModuleType:
    """Import and return backend ``name`` (``"native"`` or ``"jax"``)."""
    if name not in BACKENDS:
        raise ValueError(f"Unknown backend {name!r}; choose one of {BACKENDS}.")
    module = importlib.import_module(f"{__name__}.{name}")
    if name == "native":
        from .native._compat import native_required

        native_required()
    return module


def available_backends() -> tuple[str, ...]:
    """Backends whose runtime dependencies import."""
    found = []
    for name in BACKENDS:
        try:
            get_backend(name)
        except (ImportError, RuntimeError):
            continue
        found.append(name)
    return tuple(found)
