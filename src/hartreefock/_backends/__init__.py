"""Compute backends: ``native`` (C++ extension) and ``jax`` (optional).

Each backend module provides ``solve``, ``solve_scf``, ``build_fock``,
``fock_operators`` and ``resample_kgrid`` for the shared :class:`hartreefock.HartreeFockKernel`.
"""

from __future__ import annotations

import importlib
from types import ModuleType

BACKENDS = ("native", "jax")


def get_backend(name: str) -> ModuleType:
    """Import and return a backend module.

    Parameters
    ----------
    name : {"native", "jax"}
        The backend.

    Returns
    -------
    types.ModuleType
        The backend, with the solver entry points the top-level functions call.

    Raises
    ------
    RuntimeError
        For ``"native"`` when the compiled extension did not build.
    """
    if name not in BACKENDS:
        raise ValueError(f"Unknown backend {name!r}; choose one of {BACKENDS}.")
    module = importlib.import_module(f"{__name__}.{name}")
    if name == "native":
        from .native._compat import native_required

        native_required()
    return module


def available_backends() -> tuple[str, ...]:
    """List the backends whose runtime dependencies import.

    Returns
    -------
    tuple of str
        A subset of ``("native", "jax")``.
    """
    found = []
    for name in BACKENDS:
        try:
            get_backend(name)
        except (ImportError, RuntimeError):
            continue
        found.append(name)
    return tuple(found)
