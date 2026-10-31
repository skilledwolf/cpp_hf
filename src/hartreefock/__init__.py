"""Hartree-Fock solvers on 2D k-meshes with native (C++) and JAX backends."""

from importlib import metadata

try:  # pragma: no cover - metadata only
    __version__ = metadata.version("hartreefock")
except metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"
