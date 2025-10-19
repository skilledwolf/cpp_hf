"""
Python package facade for the cpp_hf extension.

This package re-exports the symbols from the compiled extension module
that is installed as a submodule `cpp_hf.cpp_hf` inside this package.
"""

# Re-export the pybind11 functions at the package level
from .cpp_hf import hartreefock_iteration_cpp#, prof_dump  # type: ignore

__all__ = [
    "hartreefock_iteration_cpp",
    # "prof_dump",
]

