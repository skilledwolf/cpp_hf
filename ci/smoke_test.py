"""Installed-wheel smoke test used by cibuildwheel.

This intentionally exercises the compiled extension; importing ``cpp_hf``
alone is insufficient because the Python package degrades gracefully when the
native module cannot be loaded.
"""

from __future__ import annotations

import numpy as np

import cpp_hf
from cpp_hf._compat import HAVE_NATIVE


assert HAVE_NATIVE, "cpp_hf._native failed to load from the installed wheel"

weights = np.ones((1, 1), dtype=np.float64)
h = np.array([[[[-1.0, 0.0], [0.0, 1.0]]]], dtype=np.complex128)
interaction = np.ones((1, 1, 1, 1), dtype=np.float64)
kernel = cpp_hf.HartreeFockKernel(weights, h, interaction, T=0.2)
result = cpp_hf.solve_scf(
    kernel,
    np.zeros_like(h),
    1.0,
    config=cpp_hf.SCFConfig(max_iter=4, mixing=1.0),
)

assert result.density_matrix is not None
assert result.density_matrix.shape == h.shape
assert np.isfinite(float(result.energy))
print("wheel native smoke test passed")
