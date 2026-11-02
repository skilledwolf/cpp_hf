"""Installed-wheel smoke test used by cibuildwheel.

``import hartreefock`` succeeds without the compiled extension, so check that
the C++ kernels registered with JAX and run one solve through them.
"""

from __future__ import annotations

import numpy as np

import hartreefock as hf
from hartreefock._jax import ffi

assert all(ffi.available(name) for name in (ffi.EIGH, ffi.EIGVALSH, ffi.MATMUL, ffi.SUPERLATTICE_FOCK))

weights = np.ones((1, 1), dtype=np.float64)
h = np.array([[[[-1.0, 0.0], [0.0, 1.0]]]], dtype=np.complex128)
kernel = hf.HartreeFockKernel(weights, h, np.ones((1, 1, 1, 1)), T=0.2)
result = hf.solve_scf(kernel, np.zeros_like(h), 1.0, config=hf.SCFConfig(max_iter=4, mixing=1.0))

assert result.density.shape == h.shape
assert np.isfinite(float(result.energy))
print("wheel smoke test passed")
