"""Installed-wheel smoke test used by cibuildwheel.

Runs an actual native solve: ``import hartreefock`` alone succeeds even when
the compiled extension fails to load.
"""

from __future__ import annotations

import numpy as np

import hartreefock as hf

weights = np.ones((1, 1), dtype=np.float64)
h = np.array([[[[-1.0, 0.0], [0.0, 1.0]]]], dtype=np.complex128)
kernel = hf.HartreeFockKernel(weights, h, np.ones((1, 1, 1, 1)), T=0.2)
result = hf.solve_scf(kernel, np.zeros_like(h), 1.0, config=hf.SCFConfig(max_iter=4, mixing=1.0), backend="native")

assert result.density.shape == h.shape
assert np.isfinite(float(result.energy))
print("wheel native smoke test passed")
