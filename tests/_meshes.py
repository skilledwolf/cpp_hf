"""Meshes for the synthetic test models, which state their k weights directly."""

import numpy as np

from hartreefock import Mesh


def mesh_for(weights) -> Mesh:
    """The square mesh whose uniform point weight is that of ``weights`` (shape ``(nk, nk)``)."""
    w = np.asarray(weights, dtype=float)
    nk = w.shape[0]
    if w.shape != (nk, nk) or not np.allclose(w, w.flat[0]):
        raise ValueError(f"weights must be uniform and square, got {w.shape}")
    # weight = |det cell| / nk² / (2π)²
    return Mesh(2 * np.pi * nk * np.sqrt(w.flat[0]) * np.eye(2), nk)
