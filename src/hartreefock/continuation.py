"""Coarse-to-fine k-mesh continuation: the seed of ``hf.solve(..., coarse=)``."""

from __future__ import annotations

import numpy as np

from . import _jax
from .kernel import HartreeFockKernel
from .superlattice import upsample_density


def resample_to(coarse: HartreeFockKernel, fine: HartreeFockKernel, density) -> np.ndarray:
    """``density`` on the mesh of ``coarse`` resampled to the mesh of ``fine``."""
    if coarse.h.shape[-2:] != fine.h.shape[-2:]:
        raise ValueError(
            "coarse and fine kernels must share orbital dimensions "
            f"(..., nb, nb); got {tuple(coarse.h.shape[-2:])} vs {tuple(fine.h.shape[-2:])}."
        )
    for flag in ("include_hartree", "include_exchange"):
        if getattr(coarse, flag) != getattr(fine, flag):
            raise ValueError(f"coarse and fine kernels must agree on {flag}.")
    if density is None:
        raise ValueError("the coarse stage must return its density (return_density=True).")
    if fine.superlattice:
        return upsample_density(density, coarse.mesh, fine.mesh)
    P = np.asarray(_jax.resample_kgrid(density, int(fine.h.shape[0])))
    return 0.5 * (P + P.conj().swapaxes(-1, -2))
