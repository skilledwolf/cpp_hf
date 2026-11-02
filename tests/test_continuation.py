"""Coarse-to-fine continuation driver."""

from __future__ import annotations

import numpy as np
import pytest
from _meshes import mesh_for

import hartreefock as hf
from hartreefock._jax import resample_kgrid


def _kernel(nk, *, exchange=0.6, nb=2):
    h = np.zeros((nk, nk, nb, nb), dtype=np.complex128)
    h[..., 0, 0], h[..., 1, 1] = -0.5, 0.5
    return hf.HartreeFockKernel(mesh_for(np.ones((nk, nk))), h, np.full((nk, nk, 1, 1), exchange), T=0.2)


def test_continuation_matches_manual_coarse_then_fine():
    kc, kf = _kernel(2), _kernel(4)
    P0 = np.zeros_like(kc.h)
    cfg = hf.SolverConfig(method="cg", max_iter=200, tol_E=1e-10)

    combined = hf.solve(kf, P0, 1.0, config=cfg, coarse=kc)

    coarse = hf.solve(kc, P0, 1.0, config=cfg)
    seed = np.asarray(resample_kgrid(coarse.density, 4))
    seed = 0.5 * (seed + seed.conj().swapaxes(-1, -2))
    fine = hf.solve(kf, seed, 1.0, config=cfg)
    cold = hf.solve(kf, np.zeros_like(kf.h), 1.0, config=cfg)

    assert isinstance(combined.coarse, hf.SolveResult) and combined.coarse.coarse is None
    np.testing.assert_allclose(combined.coarse.density, coarse.density, atol=1e-12)
    np.testing.assert_allclose(combined.density, fine.density, atol=1e-10)
    assert combined.converged
    np.testing.assert_allclose(combined.energy, cold.energy, atol=1e-6)


def test_continuation_validates_inputs():
    kc = _kernel(2)
    mismatched = hf.HartreeFockKernel(mesh_for(np.ones((4, 4))), np.zeros((4, 4, 3, 3), complex), np.full((4, 4, 1, 1), 0.1), T=0.1)
    with pytest.raises(ValueError, match="orbital dimensions"):
        hf.solve(mismatched, np.zeros_like(kc.h), 1.0, coarse=kc)
    hartree = hf.HartreeFockKernel(mesh_for(np.ones((4, 4))), np.zeros((4, 4, 2, 2), complex), np.full((4, 4, 1, 1), 0.1), hartree=np.eye(2), T=0.1, reference=np.zeros((4, 4, 2, 2), complex))
    with pytest.raises(ValueError, match="include_hartree"):
        hf.solve(hartree, np.zeros_like(kc.h), 1.0, coarse=kc)
