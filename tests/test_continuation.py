"""Coarse-to-fine continuation driver."""

from __future__ import annotations

import numpy as np
import pytest

import hartreefock as hf
from hartreefock._jax import resample_kgrid



def _kernel(nk, *, exchange=0.6, nb=2):
    h = np.zeros((nk, nk, nb, nb), dtype=np.complex128)
    h[..., 0, 0], h[..., 1, 1] = -0.5, 0.5
    return hf.HartreeFockKernel(
        np.ones((nk, nk)), h, np.full((nk, nk, 1, 1), exchange), T=0.2
    )


def test_continuation_matches_manual_coarse_then_fine():
    kc, kf = _kernel(2), _kernel(4)
    P0 = np.zeros_like(kc.h)
    cfg = hf.SolverConfig(max_iter=200, tol_E=1e-10)

    combined = hf.solve_continuation(
        kc, kf, P0, 1.0, 1.0, coarse_config=cfg, fine_config=cfg
    )

    coarse = hf.solve(kc, P0, 1.0, config=cfg)
    seed = np.asarray(resample_kgrid(coarse.density, 4))
    seed = 0.5 * (seed + seed.conj().swapaxes(-1, -2))
    fine = hf.solve(kf, seed, 1.0, config=cfg)
    cold = hf.solve(kf, np.zeros_like(kf.h), 1.0, config=cfg)

    assert isinstance(combined.fine, hf.SolveResult)
    np.testing.assert_allclose(np.asarray(combined.P0_fine), seed, atol=1e-10)
    np.testing.assert_allclose(np.asarray(combined.fine.density), np.asarray(fine.density), atol=1e-10)
    assert bool(combined.fine.converged)
    np.testing.assert_allclose(float(combined.fine.energy), float(cold.energy), atol=1e-6)


def test_stages_can_mix_scf_and_minimization():
    kc, kf = _kernel(2), _kernel(4)
    res = hf.solve_continuation(
        kc, kf, np.zeros_like(kc.h), 1.0, 1.0,
        coarse_config=hf.SCFConfig(max_iter=40, mixing=0.5),
        fine_config=hf.SolverConfig(max_iter=50),
        retain_P0_fine=False,
    )
    assert isinstance(res.coarse, hf.SCFResult)
    assert isinstance(res.fine, hf.SolveResult)
    assert res.P0_fine is None


def test_continuation_validates_inputs():
    kc = _kernel(2)
    mismatched = hf.HartreeFockKernel(
        np.ones((4, 4)), np.zeros((4, 4, 3, 3), complex), np.full((4, 4, 1, 1), 0.1), T=0.1
    )
    with pytest.raises(ValueError, match="orbital dimensions"):
        hf.solve_continuation(kc, mismatched, np.zeros_like(kc.h), 1.0, 1.0)
    hartree = hf.HartreeFockKernel(
        np.ones((4, 4)), np.zeros((4, 4, 2, 2), complex), np.full((4, 4, 1, 1), 0.1), T=0.1,
        include_hartree=True, reference_density=np.zeros((4, 4, 2, 2), complex),
        hartree_matrix=np.eye(2),
    )
    with pytest.raises(ValueError, match="include_hartree"):
        hf.solve_continuation(kc, hartree, np.zeros_like(kc.h), 1.0, 1.0)
    with pytest.raises(ValueError, match="P0_coarse shape"):
        hf.solve_continuation(kc, _kernel(4), np.zeros((3, 3, 2, 2), complex), 1.0, 1.0)
    with pytest.raises(TypeError, match="coarse_config"):
        hf.solve_continuation(
            kc, _kernel(4), np.zeros_like(kc.h), 1.0, 1.0, coarse_config={"max_iter": 5}
        )
