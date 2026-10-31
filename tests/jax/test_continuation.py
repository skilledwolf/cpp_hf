"""Tests for the coarse-to-fine multigrid driver."""

from __future__ import annotations

import numpy as np
import pytest

import jax.numpy as jnp

from hartreefock.backends.jax import (
    ContinuationResult,
    HartreeFockKernel,
    SCFConfig,
    SCFResult,
    SolverConfig,
    SolveResult,
    resample_kgrid,
    solve,
    solve_continuation,
    solve_scf,
)


# ---------------------------------------------------------------------------
# Test kernels
# ---------------------------------------------------------------------------


def _two_band_kernel(nk: int, *, T: float = 0.2, exchange: float = 0.25, dtype=jnp.complex64):
    real_dtype = jnp.finfo(dtype).dtype
    weights = jnp.ones((nk, nk), dtype=real_dtype)
    hamiltonian = np.zeros((nk, nk, 2, 2), dtype=dtype)
    hamiltonian[..., 0, 0] = -0.5
    hamiltonian[..., 1, 1] = 0.5
    coulomb_q = jnp.full((nk, nk, 1, 1), exchange, dtype=dtype)
    return HartreeFockKernel(
        weights=weights,
        hamiltonian=jnp.asarray(hamiltonian),
        coulomb_q=coulomb_q,
        T=T,
    )


# ---------------------------------------------------------------------------
# Happy-path agreement with manual coarse-then-fine
# ---------------------------------------------------------------------------


class TestContinuationMatchesManualDriver:
    def test_direct_minimization_stages(self):
        nk_c, nk_f = 2, 4
        kc = _two_band_kernel(nk_c)
        kf = _two_band_kernel(nk_f)
        P0_c = jnp.zeros_like(kc.h)
        cfg_c = SolverConfig(max_iter=50, tol_E=1e-8)
        cfg_f = SolverConfig(max_iter=50, tol_E=1e-8)

        combined = solve_continuation(
            kc, kf, P0_c, 1.0, 1.0,
            coarse_config=cfg_c, fine_config=cfg_f,
        )

        expected_coarse = solve(kc, P0_c, 1.0, config=cfg_c)
        P0_f_expected = resample_kgrid(expected_coarse.density, nk_f, method="linear")
        P0_f_expected = 0.5 * (
            P0_f_expected + jnp.conj(jnp.swapaxes(P0_f_expected, -1, -2))
        )
        expected_fine = solve(kf, P0_f_expected, 1.0, config=cfg_f)

        assert isinstance(combined, ContinuationResult)
        assert isinstance(combined.coarse, SolveResult)
        assert isinstance(combined.fine, SolveResult)
        np.testing.assert_allclose(
            np.asarray(combined.coarse.density), np.asarray(expected_coarse.density),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(combined.fine.density), np.asarray(expected_fine.density),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(combined.P0_fine), np.asarray(P0_f_expected), atol=1e-6,
        )

    def test_scf_stages(self):
        nk_c, nk_f = 2, 4
        kc = _two_band_kernel(nk_c)
        kf = _two_band_kernel(nk_f)
        P0_c = jnp.zeros_like(kc.h)
        cfg_c = SCFConfig(max_iter=40, density_tol=1e-6, comm_tol=1e-5, mixing=0.5)
        cfg_f = SCFConfig(max_iter=40, density_tol=1e-6, comm_tol=1e-5, mixing=0.5)

        combined = solve_continuation(
            kc, kf, P0_c, 1.0, 1.0,
            coarse_config=cfg_c, fine_config=cfg_f,
        )

        assert isinstance(combined.coarse, SCFResult)
        assert isinstance(combined.fine, SCFResult)
        assert combined.fine.converged
        assert combined.P0_fine.shape == kf.h.shape

    def test_mixed_scf_coarse_dm_fine(self):
        kc = _two_band_kernel(2)
        kf = _two_band_kernel(4)
        P0_c = jnp.zeros_like(kc.h)

        combined = solve_continuation(
            kc, kf, P0_c, 1.0, 1.0,
            coarse_config=SCFConfig(max_iter=30, mixing=0.5),
            fine_config=SolverConfig(max_iter=30, tol_E=1e-8),
        )
        assert isinstance(combined.coarse, SCFResult)
        assert isinstance(combined.fine, SolveResult)

    def test_default_configs_use_direct_minimization(self):
        kc = _two_band_kernel(2)
        kf = _two_band_kernel(4)
        P0_c = jnp.zeros_like(kc.h)

        combined = solve_continuation(kc, kf, P0_c, 1.0, 1.0)
        assert isinstance(combined.coarse, SolveResult)
        assert isinstance(combined.fine, SolveResult)


# ---------------------------------------------------------------------------
# Seeding benefit: continuation should converge the fine solve faster than
# starting from a cold zero seed when the coarse answer is a decent guess.
# ---------------------------------------------------------------------------


class TestContinuationConvergesToSameFixedPoint:
    def test_fine_solve_matches_cold_start_result(self, enable_x64):
        """Continuation + cold fine solve should land at the same fixed point.

        Run in float64: with ``tol_E=1e-10`` a float32 solve stops on
        representable-update grounds, and two such solves from different
        seeds can legitimately differ by tens of ulp (about 4e-6 here).
        """
        with enable_x64():
            kc = _two_band_kernel(4, exchange=0.6, dtype=jnp.complex128)
            kf = _two_band_kernel(8, exchange=0.6, dtype=jnp.complex128)
            P0_c = jnp.zeros_like(kc.h)
            P0_f_cold = jnp.zeros_like(kf.h)

            cfg = SolverConfig(max_iter=200, tol_E=1e-10)
            cold = solve(kf, P0_f_cold, 1.0, config=cfg)
            continued = solve_continuation(
                kc, kf, P0_c, 1.0, 1.0,
                coarse_config=cfg, fine_config=cfg,
            )

        assert bool(cold.converged)
        assert bool(continued.fine.converged)
        np.testing.assert_allclose(
            float(continued.fine.energy), float(cold.energy),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(continued.fine.density), np.asarray(cold.density),
            atol=1e-5,
        )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestContinuationValidation:
    def test_rejects_mismatched_orbital_dims(self):
        kc = _two_band_kernel(2)
        weights = jnp.ones((4, 4), dtype=jnp.float32)
        hamiltonian = jnp.zeros((4, 4, 3, 3), dtype=jnp.complex64)
        coulomb_q = jnp.ones((4, 4, 1, 1), dtype=jnp.complex64) * 0.1
        kf = HartreeFockKernel(weights, hamiltonian, coulomb_q, T=0.1)

        with pytest.raises(ValueError, match="orbital dimensions"):
            solve_continuation(kc, kf, jnp.zeros_like(kc.h), 1.0, 1.0)

    def test_rejects_mismatched_hartree_flag(self):
        kc = _two_band_kernel(2)
        weights = jnp.ones((4, 4), dtype=jnp.float32)
        hamiltonian = jnp.zeros((4, 4, 2, 2), dtype=jnp.complex64)
        coulomb_q = jnp.ones((4, 4, 1, 1), dtype=jnp.complex64) * 0.1
        hartree_matrix = jnp.eye(2, dtype=jnp.float32)
        ref = jnp.zeros_like(hamiltonian)
        kf = HartreeFockKernel(
            weights, hamiltonian, coulomb_q, T=0.1,
            include_hartree=True, include_exchange=True,
            reference_density=ref, hartree_matrix=hartree_matrix,
        )

        with pytest.raises(ValueError, match="include_hartree"):
            solve_continuation(kc, kf, jnp.zeros_like(kc.h), 1.0, 1.0)

    def test_rejects_wrong_P0_shape(self):
        kc = _two_band_kernel(2)
        kf = _two_band_kernel(4)
        wrong_P0 = jnp.zeros((3, 3, 2, 2), dtype=kc.h.dtype)
        with pytest.raises(ValueError, match="P0_coarse shape"):
            solve_continuation(kc, kf, wrong_P0, 1.0, 1.0)

    def test_rejects_unknown_config_type(self):
        kc = _two_band_kernel(2)
        kf = _two_band_kernel(4)
        with pytest.raises(TypeError, match="coarse_config"):
            solve_continuation(
                kc, kf, jnp.zeros_like(kc.h), 1.0, 1.0,
                coarse_config={"max_iter": 5},  # dict not accepted; must be config
            )
