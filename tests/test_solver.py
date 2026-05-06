"""Tests for the direct-minimization solver (port of jax_hf.tests.test_solver)."""

import numpy as np
import pytest

from cpp_hf import (
    HartreeFockKernel,
    SCFConfig,
    SolverConfig,
    SolveResult,
    solve,
    solve_scf,
)


def _comm_rms(F, P, weights_2d):
    comm = F @ P - P @ F
    sq = np.abs(comm) ** 2
    per_k = np.sum(sq, axis=(-2, -1))
    weight_sum = np.sum(weights_2d)
    return float(np.sqrt(np.sum(weights_2d * per_k) / max(float(weight_sum), 1e-30)))


def _make_two_band_kernel(nk=1, T=0.2, exchange_strength=0.25):
    weights = np.ones((nk, nk), dtype=np.float32)
    hamiltonian = np.zeros((nk, nk, 2, 2), dtype=np.complex64)
    hamiltonian[..., 0, 0] = -0.5
    hamiltonian[..., 1, 1] = 0.5
    coulomb_q = np.full((nk, nk, 1, 1), exchange_strength, dtype=np.complex64)
    return HartreeFockKernel(
        weights=weights,
        hamiltonian=hamiltonian,
        coulomb_q=coulomb_q,
        T=T,
    )


def _solve_problem(kernel, n_electrons=1.0, config=None):
    if config is None:
        config = SolverConfig(max_iter=100, tol_E=1e-8)
    P0 = np.zeros_like(kernel.h)
    return solve(kernel, P0, n_electrons, config=config)


class TestBasicConvergence:
    def test_converges_on_tiny_model(self):
        kernel = _make_two_band_kernel()
        result = _solve_problem(kernel)

        assert int(result.n_iter) <= 100
        assert bool(result.converged)
        assert np.isfinite(float(result.energy))
        assert np.isfinite(float(result.mu))

    def test_density_is_hermitian(self):
        kernel = _make_two_band_kernel()
        result = _solve_problem(kernel)

        np.testing.assert_allclose(
            np.array(result.density),
            np.array(np.conj(np.swapaxes(result.density, -1, -2))),
            atol=1e-6,
        )

    def test_self_consistency(self):
        kernel = _make_two_band_kernel()
        result = _solve_problem(kernel)
        rms = _comm_rms(result.fock, result.density, kernel.w2d)
        assert rms < 1e-4

    def test_particle_number_conserved(self):
        kernel = _make_two_band_kernel()
        result = _solve_problem(kernel)
        n_total = float(np.sum(kernel.w2d[..., None] * result.p))
        np.testing.assert_allclose(n_total, 1.0, atol=1e-4)

    def test_history_has_correct_length(self):
        kernel = _make_two_band_kernel()
        result = _solve_problem(kernel)
        n = int(result.n_iter)
        assert n > 0
        assert np.all(np.isfinite(np.array(result.history["E"][:n])))
        assert np.all(np.isfinite(np.array(result.history["grad_norm"][:n])))


class TestNonInteracting:
    def test_converges_to_exact_occupations(self):
        kernel = _make_two_band_kernel(exchange_strength=0.0, T=0.01)
        result = _solve_problem(kernel, config=SolverConfig(max_iter=50, tol_E=1e-10))
        p = np.array(result.p[0, 0])
        assert p[0] > 0.99 or p[1] > 0.99


class TestMultiKPoint:
    def test_2x2_grid_converges(self):
        kernel = _make_two_band_kernel(nk=2, T=0.1)
        result = _solve_problem(
            kernel, n_electrons=4.0,
            config=SolverConfig(max_iter=100, tol_E=1e-7),
        )
        assert bool(result.converged)
        rms = _comm_rms(result.fock, result.density, kernel.w2d)
        assert rms < 1e-3

    def test_4x4_grid_converges(self):
        kernel = _make_two_band_kernel(nk=4, T=0.1)
        result = _solve_problem(
            kernel, n_electrons=16.0,
            config=SolverConfig(max_iter=200, tol_E=1e-7),
        )
        assert bool(result.converged)


class TestSolveResult:
    def test_result_is_named_tuple(self):
        kernel = _make_two_band_kernel()
        result = _solve_problem(kernel)
        assert isinstance(result, SolveResult)

    def test_shapes_match_problem(self):
        kernel = _make_two_band_kernel(nk=2)
        result = _solve_problem(kernel, n_electrons=4.0)

        assert result.density.shape == kernel.h.shape
        assert result.fock.shape == kernel.h.shape
        assert result.Q.shape == kernel.h.shape
        assert result.p.shape == kernel.h.shape[:-1]


class TestContactTerms:
    @staticmethod
    def _kernel_with_contact(g, nk=2, T=0.05):
        weights = np.ones((nk, nk), dtype=np.float32)
        h = np.zeros((nk, nk, 2, 2), dtype=np.complex64)
        h[..., 0, 0] = -1.0
        h[..., 1, 1] = 1.0
        for i in range(nk):
            for j in range(nk):
                h[i, j, 0, 1] = 0.4 + 0.1 * (i + j)
                h[i, j, 1, 0] = 0.4 + 0.1 * (i + j)
        coulomb_q = np.full((nk, nk, 1, 1), 0.1, dtype=np.complex64)
        sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex64)
        contact_terms = [(np.float32(g), sigma_z, sigma_z)]
        return HartreeFockKernel(
            weights=weights,
            hamiltonian=h,
            coulomb_q=coulomb_q,
            T=T,
            contact_terms=contact_terms,
        )

    def test_direct_minimization_matches_scf_with_contact(self):
        kernel = self._kernel_with_contact(g=0.3, nk=2)
        P0 = np.zeros_like(kernel.h)

        dm = solve(
            kernel, P0, n_electrons=4.0,
            config=SolverConfig(max_iter=200, tol_E=1e-9),
        )
        scf = solve_scf(
            kernel, P0, n_electrons=4.0,
            config=SCFConfig(max_iter=400, mixing=0.3,
                             density_tol=1e-7, comm_tol=1e-6),
        )

        assert bool(dm.converged)
        assert bool(scf.converged)
        np.testing.assert_allclose(
            float(dm.energy), float(scf.energy), atol=1e-5, rtol=1e-6,
        )
        density_diff = float(np.linalg.norm(dm.density - scf.density_matrix))
        assert density_diff < 1e-4, (
            f"DM and SCF densities disagree (||dP||_F = {density_diff:.3e}); "
            "indicates the contact term is missing from the inner-loop Fock build."
        )

    def test_direct_minimization_loop_history_matches_final_energy(self):
        kernel = self._kernel_with_contact(g=0.3, nk=2)
        P0 = np.zeros_like(kernel.h)
        result = solve(
            kernel, P0, n_electrons=4.0,
            config=SolverConfig(max_iter=200, tol_E=1e-9),
        )
        assert bool(result.converged)
        n = int(result.n_iter)
        last_loop_E = float(np.array(result.history["E"])[n - 1])
        np.testing.assert_allclose(
            last_loop_E, float(result.energy), atol=1e-5, rtol=1e-6,
        )

    def test_direct_minimization_self_consistent_with_contact(self):
        kernel = self._kernel_with_contact(g=0.3, nk=2)
        P0 = np.zeros_like(kernel.h)
        result = solve(
            kernel, P0, n_electrons=4.0,
            config=SolverConfig(max_iter=200, tol_E=1e-9),
        )
        assert bool(result.converged)
        rms = _comm_rms(result.fock, result.density, kernel.w2d)
        assert rms < 1e-3

    def test_contact_term_changes_solution(self):
        kernel_off = self._kernel_with_contact(g=0.0, nk=2)
        kernel_on = self._kernel_with_contact(g=0.3, nk=2)
        P0 = np.zeros_like(kernel_off.h)
        cfg = SolverConfig(max_iter=200, tol_E=1e-9)

        E_off = float(solve(kernel_off, P0, 4.0, config=cfg).energy)
        E_on = float(solve(kernel_on, P0, 4.0, config=cfg).energy)

        assert abs(E_on - E_off) > 1e-3


class TestEdgeCases:
    def test_rejects_unreachable_density(self):
        kernel = _make_two_band_kernel()
        P0 = np.zeros_like(kernel.h)

        with pytest.raises(ValueError, match="physically reachable range"):
            solve(kernel, P0, n_electrons=3.0)

    def test_zero_exchange_strength(self):
        kernel = _make_two_band_kernel(exchange_strength=0.0)
        result = _solve_problem(kernel, config=SolverConfig(max_iter=50, tol_E=1e-8))

        assert int(result.n_iter) <= 50
        assert np.isfinite(float(result.energy))


class TestSpectralCayley:
    @staticmethod
    def _make_skew(rng, shape, dtype=np.complex128):
        d = rng.normal(size=shape) + 1j * rng.normal(size=shape)
        d = 0.5 * (d - np.conj(d.swapaxes(-1, -2)))
        return d.astype(dtype)

    @staticmethod
    def _make_herm(rng, shape, dtype=np.complex128):
        F = rng.normal(size=shape) + 1j * rng.normal(size=shape)
        F = 0.5 * (F + np.conj(F.swapaxes(-1, -2)))
        return F.astype(dtype)

    def test_spectral_unitary_matches_cayley_lu(self):
        from cpp_hf.solver import _cayley_retract
        from cpp_hf import _native

        rng = np.random.default_rng(0)
        d = self._make_skew(rng, (3, 3, 6, 6))
        V_d, lam_d = _native._cayley_spectral_setup(d)

        for tau_val in [0.0, 0.1, 0.5, 1.0, -0.3]:
            U_lu = _cayley_retract(d, tau_val)
            U_sp = _native._cayley_unitary_from_spectrum(V_d, lam_d, tau_val)
            np.testing.assert_allclose(
                np.asarray(U_sp), np.asarray(U_lu),
                atol=1e-12, rtol=1e-12,
                err_msg=f"spectral U mismatch at tau={tau_val}",
            )

    def test_spectral_diag_matches_lu_diag(self):
        from cpp_hf.solver import _cayley_retract
        from cpp_hf import _native

        rng = np.random.default_rng(1)
        d = self._make_skew(rng, (3, 3, 6, 6))
        Ft = self._make_herm(rng, (3, 3, 6, 6))

        V_d, lam_d = _native._cayley_spectral_setup(d)
        Ft_eig = np.conj(np.swapaxes(V_d, -2, -1)) @ Ft @ V_d

        for tau_val in [0.0, 0.05, 0.3, 0.7, -0.5]:
            diag_sp = _native._diag_UFU_from_spectrum(V_d, Ft_eig, lam_d, tau_val)
            U_lu = _cayley_retract(d, tau_val)
            Ft_trial = np.conj(np.swapaxes(U_lu, -2, -1)) @ Ft @ U_lu
            diag_lu = np.real(np.diagonal(Ft_trial, axis1=-2, axis2=-1))
            np.testing.assert_allclose(
                np.asarray(diag_sp), np.asarray(diag_lu),
                atol=1e-12, rtol=1e-12,
                err_msg=f"spectral diag mismatch at tau={tau_val}",
            )

    def test_spectral_unitary_is_unitary(self):
        from cpp_hf import _native

        rng = np.random.default_rng(2)
        d = self._make_skew(rng, (2, 2, 8, 8))
        V_d, lam_d = _native._cayley_spectral_setup(d)

        for tau_val in [0.0, 0.5, 1.0]:
            U = _native._cayley_unitary_from_spectrum(V_d, lam_d, tau_val)
            UUH = U @ np.conj(np.swapaxes(U, -1, -2))
            eye = np.eye(8, dtype=U.dtype)
            np.testing.assert_allclose(
                np.asarray(UUH),
                np.asarray(eye[None, None, ...] * np.ones((2, 2, 1, 1), dtype=U.dtype)),
                atol=1e-12, rtol=1e-12,
                err_msg=f"non-unitary at tau={tau_val}",
            )
