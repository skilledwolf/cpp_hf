"""Tests for the new direct-minimization solver."""

import jax.numpy as jnp
import numpy as np
import pytest
from _meshes import mesh_for

from hartreefock import CG, RDIIS, HartreeFockKernel, SolveResult, solve


def _comm_rms(F, P, weights_2d):
    """Weighted RMS of the commutator [F, P]."""
    comm = F @ P - P @ F
    sq = jnp.abs(comm) ** 2
    per_k = jnp.sum(sq, axis=(-2, -1))
    weight_sum = jnp.sum(weights_2d)
    return float(jnp.sqrt(jnp.sum(weights_2d * per_k) / jnp.maximum(weight_sum, 1e-30)))


def _make_two_band_kernel(nk=1, T=0.2, exchange_strength=0.25):
    """Two-band model with weak exchange on an nk x nk grid."""
    weights = jnp.ones((nk, nk), dtype=jnp.float32)
    hamiltonian = np.zeros((nk, nk, 2, 2), dtype=np.complex64)
    hamiltonian[..., 0, 0] = -0.5
    hamiltonian[..., 1, 1] = 0.5
    coulomb_q = jnp.full((nk, nk, 1, 1), exchange_strength, dtype=jnp.complex64)
    return HartreeFockKernel(mesh_for(weights), jnp.asarray(hamiltonian), coulomb_q, T=T)


def _solve_problem(kernel, n_electrons=1.0, config=None):
    """Helper: solve, return result."""
    if config is None:
        config = CG(max_iter=100, tol_E=1e-8)
    P0 = jnp.zeros_like(kernel.h)
    return solve(kernel, P0, n_electrons, config=config)


class TestBasicConvergence:
    def test_converges_on_tiny_model(self):
        kernel = _make_two_band_kernel()
        result = _solve_problem(kernel)

        assert result.n_iter <= 100
        assert result.converged
        assert np.isfinite(result.energy)
        assert np.isfinite(result.mu)

    def test_density_is_hermitian(self):
        kernel = _make_two_band_kernel()
        result = _solve_problem(kernel)

        np.testing.assert_allclose(
            np.array(result.density),
            np.array(jnp.conj(jnp.swapaxes(result.density, -1, -2))),
            atol=1e-6,
        )

    def test_self_consistency(self):
        """At convergence, [F, P] should be near zero."""
        kernel = _make_two_band_kernel()
        result = _solve_problem(kernel)

        rms = _comm_rms(result.fock, result.density, kernel.w2d)
        assert rms < 1e-4

    def test_particle_number_conserved(self):
        kernel = _make_two_band_kernel()
        result = _solve_problem(kernel)

        n_total = float(jnp.sum(
            kernel.w2d[..., None] * result.p
        ))
        np.testing.assert_allclose(n_total, 1.0, atol=1e-4)

    def test_history_has_correct_length(self):
        kernel = _make_two_band_kernel()
        result = _solve_problem(kernel)

        n = result.n_iter
        # History arrays are pre-allocated to max_iter, but first n entries
        # should be populated with finite values
        assert n > 0
        assert np.all(np.isfinite(np.array(result.history["E"][:n])))
        assert np.all(np.isfinite(np.array(result.history["grad_norm"][:n])))


class TestNonInteracting:
    """Non-interacting limit: exchange=0, solution is just diag(h)."""

    def test_converges_to_exact_occupations(self):
        kernel = _make_two_band_kernel(exchange_strength=0.0, T=0.01)
        result = _solve_problem(kernel, config=CG(max_iter=50, tol_E=1e-10))

        # With eps = [-0.5, 0.5] and T=0.01, occupation should be ~[1, 0]
        p = np.array(result.p[0, 0])
        assert p[0] > 0.99 or p[1] > 0.99  # one of them should be ~1


class TestMultiKPoint:
    def test_2x2_grid_converges(self):
        kernel = _make_two_band_kernel(nk=2, T=0.1)
        result = _solve_problem(
            kernel, n_electrons=4.0,
            config=CG(max_iter=100, tol_E=1e-7),
        )

        assert result.converged
        rms = _comm_rms(result.fock, result.density, kernel.w2d)
        assert rms < 1e-3

    def test_4x4_grid_converges(self):
        kernel = _make_two_band_kernel(nk=4, T=0.1)
        result = _solve_problem(
            kernel,
            n_electrons=16.0,
            config=CG(max_iter=200, tol_E=1e-7),
        )

        assert result.converged


class TestPlateauWindow:
    """Windowed-energy convergence (plateau_window) for the CG path."""

    def test_windowed_and_single_step_agree(self):
        kernel = _make_two_band_kernel(nk=2, T=0.1)
        P0 = jnp.zeros_like(kernel.h)
        # plateau_window=0 is the per-iteration |dE| test; the default 5 is the
        # windowed test.  Both must reach the same converged solution.
        r_single = solve(kernel, P0, 4.0, config=CG(
            max_iter=200, tol_E=1e-8, plateau_window=0))
        r_window = solve(kernel, P0, 4.0, config=CG(
            max_iter=200, tol_E=1e-8, plateau_window=5))
        assert r_single.converged and r_window.converged
        np.testing.assert_allclose(
            r_window.energy, r_single.energy, atol=1e-6)

    def test_default_is_windowed(self):
        assert CG().plateau_window == 5


class TestGradientStop:
    """tol_grad > 0 is a sufficient stop for the CG path (mirrors cpp_hf).

    Needs a kernel with band hybridization: without off-diagonal h the
    orbital gradient is exactly zero from iteration 0 (all relaxation lives
    in the occupation channel, which tol_grad intentionally excludes).

    Runs in float64 (like the Newton tests): the float32 orbital-gradient
    noise floor is platform-dependent (~1e-5 on macOS ARM, ~2e-4 on Linux
    CI), so float32 gradient tolerances are not portable.
    """

    @pytest.fixture(autouse=True)
    def _x64(self):
        import jax
        old = jax.config.jax_enable_x64
        jax.config.update("jax_enable_x64", True)
        yield
        jax.config.update("jax_enable_x64", old)

    @staticmethod
    def _hybridized_kernel(nk=2, T=0.1):
        h = np.zeros((nk, nk, 2, 2), dtype=np.complex128)
        h[..., 0, 0] = -0.5
        h[..., 1, 1] = 0.5
        h[..., 0, 1] = h[..., 1, 0] = 0.2
        return HartreeFockKernel(mesh_for(jnp.ones((nk, nk), dtype=jnp.float64)), jnp.asarray(h), jnp.full((nk, nk, 1, 1), 0.5, dtype=jnp.complex128), T=T)

    def _solve(self, config):
        kernel = self._hybridized_kernel()
        P0 = np.zeros(kernel.h.shape, dtype=np.complex128)
        # Seed off the h-eigenbasis so the orbital gradient starts nonzero.
        P0[..., 0, 0] = 0.6
        P0[..., 1, 1] = 0.15
        P0[..., 0, 1] = P0[..., 1, 0] = 0.25
        return solve(kernel, jnp.asarray(P0), 3.0, config=config)

    def test_cg_stops_at_first_crossing_and_flags_converged(self):
        tol = 1e-4
        result = self._solve(CG(max_iter=300, tol_grad=tol))
        assert result.converged
        n = result.n_iter
        assert 1 < n < 300
        hG = np.asarray(result.history["grad_norm"][:n])
        # Histories end with the qualifying point, and it is the first
        # crossing — the run must not grind on after reaching tolerance.
        assert hG[-1] <= tol
        assert np.all(hG[:-1] > tol)

    def test_cg_gradient_stop_matches_energy_stop_solution(self):
        by_grad = self._solve(CG(max_iter=300, tol_grad=1e-7))
        by_energy = self._solve(CG(max_iter=300, tol_E=1e-10))
        assert bool(by_grad.converged) and bool(by_energy.converged)
        np.testing.assert_allclose(
            float(by_grad.energy), float(by_energy.energy), atol=1e-6)

    def test_cg_unreachable_tolerance_flags_unconverged(self):
        result = self._solve(CG(max_iter=3, tol_grad=1e-14))
        assert not result.converged
        assert result.n_iter == 3


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
    """Contact (q-independent) flavor-bilinear terms must enter the inner loop.

    Regression for the bug where ``_solve_impl`` body forgot to forward
    ``contact_g/Oi/Oj`` to ``build_fock`` — the optimizer minimised the
    contact-free energy and re-introduced the contact contribution only at
    the final eval, so the returned density was not a stationary point of
    the augmented energy and ``history["E"]`` jumped at the boundary.
    """

    @staticmethod
    def _kernel_with_contact(g, nk=2, T=0.05):
        """Two-band kernel with off-diagonal h (so density mixes flavors) and σ_z⊗σ_z contact."""
        weights = jnp.ones((nk, nk), dtype=jnp.float32)
        h = np.zeros((nk, nk, 2, 2), dtype=np.complex64)
        h[..., 0, 0] = -1.0
        h[..., 1, 1] = 1.0
        # k-dependent off-diagonal hopping — gives orbitals that mix flavors,
        # so ρ̄ acquires nontrivial off-diagonal entries that the contact term
        # actually couples to.
        for i in range(nk):
            for j in range(nk):
                h[i, j, 0, 1] = 0.4 + 0.1 * (i + j)
                h[i, j, 1, 0] = 0.4 + 0.1 * (i + j)
        coulomb_q = jnp.full((nk, nk, 1, 1), 0.1, dtype=jnp.complex64)
        sigma_z = jnp.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=jnp.complex64)
        contact_terms = [(jnp.float32(g), sigma_z, sigma_z)]
        return HartreeFockKernel(mesh_for(weights), jnp.asarray(h), coulomb_q, T=T, contact_terms=contact_terms)

    def test_direct_minimization_matches_scf_with_contact(self):
        """Both solvers must land on the same density and energy with a contact term.

        Energy alone is insensitive (it's stationary at both fixed points, so a
        small density miss costs only O(δP²)).  The Frobenius density distance
        is the load-bearing check — pre-fix, DM converged to the contact-free
        density which differs from the SCF density at ~1e-3 in this setup.
        """
        kernel = self._kernel_with_contact(g=0.3, nk=2)
        P0 = jnp.zeros_like(kernel.h)

        dm = solve(
            kernel, P0, n_electrons=4.0,
            config=CG(max_iter=200, tol_E=1e-9),
        )
        scf = solve(
            kernel, P0, n_electrons=4.0,
            config=RDIIS(max_iter=400, density_tol=1e-7, comm_tol=1e-6),
        )

        assert bool(dm.converged)
        assert bool(scf.converged)
        np.testing.assert_allclose(
            float(dm.energy), float(scf.energy), atol=1e-5, rtol=1e-6,
        )
        density_diff = float(jnp.linalg.norm(dm.density - scf.density))
        assert density_diff < 1e-4, (
            f"DM and SCF densities disagree (||ΔP||_F = {density_diff:.3e}); "
            "indicates the contact term is missing from the inner-loop Fock build."
        )

    def test_direct_minimization_loop_history_matches_final_energy(self):
        """Last logged loop energy must equal the returned energy.

        Pre-fix the loop trace logged ``E_no_contact[P]`` while the returned
        ``energy`` reported ``E_with_contact[P]`` — a visible jump at the
        loop boundary.
        """
        kernel = self._kernel_with_contact(g=0.3, nk=2)
        P0 = jnp.zeros_like(kernel.h)
        result = solve(
            kernel, P0, n_electrons=4.0,
            config=CG(max_iter=200, tol_E=1e-9),
        )

        assert result.converged
        n = result.n_iter
        last_loop_E = float(np.array(result.history["E"])[n - 1])
        np.testing.assert_allclose(
            last_loop_E, result.energy, atol=1e-5, rtol=1e-6,
        )

    def test_direct_minimization_self_consistent_with_contact(self):
        """[F, P] must vanish at convergence — F includes the contact contribution."""
        kernel = self._kernel_with_contact(g=0.3, nk=2)
        P0 = jnp.zeros_like(kernel.h)
        result = solve(
            kernel, P0, n_electrons=4.0,
            config=CG(max_iter=200, tol_E=1e-9),
        )

        assert result.converged
        rms = _comm_rms(result.fock, result.density, kernel.w2d)
        assert rms < 1e-3

    def test_contact_term_changes_solution(self):
        """Sanity check: the contact coupling must actually move the energy.

        Guards against a regression where contact terms get silently zeroed —
        without this check, a fix that wires them through incorrectly (e.g.
        multiplied by 0) could still pass the SCF-vs-DM cross-check.
        """
        kernel_off = self._kernel_with_contact(g=0.0, nk=2)
        kernel_on = self._kernel_with_contact(g=0.3, nk=2)
        P0 = jnp.zeros_like(kernel_off.h)
        cfg = CG(max_iter=200, tol_E=1e-9)

        E_off = float(solve(kernel_off, P0, 4.0, config=cfg).energy)
        E_on = float(solve(kernel_on, P0, 4.0, config=cfg).energy)

        assert abs(E_on - E_off) > 1e-3


class TestEdgeCases:
    def test_rejects_unreachable_density(self):
        kernel = _make_two_band_kernel()
        P0 = jnp.zeros_like(kernel.h)

        with pytest.raises(ValueError, match="physically reachable range"):
            solve(kernel, P0, n_electrons=3.0, config=CG())

    def test_zero_exchange_strength(self):
        """Should converge quickly with no interaction."""
        kernel = _make_two_band_kernel(exchange_strength=0.0)
        result = _solve_problem(kernel, config=CG(max_iter=50, tol_E=1e-8))

        assert result.n_iter <= 50
        assert np.isfinite(result.energy)


class TestSpectralCayley:
    """Spectral Cayley path: line search uses eigh(i*d_Q) + Hadamard scaling.

    Verifies that the spectral helpers produce results numerically equivalent
    to building U via the LU-based Cayley solve and computing diag(U†FtU)
    explicitly.
    """

    @staticmethod
    def _make_skew(rng, shape, dtype=jnp.complex64):
        d = rng.normal(size=shape) + 1j * rng.normal(size=shape)
        d = 0.5 * (d - np.conj(d.swapaxes(-1, -2)))
        return jnp.asarray(d, dtype=dtype)

    @staticmethod
    def _make_herm(rng, shape, dtype=jnp.complex64):
        F = rng.normal(size=shape) + 1j * rng.normal(size=shape)
        F = 0.5 * (F + np.conj(F.swapaxes(-1, -2)))
        return jnp.asarray(F, dtype=dtype)

    def test_spectral_unitary_matches_cayley_lu(self):
        """U(τ) from spectral form must match U(τ) from Cayley LU to roundoff."""
        from hartreefock._jax.riemannian import (
            _cayley_retract,
            _cayley_spectral_setup,
            _cayley_unitary_from_spectrum,
        )
        rng = np.random.default_rng(0)
        d = self._make_skew(rng, (3, 3, 6, 6))
        V_d, lam_d = _cayley_spectral_setup(d)

        for tau_val in [0.0, 0.1, 0.5, 1.0, -0.3]:
            tau = jnp.asarray(tau_val, dtype=jnp.float32)
            U_lu = _cayley_retract(d, tau)
            U_sp = _cayley_unitary_from_spectrum(V_d, lam_d, tau)
            np.testing.assert_allclose(
                np.asarray(U_sp), np.asarray(U_lu),
                atol=1e-5, rtol=1e-5,
                err_msg=f"spectral U mismatch at tau={tau_val}",
            )

    def test_spectral_diag_matches_lu_diag(self):
        """diag(U(τ)†·Ft·U(τ)) from spectral form must match LU-built form."""
        from hartreefock._jax.riemannian import (
            _cayley_retract,
            _cayley_spectral_setup,
            _diag_UFU_from_spectrum,
        )
        rng = np.random.default_rng(1)
        d = self._make_skew(rng, (3, 3, 6, 6))
        Ft = self._make_herm(rng, (3, 3, 6, 6))

        V_d, lam_d = _cayley_spectral_setup(d)
        Ft_eig = jnp.conj(jnp.swapaxes(V_d, -2, -1)) @ Ft @ V_d

        for tau_val in [0.0, 0.05, 0.3, 0.7, -0.5]:
            tau = jnp.asarray(tau_val, dtype=jnp.float32)
            diag_sp = _diag_UFU_from_spectrum(V_d, Ft_eig, lam_d, tau)
            U_lu = _cayley_retract(d, tau)
            Ft_trial = jnp.conj(jnp.swapaxes(U_lu, -2, -1)) @ Ft @ U_lu
            diag_lu = jnp.real(jnp.diagonal(Ft_trial, axis1=-2, axis2=-1))
            np.testing.assert_allclose(
                np.asarray(diag_sp), np.asarray(diag_lu),
                atol=1e-4, rtol=1e-4,
                err_msg=f"spectral diag mismatch at tau={tau_val}",
            )

    def test_spectral_unitary_is_unitary(self):
        """Even for τ=0 (eigh roundoff worst case), U†U must equal I."""
        from hartreefock._jax.riemannian import (
            _cayley_spectral_setup,
            _cayley_unitary_from_spectrum,
        )
        rng = np.random.default_rng(2)
        d = self._make_skew(rng, (2, 2, 8, 8))
        V_d, lam_d = _cayley_spectral_setup(d)

        for tau_val in [0.0, 0.5, 1.0]:
            tau = jnp.asarray(tau_val, dtype=jnp.float32)
            U = _cayley_unitary_from_spectrum(V_d, lam_d, tau)
            UUH = U @ jnp.conj(jnp.swapaxes(U, -1, -2))
            eye = jnp.eye(8, dtype=U.dtype)
            np.testing.assert_allclose(
                np.asarray(UUH), np.asarray(eye[None, None, ...] * jnp.ones((2, 2, 1, 1), dtype=U.dtype)),
                atol=1e-5, rtol=1e-5,
                err_msg=f"non-unitary at tau={tau_val}",
            )


class TestAdversarialStationarity:
    @staticmethod
    def _repulsive_hartree(T=0.5):
        h = jnp.zeros((1, 1, 2, 2), dtype=jnp.complex64)
        return HartreeFockKernel(mesh_for(jnp.ones((1, 1), dtype=jnp.float32)), h, jnp.zeros((1, 1, 1, 1), dtype=jnp.complex64), hartree=jnp.eye(2, dtype=jnp.float32), T=T, exchange=False, reference=jnp.zeros_like(h))

    def test_zero_orbital_gradient_does_not_hide_wrong_occupations(self):
        """Diagonal F and P have zero commutator even at a maximally wrong P."""
        kernel = self._repulsive_hartree()
        P0 = jnp.array([[[[1.0, 0.0], [0.0, 0.0]]]], dtype=jnp.complex64)
        result = solve(
            kernel, P0, 1.0,
            config=CG(max_iter=300, tol_grad=2e-4),
        )
        assert result.converged
        np.testing.assert_allclose(
            np.asarray(result.density[0, 0]),
            0.5 * np.eye(2),
            atol=2e-3,
        )
        n = result.n_iter
        free_history = np.asarray(result.history["free_energy"][:n])
        assert np.all(np.diff(free_history) <= 2e-5)

    @pytest.mark.parametrize("n_electrons", [0.0, 2.0])
    def test_exact_empty_and_full_fillings(self, n_electrons):
        kernel = self._repulsive_hartree()
        result = solve(
            kernel, jnp.zeros_like(kernel.h), n_electrons,
            config=CG(max_iter=20, tol_grad=1e-4),
        )
        assert result.converged
        expected = np.eye(2) * (n_electrons / 2.0)
        np.testing.assert_array_equal(np.asarray(result.density[0, 0]), expected)


class TestSolverInputValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"max_iter": 1.5},
            {"tol_E": np.nan},
            {"tol_grad": np.inf},
            {"cg_restart": 0},
            {"bt_shrink": 1.0},
        ],
    )
    def test_invalid_config_domains_raise(self, kwargs):
        with pytest.raises(ValueError):
            CG(**kwargs)

    def test_wrong_P0_shape_raises_before_jit(self):
        kernel = _make_two_band_kernel(nk=2)
        with pytest.raises(ValueError, match="P0 must have shape"):
            solve(kernel, jnp.zeros((2, 2, 3, 3)), 1.0, config=CG())

    def test_negative_kernel_temperature_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            _make_two_band_kernel(T=-0.1)
