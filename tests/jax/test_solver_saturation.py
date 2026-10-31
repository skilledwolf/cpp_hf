"""Finite-temperature occupation handling at saturated occupations.

At low temperature most occupations sit exactly at 0 or 1 in floating point.
The clipped logit of such a band is not a gradient; before the fix it entered
the uniform particle-conservation shift, turned the occupation search direction
into an ascent direction, and made the true-objective line search reject every
candidate, so the solver advanced only through its damped Fermi-Dirac fallback
at about ten Fock builds per iteration.

Saturated bands are now treated as active bound constraints: held fixed, given
zero Fermi-Dirac curvature, and excluded from the particle-conservation
projection, which is enforced over the resolvable bands.  The tests below pin
that behaviour, the descent property of the Fermi-Dirac direction the fallback
still relies on, and the early-stopping chemical-potential solver, whose
accuracy carries the particle count.  They run in float64 like the other
gradient-level tests, except where a float32 regime is the point.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from hartreefock.backends.jax import HartreeFockKernel, SCFConfig, SolverConfig, solve, solve_scf
from hartreefock.backends.jax.fock import build_fock, free_energy, hf_energy
import hartreefock.backends.jax.solver as S


@pytest.fixture(autouse=True)
def _x64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


def _wide_band_kernel(nk=6, nb=4, T=0.01, ex=0.3, seed=0):
    """Bands spanning +-20 at T=0.01: nearly every occupation saturates."""
    rng = np.random.default_rng(seed)
    h = np.zeros((nk, nk, nb, nb), dtype=np.complex128)
    h[..., np.arange(nb), np.arange(nb)] = np.linspace(-20.0, 20.0, nb)
    A = rng.normal(scale=0.5, size=(nk, nk, nb, nb)) + 1j * rng.normal(scale=0.5, size=(nk, nk, nb, nb))
    h = h + 0.5 * (A + np.conj(A.swapaxes(-1, -2)))
    return HartreeFockKernel(
        weights=jnp.ones((nk, nk), dtype=jnp.float64) / (nk * nk),
        hamiltonian=jnp.asarray(h),
        coulomb_q=jnp.full((nk, nk, 1, 1), ex, dtype=jnp.complex128),
        T=T,
    )


def _fock(kernel, P):
    a = kernel.as_args()
    return build_fock(
        P, h=a["h"], VR=a["VR"], refP=a["refP"], HH=a["HH"], w2d=kernel.w2d,
        include_exchange=a["include_exchange"], include_hartree=a["include_hartree"],
        exchange_hermitian_channel_packing=a["exchange_hermitian_channel_packing"],
        contact_g=a["contact_g"], contact_Oi=a["contact_Oi"], contact_Oj=a["contact_Oj"],
    )


def _initial_state(kernel, n_electrons):
    """(Q, p, mu) as the CG solver initialises them from P0 = 0."""
    _, _, F0 = _fock(kernel, jnp.zeros_like(kernel.h))
    eps0, Q = jnp.linalg.eigh(F0)
    w_norm = kernel.w2d / kernel.weight_sum
    ntn = n_electrons / float(kernel.weight_sum)
    mu = S._solve_mu(eps0, w_norm, ntn, 0.0, kernel.T, maxiter=50)
    p = jax.nn.sigmoid((mu - eps0) / kernel.T)
    return Q, p, mu, w_norm, ntn


def _free_energy(kernel, Q, p):
    a = kernel.as_args()
    P = S._herm(S._density_from_Qp(Q, p))
    Sigma, H, _ = _fock(kernel, P)
    E = hf_energy(P, h=a["h"], Sigma=Sigma, H=H, weights_b=kernel.weights_b, refP=a["refP"])
    return free_energy(E, p, kernel.w2d, kernel.T)


class TestSaturatedOccupationGradient:
    def test_gradient_vanishes_at_fermi_dirac_state(self):
        kernel = _wide_band_kernel()
        n_e = 1.5 * float(kernel.weight_sum)          # metallic: band 2 partially filled
        Q, p, mu, w_norm, _ = _initial_state(kernel, n_e)
        _, _, F0 = _fock(kernel, jnp.zeros_like(kernel.h))
        eps = jnp.real(jnp.diagonal(S._fock_in_orbital_basis(Q, F0), axis1=-2, axis2=-1))

        g, h, curv, active = S._occupation_gradient(eps, p, kernel.T, w_norm)
        frac_saturated = 1.0 - float(jnp.mean(active))
        assert frac_saturated > 0.5, "fixture should saturate most occupations"
        assert bool(jnp.any(active)), "fixture should keep some bands partially filled"

        # (eps, p) is Fermi-Dirac consistent, so the projected gradient and its
        # preconditioned form must vanish relative to the band scale.  The only
        # residue is the logit round trip of nearly saturated active bands
        # (relative error ~eps/(1-p) in logit, damped by curv ~ (1-p) in h).
        scale = float(S._norm_vec(eps, w_norm))
        assert float(S._norm_vec(g, w_norm)) < 1e-7 * scale
        assert float(S._norm_vec(h, w_norm)) < 1e-10 * scale
        assert bool(jnp.all(curv[~active] == 0.0))

        # The unmasked formula this replaces is not small: the clipped logit of
        # the saturated bands leaks through the uniform particle shift.
        g_unmasked = S._project_occ(eps + kernel.T * S._logit(p), w_norm)
        assert float(S._norm_vec(g_unmasked, w_norm)) > 1e-2 * scale

    def test_preconditioned_direction_is_descent(self):
        kernel = _wide_band_kernel()
        n_e = 1.5 * float(kernel.weight_sum)
        Q, p, mu, w_norm, ntn = _initial_state(kernel, n_e)
        # Interacting Fock at the initial density: not stationary in p.
        P = S._herm(S._density_from_Qp(Q, p))
        _, _, F = _fock(kernel, P)
        eps = jnp.real(jnp.diagonal(S._fock_in_orbital_basis(Q, F), axis1=-2, axis2=-1))
        g_raw = eps + kernel.T * S._logit(p)
        _, h, curv, _ = S._occupation_gradient(eps, p, kernel.T, w_norm)

        # Analytic directional derivative along p -> p - tau h, in the same
        # (unnormalised) k-weights the free energy uses.
        slope = -float(jnp.sum(kernel.w2d[..., None] * g_raw * h))
        assert slope < 0.0
        Om0 = float(_free_energy(kernel, Q, p))
        tau = 1e-4
        p_t = S._project_capped_occupations(p - tau * h, w_norm, ntn)
        fd_slope = (float(_free_energy(kernel, Q, p_t)) - Om0) / tau
        np.testing.assert_allclose(fd_slope, slope, rtol=2e-2)
        # Particle number is preserved by the direction itself, up to the
        # rounding of the band-scale shift that was subtracted to form h.
        scale = float(S._norm_vec(eps, w_norm))
        assert abs(float(jnp.sum(w_norm[..., None] * h))) < 1e-13 * scale

    def test_masked_projection_reduces_to_uniform_when_all_active(self):
        rng = np.random.default_rng(3)
        dp = jnp.asarray(rng.normal(size=(3, 3, 5)))
        w2d = jnp.asarray(rng.uniform(0.5, 1.5, size=(3, 3)))
        active = jnp.ones(dp.shape, dtype=bool)
        np.testing.assert_allclose(
            np.asarray(S._project_occ_masked(dp, w2d, active)),
            np.asarray(S._project_occ(dp, w2d)), atol=1e-14,
        )
        curv = jnp.asarray(rng.uniform(0.1, 1.0, size=dp.shape))
        proj = S._project_occ_curvature(dp, w2d, curv)
        assert abs(float(jnp.sum(w2d[..., None] * proj))) < 1e-13


class TestFrozenFockOccupationStep:
    """The CG occupation move: p -> p + s (p_FD(Ft) - p), s in (0, 1].

    Exercised on a fully saturated state (every occupation exactly 0 or 1, as a
    cold T -> 0 seed produces), which is precisely where an occupation gradient
    step cannot move at all.
    """

    @staticmethod
    def _integer_occupations(kernel, n_electrons):
        """Q from the non-interacting Fock, p integer-valued at the target count.

        Half the k-points get two filled bands and half get one, so the total is
        1.5 per k-point while every individual occupation sits on a bound.
        """
        Q, _p, _mu, w_norm, ntn = _initial_state(kernel, n_electrons)
        nk1, nk2, nb = Q.shape[0], Q.shape[1], Q.shape[-1]
        flat = np.zeros((nk1 * nk2, nb))
        flat[:, 0] = 1.0
        flat[: (nk1 * nk2) // 2, 1] = 1.0
        p = jnp.asarray(flat.reshape(nk1, nk2, nb))
        return Q, p, w_norm, ntn

    @staticmethod
    def _frozen_fock_occupations(kernel, Q, p, w_norm, ntn):
        P = S._herm(S._density_from_Qp(Q, p))
        _, _, F = _fock(kernel, P)
        eps = jnp.real(jnp.diagonal(S._fock_in_orbital_basis(Q, F), axis1=-2, axis2=-1))
        mu = S._solve_mu(eps, w_norm, ntn, 0.0, kernel.T, maxiter=50)
        return eps, mu, jax.nn.sigmoid((mu - eps) / kernel.T)

    def test_direction_descends_and_conserves_particles(self):
        kernel = _wide_band_kernel()
        n_e = 1.5 * float(kernel.weight_sum)
        Q, p, w_norm, ntn = self._integer_occupations(kernel, n_e)
        assert float(jnp.mean(S._active_occupations(p))) == 0.0, "fixture must be fully saturated"
        np.testing.assert_allclose(float(jnp.sum(w_norm[..., None] * p)), ntn, atol=1e-14)

        eps, mu, p_fd = self._frozen_fock_occupations(kernel, Q, p, w_norm, ntn)
        d = p_fd - p
        assert float(S._norm_vec(d, w_norm)) > 1e-3, "fixture must be off the FD manifold"
        # Particle-conserving to the accuracy of the chemical-potential solve.
        assert abs(float(jnp.sum(w_norm[..., None] * d))) < 1e-12

        # Slope of Omega along p -> p + s d.  Writing x_i = (mu - eps_i)/T, so
        # that p_FD_i = sigmoid(x_i), each term of the sum is
        #   w_i T (logit(p_i) - x_i) (sigmoid(x_i) - p_i) <= 0
        # because logit and sigmoid are inverse increasing functions: the move
        # descends whatever the clipping does to the logit of a saturated band.
        # (Clipping can only flip a term for a band more than |log eps| * T
        # above mu, where p_FD_i itself is below machine epsilon.)
        integrand = (eps + kernel.T * S._logit(p) - mu) * d
        assert float(jnp.max(integrand)) < 1e-15
        slope = float(jnp.sum(kernel.w2d[..., None] * integrand))
        np.testing.assert_allclose(
            slope,
            float(jnp.sum(kernel.w2d[..., None] * (eps + kernel.T * S._logit(p)) * d)),
            atol=1e-10,
        )
        assert slope < 0.0

        # The full step is the frozen-Fock minimiser, so it must beat both the
        # starting point and every partial step along the same path.  (At an
        # exact bound the entropy derivative is not representable, so the value
        # of the objective, not a finite-difference slope, is the meaningful
        # check here; the slope is verified below on an unsaturated state.)
        omegas = [float(_free_energy(kernel, Q, p + si * d)) for si in (0.0, 0.25, 0.5, 1.0)]
        assert omegas == sorted(omegas, reverse=True), omegas

    def test_slope_matches_finite_difference_when_unsaturated(self):
        # T = 5 against a +-20 band span keeps every occupation fractional, so
        # the entropy derivative is finite and the analytic slope is testable.
        kernel = _wide_band_kernel(T=5.0)
        n_e = 1.5 * float(kernel.weight_sum)
        Q, p_start, _mu0, w_norm, ntn = _initial_state(kernel, n_e)
        assert bool(jnp.all(S._active_occupations(p_start)))
        # Perturb off the Fermi-Dirac manifold, conserving the particle count.
        rng = np.random.default_rng(11)
        noise = jnp.asarray(rng.normal(scale=0.05, size=p_start.shape))
        p = jnp.clip(p_start + S._project_occ(noise, w_norm), 0.02, 0.98)
        p = S._project_capped_occupations(p, w_norm, ntn)

        eps, mu, p_fd = self._frozen_fock_occupations(kernel, Q, p, w_norm, ntn)
        d = p_fd - p
        assert abs(float(jnp.sum(w_norm[..., None] * d))) < 1e-12
        slope = float(jnp.sum(kernel.w2d[..., None] * (eps + kernel.T * S._logit(p) - mu) * d))
        assert slope < 0.0
        Om0 = float(_free_energy(kernel, Q, p))
        step = 1e-6
        fd_slope = (float(_free_energy(kernel, Q, p + step * d)) - Om0) / step
        np.testing.assert_allclose(fd_slope, slope, rtol=1e-3)

    def test_gradient_step_is_frozen_but_fermi_dirac_step_is_not(self):
        """A preconditioned gradient step cannot leave an exact bound."""
        kernel = _wide_band_kernel()
        n_e = 1.5 * float(kernel.weight_sum)
        Q, p, w_norm, ntn = self._integer_occupations(kernel, n_e)
        eps, mu, p_fd = self._frozen_fock_occupations(kernel, Q, p, w_norm, ntn)

        _g, h, curv, active = S._occupation_gradient(eps, p, kernel.T, w_norm)
        assert not bool(jnp.any(active))
        assert float(jnp.max(jnp.abs(curv))) == 0.0
        assert float(jnp.max(jnp.abs(h))) == 0.0          # gradient step: no motion
        assert float(jnp.max(jnp.abs(p_fd - p))) > 1e-3   # Fermi-Dirac step: motion

    def test_solver_escapes_a_fully_saturated_start(self):
        """End to end: a cold integer-occupation seed must still reach the SCF answer."""
        kernel = _wide_band_kernel()
        n_e = 1.5 * float(kernel.weight_sum)
        Q, p, _w_norm, _ntn = self._integer_occupations(kernel, n_e)
        P0 = S._herm(S._density_from_Qp(Q, p))
        cg = solve(kernel, P0, n_e, config=SolverConfig(max_iter=300, tol_E=1e-10))
        scf = solve_scf(
            kernel, P0, n_e,
            config=SCFConfig(max_iter=500, acceleration="diis",
                             density_tol=1e-10, comm_tol=1e-9),
        )
        assert bool(cg.converged) and bool(scf.converged)
        np.testing.assert_allclose(float(cg.energy), float(scf.energy), atol=1e-7)


class TestLineSearchAcceptsWhenSaturated:
    """The defect and its fix, at the level of the mechanism that broke."""

    @staticmethod
    def _cg_state(kernel, n_electrons):
        """(Q, p, Omega, Ft, eps) exactly as the solver's first iteration sees them."""
        Q, p, _mu, w_norm, ntn = _initial_state(kernel, n_electrons)
        P = S._herm(S._density_from_Qp(Q, p))
        _, _, F = _fock(kernel, P)
        Ft = S._fock_in_orbital_basis(Q, F)
        eps = jnp.real(jnp.diagonal(Ft, axis1=-2, axis2=-1))
        return Q, p, float(_free_energy(kernel, Q, p)), Ft, eps, w_norm, ntn

    @staticmethod
    def _directions(kernel, p, eps, Ft, w_norm, masked):
        """Preconditioned CG search directions, with and without the saturation mask."""
        nb = p.shape[-1]
        offdiag = 1.0 - jnp.eye(nb)
        diff_p = p[..., None, :] - p[..., :, None]
        G_Q = S._skew_hermitian(diff_p * Ft) * offdiag
        gap = eps[..., :, None] - eps[..., None, :]
        lam = jnp.maximum(kernel.T, 1e-3 * jnp.sqrt(jnp.mean(eps ** 2)))
        d_Q = G_Q / jnp.sqrt(gap ** 2 + lam ** 2)
        if masked:
            _g, d_p, _c, _a = S._occupation_gradient(eps, p, kernel.T, w_norm)
        else:
            g_p = S._project_occ(eps + kernel.T * S._logit(p), w_norm)
            p_safe = jnp.clip(p, 1e-8, 1.0 - 1e-8)
            d_p = S._project_occ(g_p * p_safe * (1.0 - p_safe) / kernel.T, w_norm)
        return d_Q, d_p

    def _best_trial(self, kernel, Q, p, d_Q, d_p, Omega, w_norm, ntn, bt_max=8):
        """Lowest free energy over the solver's backtracking sequence."""
        V_d, lam_d = S._cayley_spectral_setup(d_Q)
        tau = float(jnp.minimum(1.0, 0.6 / jnp.maximum(S._norm_matrix(d_Q, w_norm), 1e-30)))
        best = Omega
        for _ in range(bt_max):
            U = S._cayley_unitary_from_spectrum(V_d, lam_d, tau)
            p_t = S._project_capped_occupations(p - tau * d_p, w_norm, ntn)
            best = min(best, float(_free_energy(kernel, Q @ U, p_t)))
            tau *= 0.5
        return best

    def test_masked_direction_yields_a_descent_step_where_unmasked_does_not(self):
        kernel = _wide_band_kernel()
        n_e = 1.5 * float(kernel.weight_sum)
        Q, p, Omega, Ft, eps, w_norm, ntn = self._cg_state(kernel, n_e)
        assert float(jnp.mean(S._active_occupations(p))) < 0.5, "fixture must saturate"

        dq_m, dp_m = self._directions(kernel, p, eps, Ft, w_norm, masked=True)
        dq_u, dp_u = self._directions(kernel, p, eps, Ft, w_norm, masked=False)
        # Same orbital direction; only the occupation direction differs.
        np.testing.assert_allclose(np.asarray(dq_m), np.asarray(dq_u), atol=0)

        best_masked = self._best_trial(kernel, Q, p, dq_m, dp_m, Omega, w_norm, ntn)
        best_unmasked = self._best_trial(kernel, Q, p, dq_u, dp_u, Omega, w_norm, ntn)
        # The fix: some backtracking candidate now descends...
        assert best_masked < Omega, (best_masked, Omega)
        # ...where every candidate along the unmasked direction climbed, which is
        # what made the solver fall back to a plain Roothaan step every iteration.
        assert best_unmasked >= Omega, (best_unmasked, Omega)

    def test_cg_costs_fewer_fock_builds_than_an_exhausted_line_search(self):
        """End to end: the line search must not burn its whole budget every step.

        The default fixture is essentially converged after one CG iteration;
        the remaining iterations sit at the rounding floor, where the number of
        rejected trials depends on the platform (5 to 10 builds per iteration
        across jax versions and CPUs). A stronger exchange keeps the solver
        descending for several iterations, where the masked occupation step
        accepts after about three builds everywhere it was measured.
        """
        kernel = _wide_band_kernel(ex=4.0)
        n_e = 1.5 * float(kernel.weight_sum)
        P0 = jnp.zeros_like(kernel.h)
        counter = {"n": 0}

        def fock_fn(*args, **kwargs):
            jax.debug.callback(lambda: counter.__setitem__("n", counter["n"] + 1))
            return build_fock(*args, **kwargs)

        cfg = SolverConfig(max_iter=200, tol_E=1e-9)
        result = solve(kernel, P0, n_e, config=cfg, fock_build_fn=fock_fn)
        jax.block_until_ready(result.energy)
        assert bool(result.converged)
        n = int(result.n_iter)
        assert n > 0
        # Ceiling when every candidate is rejected: one build at the current
        # state, ``bt_max`` line-search trials, and a fallback trial.
        exhausted = 1 + cfg.bt_max + 1
        builds_per_iter = (counter["n"] - 3) / n      # init, finalize, polish
        assert builds_per_iter < exhausted - 1.0, builds_per_iter

        scf = solve_scf(
            kernel, P0, n_e,
            config=SCFConfig(max_iter=500, acceleration="diis",
                             density_tol=1e-10, comm_tol=1e-9),
        )
        assert bool(scf.converged)
        np.testing.assert_allclose(float(result.energy), float(scf.energy), atol=1e-7)

    def test_newton_converges_and_matches_cg(self):
        kernel = _wide_band_kernel()
        n_e = 1.5 * float(kernel.weight_sum)
        P0 = jnp.zeros_like(kernel.h)
        cg = solve(kernel, P0, n_e, config=SolverConfig(max_iter=300, tol_grad=1e-8))
        nw = solve(kernel, P0, n_e, config=SolverConfig(max_iter=100, tol_grad=1e-8, optimizer="newton"))
        assert bool(cg.converged) and bool(nw.converged)
        np.testing.assert_allclose(float(nw.energy), float(cg.energy), atol=1e-7)


class TestSolveMuEarlyStop:
    @pytest.mark.parametrize("T", [0.01, 0.5])
    @pytest.mark.parametrize("target", [0.3, 1.5, 2.7])
    def test_hits_target_count(self, T, target):
        rng = np.random.default_rng(7)
        eps = jnp.asarray(rng.uniform(-20.0, 20.0, size=(6, 6, 4)))
        w_norm = jnp.ones((6, 6)) / 36.0
        mu = S._solve_mu(eps, w_norm, target, 0.0, T, maxiter=50)
        count = float(jnp.sum(w_norm[..., None] * jax.nn.sigmoid((mu - eps) / T)))
        assert abs(count - target) < 1e-10
        # Warm start from a perturbed guess reaches the same mu.
        mu_warm = S._solve_mu(eps, w_norm, target, mu + 3.0 * T, T, maxiter=50)
        np.testing.assert_allclose(float(mu_warm), float(mu), atol=1e-9 * max(1.0, abs(float(mu))))

    def test_float32_low_filling_conserves_particles(self):
        """Particle count at low filling, in float32, where the tolerance bit.

        The CG path takes its occupations straight from ``_solve_mu`` (the
        capped-simplex projection that used to re-impose the count is gone), so
        the solve is only as particle-conserving as the chemical potential.  A
        count tolerance scaled to the *capacity* rather than to the target left
        a ~6e-5 electron error here out of a target of 1.0 -- invisible to a
        1e-4 absolute check at half filling, but a 6e-5 relative error that
        moved the converged energy in the fifth decimal.
        """
        import jax as _jax

        old = _jax.config.jax_enable_x64
        _jax.config.update("jax_enable_x64", False)
        try:
            nk, nb = 8, 2
            ham = np.zeros((nk, nk, nb, nb), dtype=np.complex64)
            ham[..., 0, 0] = -0.5
            ham[..., 1, 1] = 0.5
            kernel = HartreeFockKernel(
                weights=jnp.ones((nk, nk), dtype=jnp.float32),
                hamiltonian=jnp.asarray(ham),
                coulomb_q=jnp.full((nk, nk, 1, 1), 0.6, dtype=jnp.complex64),
                T=0.2,
            )
            assert kernel.h.dtype == jnp.complex64
            n_e = 1.0                      # 1 electron in 128 states
            capacity = float(kernel.weight_sum) * nb
            assert n_e < 0.01 * capacity, "fixture must be at low filling"

            result = solve(
                kernel, jnp.zeros_like(kernel.h), n_e,
                config=SolverConfig(max_iter=200, tol_E=1e-10),
            )
            assert bool(result.converged)
            for label, count in (
                ("occupations", float(jnp.sum(kernel.w2d[..., None] * result.p))),
                ("density", float(jnp.sum(
                    kernel.w2d * jnp.real(jnp.trace(result.density, axis1=-2, axis2=-1))
                ))),
            ):
                assert abs(count - n_e) < 1e-5 * n_e, (label, count)
        finally:
            _jax.config.update("jax_enable_x64", old)

    @staticmethod
    def _float32_bands(nk=8, nb=6, seed=0):
        rng = np.random.default_rng(seed)
        eps = jnp.asarray(rng.uniform(-20.0, 20.0, size=(nk, nk, nb)).astype(np.float32))
        w_norm = jnp.full((nk, nk), 1.0 / (nk * nk), dtype=jnp.float32)
        return eps, w_norm, float(jnp.sum(w_norm)) * nb

    def test_count_tolerance_tracks_the_filled_fraction(self):
        """Resolve the count relative to the occupied fraction, not the capacity.

        Float32 at low filling from a warm start well away from the answer --
        the regime where the early stop can fire before the Newton step has
        converged.  A tolerance scaled to the band capacity stops at a relative
        particle error of ~8e-4 here; scaling it to the target count reaches
        ~9e-9.  That matters because the CG path takes its occupations straight
        from this solve, so the particle count is only as good as ``mu``.
        """
        import jax as _jax

        old = _jax.config.jax_enable_x64
        _jax.config.update("jax_enable_x64", False)
        try:
            nk, nb, T, mu0 = 8, 6, 0.1, 25.0
            eps, w_norm, capacity = self._float32_bands(nk=nk, nb=nb, seed=8)
            target = np.float32(0.0005 * capacity)
            mu = S._solve_mu(
                eps, w_norm, target, np.float32(mu0), np.float32(T), maxiter=60,
            )
            count = float(jnp.sum(w_norm[..., None] * jax.nn.sigmoid((mu - eps) / T)))
            assert abs(count - float(target)) < 1e-6 * float(target)
        finally:
            _jax.config.update("jax_enable_x64", old)

    @pytest.mark.parametrize("filling", [0.5, 1.0 - 1e-3])
    def test_count_is_accurate_to_the_representable_limit(self, filling):
        """Away from low filling, hit the count to the precision of the sum itself.

        The particle count is a sum of order ``capacity`` sigmoids, so its
        absolute round-off floor in float32 is a few ULPs of the capacity; no
        choice of stopping tolerance can do better.
        """
        import jax as _jax

        old = _jax.config.jax_enable_x64
        _jax.config.update("jax_enable_x64", False)
        try:
            T = 0.1
            eps, w_norm, capacity = self._float32_bands()
            target = np.float32(filling * capacity)
            mu = S._solve_mu(eps, w_norm, target, np.float32(0.0), np.float32(T), maxiter=60)
            count = float(jnp.sum(w_norm[..., None] * jax.nn.sigmoid((mu - eps) / T)))
            assert abs(count - float(target)) < 8 * np.finfo(np.float32).eps * capacity
        finally:
            _jax.config.update("jax_enable_x64", old)

    def test_endpoints_are_infinite(self):
        eps = jnp.asarray(np.linspace(-1.0, 1.0, 8).reshape(2, 2, 2))
        w_norm = jnp.ones((2, 2)) / 4.0
        assert float(S._solve_mu(eps, w_norm, 0.0, 0.0, 0.1, maxiter=50)) == -np.inf
        assert float(S._solve_mu(eps, w_norm, 2.0, 0.0, 0.1, maxiter=50)) == np.inf

    def test_converged_warm_start_is_kept(self):
        """A warm start already at the solution must come back unchanged.

        The Newton update from a converged mu rounds back onto mu, which has
        just become a bracket end.  Treating that as an out-of-bracket step and
        bisecting from it throws the solution away, after which the iterate
        cycles through bisection midpoints for the rest of the budget and the
        returned mu depends on where the budget happens to run out.  (Ported
        from hartreefock.backends.native, where the unguarded solver showed exactly that.)
        """
        rng = np.random.default_rng(3)
        eps = jnp.asarray(rng.uniform(-20.0, 20.0, size=(6, 6, 4)))
        w_norm = jnp.ones((6, 6)) / 36.0
        T, target = 0.01, 1.5
        mu = float(S._solve_mu(eps, w_norm, target, 0.0, T, maxiter=50))
        tol = 4.0 * np.finfo(np.float64).eps * max(1.0, abs(mu))
        for maxiter in (1, 2, 3, 5, 8, 13, 25, 50):
            mu_warm = float(S._solve_mu(eps, w_norm, target, mu, T, maxiter=maxiter))
            assert abs(mu_warm - mu) <= tol, (maxiter, mu_warm, mu)

    def test_result_is_independent_of_the_iteration_budget(self):
        """Once converged the iterate stays put, so any sufficient budget agrees."""
        rng = np.random.default_rng(5)
        eps = jnp.asarray(rng.uniform(-20.0, 20.0, size=(6, 6, 4)))
        w_norm = jnp.ones((6, 6)) / 36.0
        T, target = 0.05, 1.5
        mus = [float(S._solve_mu(eps, w_norm, target, 0.0, T, maxiter=m)) for m in (25, 50, 200)]
        tol = 8.0 * np.finfo(np.float64).eps * max(1.0, abs(mus[0]))
        for m in mus[1:]:
            assert abs(m - mus[0]) <= tol, mus


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("remote_energy", [100.0, 1000.0])
def test_chemical_potential_resolves_low_temperature_despite_remote_bands(dtype, remote_energy):
    eps = jnp.asarray([[[0.0, remote_energy]]], dtype=dtype)
    weights = jnp.ones((1, 1), dtype=dtype)
    target = jnp.asarray(0.1, dtype=dtype)
    temperature = jnp.asarray(1e-12, dtype=dtype)
    mu = S._solve_mu(eps, weights, target, 0.0, temperature, maxiter=50)
    count = jnp.sum(jax.nn.sigmoid((mu - eps) / temperature))
    assert mu.dtype == dtype
    np.testing.assert_allclose(count, target, rtol=16 * np.finfo(dtype).eps, atol=0.0)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_chemical_potential_bracket_reaches_dilute_filling(dtype):
    eps = jnp.zeros((1, 1, 2), dtype=dtype)
    weights = jnp.ones((1, 1), dtype=dtype)
    target = jnp.asarray(1e-25, dtype=dtype)
    temperature = jnp.asarray(0.1, dtype=dtype)
    mu = S._solve_mu(eps, weights, target, 0.0, temperature, maxiter=50)
    count = jnp.sum(jax.nn.sigmoid((mu - eps) / temperature))
    assert mu.dtype == dtype
    np.testing.assert_allclose(count, target, rtol=64 * np.finfo(dtype).eps, atol=0.0)
