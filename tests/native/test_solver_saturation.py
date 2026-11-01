"""Finite-temperature occupation handling at saturated occupations.

Port of ``jax_hf/tests/test_solver_saturation.py``.

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
accuracy carries the particle count.  The native helpers are reached through
the native extension test hooks.
"""

from __future__ import annotations

import numpy as np
import pytest

from hartreefock import HartreeFockKernel, SCFConfig, SolverConfig, solve, solve_scf
from scipy.special import expit
from hartreefock._backends.native._compat import _native
from scipy.special import xlogy
from hartreefock._backends.native._lower import kernel_args

pytestmark = pytest.mark.skipif(_native is None, reason="native extension not built")

LOGIT_CLIP = 1.0e-14  # cpp/include/cpp_hf/types.hpp
OCC_CLIP = 1.0e-8
EPS64 = float(np.finfo(np.float64).eps)


# ---------------------------------------------------------------------------
# numpy mirrors of the solver's small helpers
# ---------------------------------------------------------------------------

def _herm(X):
    return 0.5 * (X + np.conj(np.swapaxes(X, -1, -2)))


def _skew(X):
    return 0.5 * (X - np.conj(np.swapaxes(X, -1, -2)))


def _density(Q, p):
    return np.einsum("...in,...n,...jn->...ij", Q, p, np.conj(Q))


def _orbital_basis(Q, F):
    return np.einsum("...in,...ij,...jm->...nm", np.conj(Q), F, Q)


def _logit(p):
    ps = np.clip(p, LOGIT_CLIP, 1.0 - LOGIT_CLIP)
    return np.log(ps) - np.log1p(-ps)


def _active(p):
    return (p > LOGIT_CLIP) & (p < 1.0 - LOGIT_CLIP)


def _project_uniform(dp, w):
    """Unmasked particle-conserving projection: the pre-fix formula."""
    return dp - np.sum(w[..., None] * dp) / (np.sum(w) * dp.shape[-1])


def _project_capped(p, w, n_target, maxiter=50):
    """Euclidean projection onto 0 <= p <= 1 at a fixed weighted count."""
    lo = -np.max(p) - 1.0
    hi = 1.0 - np.min(p) + 1.0
    for _ in range(maxiter):
        mid = 0.5 * (lo + hi)
        count = np.sum(w[..., None] * np.clip(p + mid, 0.0, 1.0))
        if count < n_target:
            lo = mid
        elif count > n_target:
            hi = mid
    return np.clip(p + 0.5 * (lo + hi), 0.0, 1.0)


def _norm_vec(x, w):
    return float(np.sqrt(np.sum(w[..., None] * x * x)))


def _norm_mat(X, w):
    return float(np.sqrt(np.sum(w * np.sum(np.abs(X) ** 2, axis=(-2, -1)))))


def _fock(kernel, P):
    """(F, E) from the native Fock build at density ``P``."""
    _, _, F, E = _native.build_fock_apply(
        kernel_args(kernel),
        np.ascontiguousarray(P, dtype=np.complex128), None,
    )
    return np.asarray(F), float(E)


def _wide_band_kernel(nk=6, nb=4, T=0.01, ex=0.3, seed=0):
    """Bands spanning +-20 at T=0.01: nearly every occupation saturates."""
    rng = np.random.default_rng(seed)
    h = np.zeros((nk, nk, nb, nb), dtype=np.complex128)
    h[..., np.arange(nb), np.arange(nb)] = np.linspace(-20.0, 20.0, nb)
    A = (rng.normal(scale=0.5, size=(nk, nk, nb, nb))
         + 1j * rng.normal(scale=0.5, size=(nk, nk, nb, nb)))
    h = h + 0.5 * (A + np.conj(A.swapaxes(-1, -2)))
    return HartreeFockKernel(
        weights=np.ones((nk, nk), dtype=np.float64) / (nk * nk),
        hamiltonian=h,
        coulomb_q=np.full((nk, nk, 1, 1), ex, dtype=np.complex128),
        T=T,
    )


def _initial_state(kernel, n_electrons):
    """(Q, p, mu, w_norm, n_target_norm) as the CG solver initialises them from P0 = 0."""
    F0, _ = _fock(kernel, np.zeros_like(kernel.h))
    eps0, Q = np.linalg.eigh(F0)
    w_norm = np.asarray(kernel.w2d) / float(kernel.weight_sum)
    ntn = n_electrons / float(kernel.weight_sum)
    mu = _native._solve_mu_inloop(eps0, w_norm, ntn, 0.0, float(kernel.T), 50)
    p = expit((mu - eps0) / float(kernel.T))
    return Q, p, mu, w_norm, ntn


def _free_energy(kernel, Q, p):
    _, E = _fock(kernel, _herm(_density(Q, p)))
    w = np.asarray(kernel.w2d)[..., None]
    return float(E + kernel.T * np.sum(w * (xlogy(p, p) + xlogy(1 - p, 1 - p))))


class TestSaturatedOccupationGradient:
    def test_gradient_vanishes_at_fermi_dirac_state(self):
        kernel = _wide_band_kernel()
        n_e = 1.5 * float(kernel.weight_sum)          # metallic: band 2 partially filled
        Q, p, mu, w_norm, _ = _initial_state(kernel, n_e)
        F0, _ = _fock(kernel, np.zeros_like(kernel.h))
        eps = np.real(np.diagonal(_orbital_basis(Q, F0), axis1=-2, axis2=-1))
        T = float(kernel.T)

        g, h, curv, active = _native._occupation_gradient(eps, p, T, w_norm)
        np.testing.assert_array_equal(active, _active(p))
        frac_saturated = 1.0 - float(np.mean(active))
        assert frac_saturated > 0.5, "fixture should saturate most occupations"
        assert bool(np.any(active)), "fixture should keep some bands partially filled"

        # (eps, p) is Fermi-Dirac consistent, so the projected gradient and its
        # preconditioned form must vanish relative to the band scale.  The only
        # residue is the logit round trip of nearly saturated active bands
        # (relative error ~eps/(1-p) in logit, damped by curv ~ (1-p) in h).
        scale = _norm_vec(eps, w_norm)
        assert _norm_vec(g, w_norm) < 1e-7 * scale
        assert _norm_vec(h, w_norm) < 1e-10 * scale
        assert np.all(curv[~active] == 0.0)
        assert np.all(g[~active] == 0.0)

        # The unmasked formula this replaces is not small: the clipped logit of
        # the saturated bands leaks through the uniform particle shift.
        g_unmasked = _project_uniform(eps + T * _logit(p), w_norm)
        assert _norm_vec(g_unmasked, w_norm) > 1e-2 * scale

    def test_preconditioned_direction_is_descent(self):
        kernel = _wide_band_kernel()
        n_e = 1.5 * float(kernel.weight_sum)
        Q, p, mu, w_norm, ntn = _initial_state(kernel, n_e)
        T = float(kernel.T)
        # Interacting Fock at the initial density: not stationary in p.
        F, _ = _fock(kernel, _herm(_density(Q, p)))
        eps = np.real(np.diagonal(_orbital_basis(Q, F), axis1=-2, axis2=-1))
        g_raw = eps + T * _logit(p)
        _, h, curv, _ = _native._occupation_gradient(eps, p, T, w_norm)
        w2d = np.asarray(kernel.w2d)

        # Analytic directional derivative along p -> p - tau h, in the same
        # (unnormalised) k-weights the free energy uses.
        slope = -float(np.sum(w2d[..., None] * g_raw * h))
        assert slope < 0.0
        Om0 = _free_energy(kernel, Q, p)
        tau = 1e-4
        p_t = _project_capped(p - tau * h, w_norm, ntn)
        fd_slope = (_free_energy(kernel, Q, p_t) - Om0) / tau
        np.testing.assert_allclose(fd_slope, slope, rtol=2e-2)
        # Particle number is preserved by the direction itself, to the
        # round-off of the curvature-weighted mean that enforces it.
        floor = 64 * EPS64 * float(np.sum(w_norm[..., None] * np.abs(curv * g_raw)))
        assert abs(float(np.sum(w_norm[..., None] * h))) < floor

    def test_masked_projection_reduces_to_uniform_when_all_active(self):
        rng = np.random.default_rng(3)
        eps = rng.normal(size=(3, 3, 5))
        p = rng.uniform(0.1, 0.9, size=(3, 3, 5))
        w = rng.uniform(0.5, 1.5, size=(3, 3))
        T = 0.3
        g, h, curv, active = _native._occupation_gradient(eps, p, T, w)
        assert bool(np.all(active))
        g_raw = eps + T * _logit(p)
        np.testing.assert_allclose(g, _project_uniform(g_raw, w), atol=1e-14)
        np.testing.assert_allclose(
            g, np.asarray(_native._project_occupation_tangent(g_raw, w)), atol=1e-14,
        )
        np.testing.assert_allclose(curv, p * (1.0 - p) / T, rtol=1e-14)
        # h = curv (g_raw - lam) with the curvature-weighted mean removed.
        lam = np.sum(w[..., None] * curv * g_raw) / np.sum(w[..., None] * curv)
        np.testing.assert_allclose(h, curv * (g_raw - lam), atol=1e-14)
        assert abs(float(np.sum(w[..., None] * h))) < 1e-13


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
        return Q, flat.reshape(nk1, nk2, nb), w_norm, ntn

    @staticmethod
    def _frozen_fock_occupations(kernel, Q, p, w_norm, ntn):
        F, _ = _fock(kernel, _herm(_density(Q, p)))
        eps = np.real(np.diagonal(_orbital_basis(Q, F), axis1=-2, axis2=-1))
        T = float(kernel.T)
        mu = _native._solve_mu_inloop(eps, w_norm, ntn, 0.0, T, 50)
        return eps, mu, expit((mu - eps) / T)

    def test_direction_descends_and_conserves_particles(self):
        kernel = _wide_band_kernel()
        n_e = 1.5 * float(kernel.weight_sum)
        Q, p, w_norm, ntn = self._integer_occupations(kernel, n_e)
        T = float(kernel.T)
        assert not np.any(_active(p)), "fixture must be fully saturated"
        np.testing.assert_allclose(float(np.sum(w_norm[..., None] * p)), ntn, atol=1e-14)

        eps, mu, p_fd = self._frozen_fock_occupations(kernel, Q, p, w_norm, ntn)
        d = p_fd - p
        assert _norm_vec(d, w_norm) > 1e-3, "fixture must be off the FD manifold"
        # Particle-conserving to the accuracy of the chemical-potential solve.
        assert abs(float(np.sum(w_norm[..., None] * d))) < 1e-12

        # Slope of Omega along p -> p + s d.  Writing x_i = (mu - eps_i)/T, so
        # that p_FD_i = sigmoid(x_i), each term of the sum is
        #   w_i T (logit(p_i) - x_i) (sigmoid(x_i) - p_i) <= 0
        # because logit and sigmoid are inverse increasing functions: the move
        # descends whatever the clipping does to the logit of a saturated band.
        # (Clipping can only flip a term for a band more than |log eps| * T
        # above mu, where p_FD_i itself is below machine epsilon.)
        w2d = np.asarray(kernel.w2d)
        integrand = (eps + T * _logit(p) - mu) * d
        assert float(np.max(integrand)) < 1e-15
        slope = float(np.sum(w2d[..., None] * integrand))
        np.testing.assert_allclose(
            slope, float(np.sum(w2d[..., None] * (eps + T * _logit(p)) * d)), atol=1e-10,
        )
        assert slope < 0.0

        # The full step is the frozen-Fock minimiser, so it must beat both the
        # starting point and every partial step along the same path.  (At an
        # exact bound the entropy derivative is not representable, so the value
        # of the objective, not a finite-difference slope, is the meaningful
        # check here; the slope is verified below on an unsaturated state.)
        omegas = [_free_energy(kernel, Q, p + si * d) for si in (0.0, 0.25, 0.5, 1.0)]
        assert omegas == sorted(omegas, reverse=True), omegas

    def test_slope_matches_finite_difference_when_unsaturated(self):
        # T = 5 against a +-20 band span keeps every occupation fractional, so
        # the entropy derivative is finite and the analytic slope is testable.
        kernel = _wide_band_kernel(T=5.0)
        n_e = 1.5 * float(kernel.weight_sum)
        Q, p_start, _mu0, w_norm, ntn = _initial_state(kernel, n_e)
        T = float(kernel.T)
        assert bool(np.all(_active(p_start)))
        # Perturb off the Fermi-Dirac manifold, conserving the particle count.
        rng = np.random.default_rng(11)
        noise = rng.normal(scale=0.05, size=p_start.shape)
        p = np.clip(p_start + _project_uniform(noise, w_norm), 0.02, 0.98)
        p = _project_capped(p, w_norm, ntn)

        eps, mu, p_fd = self._frozen_fock_occupations(kernel, Q, p, w_norm, ntn)
        d = p_fd - p
        assert abs(float(np.sum(w_norm[..., None] * d))) < 1e-12
        w2d = np.asarray(kernel.w2d)
        slope = float(np.sum(w2d[..., None] * (eps + T * _logit(p) - mu) * d))
        assert slope < 0.0
        Om0 = _free_energy(kernel, Q, p)
        step = 1e-6
        fd_slope = (_free_energy(kernel, Q, p + step * d) - Om0) / step
        np.testing.assert_allclose(fd_slope, slope, rtol=1e-3)

    def test_gradient_step_is_frozen_but_fermi_dirac_step_is_not(self):
        """A preconditioned gradient step cannot leave an exact bound."""
        kernel = _wide_band_kernel()
        n_e = 1.5 * float(kernel.weight_sum)
        Q, p, w_norm, ntn = self._integer_occupations(kernel, n_e)
        eps, mu, p_fd = self._frozen_fock_occupations(kernel, Q, p, w_norm, ntn)

        g, h, curv, active = _native._occupation_gradient(eps, p, float(kernel.T), w_norm)
        assert not bool(np.any(active))
        assert float(np.max(np.abs(curv))) == 0.0
        assert float(np.max(np.abs(g))) == 0.0
        assert float(np.max(np.abs(h))) == 0.0          # gradient step: no motion
        assert float(np.max(np.abs(p_fd - p))) > 1e-3   # Fermi-Dirac step: motion

    def test_solver_escapes_a_fully_saturated_start(self):
        """End to end: a cold integer-occupation seed must still reach the SCF answer."""
        kernel = _wide_band_kernel()
        n_e = 1.5 * float(kernel.weight_sum)
        Q, p, _w_norm, _ntn = self._integer_occupations(kernel, n_e)
        P0 = _herm(_density(Q, p))
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
        """(Q, p, Omega, Ft, eps, w_norm, ntn) exactly as the solver's first iteration sees them."""
        Q, p, _mu, w_norm, ntn = _initial_state(kernel, n_electrons)
        F, _ = _fock(kernel, _herm(_density(Q, p)))
        Ft = _orbital_basis(Q, F)
        eps = np.real(np.diagonal(Ft, axis1=-2, axis2=-1))
        return Q, p, _free_energy(kernel, Q, p), Ft, eps, w_norm, ntn

    @staticmethod
    def _directions(kernel, p, eps, Ft, w_norm, masked):
        """Preconditioned CG search directions, with and without the saturation mask."""
        T = float(kernel.T)
        nb = p.shape[-1]
        offdiag = 1.0 - np.eye(nb)
        diff_p = p[..., None, :] - p[..., :, None]
        G_Q = _skew(diff_p * Ft) * offdiag
        gap = eps[..., :, None] - eps[..., None, :]
        lam = max(T, 1e-3 * float(np.sqrt(np.mean(eps ** 2))))
        d_Q = G_Q / np.sqrt(gap ** 2 + lam ** 2)
        if masked:
            _g, d_p, _c, _a = _native._occupation_gradient(eps, p, T, w_norm)
        else:
            g_p = _project_uniform(eps + T * _logit(p), w_norm)
            p_safe = np.clip(p, OCC_CLIP, 1.0 - OCC_CLIP)
            d_p = _project_uniform(g_p * p_safe * (1.0 - p_safe) / T, w_norm)
        return d_Q, np.asarray(d_p)

    def _best_trial(self, kernel, Q, p, d_Q, d_p, Omega, w_norm, ntn, bt_max=8):
        """Lowest free energy over the solver's backtracking sequence."""
        V_d, lam_d = _native._cayley_spectral_setup(np.ascontiguousarray(d_Q))
        tau = min(1.0, 0.6 / max(_norm_mat(d_Q, w_norm), 1e-30))
        best = Omega
        for _ in range(bt_max):
            U = np.asarray(_native._cayley_unitary_from_spectrum(V_d, lam_d, tau))
            p_t = _project_capped(p - tau * d_p, w_norm, ntn)
            best = min(best, _free_energy(kernel, Q @ U, p_t))
            tau *= 0.5
        return best

    def test_masked_direction_yields_a_descent_step_where_unmasked_does_not(self):
        kernel = _wide_band_kernel()
        n_e = 1.5 * float(kernel.weight_sum)
        Q, p, Omega, Ft, eps, w_norm, ntn = self._cg_state(kernel, n_e)
        assert float(np.mean(_active(p))) < 0.5, "fixture must saturate"

        dq_m, dp_m = self._directions(kernel, p, eps, Ft, w_norm, masked=True)
        dq_u, dp_u = self._directions(kernel, p, eps, Ft, w_norm, masked=False)
        # Same orbital direction; only the occupation direction differs.
        np.testing.assert_allclose(dq_m, dq_u, atol=0)

        best_masked = self._best_trial(kernel, Q, p, dq_m, dp_m, Omega, w_norm, ntn)
        best_unmasked = self._best_trial(kernel, Q, p, dq_u, dp_u, Omega, w_norm, ntn)
        # The fix: some backtracking candidate now descends...
        assert best_masked < Omega, (best_masked, Omega)
        # ...where every candidate along the unmasked direction climbed, which is
        # what made the solver fall back to a plain Roothaan step every iteration.
        assert best_unmasked >= Omega, (best_unmasked, Omega)

    def test_first_iteration_does_not_exhaust_the_line_search(self):
        """End to end: the line search must not burn its whole budget.

        The native backend has no Fock-build hook, but every density about to enter a
        Fock build first passes through ``project_fn`` -- the current state,
        each line-search candidate, the Fermi-Dirac fixed point, and each
        fixed-point candidate (twice) -- so an identity projector counts an
        upper bound on the Fock builds.  The defect exhausted the line search
        from the very first iteration (``bt_max`` rejected candidates before
        the fixed-point step), so one iteration with a generous ``bt_max``
        separates the two regimes whatever the fixed-point step then costs.
        """
        kernel = _wide_band_kernel()
        n_e = 1.5 * float(kernel.weight_sum)
        P0 = np.zeros_like(kernel.h)
        counter = {"n": 0}

        def probe(P):
            counter["n"] += 1
            return P

        bt_max = 16
        solve(kernel, P0, n_e, config=SolverConfig(max_iter=1, bt_max=bt_max, project_fn=probe))
        # Exhausted: state + bt_max rejected candidates + fixed point + at least
        # one fixed-point candidate (2) + finalize (2) >= bt_max + 6.
        assert counter["n"] < bt_max + 3, counter["n"]

    def test_cg_matches_scf_on_the_saturated_problem(self):
        kernel = _wide_band_kernel()
        n_e = 1.5 * float(kernel.weight_sum)
        P0 = np.zeros_like(kernel.h)
        result = solve(kernel, P0, n_e, config=SolverConfig(max_iter=200, tol_E=1e-9))
        assert bool(result.converged)
        scf = solve_scf(
            kernel, P0, n_e,
            config=SCFConfig(max_iter=500, acceleration="diis",
                             density_tol=1e-10, comm_tol=1e-9),
        )
        assert bool(scf.converged)
        np.testing.assert_allclose(float(result.energy), float(scf.energy), atol=1e-7)

    def test_stuck_solve_exits_early_and_unconverged(self):
        """An unreachable tolerance must not burn the whole budget.

        Once neither a line-search candidate nor a fixed-point candidate
        descends, the state is unchanged and the CG history is reset; a second
        such iteration proves every further one an exact repeat (1 + 3*bt_max
        Fock builds each), so the loop exits with ``converged=False``.
        """
        kernel = _wide_band_kernel()
        n_e = 1.5 * float(kernel.weight_sum)
        P0 = np.zeros_like(kernel.h)
        result = solve(kernel, P0, n_e, config=SolverConfig(max_iter=300, tol_grad=1e-12))
        assert not bool(result.converged)
        assert int(result.n_iter) < 100, int(result.n_iter)
        hG = np.asarray(result.history["grad_norm"])
        assert hG.shape == (int(result.n_iter),)
        assert hG[-1] > 1e-12 and hG[-1] < 1e-6

    def test_newton_converges_and_matches_cg(self):
        # tol_grad=1e-7: the stationarity residual is a density difference whose
        # double-precision floor on this |Omega| ~ 35 fixture sits near 3e-8
        # (the line search cannot resolve free-energy decreases below ~1e-15),
        # so a 1e-8 target is not reliably reachable by either optimizer.
        kernel = _wide_band_kernel()
        n_e = 1.5 * float(kernel.weight_sum)
        P0 = np.zeros_like(kernel.h)
        cg = solve(kernel, P0, n_e, config=SolverConfig(max_iter=300, tol_grad=1e-7))
        nw = solve(kernel, P0, n_e,
                   config=SolverConfig(max_iter=100, tol_grad=1e-7, optimizer="newton"))
        assert bool(cg.converged), int(cg.n_iter)
        assert bool(nw.converged), int(nw.n_iter)
        np.testing.assert_allclose(float(nw.energy), float(cg.energy), atol=1e-7)


class TestSolveMuEarlyStop:
    @staticmethod
    def _count(eps, w_norm, mu, T):
        return float(np.sum(w_norm[..., None] * expit((mu - eps) / T)))

    @staticmethod
    def _bands(nk, nb, seed):
        rng = np.random.default_rng(seed)
        eps = rng.uniform(-20.0, 20.0, size=(nk, nk, nb))
        w_norm = np.full((nk, nk), 1.0 / (nk * nk))
        return eps, w_norm, float(np.sum(w_norm)) * nb

    @pytest.mark.parametrize("T", [0.01, 0.5])
    @pytest.mark.parametrize("target", [0.3, 1.5, 2.7])
    def test_hits_target_count(self, T, target):
        eps, w_norm, _ = self._bands(6, 4, seed=7)
        mu = _native._solve_mu_inloop(eps, w_norm, target, 0.0, T, 50)
        assert abs(self._count(eps, w_norm, mu, T) - target) < 1e-10
        # Warm start from a perturbed guess reaches the same mu.
        mu_warm = _native._solve_mu_inloop(eps, w_norm, target, mu + 3.0 * T, T, 50)
        np.testing.assert_allclose(mu_warm, mu, atol=1e-9 * max(1.0, abs(mu)))

    def test_converged_warm_start_is_kept(self):
        """A warm start already at the solution must come back unchanged.

        The Newton update from a converged mu rounds back onto mu, which has
        just become a bracket end.  Treating that as an out-of-bracket step and
        bisecting from it throws the solution away, after which the iterate
        cycles through bisection midpoints for the rest of the budget and the
        returned mu depends on where the budget happens to run out.
        """
        eps, w_norm, _ = self._bands(6, 4, seed=3)
        T, target = 0.01, 1.5
        mu = _native._solve_mu_inloop(eps, w_norm, target, 0.0, T, 50)
        tol = 4.0 * EPS64 * max(1.0, abs(mu))
        for maxiter in (1, 2, 3, 5, 8, 13, 25, 50):
            mu_warm = _native._solve_mu_inloop(eps, w_norm, target, mu, T, maxiter)
            assert abs(mu_warm - mu) <= tol, (maxiter, mu_warm, mu)

    def test_result_is_independent_of_the_iteration_budget(self):
        """Once converged the iterate stays put, so any sufficient budget agrees."""
        eps, w_norm, _ = self._bands(6, 4, seed=5)
        T, target = 0.05, 1.5
        mus = [_native._solve_mu_inloop(eps, w_norm, target, 0.0, T, m) for m in (25, 50, 200)]
        tol = 8.0 * EPS64 * max(1.0, abs(mus[0]))
        for m in mus[1:]:
            assert abs(m - mus[0]) <= tol, mus

    def test_low_filling_count_is_accurate_relative_to_target(self):
        """Resolve the count relative to the occupied fraction, not the capacity.

        A warm start well away from the answer at very low filling; the CG path
        takes its occupations straight from this solve, so the particle count
        is only as good as ``mu``.  A tolerance scaled to the band capacity
        would stop at a relative particle error of order capacity/n_target.
        """
        eps, w_norm, capacity = self._bands(8, 6, seed=8)
        T, mu0 = 0.1, 25.0
        target = 0.0005 * capacity
        mu = _native._solve_mu_inloop(eps, w_norm, target, mu0, T, 60)
        assert abs(self._count(eps, w_norm, mu, T) - target) < 1e-12 * target

    @pytest.mark.parametrize("filling", [0.5, 1.0 - 1e-3])
    def test_count_is_accurate_to_the_representable_limit(self, filling):
        """Away from low filling, hit the count to the precision of the sum itself.

        The particle count is a sum of order ``capacity`` sigmoids, so its
        absolute round-off floor is a few ULPs of the capacity; no choice of
        stopping tolerance can do better.
        """
        eps, w_norm, capacity = self._bands(8, 6, seed=0)
        T = 0.1
        target = filling * capacity
        mu = _native._solve_mu_inloop(eps, w_norm, target, 0.0, T, 60)
        assert abs(self._count(eps, w_norm, mu, T) - target) < 8 * EPS64 * capacity

    def test_endpoints_are_infinite(self):
        eps = np.linspace(-1.0, 1.0, 8).reshape(2, 2, 2)
        w_norm = np.ones((2, 2)) / 4.0
        assert _native._solve_mu_inloop(eps, w_norm, 0.0, 0.0, 0.1, 50) == -np.inf
        assert _native._solve_mu_inloop(eps, w_norm, 2.0, 0.0, 0.1, 50) == np.inf

    @pytest.mark.parametrize("distant_band", [1000.0, 10000.0])
    def test_zero_temperature_resolution_does_not_depend_on_distant_band(self, distant_band):
        eps = np.array([[[0.0, distant_band]]])
        weights = np.ones((1, 1))
        mu = _native._solve_mu_inloop(eps, weights, 0.1, 0.0, 0.0, 50)
        np.testing.assert_allclose(self._count(eps, weights, mu, 1e-12), 0.1,
                                   rtol=1e-13, atol=0.0)

    def test_extremely_dilute_filling_is_not_limited_by_fixed_thermal_bounds(self):
        eps = np.array([[[0.0, 0.2]]])
        weights = np.ones((1, 1))
        target = 1e-25
        mu = _native._solve_mu_inloop(eps, weights, target, 0.0, 0.1, 50)
        np.testing.assert_allclose(self._count(eps, weights, mu, 0.1), target,
                                   rtol=1e-13, atol=0.0)
