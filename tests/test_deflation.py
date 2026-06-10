"""Tests for deflated trust-region Newton (finding distinct HF solutions) and
the project_fn wiring in solve_rtr.

Covers:
  1. project_fn is now honoured in the Newton path (was silently dropped).
  2. Deflation OFF is a strict no-op (the existing Newton path is unchanged).
  3. The deflation-bias math (Φ, S_pen gradient identity, diagonal coefficient)
     against the native helper, by finite differences.
  4. End-to-end: solve_deflated finds a distinct second HF minimum on a kernel
     with two known self-consistent solutions.
"""

import numpy as np
import pytest

from cpp_hf import HartreeFockKernel, SolverConfig, solve, solve_deflated


def _comm_rms(F, P, w2d):
    comm = F @ P - P @ F
    per_k = np.sum(np.abs(comm) ** 2, axis=(-2, -1))
    return float(np.sqrt(np.sum(w2d * per_k) / max(float(np.sum(w2d)), 1e-30)))


def _hybridized(nk=2, T=0.05, ex=0.25):
    """Two-band kernel with off-diagonal h (band hybridization): the converged
    density carries nonzero (0,1) coherence, so a diagonal-projector test bites."""
    w = np.ones((nk, nk), np.float32)
    h = np.zeros((nk, nk, 2, 2), np.complex64)
    h[..., 0, 0] = -0.5
    h[..., 1, 1] = 0.5
    h[..., 0, 1] = 0.3
    h[..., 1, 0] = 0.3
    return HartreeFockKernel(
        weights=w, hamiltonian=h,
        coulomb_q=np.full((nk, nk, 1, 1), ex, np.complex64), T=T,
    )


def _bistable(nk=1, T=0.02, ex=1.0, delta=0.1):
    """Near-degenerate two-band with strong exchange (Σ = -ex·P at one k-point).

    Two self-consistent solutions exist when ex > 2·delta:
      P = diag(1,0): E = -delta - ex/2  (ground; where cold-start from P0=0 lands)
      P = diag(0,1): E = +delta - ex/2  (metastable; deflation must discover it)
    """
    w = np.ones((nk, nk), np.float32)
    h = np.zeros((nk, nk, 2, 2), np.complex64)
    h[..., 0, 0] = -delta
    h[..., 1, 1] = +delta
    return HartreeFockKernel(
        weights=w, hamiltonian=h,
        coulomb_q=np.full((nk, nk, 1, 1), ex, np.complex64), T=T,
    )


# --- 1. project_fn now respected in the Newton path -------------------------
class TestProjectFnNewton:
    def test_diagonal_projector_enforced(self):
        K = _hybridized(nk=2, ex=0.25)
        P0 = np.zeros_like(K.h)
        ne = 4.0

        def proj_diag(P):
            P = np.array(P, copy=True)
            P[..., 0, 1] = 0.0
            P[..., 1, 0] = 0.0
            return P

        r = solve(K, P0, ne, config=SolverConfig(
            optimizer="newton", tol_grad=1e-6, max_iter=200, project_fn=proj_diag))
        off = float(np.max(np.abs(np.asarray(r.density)[..., 0, 1])))
        assert off < 1e-8, f"projector not enforced in Newton path: off-diag {off}"

        # control: without the projector the coherence is clearly nonzero
        r0 = solve(K, P0, ne, config=SolverConfig(
            optimizer="newton", tol_grad=1e-6, max_iter=200))
        assert float(np.max(np.abs(np.asarray(r0.density)[..., 0, 1]))) > 1e-3


# --- 2. deflation OFF is a strict no-op -------------------------------------
class TestDeflationNoOp:
    def test_off_matches_plain_newton(self):
        K = _hybridized(nk=2, ex=0.3)
        P0 = np.zeros_like(K.h)
        ne = 4.0
        base = dict(optimizer="newton", tol_grad=1e-7, max_iter=200)
        r1 = solve(K, P0, ne, config=SolverConfig(**base))
        # no targets, sigma 0
        r2 = solve(K, P0, ne, config=SolverConfig(deflation_sigma=0.0, **base))
        np.testing.assert_allclose(np.asarray(r2.density), np.asarray(r1.density), atol=1e-12)
        np.testing.assert_allclose(float(r2.energy), float(r1.energy), atol=1e-12)
        # a target present but sigma 0 -> still a strict no-op
        r3 = solve(K, P0, ne, config=SolverConfig(
            deflation_targets=np.asarray(r1.density)[None], deflation_sigma=0.0, **base))
        np.testing.assert_allclose(np.asarray(r3.density), np.asarray(r1.density), atol=1e-12)

    def test_cg_rejects_deflation(self):
        K = _hybridized(nk=2)
        P0 = np.zeros_like(K.h)
        with pytest.raises(ValueError):
            solve(K, P0, 4.0, config=SolverConfig(
                optimizer="cg",
                deflation_targets=np.asarray(P0)[None], deflation_sigma=1.0))


# --- 3. deflation-bias math (native helper, finite differences) -------------
class TestDeflationBiasMath:
    def test_gradient_identity_phi_and_diag(self):
        from cpp_hf import _native

        rng = np.random.default_rng(0)
        nk, nb = 2, 2
        # nonuniform weights to exercise the w2d weighting
        w2d = (np.arange(nk * nk).reshape(nk, nk).astype(np.float64) + 1.0)
        sigma, length = 0.7, 0.5

        def herm(shape):
            A = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
            return (0.5 * (A + np.conj(np.swapaxes(A, -1, -2)))).astype(np.complex128)

        P = herm((nk, nk, nb, nb))
        targets = np.stack([herm((nk, nk, nb, nb)) for _ in range(2)])

        def bias(Pin):
            return _native._deflation_bias(
                np.ascontiguousarray(Pin),
                np.ascontiguousarray(targets),
                np.ascontiguousarray(w2d, np.float64), sigma, length)

        Phi, S_pen, diag = bias(P)
        S_pen = np.asarray(S_pen)

        # gradient identity  dΦ/dP_k = w_k S_pen_k  (central FD along a Hermitian dir)
        dP = herm((nk, nk, nb, nb))
        analytic = float(np.sum(w2d[..., None, None] * np.real(np.conj(S_pen) * dP)))
        fd = []
        for eps in (1e-4, 1e-5):
            fd.append((float(bias(P + eps * dP)[0]) - float(bias(P - eps * dP)[0])) / (2 * eps))
        assert abs(fd[1] - analytic) < 1e-4 * (abs(analytic) + 1.0)
        assert abs(fd[1] - analytic) < 0.2 * abs(fd[0] - analytic) + 1e-9  # O(eps^2)

        # Φ and diag against the closed form
        inv2L2 = 1.0 / (2.0 * length * length)
        d2 = np.array([float(np.sum(w2d[..., None, None] * np.abs(P - t) ** 2))
                       for t in targets])
        phi = sigma * np.exp(-d2 * inv2L2)
        phip = -phi * inv2L2
        assert abs(float(Phi) - float(np.sum(phi))) < 1e-6 * (abs(float(np.sum(phi))) + 1.0)
        assert abs(float(diag) - 2.0 * float(np.sum(phip))) < 1e-6 * (2 * abs(float(np.sum(phip))) + 1.0)

    def test_empty_targets_is_zero(self):
        from cpp_hf import _native
        nk, nb = 2, 2
        w2d = np.ones((nk, nk), np.float64)
        P = np.zeros((nk, nk, nb, nb), np.complex128)
        empty = np.empty((0, nk, nk, nb, nb), np.complex128)
        Phi, S_pen, diag = _native._deflation_bias(P, empty, w2d, 1.0, 1.0)
        assert float(Phi) == 0.0 and float(diag) == 0.0
        assert np.max(np.abs(np.asarray(S_pen))) == 0.0


# --- 4. end-to-end: deflation finds a distinct second minimum ---------------
class TestDeflationFindsSolutions:
    def test_finds_distinct_second_minimum(self):
        K = _bistable(nk=1, T=0.02, ex=1.0, delta=0.1)
        P0 = np.zeros_like(K.h)
        ne = 1.0
        base = SolverConfig(optimizer="newton", tol_grad=1e-6, max_iter=300)

        res = solve_deflated(K, P0, ne, base_config=base, n_solutions=2, seed=0)
        assert res.n_found >= 2, f"only found {res.n_found} solution(s)"

        # both are genuine (unbiased) stationary points
        for s in res.solutions:
            assert _comm_rms(np.asarray(s.fock), np.asarray(s.density), K.w2d) < 1e-3

        # the two densities are clearly distinct (different polarization)
        d = np.asarray(res.solutions[0].density) - np.asarray(res.solutions[1].density)
        assert float(np.max(np.abs(d))) > 1e-2

        # deflated best is no worse than the cold start, and matches the analytic ground
        cold = solve(K, P0, ne, config=base)
        assert float(res.best.energy) <= float(cold.energy) + 1e-7
        np.testing.assert_allclose(float(res.best.energy), -0.6, atol=2e-2)
        # energies bracket the two analytic solutions  (-0.6 ground, -0.4 metastable)
        np.testing.assert_allclose(sorted(res.energies)[:2], [-0.6, -0.4], atol=3e-2)
