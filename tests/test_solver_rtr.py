"""Tests for the trust-region Newton path (optimizer="newton") of the solver.

The Newton step uses the joint (Q,p) Hessian with the EXACT linear interaction
response Σ[δP] = F[δP]-h (one Fock build per Hessian-vector product), solved by
a Steihaug truncated-CG within a trust region. It can need far fewer outer
iterations than CG on stiff problems, although each outer iteration performs
multiple Hessian-vector Fock builds.

Note: like CG and SCF, Newton converges to the self-consistent solution in the
basin of its initial guess; on problems with multiple HF solutions it may find a
different stationary point.  Tests that compare energies therefore use a
unique-minimum kernel (the contact case) or assert only stationarity.
"""

import numpy as np
import pytest

from cpp_hf import (
    HartreeFockKernel,
    SCFConfig,
    SolverConfig,
    solve,
    solve_scf,
)
from cpp_hf.solver import _kernel_args_for_native


def _comm_rms(F, P, w2d):
    comm = F @ P - P @ F
    per_k = np.sum(np.abs(comm) ** 2, axis=(-2, -1))
    return float(np.sqrt(np.sum(w2d * per_k) / max(float(np.sum(w2d)), 1e-30)))


def _two_band(nk=2, T=0.1, ex=0.25):
    w = np.ones((nk, nk), dtype=np.float32)
    h = np.zeros((nk, nk, 2, 2), dtype=np.complex64)
    h[..., 0, 0] = -0.5
    h[..., 1, 1] = 0.5
    return HartreeFockKernel(weights=w, hamiltonian=h,
                             coulomb_q=np.full((nk, nk, 1, 1), ex, np.complex64), T=T)


def _contact(g=0.3, nk=2, T=0.05):
    w = np.ones((nk, nk), dtype=np.float32)
    h = np.zeros((nk, nk, 2, 2), dtype=np.complex64)
    h[..., 0, 0] = -1.0
    h[..., 1, 1] = 1.0
    for i in range(nk):
        for j in range(nk):
            h[i, j, 0, 1] = 0.4 + 0.1 * (i + j)
            h[i, j, 1, 0] = 0.4 + 0.1 * (i + j)
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex64)
    return HartreeFockKernel(weights=w, hamiltonian=h,
                             coulomb_q=np.full((nk, nk, 1, 1), 0.1, np.complex64),
                             T=T, contact_terms=[(np.float32(g), sz, sz)])


def _multiband(nk, nb, T, ex, seed):
    rng = np.random.default_rng(seed)
    w = np.ones((nk, nk), dtype=np.float32)
    diag = np.linspace(-1.0, 1.0, nb)
    A = (rng.standard_normal((nb, nb)) + 1j * rng.standard_normal((nb, nb))) * 0.35
    A = 0.5 * (A + A.conj().T)
    h = np.zeros((nk, nk, nb, nb), dtype=np.complex128)
    kx = np.arange(nk) / nk * 2 * np.pi
    for i in range(nk):
        for j in range(nk):
            hk = np.diag(diag).astype(np.complex128) + A * (np.cos(kx[i]) + np.cos(kx[j]))
            h[i, j] = 0.5 * (hk + hk.conj().T)
    return HartreeFockKernel(weights=w, hamiltonian=h.astype(np.complex64),
                             coulomb_q=np.full((nk, nk, 1, 1), ex, np.complex64), T=T)


# ---------------------------------------------------------------------------
# Hessian-vector product (the core; validated against finite differences)
# ---------------------------------------------------------------------------

class TestRtrHvp:
    def test_matches_finite_difference(self):
        from scipy.linalg import expm
        from cpp_hf import _native

        def offdiag(X):
            X = X.copy(); idx = np.arange(X.shape[-1]); X[..., idx, idx] = 0; return X
        def skew(X):
            return 0.5 * (X - np.conj(np.swapaxes(X, -1, -2)))
        def ft(Q, M):
            return np.einsum('...in,...ij,...jm->...nm', np.conj(Q), M, Q)
        def dens(Q, p):
            return np.einsum('...in,...n,...jn->...ij', Q, p, np.conj(Q))

        K = _multiband(5, 5, 0.05, 0.4, 4)
        args = _kernel_args_for_native(K)
        hm = np.asarray(K.h); T = float(K.T); nk1, nk2, nb = K.h.shape[:3]
        def Fock(P):
            return np.asarray(_native.build_fock_apply(args, np.ascontiguousarray(P, np.complex128), None)[2])

        rng = np.random.default_rng(1)
        eps, Q = np.linalg.eigh(Fock(np.zeros_like(K.h)))
        X0 = skew(offdiag(rng.standard_normal(Q.shape) + 1j * rng.standard_normal(Q.shape)))
        Q = Q @ np.stack([expm(0.2 * X0[i, j]) for i in range(nk1) for j in range(nk2)]).reshape(Q.shape)
        p = np.clip(1.0 / (1 + np.exp((eps - np.median(eps)) / T)), 0.05, 0.95)
        mu = float(np.median(eps))

        def grad(Q, p):
            Ft = ft(Q, Fock(dens(Q, p)))
            GQ = offdiag((p[..., None, :] - p[..., :, None]) * Ft)
            gp = np.real(np.diagonal(Ft, axis1=-2, axis2=-1)) + T * np.log(p / (1 - p)) - mu
            return GQ, gp, Ft

        GQ, gp, Ft = grad(Q, p)
        X = skew(offdiag(rng.standard_normal(Q.shape) + 1j * rng.standard_normal(Q.shape)))
        dp = rng.standard_normal(p.shape) * 0.1
        HX, Hp = _native._rtr_joint_hvp(
            args, np.ascontiguousarray(X), np.ascontiguousarray(dp, np.float64),
            np.ascontiguousarray(Q), np.ascontiguousarray(p, np.float64),
            np.ascontiguousarray(Ft))
        HX = np.asarray(HX); Hp = np.asarray(Hp)
        w2 = np.asarray(K.w2d)
        # finite difference at two scales; error must shrink ~linearly.
        errs_Q, errs_p = [], []
        for e in (1e-5, 1e-6):
            Qe = Q @ np.stack([expm(e * X[i, j]) for i in range(nk1) for j in range(nk2)]).reshape(Q.shape)
            GQe, gpe, _ = grad(Qe, p + e * dp)
            transport = np.stack([
                expm(0.5 * e * X[i, j])
                for i in range(nk1) for j in range(nk2)
            ]).reshape(Q.shape)
            GQe_back = transport @ GQe @ np.conj(transport.swapaxes(-2, -1))
            fdQ = offdiag((GQe_back - GQ) / e)
            fdp = (gpe - gp) / e
            fdp_pr = fdp - np.sum(w2[..., None] * fdp) / (np.sum(w2) * nb)
            errs_Q.append(np.max(np.abs(fdQ - HX)) / max(np.max(np.abs(HX)), 1e-30))
            errs_p.append(np.max(np.abs(fdp_pr - Hp)) / max(np.max(np.abs(Hp)), 1e-30))
        assert errs_Q[1] < 1e-4 and errs_p[1] < 1e-4
        assert errs_Q[1] < 0.2 * errs_Q[0] + 1e-12   # O(eps) convergence
        assert errs_p[1] < 0.2 * errs_p[0] + 1e-12


# ---------------------------------------------------------------------------
# Solver behaviour
# ---------------------------------------------------------------------------

class TestRtrConvergence:
    def test_matches_cg_and_scf_unique_minimum(self):
        # Contact kernel has a unique HF minimum: Newton must match CG and SCF.
        kernel = _contact(g=0.3, nk=2)
        P0 = np.zeros_like(kernel.h)
        cg = solve(kernel, P0, 4.0, config=SolverConfig(max_iter=400, tol_E=1e-10))
        nw = solve(kernel, P0, 4.0, config=SolverConfig(max_iter=200, tol_grad=1e-7, optimizer="newton"))
        scf = solve_scf(kernel, P0, 4.0,
                        config=SCFConfig(max_iter=600, mixing=0.3, density_tol=1e-9, comm_tol=1e-8))
        assert bool(nw.converged)
        np.testing.assert_allclose(float(nw.energy), float(cg.energy), atol=1e-6, rtol=1e-7)
        np.testing.assert_allclose(float(nw.energy), float(scf.energy), atol=1e-6, rtol=1e-7)

    def test_stationary_and_conserves_particles(self):
        kernel = _multiband(4, 4, 0.05, 0.3, 7)
        P0 = np.zeros_like(kernel.h)
        ne = float(4 * 4 * 4 * 0.5)
        r = solve(kernel, P0, ne, config=SolverConfig(max_iter=200, tol_grad=1e-6, optimizer="newton"))
        assert bool(r.converged)
        assert _comm_rms(np.asarray(r.fock), np.asarray(r.density), kernel.w2d) < 1e-4
        n_total = float(np.sum(kernel.w2d[..., None] * np.asarray(r.p)))
        np.testing.assert_allclose(n_total, ne, atol=1e-4)
        np.testing.assert_allclose(
            np.asarray(r.density), np.conj(np.swapaxes(np.asarray(r.density), -1, -2)), atol=1e-6)

    def test_newton_reaches_full_stationarity_threshold_before_cg(self):
        # On this stiff multi-k problem the stricter full Fermi-density
        # residual converges slowly for both optimizers. Compare them at the
        # same residual threshold; their final iteration counts correspond to
        # different accuracies and are not a meaningful speed ratio.
        kernel = _multiband(6, 6, 0.015, 0.4, 21)
        P0 = np.zeros_like(kernel.h)
        ne = float(6 * 6 * 6 * 0.5)
        cg = solve(kernel, P0, ne, config=SolverConfig(
            max_iter=2000, tol_E=0.0, tol_grad=1e-6, plateau_window=0))
        nw = solve(kernel, P0, ne, config=SolverConfig(
            max_iter=200, tol_grad=1e-6, optimizer="newton"))
        threshold = 1e-4
        cg_crossing = np.flatnonzero(
            np.asarray(cg.history["grad_norm"]) < threshold)
        nw_crossing = np.flatnonzero(
            np.asarray(nw.history["grad_norm"]) < threshold)
        assert cg_crossing.size
        assert nw_crossing.size
        assert int(nw_crossing[0]) < int(cg_crossing[0])
        # Same near-stationary solution as CG on this well-posed problem.
        np.testing.assert_allclose(float(nw.energy), float(cg.energy), atol=1e-4, rtol=1e-6)


class TestRtrConfig:
    def test_newton_optimizer_recognised(self):
        kernel = _two_band(nk=2, T=0.1)
        P0 = np.zeros_like(kernel.h)
        r = solve(kernel, P0, 4.0, config=SolverConfig(max_iter=100, tol_grad=1e-6, optimizer="newton"))
        # converges to a stationary point (energy finite, particles conserved)
        assert np.isfinite(float(r.energy))
        assert _comm_rms(np.asarray(r.fock), np.asarray(r.density), kernel.w2d) < 1e-4


@pytest.mark.parametrize("optimizer", ["cg", "newton"])
def test_degenerate_initial_chart_uses_projected_density_entropy(optimizer):
    from scipy.special import xlogy
    from cpp_hf.symmetry import make_project_fn
    from cpp_hf.utils import density_matrix_from_fock

    theta = 0.7
    symmetry = np.array([[np.cos(theta), np.sin(theta)],
                         [np.sin(theta), -np.cos(theta)]])
    project = make_project_fn(unitary_group=np.array([np.eye(2), symmetry]))
    h = np.zeros((1, 1, 2, 2), complex)
    kernel = HartreeFockKernel(
        np.ones((1, 1)), h, np.zeros((1, 1, 1, 1)), T=0.1,
        include_exchange=False, include_hartree=True,
        reference_density=np.zeros_like(h), hartree_matrix=np.diag([2.0, 0.0]),
    )
    # F[P0]=0 selects an arbitrary orbital gauge. The next Fock has a
    # nontrivial symmetry axis, so projecting an occupation trial changes its
    # spectrum. Entropy must use that new spectrum, including before stopping.
    def run(iterations):
        return solve(kernel, h, 1.0, config=SolverConfig(
            optimizer=optimizer, project_fn=project, max_iter=iterations, tol_grad=1e-7,
        ))

    def actual_free_energy(result):
        p = np.clip(np.linalg.eigvalsh(result.density), 0.0, 1.0)
        return float(result.energy + kernel.T * np.sum(
            xlogy(p, p) + xlogy(1.0 - p, 1.0 - p)))

    first, second, converged = run(1), run(2), run(200)
    np.testing.assert_allclose(second.history["free_energy"][1],
                               actual_free_energy(first), atol=1e-12, rtol=0.0)
    assert actual_free_energy(second) <= actual_free_energy(first) + 1e-12
    for result in (first, second, converged):
        reconstructed = (result.Q * result.p[..., None, :]) @ result.Q.conj().swapaxes(-2, -1)
        np.testing.assert_allclose(result.density, reconstructed, atol=1e-12)
    assert converged.converged
    fixed_point, _ = density_matrix_from_fock(converged.fock, kernel.w2d, 1.0, kernel.T)
    np.testing.assert_allclose(converged.density, project(fixed_point), atol=1e-7)
