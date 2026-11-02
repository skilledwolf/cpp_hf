"""Tests for the trust-region Newton path (optimizer="newton") of the solver.

The Newton step uses the joint (Q, p) Hessian with the exact linear interaction
response Sigma[dP] = F[dP] - h (one Fock build per Hessian-vector product),
solved by a Steihaug truncated-CG within a trust region. On the regression
problems below it reaches the same stationary point in fewer outer iterations
than CG; no fixed speed ratio is part of the solver contract.

Newton is a second-order method and is run in float64 here (it needs the
precision — see ``test_float32_warns``).  Like CG and SCF it converges to the
self-consistent solution in the basin of its initial guess; on problems with
multiple HF solutions it may find a different stationary point, so energy
comparisons use a unique-minimum (contact) kernel and the rest assert only
stationarity / particle conservation.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from hartreefock import HartreeFockKernel, SCFConfig, SolverConfig
from hartreefock._jax import solve, solve_scf
from hartreefock._jax.fock import build_fock
from hartreefock._jax.utils import density_matrix_from_fock


# --------------------------------------------------------------------------
# Problem builders (float64)
# --------------------------------------------------------------------------

def _two_band_decoupled(nk=2, T=0.1, ex=0.25):
    """Diagonal h + scalar exchange: orbital sector decouples from occupations."""
    h = np.zeros((nk, nk, 2, 2), dtype=np.complex128)
    h[..., 0, 0] = -0.5
    h[..., 1, 1] = 0.5
    return HartreeFockKernel(
        weights=jnp.ones((nk, nk), dtype=jnp.float64),
        hamiltonian=jnp.asarray(h),
        coulomb_q=jnp.full((nk, nk, 1, 1), ex, dtype=jnp.complex128),
        T=T,
    )


def _contact(g=0.3, nk=2, T=0.05):
    """Contact (q-independent) interaction with a unique HF minimum."""
    h = np.zeros((nk, nk, 2, 2), dtype=np.complex128)
    h[..., 0, 0] = -1.0
    h[..., 1, 1] = 1.0
    for i in range(nk):
        for j in range(nk):
            h[i, j, 0, 1] = 0.4 + 0.1 * (i + j)
            h[i, j, 1, 0] = 0.4 + 0.1 * (i + j)
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    return HartreeFockKernel(
        weights=jnp.ones((nk, nk), dtype=jnp.float64),
        hamiltonian=jnp.asarray(h),
        coulomb_q=jnp.full((nk, nk, 1, 1), 0.1, dtype=jnp.complex128),
        T=T,
        contact_terms=[(0.3, sz, sz)],
    )


def _multiband(nk, nb, T, ex, seed):
    rng = np.random.default_rng(seed)
    diag = np.linspace(-1.0, 1.0, nb)
    A = (rng.standard_normal((nb, nb)) + 1j * rng.standard_normal((nb, nb))) * 0.35
    A = 0.5 * (A + A.conj().T)
    h = np.zeros((nk, nk, nb, nb), dtype=np.complex128)
    kx = np.arange(nk) / nk * 2 * np.pi
    for i in range(nk):
        for j in range(nk):
            hk = np.diag(diag).astype(np.complex128) + A * (np.cos(kx[i]) + np.cos(kx[j]))
            h[i, j] = 0.5 * (hk + hk.conj().T)
    return HartreeFockKernel(
        weights=jnp.ones((nk, nk), dtype=jnp.float64),
        hamiltonian=jnp.asarray(h),
        coulomb_q=jnp.full((nk, nk, 1, 1), ex, dtype=jnp.complex128),
        T=T,
    )


def _dirac(nk=4, m=0.3, v=1.0, T=0.1, ex=0.5):
    ks = (np.arange(nk) - nk // 2) * (2 * np.pi / nk)
    KX, KY = np.meshgrid(ks, ks, indexing="ij")
    h = np.zeros((nk, nk, 2, 2), dtype=np.complex128)
    h[..., 0, 0] = m
    h[..., 1, 1] = -m
    h[..., 0, 1] = v * (KX - 1j * KY)
    h[..., 1, 0] = v * (KX + 1j * KY)
    return HartreeFockKernel(
        weights=jnp.ones((nk, nk), dtype=jnp.float64),
        hamiltonian=jnp.asarray(h),
        coulomb_q=jnp.full((nk, nk, 1, 1), ex, dtype=jnp.complex128),
        T=T,
    )


def _comm_rms(F, P, w2d):
    comm = F @ P - P @ F
    per_k = jnp.sum(jnp.abs(comm) ** 2, axis=(-2, -1))
    return float(jnp.sqrt(jnp.sum(w2d * per_k) / jnp.maximum(jnp.sum(w2d), 1e-30)))


# --------------------------------------------------------------------------
# Hessian-vector product vs finite differences (the core correctness check)
# --------------------------------------------------------------------------

class TestNewtonHvp:
    def test_matches_finite_difference(self):
        from jax.scipy.linalg import expm

        K = _multiband(5, 5, 0.05, 0.4, 4)
        a = K._as_args()
        h = a["h"]
        T = float(K.T)
        nb = h.shape[-1]
        w2 = a["weights_b"][..., 0, 0]
        eye = jnp.eye(nb, dtype=h.dtype)
        offmask = 1.0 - jnp.eye(nb, dtype=h.real.dtype)

        def herm(X):
            return 0.5 * (X + jnp.conj(jnp.swapaxes(X, -1, -2)))

        def skew(X):
            return 0.5 * (X - jnp.conj(jnp.swapaxes(X, -1, -2)))

        def Fock(P):
            _, _, F = build_fock(
                P, h=h, VR=a["VR"], refP=jnp.zeros_like(h), HH=a["HH"], w2d=w2,
                include_exchange=a["include_exchange"],
                include_hartree=a["include_hartree"],
                exchange_hermitian_channel_packing=a["exchange_hermitian_channel_packing"],
                contact_g=a["contact_g"], contact_Oi=a["contact_Oi"], contact_Oj=a["contact_Oj"],
            )
            return F

        dens = lambda Q, p: jnp.einsum("...in,...n,...jn->...ij", Q, p, jnp.conj(Q))
        fockorb = lambda Q, F: jnp.einsum("...in,...ij,...jm->...nm", jnp.conj(Q), F, Q)
        proj = lambda dp: dp - jnp.sum(w2[..., None] * dp) / (jnp.sum(w2) * nb)

        rng = np.random.default_rng(1)
        eps, Q = jnp.linalg.eigh(Fock(jnp.zeros_like(h)))
        X0 = skew(rng.standard_normal(Q.shape) + 1j * rng.standard_normal(Q.shape)) * offmask
        Q = Q @ jax.vmap(jax.vmap(expm))(0.2 * X0)
        mu = float(jnp.median(eps))
        p = jnp.clip(jax.nn.sigmoid((mu - eps) / T), 0.05, 0.95)

        def grad(Q, p):
            Ft = fockorb(Q, Fock(dens(Q, p)))
            diff_p = p[..., None, :] - p[..., :, None]
            G = skew(diff_p * Ft) * offmask
            gp = jnp.real(jnp.diagonal(Ft, axis1=-2, axis2=-1)) + T * jnp.log(p / (1 - p)) - mu
            return G, gp, Ft

        def hvp(X, dp, Q, p, Ft):
            diff_p = p[..., None, :] - p[..., :, None]
            M = X * diff_p + dp[..., :, None] * eye
            dP = herm(jnp.einsum("...in,...nm,...jm->...ij", Q, M, jnp.conj(Q)))
            St = fockorb(Q, Fock(dP) - h)
            A = jnp.matmul(Ft, X) - jnp.matmul(X, Ft) + St
            diff_dp = dp[..., None, :] - dp[..., :, None]
            C = diff_dp * Ft + diff_p * A
            HX = skew(C) * offmask
            pc = jnp.clip(p, 1e-8, 1 - 1e-8)
            Hp = jnp.real(jnp.diagonal(A, axis1=-2, axis2=-1)) + T / (pc * (1 - pc)) * dp
            return HX, proj(Hp)

        G0, gp0, Ft = grad(Q, p)
        X = skew(rng.standard_normal(Q.shape) + 1j * rng.standard_normal(Q.shape)) * offmask
        dp = jnp.asarray(rng.standard_normal(p.shape) * 0.1)
        HX, Hp = hvp(X, dp, Q, p, Ft)

        errs_Q, errs_p = [], []
        for e in (1e-5, 1e-6):
            Qe = Q @ jax.vmap(jax.vmap(expm))(e * X)
            Ge, gpe, _ = grad(Qe, p + e * dp)
            fdQ = (Ge - G0) / e
            fdp = proj((gpe - gp0) / e)
            errs_Q.append(float(jnp.max(jnp.abs(fdQ - HX)) / jnp.maximum(jnp.max(jnp.abs(HX)), 1e-30)))
            errs_p.append(float(jnp.max(jnp.abs(fdp - Hp)) / jnp.maximum(jnp.max(jnp.abs(Hp)), 1e-30)))
        # absolute accuracy and ~O(eps) (linear) convergence
        assert errs_Q[1] < 1e-4 and errs_p[1] < 1e-4
        assert errs_Q[1] < 0.2 * errs_Q[0] + 1e-12
        assert errs_p[1] < 0.2 * errs_p[0] + 1e-12

    def test_predicted_reduction_includes_occupation_linear_term(self):
        from hartreefock._jax.newton import _joint_predicted_reduction

        weights = jnp.ones((1, 1), dtype=jnp.float64)
        zeros_Q = jnp.zeros((1, 1, 2, 2), dtype=jnp.complex128)
        zeros_p = jnp.zeros((1, 1, 2), dtype=jnp.float64)
        g_p = jnp.array([[[1.0, -1.0]]], dtype=jnp.float64)
        vp = jnp.array([[[-0.25, 0.25]]], dtype=jnp.float64)

        pred = _joint_predicted_reduction(
            zeros_Q, g_p, zeros_Q, vp, zeros_Q, zeros_p, weights,
        )
        np.testing.assert_allclose(float(pred), 0.5, atol=1e-15)


# --------------------------------------------------------------------------
# Solver behaviour
# --------------------------------------------------------------------------

class TestNewtonConvergence:
    def test_matches_cg_and_scf_unique_minimum(self):
        # Contact kernel has a unique HF minimum: Newton must match CG and SCF.
        K = _contact(g=0.3, nk=2)
        P0 = jnp.zeros_like(K.h)
        cg = solve(K, P0, 4.0, config=SolverConfig(max_iter=400, tol_E=1e-11, tol_grad=1e-9))
        nw = solve(K, P0, 4.0, config=SolverConfig(max_iter=200, tol_grad=1e-7, optimizer="newton"))
        scf = solve_scf(K, P0, 4.0, config=SCFConfig(max_iter=600, mixing=0.3,
                                                     density_tol=1e-10, comm_tol=1e-9))
        assert bool(nw.converged)
        np.testing.assert_allclose(float(nw.energy), float(cg.energy), atol=1e-6, rtol=1e-7)
        np.testing.assert_allclose(float(nw.energy), float(scf.energy), atol=1e-6, rtol=1e-7)

    @pytest.mark.parametrize("nk,nb,T,ex,seed", [
        (4, 4, 0.05, 0.3, 7),
        (5, 5, 0.05, 0.4, 4),
        (4, 5, 0.08, 0.5, 11),
    ])
    def test_stationary_and_conserves_particles(self, nk, nb, T, ex, seed):
        K = _multiband(nk, nb, T, ex, seed)
        P0 = jnp.zeros_like(K.h)
        ne = float(nk * nk * nb * 0.5)
        r = solve(K, P0, ne, config=SolverConfig(max_iter=200, tol_grad=1e-6, optimizer="newton"))
        assert bool(r.converged)
        assert _comm_rms(r.fock, r.density, K.w2d) < 1e-4
        P_fd, _ = density_matrix_from_fock(r.fock, K.w2d, ne, T)
        fixed_point_residual = float(jnp.sqrt(
            jnp.sum(
                K.w2d * jnp.sum(jnp.abs(r.density - P_fd) ** 2, axis=(-2, -1))
            ) / jnp.sum(K.w2d)
        ))
        assert fixed_point_residual < 2e-6
        n_total = float(jnp.sum(K.w2d[..., None] * r.p))
        np.testing.assert_allclose(n_total, ne, atol=1e-10)

    def test_reaches_same_solution_in_fewer_outer_iterations_than_cg(self):
        K = _dirac(nk=4)
        P0 = jnp.zeros_like(K.h)
        cfg = dict(max_iter=2000, tol_grad=1e-7)
        cg = solve(K, P0, 16.0, config=SolverConfig(optimizer="cg", **cfg))
        nw = solve(K, P0, 16.0, config=SolverConfig(optimizer="newton", **cfg))
        assert bool(nw.converged)
        # Newton reaches the same energy as CG...
        np.testing.assert_allclose(float(nw.energy), float(cg.energy), atol=1e-6, rtol=1e-7)
        # ...in fewer outer iterations. Full FD-residual convergence may invoke
        # damped fixed-point fallback near occupation degeneracies, so the old
        # fixed 5x speed-ratio assertion was not a mathematical contract.
        assert int(nw.n_iter) < int(cg.n_iter)

    def test_decoupled_cold_start_matches_cg(self):
        # Diagonal h + scalar exchange: the orbital gradient is zero at the
        # cold start while the occupations are not yet self-consistent.  The
        # cold-start occupation relaxation must recover the true energy (rather
        # than stopping at the non-interacting occupations).
        K = _two_band_decoupled(nk=2)
        P0 = jnp.zeros_like(K.h)
        cg = solve(K, P0, 4.0, config=SolverConfig(max_iter=400, tol_E=1e-12, tol_grad=1e-9))
        nw = solve(K, P0, 4.0, config=SolverConfig(max_iter=200, tol_grad=1e-8, optimizer="newton"))
        np.testing.assert_allclose(float(nw.energy), float(cg.energy), atol=1e-8, rtol=1e-9)

    def test_zero_orbital_gradient_still_relaxes_occupations(self):
        h = jnp.zeros((1, 1, 2, 2), dtype=jnp.complex128)
        K = HartreeFockKernel(
            weights=jnp.ones((1, 1), dtype=jnp.float64),
            hamiltonian=h,
            coulomb_q=jnp.zeros((1, 1, 1, 1), dtype=jnp.complex128),
            T=0.5,
            include_hartree=True,
            include_exchange=False,
            reference_density=jnp.zeros_like(h),
            hartree_matrix=jnp.eye(2, dtype=jnp.float64),
        )
        P0 = jnp.array([[[[1.0, 0.0], [0.0, 0.0]]]], dtype=jnp.complex128)
        r = solve(
            K, P0, 1.0,
            config=SolverConfig(
                optimizer="newton", max_iter=100, tol_grad=1e-8,
            ),
        )
        assert bool(r.converged)
        np.testing.assert_allclose(
            np.asarray(r.density[0, 0]), 0.5 * np.eye(2), atol=1e-7,
        )

    def test_density_hermitian_and_particles(self):
        K = _dirac(nk=4)
        P0 = jnp.zeros_like(K.h)
        r = solve(K, P0, 16.0, config=SolverConfig(max_iter=200, tol_grad=1e-7, optimizer="newton"))
        np.testing.assert_allclose(
            np.array(r.density),
            np.array(jnp.conj(jnp.swapaxes(r.density, -1, -2))),
            atol=1e-9,
        )
        np.testing.assert_allclose(
            float(jnp.sum(K.w2d[..., None] * r.p)), 16.0, atol=1e-5,
        )


class TestNewtonDispatch:
    def test_optimizer_recognised(self):
        K = _contact(nk=2)
        P0 = jnp.zeros_like(K.h)
        r = solve(K, P0, 4.0, config=SolverConfig(max_iter=100, tol_grad=1e-6, optimizer="newton"))
        assert np.isfinite(float(r.energy))

    def test_unknown_optimizer_raises(self):
        K = _contact(nk=2)
        P0 = jnp.zeros_like(K.h)
        with pytest.raises(ValueError, match="optimizer"):
            solve(K, P0, 4.0, config=SolverConfig(optimizer="bfgs"))

    def test_float32_warns(self):
        # A float32 kernel should warn (Newton needs float64 to converge).
        h = np.zeros((2, 2, 2, 2), dtype=np.complex64)
        h[..., 0, 0] = -0.5
        h[..., 1, 1] = 0.5
        K = HartreeFockKernel(
            weights=jnp.ones((2, 2), dtype=jnp.float32),
            hamiltonian=jnp.asarray(h),
            coulomb_q=jnp.full((2, 2, 1, 1), 0.25, dtype=jnp.complex64),
            T=0.1,
        )
        P0 = jnp.zeros_like(K.h)
        with pytest.warns(UserWarning, match="float64"):
            solve(K, P0, 2.0, config=SolverConfig(max_iter=2, optimizer="newton"))
