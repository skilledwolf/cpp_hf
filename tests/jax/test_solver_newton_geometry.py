"""Finite-difference and adjoint checks of the production Newton operator."""
import numpy as np
import pytest
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from scipy.linalg import expm

from hartreefock import HartreeFockKernel, SolverConfig
from hartreefock.backends.jax import solve
from hartreefock.backends.jax.fock import build_fock
from hartreefock.backends.jax.solver import (
    _deflation_bias, _deflation_response, _density_from_Qp,
    _fock_in_orbital_basis, _herm, _joint_hvp, _joint_inner_product,
    _occupation_gradient, _project_occ, _scale_joint_step_to_bounds,
    _skew_hermitian,
)


def _random_geometry(seed=12):
    rng = np.random.default_rng(seed)
    shape = (2, 2, 3, 3)

    def herm():
        return _herm(jnp.asarray(rng.normal(size=shape) + 1j * rng.normal(size=shape)))

    def tangent():
        X = _skew_hermitian(jnp.asarray(rng.normal(size=shape) + 1j * rng.normal(size=shape)))
        return X * (1.0 - jnp.eye(3)), jnp.asarray(rng.normal(size=shape[:-1]))

    _, Q = jnp.linalg.eigh(herm())
    p = jnp.asarray(rng.uniform(0.2, 0.8, size=shape[:-1]))
    return Q, p, herm, tangent


def _batch_expm(X):
    return jnp.asarray(np.asarray([expm(x) for x in np.asarray(X).reshape(-1, 3, 3)]).reshape(X.shape))


@pytest.mark.parametrize("weights", [np.full((2, 2), 0.013), np.array([[0.07, 0.3], [0.5, 0.11]])])
@pytest.mark.parametrize("deflated", [False, True])
def test_joint_hessian_parallel_transport_and_weighted_adjoint(weights, deflated):
    Q, p, herm, tangent = _random_geometry()
    w = jnp.asarray(weights)
    wn = w / w.sum()
    h = herm()
    O = herm()[0, 0]
    T = 0.7
    targets = jnp.stack([0.4 * herm(), 0.2 * herm()])
    sigma, length = 0.8, 1.3

    def response(dP):
        # Weighted, globally coupled, self-adjoint interaction.
        charge = jnp.sum(w[..., None, None] * jnp.real(jnp.conj(O) * dP))
        return 0.6 * dP + 0.4 * charge * O

    def fock(P):
        F = h + response(P)
        return F + (_deflation_bias(P, targets, w, sigma, length)[1] if deflated else 0.0)

    def gradient(Qi, pi):
        Ft = _fock_in_orbital_basis(Qi, fock(_density_from_Qp(Qi, pi)))
        G = _skew_hermitian((pi[..., None, :] - pi[..., :, None]) * Ft)
        gp = _occupation_gradient(jnp.real(jnp.diagonal(Ft, axis1=-2, axis2=-1)), pi, T, w)[0]
        return G, gp, Ft

    P = _density_from_Qp(Q, p)
    _, _, Ft = gradient(Q, p)

    def effective_response(dP):
        R = response(dP)
        return R + (_deflation_response(P, dP, targets, w, sigma, length) if deflated else 0.0)

    X, dp = tangent()
    dp = _project_occ(dp, w)
    Y, dq = tangent()
    dq = _project_occ(dq, w)
    HX, Hp = _joint_hvp(Q, p, Ft, X, dp, T, w, effective_response)
    HY, Hq = _joint_hvp(Q, p, Ft, Y, dq, T, w, effective_response)
    np.testing.assert_allclose(
        _joint_inner_product(X, dp, HY, Hq, wn),
        _joint_inner_product(HX, Hp, Y, dq, wn), rtol=2e-12, atol=2e-12,
    )
    step = 1e-5
    grads = []
    for sign in (-1, 1):
        t = sign * step
        Gt, gpt, _ = gradient(Q @ _batch_expm(t * X), p + t * dp)
        # Body gradients need Levi-Civita transport before differentiating.
        Uhalf = _batch_expm(0.5 * t * X)
        Gback = Uhalf @ Gt @ jnp.conj(jnp.swapaxes(Uhalf, -1, -2))
        grads.append((Gback, gpt))
    fdX = (grads[1][0] - grads[0][0]) / (2 * step) * (1.0 - jnp.eye(3))
    fdp = (grads[1][1] - grads[0][1]) / (2 * step)
    np.testing.assert_allclose(HX, fdX, rtol=2e-7, atol=3e-8)
    np.testing.assert_allclose(Hp, fdp, rtol=2e-7, atol=3e-8)


def test_fraction_to_boundary_preserves_whole_model_and_particle_tangent():
    Q, p, _, tangent = _random_geometry()
    w = jnp.array([[0.07, 0.3], [0.5, 0.11]])
    X, dp = tangent()
    dp = 9.0 * _project_occ(dp, w)
    Hx, Hp = 1.7 * X, -0.2 * dp
    xs, ps, hs, hps = _scale_joint_step_to_bounds(p, X, dp, Hx, Hp)
    factor = float(jnp.linalg.norm(ps) / jnp.linalg.norm(dp))
    assert 0 < factor < 1
    for actual, original in [(xs, X), (ps, dp), (hs, Hx), (hps, Hp)]:
        np.testing.assert_allclose(actual, factor * original, atol=2e-14)
    assert np.min(p + ps) > 0 and np.max(p + ps) < 1
    assert abs(float(jnp.sum(w[..., None] * ps))) < 1e-14


def test_stiff_coupled_occupation_problem_converges():
    w = jnp.array([[0.2, 0.7], [0.4, 0.1]])
    h = jnp.broadcast_to(jnp.array([[-0.6, 0.12], [0.12, 0.4]], dtype=jnp.complex128), (2, 2, 2, 2))
    K = HartreeFockKernel(weights=w, hamiltonian=h, coulomb_q=jnp.zeros((2, 2, 1, 1)),
                         T=0.2, include_exchange=False, include_hartree=True,
                         reference_density=jnp.zeros_like(h),
                         hartree_matrix=jnp.array([[30., 3.], [3., 20.]]))
    result = solve(K, jnp.zeros_like(h), 1.2, config=SolverConfig(optimizer="newton", max_iter=120, tol_grad=1e-8))
    assert bool(result.converged)
    np.testing.assert_allclose(jnp.sum(w[..., None] * result.p), 1.2, atol=1e-11)
    assert int(result.n_iter) < 30


@pytest.mark.parametrize("optimizer", ["cg", "newton"])
def test_projected_solver_keeps_density_spectrum_and_stationarity_consistent(optimizer):
    h = jnp.array([[[[-0.3, 0.4j], [-0.4j, 0.2]]]], dtype=jnp.complex128)
    K = HartreeFockKernel(weights=jnp.ones((1, 1)), hamiltonian=h,
                         coulomb_q=jnp.full((1, 1, 1, 1), 0.3), T=0.4)
    project = lambda A: A * jnp.eye(2)
    tolerance = 1e-8 if optimizer == "cg" else 1e-9
    result = solve(K, jnp.zeros_like(h), 1.0, config=SolverConfig(
        optimizer=optimizer, max_iter=100, tol_grad=tolerance, project_fn=project))
    assert bool(result.converged)
    np.testing.assert_allclose(result.density, _density_from_Qp(result.Q, result.p), atol=1e-13)
    np.testing.assert_allclose(result.fock, project(result.fock), atol=1e-13)
    eps = jnp.real(jnp.diagonal(result.fock, axis1=-2, axis2=-1))
    expected_p = jax.nn.sigmoid((result.mu - eps) / K.T)
    np.testing.assert_allclose(jnp.real(jnp.diagonal(result.density, axis1=-2, axis2=-1)), expected_p, atol=2e-9)


def test_custom_fock_with_captured_reference_has_exact_newton_response():
    h = jnp.array([[[[-0.3, 0.1j], [-0.1j, 0.2]]]], dtype=jnp.complex128)
    ref = jnp.array([[[[0.7, 0.05], [0.05, 0.3]]]], dtype=jnp.complex128)
    K = HartreeFockKernel(weights=jnp.ones((1, 1)), hamiltonian=h,
                         reference_density=ref, coulomb_q=jnp.full((1, 1, 1, 1), 0.4), T=0.3)

    def captured_builder(P, **kwargs):
        # Zeroing the arguments is insufficient to compute this response.
        kwargs["h"], kwargs["refP"] = h, ref
        return build_fock(P, **kwargs)

    cfg = SolverConfig(optimizer="newton", max_iter=80, tol_grad=1e-9)
    standard = solve(K, jnp.zeros_like(h), 0.8, config=cfg)
    custom = solve(K, jnp.zeros_like(h), 0.8, config=cfg, fock_build_fn=captured_builder)
    assert bool(standard.converged) and bool(custom.converged)
    np.testing.assert_allclose(custom.density, standard.density, atol=1e-11)
    np.testing.assert_allclose(custom.energy, standard.energy, atol=1e-12)


@pytest.mark.parametrize("optimizer", ["cg", "newton"])
def test_projected_degenerate_trials_use_the_physical_entropy(optimizer):
    from scipy.special import xlogy

    theta = 0.7
    symmetry = jnp.asarray([[np.cos(theta), np.sin(theta)],
                            [np.sin(theta), -np.cos(theta)]])
    project = lambda A: 0.5 * (A + symmetry @ A @ symmetry)
    h = jnp.zeros((1, 1, 2, 2), dtype=jnp.complex128)
    kernel = HartreeFockKernel(
        weights=jnp.ones((1, 1)), hamiltonian=h,
        coulomb_q=jnp.zeros((1, 1, 1, 1)), T=0.1,
        reference_density=jnp.zeros_like(h), include_exchange=False,
        include_hartree=True, hartree_matrix=jnp.diag(jnp.array([2.0, 0.0])),
    )

    def run(max_iter):
        return solve(kernel, jnp.zeros_like(h), 1.0, config=SolverConfig(
            optimizer=optimizer, max_iter=max_iter, tol_grad=1e-7,
            project_fn=project,
        ))

    def physical_free_energy(result):
        occupations = np.linalg.eigvalsh(np.asarray(result.density))
        entropy = -np.sum(xlogy(occupations, occupations)
                          + xlogy(1.0 - occupations, 1.0 - occupations))
        return float(result.energy) - kernel.T * entropy

    first, second = run(1), run(2)
    for result in (first, second):
        np.testing.assert_allclose(result.density, _density_from_Qp(result.Q, result.p), atol=1e-13)
        np.testing.assert_allclose(result.density, project(result.density), atol=1e-13)
    np.testing.assert_allclose(second.history["free_energy"][1], physical_free_energy(first), atol=2e-13)
    assert physical_free_energy(second) <= physical_free_energy(first) + 2e-13
    result = run(200)
    assert bool(result.converged)
    eps, Q = jnp.linalg.eigh(result.fock)
    expected = _density_from_Qp(Q, jax.nn.sigmoid((result.mu - eps) / kernel.T))
    np.testing.assert_allclose(result.density, expected, atol=2e-7)
    history = np.asarray(result.history["free_energy"])[:int(result.n_iter)]
    assert np.max(np.diff(history)) <= 2e-12
