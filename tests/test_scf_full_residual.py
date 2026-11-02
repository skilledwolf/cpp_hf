"""Independent objective, feasibility and fixed-point checks for RDIIS/Anderson."""


import jax.numpy as jnp
import numpy as np
import pytest
from _meshes import mesh_for

from hartreefock import HartreeFockKernel, SolverConfig, solve
from hartreefock._jax.fock import build_fock, free_energy, hf_energy
from hartreefock._jax.scf import _density_from_fock
from hartreefock._jax.scf_acceleration import _residual_coefficients


@pytest.fixture(autouse=True)
def double_precision(enable_x64):
    with enable_x64():
        yield


def charge_kernel(strength=30.0):
    h = np.broadcast_to(np.diag([-.2, .1, .3]), (2, 2, 3, 3)).astype(complex).copy()
    kernel = HartreeFockKernel(mesh_for(np.full((2, 2), .25)), h, np.zeros((2, 2, 1, 1)), hartree=strength * (np.eye(3) + .2 * np.ones((3, 3))), T=.5, exchange=False, reference=np.zeros_like(h))
    return kernel, np.broadcast_to(np.diag([.8, .4, .3]), h.shape).astype(complex).copy()


def independent_state(kernel, density, n_electrons, builder=build_fock):
    args = kernel._as_args()
    Sigma, H, F = builder(density, **{key: args[key] for key in (
        "h", "VR", "refP", "HH", "include_exchange", "include_hartree",
        "exchange_hermitian_channel_packing", "contact_g", "contact_Oi", "contact_Oj")},
        w2d=kernel.w2d)
    E = hf_energy(density, h=args["h"], Sigma=Sigma, H=H,
                  weights_b=args["weights_b"], refP=args["refP"])
    D, _ = _density_from_fock(F, weights=kernel.w2d, n_electrons=n_electrons,
                             temperature=kernel.T, level_shift=0.0)
    weights = np.asarray(kernel.w2d)
    dr = np.sqrt(np.sum(weights[..., None, None] * np.abs(np.asarray(D - density))**2) / weights.sum())
    cr = np.sqrt(np.sum(weights[..., None, None] * np.abs(np.asarray(F @ density - density @ F))**2) / weights.sum())
    return F, E, dr, cr


@pytest.mark.parametrize("method", ["rdiis", "anderson"])
def test_strong_hartree_occupation_residual_and_monotone_free_energy(method):
    kernel, P = charge_kernel()
    # These diagonal states have identically zero commutator, including the
    # initial state; convergence must resolve occupations, not just orbitals.
    result = solve(kernel, P, 1.5, config=SolverConfig(
        method=method, diis_size=8, mixing=1.0, max_iter=120,
        density_tol=1e-8, comm_tol=1e-8))
    assert result.converged, result.message
    F, E, dr, cr = independent_state(kernel, result.density, 1.5)
    assert dr <= 1e-8 and cr <= 1e-8
    np.testing.assert_allclose(result.fock, F, atol=2e-13)
    np.testing.assert_allclose(result.energy, E, atol=2e-13)
    assert np.all(np.diff(np.asarray(result.history["free_energy"])) <= 2e-12)
    p = np.linalg.eigvalsh(np.asarray(result.density))
    np.testing.assert_allclose(np.sum(np.asarray(kernel.w2d)[..., None] * p), 1.5, atol=2e-13)
    assert p.min() >= -1e-13 and p.max() <= 1 + 1e-13
    assert result.history["line_search_rejections"] > 0
    assert result.history["fock_builds"] <= 2 * result.n_iter + 2


@pytest.mark.parametrize("method", ["rdiis", "anderson"])
def test_budget_one_returns_repaired_exact_consistent_state(method):
    kernel, P = charge_kernel()
    P[..., 0, 0] = 1.8
    P[..., 2, 2] = -.8
    result = solve(kernel, P, 1.5, config=SolverConfig(method=method, max_iter=1))
    p = np.linalg.eigvalsh(np.asarray(result.density))
    assert p.min() >= -2e-14 and p.max() <= 1.0 + 2e-14
    np.testing.assert_allclose(np.sum(np.asarray(kernel.w2d)[..., None] * p), 1.5, atol=2e-13)
    F, E, dr, cr = independent_state(kernel, result.density, 1.5)
    np.testing.assert_allclose(result.fock, F, atol=2e-13)
    np.testing.assert_allclose(result.energy, E, atol=2e-13)
    np.testing.assert_allclose(result.history["density_residual"][-1], dr, atol=2e-13)
    np.testing.assert_allclose(result.history["commutator_residual"][-1], cr, atol=2e-13)
    assert not result.converged
    assert result.history["fock_builds"] == 2


def test_duplicate_residual_history_stays_finite_and_affine():
    R = jnp.array([[[[.3, .2j], [-.2j, -.3]]]], dtype=jnp.complex128)
    hist = jnp.stack((R, R, R, jnp.zeros_like(R)))
    c = _residual_coefficients(hist, jnp.array([True, True, True, False]), 2,
                               jnp.ones((1, 1)))
    assert np.isfinite(c).all()
    np.testing.assert_allclose(c.sum(), 1.0, atol=1e-14)
    assert c[3] == 0.0


@pytest.mark.parametrize("method", ["rdiis", "anderson"])
def test_entropy_descent_can_increase_internal_energy_and_shift_does_not_bias_solution(method):
    h = np.diag([-1., 1.]).astype(complex)[None, None]
    kernel = HartreeFockKernel(mesh_for(np.ones((1, 1))), h, np.zeros((1, 1, 1, 1)), T=1.0)
    P = np.diag([1., 0.]).astype(complex)[None, None]
    result = solve(kernel, P, 1.0, config=SolverConfig(
        method=method, mixing=1.0, level_shift=.8,
        density_tol=1e-9, comm_tol=1e-9, max_iter=20))
    assert result.converged, result.message
    expected = np.diag([1.0 / (1.0 + np.exp(-1.0)), 1.0 / (1.0 + np.exp(1.0))])
    np.testing.assert_allclose(np.asarray(result.density)[0, 0], expected, atol=1e-11)
    assert result.energy > -1.0
    assert result.history["free_energy"][-1] < -1.0


@pytest.mark.parametrize("n_electrons", [0.0, 3.0])
def test_exact_empty_full_fillings(n_electrons):
    kernel, P = charge_kernel()
    result = solve(kernel, P, n_electrons, config=SolverConfig(method="rdiis"))
    assert result.converged
    np.testing.assert_allclose(result.density, np.broadcast_to(
        np.eye(3) * (n_electrons / 3.0), P.shape), atol=2e-14)


def test_symmetry_projection_uses_entropy_of_returned_density():
    kernel, P = charge_kernel(1.0)
    # Group averaging under exchange of the first two orbitals preserves
    # positivity and trace, but does not commute with the starting density.
    swap = jnp.array([[0., 1., 0.], [1., 0., 0.], [0., 0., 1.]])

    def project(A):
        return .5 * (A + swap @ A @ swap)

    result = solve(kernel, P, 1.5, config=SolverConfig(
        method="rdiis", project_fn=project, max_iter=100,
        density_tol=1e-8, comm_tol=1e-8))
    assert result.converged, result.message
    np.testing.assert_allclose(result.density, project(result.density), atol=1e-13)
    Omega = free_energy(result.energy, jnp.linalg.eigvalsh(result.density), kernel.w2d, kernel.T)
    np.testing.assert_allclose(result.history["free_energy"][-1], Omega, atol=1e-13)


def test_safeguard_validation_and_nonfinite_input():
    old = SolverConfig(method="rdiis", diis_size=7)
    assert old.safeguard
    with pytest.raises(ValueError, match="safeguard"):
        SolverConfig(safeguard="yes")
    kernel, P = charge_kernel()
    P[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        solve(kernel, P, 1.5, config=old)


@pytest.mark.parametrize("method", ["rdiis", "anderson"])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_complex_coupled_problem_damping_trust_and_precision(method, dtype):
    rng = np.random.default_rng(712)
    A = rng.normal(size=(2, 2, 4, 4)) + 1j * rng.normal(size=(2, 2, 4, 4))
    h = (.5 * (A + A.conj().swapaxes(-1, -2))).astype(dtype)
    kernel = HartreeFockKernel(mesh_for(np.full((2, 2), .25)), h, np.full((2, 2, 1, 1), .4), hartree=2.0 * np.eye(4), T=.4, exchange=True, reference=np.zeros_like(h))
    tol = 3e-6 if dtype == np.complex64 else 1e-8
    result = solve(kernel, np.zeros_like(h), 1.6, config=SolverConfig(
        method=method, diis_size=8, mixing=.7, diis_damping=.8,
        trust_radius=.2, max_iter=150, density_tol=tol, comm_tol=tol))
    assert result.converged, result.message
    F, E, dr, cr = independent_state(kernel, result.density, 1.6)
    assert max(dr, cr) <= tol
    np.testing.assert_allclose(result.fock, F, atol=tol)
    np.testing.assert_allclose(result.energy, E, atol=tol)
    p = np.linalg.eigvalsh(np.asarray(result.density))
    assert p.min() >= -tol and p.max() <= 1 + tol
    np.testing.assert_allclose(np.sum(np.asarray(kernel.w2d)[..., None] * p), 1.6, atol=tol)
    assert np.max(np.abs(np.asarray(result.density).imag)) > .01


@pytest.mark.parametrize("method", ["linear", "oda", "rdiis"])
def test_legacy_scf_returns_the_density_whose_residual_was_certified(method):
    # This Fermi map amplifies density errors by 500.  A final undamped step
    # would move a certified state outside the requested tolerance.
    h = np.zeros((1, 1, 2, 2), dtype=complex)
    kernel = HartreeFockKernel(mesh_for(np.ones((1, 1))), h, np.zeros((1, 1, 1, 1)), hartree=1000.0 * np.eye(2), T=.5, exchange=False, reference=h)
    initial = np.diag([.5 + 1e-10, .5 - 1e-10]).astype(complex)[None, None]
    tolerance = 1e-7
    result = solve(kernel, initial, 1.0, config=SolverConfig(
        method=method, max_iter=1,
        density_tol=tolerance, comm_tol=tolerance,
    ))
    assert result.converged
    F, E, dr, cr = independent_state(kernel, result.density, 1.0)
    assert max(dr, cr) <= tolerance
    np.testing.assert_allclose(result.fock, F, atol=1e-12)
    np.testing.assert_allclose(result.energy, E, atol=1e-12)
    np.testing.assert_allclose(result.history["density_residual"][-1], dr, atol=1e-12)
