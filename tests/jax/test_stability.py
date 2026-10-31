import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import xlogy

from hartreefock import HartreeFockKernel, SCFConfig
from hartreefock.backends.jax import solve_scf, stability_analysis
from hartreefock.backends.jax.stability import _evaluate_fock, _fock_kwargs

jax.config.update("jax_enable_x64", True)


def test_degenerate_paramagnet_has_negative_exchange_curvature():
    kernel = HartreeFockKernel(np.ones((1, 1)), np.zeros((1, 1, 2, 2), complex),
                              np.ones((1, 1, 1, 1)), T=.1)
    P = np.eye(2, dtype=complex)[None, None]/2
    result = stability_analysis(kernel, P, 1.)
    assert result.stationary and result.eigen_converged and result.stable is False
    np.testing.assert_allclose(result.lowest_eigenvalue, -1.5, atol=1e-10)
    np.testing.assert_allclose(np.trace(result.direction[0, 0]), 0., atol=1e-14)


def test_repulsive_hartree_minimum_has_positive_curvature():
    h = np.diag([-.2, .3]).astype(complex)[None, None]
    kernel = HartreeFockKernel(np.ones((1, 1)), h, np.zeros((1, 1, 1, 1)), T=.5,
        include_exchange=False, include_hartree=True, reference_density=np.zeros_like(h),
        hartree_matrix=3*np.eye(2))
    result = solve_scf(kernel, np.eye(2)[None, None]/2, 1.,
        config=SCFConfig(acceleration="rdiis", density_tol=1e-9, comm_tol=1e-9))
    stability = stability_analysis(kernel, result.density, 1.)
    assert stability.stationary and stability.eigen_converged and stability.stable is True
    assert stability.lowest_eigenvalue > 0


def test_nonstationary_density_does_not_receive_stability_certificate():
    kernel = HartreeFockKernel(np.ones((1, 1)), np.diag([-1., 1.])[None, None].astype(complex),
                              np.full((1, 1, 1, 1), .1), T=.2)
    stability = stability_analysis(kernel, np.eye(2)[None, None]/2, 1.)
    assert not stability.stationary and stability.stable is None


def test_curvature_matches_actual_free_energy_second_difference():
    w = np.array([[.2, .8]])
    h = np.broadcast_to(np.array([[-.2, .1j], [-.1j, .3]]), (1, 2, 2, 2)).copy()
    kernel = HartreeFockKernel(w, h, np.full((1, 2, 1, 1), .2), T=.4)
    result = solve_scf(kernel, np.zeros_like(h), .8,
        config=SCFConfig(acceleration="rdiis", density_tol=1e-11, comm_tol=1e-11))
    stability = stability_analysis(kernel, result.density, .8)
    assert stability.stable
    D = stability.direction
    np.testing.assert_allclose(np.sum(w*np.trace(D, axis1=-2, axis2=-1)), 0., atol=1e-14)
    def omega(P):
        E = float(_evaluate_fock(jnp.asarray(P), **_fock_kwargs(kernel))[1])
        p = np.linalg.eigvalsh(P)
        return E+kernel.T*np.sum(w[..., None]*(xlogy(p, p)+xlogy(1-p, 1-p)))
    e, P = 1e-4, np.asarray(result.density)
    curvature = (omega(P+e*D)+omega(P-e*D)-2*omega(P))/(e*e*w.sum())
    np.testing.assert_allclose(curvature, stability.lowest_eigenvalue, rtol=2e-6, atol=1e-7)


def test_custom_fock_and_contact_response_match_the_same_physics():
    h = np.zeros((1, 1, 2, 2), complex)
    # For a two-flavor system the identity contact has tangent response -g D;
    # its scalar Hartree part is removed by the fixed-trace constraint.
    kernel = HartreeFockKernel(np.ones((1, 1)), h, np.zeros((1, 1, 1, 1)), T=.1,
                              contact_terms=[(1., np.eye(2), np.eye(2))])
    P = np.eye(2)[None, None]/2
    contact = stability_analysis(kernel, P, 1.)
    def custom(P, *, h, refP, w2d, **kwargs):
        rho = jnp.einsum("ij,ijab->ab", w2d, P-refP)
        sigma = (jnp.trace(rho)*jnp.eye(2)-rho)[None, None]
        return sigma, jnp.zeros_like(P), h+sigma
    replacement = stability_analysis(kernel, P, 1., fock_build_fn=custom)
    np.testing.assert_allclose(contact.lowest_eigenvalue, -1.5, atol=1e-10)
    np.testing.assert_allclose(replacement.lowest_eigenvalue, contact.lowest_eigenvalue, atol=1e-10)


def test_non_self_adjoint_custom_response_is_rejected():
    kernel = HartreeFockKernel(np.ones((1, 1)), np.zeros((1, 1, 2, 2), complex),
                              np.zeros((1, 1, 1, 1)), T=.1)
    def custom(P, *, h, **kwargs):
        # A Hermitian potential does not itself imply a variational response.
        sigma = jnp.zeros_like(P).at[..., 0, 0].set(P[..., 0, 1].real)
        return sigma, jnp.zeros_like(P), h+sigma
    with pytest.raises(ValueError, match="self-adjoint"):
        stability_analysis(kernel, np.eye(2)[None, None]/2, 1., fock_build_fn=custom)


def test_precision_limited_kernel_is_not_given_a_tight_certificate():
    kernel = HartreeFockKernel(np.ones((1, 1), np.float32),
                              np.zeros((1, 1, 2, 2), np.complex64),
                              np.zeros((1, 1, 1, 1), np.float32), T=.1)
    with pytest.raises(ValueError, match="precision"):
        stability_analysis(kernel, np.eye(2)[None, None]/2, 1.)


def test_failed_iterative_eigenvalue_solve_has_no_certificate(monkeypatch):
    from scipy.sparse.linalg import ArpackNoConvergence
    import hartreefock.backends.jax.stability as module
    h = np.zeros((1, 1, 9, 9), complex)
    kernel = HartreeFockKernel(np.ones((1, 1)), h, np.zeros((1, 1, 1, 1)), T=.1)
    def fail(*args, **kwargs):
        raise ArpackNoConvergence("forced nonconvergence", np.empty(0), np.empty((80, 0)))
    monkeypatch.setattr(module, "eigsh", fail)
    result = stability_analysis(kernel, np.eye(9)[None, None]/2, 4.5)
    assert result.stationary and not result.eigen_converged and result.stable is None


def test_eigenvalue_requires_an_independently_verified_residual(monkeypatch):
    import hartreefock.backends.jax.stability as module
    h = np.zeros((1, 1, 9, 9), complex)
    kernel = HartreeFockKernel(np.ones((1, 1)), h, np.zeros((1, 1, 1, 1)), T=.1)
    def wrong_eigenpair(operator, **kwargs):
        size = operator.shape[0]
        return np.array([2.]), np.ones((size, 1))/np.sqrt(size)
    monkeypatch.setattr(module, "eigsh", wrong_eigenpair)
    result = stability_analysis(kernel, np.eye(9)[None, None]/2, 4.5)
    assert result.stationary and not result.eigen_converged and result.stable is None
    np.testing.assert_allclose(result.eigen_residual, 1., atol=1e-12)


@pytest.mark.parametrize("number", [1e-60, 1e-6, 2.0 - 1e-6])
def test_host_fermi_density_preserves_particle_count_in_extreme_tails(number):
    from hartreefock.backends.jax.stability import _host_fermi_density
    fock = np.zeros((1, 1, 2, 2), complex)
    weights = np.ones((1, 1))
    density, _, _, occupations = _host_fermi_density(fock, weights, number, 0.1)
    np.testing.assert_allclose(np.trace(density[0, 0]).real, number, rtol=1e-13, atol=0.0)
    np.testing.assert_allclose(occupations, number / 2, rtol=1e-13, atol=0.0)
