"""The FFT Fock must differentiate the reported energy on oblique meshes."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _meshes import mesh_for

from hartreefock import HartreeFockKernel, Mesh
from hartreefock._jax.operators import _evaluate_fock, _fock_kwargs

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("nk", [3, 4, 6])
@pytest.mark.parametrize("orbital_resolved", [False, True])
def test_sampled_exchange_preserves_energy_and_restores_its_gradient(nk, orbital_resolved):
    rng = np.random.default_rng(38)
    nb = 3
    qx, qy = np.meshgrid(np.arange(nk)-nk//2, np.arange(nk)-nk//2, indexing="ij")
    V = (1/np.sqrt(.3+qx*qx+qy*qy+qx*qy))[..., None, None]
    if orbital_resolved:
        V = V*np.array([[1, .4, .2], [.4, 1, .6], [.2, .6, 1]])
    w = np.full((nk, nk), .5)
    w /= w.sum()
    h = np.broadcast_to(np.diag([-.2, .1, .4]), (nk, nk, nb, nb)).astype(complex)
    def hermitian():
        a = rng.normal(size=h.shape)+1j*rng.normal(size=h.shape)
        return .5*(a+a.conj().swapaxes(-2, -1))
    P, R, A, B = hermitian(), hermitian(), hermitian(), hermitian()
    kernel = HartreeFockKernel(mesh_for(w), h, V, T=.2, reference=R)
    kwargs = _fock_kwargs(kernel)
    def evaluate(density, args=kwargs):
        return _evaluate_fock(jnp.asarray(density), **args)
    F = np.asarray(evaluate(P)[0])
    e = 1e-5
    numeric = (float(evaluate(P+e*A)[1])-float(evaluate(P-e*A)[1]))/(2*e)
    inner = lambda a, b: np.sum(w[..., None, None]*(a.conj()*b).real)
    np.testing.assert_allclose(numeric, inner(F, A), rtol=1e-9, atol=1e-10)
    response = dict(kwargs, h=jnp.zeros_like(kernel.h), refP=jnp.zeros_like(kernel.h))
    LA, LB = np.asarray(evaluate(A, response)[0]), np.asarray(evaluate(B, response)[0])
    np.testing.assert_allclose(inner(A, LB), inner(LA, B), rtol=1e-12, atol=1e-12)
    raw = dict(kwargs, VR=jnp.asarray(np.fft.fftn(np.fft.ifftshift(V, axes=(0, 1)), axes=(0, 1))))
    np.testing.assert_allclose(evaluate(P)[1], evaluate(P, raw)[1], rtol=1e-12, atol=1e-12)
    assert kernel.exchange_reciprocity_defect >= 0


@pytest.mark.parametrize("cell, nk, match", [
    (np.zeros((2, 2)), 2, "non-singular"),
    (np.array([[1., np.nan], [0., 1.]]), 2, "finite"),
    (np.eye(2), 0, "positive"),
])
def test_mesh_rejects_invalid_cells(cell, nk, match):
    with pytest.raises(ValueError, match=match):
        Mesh(cell, nk)


def test_kernel_rejects_a_hamiltonian_off_the_mesh():
    with pytest.raises(ValueError, match="h must have shape"):
        HartreeFockKernel(Mesh(np.eye(2), 2), np.zeros((1, 2, 2, 2), complex), np.ones((1, 2, 1, 1)), T=.2)
