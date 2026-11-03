"""Full-residual acceleration: monotone descent, feasibility, and Fock reuse."""
import numpy as np
from _meshes import mesh_for

from hartreefock import HartreeFockKernel, SolverConfig, solve
from hartreefock._jax import fock_operators
from hartreefock._jax.utils import density_matrix_from_fock


def charge_kernel(strength=30.):
    h=np.broadcast_to(np.diag([-.2,.1,.3]),(2,2,3,3)).astype(complex).copy()
    K=HartreeFockKernel(mesh_for(np.full((2, 2), .25)), h, np.zeros((2,2,1,1)), hartree=strength*(np.eye(3)+.2*np.ones((3,3))), T=.5, exchange=False, reference=np.zeros_like(h))
    return K,np.broadcast_to(np.diag([.8,.4,.3]),h.shape).astype(complex).copy()


def test_strong_diagonal_hartree_converges_monotonically():
    K,P=charge_kernel()
    r=solve(K,P,1.5,config=SolverConfig(method="rdiis",
        max_iter=100,density_tol=1e-8,comm_tol=1e-8,mixing=1.))
    assert r.converged
    D,_=density_matrix_from_fock(r.fock,K.w2d,1.5,K.T)
    np.testing.assert_allclose(r.density,D,atol=2e-8)
    assert np.all(np.diff(r.history['free_energy'])<=1e-11)
    np.testing.assert_allclose(np.sum(K.w2d[...,None]*np.linalg.eigvalsh(r.density)),1.5,atol=1e-12)
    # Rejected damping trials reuse F(P) along the affine density segment.
    assert r.history['fock_builds'] <= 2*r.n_iter+2


def test_returned_state_is_feasible_even_after_one_iteration():
    K,P=charge_kernel()
    P[...,0,0]=1.8;P[...,2,2]=-.8
    r=solve(K,P,1.5,config=SolverConfig(method="rdiis",max_iter=1))
    p=np.linalg.eigvalsh(r.density)
    assert p.min()>=-1e-14 and p.max()<=1+1e-14
    np.testing.assert_allclose(np.sum(K.w2d[...,None]*p),1.5,atol=1e-12)
    F,E=fock_operators(K).evaluate(r.density)
    np.testing.assert_allclose(r.fock,F,atol=1e-13)
    np.testing.assert_allclose(r.energy,E,atol=1e-13)
    assert not r.converged
