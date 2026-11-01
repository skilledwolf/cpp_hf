"""Full-residual acceleration, response, feasibility, and Fock reuse checks."""
from dataclasses import replace
import numpy as np
import pytest
from scipy.special import expit
from hartreefock._backends.native._compat import _native
from hartreefock import HartreeFockKernel, SCFConfig, solve_scf
from hartreefock._backends.native._lower import kernel_args
from hartreefock._backends.native import density_matrix_from_fock


def charge_kernel(strength=30.):
    h=np.broadcast_to(np.diag([-.2,.1,.3]),(2,2,3,3)).astype(complex).copy()
    K=HartreeFockKernel(np.array([[.1,.2],[.3,.4]]),h,np.zeros((2,2,1,1)),T=.5,
        include_hartree=True,include_exchange=False,reference_density=np.zeros_like(h),
        hartree_matrix=strength*(np.eye(3)+.2*np.ones((3,3))))
    return K,np.broadcast_to(np.diag([.8,.4,.3]),h.shape).astype(complex).copy()


@pytest.mark.parametrize('method',['anderson','rdiis'])
@pytest.mark.parametrize('preconditioner',['none','hartree'])
def test_strong_diagonal_hartree_converges_monotonically(method,preconditioner):
    K,P=charge_kernel()
    r=solve_scf(K,P,1.5,config=SCFConfig(acceleration=method,preconditioner=preconditioner,
        max_iter=100,density_tol=1e-8,comm_tol=1e-8,mixing=1.))
    assert r.converged
    D,_=density_matrix_from_fock(r.fock,K.w2d,1.5,K.T)
    np.testing.assert_allclose(r.density,D,atol=2e-8)
    assert np.all(np.diff(r.history['free_energy'])<=1e-11)
    np.testing.assert_allclose(np.sum(K.w2d[...,None]*np.linalg.eigvalsh(r.density)),1.5,atol=1e-12)
    # Rejected damping trials reuse F(P) along the affine density segment.
    assert r.history['fock_builds'] <= 2*r.n_iter+2


@pytest.mark.parametrize('method',['anderson','rdiis'])
def test_returned_state_is_feasible_even_after_one_iteration(method):
    K,P=charge_kernel()
    P[...,0,0]=1.8;P[...,2,2]=-.8
    r=solve_scf(K,P,1.5,config=SCFConfig(acceleration=method,max_iter=1))
    p=np.linalg.eigvalsh(r.density)
    assert p.min()>=-1e-14 and p.max()<=1+1e-14
    np.testing.assert_allclose(np.sum(K.w2d[...,None]*p),1.5,atol=1e-12)
    _,_,F,E=_native.build_fock_apply(kernel_args(K),r.density,None)
    np.testing.assert_allclose(r.fock,F,atol=1e-13)
    np.testing.assert_allclose(r.energy,E,atol=1e-13)
    assert not r.converged


@pytest.mark.parametrize('degenerate',[False,True])
def test_fermi_response_matches_fixed_particle_central_difference(degenerate):
    rng=np.random.default_rng(817);shape=(2,2,3,3)
    A=rng.normal(size=shape)+1j*rng.normal(size=shape)
    Q=np.linalg.qr(A)[0]
    eps=np.broadcast_to(np.array([-.4,-.4 if degenerate else .1,.7]),(2,2,3)).copy()
    F=(Q*eps[...,None,:])@Q.conj().swapaxes(-2,-1)
    A=rng.normal(size=shape)+1j*rng.normal(size=shape)
    dF=.5*(A+A.conj().swapaxes(-2,-1))
    w=np.array([[.1,.2],[.3,.4]]);T=.3;ne=1.2
    _,mu=density_matrix_from_fock(F,w,ne,T)
    p=expit((mu-eps)/T)
    response=_native._fermi_response(dF,Q,eps,p,w,T)
    step=2e-5
    plus,_=density_matrix_from_fock(F+step*dF,w,ne,T)
    minus,_=density_matrix_from_fock(F-step*dF,w,ne,T)
    np.testing.assert_allclose(response,(plus-minus)/(2*step),rtol=3e-7,atol=1e-8)
    np.testing.assert_allclose(np.sum(w[...,None]*np.diagonal(response,axis1=-2,axis2=-1)),0.,atol=1e-13)


def test_checked_symmetry_blocks_preserve_energy_and_density():
    rng=np.random.default_rng(4);nk,nb=4,8
    h=np.zeros((nk,nk,nb,nb),complex)
    for off in (0,4):
        A=rng.normal(size=(nk,nk,4,4))+1j*rng.normal(size=(nk,nk,4,4))
        h[...,off:off+4,off:off+4]=.5*(A+A.conj().swapaxes(-2,-1))
    V=np.full((nk,nk,nb,nb),.2)
    K=HartreeFockKernel(np.ones((nk,nk))/(nk*nk),h,V,T=.4)
    cfg=SCFConfig(acceleration='rdiis',max_iter=100,density_tol=1e-8,comm_tol=1e-8)
    a=solve_scf(K,np.zeros_like(h),4.,config=cfg)
    b=solve_scf(K,np.zeros_like(h),4.,config=replace(cfg,block_sizes=(4,4)))
    assert a.converged and b.converged
    np.testing.assert_allclose(a.density,b.density,atol=1e-7)
    np.testing.assert_allclose(a.energy,b.energy,atol=1e-10)
    # A permitted coherence must never be silently dropped by a block hint.
    K.h[...,0,7]=K.h[...,7,0]=.3
    a=solve_scf(K,np.zeros_like(h),4.,config=cfg)
    b=solve_scf(K,np.zeros_like(h),4.,config=replace(cfg,block_sizes=(4,4)))
    np.testing.assert_allclose(a.density,b.density,atol=1e-8)
    assert np.max(abs(b.density[...,0,7]))>1e-3
