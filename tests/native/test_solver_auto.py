import numpy as np
import pytest
from hartreefock import HartreeFockKernel, SolverConfig, solve


@pytest.mark.parametrize('method',['cg','auto','newton'])
@pytest.mark.parametrize('exchange',[0.,.1])
def test_projected_solution_has_consistent_occupations_entropy_and_stationarity(method,exchange):
    h=np.array([[-.2,.3],[.3,.2]],complex)[None,None]
    K=HartreeFockKernel(np.ones((1,1)),h,np.full((1,1,1,1),exchange),T=.3)
    project=lambda P:P*np.eye(2)
    r=solve(K,np.zeros_like(h),1.,config=SolverConfig(optimizer=method,project_fn=project,tol_grad=1e-9,max_iter=200))
    assert r.converged
    np.testing.assert_allclose(r.density,(r.Q*r.p[...,None,:])@r.Q.conj().swapaxes(-2,-1),atol=1e-12)
    np.testing.assert_allclose(r.fock,project(r.fock),atol=1e-12)
    from hartreefock._backends.native import density_matrix_from_fock
    D,_=density_matrix_from_fock(r.fock,K.w2d,1.,K.T)
    np.testing.assert_allclose(r.density,D,atol=1e-9)
    from scipy.special import xlogy
    p=np.clip(np.linalg.eigvalsh(r.density),0,1)
    actual_free_energy=r.energy+K.T*np.sum(K.w2d[...,None]*(xlogy(p,p)+xlogy(1-p,1-p)))
    np.testing.assert_allclose(r.history['free_energy'][-1],actual_free_energy,rtol=0,atol=1e-11)
    assert np.all(np.diff(r.history['free_energy'])<=1e-12)
    if exchange:
        assert int(r.n_iter)>1
    else:
        expected=np.diag(1/(1+np.exp(np.diag(h[0,0]).real/K.T)))
        np.testing.assert_allclose(r.density[0,0],expected,rtol=0,atol=1e-9)


def test_auto_return_flags_and_one_iteration_budget():
    h=np.diag([-.2,.3]).astype(complex)[None,None]
    K=HartreeFockKernel(np.ones((1,1)),h,np.ones((1,1,1,1)),T=.3)
    r=solve(K,np.eye(2)[None,None]/2,1.,config=SolverConfig(optimizer='auto',max_iter=1,tol_grad=1e-9,
                                                              return_Q=False,return_fock=False,return_density=False))
    assert r.n_iter==1 and not r.converged
    assert r.Q is None and r.fock is None and r.density is None
    assert r.history['fock_builds']==2
    np.testing.assert_allclose(r.p.sum(),1.)
