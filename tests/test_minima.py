import numpy as np
import pytest
from cpp_hf import HartreeFockKernel, SCFConfig, solve_minimum


def test_negative_curvature_recovery_escapes_exact_paramagnetic_saddle():
    K=HartreeFockKernel(np.ones((1,1)),np.zeros((1,1,2,2),complex),np.ones((1,1,1,1)),T=.1)
    P=np.eye(2,dtype=complex)[None,None]/2
    r=solve_minimum(K,P,1.)
    assert r.converged and r.stability.stable
    assert r.escapes>=1 and r.total_iterations>r.solution.iterations
    assert r.fock_builds>r.solution.history['fock_builds']
    assert np.all(np.diff(r.free_energies)<0)
    p=np.linalg.eigvalsh(r.solution.density_matrix)
    assert p.max()>.99 and p.min()<.01
    np.testing.assert_allclose(p.sum(),1.,atol=1e-13)


def test_zero_escape_budget_reports_saddle_without_false_success():
    K=HartreeFockKernel(np.ones((1,1)),np.zeros((1,1,2,2),complex),np.ones((1,1,1,1)),T=.1)
    r=solve_minimum(K,np.eye(2)[None,None]/2,1.,max_escapes=0)
    assert r.solution.converged and not r.converged and r.stability.stable is False
    assert r.escapes==0


def test_stability_recovery_rejects_conflicting_symmetry_constraint():
    K=HartreeFockKernel(np.ones((1,1)),np.zeros((1,1,2,2),complex),np.ones((1,1,1,1)),T=.1)
    with pytest.raises(ValueError,match='unconstrained'):
        solve_minimum(K,np.eye(2)[None,None]/2,1.,config=SCFConfig(project_fn=lambda P:P))


def test_nonstationary_iterate_is_not_reported_as_a_minimum():
    h=np.diag([-1.,1.]).astype(complex)[None,None]
    K=HartreeFockKernel(np.ones((1,1)),h,np.full((1,1,1,1),.1),T=.1)
    r=solve_minimum(K,np.eye(2)[None,None]/2,1.,
        config=SCFConfig(acceleration='rdiis',max_iter=1,density_tol=1e-8,comm_tol=1e-8))
    assert not r.converged and r.stability is None
    assert r.escapes==0 and r.total_iterations==1


def test_fermi_escape_preserves_unstable_tangent_and_saturated_occupations():
    from scipy.special import xlogy
    from cpp_hf import _native
    from cpp_hf.solver import _kernel_args_for_native
    from cpp_hf.stability import stability_analysis
    from cpp_hf.utils import density_matrix_from_fock
    rng=np.random.default_rng(37)
    Q=np.linalg.qr(rng.normal(size=(1,2,4,4))+1j*rng.normal(size=(1,2,4,4)))[0]
    F=(Q*np.array([-4.,-.03,.03,4.])[None,None,None,:])@Q.conj().swapaxes(-2,-1)
    w=np.array([[.2,.8]])
    P,_=density_matrix_from_fock(F,w,2.,.1)
    K=HartreeFockKernel(w,F,np.ones((1,2,1,1)),T=.1,reference_density=P)
    s=stability_analysis(K,P,2.)
    assert s.stationary and s.stable is False
    args=_kernel_args_for_native(K)
    response_args=dict(args,h=np.zeros_like(P),has_refP=False,refP=np.empty(0,complex))
    response=_native.build_fock_apply(response_args,s.direction,None)[2]/(1-s.lowest_eigenvalue)
    def retract(alpha):
        return density_matrix_from_fock(F+alpha*response,w,2.,K.T)[0]
    eps=1e-5
    np.testing.assert_allclose((retract(eps)-retract(-eps))/(2*eps),s.direction,
                               rtol=2e-6,atol=1e-8)
    def omega(density):
        E=_native.build_fock_apply(args,density,None)[3]
        p=np.clip(np.linalg.eigvalsh(density),0,1)
        return E+K.T*np.sum(w[...,None]*(xlogy(p,p)+xlogy(1-p,1-p)))
    energies=[]
    for sign in (-1,1):
        trial=retract(sign*.1)
        occupations=np.linalg.eigvalsh(trial)
        assert occupations.min()>-1e-13 and occupations.max()<1+1e-13
        np.testing.assert_allclose(np.sum(w[...,None]*occupations),2.,atol=1e-13)
        energies.append(omega(trial))
    assert min(energies)<omega(P)-1e-6


def test_minimum_uses_fermi_escape_before_clipped_density_trials(monkeypatch):
    import cpp_hf.minima as minima
    def clipping_not_needed(*args):
        raise AssertionError('Fermi escape should resolve this saddle')
    monkeypatch.setattr(minima,'_feasible_density',clipping_not_needed)
    K=HartreeFockKernel(np.ones((1,1)),np.zeros((1,1,2,2),complex),np.ones((1,1,1,1)),T=.1)
    r=solve_minimum(K,np.eye(2)[None,None]/2,1.)
    assert r.converged and r.escapes>=1


def test_minimum_retains_projected_escape_fallback(monkeypatch):
    import cpp_hf.minima as minima
    P=np.eye(2,dtype=complex)[None,None]/2
    monkeypatch.setattr(minima,'density_matrix_from_fock',lambda *args:(P.copy(),0.))
    K=HartreeFockKernel(np.ones((1,1)),np.zeros((1,1,2,2),complex),np.ones((1,1,1,1)),T=.1)
    r=solve_minimum(K,P,1.)
    assert r.converged and r.escapes>=1


def test_minimum_free_energy_describes_returned_state_not_stale_history(monkeypatch):
    from dataclasses import replace
    import cpp_hf.minima as minima
    original=minima.solve_scf
    def stale_history(*args,**kwargs):
        result=original(*args,**kwargs)
        history=dict(result.history,free_energy=np.zeros_like(result.history['free_energy']))
        return replace(result,history=history)
    monkeypatch.setattr(minima,'solve_scf',stale_history)
    K=HartreeFockKernel(np.ones((1,1)),np.zeros((1,1,2,2),complex),np.zeros((1,1,1,1)),T=.1)
    r=solve_minimum(K,np.eye(2)[None,None]/2,1.)
    assert r.converged
    np.testing.assert_allclose(r.free_energies,[-.2*np.log(2)],atol=1e-14)
