import numpy as np
import pytest
from cpp_hf import HartreeFockKernel, SCFConfig, solve_minimum, solve_scf
from cpp_hf.stability import stability_analysis


def test_degenerate_paramagnet_has_negative_exchange_curvature():
    # P=I/2 is exactly self-consistent, but exchange > 4T makes it unstable.
    K=HartreeFockKernel(np.ones((1,1)),np.zeros((1,1,2,2),complex),
                       np.ones((1,1,1,1)),T=.1)
    P=np.eye(2,dtype=complex)[None,None]/2
    s=stability_analysis(K,P,1.)
    assert s.stationary and s.eigen_converged and s.stable is False
    np.testing.assert_allclose(s.lowest_eigenvalue,-1.5,atol=1e-10)
    np.testing.assert_allclose(np.trace(s.direction[0,0]),0.,atol=1e-14)


def test_repulsive_hartree_minimum_has_positive_curvature():
    h=np.diag([-.2,.3]).astype(complex)[None,None]
    K=HartreeFockKernel(np.ones((1,1)),h,np.zeros((1,1,1,1)),T=.5,
        include_exchange=False,include_hartree=True,reference_density=np.zeros_like(h),
        hartree_matrix=3*np.eye(2))
    r=solve_scf(K,np.eye(2)[None,None]/2,1.,config=SCFConfig(acceleration='rdiis',density_tol=1e-9,comm_tol=1e-9))
    s=stability_analysis(K,r.density_matrix,1.)
    assert s.stationary and s.eigen_converged and s.stable is True
    assert s.lowest_eigenvalue>0


def test_nonstationary_density_does_not_receive_stability_certificate():
    K=HartreeFockKernel(np.ones((1,1)),np.diag([-1.,1.])[None,None].astype(complex),
                       np.full((1,1,1,1),.1),T=.2)
    s=stability_analysis(K,np.eye(2)[None,None]/2,1.)
    assert not s.stationary and s.stable is None


@pytest.mark.parametrize("number", [1e-6, 2.0 - 1e-6])
def test_flat_band_minimum_is_certified_near_empty_and_full_filling(number):
    h = np.zeros((1, 1, 2, 2), complex)
    kernel = HartreeFockKernel(np.ones((1, 1)), h, np.zeros((1, 1, 1, 1)), T=0.1)
    density = np.eye(2)[None, None] * (number / 2.0)
    result = solve_minimum(kernel, density, number)
    assert result.converged and result.stability.stable
    assert result.stability.density_residual < 1e-12
    assert result.escapes == 0


def test_curvature_matches_physical_free_energy_second_difference():
    from scipy.special import xlogy
    from cpp_hf import _native
    from cpp_hf.solver import _kernel_args_for_native
    w=np.array([[.2,.8]])
    h=np.broadcast_to(np.array([[-.2,.1j],[-.1j,.3]]),(1,2,2,2)).copy()
    K=HartreeFockKernel(w,h,np.full((1,2,1,1),.2),T=.4)
    r=solve_scf(K,np.zeros_like(h),.8,config=SCFConfig(acceleration='rdiis',density_tol=1e-11,comm_tol=1e-11))
    s=stability_analysis(K,r.density_matrix,.8)
    assert s.stable
    D=s.direction
    np.testing.assert_allclose(np.sum(w*np.trace(D,axis1=-2,axis2=-1)),0,atol=1e-14)
    def omega(P):
        E=_native.build_fock_apply(_kernel_args_for_native(K),P,None)[3]
        p=np.linalg.eigvalsh(P)
        return E+K.T*np.sum(w[...,None]*(xlogy(p,p)+xlogy(1-p,1-p)))
    e=1e-4;P=r.density_matrix
    curvature=(omega(P+e*D)+omega(P-e*D)-2*omega(P))/(e*e*w.sum())
    np.testing.assert_allclose(curvature,s.lowest_eigenvalue,rtol=2e-6,atol=1e-7)
