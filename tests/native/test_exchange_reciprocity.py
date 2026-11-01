"""The Fock gradient must differentiate the scalar energy on oblique grids."""
import numpy as np
import pytest
from hartreefock._backends.native._compat import _native
from hartreefock import HartreeFockKernel
from hartreefock._backends.native._lower import kernel_args


@pytest.mark.parametrize('nk',[3,4,6])
@pytest.mark.parametrize('orbital_resolved',[False,True])
def test_nonreciprocal_sampled_exchange_has_exact_energy_gradient(nk,orbital_resolved):
    rng=np.random.default_rng(38);nb=3
    x=np.arange(nk)-nk//2
    qx,qy=np.meshgrid(x,x,indexing='ij')
    V=1/np.sqrt(.3+qx*qx+qy*qy+qx*qy)
    V=V[...,None,None]
    if orbital_resolved:V=V*np.array([[1,.4,.2],[.4,1,.6],[.2,.6,1]])
    w=rng.uniform(.1,1,size=(nk,nk));w/=w.sum()
    h=np.broadcast_to(np.diag([-.2,.1,.4]),(nk,nk,nb,nb)).astype(complex).copy()
    def hermitian():
        a=rng.normal(size=h.shape)+1j*rng.normal(size=h.shape)
        return .5*(a+a.conj().swapaxes(-2,-1))
    P,R,A,B=hermitian(),hermitian(),hermitian(),hermitian()
    K=HartreeFockKernel(w,h,V,T=.2,reference_density=R)
    args=kernel_args(K)
    def evaluate(P):return _native.build_fock_apply(args,P,None)
    F=evaluate(P)[2];e=1e-5
    numeric=(evaluate(P+e*A)[3]-evaluate(P-e*A)[3])/(2*e)
    analytic=np.sum(w[...,None,None]*(F.conj()*A).real)
    np.testing.assert_allclose(numeric,analytic,rtol=1e-9,atol=1e-10)
    response_args=dict(args,h=np.zeros_like(h),has_refP=False,refP=np.empty(0,complex))
    LA=_native.build_fock_apply(response_args,A,None)[2]
    LB=_native.build_fock_apply(response_args,B,None)[2]
    inner=lambda a,b:np.sum(w[...,None,None]*(a.conj()*b).real)
    np.testing.assert_allclose(inner(A,LB),inner(LA,B),rtol=1e-12,atol=1e-12)
    # Symmetrizing a quadratic form changes its gradient, not its value.
    raw_args=dict(args,VR=np.ascontiguousarray(np.fft.fftn(np.fft.ifftshift(V,axes=(0,1)),axes=(0,1))))
    raw_E=_native.build_fock_apply(raw_args,P,None)[3]
    np.testing.assert_allclose(evaluate(P)[3],raw_E,rtol=1e-12,atol=1e-12)
