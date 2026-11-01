"""Independent geometric identities needed by the Newton Krylov solver."""
import numpy as np
import pytest

from hartreefock._backends.native._compat import _native
from hartreefock import HartreeFockKernel, SolverConfig, solve
from hartreefock._backends.native._lower import kernel_args


@pytest.mark.parametrize("weight_scale", [1.0, 0.001])
def test_joint_hessian_is_self_adjoint_away_from_stationarity(weight_scale):
    rng = np.random.default_rng(713)
    shape = (2, 2, 3, 3)
    w = weight_scale * np.array([[0.1, 0.2], [0.3, 0.4]])
    A = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    h = 0.5 * (A + A.conj().swapaxes(-2, -1))
    K = HartreeFockKernel(w, h, np.full((2, 2, 1, 1), 0.4), T=0.5)
    Q = np.linalg.qr(rng.normal(size=shape) + 1j * rng.normal(size=shape))[0]
    p = rng.uniform(0.15, 0.85, size=(2, 2, 3))
    P = (Q * p[..., None, :]) @ Q.conj().swapaxes(-2, -1)
    args = kernel_args(K)
    F = _native.build_fock_apply(args, P, None)[2]
    Ft = Q.conj().swapaxes(-2, -1) @ F @ Q

    def tangent():
        A = rng.normal(size=shape) + 1j * rng.normal(size=shape)
        X = 0.5 * (A - A.conj().swapaxes(-2, -1))
        X[..., np.arange(3), np.arange(3)] = 0.0
        dp = rng.normal(size=p.shape)
        dp -= np.sum(w[..., None] * dp) / (3 * w.sum())
        return X, dp

    def inner(a, b):
        return (np.sum(w[..., None, None] * (a[0].conj() * b[0]).real)
                + np.sum(w[..., None] * a[1] * b[1]))

    for _ in range(5):
        a, b = tangent(), tangent()
        Ha = _native._rtr_joint_hvp(args, *a, Q, p, Ft)
        Hb = _native._rtr_joint_hvp(args, *b, Q, p, Ft)
        np.testing.assert_allclose(inner(a, Hb), inner(Ha, b), rtol=2e-12, atol=2e-13)


def test_newton_solves_coupled_hartree_occupations_without_frozen_fock_overshoot():
    # Purely diagonal example: every degree of freedom is an occupation.
    # HH/T >> 1 makes a frozen-F occupation replacement strongly unstable.
    h = np.diag([-0.2, 0.1, 0.3]).astype(complex)[None, None]
    K = HartreeFockKernel(np.ones((1, 1)), h, np.zeros((1, 1, 1, 1)),
                          T=0.5, include_exchange=False, include_hartree=True,
                          reference_density=np.zeros_like(h), hartree_matrix=10*np.eye(3))
    r = solve(K, np.eye(3)[None, None]/2, 1.5,
              config=SolverConfig(optimizer="newton", tol_grad=1e-8, max_iter=100))
    assert r.converged
    assert r.n_iter < 20
    assert np.all(np.diff(r.history["free_energy"]) <= 1e-12)
    np.testing.assert_allclose(r.p.sum(), 1.5, atol=1e-12)
    assert np.all((r.p >= 0) & (r.p <= 1))


def test_full_deflation_hessian_matches_transported_finite_difference():
    from scipy.linalg import expm
    rng=np.random.default_rng(98)
    w=np.array([[.2,.7]])
    h=np.broadcast_to(np.diag([-.2,.1,.4]),(1,2,3,3)).astype(complex).copy()
    K=HartreeFockKernel(w,h,np.full((1,2,1,1),.1),T=.3)
    Q=np.linalg.qr(rng.normal(size=h.shape)+1j*rng.normal(size=h.shape))[0]
    p=rng.uniform(.2,.8,size=(1,2,3))
    targets=np.broadcast_to(np.diag([.8,.1,.1]),(2,*h.shape)).astype(complex).copy()
    targets[1]=np.eye(3)*.4
    sigma,length=.7,.35
    args=kernel_args(K)
    args.update(test_deflation_targets=targets,test_deflation_sigma=sigma,test_deflation_length=length)
    X=rng.normal(size=h.shape)+1j*rng.normal(size=h.shape)
    X=(X-X.conj().swapaxes(-2,-1))*.1;X[...,np.arange(3),np.arange(3)]=0
    dp=rng.normal(size=p.shape)*.1;dp-=np.sum(w[...,None]*dp)/(w.sum()*3)
    def grad(Q,p):
        P=(Q*p[...,None,:])@Q.conj().swapaxes(-2,-1)
        F=_native.build_fock_apply(args,P,None)[2]
        penalty=_native._deflation_bias(P,targets,w,sigma,length)[1]
        Ft=Q.conj().swapaxes(-2,-1)@(F+penalty)@Q
        G=(p[...,None,:]-p[..., :,None])*Ft
        g=Ft.diagonal(axis1=-2,axis2=-1).real+K.T*np.log(p/(1-p))
        g-=np.sum(w[...,None]*g)/(3*w.sum())
        return G,g,Ft
    G,g,Ft=grad(Q,p)
    HX,Hp=_native._rtr_joint_hvp(args,X,dp,Q,p,Ft)
    def shifted(t):
        U=np.stack([expm(t*M) for M in X.reshape(-1,3,3)]).reshape(X.shape)
        V=np.stack([expm(.5*t*M) for M in X.reshape(-1,3,3)]).reshape(X.shape)
        Gt,gt,_=grad(Q@U,p+t*dp)
        return V@Gt@V.conj().swapaxes(-2,-1),gt
    e=1e-5;a,b=shifted(e),shifted(-e)
    Hfd=(a[0]-b[0])/(2*e);Hfd[...,np.arange(3),np.arange(3)]=0
    np.testing.assert_allclose(HX,Hfd,rtol=1e-7,atol=1e-9)
    np.testing.assert_allclose(Hp,(a[1]-b[1])/(2*e),rtol=1e-7,atol=1e-9)
