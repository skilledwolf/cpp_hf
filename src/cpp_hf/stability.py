"""Matrix-free finite-temperature stability of a self-consistent HF density.

The tested operator is I + B^(1/2) L B^(1/2), restricted to fixed particle
count, where B=-df(F) and L=dF/dP. Congruence with the free-energy Hessian
preserves its number of negative directions while avoiding the ill-conditioned
entropy curvature. Equal occupations retain all resolvable density directions.
"""
from dataclasses import dataclass
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh, ArpackNoConvergence
from ._compat import native_required, _native
from .linalg import eigh
from .solver import _kernel_args_for_native
from .utils import fermidirac, find_chemical_potential


@dataclass(frozen=True)
class StabilityResult:
    lowest_eigenvalue: float
    eigen_residual: float
    eigen_converged: bool
    stationary: bool
    stable: bool | None
    density_residual: float
    commutator_residual: float
    response_builds: int
    direction: np.ndarray | None


def stability_analysis(kernel, density, n_electrons, *, stationarity_tol=1e-6,
                       eigen_tol=1e-7, maxiter=200, seed=0) -> StabilityResult:
    """Test an unconstrained finite-T solution for negative free-energy curvature.

    ``stable=True`` means no resolved negative curvature was found after a
    converged eigenvalue solve at a stationary density. It is not a global
    ground-state certificate. ``stable=None`` means the test is inconclusive.
    The returned direction is a Hermitian, particle-conserving density
    variation; its scale is arbitrary. Extra imposed symmetry constraints are
    intentionally not accepted: this tests stability against symmetry breaking.
    """
    native_required()
    if not np.isfinite(kernel.T) or kernel.T <= 0:
        raise ValueError('stability_analysis requires positive temperature')
    if (not np.isfinite(stationarity_tol) or not np.isfinite(eigen_tol)
            or stationarity_tol <= 0 or eigen_tol <= 0
            or not isinstance(maxiter, (int, np.integer)) or maxiter <= 0):
        raise ValueError('tolerances and maxiter must be positive')
    P=np.ascontiguousarray(density,dtype=complex)
    if P.shape != kernel.h.shape or not np.all(np.isfinite(P)):
        raise ValueError('density must be finite and match the kernel shape')
    w=np.asarray(kernel.w2d);wn=w/w.sum()
    if np.any(w<=0):
        raise ValueError('stability_analysis requires positive quadrature weights')
    if not np.allclose(P,P.conj().swapaxes(-2,-1),rtol=0.,atol=1e-10):
        raise ValueError('density must be Hermitian')
    occupations=np.linalg.eigvalsh(P)
    if occupations.min() < -1e-10 or occupations.max() > 1+1e-10:
        raise ValueError('density eigenvalues must lie in [0,1]')
    count=float(np.sum(w[...,None]*occupations))
    if not np.isfinite(n_electrons) or abs(count-n_electrons)>1e-8*w.sum():
        raise ValueError('density must have the specified particle count')
    args=_kernel_args_for_native(kernel)
    _,_,F,_=_native.build_fock_apply(args,P,None)
    eps,Q=eigh(F)
    mu=find_chemical_potential(eps,w,n_electrons,kernel.T)
    p=fermidirac(eps-mu,kernel.T)
    D=(Q*p[...,None,:])@Q.conj().swapaxes(-2,-1)
    norm=lambda A:float(np.sqrt(np.sum(wn*np.sum(abs(A)**2,axis=(-2,-1)))))
    dr,cr=norm(P-D),norm(F@P-P@F)
    stationary=max(dr,cr)<=stationarity_tol
    shape=P.shape;nk=int(np.prod(shape[:2]));nb=shape[-1]
    eps=eps.reshape(nk,nb);p=p.reshape(nk,nb);Q=Q.reshape(nk,nb,nb)
    wn=wn.reshape(nk);rows,cols=np.triu_indices(nb,1)
    gap=eps[:,cols]-eps[:,rows]
    pdiff=p[:,rows]-p[:,cols]
    b=np.divide(pdiff,gap,out=np.zeros_like(gap),where=abs(gap)>1e-7*kernel.T)
    same=abs(gap)<=1e-7*kernel.T
    b[same]=(.5*(p[:,rows]*(1-p[:,rows])+p[:,cols]*(1-p[:,cols]))/kernel.T)[same]
    b=np.maximum(b,0.)
    indices=np.nonzero(b>0)
    bsqrt=np.sqrt(b[indices]);n_off=len(bsqrt)
    diag=(p*(1-p)/kernel.T)
    active=np.nonzero(diag>0)
    dsqrt=np.sqrt(diag[active]);n_diag=len(dsqrt)
    # A Householder basis eliminates the trace direction in O(n), without a
    # dense null-space matrix. The constraint vector includes sqrt(B).
    q=np.sqrt(wn[active[0]])*dsqrt
    v=q/np.linalg.norm(q) if n_diag else np.empty(0)
    if n_diag:
        v[0]+=1.;v/=np.linalg.norm(v)
    def house(x):return x-2*v*np.dot(v,x)
    n=n_off*2+max(n_diag-1,0)
    def unpack(x):
        M=np.zeros((nk,nb,nb),complex)
        k,t=indices;i,j=rows[t],cols[t]
        z=(x[:n_off]+1j*x[n_off:2*n_off])*bsqrt/np.sqrt(2*wn[k])
        M[k,i,j]=z;M[k,j,i]=z.conj()
        if n_diag>1:
            y=house(np.r_[0.,x[2*n_off:]])
            M[active[0],active[1],active[1]]=y*dsqrt/np.sqrt(wn[active[0]])
        return (Q@M@Q.conj().swapaxes(-2,-1)).reshape(shape)
    def pack(A):
        M=Q.conj().swapaxes(-2,-1)@A.reshape(nk,nb,nb)@Q
        k,t=indices;i,j=rows[t],cols[t]
        z=M[k,i,j]*bsqrt*np.sqrt(2*wn[k])
        d=np.empty(0)
        if n_diag>1:
            d=house(M[active[0],active[1],active[1]].real*dsqrt*np.sqrt(wn[active[0]]))[1:]
        return np.r_[z.real,z.imag,d]
    # Removing h before response evaluation prevents subtractive cancellation.
    response_args=dict(args,h=np.zeros_like(P),has_refP=False,refP=np.empty(0,dtype=complex))
    builds=0
    def action(x):
        nonlocal builds
        _,_,response,_=_native.build_fock_apply(response_args,np.ascontiguousarray(unpack(x)),None)
        builds+=1
        return x+pack(response)
    if n==0:
        return StabilityResult(float('inf'),0.,True,stationary,True if stationary else None,dr,cr,0,None)
    operator=LinearOperator((n,n),matvec=action,dtype=float)
    if n<=64:
        H=np.column_stack([action(x) for x in np.eye(n)])
        if not np.allclose(H,H.T,rtol=1e-8,atol=1e-9):
            raise ValueError('interaction response is not self-adjoint in the quadrature metric')
        eig,V=np.linalg.eigh(.5*(H+H.T));value=float(eig[0]);vector=V[:,0]
        ok=True
    else:
        # ARPACK assumes self-adjointness. Catch incompatible interactions
        # before trusting its curvature estimate (without assembling H).
        rng=np.random.default_rng(seed)
        a=rng.normal(size=n);a/=np.linalg.norm(a)
        b=rng.normal(size=n);b/=np.linalg.norm(b)
        Ha,Hb=action(a),action(b)
        if abs(np.dot(a,Hb)-np.dot(Ha,b))>1e-8*max(1.,np.linalg.norm(Ha),np.linalg.norm(Hb)):
            raise ValueError('interaction response is not self-adjoint in the quadrature metric')
        try:
            eig,V=eigsh(operator,k=1,which='SA',tol=eigen_tol,maxiter=maxiter,
                        v0=np.random.default_rng(seed).normal(size=n))
            value=float(eig[0]);vector=V[:,0];ok=True
        except ArpackNoConvergence:
            return StabilityResult(float('nan'),float('inf'),False,stationary,None,dr,cr,builds,None)
    residual=float(np.linalg.norm(action(vector)-value*vector))
    ok=ok and residual<=max(10*eigen_tol,1e-10)*max(1.,abs(value))
    stable=bool(value>=-10*eigen_tol) if ok and stationary else None
    return StabilityResult(value,residual,ok,stationary,stable,dr,cr,builds,unpack(vector))
