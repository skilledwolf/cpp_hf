"""Finite-temperature density stability with a matrix-free JAX response.

The tested operator is ``I + B**(1/2) L B**(1/2)`` on the fixed-particle
density tangent, where ``B = -df(F)`` and ``L = dF/dP``. Its negative
eigenvalues identify negative free-energy curvature, including directions
between equally occupied orbitals. JAX compiles Fock evaluations and exact
directional derivatives; NumPy/SciPy orchestrate the outer eigenvalue solve.
The public driver itself is not JIT-compatible.
"""

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import expit
from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigsh

from .fock import build_fock, hf_energy
from .utils import device_args


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


_STATIC_FOCK_ARGS = (
    "include_exchange", "include_hartree",
    "exchange_hermitian_channel_packing", "fock_build_fn",
)


@partial(jax.jit, static_argnames=_STATIC_FOCK_ARGS)
def _evaluate_fock(P, *, fock_build_fn, **kwargs):
    sigma, hartree, fock = fock_build_fn(P, project_fn=None, **kwargs)
    energy = hf_energy(P, h=kwargs["h"], Sigma=sigma, H=hartree,
                       weights_b=kwargs["w2d"][..., None, None],
                       refP=kwargs["refP"])
    return fock, energy


@partial(jax.jit, static_argnames=_STATIC_FOCK_ARGS)
def _response_fock(P, direction, *, fock_build_fn, **kwargs):
    return jax.jvp(
        lambda density: _evaluate_fock(
            density, fock_build_fn=fock_build_fn, **kwargs)[0],
        (P,), (direction,),
    )[1]


def _fock_kwargs(kernel, fock_build_fn=None):
    args = device_args(kernel)
    names = ("h", "VR", "refP", "HH", "include_exchange", "include_hartree",
             "exchange_hermitian_channel_packing", "contact_g", "contact_Oi",
             "contact_Oj")
    kwargs = {name: args[name] for name in names}
    kwargs.update(w2d=kernel.w2d,
                  fock_build_fn=build_fock if fock_build_fn is None else fock_build_fn)
    return kwargs


def _host_fermi_density(fock, weights, number, temperature):
    """Host diagonalization and a fixed-particle Fermi solve in double precision."""
    eps, orbitals = np.linalg.eigh(fock)
    capacity = float(weights.sum()) * fock.shape[-1]
    if number == 0 or number == capacity:
        mu = -np.inf if number == 0 else np.inf
    else:
        shift = temperature * (np.log(number) - np.log(capacity - number))
        lo = np.nextafter(float(eps.min()) + shift, -np.inf)
        hi = np.nextafter(float(eps.max()) + shift, np.inf)
        for _ in range(100):
            mid = .5*(lo+hi)
            count = np.sum(weights[..., None]*expit((mid-eps)/temperature))
            if count < number:
                lo = mid
            else:
                hi = mid
        mu = .5*(lo+hi)
    p = expit((mu-eps)/temperature)
    density = (orbitals*p[..., None, :]) @ orbitals.conj().swapaxes(-2, -1)
    return density, eps, orbitals, p


def stability_analysis(kernel, density, n_electrons, *, stationarity_tol=1e-6,
                       eigen_tol=1e-7, maxiter=200, seed=0,
                       fock_build_fn=None) -> StabilityResult:
    """Check an unconstrained finite-T density for negative local curvature.

    ``stable=True`` requires stationarity and a verified eigenvalue solve;
    it is a local test in the supplied basis, not a global minimum guarantee.
    ``stable=None`` is inconclusive. The returned direction is Hermitian and
    conserves particle number in the quadrature metric. Imposed symmetry
    callbacks are intentionally unsupported: the test allows symmetry breaking.

    ``fock_build_fn`` has the same signature as in :func:`solve_scf`. It must
    be JAX-differentiable, affine in density, and variational for the quadratic
    energy formed from its returned ``(Sigma, H, F)``. The response is computed
    by JVP, including contact and custom superlattice interactions. Adjoint
    consistency is checked before accepting an eigenvalue; arbitrary custom
    nonlinear/nonvariational builders are outside the API contract.

    Enable ``jax_enable_x64`` before constructing the kernel for the default
    tolerances. Requests below ten times the kernel dtype epsilon are rejected
    instead of issuing a precision-limited stability certificate.
    """
    if not np.isfinite(kernel.T) or kernel.T <= 0:
        raise ValueError("stability_analysis requires positive temperature")
    if (not np.isfinite(stationarity_tol) or not np.isfinite(eigen_tol)
            or stationarity_tol <= 0 or eigen_tol <= 0
            or isinstance(maxiter, bool)
            or not isinstance(maxiter, (int, np.integer)) or maxiter <= 0):
        raise ValueError("tolerances and maxiter must be positive")
    dtype = np.asarray(kernel.h).dtype
    precision = np.finfo(np.empty((), dtype=dtype).real.dtype).eps
    if min(stationarity_tol, eigen_tol) < 10*precision:
        raise ValueError("stability tolerances require greater kernel precision; "
                         "enable jax_enable_x64 before constructing the kernel")
    P = np.ascontiguousarray(density, dtype=np.result_type(dtype, np.complex64))
    if P.shape != kernel.h.shape or not np.all(np.isfinite(P)):
        raise ValueError("density must be finite and match the kernel shape")
    w = np.asarray(kernel.w2d, dtype=float)
    wn = w/w.sum()
    if not np.all(np.isfinite(w)) or np.any(w <= 0):
        raise ValueError("stability_analysis requires positive quadrature weights")
    if not np.allclose(P, P.conj().swapaxes(-2, -1), rtol=0., atol=10*precision):
        raise ValueError("density must be Hermitian")
    occupations = np.linalg.eigvalsh(P)
    feasibility_tol = max(1e-10, 10*precision)
    if occupations.min() < -feasibility_tol or occupations.max() > 1+feasibility_tol:
        raise ValueError("density eigenvalues must lie in [0,1]")
    count = float(np.sum(w[..., None]*occupations))
    if (not np.isfinite(n_electrons) or n_electrons < 0
            or n_electrons > w.sum()*P.shape[-1]
            or abs(count-n_electrons) > max(1e-8, 10*precision)*w.sum()):
        raise ValueError("density must have the specified particle count")
    kwargs = _fock_kwargs(kernel, fock_build_fn)
    device_density = jnp.asarray(P)
    F = np.asarray(_evaluate_fock(device_density, **kwargs)[0], dtype=complex)
    if not np.all(np.isfinite(F)) or not np.allclose(
            F, F.conj().swapaxes(-2, -1), rtol=0., atol=100*precision):
        raise ValueError("Fock builder must return a finite Hermitian matrix")
    D, eps, Q, p = _host_fermi_density(F, w, n_electrons, kernel.T)
    def norm(A):
        return float(np.sqrt(np.sum(wn*np.sum(abs(A)**2, axis=(-2, -1)))))
    dr, cr = norm(P-D), norm(F@P-P@F)
    stationary = max(dr, cr) <= stationarity_tol
    shape = P.shape
    nk, nb = int(np.prod(shape[:2])), shape[-1]
    eps, p, Q = eps.reshape(nk, nb), p.reshape(nk, nb), Q.reshape(nk, nb, nb)
    wn = wn.reshape(nk)
    rows, cols = np.triu_indices(nb, 1)
    gap, pdiff = eps[:, cols]-eps[:, rows], p[:, rows]-p[:, cols]
    b = np.divide(pdiff, gap, out=np.zeros_like(gap), where=abs(gap)>1e-7*kernel.T)
    same = abs(gap) <= 1e-7*kernel.T
    b[same] = (.5*(p[:, rows]*(1-p[:, rows])+p[:, cols]*(1-p[:, cols]))/kernel.T)[same]
    b = np.maximum(b, 0.)
    indices = np.nonzero(b > 0)
    bsqrt = np.sqrt(b[indices])
    n_off = len(bsqrt)
    diag = p*(1-p)/kernel.T
    active = np.nonzero(diag > 0)
    dsqrt = np.sqrt(diag[active])
    n_diag = len(dsqrt)
    # Eliminate the weighted fixed-trace constraint with a Householder basis.
    q = np.sqrt(wn[active[0]])*dsqrt
    v = q/np.linalg.norm(q) if n_diag else np.empty(0)
    if n_diag:
        v[0] += 1.
        v /= np.linalg.norm(v)
    def house(x):
        return x-2*v*np.dot(v, x)
    n = 2*n_off+max(n_diag-1, 0)
    def unpack(x):
        M = np.zeros((nk, nb, nb), complex)
        k, t = indices
        i, j = rows[t], cols[t]
        z = (x[:n_off]+1j*x[n_off:2*n_off])*bsqrt/np.sqrt(2*wn[k])
        M[k, i, j], M[k, j, i] = z, z.conj()
        if n_diag > 1:
            y = house(np.r_[0., x[2*n_off:]])
            M[active[0], active[1], active[1]] = y*dsqrt/np.sqrt(wn[active[0]])
        return (Q@M@Q.conj().swapaxes(-2, -1)).reshape(shape)
    def pack(A):
        M = Q.conj().swapaxes(-2, -1)@A.reshape(nk, nb, nb)@Q
        k, t = indices
        i, j = rows[t], cols[t]
        z = M[k, i, j]*bsqrt*np.sqrt(2*wn[k])
        d = np.empty(0)
        if n_diag > 1:
            d = house(M[active[0], active[1], active[1]].real
                      * dsqrt*np.sqrt(wn[active[0]]))[1:]
        return np.r_[z.real, z.imag, d]
    builds = 0
    def action(x):
        nonlocal builds
        direction = jnp.asarray(unpack(x), dtype=device_density.dtype)
        response = np.asarray(_response_fock(device_density, direction, **kwargs))
        builds += 1
        answer = x+pack(response)
        if not np.all(np.isfinite(answer)):
            raise ValueError("Fock response must be finite")
        return answer
    if n == 0:
        return StabilityResult(float("inf"), 0., True, stationary,
                               True if stationary else None, dr, cr, 0, None)
    operator = LinearOperator((n, n), matvec=action, dtype=float)
    adjoint_tol = max(1e-8, 100*precision)
    if n <= 64:
        H = np.column_stack([action(x) for x in np.eye(n)])
        if not np.allclose(H, H.T, rtol=adjoint_tol, atol=adjoint_tol*.1):
            raise ValueError("interaction response is not self-adjoint in the quadrature metric")
        eig, V = np.linalg.eigh(.5*(H+H.T))
        value, vector = float(eig[0]), V[:, 0]
        ok = True
    else:
        rng = np.random.default_rng(seed)
        a, b = rng.normal(size=(2, n))
        a, b = a/np.linalg.norm(a), b/np.linalg.norm(b)
        Ha, Hb = action(a), action(b)
        if abs(np.dot(a, Hb)-np.dot(Ha, b)) > adjoint_tol*max(
                1., np.linalg.norm(Ha), np.linalg.norm(Hb)):
            raise ValueError("interaction response is not self-adjoint in the quadrature metric")
        try:
            eig, V = eigsh(operator, k=1, which="SA", tol=eigen_tol, maxiter=maxiter,
                           v0=np.random.default_rng(seed).normal(size=n))
            value, vector, ok = float(eig[0]), V[:, 0], True
        except ArpackNoConvergence:
            return StabilityResult(float("nan"), float("inf"), False, stationary,
                                   None, dr, cr, builds, None)
    residual = float(np.linalg.norm(action(vector)-value*vector))
    ok = ok and residual <= max(10*eigen_tol, 1e-10)*max(1., abs(value))
    stable = bool(value >= -10*eigen_tol) if ok and stationary else None
    return StabilityResult(value, residual, ok, stationary, stable, dr, cr,
                           builds, unpack(vector))
