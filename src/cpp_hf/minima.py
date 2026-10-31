"""Stability-checked local minimization with bounded negative-curvature recovery."""
from dataclasses import dataclass, replace
import numpy as np
from scipy.special import xlogy
from ._compat import _native
from .reference_scf import SCFConfig, SCFResult, solve_scf
from .solver import _kernel_args_for_native
from .stability import StabilityResult, stability_analysis
from .utils import density_matrix_from_fock


@dataclass(frozen=True)
class MinimumResult:
    solution: SCFResult
    stability: StabilityResult | None
    converged: bool
    escapes: int
    total_iterations: int
    fock_builds: int
    free_energies: tuple[float, ...]
    message: str


def _feasible_density(P, weights, number):
    """Weighted Frobenius projection onto the fixed-trace density spectrahedron."""
    p, Q = np.linalg.eigh(.5*(P+P.conj().swapaxes(-2,-1)))
    lo, hi = -float(p.max())-1., 1.-float(p.min())+1.
    for _ in range(60):
        mid = .5*(lo+hi)
        if np.sum(weights[...,None]*np.clip(p+mid,0.,1.)) < number:
            lo = mid
        else:
            hi = mid
    p = np.clip(p+.5*(lo+hi),0.,1.)
    return np.ascontiguousarray((Q*p[...,None,:])@Q.conj().swapaxes(-2,-1))


def solve_minimum(kernel, P0, n_electrons, *, config=None, max_escapes=4,
                  eigen_tol=1e-7, stability_maxiter=200) -> MinimumResult:
    """Find a stationary density with no resolved negative curvature.

    Each local solve has ``config.max_iter`` iterations; at most
    ``max_escapes + 1`` local solves are performed. All additional Fock builds,
    including the stability test and trial perturbations, are counted. The
    default is safeguarded RDIIS at 1e-8 residuals with 500 iterations.
    ``converged`` requires both a converged local solve and a conclusive stable
    Hessian test. This is a local minimum check within the supplied basis, not
    a global ground-state certificate or a test of omitted spin/valley sectors.

    Escapes first perturb the physical Fock through its affine response and
    rebuild a fixed-particle Fermi density. This retains the unstable tangent
    without clipping nearly saturated occupations. Projected density trials
    remain a fallback. Every accepted escape must lower the actual free energy.

    Extra symmetry callbacks are not supported: recovery must be allowed to
    break symmetry. ``block_sizes`` remains a checked acceleration hint.
    The returned SCF solution always contains its density and Fock matrix.
    """
    if not isinstance(max_escapes,(int,np.integer)) or isinstance(max_escapes,bool) or max_escapes < 0:
        raise ValueError('max_escapes must be a nonnegative integer')
    if kernel.T <= 0 or np.any(np.asarray(kernel.w2d) <= 0):
        raise ValueError('solve_minimum requires positive temperature and weights')
    if config is None:
        config = SCFConfig(acceleration='rdiis',diis_size=8,max_iter=500,
                           density_tol=1e-8,comm_tol=1e-8)
    if config.project_fn is not None:
        raise ValueError('solve_minimum requires an unconstrained kernel; remove project_fn')
    cfg = replace(config,return_density=True,return_fock=True)
    args = _kernel_args_for_native(kernel)
    w = np.asarray(kernel.w2d)
    iterations = builds = escapes = 0
    energies = []
    initial = P0
    stability = None
    message = 'Local solve did not converge'
    for cycle in range(max_escapes+1):
        result = solve_scf(kernel,initial,n_electrons,config=cfg)
        iterations += result.iterations
        builds += result.history['fock_builds']
        p = np.clip(np.linalg.eigvalsh(result.density_matrix),0.,1.)
        omega = float(result.energy+kernel.T*np.sum(
            w[...,None]*(xlogy(p,p)+xlogy(1-p,1-p))))
        energies.append(omega)
        # A saddle can stall just above a tight requested residual. A nearby
        # Fermi-response Hessian can still propose an escape direction; actual
        # energy descent validates it. Only strict local convergence plus a
        # stable eigenvalue test can produce a successful final result.
        recovery_tol = max(1e-6,cfg.density_tol,cfg.comm_tol)
        residual = max(float(result.history['density_residual'][-1]),
                       float(result.history['commutator_residual'][-1]))
        if not result.converged and residual > recovery_tol:
            stability = None
            break
        stability = stability_analysis(kernel,result.density_matrix,n_electrons,
            stationarity_tol=recovery_tol,eigen_tol=eigen_tol,
            maxiter=stability_maxiter)
        builds += 1 + stability.response_builds
        if stability.stable is True and not result.converged:
            message = "No negative curvature found, but the requested residual was not reached"
            break
        if stability.stable is True:
            return MinimumResult(result,stability,True,escapes,iterations,builds,
                                 tuple(energies),'Converged to a locally stable density')
        if stability.stable is None:
            message = 'Stability eigenvalue solve was inconclusive'
            break
        if cycle == max_escapes:
            message = 'Negative curvature remains after the escape budget'
            break
        direction = stability.direction
        norm = np.sqrt(np.sum(w*np.sum(abs(direction)**2,axis=(-2,-1)))/w.sum())
        direction = direction/max(norm,np.finfo(float).tiny)
        best, best_energy = None, omega
        # If (I + sqrt(B) L sqrt(B))v = lambda*v, dP=sqrt(B)v obeys
        # -B L dP = (1-lambda)dP on the fixed-particle tangent. The Fermi
        # retraction below therefore follows that direction to first order,
        # including near-saturated bands where Euclidean clipping can erase
        # the negative curvature. Check finite steps against the actual energy.
        noise = 64*np.finfo(float).eps*max(1.,abs(omega))
        response_args = dict(args,h=np.zeros_like(direction),has_refP=False,
                             refP=np.empty(0,dtype=complex))
        response = _native.build_fock_apply(response_args,direction,None)[2]
        builds += 1
        response = response/(1.-stability.lowest_eigenvalue)
        for retraction in ('fermi','projected'):
            for amplitude in (.25,.1,.03,.01,.003,.001):
                for sign in (-1.,1.):
                    if retraction == 'fermi':
                        trial,_ = density_matrix_from_fock(
                            result.fock_matrix+sign*amplitude*response,w,n_electrons,kernel.T)
                    else:
                        trial = _feasible_density(result.density_matrix+sign*amplitude*direction,w,n_electrons)
                    E = float(_native.build_fock_apply(args,trial,None)[3])
                    builds += 1
                    p = np.clip(np.linalg.eigvalsh(trial),0.,1.)
                    trial_energy = E+kernel.T*np.sum(w[...,None]*(xlogy(p,p)+xlogy(1-p,1-p)))
                    if trial_energy < best_energy-noise:
                        best, best_energy = trial, float(trial_energy)
            if best is not None:
                break
        if best is None:
            message = 'Negative curvature found, but no finite feasible descent step resolved'
            break
        initial = best
        escapes += 1
    return MinimumResult(result,stability,False,escapes,iterations,builds,tuple(energies),message)
