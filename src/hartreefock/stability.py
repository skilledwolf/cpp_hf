"""Finite-temperature stability analysis and stability-checked minimization.

The tested operator is ``I + B**(1/2) L B**(1/2)`` on the fixed-particle
density tangent, where ``B = -df(F)`` and ``L = dF/dP``. Its negative
eigenvalues identify negative free-energy curvature, including directions
between equally occupied orbitals. JAX supplies Fock evaluations and the
linear response ``L``; NumPy/SciPy orchestrate the eigenvalue solve and
the escape search, so neither driver is JIT-compatible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigsh
from scipy.special import expit, xlogy

from . import _jax
from .config import SolverConfig, SolveResult


@dataclass(frozen=True)
class StabilityResult:
    """Outcome of :func:`stability_analysis`.

    Attributes
    ----------
    lowest_eigenvalue : float
        Lowest eigenvalue of the free-energy Hessian; negative means unstable.
        ``inf`` when the test was skipped, ``nan`` when the solve failed.
    eigen_residual : float
        Residual norm of the eigenpair.
    eigen_converged : bool
        Whether the eigenvalue solve converged and passed its residual check.
    stationary : bool
        Whether the density meets the stationarity tolerance.
    stable : bool or None
        ``True`` for stationary with no negative curvature, ``False`` for a
        resolved negative mode, ``None`` when the test is inconclusive.
    density_residual, commutator_residual : float
        Stationarity residuals of the input density.
    response_builds : int
        Linear-response Fock builds the test used.
    direction : ndarray or None
        The lowest mode as a Hermitian, particle-conserving density
        perturbation, or ``None`` when there is none.
    """

    lowest_eigenvalue: float
    eigen_residual: float
    eigen_converged: bool
    stationary: bool
    stable: bool | None
    density_residual: float
    commutator_residual: float
    response_builds: int
    direction: np.ndarray | None


@dataclass(frozen=True)
class MinimumResult:
    """Outcome of :func:`solve_minimum`.

    Attributes
    ----------
    solution : SolveResult
        The last local solve, with its density and Fock matrix.
    stability : StabilityResult or None
        The stability test of ``solution``, or ``None`` when none ran.
    converged : bool
        Whether the local solve converged and the stability test found no
        negative curvature.
    escapes : int
        Descent steps taken along unstable modes.
    total_iterations : int
        Iterations summed over all local solves.
    fock_builds : int
        All Fock builds, including the stability tests and trial steps; -1
        when the SCF method does not count them.
    free_energies : tuple of float
        Free energy after each local solve.
    message : str
        Why the search stopped.
    """

    solution: SolveResult
    stability: StabilityResult | None
    converged: bool
    escapes: int
    total_iterations: int
    fock_builds: int
    free_energies: tuple[float, ...]
    message: str


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
            mid = 0.5 * (lo + hi)
            count = np.sum(weights[..., None] * expit((mid - eps) / temperature))
            if count < number:
                lo = mid
            else:
                hi = mid
        mu = 0.5 * (lo + hi)
    p = expit((mu - eps) / temperature)
    density = (orbitals * p[..., None, :]) @ orbitals.conj().swapaxes(-2, -1)
    return density, eps, orbitals, p


def stability_analysis(
    kernel,
    density,
    n_electrons,
    *,
    stationarity_tol=1e-6,
    eigen_tol=1e-7,
    maxiter=200,
    seed=0,
) -> StabilityResult:
    """Test a converged finite-temperature density for negative free-energy curvature.

    Computes the lowest eigenvalue of the free-energy Hessian at ``density``
    over all Hermitian, particle-conserving perturbations, spin and orbital
    mixing included: the test allows symmetry breaking, so symmetry
    callbacks are not supported. The response is the linear part of the
    kernel's Fock build (dense, contact or superlattice), and its adjoint
    consistency is checked before an eigenvalue is accepted. It is a local
    test in the basis supplied, not a proof of a global minimum.

    Tolerances below ten times the kernel dtype's epsilon are rejected.

    Parameters
    ----------
    kernel : HartreeFockKernel
        The problem; needs ``T > 0`` and positive weights.
    density : (nk1, nk2, nb, nb) array
        A Hermitian density with eigenvalues in [0, 1], such as a solver's
        ``result.density``.
    n_electrons : float
        Its particle number, counted with the weights.
    stationarity_tol : float
        Largest residual (density and commutator) for which the density counts
        as stationary.
    eigen_tol : float
        Tolerance of the eigenvalue solve.
    maxiter : int
        Iterations of the eigenvalue solve.
    seed : int
        Seed of the random starting vector.

    Returns
    -------
    StabilityResult
        ``lowest_eigenvalue``, its direction, and ``stable``: ``True`` for a
        stationary density with no negative curvature, ``False`` for negative
        curvature, ``None`` when the test is inconclusive.
    """
    if not np.isfinite(kernel.T) or kernel.T <= 0:
        raise ValueError("stability_analysis requires positive temperature")
    if (
        not np.isfinite(stationarity_tol)
        or not np.isfinite(eigen_tol)
        or stationarity_tol <= 0
        or eigen_tol <= 0
        or isinstance(maxiter, bool)
        or not isinstance(maxiter, (int, np.integer))
        or maxiter <= 0
    ):
        raise ValueError("tolerances and maxiter must be positive")
    dtype = np.asarray(kernel.h).dtype
    precision = np.finfo(np.empty((), dtype=dtype).real.dtype).eps
    if min(stationarity_tol, eigen_tol) < 10 * precision:
        raise ValueError(
            "stability tolerances require greater kernel precision; "
            "build the kernel from float64 arrays"
        )
    P = np.ascontiguousarray(density, dtype=np.result_type(dtype, np.complex64))
    if P.shape != kernel.h.shape or not np.all(np.isfinite(P)):
        raise ValueError("density must be finite and match the kernel shape")
    w = np.asarray(kernel.w2d, dtype=float)
    wn = w / w.sum()
    if not np.all(np.isfinite(w)) or np.any(w <= 0):
        raise ValueError("stability_analysis requires positive quadrature weights")
    if not np.allclose(P, P.conj().swapaxes(-2, -1), rtol=0.0, atol=10 * precision):
        raise ValueError("density must be Hermitian")
    occupations = np.linalg.eigvalsh(P)
    feasibility_tol = max(1e-10, 10 * precision)
    if occupations.min() < -feasibility_tol or occupations.max() > 1 + feasibility_tol:
        raise ValueError("density eigenvalues must lie in [0,1]")
    count = float(np.sum(w[..., None] * occupations))
    if (
        not np.isfinite(n_electrons)
        or n_electrons < 0
        or n_electrons > w.sum() * P.shape[-1]
        or abs(count - n_electrons) > max(1e-8, 10 * precision) * w.sum()
    ):
        raise ValueError("density must have the specified particle count")
    ops = _jax.fock_operators(kernel)
    F = np.asarray(ops.evaluate(P)[0], dtype=complex)
    if not np.all(np.isfinite(F)) or not np.allclose(
        F, F.conj().swapaxes(-2, -1), rtol=0.0, atol=100 * precision
    ):
        raise ValueError("Fock builder must return a finite Hermitian matrix")
    D, eps, Q, p = _host_fermi_density(F, w, n_electrons, kernel.T)

    def norm(A):
        return float(np.sqrt(np.sum(wn * np.sum(abs(A) ** 2, axis=(-2, -1)))))

    dr, cr = norm(P - D), norm(F @ P - P @ F)
    stationary = max(dr, cr) <= stationarity_tol
    shape = P.shape
    nk, nb = int(np.prod(shape[:2])), shape[-1]
    eps, p, Q = eps.reshape(nk, nb), p.reshape(nk, nb), Q.reshape(nk, nb, nb)
    wn = wn.reshape(nk)
    rows, cols = np.triu_indices(nb, 1)
    gap, pdiff = eps[:, cols] - eps[:, rows], p[:, rows] - p[:, cols]
    b = np.divide(pdiff, gap, out=np.zeros_like(gap), where=abs(gap) > 1e-7 * kernel.T)
    same = abs(gap) <= 1e-7 * kernel.T
    b[same] = (
        0.5 * (p[:, rows] * (1 - p[:, rows]) + p[:, cols] * (1 - p[:, cols])) / kernel.T
    )[same]
    b = np.maximum(b, 0.0)
    indices = np.nonzero(b > 0)
    bsqrt = np.sqrt(b[indices])
    n_off = len(bsqrt)
    diag = p * (1 - p) / kernel.T
    active = np.nonzero(diag > 0)
    dsqrt = np.sqrt(diag[active])
    n_diag = len(dsqrt)
    # Eliminate the weighted fixed-trace constraint with a Householder basis.
    q = np.sqrt(wn[active[0]]) * dsqrt
    v = q / np.linalg.norm(q) if n_diag else np.empty(0)
    if n_diag:
        v[0] += 1.0
        v /= np.linalg.norm(v)

    def house(x):
        return x - 2 * v * np.dot(v, x)

    n = 2 * n_off + max(n_diag - 1, 0)

    def unpack(x):
        M = np.zeros((nk, nb, nb), complex)
        k, t = indices
        i, j = rows[t], cols[t]
        z = (x[:n_off] + 1j * x[n_off : 2 * n_off]) * bsqrt / np.sqrt(2 * wn[k])
        M[k, i, j], M[k, j, i] = z, z.conj()
        if n_diag > 1:
            y = house(np.r_[0.0, x[2 * n_off :]])
            M[active[0], active[1], active[1]] = y * dsqrt / np.sqrt(wn[active[0]])
        return (Q @ M @ Q.conj().swapaxes(-2, -1)).reshape(shape)

    def pack(A):
        M = Q.conj().swapaxes(-2, -1) @ A.reshape(nk, nb, nb) @ Q
        k, t = indices
        i, j = rows[t], cols[t]
        z = M[k, i, j] * bsqrt * np.sqrt(2 * wn[k])
        d = np.empty(0)
        if n_diag > 1:
            d = house(
                M[active[0], active[1], active[1]].real * dsqrt * np.sqrt(wn[active[0]])
            )[1:]
        return np.r_[z.real, z.imag, d]

    builds = 0

    def action(x):
        nonlocal builds
        response = np.asarray(ops.response(P, unpack(x).astype(P.dtype)))
        builds += 1
        answer = x + pack(response)
        if not np.all(np.isfinite(answer)):
            raise ValueError("Fock response must be finite")
        return answer

    if n == 0:
        return StabilityResult(
            float("inf"),
            0.0,
            True,
            stationary,
            True if stationary else None,
            dr,
            cr,
            0,
            None,
        )
    operator = LinearOperator((n, n), matvec=action, dtype=float)
    adjoint_tol = max(1e-8, 100 * precision)
    if n <= 64:
        H = np.column_stack([action(x) for x in np.eye(n)])
        if not np.allclose(H, H.T, rtol=adjoint_tol, atol=adjoint_tol * 0.1):
            raise ValueError(
                "interaction response is not self-adjoint in the quadrature metric"
            )
        eig, V = np.linalg.eigh(0.5 * (H + H.T))
        value, vector = float(eig[0]), V[:, 0]
        ok = True
    else:
        rng = np.random.default_rng(seed)
        a, b = rng.normal(size=(2, n))
        a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
        Ha, Hb = action(a), action(b)
        if abs(np.dot(a, Hb) - np.dot(Ha, b)) > adjoint_tol * max(
            1.0, np.linalg.norm(Ha), np.linalg.norm(Hb)
        ):
            raise ValueError(
                "interaction response is not self-adjoint in the quadrature metric"
            )
        try:
            eig, V = eigsh(
                operator,
                k=1,
                which="SA",
                tol=eigen_tol,
                maxiter=maxiter,
                v0=np.random.default_rng(seed).normal(size=n),
            )
            value, vector, ok = float(eig[0]), V[:, 0], True
        except ArpackNoConvergence:
            return StabilityResult(
                float("nan"),
                float("inf"),
                False,
                stationary,
                None,
                dr,
                cr,
                builds,
                None,
            )
    residual = float(np.linalg.norm(action(vector) - value * vector))
    ok = ok and residual <= max(10 * eigen_tol, 1e-10) * max(1.0, abs(value))
    stable = bool(value >= -10 * eigen_tol) if ok and stationary else None
    return StabilityResult(
        value, residual, ok, stationary, stable, dr, cr, builds, unpack(vector)
    )


def _feasible_density(P, weights, number):
    """Weighted Frobenius projection onto the fixed-trace density spectrahedron."""
    p, Q = np.linalg.eigh(0.5 * (P + P.conj().swapaxes(-2, -1)))
    lo, hi = -float(p.max()) - 1.0, 1.0 - float(p.min()) + 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if np.sum(weights[..., None] * np.clip(p + mid, 0.0, 1.0)) < number:
            lo = mid
        else:
            hi = mid
    p = np.clip(p + 0.5 * (lo + hi), 0.0, 1.0)
    return np.ascontiguousarray((Q * p[..., None, :]) @ Q.conj().swapaxes(-2, -1))


def _fermi_escape_density(fock, response_field, weights, number, temperature, step):
    """Retract a response-field step through fixed-particle Fermi occupation.

    For an unstable congruence eigenmode dP with eigenvalue lambda, use
    response_field = L(dP)/(1-lambda). Since -B L(dP)=(1-lambda)dP,
    the derivative at a stationary density is dP. Re-occupation respects
    spectral bounds smoothly even when density projection would clip a
    tangent perturbation at saturated occupations.
    """
    field = np.asarray(fock) + step * np.asarray(response_field)
    field = 0.5 * (field + field.conj().swapaxes(-2, -1))
    return _host_fermi_density(field, weights, number, temperature)[0]


def solve_minimum(
    kernel,
    P0,
    n_electrons,
    *,
    config=None,
    max_escapes=4,
    eigen_tol=1e-7,
    stability_maxiter=200,
) -> MinimumResult:
    """Solve, test the stability of the solution, and step downhill until it is a local minimum.

    Each round runs a local SCF solve and :func:`stability_analysis`. On
    negative curvature it refills a Fock field perturbed along the unstable
    direction at fixed particle number (falling back to a spectral projection
    of the density perturbation), accepts the step only if it lowers the free
    energy by more than the kernel precision, and solves again. Symmetry
    callbacks are not supported, since the escape must be allowed to break
    symmetry. ``converged`` is a local statement within the basis supplied, not
    a proof of the ground state or a test of sectors left out of it.

    Parameters
    ----------
    kernel : HartreeFockKernel
        The problem; needs ``T > 0`` and positive weights.
    P0 : (nk1, nk2, nb, nb) array
        Starting density.
    n_electrons : float
        Particle number, counted with the weights.
    config : SolverConfig, optional
        Configuration of each local solve. The default is safeguarded RDIIS
        with 1e-8 residuals and 500 iterations.
    max_escapes : int
        Most descent steps; at most ``max_escapes + 1`` local solves run.
    eigen_tol : float
        Tolerance of each stability test.
    stability_maxiter : int
        Iterations of each stability test.

    Returns
    -------
    MinimumResult
        The last local solve (``solution``), its stability test, the number of
        escapes, the free energy after each solve, and the Fock builds spent
        (``-1`` when the SCF method does not count them).
    """
    if (
        not isinstance(max_escapes, (int, np.integer))
        or isinstance(max_escapes, bool)
        or max_escapes < 0
    ):
        raise ValueError("max_escapes must be a nonnegative integer")
    if kernel.T <= 0 or np.any(np.asarray(kernel.w2d) <= 0):
        raise ValueError("solve_minimum requires positive temperature and weights")
    if (
        not np.isfinite(eigen_tol)
        or eigen_tol <= 0
        or not isinstance(stability_maxiter, (int, np.integer))
        or isinstance(stability_maxiter, bool)
        or stability_maxiter <= 0
    ):
        raise ValueError("eigen_tol and stability_maxiter must be positive")
    precision = np.finfo(np.asarray(kernel.h).real.dtype).eps
    if eigen_tol < 10 * precision:
        raise ValueError(
            "stability tolerance requires greater kernel precision; "
            "build the kernel from float64 arrays"
        )
    if config is None:
        config = SolverConfig(
            diis_size=8,
            max_iter=500,
            density_tol=1e-8,
            comm_tol=1e-8,
        )
    if config.project_fn is not None:
        raise ValueError(
            "solve_minimum requires an unconstrained kernel; remove project_fn"
        )
    cfg = config
    ops = _jax.fock_operators(kernel)
    w = np.asarray(kernel.w2d)
    iterations = builds = escapes = 0
    energies = []
    initial = P0
    stability = None
    message = "Local solve did not converge"
    for cycle in range(max_escapes + 1):
        result = _jax.solve(kernel, initial, n_electrons, config=cfg)
        iterations += result.n_iter
        # The new RDIIS/Anderson paths count every build. For legacy methods
        # without counters, -1 propagates "unavailable" rather than fabricating
        # a total from iteration counts.
        local_builds = result.history.get("fock_builds", -1)
        if builds >= 0:
            builds = builds + int(local_builds) if local_builds >= 0 else -1
        # Legacy SCF histories can stop before the final density/Fock rebuild.
        # Compare every escape against the actual returned state's free energy,
        # using the same spectral entropy as the accepted finite-step trials.
        p = np.clip(np.linalg.eigvalsh(np.asarray(result.density)), 0.0, 1.0)
        omega = float(result.energy) + kernel.T * np.sum(
            w[..., None] * (xlogy(p, p) + xlogy(1 - p, 1 - p))
        )
        omega = float(omega)
        energies.append(omega)
        # A saddle can stall just above a tight requested residual. A nearby
        # Fermi-response Hessian can still propose an escape direction; actual
        # energy descent validates it. Only strict local convergence plus a
        # stable eigenvalue test can produce a successful final result.
        recovery_tol = max(1e-6, cfg.density_tol, cfg.comm_tol)
        residual = max(
            float(result.history["density_residual"][-1]),
            float(result.history["commutator_residual"][-1]),
        )
        if not result.converged and residual > recovery_tol:
            stability = None
            break
        stability = stability_analysis(
            kernel,
            result.density,
            n_electrons,
            stationarity_tol=recovery_tol,
            eigen_tol=eigen_tol,
            maxiter=stability_maxiter,
        )
        if builds >= 0:
            builds += 1 + stability.response_builds
        if stability.stable is True and not result.converged:
            message = "No negative curvature found, but the requested residual was not reached"
            break
        if stability.stable is True:
            return MinimumResult(
                result,
                stability,
                True,
                escapes,
                iterations,
                builds,
                tuple(energies),
                "Converged to a locally stable density",
            )
        if stability.stable is None:
            message = "Stability eigenvalue solve was inconclusive"
            break
        if cycle == max_escapes:
            message = "Negative curvature remains after the escape budget"
            break
        direction = stability.direction
        norm = np.sqrt(np.sum(w * np.sum(abs(direction) ** 2, axis=(-2, -1))) / w.sum())
        direction = direction / max(norm, np.finfo(float).tiny)
        best, best_energy = None, omega
        # At stationarity the Fermi map has derivative -B. The congruence
        # eigen-equation gives -B L(dP)=(1-lambda)dP, including the chemical
        # potential response that enforces the weighted particle constraint.
        # The actual finite-step energy check also covers slightly stalled,
        # nonstationary starting densities where this tangent equality is
        # approximate. Count the exact JVP response as one additional build.
        density = np.asarray(result.density).astype(ops.dtype)
        response_field = np.asarray(
            ops.response(density, direction.astype(ops.dtype))
        ) / (1 - stability.lowest_eigenvalue)
        if builds >= 0:
            builds += 1
        noise = 64 * precision * max(1.0, abs(omega))
        for use_fermi in (True, False):
            for amplitude in (0.25, 0.1, 0.03, 0.01, 0.003, 0.001):
                for sign in (-1.0, 1.0):
                    step = sign * amplitude
                    if use_fermi:
                        trial = _fermi_escape_density(
                            result.fock, response_field, w, n_electrons, kernel.T, step
                        )
                    else:
                        trial = _feasible_density(
                            result.density + step * direction, w, n_electrons
                        )
                    # Judge and retain exactly the density representable by this
                    # kernel (a float32 JAX kernel must not silently see float64).
                    trial = np.asarray(trial).astype(ops.dtype)
                    E = float(ops.evaluate(trial)[1])
                    if builds >= 0:
                        builds += 1
                    p = np.clip(np.linalg.eigvalsh(trial), 0.0, 1.0)
                    trial_energy = E + kernel.T * np.sum(
                        w[..., None] * (xlogy(p, p) + xlogy(1 - p, 1 - p))
                    )
                    if trial_energy < best_energy - noise:
                        best, best_energy = trial, float(trial_energy)
            if best is not None:
                break
        if best is None:
            message = (
                "Negative curvature found, but no finite feasible descent step resolved"
            )
            break
        initial = best
        escapes += 1
    return MinimumResult(
        result, stability, False, escapes, iterations, builds, tuple(energies), message
    )
