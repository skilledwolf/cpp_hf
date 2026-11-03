"""Reference SCF solver for generating gold-standard baselines.

Roothaan iteration with linear mixing or RDIIS:
  1. Build Fock F[P]
  2. Diagonalize -> eigenvalues, eigenvectors
  3. Find mu via Fermi-Dirac, construct P_new = V diag(f) V†
  4. Update the density per ``config.method``:
       - "linear": P <- (1 - mixing) P + mixing P_new
       - "rdiis":  safeguarded full-density-residual Fock extrapolation
     optionally clipped to a ``trust_radius`` step.
  5. Check convergence on density change and commutator norm.

This solver is intentionally separate from the direct-minimization solver and
serves as a reference / fallback.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from ..config import RDIIS, SolveResult
from .fock import fock_builder, free_energy, hf_energy
from .linalg import adjoint, bmm, eigh, hermitian_eigvalsh
from .utils import hermitize as _herm
from .utils import (
    chemical_potential,
    device_args,
    fermidirac,
    validate_electron_count,
    validate_temperature,
)



def _weighted_matrix_norm(matrix, weights, weight_sum):
    per_k = jnp.sum(jnp.abs(matrix) ** 2, axis=(-2, -1))
    return jnp.sqrt(jnp.sum(weights * per_k) / weight_sum)


def _density_from_fock(fock, *, weights, n_electrons, temperature, level_shift, sectors=None):
    """Diagonalize Fock, find mu, build P = V diag(f) V†."""
    eigenvalues, orbitals = eigh(_herm(fock), sectors)
    mu_raw = chemical_potential(eigenvalues, weights, n_electrons, temperature)
    occ_raw = fermidirac(eigenvalues - mu_raw, temperature)
    level_shift = jnp.asarray(level_shift, dtype=eigenvalues.dtype)

    def shifted_occupations(_):
        shifted = eigenvalues + level_shift * (1.0 - occ_raw)
        mu = chemical_potential(shifted, weights, n_electrons, temperature)
        return fermidirac(shifted - mu, temperature), mu

    # Level shifts are dynamic JIT inputs.  Reuse the occupations at zero
    # shift without executing a second chemical-potential solve.
    occupations, mu = jax.lax.cond(
        level_shift == 0.0, lambda _: (occ_raw, mu_raw), shifted_occupations, None,
    )
    density = bmm(orbitals * occupations[..., None, :].astype(orbitals.dtype), adjoint(orbitals))
    return _herm(density), mu


def roothaan(
    kernel,
    P0,
    n_electrons: float,
    *,
    config=None,
) -> SolveResult:
    """Reference self-consistent field (SCF) solver.

    Standard Roothaan iteration: build Fock, diagonalise, re-occupy via
    Fermi-Dirac, update the density (linear mixing or Pulay DIIS per
    ``config.method``), repeat until converged.  Useful as a
    baseline or fallback when the direct-minimization solver struggles.

    Parameters
    ----------
    kernel
        :class:`HartreeFockKernel` with Hamiltonian, interaction, reference
        density, etc.
    P0
        Initial density matrix, shape ``(nk1, nk2, nb, nb)``.
    n_electrons
        Target electron count.
    config
        :class:`~hartreefock.RDIIS` (the default) or
        :class:`~hartreefock.Linear`.

    Notes
    -----
    ``rdiis`` fits the full fixed-particle Fermi-density residual and
    extrapolates Focks. It repairs occupation bounds and particle count, and
    safeguards the actual finite-temperature free energy.
    Its convergence residual is always the unshifted Fermi map;
    ``level_shift`` only modifies an RDIIS proposal. Custom Fock builders
    must be affine and variational, matching ``hf_energy``. A symmetry
    ``project_fn`` must be a linear, orthogonal, trace- and positivity-
    preserving projection onto a matrix algebra, as for group averaging.
    These are fixed-point solvers and do not certify a local minimum.

    Returns
    -------
    SolveResult
        ``density, fock, energy, mu, n_iter, converged, history``.
    """
    if config is None:
        config = RDIIS()

    args = device_args(kernel)
    h = args["h"]
    P0_arr = jnp.asarray(P0)
    if P0_arr.shape != h.shape:
        raise ValueError(
            f"P0 must have shape {tuple(h.shape)}, got {tuple(P0_arr.shape)}."
        )
    validate_electron_count(
        kernel.w2d,
        kernel.h.shape[-1],
        n_electrons,
        context="n_electrons",
    )
    validate_temperature(kernel.T, context="kernel.T")

    run_scf = _run_scf
    if isinstance(config, RDIIS):
        from .scf_acceleration import _run_accelerated_scf
        if not bool(jnp.all(jnp.isfinite(P0_arr))):
            raise ValueError("P0 must contain only finite values")
        run_scf = _run_accelerated_scf
        extra_args = dict(safeguard=config.safeguard, diis_size=int(config.diis_size),
                          diis_start=int(config.diis_start),
                          diis_damping=float(config.diis_damping))
    else:
        extra_args = {"mixing": float(config.mixing)}

    # At either exact filling endpoint the physical density is known.  Seed
    # it directly so mixing cannot leave a one-ULP particle deficit in a
    # state whose fixed-point residual already passes the stopping tolerance.
    capacity = float(kernel.weight_sum) * h.shape[-1]
    if n_electrons <= 0.0:
        P0_arr = jnp.zeros_like(h)
    elif n_electrons >= capacity:
        P0_arr = jnp.broadcast_to(jnp.eye(h.shape[-1], dtype=h.dtype), h.shape)

    run_result = run_scf(
            jnp.asarray(P0_arr, dtype=h.dtype),
            h=h, weights_b=args["weights_b"], weight_sum=args["weight_sum"],
            VR=args["VR"], T=args["T"], refP=args["refP"], HH=args["HH"],
            w2d=kernel.w2d,
            n_electrons=float(n_electrons),
            include_hartree=args["include_hartree"],
            include_exchange=args["include_exchange"],
            exchange_hermitian_channel_packing=args["exchange_hermitian_channel_packing"],
            contact_g=args["contact_g"], contact_Oi=args["contact_Oi"],
            contact_Oj=args["contact_Oj"],
            level_shift=float(config.level_shift),
            density_tol=float(config.density_tol),
            comm_tol=float(config.comm_tol),
            trust_radius=float(config.trust_radius),
            max_iter=int(config.max_iter),
            project_fn=config.project_fn,
            fock_build_fn=fock_builder(kernel),
            sectors=kernel.sectors,
            **extra_args,
        )

    density, fock, energy, mu, iterations, converged, \
        hist_E, hist_free, hist_density, hist_comm = run_result[:10]
    diagnostics = {}
    if len(run_result) > 10:
        diagnostics = {"fock_builds": int(run_result[10]),
                       "line_search_rejections": int(run_result[11])}

    n_iter = int(iterations)
    return SolveResult(
        density=density,
        fock=fock,
        energy=energy,
        mu=mu,
        n_iter=n_iter,
        converged=bool(converged),
        history={
            "E": hist_E[:n_iter],
            "free_energy": hist_free[:n_iter],
            "density_residual": hist_density[:n_iter],
            "commutator_residual": hist_comm[:n_iter],
            **diagnostics,
        },
    )


@partial(
    jax.jit,
    static_argnames=(
        "include_hartree", "include_exchange",
        "exchange_hermitian_channel_packing",
        "max_iter", "project_fn", "fock_build_fn", "sectors",
    ),
)
def _run_scf(
    density0,
    *,
    h, weights_b, weight_sum, VR, T, refP, HH, w2d,
    n_electrons,
    include_hartree, include_exchange,
    exchange_hermitian_channel_packing,
    contact_g, contact_Oi, contact_Oj,
    mixing, level_shift, density_tol, comm_tol,
    trust_radius, max_iter, project_fn,
    fock_build_fn, sectors,
):
    real_dtype = jnp.zeros((), dtype=h.dtype).real.dtype
    weights_2d = w2d
    ws = jnp.maximum(weight_sum, jnp.asarray(1e-30, dtype=real_dtype))
    tiny = jnp.asarray(1e-30, dtype=real_dtype)

    hist_E = jnp.zeros(max_iter, dtype=real_dtype)
    hist_free = jnp.zeros(max_iter, dtype=real_dtype)
    hist_density = jnp.zeros(max_iter, dtype=real_dtype)
    hist_comm = jnp.zeros(max_iter, dtype=real_dtype)

    _project = project_fn if project_fn is not None else (lambda A: A)

    def _fock(density_h):
        return fock_build_fn(
            density_h, h=h, VR=VR, refP=refP, HH=HH, w2d=w2d,
            include_exchange=include_exchange, include_hartree=include_hartree,
            exchange_hermitian_channel_packing=exchange_hermitian_channel_packing,
            contact_g=contact_g, contact_Oi=contact_Oi, contact_Oj=contact_Oj, sectors=sectors,
            project_fn=project_fn,
        )

    def _build_and_occupy(density):
        density_h = _herm(jnp.asarray(_project(density), dtype=h.dtype))
        Sigma, H, F = _fock(density_h)
        E = hf_energy(
            density_h, h=h, Sigma=Sigma, H=H, weights_b=weights_b, refP=refP,
        )
        density_eigs = hermitian_eigvalsh(density_h).astype(real_dtype)
        Omega = free_energy(E, density_eigs, weights_2d, T)
        P_new, mu = _density_from_fock(
            F, weights=weights_2d, n_electrons=n_electrons,
            temperature=T, level_shift=level_shift, sectors=sectors,
        )
        P_new = _herm(jnp.asarray(_project(P_new), dtype=h.dtype))
        return P_new, F, E, Omega, mu

    def _linear(density, delta):
        return _herm(jnp.asarray(_project(density + mixing * delta), dtype=h.dtype))

    def cond(carry):
        k, _, _, _, _, converged, _, _, _, _ = carry
        return jnp.logical_and(k < max_iter, jnp.logical_not(converged))

    def body(carry):
        k, density, _fk, _en, _mu, _conv, hE, hO, hD, hC = carry

        P_new, F, E, Omega, mu = _build_and_occupy(density)
        delta = P_new - density
        comm = bmm(F, density) - bmm(density, F)
        d_res = _weighted_matrix_norm(delta, weights_2d, ws)
        c_res = _weighted_matrix_norm(comm, weights_2d, ws)
        converged = jnp.logical_and(d_res <= density_tol, c_res <= comm_tol)

        updated = _linear(density, delta)

        # --- trust-region clip on the proposed step (when not converged) -----
        step = updated - density
        step_norm = _weighted_matrix_norm(step, weights_2d, ws)
        do_clip = jnp.logical_and(trust_radius > 0.0, step_norm > trust_radius)
        scale = jnp.where(do_clip, trust_radius / jnp.maximum(step_norm, tiny), 1.0)
        clipped = _herm(jnp.asarray(
            _project(density + scale.astype(h.dtype) * step), dtype=h.dtype))
        updated = jnp.where(do_clip, clipped, updated)

        # The residual above certifies density.  An undamped Fermi-map step
        # can amplify its error and invalidate that convergence certificate.
        density_out = jnp.where(converged, density, updated)

        hE = hE.at[k].set(E)
        hO = hO.at[k].set(Omega)
        hD = hD.at[k].set(d_res)
        hC = hC.at[k].set(c_res)
        return (k + 1, density_out, F, E, mu, converged,
                hE, hO, hD, hC)

    carry0 = (
        jnp.int32(0),
        jnp.asarray(density0, dtype=h.dtype),
        jnp.zeros_like(h),
        jnp.asarray(0.0, dtype=real_dtype),
        jnp.asarray(0.0, dtype=real_dtype),
        jnp.bool_(False),
        hist_E, hist_free, hist_density, hist_comm,
    )

    k, density, fock, energy, mu, converged, hE, hO, hD, hC = \
        jax.lax.while_loop(cond, body, carry0)

    # Final evaluation at the converged density (self-consistent density returned).
    _P_final, F_final, E_final, _Omega_final, mu_final = _build_and_occupy(density)
    return density, F_final, E_final, mu_final, k, converged, hE, hO, hD, hC
