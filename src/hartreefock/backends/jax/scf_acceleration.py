"""JAX-native safeguarded full-density-residual SCF acceleration.

The least-squares solve acts on residual differences, without forming a
Gram matrix. All density endpoints have occupations in [0, 1] and the
requested weighted trace. The affine Hartree-Fock map supplies exact Focks
and quadratic electronic energies along backtracking segments; scalar
trials require eigensystems, but no interaction builds.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .fock import hf_energy, occupation_entropy
from .reference_scf import _herm, _weighted_matrix_norm
from .utils import fermidirac, find_chemical_potential


class _State(NamedTuple):
    P: jax.Array
    F: jax.Array
    D: jax.Array
    E: jax.Array
    S: jax.Array
    mu: jax.Array
    dr: jax.Array
    cr: jax.Array


def _residual_coefficients(history, active, latest, weights):
    """Real affine coefficients from a rank-truncated residual-difference SVD."""
    n_hist = history.shape[0]
    dtype = history.real.dtype
    r = history[latest]
    differences = (history - r) * active.reshape((-1, 1, 1, 1, 1))
    scale = jnp.sqrt(weights / jnp.sum(weights))[..., None, None]
    columns = (differences * scale).reshape(n_hist, -1).T
    rhs = (-r * scale).reshape(-1)
    A = jnp.concatenate((columns.real, columns.imag), axis=0)
    b = jnp.concatenate((rhs.real, rhs.imag), axis=0)
    # Direct SVD avoids squaring the condition number, which is particularly
    # important for nearly dependent histories near a fixed point.
    U, singular, Vh = jnp.linalg.svd(A, full_matrices=False)
    threshold = jnp.asarray(1e-10 if dtype == jnp.float64 else 1e-5, dtype)
    keep = singular > threshold * jnp.max(singular)
    inverse = jnp.where(keep, 1.0 / jnp.where(keep, singular, 1.0), 0.0)
    x = Vh.T @ (inverse * (U.T @ b))
    coeffs = (x * active).at[latest].add(1.0 - jnp.sum(x * active))
    fallback = jnp.zeros(n_hist, dtype=dtype).at[latest].set(1.0)
    return jnp.where(jnp.all(jnp.isfinite(coeffs)), coeffs, fallback)


def _capped_occupations(eigenvalues, weights, count):
    """Weighted Frobenius projection onto 0 <= p <= 1, sum(w*p) = count."""
    dtype = eigenvalues.dtype
    weights = weights / jnp.sum(weights)
    lo = -jnp.max(eigenvalues)
    hi = 1.0 - jnp.min(eigenvalues)

    def body(_, interval):
        low, high = interval
        mid = low + 0.5 * (high - low)
        value = jnp.sum(weights[..., None] * jnp.clip(eigenvalues + mid, 0.0, 1.0))
        return jnp.where(value < count, mid, low), jnp.where(value < count, high, mid)

    lo, hi = jax.lax.fori_loop(0, 64 if dtype == jnp.float64 else 32, body, (lo, hi))
    return jnp.clip(eigenvalues + (lo + 0.5 * (hi - lo)), 0.0, 1.0)


@partial(jax.jit, static_argnames=(
    "include_hartree", "include_exchange", "exchange_hermitian_channel_packing",
    "max_iter", "acceleration", "diis_size", "project_fn", "fock_build_fn", "safeguard",
))
def _run_accelerated_scf(
    density0, *, h, weights_b, weight_sum, VR, T, refP, HH, w2d,
    n_electrons, include_hartree, include_exchange,
    exchange_hermitian_channel_packing, contact_g, contact_Oi, contact_Oj,
    mixing, level_shift, density_tol, comm_tol, diis_start, diis_damping,
    trust_radius, max_iter, acceleration, diis_size, project_fn, fock_build_fn,
    safeguard,
):
    dtype = h.real.dtype
    eps = jnp.finfo(dtype).eps
    ws = weight_sum
    normalized_count = n_electrons / ws
    tiny = jnp.finfo(dtype).tiny

    def project(P):
        return _herm(jnp.asarray(project_fn(P) if project_fn is not None else P, h.dtype))

    def entropy_and_repair(P):
        P = project(P)
        p, Q = jnp.linalg.eigh(P)
        count = jnp.sum(w2d[..., None] * p) / ws
        bad = ((jnp.min(p) < -8.0 * eps) | (jnp.max(p) > 1.0 + 8.0 * eps)
               | (jnp.abs(count - normalized_count) > 8.0 * eps * max(1, h.shape[-1])))

        def repair(_):
            occupations = _capped_occupations(p, w2d, normalized_count)
            repaired = project((Q * occupations[..., None, :]) @ jnp.conj(Q.swapaxes(-1, -2)))
            return repaired, jnp.linalg.eigvalsh(repaired)

        P, p = jax.lax.cond(bad, repair, lambda _: (P, p), operand=None)
        return P, occupation_entropy(p, w2d)

    def fermi_density(F, shift=0.0):
        bands, Q = jnp.linalg.eigh(_herm(F))
        mu = find_chemical_potential(bands, w2d, n_electrons, T)
        p = fermidirac(bands - mu, T)

        def shifted(_):
            shifted_bands = bands + shift * (1.0 - p)
            shifted_mu = find_chemical_potential(shifted_bands, w2d, n_electrons, T)
            return fermidirac(shifted_bands - shifted_mu, T), shifted_mu

        p, mu = jax.lax.cond(jnp.asarray(shift) != 0.0, shifted, lambda _: (p, mu), None)
        return project((Q * p[..., None, :]) @ jnp.conj(Q.swapaxes(-1, -2))), mu

    def state(P, F, E, S):
        D, mu = fermi_density(F)
        dr = _weighted_matrix_norm(D - P, w2d, ws)
        cr = _weighted_matrix_norm(F @ P - P @ F, w2d, ws)
        return _State(P, F, D, E, S, mu, dr, cr)

    def evaluate(P):
        P, S = entropy_and_repair(P)
        Sigma, H, F = fock_build_fn(
            P, h=h, VR=VR, refP=refP, HH=HH, w2d=w2d,
            include_exchange=include_exchange, include_hartree=include_hartree,
            exchange_hermitian_channel_packing=exchange_hermitian_channel_packing,
            contact_g=contact_g, contact_Oi=contact_Oi, contact_Oj=contact_Oj,
            project_fn=project_fn,
        )
        E = hf_energy(P, h=h, Sigma=Sigma, H=H, weights_b=weights_b, refP=refP)
        return state(P, project(F), E, S)

    def inner(A, B):
        return jnp.sum(w2d[..., None, None] * jnp.real(jnp.conj(A) * B))

    def acceptable(current, candidate):
        delta = inner(candidate.P - current.P, 0.5 * (candidate.F + current.F))
        delta = delta - T * (candidate.S - current.S)
        noise = 32.0 * eps * jnp.maximum(jnp.abs(current.E) + T * jnp.abs(current.S), ws)
        improved = jnp.maximum(candidate.dr, candidate.cr) < jnp.maximum(current.dr, current.cr)
        finite = (jnp.isfinite(delta) & jnp.isfinite(candidate.E)
                  & jnp.isfinite(candidate.dr) & jnp.isfinite(candidate.cr))
        if not safeguard:
            return finite
        return finite & ((delta < -noise) | ((delta <= noise) & improved))

    def line_search(current, endpoint):
        accepted = acceptable(current, endpoint)
        direction = endpoint.P - current.P
        delta_F = endpoint.F - current.F
        slope = inner(direction, current.F)
        quadratic = 0.5 * inner(direction, delta_F)

        def cond(carry):
            i, _alpha, _trial, accepted = carry
            return (i < 16) & ~accepted

        def body(carry):
            i, alpha, _trial, _accepted = carry
            P = _herm(current.P + alpha * direction)
            F = _herm(current.F + alpha * delta_F)
            S = occupation_entropy(jnp.linalg.eigvalsh(P), w2d)
            E = current.E + alpha * slope + alpha * alpha * quadratic
            trial = state(P, F, E, S)
            return i + 1, 0.5 * alpha, trial, acceptable(current, trial)

        n_bt, _, candidate, accepted = jax.lax.while_loop(
            cond, body, (jnp.int32(0), jnp.asarray(0.5, dtype), endpoint, accepted))
        result = jax.lax.cond(accepted, lambda _: candidate, lambda _: current, None)
        return result, accepted, n_bt + (~accepted).astype(jnp.int32)

    def converged(current):
        return (current.dr <= density_tol) & (current.cr <= comm_tol)

    histories = tuple(jnp.zeros(max_iter, dtype=dtype) for _ in range(4))
    R_hist0 = jnp.zeros((diis_size,) + h.shape, dtype=h.dtype)
    U_hist0 = jnp.zeros_like(R_hist0)
    initial = evaluate(density0)

    def cond(carry):
        k, _current, _filled, _Rh, _Uh, _hist, done, _nf, _nrej, _failed = carry
        return (k < max_iter) & ~done

    def body(carry):
        k, current, filled, R_hist, U_hist, histories, _done, n_fock, n_rej, failed = carry
        values = (current.E, current.E - T * current.S, current.dr, current.cr)
        histories = tuple(history.at[k].set(value) for history, value in zip(histories, values))
        done = converged(current) | (k + 1 >= max_iter)

        def advance(_):
            idx = filled % diis_size
            R = current.D - current.P
            Rh = R_hist.at[idx].set(R)
            update = current.F if acceleration == "rdiis" else current.P + mixing * R
            Uh = U_hist.at[idx].set(update)
            count = jnp.minimum(filled + 1, diis_size)
            coeffs = _residual_coefficients(Rh, jnp.arange(diis_size) < count, idx, w2d)
            fallback = jnp.zeros(diis_size, dtype=dtype).at[idx].set(1.0)
            coeffs = jnp.where(count >= jnp.maximum(2, diis_start), coeffs, fallback)
            proposal = jnp.einsum("i,iklab->klab", coeffs.astype(h.dtype), Uh)
            if acceleration == "rdiis":
                F = diis_damping * proposal + (1.0 - diis_damping) * current.F
                proposal, _ = fermi_density(F, level_shift)
            direction = proposal - current.P
            distance = _weighted_matrix_norm(direction, w2d, ws)
            scale = jnp.where((trust_radius > 0.0) & (distance > trust_radius),
                              trust_radius / jnp.maximum(distance, tiny), 1.0)
            endpoint = evaluate(current.P + scale * direction)
            updated, accepted, rejected = line_search(current, endpoint)

            def recover(_):
                endpoint = evaluate(current.D)
                updated, accepted, rejected = line_search(current, endpoint)
                return updated, accepted, rejected, jnp.int32(1), jnp.int32(0)

            updated, accepted, more_rej, more_fock, next_filled = jax.lax.cond(
                accepted,
                lambda _: (updated, accepted, jnp.int32(0), jnp.int32(0), filled + 1),
                recover, None,
            )
            next_failed = jnp.where(accepted, jnp.int32(0), failed + 1)
            return (updated, next_filled, Rh, Uh, next_failed >= 2,
                    n_fock + 1 + more_fock, n_rej + rejected + more_rej, next_failed)

        next_state = jax.lax.cond(
            done, lambda _: (current, filled, R_hist, U_hist, done, n_fock, n_rej, failed),
            advance, None)
        current, filled, R_hist, U_hist, done, n_fock, n_rej, failed = next_state
        return k + 1, current, filled, R_hist, U_hist, histories, done, n_fock, n_rej, failed

    result = jax.lax.while_loop(cond, body, (
        jnp.int32(0), initial, jnp.int32(0), R_hist0, U_hist0, histories,
        jnp.bool_(False), jnp.int32(1), jnp.int32(0), jnp.int32(0)))
    k, current, _, _, _, histories, _, n_fock, n_rej, _ = result
    # Remove accumulated interpolation roundoff and check precisely the
    # density/Fock pair exposed to the caller, including a capped run.
    current = evaluate(current.P)
    values = (current.E, current.E - T * current.S, current.dr, current.cr)
    histories = tuple(history.at[k - 1].set(value) for history, value in zip(histories, values))
    return (current.P, current.F, current.E, current.mu, k, converged(current),
            *histories, n_fock + 1, n_rej)
