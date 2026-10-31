"""Generic symmetry projectors for JAX Hartree-Fock solvers.

This module provides ``make_project_fn``, a JAX-traceable callable builder
for averaging density / Fock matrices over unitary, spatial, and
time-reversal symmetries.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jax import lax


ProjectFn = Callable[[jax.Array], jax.Array]

__all__ = ["ProjectFn", "make_project_fn"]


def _flip_k(
    A: jax.Array,
    k_convention: str,
    flip_axes: tuple[int, ...] = (0, 1),
) -> jax.Array:
    """Negate selected k-grid axes under either ``flip`` or ``mod`` convention."""
    if k_convention == "flip":
        return jnp.flip(A, axis=flip_axes)
    if k_convention == "mod":
        nk1, nk2 = A.shape[0], A.shape[1]
        i = (
            (-jnp.arange(nk1, dtype=jnp.int32)) % nk1
            if 0 in flip_axes
            else jnp.arange(nk1, dtype=jnp.int32)
        )
        j = (
            (-jnp.arange(nk2, dtype=jnp.int32)) % nk2
            if 1 in flip_axes
            else jnp.arange(nk2, dtype=jnp.int32)
        )
        return A[i[:, None], j[None, :], ...]
    raise ValueError(f"k_convention must be 'mod' or 'flip', got {k_convention!r}")


# Orbital dimension up to which the group sum is applied as one precomputed
# GEMM on vec(A) (an (nb², nb²) matrix: 16 bytes × nb⁴, i.e. 5 MB at nb=24).
# Measured on the nk=49, nb=16 bilayer: the per-element conjugation loop costs
# 30-40 ms per call (16 batched 16×16 matmuls that XLA:CPU runs far below
# peak) against 5.7 ms for the single (2401, 256) @ (256, 256) product, and
# the projector is applied about six times per CG iteration.  Above the
# threshold the nb⁴ matrix outgrows the 2·|G|·nb³ loop and its memory.
_GEMM_MAX_NB = 24


def _conj_sum_matrix(G: jax.Array) -> jax.Array:
    """Matrix of ``vec(A) -> vec(sum_i g_i A g_i†)`` for row-major ``vec``.

    ``vec(g A g†) = (g ⊗ conj(g)) vec(A)``; the sum over the group elements is
    formed once, at ``make_project_fn`` time, from concrete unitaries.
    """
    G = jnp.asarray(G)
    ng, nb = G.shape[0], G.shape[-1]
    M = jnp.zeros((nb * nb, nb * nb), dtype=G.dtype)
    for i in range(ng):
        M = M + jnp.kron(G[i], jnp.conj(G[i]))
    return M


def _apply_vec_matrix(A: jax.Array, M: jax.Array) -> jax.Array:
    """Apply a vec-space matrix ``M`` to the last two axes of ``A``."""
    nb = A.shape[-1]
    flat = A.reshape(A.shape[:-2] + (nb * nb,))
    return (flat @ jnp.swapaxes(M.astype(A.dtype), -1, -2)).reshape(A.shape)


def _sum_unitary_conj(
    A: jax.Array, G: jax.Array, M_sum: jax.Array | None = None,
) -> jax.Array:
    """Return ``sum_i g_i @ A @ g_i†`` without normalization.

    ``M_sum`` is the precomputed :func:`_conj_sum_matrix` of ``G``; when given
    the sum is one GEMM instead of ``len(G)`` batched conjugations.
    """
    if M_sum is not None:
        return _apply_vec_matrix(A, M_sum)
    # The group elements follow the working dtype of ``A``: a complex128 group
    # applied to a complex64 density must not promote the loop carry.
    G = G.astype(A.dtype)
    ng = G.shape[0]

    def body(i, acc):
        g = G[i]
        gH = jnp.conj(jnp.swapaxes(g, -1, -2))
        return acc + (g @ A) @ gH

    return lax.fori_loop(0, ng, body, jnp.zeros_like(A))


def _avg_unitary_conj(
    A: jax.Array, G: jax.Array, M_sum: jax.Array | None = None,
) -> jax.Array:
    """Average ``A`` over unitary conjugation by the elements of ``G``."""
    ng = G.shape[0]
    return _sum_unitary_conj(A, G, M_sum) / jnp.asarray(float(ng), dtype=A.dtype)


def _avg_combined_group(
    A: jax.Array,
    G_same: jax.Array,
    G_flip: jax.Array,
    k_convention: str,
    flip_axes: tuple[int, ...] = (0, 1),
    M_same: jax.Array | None = None,
    M_flip: jax.Array | None = None,
) -> jax.Array:
    """Average over a group split into same-k and flipped-k elements."""
    acc = _sum_unitary_conj(A, G_same, M_same)
    A_neg = _flip_k(A, k_convention, flip_axes)
    acc = acc + _sum_unitary_conj(A_neg, G_flip, M_flip)
    N = G_same.shape[0] + G_flip.shape[0]
    return acc / jnp.asarray(float(N), dtype=A.dtype)


def _avg_spatial_group_via_k_maps(
    A: jax.Array,
    spatial_group: jax.Array,
    k_index_i: jax.Array,
    k_index_j: jax.Array,
) -> jax.Array:
    """Average over a spatial group with per-element k-index maps.

    For each element ``g`` in ``spatial_group`` (orbital unitary ``U_g``),
    take ``A[k_index_i[g], k_index_j[g]]``, conjugate by ``U_g``, and average.
    The k-index maps must be precomputed against the actual k-mesh — see
    ``contimod.symmetry.continuum.cyclic_rotation_index_maps`` for one builder.

    Shapes:
        A             : (nk1, nk2, nb, nb)
        spatial_group : (n_g, nb, nb)
        k_index_i,j   : (n_g, nk1, nk2)
    """
    # Coerce up front so callers passing numpy arrays (e.g. the cpp_hf
    # backend, which uses np internally) don't trip the fori_loop tracer.
    A_j = jnp.asarray(A)
    spatial_group = jnp.asarray(spatial_group).astype(A_j.dtype)
    k_index_i = jnp.asarray(k_index_i)
    k_index_j = jnp.asarray(k_index_j)
    n_g = spatial_group.shape[0]

    def body(i, acc):
        gi = spatial_group[i]
        giH = jnp.conj(jnp.swapaxes(gi, -1, -2))
        A_g = A_j[k_index_i[i], k_index_j[i]]  # (nk1, nk2, nb, nb)
        return acc + (gi @ A_g) @ giH

    return lax.fori_loop(0, n_g, body, jnp.zeros_like(A_j)) / jnp.asarray(
        float(n_g), dtype=A_j.dtype
    )


def _avg_time_reversal_via_k_map(
    A: jax.Array,
    U: jax.Array,
    k_index_i: jax.Array,
    k_index_j: jax.Array,
    valid: jax.Array,
) -> jax.Array:
    """Time-reversal average using a precomputed k-index map.

    ``valid`` is the (nk1, nk2) bool mask that selects k whose ``-k`` partner
    lies inside the patch.  Where invalid, ``A[k]`` is left untouched.
    """
    A_j = jnp.asarray(A)
    U = jnp.asarray(U).astype(A_j.dtype)
    k_index_i = jnp.asarray(k_index_i)
    k_index_j = jnp.asarray(k_index_j)
    valid = jnp.asarray(valid)
    UH = jnp.conj(jnp.swapaxes(U, -1, -2))
    A_at_neg = A_j[k_index_i, k_index_j]
    A_tr = U @ jnp.conj(A_at_neg) @ UH
    avg = 0.5 * (A_j + A_tr)
    return jnp.where(valid[..., None, None], avg, A_j)


def _avg_time_reversal(
    A: jax.Array,
    U: jax.Array,
    k_convention: str,
    flip_axes: tuple[int, ...] = (0, 1),
) -> jax.Array:
    """Average ``A`` with its time-reversed partner."""
    U = U.astype(A.dtype)
    UH = jnp.conj(jnp.swapaxes(U, -1, -2))
    A_neg = _flip_k(A, k_convention, flip_axes)
    A_tr = U @ jnp.conj(A_neg) @ UH
    return 0.5 * (A + A_tr)


def make_project_fn(
    *,
    unitary_group: jax.Array | None = None,
    spatial_group: jax.Array | None = None,
    spatial_k_convention: str = "mod",
    spatial_k_flip_axes: tuple[int, ...] = (0, 1),
    spatial_k_index_maps: tuple[jax.Array, jax.Array] | None = None,
    spatial_k_orbit_mask: jax.Array | None = None,
    time_reversal_U: jax.Array | None = None,
    time_reversal_k_convention: str = "mod",
    time_reversal_k_flip_axes: tuple[int, ...] = (0, 1),
    time_reversal_k_index_map: tuple[jax.Array, jax.Array, jax.Array] | None = None,
) -> ProjectFn:
    """Build a symmetry-averaging projection function.

    Two ways to specify how the k-coordinate transforms under a symmetry:

    1. **Convention strings** (``spatial_k_convention``,
       ``time_reversal_k_convention``): ``"mod"`` uses periodic-cell wraps
       and ``"flip"`` reflects selected axes — useful when the k-mesh is a
       periodic Brillouin zone that the symmetry maps to itself by index
       arithmetic.

    2. **Precomputed index maps** (``spatial_k_index_maps``,
       ``time_reversal_k_index_map``): pass per-element / per-k tables that
       say where each k goes under the symmetry.  Required for arbitrary
       rotations on a finite continuum patch where the symmetry is not a
       trivial axis flip — e.g. C3z on a rhombic graphene-K patch.  See
       :func:`contimod.symmetry.continuum.cyclic_rotation_index_maps` for a
       builder.

    Parameters
    ----------
    unitary_group, spatial_group, time_reversal_U :
        Orbital unitaries (no k-dependence).  Same as before.
    spatial_k_index_maps :
        Optional ``(idx_i, idx_j)`` tuple where ``idx_i`` and ``idx_j`` each
        have shape ``(n_g_spatial, nk1, nk2)``.  When given, the spatial
        average uses ``A[idx_i[g], idx_j[g]]`` rather than the convention-
        based k-flip.  ``unitary_group`` (if any) is averaged over identity-
        k as before, multiplicatively combined with the spatial average.
    spatial_k_orbit_mask :
        Optional ``(nk1, nk2)`` bool mask.  Where ``False``, ``A[k]`` is
        left unchanged (the partial orbital average is suppressed).  Use
        for finite continuum patches where the full rotational orbit of
        boundary k-points lies outside the sampled region.
    time_reversal_k_index_map :
        Optional ``(idx_i, idx_j, valid)`` triple.  ``valid`` is a
        ``(nk1, nk2)`` bool mask: where ``False``, ``A[k]`` is left alone.
    """
    has_group = (
        unitary_group is not None
        or spatial_group is not None
        or spatial_k_index_maps is not None
    )
    if not has_group and time_reversal_U is None:
        return lambda A: A

    G = None if unitary_group is None else jnp.asarray(unitary_group)
    S = None if spatial_group is None else jnp.asarray(spatial_group)
    U = None if time_reversal_U is None else jnp.asarray(time_reversal_U)
    requires_complex = any(jnp.iscomplexobj(op) for op in (G, S, U) if op is not None)
    # Precomputed vec-space group sums for the GEMM form (see _GEMM_MAX_NB).
    nb_g = None
    for ops in (G, S):
        if ops is not None:
            nb_g = int(ops.shape[-1])
    use_gemm = nb_g is not None and nb_g <= _GEMM_MAX_NB
    M_G = _conj_sum_matrix(G) if (use_gemm and G is not None) else None
    M_S = _conj_sum_matrix(S) if (use_gemm and S is not None) else None
    s_k_conv = str(spatial_k_convention)
    s_k_axes = tuple(spatial_k_flip_axes)
    t_k_conv = str(time_reversal_k_convention)
    t_k_axes = tuple(time_reversal_k_flip_axes)

    if spatial_k_index_maps is not None:
        s_idx_i = jnp.asarray(spatial_k_index_maps[0], dtype=jnp.int32)
        s_idx_j = jnp.asarray(spatial_k_index_maps[1], dtype=jnp.int32)
        if s_idx_i.shape != s_idx_j.shape:
            raise ValueError(
                "spatial_k_index_maps[0] and [1] must have the same shape; "
                f"got {s_idx_i.shape} vs {s_idx_j.shape}."
            )
        if S is None:
            raise ValueError(
                "spatial_k_index_maps requires spatial_group "
                "(the orbital unitaries to average over)."
            )
        if S.shape[0] != s_idx_i.shape[0]:
            raise ValueError(
                "spatial_group and spatial_k_index_maps must have matching "
                f"leading dim; got {S.shape[0]} and {s_idx_i.shape[0]}."
            )
    else:
        s_idx_i = s_idx_j = None

    s_orbit_mask = (
        None if spatial_k_orbit_mask is None
        else jnp.asarray(spatial_k_orbit_mask, dtype=bool)
    )

    if time_reversal_k_index_map is not None:
        if U is None:
            raise ValueError(
                "time_reversal_k_index_map requires time_reversal_U."
            )
        t_idx_i = jnp.asarray(time_reversal_k_index_map[0], dtype=jnp.int32)
        t_idx_j = jnp.asarray(time_reversal_k_index_map[1], dtype=jnp.int32)
        t_valid = jnp.asarray(time_reversal_k_index_map[2], dtype=bool)
    else:
        t_idx_i = t_idx_j = t_valid = None

    def project(A: jax.Array) -> jax.Array:
        out = jnp.asarray(A)
        if requires_complex and not jnp.iscomplexobj(out):
            out = out.astype(jnp.result_type(out.dtype, jnp.complex64))
        if s_idx_i is not None:
            avg = _avg_spatial_group_via_k_maps(out, S, s_idx_i, s_idx_j)
            if s_orbit_mask is not None:
                avg = jnp.where(s_orbit_mask[..., None, None], avg, out)
            out = avg
            if G is not None:
                # Compose orbital-only and spatial averages: the orbital
                # average commutes with k-indexing, so do it after.
                out = _avg_unitary_conj(out, G, M_G)
        elif G is not None and S is not None:
            out = _avg_combined_group(
                out, G, S, s_k_conv, s_k_axes, M_same=M_G, M_flip=M_S,
            )
        elif G is not None:
            out = _avg_unitary_conj(out, G, M_G)
        elif S is not None:
            I_mat = jnp.eye(S.shape[-1], dtype=S.dtype)[None]
            out = _avg_combined_group(
                out, I_mat, S, s_k_conv, s_k_axes, M_flip=M_S,
            )
        if U is not None:
            if t_idx_i is not None:
                out = _avg_time_reversal_via_k_map(out, U, t_idx_i, t_idx_j, t_valid)
            else:
                out = _avg_time_reversal(out, U, t_k_conv, t_k_axes)
        return out

    return project
