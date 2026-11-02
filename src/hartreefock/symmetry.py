"""Symmetry projectors for density and Fock matrices.

:func:`make_project_fn` returns a callable that averages a
``(nk1, nk2, nb, nb)`` array over unitary, spatial and time-reversal
symmetries.  It works on NumPy arrays and on JAX arrays, including under
``jax.jit``: the array namespace is taken from the input on every call.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

ProjectFn = Callable[[Any], Any]

__all__ = ["ProjectFn", "array_namespace", "flip_k", "make_project_fn"]

# Orbital dimension up to which the group sum is applied as one precomputed
# GEMM on vec(A) (an (nb², nb²) matrix: 16 bytes × nb⁴, i.e. 5 MB at nb=24).
# On the nk=49, nb=16 bilayer the per-element conjugation loop costs 30-40 ms
# per call against 5.7 ms for the single (2401, 256) @ (256, 256) product,
# and the projector runs about six times per CG iteration.  Above the
# threshold the nb⁴ matrix outgrows the 2·|G|·nb³ loop and its memory.
_GEMM_MAX_NB = 24


def array_namespace(A):
    """``jax.numpy`` for JAX arrays and tracers, ``numpy`` otherwise."""
    if type(A).__module__.startswith("jax"):
        import jax.numpy as jnp

        return jnp
    return np


def _sum_over(xp, n: int, term, init):
    """``init + sum_i term(i)``; a ``fori_loop`` under JAX keeps traces small."""
    if xp is np:
        acc = init
        for i in range(n):
            acc = acc + term(i)
        return acc
    from jax import lax

    return lax.fori_loop(0, n, lambda i, acc: acc + term(i), init)


def _dagger(xp, X):
    return xp.conj(xp.swapaxes(X, -1, -2))


def flip_k(xp, A, k_convention: str, flip_axes: tuple[int, ...] = (0, 1)):
    """Negate selected k-grid axes under the ``flip`` or ``mod`` convention."""
    if k_convention == "flip":
        return xp.flip(A, axis=flip_axes)
    if k_convention == "mod":
        nk1, nk2 = A.shape[0], A.shape[1]
        i = (-np.arange(nk1)) % nk1 if 0 in flip_axes else np.arange(nk1)
        j = (-np.arange(nk2)) % nk2 if 1 in flip_axes else np.arange(nk2)
        return A[i[:, None], j[None, :], ...]
    raise ValueError(f"k_convention must be 'mod' or 'flip', got {k_convention!r}")


def _conj_sum_matrix(G: np.ndarray) -> np.ndarray:
    """Matrix of ``vec(A) -> vec(sum_i g_i A g_i†)`` for row-major ``vec``.

    ``vec(g A g†) = (g ⊗ conj(g)) vec(A)``, summed once over the concrete group.
    """
    return sum(np.kron(g, np.conj(g)) for g in G)


def _sum_unitary_conj(xp, A, G, M_sum=None):
    """``sum_i g_i A g_i†`` (one GEMM when the vec-space matrix ``M_sum`` is given)."""
    if M_sum is not None:
        nb = A.shape[-1]
        flat = A.reshape(A.shape[:-2] + (nb * nb,))
        M = xp.asarray(M_sum).astype(A.dtype)
        return (flat @ xp.swapaxes(M, -1, -2)).reshape(A.shape)
    # The group follows the working dtype: a complex128 group applied to a
    # complex64 density must not promote the accumulator.
    G = xp.asarray(G).astype(A.dtype)
    return _sum_over(xp, G.shape[0], lambda i: (G[i] @ A) @ _dagger(xp, G[i]), xp.zeros_like(A))


def _avg_unitary_conj(xp, A, G, M_sum=None):
    return _sum_unitary_conj(xp, A, G, M_sum) / G.shape[0]


def _avg_combined_group(xp, A, G_same, G_flip, k_convention, flip_axes, M_same=None, M_flip=None):
    """Average over a group split into same-k and flipped-k elements."""
    acc = _sum_unitary_conj(xp, A, G_same, M_same)
    acc = acc + _sum_unitary_conj(xp, flip_k(xp, A, k_convention, flip_axes), G_flip, M_flip)
    return acc / (G_same.shape[0] + G_flip.shape[0])


def _avg_spatial_group_via_k_maps(xp, A, S, k_index_i, k_index_j):
    """Average ``U_g A[k_index_i[g], k_index_j[g]] U_g†`` over the spatial group.

    Shapes: ``A`` (nk1, nk2, nb, nb), ``S`` (n_g, nb, nb), index maps
    (n_g, nk1, nk2).
    """
    S = xp.asarray(S).astype(A.dtype)
    ki, kj = xp.asarray(k_index_i), xp.asarray(k_index_j)

    def term(i):
        return (S[i] @ A[ki[i], kj[i]]) @ _dagger(xp, S[i])

    return _sum_over(xp, S.shape[0], term, xp.zeros_like(A)) / S.shape[0]


def _avg_time_reversal(xp, A, U, k_convention, flip_axes):
    U = xp.asarray(U).astype(A.dtype)
    A_tr = U @ xp.conj(flip_k(xp, A, k_convention, flip_axes)) @ _dagger(xp, U)
    return 0.5 * (A + A_tr)


def _avg_time_reversal_via_k_map(xp, A, U, k_index_i, k_index_j, valid):
    """Time-reversal average with a precomputed ``-k`` map; invalid k are left alone."""
    U = xp.asarray(U).astype(A.dtype)
    A_neg = A[xp.asarray(k_index_i), xp.asarray(k_index_j)]
    avg = 0.5 * (A + U @ xp.conj(A_neg) @ _dagger(xp, U))
    return xp.where(xp.asarray(valid)[..., None, None], avg, A)


def make_project_fn(
    *,
    unitary_group=None,
    spatial_group=None,
    spatial_k_convention: str = "mod",
    spatial_k_flip_axes: tuple[int, ...] = (0, 1),
    spatial_k_index_maps=None,
    spatial_k_orbit_mask=None,
    time_reversal_U=None,
    time_reversal_k_convention: str = "mod",
    time_reversal_k_flip_axes: tuple[int, ...] = (0, 1),
    time_reversal_k_index_map=None,
) -> ProjectFn:
    """Build a symmetry-averaging projection function.

    How k transforms under a symmetry is given either by a convention string
    (``"mod"``: periodic-cell wrap, ``"flip"``: reflect the selected axes),
    for k-meshes the symmetry maps onto themselves by index arithmetic, or by
    precomputed index maps, needed for general rotations on a finite
    continuum patch (e.g. C3z on a rhombic graphene-K patch; see
    :func:`contimod.symmetry.continuum.cyclic_rotation_index_maps`).

    Parameters
    ----------
    unitary_group, spatial_group, time_reversal_U :
        Orbital unitaries (no k-dependence).  ``unitary_group`` acts at the
        same k; ``spatial_group`` elements also map k to its image.
    spatial_k_index_maps :
        Optional ``(idx_i, idx_j)``, each ``(n_g_spatial, nk1, nk2)``: the
        spatial average uses ``A[idx_i[g], idx_j[g]]`` instead of the
        convention flip.  ``unitary_group`` (if any) is applied after it.
    spatial_k_orbit_mask :
        Optional ``(nk1, nk2)`` bool mask; where False, ``A[k]`` is left
        unchanged (for boundary k whose orbit leaves the sampled patch).
    time_reversal_k_index_map :
        Optional ``(idx_i, idx_j, valid)``; where ``valid`` is False the
        time-reversal average leaves ``A[k]`` alone.
    """
    has_group = (
        unitary_group is not None
        or spatial_group is not None
        or spatial_k_index_maps is not None
    )
    if not has_group and time_reversal_U is None:
        return lambda A: A

    G = None if unitary_group is None else np.asarray(unitary_group)
    S = None if spatial_group is None else np.asarray(spatial_group)
    U = None if time_reversal_U is None else np.asarray(time_reversal_U)
    requires_complex = any(np.iscomplexobj(op) for op in (G, S, U) if op is not None)
    nb_g = next((int(ops.shape[-1]) for ops in (S, G) if ops is not None), None)
    use_gemm = nb_g is not None and nb_g <= _GEMM_MAX_NB
    M_G = _conj_sum_matrix(G) if (use_gemm and G is not None) else None
    M_S = _conj_sum_matrix(S) if (use_gemm and S is not None) else None
    s_conv, s_axes = str(spatial_k_convention), tuple(spatial_k_flip_axes)
    t_conv, t_axes = str(time_reversal_k_convention), tuple(time_reversal_k_flip_axes)

    s_maps = None
    if spatial_k_index_maps is not None:
        s_maps = tuple(np.asarray(m, dtype=np.int32) for m in spatial_k_index_maps)
        if s_maps[0].shape != s_maps[1].shape:
            raise ValueError(
                "spatial_k_index_maps[0] and [1] must have the same shape; "
                f"got {s_maps[0].shape} vs {s_maps[1].shape}."
            )
        if S is None:
            raise ValueError(
                "spatial_k_index_maps requires spatial_group "
                "(the orbital unitaries to average over)."
            )
        if S.shape[0] != s_maps[0].shape[0]:
            raise ValueError(
                "spatial_group and spatial_k_index_maps must have matching "
                f"leading dim; got {S.shape[0]} and {s_maps[0].shape[0]}."
            )
    orbit_mask = None if spatial_k_orbit_mask is None else np.asarray(spatial_k_orbit_mask, dtype=bool)

    t_map = None
    if time_reversal_k_index_map is not None:
        if U is None:
            raise ValueError("time_reversal_k_index_map requires time_reversal_U.")
        idx_i, idx_j, valid = time_reversal_k_index_map
        t_map = (np.asarray(idx_i, np.int32), np.asarray(idx_j, np.int32), np.asarray(valid, bool))

    def project(A):
        xp = array_namespace(A)
        out = xp.asarray(A)
        if requires_complex and not xp.iscomplexobj(out):
            out = out.astype(xp.result_type(out.dtype, xp.complex64))
        if s_maps is not None:
            avg = _avg_spatial_group_via_k_maps(xp, out, S, *s_maps)
            out = avg if orbit_mask is None else xp.where(xp.asarray(orbit_mask)[..., None, None], avg, out)
            if G is not None:
                # The orbital average commutes with k-indexing.
                out = _avg_unitary_conj(xp, out, G, M_G)
        elif G is not None and S is not None:
            out = _avg_combined_group(xp, out, G, S, s_conv, s_axes, M_same=M_G, M_flip=M_S)
        elif G is not None:
            out = _avg_unitary_conj(xp, out, G, M_G)
        elif S is not None:
            identity = np.eye(S.shape[-1], dtype=S.dtype)[None]
            out = _avg_combined_group(xp, out, identity, S, s_conv, s_axes, M_flip=M_S)
        if U is not None:
            if t_map is not None:
                out = _avg_time_reversal_via_k_map(xp, out, U, *t_map)
            else:
                out = _avg_time_reversal(xp, out, U, t_conv, t_axes)
        return out

    return project
