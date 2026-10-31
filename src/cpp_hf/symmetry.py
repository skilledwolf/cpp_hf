"""Generic symmetry projectors (numpy port of ``jax_hf.symmetry``).

``make_project_fn`` builds a callable that averages a density / Fock matrix
over unitary, spatial, and time-reversal symmetries.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

ProjectFn = Callable[[np.ndarray], np.ndarray]

__all__ = ["ProjectFn", "make_project_fn"]


def _flip_k(
    A: np.ndarray,
    k_convention: str,
    flip_axes: tuple[int, ...] = (0, 1),
) -> np.ndarray:
    if k_convention == "flip":
        return np.flip(A, axis=flip_axes)
    if k_convention == "mod":
        nk1, nk2 = A.shape[0], A.shape[1]
        i = (
            (-np.arange(nk1, dtype=np.int32)) % nk1
            if 0 in flip_axes
            else np.arange(nk1, dtype=np.int32)
        )
        j = (
            (-np.arange(nk2, dtype=np.int32)) % nk2
            if 1 in flip_axes
            else np.arange(nk2, dtype=np.int32)
        )
        return A[i[:, None], j[None, :], ...]
    raise ValueError(f"k_convention must be 'mod' or 'flip', got {k_convention!r}")


# Orbital dimension up to which the group sum is applied as one precomputed
# GEMM on vec(A) (an (nb², nb²) matrix) instead of |G| batched small
# conjugations.  Mirrors jax_hf.symmetry._GEMM_MAX_NB, where the GEMM form is
# on (nb <= 24) and speeds the bilayer CG solve up 1.5-1.75x.  Here it is OFF
# by default: this projector runs as a Python callback inside the OpenMP
# native solver, and numpy's threaded BLAS GEMM contends with the solver's
# worker threads (measured on the nk=49, nb=16 bilayer: standalone 43 -> 12 ms
# per call, but the CG solve slowed from 3.3 s to 5.4 s; with
# OPENBLAS_NUM_THREADS=1 it merely broke even).  Set to e.g. 24 to enable; the
# real fix is a native projector inside the C++ loop.
_GEMM_MAX_NB = 0


def _conj_sum_matrix(G: np.ndarray) -> np.ndarray:
    """Matrix of ``vec(A) -> vec(sum_i g_i A g_i†)`` for row-major ``vec``.

    ``vec(g A g†) = (g ⊗ conj(g)) vec(A)``; formed once from the concrete
    unitaries at ``make_project_fn`` time.
    """
    G = np.asarray(G)
    nb = G.shape[-1]
    M = np.zeros((nb * nb, nb * nb), dtype=G.dtype)
    for g in G:
        M += np.kron(g, np.conj(g))
    return M


def _apply_vec_matrix(A: np.ndarray, M: np.ndarray) -> np.ndarray:
    # One 2-D BLAS GEMM over all k-points.  A stacked (nk1, nk2, nb²) @ (nb², nb²)
    # product with a transposed view of ``M`` makes numpy fall back to its
    # non-BLAS loop (measured 1.5 s instead of ~10 ms on (49, 49, 16, 16)).
    nb = A.shape[-1]
    flat = np.ascontiguousarray(A).reshape(-1, nb * nb)
    MT = np.ascontiguousarray(M.astype(A.dtype, copy=False).T)
    return (flat @ MT).reshape(A.shape)


def _sum_unitary_conj(
    A: np.ndarray, G: np.ndarray, M_sum: np.ndarray | None = None,
) -> np.ndarray:
    if M_sum is not None:
        return _apply_vec_matrix(A, M_sum)
    G = G.astype(A.dtype, copy=False)
    acc = np.zeros_like(A)
    for i in range(G.shape[0]):
        g = G[i]
        gH = np.conj(np.swapaxes(g, -1, -2))
        acc += (g @ A) @ gH
    return acc


def _avg_unitary_conj(
    A: np.ndarray, G: np.ndarray, M_sum: np.ndarray | None = None,
) -> np.ndarray:
    ng = G.shape[0]
    return _sum_unitary_conj(A, G, M_sum) / np.asarray(float(ng), dtype=A.dtype)


def _avg_combined_group(
    A: np.ndarray,
    G_same: np.ndarray,
    G_flip: np.ndarray,
    k_convention: str,
    flip_axes: tuple[int, ...] = (0, 1),
    M_same: np.ndarray | None = None,
    M_flip: np.ndarray | None = None,
) -> np.ndarray:
    acc = _sum_unitary_conj(A, G_same, M_same)
    A_neg = _flip_k(A, k_convention, flip_axes)
    acc += _sum_unitary_conj(A_neg, G_flip, M_flip)
    N = G_same.shape[0] + G_flip.shape[0]
    return acc / np.asarray(float(N), dtype=A.dtype)


def _avg_time_reversal(
    A: np.ndarray,
    U: np.ndarray,
    k_convention: str,
    flip_axes: tuple[int, ...] = (0, 1),
) -> np.ndarray:
    U = U.astype(A.dtype, copy=False)
    UH = np.conj(np.swapaxes(U, -1, -2))
    A_neg = _flip_k(A, k_convention, flip_axes)
    A_tr = U @ np.conj(A_neg) @ UH
    return 0.5 * (A + A_tr)


def make_project_fn(
    *,
    unitary_group: np.ndarray | None = None,
    spatial_group: np.ndarray | None = None,
    spatial_k_convention: str = "mod",
    spatial_k_flip_axes: tuple[int, ...] = (0, 1),
    time_reversal_U: np.ndarray | None = None,
    time_reversal_k_convention: str = "mod",
    time_reversal_k_flip_axes: tuple[int, ...] = (0, 1),
) -> ProjectFn:
    """Build a symmetry-averaging projection function."""
    has_group = unitary_group is not None or spatial_group is not None
    if not has_group and time_reversal_U is None:
        return lambda A: A

    G = None if unitary_group is None else np.asarray(unitary_group)
    S = None if spatial_group is None else np.asarray(spatial_group)
    U = None if time_reversal_U is None else np.asarray(time_reversal_U)
    requires_complex = any(np.iscomplexobj(op) for op in (G, S, U))
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

    def project(A: np.ndarray) -> np.ndarray:
        out = np.asarray(A)
        if requires_complex and not np.iscomplexobj(out):
            out = out.astype(np.result_type(out.dtype, np.complex64))
        if G is not None and S is not None:
            out = _avg_combined_group(
                out, G, S, s_k_conv, s_k_axes, M_same=M_G, M_flip=M_S,
            )
        elif G is not None:
            out = _avg_unitary_conj(out, G, M_G)
        elif S is not None:
            I_mat = np.eye(S.shape[-1], dtype=S.dtype)[None]
            out = _avg_combined_group(
                out, I_mat, S, s_k_conv, s_k_axes, M_flip=M_S,
            )
        if U is not None:
            out = _avg_time_reversal(out, U, t_k_conv, t_k_axes)
        return out

    return project
